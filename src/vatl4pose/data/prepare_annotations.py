"""Per-video annotation files of PoseTrack21 and JRDB-Pose, made from the official releases.

The readers in :mod:`vatl4pose.data.annotations` take one JSON file per video,
``<root>/activelearning/<split>/<video_id>_*.json``, in the COCO keypoint layout with the size of
every image. The official releases hold other layouts. This module writes the per-video files from
them; ``vatl4pose-prepare-annotations`` is its command.

PoseTrack21 (``root`` is the directory of the release)::

    <root>/posetrack_data/<train|val>/<video_id>_*.json   input, one file per video
    <root>/<file_name>                                    the frame, ``file_name`` from the JSON
    <root>/activelearning/<split>/<same file name>        output

The release keeps the 593 training videos in ``posetrack_data/train``. The splits ``train`` and
``train_val`` take their videos from that directory by id list (579 and 14 videos, as in the paper).
Their output holds every image and annotation of the video, with the size of each image added.
The split ``val`` keeps, of the 33 frames around the middle frame of each validation video, those
that carry labels (the *dense window*): the labeled images whose id lies within 16 of
``int(f"1{vid_id}{nframes // 2:04d}")``, and the annotations of those images.

JRDB-Pose (``root`` is the directory with the images and the labels of the release)::

    <root>/jrdb2022/labels/labels_2d_pose_stitched_coco/<scene>.json   keypoints of the people
    <root>/jrdb2022/labels/labels_2d_stitched/<scene>.json             boxes of the people
    <root>/images/<file_name>                                          the stitched frame
    <root>/activelearning/<split>/<video_id>_jrdb-pose.json            output

``<root>/jrdb2022`` is the unpacked label release (or a link to it); the stitched frames are in
``<root>/images/image_stitched/<scene>/``.
The scene at position ``i`` of the scene list of a split is the video ``f"{i:02d}"``. Of each scene
the first 149 frames are kept (image ids below 150). An image gets the id
``int(f"1{video_id}{image_id:05d}")`` and an annotation the id ``int(f"{new_image_id}{track_id:03d}")``.
The box and the area of an annotation come from the detection ``pedestrian:<track_id>`` of the
same frame; an annotation without one is dropped, but its image stays in the file. A joint
visibility of 0 stays 0, any other value becomes 1.0.

Image sizes are read from the image header, without decoding the pixels. A file that exists is
kept when it holds the same data and is otherwise refused, unless the caller allows replacing it.
Every file is written to a temporary name and moved into place, so a reader never sees half a file.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import pathlib
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from PIL import Image

_LOG = logging.getLogger(__name__)

POSETRACK21_SPLITS = ("train", "train_val", "val")
"""The splits of PoseTrack21; ``train`` and ``train_val`` share the raw directory ``train``."""

JRDB_POSE_SPLITS = ("train", "val", "test")
"""The splits of JRDB-Pose."""

DENSE_RADIUS = 16
"""A validation frame is kept if its image id differs from the middle frame's by at most this."""

JRDB_IMAGE_ID_LIMIT = 150
"""JRDB-Pose images with an id below this are kept."""


@dataclass(frozen=True)
class PreparedVideo:
    """What happened to the file of one video.

    Attributes:
        video_id: The video, e.g. ``"000342"`` or ``"03"``.
        path: The output file.
        status: ``written`` (new file), ``up to date`` (an equal file existed) or ``replaced``.
        num_images: Images in the file.
        num_annotations: Annotations in the file.
        num_dropped: JRDB-Pose annotations left out for lack of a box; 0 for PoseTrack21.
    """

    video_id: str
    path: pathlib.Path
    status: str
    num_images: int
    num_annotations: int
    num_dropped: int


# ------------------------------------------------------------------ image sizes


def image_size(path: pathlib.Path) -> tuple[int, int]:
    """``(width, height)`` of an image file, from its header (the pixels are not decoded)."""
    with Image.open(path) as image:
        width, height = image.size
    return width, height


def read_image_sizes(
    paths: Sequence[pathlib.Path], num_workers: int
) -> list[tuple[int, int]]:
    """:func:`image_size` of every path, in the order of ``paths``, with ``num_workers`` threads."""
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        return list(pool.map(image_size, paths))


# ------------------------------------------------------------------ comparing and writing


def first_difference(a: Any, b: Any, path: str = "$") -> str | None:
    """Describe the first difference between two parsed JSON values, or ``None`` if they are equal.

    The description is the path of the difference and the two values as ``a != b``, for example
    ``$.images[3].width: 854 != 1280``. Dictionaries are compared key by key in the order of
    ``a``, lists element by element; ``1`` and ``1.0`` are equal, as they are in JSON.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        for key, value in a.items():
            if key not in b:
                return f"{path}: key {key!r} is missing in the second"
            difference = first_difference(value, b[key], f"{path}.{key}")
            if difference is not None:
                return difference
        for key in b:
            if key not in a:
                return f"{path}: key {key!r} is missing in the first"
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            difference = first_difference(x, y, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    return None if a == b else f"{path}: {a!r} != {b!r}"


def write_annotation(
    path: pathlib.Path, data: Mapping[str, Any], overwrite: bool
) -> str:
    """Write ``data`` as JSON to ``path`` atomically; returns the status of the file.

    The status is ``written`` (there was no file), ``up to date`` (the file holds equal data and is
    left untouched) or ``replaced`` (a file with other data, or one that is not valid JSON, was
    replaced).

    Raises:
        ValueError: If a file with other data, or one that does not parse, exists and
            ``overwrite`` is false. The message names the file and, for other data, the first
            difference; the file is not changed.
    """
    text = json.dumps(data)
    status = "written"
    if path.exists():
        try:
            with path.open(encoding="utf-8") as handle:
                existing = json.load(handle)
        except ValueError as error:  # JSONDecodeError and UnicodeDecodeError
            if not overwrite:
                raise ValueError(
                    f"{path} exists but is not valid JSON ({error}). The file is kept; run with "
                    "prepare.overwrite=true to replace it."
                ) from error
        else:
            difference = first_difference(existing, json.loads(text))
            if difference is None:
                return "up to date"
            if not overwrite:
                raise ValueError(
                    f"{path} exists and differs from the file made from the official release "
                    f"(first difference, existing != new: {difference}). The file is kept; run "
                    "with prepare.overwrite=true to replace it."
                )
        status = "replaced"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return status


# ------------------------------------------------------------------ PoseTrack21


def build_posetrack21_video(
    raw: Mapping[str, Any],
    image_root: pathlib.Path,
    *,
    dense_window: bool,
    num_workers: int,
) -> dict[str, Any]:
    """The annotation of one PoseTrack21 video with the size of every kept image added.

    Args:
        raw: The parsed raw file of the video (``images``, ``annotations``, ``categories``).
        image_root: Directory that the ``file_name`` of an image is relative to.
        dense_window: Keep only the labeled images around the middle frame (see the module
            docstring), as for the validation split; otherwise keep every image.
        num_workers: Threads that read the image headers.

    Raises:
        ValueError: If the video has no images, or (with ``dense_window``) an ``is_labeled`` is not a
            boolean.
        FileNotFoundError: If a kept image does not exist.
    """
    images = list(raw["images"])
    if not images:
        raise ValueError("the raw annotation file lists no images")
    if dense_window:
        for image in images:
            if not isinstance(image["is_labeled"], bool):
                raise ValueError(
                    f"image {image['file_name']}: is_labeled must be true or false, got "
                    f"{image['is_labeled']!r}"
                )
        nframes = int(images[0]["nframes"])
        middle = int(f"1{images[0]['vid_id']}{nframes // 2:04d}")
        images = [
            image
            for image in images
            if image["is_labeled"] and abs(image["image_id"] - middle) <= DENSE_RADIUS
        ]
    sizes = read_image_sizes(
        [image_root / image["file_name"] for image in images], num_workers
    )
    sized = [
        {**image, "width": width, "height": height}
        for image, (width, height) in zip(images, sizes, strict=True)
    ]
    kept = {image["image_id"] for image in sized}
    return {
        "images": sized,
        "annotations": [a for a in raw["annotations"] if a["image_id"] in kept],
        "categories": raw["categories"],
    }


def _find_raw_file(directory: pathlib.Path, video_id: str) -> pathlib.Path:
    pattern = f"{glob.escape(video_id)}_*.json"
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"no raw annotation file {directory / pattern}; the PoseTrack21 release holds one "
            "file per video in posetrack_data/train and posetrack_data/val"
        )
    if len(matches) > 1:
        raise ValueError(
            f"expected one raw annotation file {directory / pattern}, found "
            f"{[match.name for match in matches]}"
        )
    return matches[0]


def prepare_posetrack21(
    root: pathlib.Path,
    split: str,
    video_ids: Sequence[str],
    *,
    overwrite: bool,
    num_workers: int,
) -> list[PreparedVideo]:
    """Write ``<root>/activelearning/<split>/<raw file name>`` of every video in ``video_ids``.

    ``val`` videos are read from ``posetrack_data/val`` and cut to the dense window; ``train`` and
    ``train_val`` videos are read from ``posetrack_data/train`` and kept whole. Every raw file is
    located before the first output is written.

    Raises:
        ValueError: If ``split`` is unknown, a video has two raw files, or an existing output file
            differs and ``overwrite`` is false.
        FileNotFoundError: If a raw file or a frame does not exist.
    """
    if split not in POSETRACK21_SPLITS:
        raise ValueError(
            f"unknown PoseTrack21 split {split!r}; valid splits: {', '.join(POSETRACK21_SPLITS)}"
        )
    root = pathlib.Path(root)
    raw_dir = root / "posetrack_data" / ("val" if split == "val" else "train")
    raw_files = [_find_raw_file(raw_dir, video_id) for video_id in video_ids]
    results: list[PreparedVideo] = []
    for video_id, raw_file in zip(video_ids, raw_files, strict=True):
        with raw_file.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        video = build_posetrack21_video(
            raw, root, dense_window=split == "val", num_workers=num_workers
        )
        path = root / "activelearning" / split / raw_file.name
        status = write_annotation(path, video, overwrite)
        results.append(
            PreparedVideo(
                video_id,
                path,
                status,
                len(video["images"]),
                len(video["annotations"]),
                0,
            )
        )
    return results


# ------------------------------------------------------------------ JRDB-Pose


def build_jrdb_pose_video(
    video_id: str,
    pose: Mapping[str, Any],
    detections: Mapping[str, Any],
    image_root: pathlib.Path,
    num_workers: int,
) -> tuple[dict[str, Any], int]:
    """The annotation of one JRDB-Pose scene, and the number of annotations left out.

    Args:
        video_id: Two-digit video id, the position of the scene in its split.
        pose: The parsed ``labels_2d_pose_stitched_coco`` file of the scene.
        detections: The parsed ``labels_2d_stitched`` file of the scene.
        image_root: Directory that ``images/<file_name>`` is relative to.
        num_workers: Threads that read the image headers.

    Raises:
        ValueError: If an annotation refers to an image id that is not the id of the image at
            position ``image_id - 1`` of the pose file's images, or a frame has no entry in the
            detection file.
        FileNotFoundError: If a kept image does not exist.
    """
    pose_images = pose["images"]
    labels = detections["labels"]
    people_of: dict[str, dict[str, Mapping[str, Any]]] = {}
    images: dict[int, dict[str, Any]] = {}
    annotations: list[dict[str, Any]] = []
    dropped = 0
    for annotation in pose["annotations"]:
        image_id = annotation["image_id"]
        if image_id >= JRDB_IMAGE_ID_LIMIT:
            continue
        # the image at position image_id - 1 must have the id image_id, so that reading by
        # position and by id give the same image
        position = image_id - 1
        if (
            not 0 <= position < len(pose_images)
            or pose_images[position]["id"] != image_id
        ):
            raise ValueError(
                f"video {video_id}: annotation {annotation.get('id')} refers to image id "
                f"{image_id}, but the image at position {position} of the pose file's images "
                "does not have that id"
            )
        source = pose_images[position]
        new_image_id = int(f"1{video_id}{image_id:05d}")
        if image_id not in images:
            images[image_id] = {
                "id": new_image_id,
                "image_id": new_image_id,
                "vid_id": video_id,
                "file_name": f"images/{source['file_name']}",
                "is_labeled": True,
                "has_labeled_person": True,
            }
        frame = pathlib.PurePosixPath(source["file_name"]).name
        if frame not in people_of:
            if frame not in labels:
                raise ValueError(
                    f"video {video_id}: labels_2d_stitched has no entry for the frame {frame}"
                )
            people: dict[str, Mapping[str, Any]] = {}
            # when a track has two boxes in one frame, the first one counts
            for person in labels[frame]:
                people.setdefault(person["label_id"], person)
            people_of[frame] = people
        track_id = annotation["track_id"]
        person = people_of[frame].get(f"pedestrian:{track_id}")
        if person is None:
            dropped += 1
            continue
        flat = list(annotation["keypoints"])
        if len(flat) % 3:
            raise ValueError(
                f"video {video_id}: annotation {annotation.get('id')} has {len(flat)} keypoint "
                "values, not a multiple of 3"
            )
        for index in range(2, len(flat), 3):
            flat[index] = 0 if flat[index] == 0 else 1.0
        annotations.append(
            {
                "track_id": track_id,
                "image_id": new_image_id,
                "category_id": annotation["category_id"],
                "num_keypoints": annotation["num_keypoints"],
                "is_crowd": 0,
                "id": int(f"{new_image_id}{track_id:03d}"),
                "keypoints": flat,
                "bbox": person["box"],
                "area": person["attributes"]["area"],
            }
        )
    if dropped:
        _LOG.warning(
            "video %s: dropped %d annotation(s) without a pedestrian box in labels_2d_stitched",
            video_id,
            dropped,
        )
    kept = list(images.values())
    sizes = read_image_sizes(
        [image_root / image["file_name"] for image in kept], num_workers
    )
    for image, (width, height) in zip(kept, sizes, strict=True):
        image["height"], image["width"] = height, width
    video = {
        "images": kept,
        "annotations": annotations,
        "categories": pose["categories"],
    }
    return video, dropped


def prepare_jrdb_pose(
    root: pathlib.Path,
    split: str,
    scenes: Sequence[str],
    video_ids: Sequence[str],
    *,
    overwrite: bool,
    num_workers: int,
) -> list[PreparedVideo]:
    """Write ``<root>/activelearning/<split>/<video_id>_jrdb-pose.json`` of every video.

    Args:
        root: JRDB-Pose directory (see the module docstring).
        split: ``train``, ``val`` or ``test``.
        scenes: The scenes of the split; the scene at position ``i`` is the video ``f"{i:02d}"``.
        video_ids: The videos to write, a subset of the ids of ``scenes``.
        overwrite: Replace an existing file with other data.
        num_workers: Threads that read the image headers.

    Raises:
        ValueError: If ``split`` is unknown, a video id is not a position in ``scenes``, or an
            existing output file differs and ``overwrite`` is false.
        FileNotFoundError: If a label file or a frame does not exist.
    """
    if split not in JRDB_POSE_SPLITS:
        raise ValueError(
            f"unknown JRDB-Pose split {split!r}; valid splits: {', '.join(JRDB_POSE_SPLITS)}"
        )
    valid = [f"{index:02d}" for index in range(len(scenes))]
    unknown = [video_id for video_id in video_ids if video_id not in valid]
    if unknown:
        raise ValueError(
            f"video ids {unknown} are not positions in the scene list of the split {split!r}; "
            f"valid ids: {', '.join(valid)}"
        )
    root = pathlib.Path(root)
    labels = root / "jrdb2022" / "labels"
    files: list[tuple[pathlib.Path, pathlib.Path]] = []
    for video_id in video_ids:
        scene = scenes[int(video_id)]
        pair = (
            labels / "labels_2d_pose_stitched_coco" / f"{scene}.json",
            labels / "labels_2d_stitched" / f"{scene}.json",
        )
        for path in pair:
            if not path.is_file():
                raise FileNotFoundError(
                    f"the label file {path} of video {video_id} (scene {scene}) does not exist"
                )
        files.append(pair)
    results: list[PreparedVideo] = []
    for video_id, (pose_file, detection_file) in zip(video_ids, files, strict=True):
        with pose_file.open(encoding="utf-8") as handle:
            pose = json.load(handle)
        with detection_file.open(encoding="utf-8") as handle:
            detections = json.load(handle)
        video, dropped = build_jrdb_pose_video(
            video_id, pose, detections, root, num_workers
        )
        path = root / "activelearning" / split / f"{video_id}_jrdb-pose.json"
        status = write_annotation(path, video, overwrite)
        results.append(
            PreparedVideo(
                video_id,
                path,
                status,
                len(video["images"]),
                len(video["annotations"]),
                dropped,
            )
        )
    return results
