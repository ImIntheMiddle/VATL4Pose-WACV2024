"""Per-video person records read from PoseTrack21, JRDB-Pose and ``coco_custom`` annotations.

A *sample* is one annotated person in one frame. A reader turns the annotation file of one
video into a :class:`VideoAnnotation`: the samples in a fixed order, plus for every sample
the index of the same person (track) in the nearest earlier and later annotated frame
(:func:`link_neighbors`). Every later component (heatmap cache, query scores, evaluation)
indexes its arrays by the sample index defined here.

Layouts
-------
PoseTrack21 and JRDB-Pose (per-video files made by the dataset preparation scripts)::

    <root>/activelearning/<split>/<video_id>_*.json    exactly one match per video
    <root>/<file_name>                                 file_name is read from the JSON

``coco_custom`` (any other dataset in COCO keypoint format with track ids)::

    <root>/images/<video_id>/<frame files>
    <root>/annotations/<video_id>.json

The ``coco_custom`` JSON holds ``images`` (``id``, ``file_name`` relative to ``<root>``,
``width``, ``height``), ``annotations`` (``id``, ``image_id``, ``track_id``, ``bbox`` as
xywh, ``keypoints`` as a flat list of ``3 * K`` values, ``num_keypoints``, ``category_id``)
and ``categories``. All track ids of a file are ints or all are strings. The PoseTrack21
and JRDB-Pose files follow the same COCO structure.

Conventions
-----------
* Box. xywh becomes xyxy with ``xmax = x + max(w - 1, 0)`` and ``ymax = y + max(h - 1, 0)``;
  every coordinate is clipped to ``[0, W - 1]`` and ``[0, H - 1]``. PoseTrack21 and
  ``coco_custom`` skip an annotation whose clipped box has ``xmax <= xmin`` or
  ``ymax <= ymin``. JRDB-Pose keeps such boxes: its legacy test (``xmax < xmin``) cannot
  fire after clipping, because ``xmax >= xmin`` holds by construction. JRDB-Pose skips
  annotations without a ``bbox``.
* Keypoints. ``(K, 3)`` array of x, y, visibility with visibility ``min(1, v)``. An
  annotation without a visible keypoint is skipped. This covers the legacy test for an
  all-zero annotation, which has no visible keypoint either.
* ``frame_index``. Position of the image among the images of the annotation file when
  they are sorted by ``file_name``.
* Sample order. PoseTrack21 and JRDB-Pose use the legacy sort key
  ``int(str(ann_id)[-n:] + str(image_id))`` (``n = 2`` for PoseTrack21, ``3`` for
  JRDB-Pose) over the samples listed by ascending image id, so sample indices match the
  2023 query lists. ``coco_custom`` orders by ``(track_id, frame_index)``.
* Track id. PoseTrack21 prefixes the video id (``"000342" + "0"``), as the legacy reader
  does; the other readers use ``str(track_id)``.
"""

from __future__ import annotations

import glob
import json
import logging
import pathlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from vatl4pose.keypoints import KeypointSpec, get_spec

_LOG = logging.getLogger(__name__)


def _describe(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"array of shape {value.shape} and dtype {value.dtype}"
    return type(value).__name__


@dataclass(frozen=True, eq=False)
class PersonRecord:
    """One annotated person in one frame.

    Equality is by identity (``eq=False``) because the fields hold arrays.

    Attributes:
        sample_index: Position of the sample in its :class:`VideoAnnotation`.
        ann_id: Annotation id from the JSON file.
        image_id: Image id from the JSON file.
        frame_index: Position of the image in the video, sorted by ``file_name``.
        track_id: Person identity within the video.
        image_path: Absolute path of the frame image.
        image_size: ``(width, height)`` of the image in pixels.
        bbox_xyxy: ``(4,)`` float64 box, converted and clipped by the legacy convention.
        keypoints: ``(K, 3)`` float64 array of x, y and visibility in ``{0, 1}``.
        bbox_json_xywh: ``(4,)`` float64 raw ``bbox`` of the annotation, unclipped.
    """

    sample_index: int
    ann_id: int
    image_id: int
    frame_index: int
    track_id: str
    image_path: pathlib.Path
    image_size: tuple[int, int]
    bbox_xyxy: np.ndarray
    keypoints: np.ndarray
    bbox_json_xywh: np.ndarray

    def __post_init__(self) -> None:
        where = f"PersonRecord(ann_id={self.ann_id})"
        if not self.image_path.is_absolute():
            raise ValueError(f"{where}.image_path: {self.image_path} is not absolute")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"{where}.image_size: {self.image_size} is not positive")
        for name in ("bbox_xyxy", "bbox_json_xywh"):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (4,)
                or value.dtype != np.float64
            ):
                raise ValueError(
                    f"{where}.{name}: expected a float64 array of shape (4,), got {_describe(value)}"
                )
        keypoints = self.keypoints
        if (
            not isinstance(keypoints, np.ndarray)
            or keypoints.ndim != 2
            or keypoints.shape[1] != 3
            or keypoints.dtype != np.float64
        ):
            raise ValueError(
                f"{where}.keypoints: expected a float64 array of shape (K, 3), got {_describe(keypoints)}"
            )
        if not np.all((keypoints[:, 2] == 0.0) | (keypoints[:, 2] == 1.0)):
            raise ValueError(
                f"{where}.keypoints: visibility must be 0 or 1, got {np.unique(keypoints[:, 2]).tolist()}"
            )


@dataclass(frozen=True, eq=False)
class VideoAnnotation:
    """All samples of one video, in sample order, with their temporal neighbors.

    Attributes:
        dataset: ``"posetrack21"``, ``"jrdb_pose"`` or ``"coco_custom"``.
        video_id: Video identifier the reader was called with.
        spec: Joint layout of the keypoints.
        people: The samples; ``people[i].sample_index == i``.
        prev_index: ``(N,)`` int64. Sample of the same track in the nearest earlier
            annotated frame, or ``-1``.
        next_index: ``(N,)`` int64. Sample of the same track in the nearest later
            annotated frame, or ``-1``.
        annotation_path: Absolute path of the JSON file the samples were read from.
    """

    dataset: str
    video_id: str
    spec: KeypointSpec
    people: tuple[PersonRecord, ...]
    prev_index: np.ndarray
    next_index: np.ndarray
    annotation_path: pathlib.Path

    def __post_init__(self) -> None:
        if not self.annotation_path.is_absolute():
            raise ValueError(
                f"VideoAnnotation.annotation_path: {self.annotation_path} is not absolute"
            )
        n = len(self.people)
        for name in ("prev_index", "next_index"):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (n,)
                or value.dtype != np.int64
            ):
                raise ValueError(
                    f"VideoAnnotation.{name}: expected an int64 array of shape ({n},), got {_describe(value)}"
                )
            if n and (value.min() < -1 or value.max() >= n):
                raise ValueError(
                    f"VideoAnnotation.{name}: values must lie in [-1, {n}), got "
                    f"[{value.min()}, {value.max()}]"
                )
        for i, person in enumerate(self.people):
            if person.sample_index != i:
                raise ValueError(
                    f"VideoAnnotation.people[{i}]: sample_index is {person.sample_index}"
                )
            if person.keypoints.shape[0] != self.spec.num_joints:
                raise ValueError(
                    f"VideoAnnotation.people[{i}]: {person.keypoints.shape[0]} keypoints, "
                    f"spec {self.spec.name!r} has {self.spec.num_joints} joints"
                )


def link_neighbors(people: Sequence[PersonRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Link every sample to the same track in the nearest earlier and later annotated frame.

    A frame where the track has no sample is skipped, so a gap is bridged. Samples of
    different tracks are never linked.

    Args:
        people: The samples of one video, in any order.

    Returns:
        ``(prev_index, next_index)``, both ``(N,)`` int64. Entry ``i`` holds the position
        in ``people`` of the neighbor of ``people[i]``, or ``-1`` if there is none.

    Raises:
        ValueError: If a track has two samples in one frame; the neighbor is ambiguous.
    """
    n = len(people)
    prev_index = np.full(n, -1, dtype=np.int64)
    next_index = np.full(n, -1, dtype=np.int64)
    frames_of_track: dict[str, list[tuple[int, int]]] = {}
    for position, person in enumerate(people):
        frames_of_track.setdefault(person.track_id, []).append(
            (person.frame_index, position)
        )
    for track_id, entries in frames_of_track.items():
        entries.sort()
        for (frame, earlier), (next_frame, later) in zip(entries, entries[1:]):
            if frame == next_frame:
                raise ValueError(
                    f"track {track_id!r} has more than one person in frame {frame}"
                )
            next_index[earlier] = later
            prev_index[later] = earlier
    return prev_index, next_index


class _Raw(NamedTuple):
    """A person as found in the JSON, before the sample order is fixed."""

    ann_id: int
    image_id: int
    frame_index: int
    track: int | str
    image_path: pathlib.Path
    image_size: tuple[int, int]
    bbox_xyxy: np.ndarray
    keypoints: np.ndarray
    bbox_json_xywh: np.ndarray


def _require(mapping: Mapping[str, Any], key: str, owner: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{owner}: missing key {key!r}")
    return mapping[key]


def _legacy_box_xyxy(xywh: np.ndarray, width: int, height: int) -> np.ndarray:
    """xywh to xyxy with ``max(size - 1, 0)`` extents, clipped to ``[0, size - 1]``."""
    x, y, w, h = xywh
    xyxy = np.array(
        [x, y, x + max(w - 1.0, 0.0), y + max(h - 1.0, 0.0)], dtype=np.float64
    )
    limits = np.array([width - 1, height - 1, width - 1, height - 1], dtype=np.float64)
    return np.minimum(limits, np.maximum(0.0, xyxy))


def _collect(
    json_path: pathlib.Path,
    root: pathlib.Path,
    spec: KeypointSpec,
    *,
    skip_empty_box: bool,
    skip_without_bbox: bool,
) -> list[_Raw]:
    """Read the people of one annotation file, ordered by image id and then by file order."""
    with json_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    images = _require(data, "images", json_path.name)
    annotations = _require(data, "annotations", json_path.name)

    file_names = [
        _require(image, "file_name", f"{json_path.name} image") for image in images
    ]
    if len(set(file_names)) != len(file_names):
        raise ValueError(f"{json_path.name}: two images share one file_name")
    frame_of = {name: frame for frame, name in enumerate(sorted(file_names))}

    image_by_id: dict[int, Mapping[str, Any]] = {}
    for image in images:
        image_id = int(_require(image, "id", f"{json_path.name} image"))
        if image_id in image_by_id:
            raise ValueError(f"{json_path.name}: image id {image_id} appears twice")
        image_by_id[image_id] = image

    annotations_of: dict[int, list[Mapping[str, Any]]] = {i: [] for i in image_by_id}
    for ann in annotations:
        owner = f"{json_path.name} annotation {ann.get('id')}"
        image_id = int(_require(ann, "image_id", owner))
        if image_id not in annotations_of:
            raise ValueError(f"{owner}: refers to unknown image id {image_id}")
        annotations_of[image_id].append(ann)

    num_values = 3 * spec.num_joints
    raws: list[_Raw] = []
    for image_id in sorted(image_by_id):
        image = image_by_id[image_id]
        image_owner = f"{json_path.name} image {image_id}"
        image_path = root / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{image_owner}: image file {image_path} does not exist"
            )
        width = int(_require(image, "width", image_owner))
        height = int(_require(image, "height", image_owner))

        for ann in annotations_of[image_id]:
            ann_id = int(_require(ann, "id", image_owner))
            owner = f"{json_path.name} annotation {ann_id}"
            if "bbox" not in ann and skip_without_bbox:
                _LOG.warning("%s: no bbox, skipped", owner)
                continue
            bbox_json = np.asarray(_require(ann, "bbox", owner), dtype=np.float64)
            if bbox_json.shape != (4,):
                raise ValueError(
                    f"{owner}: bbox must hold 4 values, got {bbox_json.shape}"
                )
            bbox_xyxy = _legacy_box_xyxy(bbox_json, width, height)
            if skip_empty_box and (
                bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]
            ):
                continue

            flat = np.asarray(_require(ann, "keypoints", owner), dtype=np.float64)
            if flat.shape != (num_values,):
                raise ValueError(
                    f"{owner}: keypoints has {flat.size} values, expected {num_values} "
                    f"for spec {spec.name!r} ({spec.num_joints} joints)"
                )
            keypoints = flat.reshape(spec.num_joints, 3).copy()
            keypoints[:, 2] = np.minimum(1.0, keypoints[:, 2])
            if keypoints[:, 2].sum() < 1.0:
                continue

            raws.append(
                _Raw(
                    ann_id=ann_id,
                    image_id=image_id,
                    frame_index=frame_of[image["file_name"]],
                    track=_require(ann, "track_id", owner),
                    image_path=image_path,
                    image_size=(width, height),
                    bbox_xyxy=bbox_xyxy,
                    keypoints=keypoints,
                    bbox_json_xywh=bbox_json,
                )
            )
    return raws


def _build(
    dataset: str,
    video_id: str,
    spec: KeypointSpec,
    json_path: pathlib.Path,
    raws: Sequence[_Raw],
    track_id_of: Callable[[_Raw], str],
) -> VideoAnnotation:
    people = tuple(
        PersonRecord(
            sample_index=index,
            ann_id=raw.ann_id,
            image_id=raw.image_id,
            frame_index=raw.frame_index,
            track_id=track_id_of(raw),
            image_path=raw.image_path,
            image_size=raw.image_size,
            bbox_xyxy=raw.bbox_xyxy,
            keypoints=raw.keypoints,
            bbox_json_xywh=raw.bbox_json_xywh,
        )
        for index, raw in enumerate(raws)
    )
    prev_index, next_index = link_neighbors(people)
    return VideoAnnotation(
        dataset=dataset,
        video_id=video_id,
        spec=spec,
        people=people,
        prev_index=prev_index,
        next_index=next_index,
        annotation_path=json_path,
    )


def _find_annotation_file(directory: pathlib.Path, video_id: str) -> pathlib.Path:
    pattern = f"{glob.escape(video_id)}_*.json"
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one annotation file {directory / pattern}, found "
            f"{len(matches)}: {[match.name for match in matches]}"
        )
    return matches[0]


def _legacy_sort_key(digits: int) -> Callable[[_Raw], int]:
    """Legacy sort key: the last ``digits`` digits of the ann id followed by the image id."""

    def key(raw: _Raw) -> int:
        return int(str(raw.ann_id)[-digits:] + str(raw.image_id))

    return key


def read_posetrack21_video(
    root: pathlib.Path, video_id: str, split: str = "val"
) -> VideoAnnotation:
    """Read one PoseTrack21 video.

    Args:
        root: PoseTrack21 root; the annotation file is
            ``root/activelearning/<split>/<video_id>_*.json`` and image paths are
            ``root/<file_name>``.
        video_id: Video id such as ``"000342"``.
        split: Sub-directory of ``activelearning``.

    Raises:
        ValueError: If the annotation file is not unique or the JSON is malformed.
        FileNotFoundError: If an image listed in the JSON does not exist.
    """
    root = pathlib.Path(root).resolve()
    spec = get_spec("posetrack21")
    json_path = _find_annotation_file(root / "activelearning" / split, video_id)
    raws = _collect(json_path, root, spec, skip_empty_box=True, skip_without_bbox=False)
    raws.sort(key=_legacy_sort_key(2))
    return _build(
        "posetrack21",
        video_id,
        spec,
        json_path,
        raws,
        lambda raw: f"{video_id}{raw.track}",
    )


def read_jrdb_pose_video(
    root: pathlib.Path, video_id: str, split: str = "test"
) -> VideoAnnotation:
    """Read one JRDB-Pose video.

    Args:
        root: JRDB-Pose root; the annotation file is
            ``root/activelearning/<split>/<video_id>_*.json`` and image paths are
            ``root/<file_name>`` (for example ``images/image_stitched/<scene>/000001.jpg``).
        video_id: Video id such as ``"00"``.
        split: Sub-directory of ``activelearning``.

    Raises:
        ValueError: If the annotation file is not unique or the JSON is malformed.
        FileNotFoundError: If an image listed in the JSON does not exist.
    """
    root = pathlib.Path(root).resolve()
    spec = get_spec("jrdb_pose")
    json_path = _find_annotation_file(root / "activelearning" / split, video_id)
    raws = _collect(json_path, root, spec, skip_empty_box=False, skip_without_bbox=True)
    raws.sort(key=_legacy_sort_key(3))
    return _build(
        "jrdb_pose", video_id, spec, json_path, raws, lambda raw: str(raw.track)
    )


def read_coco_custom_video(
    root: pathlib.Path, video_id: str, spec: KeypointSpec
) -> VideoAnnotation:
    """Read one video of a ``coco_custom`` dataset (layout in the module docstring).

    Args:
        root: Dataset root; the annotation file is ``root/annotations/<video_id>.json``
            and image paths are ``root/<file_name>``.
        video_id: Video identifier.
        spec: Joint layout; every annotation needs ``3 * spec.num_joints`` keypoint values.

    Raises:
        ValueError: If the JSON is malformed or the track ids are not all ints or all strings.
        FileNotFoundError: If the annotation file or an image does not exist.
    """
    root = pathlib.Path(root).resolve()
    json_path = root / "annotations" / f"{video_id}.json"
    raws = _collect(json_path, root, spec, skip_empty_box=True, skip_without_bbox=False)
    kinds = {type(raw.track) for raw in raws}
    if len(kinds) > 1 or not kinds <= {int, str}:
        raise ValueError(
            f"{json_path.name}: track_id values must be all int or all str, got "
            f"{sorted(kind.__name__ for kind in kinds)}"
        )
    raws.sort(key=lambda raw: (raw.track, raw.frame_index))
    return _build(
        "coco_custom", video_id, spec, json_path, raws, lambda raw: str(raw.track)
    )
