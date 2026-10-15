"""Per-video person records read from PoseTrack21, JRDB-Pose and ``coco_custom`` annotations.

A reader turns the annotation file of one video into a :class:`VideoAnnotation`: the people (one
annotated person in one frame each) in a fixed order, plus for every person the index of the
same track in the nearest earlier and later annotated frame (:func:`link_neighbors`). Every
later component (heatmap cache, query scores, evaluation) indexes its arrays by the sample index
defined here, the position of a person in that order.

Layouts
-------
PoseTrack21 and JRDB-Pose (per-video files made by ``vatl4pose-prepare-annotations``)::

    <root>/activelearning/<split>/<video_id>_*.json    exactly one match per video
    <root>/<file_name>                                 file_name is read from the JSON

``coco_custom`` (any other dataset in COCO keypoint format with track ids)::

    <root>/images/<video_id>/<frame files>
    <root>/annotations/<video_id>.json

The ``coco_custom`` JSON holds ``images`` (``id``, ``file_name`` relative to ``<root>``,
``width``, ``height``), ``annotations`` (``id``, ``image_id``, ``track_id``, ``bbox`` as
xywh, ``keypoints`` as a flat list of ``3 * K`` values, ``num_keypoints``, ``category_id``)
and ``categories``. ``keypoints`` is optional in ``coco_custom`` (see Keypoints below). All
track ids of a file are ints or all are strings. The PoseTrack21 and JRDB-Pose files follow the
same COCO structure.

Conventions
-----------
* Box. xywh becomes xyxy with ``xmax = x + max(w - 1, 0)`` and ``ymax = y + max(h - 1, 0)``;
  every coordinate is clipped to ``[0, W - 1]`` and ``[0, H - 1]``. PoseTrack21 and
  ``coco_custom`` skip an annotation whose clipped box has ``xmax <= xmin`` or
  ``ymax <= ymin``. JRDB-Pose keeps such boxes: its 2023 test (``xmax < xmin``) cannot
  fire after clipping, because ``xmax >= xmin`` holds by construction. JRDB-Pose skips
  annotations without a ``bbox``.
* Panorama. A frame of a 360-degree video is a strip whose right edge continues at its left edge
  (a JRDB-Pose frame is 3760 pixels wide). The JRDB-Pose and ``coco_custom`` readers take
  ``panorama``. With it on, a person is placed in the unwrapped coordinates of the strip, next to
  the box of the person:

  - The box is clipped at the top and the bottom edge only. A box with ``x < 0`` or
    ``x + w > W`` keeps its raw coordinates, so ``xmin`` may be negative and ``xmax`` may exceed
    ``W - 1``. ``bbox_json_xywh`` is the raw box in both modes.
  - Every joint with a position is moved by a whole number of frame widths ``k * W`` to the copy
    nearest to the center of the box, ``k = floor((cx - x) / W + 0.5)`` with
    ``cx = (xmin + xmax) / 2``. A joint exactly half a frame width from ``cx`` goes to the copy
    with the larger x. A joint that is not visible but has a position moves as well, and keeps
    its visibility 0. A joint at ``(0, 0)`` is the "not annotated" marker of the files and stays
    at ``(0, 0)``, whatever its visibility. A person in the middle of the frame is unchanged.
  - The skip rules are the same. The empty-box test of ``coco_custom`` reads the box that is not
    clipped at the left and right edge, so a box that lies entirely beyond an edge is kept
    (JRDB-Pose skips no empty box). The sample order and the links between frames follow from the
    kept people.

  With ``panorama`` off the readers behave as in the 2023 code.
* Keypoints. ``(K, 3)`` array of x, y, visibility with visibility ``min(1, v)``. PoseTrack21 and
  JRDB-Pose skip an annotation without a visible joint (this covers the 2023 test for an
  all-zero annotation, which has no visible joint either) and require the ``keypoints`` key.
  ``coco_custom`` keeps every annotation with a valid box: an annotation without a ``keypoints``
  key, or without a visible joint, becomes a person whose keypoints are all zeros (positions
  and visibility). Such a person has no ground truth: the cache marks it (``has_gt``) and the loop
  reports no metrics for a video that has one.
* ``frame_index``. Position of the frame among the ``images`` of the annotation file when
  they are sorted by ``file_name``.
* Sample order. PoseTrack21 and JRDB-Pose use the 2023 sort key
  ``int(str(ann_id)[-n:] + str(image_id))`` (``n = 2`` for PoseTrack21, ``3`` for
  JRDB-Pose) over the people listed by ascending image id, so sample indices match the
  2023 query lists. ``coco_custom`` orders by ``(track_id, frame_index)``.
* Track id. PoseTrack21 prefixes the video id (``"000342" + "0"``), as the 2023 code
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
        sample_index: Position of the person in its :class:`VideoAnnotation`, the sample index.
        ann_id: Annotation id from the JSON file.
        image_id: Image id from the JSON file.
        frame_index: Position of the frame in the video, sorted by ``file_name``.
        track_id: Person identity within the video.
        image_path: Absolute path of the image file of the frame.
        image_size: ``(width, height)`` of the image in pixels.
        bbox_xyxy: ``(4,)`` float64 box, converted and clipped by the 2023 convention. In a
            panorama it is not clipped at the left and right edge (see the module docstring).
        keypoints: ``(K, 3)`` float64 array of x, y and visibility in ``{0, 1}``. All zeros for a
            ``coco_custom`` person without a visible joint (no ground truth). In a panorama the
            x of a joint is in the unwrapped coordinates of the box of the person.
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

    @property
    def has_gt(self) -> bool:
        """True if the person has a visible ground-truth joint.

        False for a ``coco_custom`` box without keypoints, whose keypoints are all zeros. The
        cache, the loop, ``vatl4pose-tune`` and the training and evaluation of a pose estimator all
        use this one definition.
        """
        return bool(self.keypoints[:, 2].any())

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
    """All people of one video, in sample order, with their temporal neighbors.

    Attributes:
        dataset: ``"posetrack21"``, ``"jrdb_pose"`` or ``"coco_custom"``.
        video_id: Video id the reader was called with.
        spec: The keypoint spec of the dataset.
        people: The people in sample order; ``people[i].sample_index == i``.
        prev_index: ``(N,)`` int64. Index of the person of the same track in the nearest earlier
            annotated frame, or ``-1``.
        next_index: ``(N,)`` int64. Index of the person of the same track in the nearest later
            annotated frame, or ``-1``.
        annotation_path: Absolute path of the JSON file the people were read from.
        panorama: Whether the frames are 360-degree panoramas and the people are placed in the
            unwrapped coordinates of the strip (see the module docstring). The crops of the cache
            and of the training read the frames the same way.
    """

    dataset: str
    video_id: str
    spec: KeypointSpec
    people: tuple[PersonRecord, ...]
    prev_index: np.ndarray
    next_index: np.ndarray
    annotation_path: pathlib.Path
    panorama: bool = False

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
    """Link every person to the same track in the nearest earlier and later annotated frame.

    A frame where the track has no person is skipped, so a gap is bridged. People of
    different tracks are never linked.

    Args:
        people: The people of one video, in any order.

    Returns:
        ``(prev_index, next_index)``, both ``(N,)`` int64. Entry ``i`` holds the position
        in ``people`` of the neighbor of ``people[i]``, or ``-1`` if there is none.

    Raises:
        ValueError: If a track has two people in one frame; the neighbor is ambiguous.
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


def _legacy_box_xyxy(
    xywh: np.ndarray, width: int, height: int, panorama: bool = False
) -> np.ndarray:
    """xywh to xyxy with ``max(size - 1, 0)`` extents, clipped to ``[0, size - 1]``.

    With ``panorama`` only the y coordinates are clipped; x is left as the box has it.
    """
    x, y, w, h = xywh
    xyxy = np.array(
        [x, y, x + max(w - 1.0, 0.0), y + max(h - 1.0, 0.0)], dtype=np.float64
    )
    limits = np.array([width - 1, height - 1, width - 1, height - 1], dtype=np.float64)
    clipped = np.minimum(limits, np.maximum(0.0, xyxy))
    if panorama:
        clipped[[0, 2]] = xyxy[[0, 2]]
    return clipped


def unwrap_keypoints(
    keypoints: np.ndarray, bbox_xyxy: np.ndarray, width: int
) -> np.ndarray:
    """Move every joint with a position by a whole number of frame widths next to the box.

    Every joint with a position moves to the copy nearest to the center of the box (rule: module
    docstring, Panorama). A joint exactly half a frame width away goes to the copy with the larger
    x, which ``np.round`` would not do. Joints at ``(0, 0)`` (not annotated) and the visibility
    column stay. Used by the readers and by the COCO-file oracle.

    Args:
        keypoints: ``(K, 3)`` float array of x, y and visibility.
        bbox_xyxy: ``(4,)`` float box of the person, not clipped at the left and right edge.
        width: Width of the frame in pixels.

    Returns:
        A new ``(K, 3)`` array.
    """
    center = 0.5 * (bbox_xyxy[0] + bbox_xyxy[2])
    has_position = (keypoints[:, 0] != 0.0) | (keypoints[:, 1] != 0.0)
    turns = np.floor((center - keypoints[:, 0]) / width + 0.5)
    moved = keypoints.copy()
    moved[:, 0] = np.where(
        has_position, keypoints[:, 0] + turns * width, keypoints[:, 0]
    )
    return moved


def _collect(
    json_path: pathlib.Path,
    root: pathlib.Path,
    spec: KeypointSpec,
    *,
    skip_empty_box: bool,
    skip_without_bbox: bool,
    require_keypoints: bool,
    panorama: bool = False,
) -> list[_Raw]:
    """Read the people of one annotation file, ordered by image id and then by file order.

    With ``require_keypoints`` (the paper datasets) an annotation needs the ``keypoints`` key and
    is skipped without a visible joint. Without it (``coco_custom``) the annotation is kept and
    gets all-zero keypoints if the key is missing or no joint is visible. With ``panorama`` the
    box is not clipped at the left and right edge and the joints move next to it (see the module
    docstring).
    """
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
            bbox_xyxy = _legacy_box_xyxy(bbox_json, width, height, panorama)
            if skip_empty_box and (
                bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]
            ):
                continue

            if require_keypoints or "keypoints" in ann:
                flat = np.asarray(_require(ann, "keypoints", owner), dtype=np.float64)
                if flat.shape != (num_values,):
                    raise ValueError(
                        f"{owner}: keypoints has {flat.size} values, expected {num_values} "
                        f"for spec {spec.name!r} ({spec.num_joints} joints)"
                    )
                keypoints = flat.reshape(spec.num_joints, 3).copy()
                keypoints[:, 2] = np.minimum(1.0, keypoints[:, 2])
            else:
                keypoints = np.zeros((spec.num_joints, 3), dtype=np.float64)
            if keypoints[:, 2].sum() < 1.0:
                if require_keypoints:
                    continue
                keypoints[:] = 0.0  # no ground truth: all zeros, whatever positions the file holds
            elif panorama:
                keypoints = unwrap_keypoints(keypoints, bbox_xyxy, width)

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
    panorama: bool,
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
        panorama=panorama,
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
    """Sort key of the 2023 code: the last ``digits`` digits of the annotation id, then image id."""

    def key(raw: _Raw) -> int:
        return int(str(raw.ann_id)[-digits:] + str(raw.image_id))

    return key


def read_posetrack21_video(
    root: pathlib.Path,
    video_id: str,
    split: str = "val",
    spec: KeypointSpec | None = None,
) -> VideoAnnotation:
    """Read one PoseTrack21 video.

    Args:
        root: PoseTrack21 root; the annotation file is
            ``root/activelearning/<split>/<video_id>_*.json`` and image paths are
            ``root/<file_name>``.
        video_id: Video id such as ``"000342"``.
        split: Sub-directory of ``activelearning``.
        spec: Keypoint spec; the ``posetrack21`` preset if omitted. A spec with another joint
            count than the annotation files raises.

    Raises:
        ValueError: If the annotation file is not unique, the JSON is malformed or its keypoints
            do not match the joint count of ``spec``.
        FileNotFoundError: If an image listed in the JSON does not exist.
    """
    root = pathlib.Path(root).resolve()
    spec = get_spec("posetrack21") if spec is None else spec
    json_path = _find_annotation_file(root / "activelearning" / split, video_id)
    raws = _collect(
        json_path,
        root,
        spec,
        skip_empty_box=True,
        skip_without_bbox=False,
        require_keypoints=True,
    )
    raws.sort(key=_legacy_sort_key(2))
    return _build(
        "posetrack21",
        video_id,
        spec,
        json_path,
        raws,
        lambda raw: f"{video_id}{raw.track}",
        panorama=False,
    )


def read_jrdb_pose_video(
    root: pathlib.Path,
    video_id: str,
    split: str = "test",
    spec: KeypointSpec | None = None,
    panorama: bool = True,
) -> VideoAnnotation:
    """Read one JRDB-Pose video.

    The frames of JRDB-Pose are 360-degree panoramas, so the reader treats the seam as part of
    the frame by default (``panorama``, see the module docstring). ``panorama=False`` gives the
    boxes and joints of the 2023 code, which clips the boxes at the left and right edge and
    leaves the joints on the side of the frame where the file records them.

    Args:
        root: JRDB-Pose root; the annotation file is
            ``root/activelearning/<split>/<video_id>_*.json`` and image paths are
            ``root/<file_name>`` (for example ``images/image_stitched/<scene>/000001.jpg``).
        video_id: Video id such as ``"00"``.
        split: Sub-directory of ``activelearning``.
        spec: Keypoint spec; the ``jrdb_pose`` preset if omitted. A spec with another joint
            count than the annotation files raises.
        panorama: Place the people in the unwrapped coordinates of the 360-degree strip.

    Raises:
        ValueError: If the annotation file is not unique, the JSON is malformed or its keypoints
            do not match the joint count of ``spec``.
        FileNotFoundError: If an image listed in the JSON does not exist.
    """
    root = pathlib.Path(root).resolve()
    spec = get_spec("jrdb_pose") if spec is None else spec
    json_path = _find_annotation_file(root / "activelearning" / split, video_id)
    raws = _collect(
        json_path,
        root,
        spec,
        skip_empty_box=False,
        skip_without_bbox=True,
        require_keypoints=True,
        panorama=panorama,
    )
    raws.sort(key=_legacy_sort_key(3))
    return _build(
        "jrdb_pose",
        video_id,
        spec,
        json_path,
        raws,
        lambda raw: str(raw.track),
        panorama=panorama,
    )


def read_coco_custom_video(
    root: pathlib.Path, video_id: str, spec: KeypointSpec, panorama: bool = False
) -> VideoAnnotation:
    """Read one video of a ``coco_custom`` dataset (layout in the module docstring).

    Every annotation with a valid box becomes a person. An annotation without a ``keypoints`` key,
    or with no visible joint, becomes a person with all-zero keypoints, which has no ground
    truth.

    Args:
        root: Dataset root; the annotation file is ``root/annotations/<video_id>.json``
            and image paths are ``root/<file_name>``.
        video_id: Video id.
        spec: Keypoint spec; a ``keypoints`` list that is given needs ``3 * spec.num_joints``
            values.
        panorama: Whether the frames are 360-degree panoramas (an own 360-degree video). The people
            are then placed in the unwrapped coordinates of the strip (see the module docstring).

    Raises:
        ValueError: If the JSON is malformed or the track ids are not all ints or all strings.
        FileNotFoundError: If the annotation file or an image does not exist.
    """
    root = pathlib.Path(root).resolve()
    json_path = root / "annotations" / f"{video_id}.json"
    raws = _collect(
        json_path,
        root,
        spec,
        skip_empty_box=True,
        skip_without_bbox=False,
        require_keypoints=False,
        panorama=panorama,
    )
    kinds = {type(raw.track) for raw in raws}
    if len(kinds) > 1 or not kinds <= {int, str}:
        raise ValueError(
            f"{json_path.name}: track_id values must be all int or all str, got "
            f"{sorted(kind.__name__ for kind in kinds)}"
        )
    raws.sort(key=lambda raw: (raw.track, raw.frame_index))
    return _build(
        "coco_custom",
        video_id,
        spec,
        json_path,
        raws,
        lambda raw: str(raw.track),
        panorama=panorama,
    )
