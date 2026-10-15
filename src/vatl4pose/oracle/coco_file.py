"""The COCO-file oracle: a human annotates the queries of a round in CVAT or Label Studio.

Every round has two files in ``<run_dir>/oracle/``, named with the round number in two digits::

    round_03_queries.json    written by this oracle
    round_03_answers.json    written by the annotation tool, put in place by the human

``request`` writes the queries file and returns ``ready=False``; the loop stops, the human
annotates, and the next ``request`` of the same round reads the answers file.

An existing queries file is kept as it is if it lists the same queries as the request: the same
``images`` and, for every annotation id, the same ``image_id``, ``file_name`` and ``bbox``. The
prefilled keypoints are not compared, because a rerun of the model can change them by float noise
while the people to annotate stay the same. If the queries differ, ``request`` raises, because an
answers file made for the old queries could otherwise be imported for the new ones.

Queries file (COCO keypoints, the format CVAT and Label Studio import)
----------------------------------------------------------------------
* ``images``: the distinct frames of the queried people, ``file_name`` relative to ``dataset_root``,
  with ``width`` and ``height``.
* ``annotations``: one per queried person, in ascending order of the sample index, so the file
  lists a set of people and does not depend on the order of the request. ``id`` is the annotation
  id of the source data, ``bbox`` is the raw ``[x, y, w, h]`` box of the source annotation (the
  box an annotator sees in the original data), and ``attributes.vatl_sample_index`` is the sample
  index in the video cache. ``keypoints`` is the prediction of the model as the starting point
  for the annotator: each joint is ``x, y, 2`` where the predicted confidence is above 0 and
  ``x, y, 0`` (not labelled, position kept) otherwise.
* ``categories``: one ``person`` category with ``keypoints`` (the joint names of the keypoint
  spec) and ``skeleton`` (the limbs of the spec, 1-based as COCO requires).

Answers file
------------
The same format, as the annotation tool exports it. Only ``annotations[*].id`` and
``annotations[*].keypoints`` are read; ``bbox``, ``images`` and the rest are ignored. Every
requested annotation id must appear exactly once and no other id may appear. ``keypoints`` holds
``3 K`` finite numbers; the visibility of each joint is 0, 1 or 2 and is mapped to 0 (not
labelled) or 1 (labelled; COCO's 1 "occluded" and 2 "visible" both become 1, as in the cache).
The answer keeps the positions the annotator left, also for joints with visibility 0.

Panorama
--------
For a cache of 360-degree frames (``cache.panorama``) the annotator works in the frame, where a
person on the seam is split in two. The queries file therefore holds the prefilled joints at
``x mod width``, and each answer is moved by whole frame widths next to the box of the person
(:func:`~vatl4pose.data.annotations.unwrap_keypoints`), so that the labels are in the unwrapped
coordinates of the box like the ground truth of the readers.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import numpy as np

from vatl4pose.data.annotations import unwrap_keypoints
from vatl4pose.data.cache import VideoCache, read_image_rgb

from .base import OracleResult, check_request

_CATEGORY_ID = 1
_CATEGORY_NAME = "person"
_MAX_IDS_SHOWN = 10


def _show(ids: list[int]) -> str:
    shown = ", ".join(str(i) for i in ids[:_MAX_IDS_SHOWN])
    if len(ids) > _MAX_IDS_SHOWN:
        shown += f" and {len(ids) - _MAX_IDS_SHOWN} more"
    return shown


def _write_atomic(path: pathlib.Path, data: bytes) -> None:
    """As :func:`vatl4pose.train._common.write_atomic`, for bytes, after creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _serialise(payload: dict[str, Any]) -> bytes:
    """The queries file as compact strict JSON (no NaN or infinity) in UTF-8, ending in a newline."""
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def _read_queries(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error})") from error


def _identity(
    payload: Any, path: pathlib.Path
) -> tuple[list[Any], dict[Any, dict[str, Any]]]:
    """The images and, per annotation id, the ``image_id``, ``file_name`` and ``bbox`` of a queries file.

    Raises:
        ValueError: If ``payload`` is not a queries file (a JSON object with an ``images`` list of
            objects and an ``annotations`` list of objects).
    """
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("images"), list)
        or not isinstance(payload.get("annotations"), list)
        or not all(isinstance(x, dict) for x in payload["images"] + payload["annotations"])
    ):
        raise ValueError(
            f"{path}: not a queries file written by this oracle, expected a JSON object with "
            "an 'images' list and an 'annotations' list of objects"
        )
    file_name = {image.get("id"): image.get("file_name") for image in payload["images"]}
    annotations = {
        ann.get("id"): {
            "image_id": ann.get("image_id"),
            "file_name": file_name.get(ann.get("image_id")),
            "bbox": ann.get("bbox"),
        }
        for ann in payload["annotations"]
    }
    return payload["images"], annotations


def _first_difference(
    existing: Any, requested: dict[str, Any], path: pathlib.Path
) -> str | None:
    """Describe the first difference between the queries of two queries files, or return ``None``.

    The queries are the annotation ids with their ``image_id``, ``file_name`` and ``bbox``, and the
    ``images`` list. The order of the annotations and the prefilled keypoints do not count. The
    annotation ids are compared first, then the fields of each annotation in the order of
    ``requested``, then the images.
    """
    old_images, old = _identity(existing, path)
    new_images, new = _identity(requested, path)
    only_old = [i for i in old if i not in new]
    only_new = [i for i in new if i not in old]
    if only_old or only_new:
        parts = []
        if only_old:
            parts.append(f"annotation id(s) only in the file: {_show(only_old)}")
        if only_new:
            parts.append(f"annotation id(s) only in the request: {_show(only_new)}")
        return "; ".join(parts)
    for ann_id, fields in new.items():
        for name, value in fields.items():
            if old[ann_id][name] != value:
                return (
                    f"annotation {ann_id}: {name} is {old[ann_id][name]!r} in the file but "
                    f"{value!r} in the request"
                )
    if old_images != new_images:
        for position, (before, after) in enumerate(zip(old_images, new_images)):
            if before != after:
                return f"images[{position}] is {before!r} in the file but {after!r} in the request"
        return f"images has {len(old_images)} entries in the file but {len(new_images)} in the request"
    return None


class CocoFileOracle:
    """Exports the queries of a round as a COCO keypoints file and imports the human answers.

    See the module docstring for the two file formats.

    Args:
        cache: Cache of the video. Its arrays give the boxes, images and annotation ids and its
            keypoint spec gives the joint names and the skeleton.
        run_dir: Directory of the run; the files go to ``run_dir / "oracle"`` (created on the
            first request).
        dataset_root: Absolute directory that the exported ``file_name`` values are relative to,
            the root of the dataset the cache was built from. Symbolic links in it are resolved,
            as the annotation readers resolve the root, because the cache holds the resolved
            image paths.

    Raises:
        ValueError: If ``dataset_root`` is not an absolute path.
    """

    def __init__(
        self, cache: VideoCache, run_dir: pathlib.Path, dataset_root: pathlib.Path
    ) -> None:
        dataset_root = pathlib.Path(dataset_root)
        if not dataset_root.is_absolute():
            raise ValueError(f"dataset_root: {dataset_root} is not an absolute path")
        self._arrays = cache.arrays
        self._spec = cache.spec
        self._panorama = cache.panorama
        self._dir = pathlib.Path(run_dir) / "oracle"
        self._dataset_root = dataset_root.resolve()

    def request(
        self, indices: np.ndarray, round_index: int, predictions: np.ndarray
    ) -> OracleResult:
        """Write the queries of the round if needed, then read the answers if they exist.

        The first call of a round writes the queries file and returns ``ready=False``; later calls
        do the same until the answers file exists, then return the answers in the order of
        ``indices``.

        Args:
            indices: ``(k,)`` distinct sample indices to annotate.
            round_index: Round number; it names the files.
            predictions: ``(N, K, 3)`` predictions (x, y, confidence); the rows of ``indices``
                prefill the keypoints of the queries file and must be finite.

        Returns:
            The result described above; ``keypoints`` is ``(k, K, 3)`` float64 with visibility
            in ``{0, 1}``.

        Raises:
            ValueError: If the request is invalid, an image lies outside ``dataset_root``, the
                queries file of the round exists but lists other queries (a stale answers file
                could be imported) or is not a queries file, or the answers file is invalid. The
                message names the problem and the first differing annotation id or field.
            FileNotFoundError: If an image file of a query does not exist.
        """
        check_request(self._arrays, indices, round_index, predictions)
        queries = self._dir / f"round_{round_index:02d}_queries.json"
        answers = self._dir / f"round_{round_index:02d}_answers.json"
        payload = self._payload(np.sort(indices), predictions, round_index)
        content = _serialise(payload)
        if queries.exists():
            difference = _first_difference(_read_queries(queries), payload, queries)
            if difference is not None:
                raise ValueError(
                    f"round {round_index:02d}: the queries file {queries} differs from the "
                    f"queries of this request: {difference}. The queried people have changed "
                    "since the file was written, so an answers file made for the old queries "
                    "could be imported for the new ones. Check that the run is the same "
                    f"(config, seed, checkpoints). To start the round over, move {queries} "
                    f"and {answers} away."
                )
        else:
            _write_atomic(queries, content)
        if not answers.exists():
            return OracleResult(
                ready=False,
                keypoints=None,
                message=f"round {round_index:02d}: {len(indices)} queries are in {queries}. "
                "Annotate them (COCO keypoints, for example in CVAT or Label Studio) and save "
                f"the result as {answers}; then run again.",
            )
        keypoints = self._import(answers, indices)
        if self._panorama:
            keypoints = self._unwrap(keypoints, indices, queries)
        return OracleResult(
            ready=True,
            keypoints=keypoints,
            message=f"round {round_index:02d}: read {len(indices)} answers from {answers}",
        )

    # ------------------------------------------------------------------ export

    def _payload(
        self, samples: np.ndarray, predictions: np.ndarray, round_index: int
    ) -> dict[str, Any]:
        """The queries file of ``samples`` (ascending sample indices) as a JSON-ready dict."""
        a = self._arrays
        prefill = predictions[samples]
        if not np.isfinite(prefill).all():
            raise ValueError(
                f"round {round_index:02d}: predictions of the requested samples must be "
                "finite, they are written to the queries file"
            )
        ann_ids = a.ann_id[samples]
        ids, counts = np.unique(ann_ids, return_counts=True)
        if (counts > 1).any():
            raise ValueError(
                f"the cache holds annotation ids {ids[counts > 1].tolist()} more than once "
                "among the requested samples, so answers could not be told apart"
            )

        frames: dict[int, pathlib.Path] = {}
        for sample in samples:
            image_id = int(a.image_id[sample])
            path = pathlib.Path(a.image_path[sample])
            if frames.setdefault(image_id, path) != path:
                raise ValueError(
                    f"the cache gives image id {image_id} two files: {frames[image_id]} "
                    f"and {path}"
                )
        images = [
            self._image_entry(image_id, path) for image_id, path in frames.items()
        ]
        width_of = {image["id"]: image["width"] for image in images}

        annotations: list[dict[str, Any]] = []
        for row, sample in enumerate(samples):
            confidence = prefill[row, :, 2]
            visibility = np.where(confidence > 0, 2, 0).tolist()
            x_drawn = prefill[row, :, 0]
            if self._panorama:
                x_drawn = np.mod(x_drawn, width_of[int(a.image_id[sample])])
            flat = [
                value
                for x, y, v in zip(
                    x_drawn.tolist(),
                    prefill[row, :, 1].tolist(),
                    visibility,
                    strict=True,
                )
                for value in (x, y, v)
            ]
            bbox = a.bbox_json_xywh[sample].tolist()
            annotations.append(
                {
                    "id": int(a.ann_id[sample]),
                    "image_id": int(a.image_id[sample]),
                    "category_id": _CATEGORY_ID,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "num_keypoints": sum(v > 0 for v in visibility),
                    "keypoints": flat,
                    "attributes": {"vatl_sample_index": int(sample)},
                }
            )

        spec = self._spec
        categories = [
            {
                "id": _CATEGORY_ID,
                "name": _CATEGORY_NAME,
                "supercategory": _CATEGORY_NAME,
                "keypoints": list(spec.joint_names),
                "skeleton": [[a_ + 1, b_ + 1] for a_, b_ in spec.skeleton],
            }
        ]
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    def _image_entry(self, image_id: int, path: pathlib.Path) -> dict[str, Any]:
        normalised = pathlib.Path(os.path.normpath(path))
        try:
            relative = normalised.relative_to(self._dataset_root)
        except ValueError:
            raise ValueError(
                f"image {path} lies outside dataset_root {self._dataset_root}; the exported "
                "file_name must be relative to dataset_root"
            ) from None
        height, width = read_image_rgb(path).shape[:2]
        return {
            "id": image_id,
            "file_name": relative.as_posix(),
            "width": int(width),
            "height": int(height),
        }

    # ------------------------------------------------------------------ import

    def _unwrap(
        self, keypoints: np.ndarray, indices: np.ndarray, queries: pathlib.Path
    ) -> np.ndarray:
        """Move the answers of a panorama next to the box of each person (see the module docstring).

        The width of each frame is read from the ``images`` of the queries file that this oracle
        wrote.
        """
        width_of = {
            image["id"]: image["width"] for image in _read_queries(queries)["images"]
        }
        a = self._arrays
        return np.stack(
            [
                unwrap_keypoints(
                    keypoints[row],
                    a.bbox_gt[sample],
                    width_of[int(a.image_id[sample])],
                )
                for row, sample in enumerate(indices)
            ]
        )

    def _import(self, path: pathlib.Path, indices: np.ndarray) -> np.ndarray:
        """Read and check the answers file; return ``(k, K, 3)`` rows in the order of ``indices``."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: not valid JSON ({error})") from error
        annotations = payload.get("annotations") if isinstance(payload, dict) else None
        if not isinstance(annotations, list):
            raise ValueError(
                f"{path}: expected a COCO file, a JSON object with an 'annotations' list"
            )
        by_id: dict[int, dict[str, Any]] = {}
        for position, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise ValueError(
                    f"{path}: annotations[{position}] is not a JSON object"
                )
            ann_id = annotation.get("id")
            if isinstance(ann_id, bool) or not isinstance(ann_id, int):
                raise ValueError(
                    f"{path}: annotations[{position}] has no integer 'id', got {ann_id!r}"
                )
            if ann_id in by_id:
                raise ValueError(
                    f"{path}: annotation id {ann_id} appears more than once"
                )
            by_id[ann_id] = annotation

        requested = self._arrays.ann_id[indices].tolist()
        missing = [i for i in requested if i not in by_id]
        if missing:
            raise ValueError(
                f"{path}: annotation id(s) missing from the answers: {_show(missing)}"
            )
        unexpected = sorted(set(by_id) - set(requested))
        if unexpected:
            raise ValueError(
                f"{path}: annotation id(s) that were not requested: {_show(unexpected)}"
            )
        rows = [self._keypoints_of(path, i, by_id[i]) for i in requested]
        keypoints = np.stack(rows)
        assert keypoints.shape == (len(indices), self._spec.num_joints, 3), (
            keypoints.shape
        )
        assert keypoints.dtype == np.float64, keypoints.dtype
        return keypoints

    def _keypoints_of(
        self, path: pathlib.Path, ann_id: int, annotation: dict[str, Any]
    ) -> np.ndarray:
        """Check the keypoints of one answer; return ``(K, 3)`` float64 with visibility 0 or 1."""
        num_joints = self._spec.num_joints
        values = annotation.get("keypoints")
        if not isinstance(values, list):
            raise ValueError(f"{path}: annotation {ann_id} has no 'keypoints' list")
        if len(values) != 3 * num_joints:
            raise ValueError(
                f"{path}: annotation {ann_id}: keypoints has {len(values)} values, expected "
                f"{3 * num_joints} (x, y, visibility for each of the {num_joints} joints)"
            )
        for position, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{path}: annotation {ann_id}: keypoints[{position}] is not a number, "
                    f"got {value!r}"
                )
        keypoints = np.array(values, dtype=np.float64).reshape(num_joints, 3)
        if not np.isfinite(keypoints).all():
            raise ValueError(
                f"{path}: annotation {ann_id}: keypoints must be finite numbers"
            )
        invalid = ~np.isin(keypoints[:, 2], (0.0, 1.0, 2.0))
        if invalid.any():
            joint = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"{path}: annotation {ann_id}: joint {joint} "
                f"({self._spec.joint_names[joint]!r}) has visibility {keypoints[joint, 2]}, "
                "expected 0, 1 or 2"
            )
        keypoints[:, 2] = np.minimum(keypoints[:, 2], 1.0)
        return keypoints
