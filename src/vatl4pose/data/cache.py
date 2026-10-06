"""Per-video HDF5 cache of person crops and annotations.

Inference needs the same normalised crop of every person in every active-learning round, and
the crop has no random part. :func:`build_video_cache` therefore crops every annotated person
of one video once and stores the crops together with the annotation arrays in one HDF5 file;
:class:`VideoCache` reads that file back. Fine-tuning does not use the stored crops: it reads the
original frames, because its augmentation needs the pixels around the person.

File layout (``N`` samples, ``K`` joints, ``T`` angles of the keypoint spec)::

    crops           uint8   (N, 256, 192, 3)  RGB, one chunk per sample, lzf compression
    bbox_crop       float64 (N, 4)            xyxy box of the cropped region (float32 values)
    bbox_gt         float64 (N, 4)            xyxy ground-truth box (clipped, as the reader gives it)
    bbox_json_xywh  float64 (N, 4)            raw xywh box of the annotation file
    keypoints_gt    float64 (N, K, 3)         x, y, visibility
    ann_id, image_id, frame_index, prev_index, next_index    int64 (N,)
    track_id, image_path                      variable-length strings (N,); paths are absolute
    hybrid_gt       float64 (N, 2K + T)       hybrid feature of the ground truth (see below)

Attributes: ``cache_version``, ``dataset``, ``video_id``, ``spec_name``, ``spec_sha256`` (of all
fields of the keypoint spec, see :func:`spec_sha256`), ``annotation_sha256`` (of the annotation
file, see :func:`annotation_sha256`), ``created_utc`` and ``vatl4pose_version``. ``VideoCache``
compares the spec digest, so a cache is refused for a spec that differs in any field, for example
in the angle triplets that ``hybrid_gt`` depends on.

``hybrid_gt`` is :func:`~vatl4pose.models.wholebody_ae.compute_hybrid` of ``bbox_json_xywh`` and
``keypoints_gt``, with the visibility as the weight of every joint. It is the training target of
the autoencoder fine-tuning.

Building decodes and crops the frames in a thread pool (OpenCV releases the interpreter lock).
One task handles one frame: it decodes the frame once and crops every person in it. The calling
thread collects the crops, writes them to the file in sample order and is the only thread that
touches the HDF5 file, so the file does not depend on the number of workers. The file is written
under ``<path>.partial`` and moved to ``<path>`` after it is closed, so a reader never sees a
half-written cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import h5py
import numpy as np
import torch

import vatl4pose
from vatl4pose.keypoints import KeypointSpec
from vatl4pose.models.wholebody_ae import compute_hybrid

from .annotations import PersonRecord, VideoAnnotation
from .geometry import INPUT_SIZE, crop_person

CACHE_VERSION: int = 1
"""Version of the file layout; a cache of another version is refused."""

# Frames handled per round of parallel work, per worker. The crops of a round wait in memory
# (147 kB each) until the main thread has written them.
_FRAMES_PER_WORKER = 4

_CROP_SHAPE = (*INPUT_SIZE, 3)


def read_image_rgb(path: pathlib.Path) -> np.ndarray:
    """Read a frame as ``(H, W, 3)`` uint8 RGB, the way the 2023 code did.

    The frame is decoded with ``cv2.imread`` (which applies the EXIF orientation) and converted
    from BGR to RGB.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
        ValueError: If OpenCV cannot decode the file.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image file {path} does not exist")
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"OpenCV cannot decode image file {path}")
    assert bgr.dtype == np.uint8 and bgr.ndim == 3 and bgr.shape[2] == 3, bgr.shape
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def spec_sha256(spec: KeypointSpec) -> str:
    """SHA-256 of a keypoint spec: the canonical JSON of all its fields.

    The JSON has sorted keys, no whitespace and tuples as lists. The digest changes when any field
    changes, including ones that only the cached features depend on, such as ``angle_triplets``.
    """
    text = json.dumps(dataclasses.asdict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def annotation_sha256(video: VideoAnnotation) -> str:
    """SHA-256 of the annotation file ``video`` was read from."""
    return hashlib.sha256(video.annotation_path.read_bytes()).hexdigest()


def _crop_frame(frame: list[PersonRecord]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decode one frame and crop every person of ``frame`` from it (runs in a worker thread).

    Args:
        frame: Samples that all show the same image file; at least one.

    Returns:
        ``(crop, bbox_crop)`` of every sample, in the order of ``frame``.
    """
    image = read_image_rgb(frame[0].image_path)
    height, width = image.shape[:2]
    for person in frame:
        if (width, height) != person.image_size:
            raise ValueError(
                f"image {person.image_path} has size (width, height) = {(width, height)}, "
                f"but the annotation of sample {person.sample_index} says {person.image_size}"
            )
    return [crop_person(image, person.bbox_xyxy) for person in frame]


def _create_strings(handle: h5py.File, name: str, values: list[str]) -> None:
    handle.create_dataset(
        name, data=np.array(values, dtype=object), dtype=h5py.string_dtype()
    )


def _write_cache(video: VideoAnnotation, path: pathlib.Path, num_workers: int) -> None:
    people = video.people
    n = len(people)
    spec = video.spec
    bbox_crop = np.empty((n, 4), dtype=np.float64)
    keypoints_gt = np.stack([p.keypoints for p in people])
    bbox_json_xywh = np.stack([p.bbox_json_xywh for p in people])
    hybrid_gt = compute_hybrid(bbox_json_xywh, keypoints_gt, spec)

    with h5py.File(path, "w") as handle:
        handle.attrs["cache_version"] = CACHE_VERSION
        handle.attrs["dataset"] = video.dataset
        handle.attrs["video_id"] = video.video_id
        handle.attrs["spec_name"] = spec.name
        handle.attrs["spec_sha256"] = spec_sha256(spec)
        handle.attrs["annotation_sha256"] = annotation_sha256(video)
        handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        handle.attrs["vatl4pose_version"] = vatl4pose.__version__

        crops = handle.create_dataset(
            "crops",
            shape=(n, *_CROP_SHAPE),
            dtype=np.uint8,
            chunks=(1, *_CROP_SHAPE),
            compression="lzf",
        )
        # Samples of one frame are spread over the video (the sample order is not the frame
        # order), so the crops are collected and written as soon as the next sample in order is
        # complete. That keeps the crops in the file in sample order, which is the order of
        # every full read.
        samples_of_frame: dict[pathlib.Path, list[int]] = {}
        for position, person in enumerate(people):
            samples_of_frame.setdefault(person.image_path, []).append(position)
        frames = list(samples_of_frame.values())
        waiting: dict[int, np.ndarray] = {}
        written = 0
        step = _FRAMES_PER_WORKER * num_workers
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for start in range(0, len(frames), step):
                batch = frames[start : start + step]
                # ``map`` yields the results in the order of its input, whatever the order in
                # which the workers finish.
                results = pool.map(
                    _crop_frame, ([people[i] for i in frame] for frame in batch)
                )
                for frame, frame_results in zip(batch, results, strict=True):
                    for position, (crop, box) in zip(frame, frame_results, strict=True):
                        waiting[position] = crop
                        bbox_crop[position] = box
                while written in waiting:
                    crops[written] = waiting.pop(written)
                    written += 1
        assert written == n and not waiting, (written, n, len(waiting))

        handle.create_dataset("bbox_crop", data=bbox_crop)
        handle.create_dataset("bbox_gt", data=np.stack([p.bbox_xyxy for p in people]))
        handle.create_dataset("bbox_json_xywh", data=bbox_json_xywh)
        handle.create_dataset("keypoints_gt", data=keypoints_gt)
        handle.create_dataset(
            "ann_id", data=np.array([p.ann_id for p in people], dtype=np.int64)
        )
        handle.create_dataset(
            "image_id", data=np.array([p.image_id for p in people], dtype=np.int64)
        )
        handle.create_dataset(
            "frame_index",
            data=np.array([p.frame_index for p in people], dtype=np.int64),
        )
        _create_strings(handle, "track_id", [p.track_id for p in people])
        handle.create_dataset("prev_index", data=video.prev_index)
        handle.create_dataset("next_index", data=video.next_index)
        _create_strings(handle, "image_path", [str(p.image_path) for p in people])
        handle.create_dataset("hybrid_gt", data=hybrid_gt)


def build_video_cache(
    video: VideoAnnotation, out_path: pathlib.Path, num_workers: int = 8
) -> None:
    """Crop every person of ``video`` and write the cache file ``out_path``.

    The frames are decoded and cropped by ``num_workers`` threads, each frame once however many
    people it shows. The file is the same for every ``num_workers``. It is written to ``<out_path>.partial`` and renamed after it is closed; if
    anything fails, the partial file is removed and ``out_path`` stays as it was. An existing
    ``out_path`` is replaced.

    Args:
        video: Samples of one video (any dataset).
        out_path: Target file; its directory is created.
        num_workers: Number of threads that decode and crop frames.

    Raises:
        ValueError: If ``num_workers`` is below 1, the video has no samples, a frame cannot be
            decoded, or a frame has another size than its annotation says.
        FileNotFoundError: If a frame file does not exist.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")
    if len(video.people) == 0:
        raise ValueError(
            f"video {video.video_id!r} ({video.annotation_path}) has no samples; "
            "there is nothing to cache"
        )
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_name(out_path.name + ".partial")
    try:
        _write_cache(video, partial, num_workers)
        os.replace(partial, out_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class CacheArrays:
    """All non-image arrays of a cache, in memory. Row ``i`` is sample ``i`` of the video.

    Attributes:
        bbox_crop: ``(N, 4)`` float64 xyxy box of the cropped region.
        bbox_gt: ``(N, 4)`` float64 xyxy ground-truth box.
        bbox_json_xywh: ``(N, 4)`` float64 raw JSON xywh box.
        keypoints_gt: ``(N, K, 3)`` float64 ground truth (x, y, visibility) in image pixels.
        ann_id: ``(N,)`` int64 annotation ids.
        image_id: ``(N,)`` int64 image ids.
        frame_index: ``(N,)`` int64 position of the frame in the video.
        track_id: ``(N,)`` object array of str.
        prev_index: ``(N,)`` int64 sample of the same track in the previous annotated frame
            or ``-1``.
        next_index: ``(N,)`` int64, likewise for the next annotated frame.
        image_path: ``(N,)`` object array of absolute path strings.
        hybrid_gt: ``(N, 2K + T)`` float64 hybrid feature of the ground truth.
    """

    bbox_crop: np.ndarray
    bbox_gt: np.ndarray
    keypoints_gt: np.ndarray
    ann_id: np.ndarray
    image_id: np.ndarray
    frame_index: np.ndarray
    track_id: np.ndarray
    prev_index: np.ndarray
    next_index: np.ndarray
    image_path: np.ndarray
    hybrid_gt: np.ndarray
    bbox_json_xywh: np.ndarray


class VideoCache:
    """Read access to a cache file written by :func:`build_video_cache`.

    The annotation arrays are loaded into memory on construction. The crops stay on disk; each
    call to :meth:`crops` opens the file, reads the requested crops and closes it again, so an
    instance can be copied into data-loader worker processes.

    Args:
        path: Cache file.
        expected_spec: Keypoint spec the caller works with; the cache must have been built for it.
        expected_annotation_sha256: SHA-256 of the annotation file the caller works with. If given,
            the cache must have been built from a file with this digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a cache, has another cache version, was built for another
            spec (another name, joint count or any other spec field, see :func:`spec_sha256`), or
            (when ``expected_annotation_sha256`` is given) from another annotation file, or if a
            dataset of the file has an unexpected shape or dtype.
    """

    def __init__(
        self,
        path: pathlib.Path,
        expected_spec: KeypointSpec,
        expected_annotation_sha256: str | None = None,
    ) -> None:
        self._path = pathlib.Path(path)
        self._spec = expected_spec
        with h5py.File(self._path, "r") as handle:
            self._check_attributes(handle, expected_spec, expected_annotation_sha256)
            self._dataset = self._text_attribute(handle, "dataset")
            self._video_id = self._text_attribute(handle, "video_id")
            self._arrays = self._load_arrays(handle, expected_spec)
            self._length = int(handle["crops"].shape[0])

    @property
    def path(self) -> pathlib.Path:
        """The cache file."""
        return self._path

    @property
    def spec(self) -> KeypointSpec:
        """The keypoint spec the cache was checked against (it matches the one it was built for)."""
        return self._spec

    @property
    def dataset(self) -> str:
        """Name of the dataset the cache was built from, e.g. ``"posetrack21"``."""
        return self._dataset

    @property
    def video_id(self) -> str:
        """Identifier of the video the cache was built from."""
        return self._video_id

    def __len__(self) -> int:
        return self._length

    @property
    def arrays(self) -> CacheArrays:
        """All non-image arrays, loaded once."""
        return self._arrays

    def crops(self, index: np.ndarray) -> torch.Tensor:
        """Read the crops of the samples ``index``.

        Args:
            index: ``(B,)`` integer array of sample indices, in any order, repeats allowed.

        Returns:
            ``(B, 256, 192, 3)`` uint8 RGB tensor.

        Raises:
            IndexError: If an index is outside ``[0, len(self))``.
        """
        idx = np.asarray(index)
        if idx.ndim != 1 or idx.dtype.kind not in "iu":
            raise ValueError(
                f"index: expected a 1-D integer array, got dtype {idx.dtype} and shape {idx.shape}"
            )
        if idx.size and (idx.min() < 0 or idx.max() >= self._length):
            raise IndexError(
                f"index out of range [0, {self._length}): "
                f"got values from {idx.min()} to {idx.max()}"
            )
        out = np.empty((idx.size, *_CROP_SHAPE), dtype=np.uint8)
        with h5py.File(self._path, "r") as handle:
            crops = handle["crops"]
            for row, sample in enumerate(idx):
                out[row] = crops[int(sample)]
        return torch.from_numpy(out)

    def _text_attribute(self, handle: h5py.File, name: str) -> str:
        if name not in handle.attrs:
            raise ValueError(
                f"{self._path} is not a vatl4pose cache: attribute {name!r} is missing"
            )
        return str(handle.attrs[name])

    def _check_attributes(
        self,
        handle: h5py.File,
        spec: KeypointSpec,
        expected_annotation_sha256: str | None,
    ) -> None:
        def attribute(name: str) -> object:
            if name not in handle.attrs:
                raise ValueError(
                    f"{self._path} is not a vatl4pose cache: attribute {name!r} is missing"
                )
            return handle.attrs[name]

        version = int(attribute("cache_version"))
        if version != CACHE_VERSION:
            raise ValueError(
                f"{self._path} has cache version {version}, this code reads version "
                f"{CACHE_VERSION}; rebuild the cache"
            )
        built_for = str(attribute("spec_name"))
        if built_for != spec.name:
            raise ValueError(
                f"{self._path} was built for spec {built_for!r}, expected {spec.name!r}"
            )
        num_joints = int(handle["keypoints_gt"].shape[1])
        if num_joints != spec.num_joints:
            raise ValueError(
                f"{self._path} holds {num_joints} joints per sample, spec {spec.name!r} "
                f"has {spec.num_joints}"
            )
        built_hash = str(attribute("spec_sha256"))
        expected_hash = spec_sha256(spec)
        if built_hash != expected_hash:
            raise ValueError(
                f"{self._path} was built for a spec {spec.name!r} with SHA-256 {built_hash}, "
                f"but the spec given has SHA-256 {expected_hash}; a field of the spec changed "
                "(for example the angle triplets), rebuild the cache"
            )
        if expected_annotation_sha256 is not None:
            digest = str(attribute("annotation_sha256"))
            if digest != expected_annotation_sha256:
                raise ValueError(
                    f"{self._path} was built from an annotation file with SHA-256 {digest}, "
                    f"expected {expected_annotation_sha256}; the annotation file changed, "
                    "rebuild the cache"
                )

    def _load_arrays(self, handle: h5py.File, spec: KeypointSpec) -> CacheArrays:
        n = int(handle["crops"].shape[0])
        if tuple(handle["crops"].shape) != (n, *_CROP_SHAPE):
            raise ValueError(
                f"{self._path}: crops have shape {tuple(handle['crops'].shape)}, "
                f"expected ({n}, {_CROP_SHAPE[0]}, {_CROP_SHAPE[1]}, {_CROP_SHAPE[2]})"
            )

        def read(name: str, kind: str, tail: tuple[int, ...]) -> np.ndarray:
            if name not in handle:
                raise ValueError(f"{self._path}: dataset {name!r} is missing")
            dataset = handle[name]
            if tuple(dataset.shape) != (n, *tail):
                raise ValueError(
                    f"{self._path}: {name} has shape {tuple(dataset.shape)}, "
                    f"expected {(n, *tail)}"
                )
            if kind == "str":
                if h5py.check_string_dtype(dataset.dtype) is None:
                    raise ValueError(
                        f"{self._path}: {name} has dtype {dataset.dtype}, expected strings"
                    )
                values = dataset.asstr()[()]
                if values.dtype != object:
                    raise ValueError(
                        f"{self._path}: {name} did not load as str objects, got {values.dtype}"
                    )
                return values
            if dataset.dtype != np.dtype(kind):
                raise ValueError(
                    f"{self._path}: {name} has dtype {dataset.dtype}, expected {kind}"
                )
            return dataset[()]

        num_joints = spec.num_joints
        arrays: dict[str, np.ndarray] = {}
        layout: dict[str, tuple[str, tuple[int, ...]]] = {
            "bbox_crop": ("float64", (4,)),
            "bbox_gt": ("float64", (4,)),
            "bbox_json_xywh": ("float64", (4,)),
            "keypoints_gt": ("float64", (num_joints, 3)),
            "ann_id": ("int64", ()),
            "image_id": ("int64", ()),
            "frame_index": ("int64", ()),
            "track_id": ("str", ()),
            "prev_index": ("int64", ()),
            "next_index": ("int64", ()),
            "image_path": ("str", ()),
            "hybrid_gt": ("float64", (spec.ae_input_dim,)),
        }
        for name, (kind, tail) in layout.items():
            arrays[name] = read(name, kind, tail)
        return CacheArrays(**arrays)

