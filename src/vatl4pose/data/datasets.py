"""Torch datasets over a :class:`~vatl4pose.data.cache.VideoCache` and over annotated videos.

* :class:`InferenceDataset` yields the normalised cached crop of every sample.
* :class:`FinetuneDataset` yields augmented training items of the labeled samples. It reads the
  original frame of each sample, crops it from the ground-truth box with a random scale, rotation
  and flip, and builds the heatmap target from the labels given by the oracle.

The initial training of the estimator has no cache, because the people of a whole training split
would need about 85 kB each. Two datasets read the frames of many :class:`VideoAnnotation` objects
instead:

* :class:`PersonTrainDataset` yields augmented training items of every annotated person. Its
  augmentation is drawn like that of :class:`FinetuneDataset`.
* :class:`PersonCropDataset` yields the normalised crop of every person and the box of the crop,
  the same crop that the active-learning loop uses.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from vatl4pose.keypoints import KeypointSpec

from .annotations import PersonRecord, VideoAnnotation
from .augment import AugmentConfig, augment_and_crop
from .cache import VideoCache, read_image_rgb
from .geometry import (
    HEATMAP_SIZE,
    INPUT_SIZE,
    crop_person,
    make_targets,
    to_input_tensor,
)


def collect_people(
    videos: Sequence[VideoAnnotation],
) -> tuple[KeypointSpec, tuple[PersonRecord, ...]]:
    """The people of all ``videos`` in order (video after video), and their keypoint spec.

    The training and the evaluation of an estimator read the keypoints of every person as labels,
    so a person without a visible keypoint (a ``coco_custom`` box without keypoints) is refused.
    The videos must agree on ``panorama``: the datasets below crop every person the same way.

    Raises:
        ValueError: If the videos do not share one keypoint spec or one ``panorama`` setting, hold
            no person, or hold a person without a visible keypoint.
    """
    specs = {video.spec for video in videos}
    if len(specs) > 1:
        raise ValueError(
            "all videos must have the same keypoint spec, got "
            f"{sorted(spec.name for spec in specs)}"
        )
    panoramas = {video.panorama for video in videos}
    if len(panoramas) > 1:
        raise ValueError(
            "all videos must have the same panorama setting, got "
            f"{[(video.video_id, video.panorama) for video in videos]}"
        )
    people = tuple(person for video in videos for person in video.people)
    if not people:
        raise ValueError(f"the {len(videos)} video(s) hold no person")
    for video in videos:
        without = [p.ann_id for p in video.people if not p.has_gt]
        if without:
            raise ValueError(
                f"video {video.video_id!r}: {len(without)} of {len(video.people)} people have no "
                f"visible keypoint (annotation ids {without[:10]}); training and evaluation need "
                "ground-truth keypoints for every person"
            )
    return next(iter(specs)), people


class InferenceDataset(Dataset[tuple[int, torch.Tensor]]):
    """Every sample of a cache, as ``(index, normalised model input)``.

    Args:
        cache: The cache to read.
    """

    def __init__(self, cache: VideoCache) -> None:
        self._cache = cache

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, i: int) -> tuple[int, torch.Tensor]:
        """Return ``(i, x)`` with ``x`` of shape ``(3, 256, 192)``, float32."""
        i = int(i)
        if not 0 <= i < len(self):
            raise IndexError(f"index {i} out of range [0, {len(self)})")
        x = to_input_tensor(self._cache.crops(np.array([i], dtype=np.int64)))[0]
        assert x.dtype == torch.float32 and tuple(x.shape) == (3, *INPUT_SIZE)
        return i, x


class FinetuneDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Augmented training items of the labeled samples of one video.

    Item ``i`` belongs to the sample ``indices[i]`` of the cache. Its augmentation is drawn from
    ``np.random.default_rng((seed, round_index, epoch, i))``, so an item is the same whenever it
    is read in the same epoch, whatever the order of reads, the batch size or the number of
    loader workers.

    The epoch is set with :meth:`set_epoch`. It is held in a shared-memory tensor, so the worker
    processes of a :class:`~torch.utils.data.DataLoader` see it, also when the loader keeps its
    workers between epochs (``persistent_workers=True``). Call :meth:`set_epoch` between two
    epochs, not while an iteration is running: a worker may already have prefetched items with
    the previous epoch.

    Args:
        cache: The cache of the video; the original frames and the ground-truth boxes come from
            it. The cached crops are not used. The crops follow the ``panorama`` setting of the
            cache (the frames wrap at the seam, the boxes and labels are in unwrapped coordinates).
        indices: ``(M,)`` int64 sample indices of the training set.
        labels: ``(M, K, 3)`` float array of x, y (image pixels) and visibility (0 or 1), one row
            per entry of ``indices``; the answers of the oracle.
        spec: Keypoint spec, ``K`` joints.
        augment: Augmentation strengths and probabilities.
        seed: Seed of the run; non-negative.
        round_index: Active-learning round; non-negative.

    Raises:
        ValueError: If an argument has a wrong shape, dtype or range, or a label visibility is
            not 0 or 1.
    """

    def __init__(
        self,
        cache: VideoCache,
        indices: np.ndarray,
        labels: np.ndarray,
        spec: KeypointSpec,
        augment: AugmentConfig,
        seed: int,
        round_index: int,
    ) -> None:
        if (
            not isinstance(indices, np.ndarray)
            or indices.ndim != 1
            or indices.dtype != np.int64
        ):
            raise ValueError(
                f"indices: expected a 1-D int64 array, got "
                f"{getattr(indices, 'dtype', type(indices).__name__)} "
                f"{getattr(indices, 'shape', '')}"
            )
        if indices.size and (indices.min() < 0 or indices.max() >= len(cache)):
            raise ValueError(
                f"indices: values must lie in [0, {len(cache)}), got range "
                f"[{indices.min()}, {indices.max()}]"
            )
        num_joints = cache.arrays.keypoints_gt.shape[1]
        if spec.num_joints != num_joints:
            raise ValueError(
                f"spec {spec.name!r} has {spec.num_joints} joints, the cache holds {num_joints}"
            )
        if (
            not isinstance(labels, np.ndarray)
            or not np.issubdtype(labels.dtype, np.floating)
            or labels.shape != (len(indices), num_joints, 3)
        ):
            raise ValueError(
                f"labels: expected a float array of shape ({len(indices)}, {num_joints}, 3), "
                f"got {getattr(labels, 'dtype', type(labels).__name__)} "
                f"{getattr(labels, 'shape', '')}"
            )
        if not np.isin(labels[..., 2], (0.0, 1.0)).all():
            raise ValueError(
                "labels: the visibility (last column) must be 0 or 1, got "
                f"{np.unique(labels[..., 2]).tolist()}"
            )
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        if round_index < 0:
            raise ValueError(f"round_index must be non-negative, got {round_index}")
        self._cache = cache
        self._indices = indices.copy()
        self._labels = labels.copy()
        self._spec = spec
        self._augment = augment
        self._seed = seed
        self._round_index = round_index
        # shared memory: the worker processes of a loader read the epoch that the main process sets
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def seed(self) -> int:
        """The seed of the run, as given to the constructor (read-only)."""
        return self._seed

    @property
    def round_index(self) -> int:
        """The active-learning round, as given to the constructor (read-only)."""
        return self._round_index

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch whose augmentation draws the following reads use.

        Raises:
            ValueError: If ``epoch`` is negative.
        """
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch.fill_(epoch)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(input, target, weight)`` of item ``i``.

        Returns:
            ``input`` ``(3, 256, 192)``, ``target`` ``(K, 64, 48)`` and ``weight`` ``(K, 1, 1)``,
            all float32. The weight is 0 for a joint that is not visible or lies outside the
            heatmap after the augmentation.
        """
        i = int(i)
        if not 0 <= i < len(self):
            raise IndexError(f"index {i} out of range [0, {len(self)})")
        sample = int(self._indices[i])
        arrays = self._cache.arrays
        epoch = int(self._epoch)
        rng = np.random.default_rng((self._seed, self._round_index, epoch, i))
        image = read_image_rgb(pathlib.Path(arrays.image_path[sample]))
        crop, joints, _ = augment_and_crop(
            image,
            arrays.bbox_gt[sample],
            self._labels[i],
            self._spec,
            self._augment,
            rng,
            panorama=self._cache.panorama,
        )
        target, weight = make_targets(joints, self._spec.num_joints)
        x = to_input_tensor(torch.from_numpy(crop).unsqueeze(0))[0]
        target_t, weight_t = torch.from_numpy(target), torch.from_numpy(weight)
        num_joints = self._spec.num_joints
        assert x.dtype == target_t.dtype == weight_t.dtype == torch.float32
        assert tuple(x.shape) == (3, *INPUT_SIZE)
        assert tuple(target_t.shape) == (num_joints, *HEATMAP_SIZE)
        assert tuple(weight_t.shape) == (num_joints, 1, 1)
        return x, target_t, weight_t


class PersonTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Augmented training items of all people of several videos, read from the original frames.

    Item ``i`` belongs to the ``i``-th person of ``videos`` (video after video). It is cropped from
    the ground-truth box of the person, which the readers clipped to the frame (not at the left and
    right edge of a panorama), with a random scale, rotation and flip, and its heatmap target is
    built from the ground-truth joints. The crops follow the ``panorama`` setting of the videos. The
    augmentation of an item is drawn from ``np.random.default_rng((seed, epoch, i))``, so an item
    is the same whenever it is read in the same epoch, whatever the order of reads, the batch size
    or the number of loader workers.

    The epoch is set with :meth:`set_epoch`. It is held in a shared-memory tensor, so the worker
    processes of a :class:`~torch.utils.data.DataLoader` see it, also when the loader keeps its
    workers between epochs (``persistent_workers=True``). Call :meth:`set_epoch` between two
    epochs, not while an iteration is running: a worker may already have prefetched items with
    the previous epoch. This is the mechanism of :class:`FinetuneDataset`.

    Args:
        videos: Videos with the same keypoint spec and ``panorama`` setting and at least one
            person in total.
        augment: Augmentation strengths and probabilities.
        seed: Seed of the run; non-negative.

    Raises:
        ValueError: If ``seed`` is negative, the videos differ in their keypoint spec or their
            ``panorama`` setting, or they hold no person.
    """

    def __init__(
        self, videos: Sequence[VideoAnnotation], augment: AugmentConfig, seed: int
    ) -> None:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        self._spec, self._people = collect_people(videos)
        self._panorama = videos[0].panorama
        self._augment = augment
        self._seed = seed
        # shared memory: the worker processes of a loader read the epoch that the main process sets
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def seed(self) -> int:
        """The seed of the run, as given to the constructor (read-only)."""
        return self._seed

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch whose augmentation draws the following reads use.

        Raises:
            ValueError: If ``epoch`` is negative.
        """
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch.fill_(epoch)

    def __len__(self) -> int:
        return len(self._people)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(input, target, weight)`` of item ``i``.

        Returns:
            ``input`` ``(3, 256, 192)``, ``target`` ``(K, 64, 48)`` and ``weight`` ``(K, 1, 1)``,
            all float32. The weight is 0 for a joint that is not visible or lies outside the
            heatmap after the augmentation.
        """
        i = int(i)
        if not 0 <= i < len(self):
            raise IndexError(f"index {i} out of range [0, {len(self)})")
        person = self._people[i]
        rng = np.random.default_rng((self._seed, int(self._epoch), i))
        crop, joints, _ = augment_and_crop(
            read_image_rgb(person.image_path),
            person.bbox_xyxy,
            person.keypoints,
            self._spec,
            self._augment,
            rng,
            panorama=self._panorama,
        )
        target, weight = make_targets(joints, self._spec.num_joints)
        x = to_input_tensor(torch.from_numpy(crop).unsqueeze(0))[0]
        target_t, weight_t = torch.from_numpy(target), torch.from_numpy(weight)
        num_joints = self._spec.num_joints
        assert x.dtype == target_t.dtype == weight_t.dtype == torch.float32
        assert tuple(x.shape) == (3, *INPUT_SIZE)
        assert tuple(target_t.shape) == (num_joints, *HEATMAP_SIZE)
        assert tuple(weight_t.shape) == (num_joints, 1, 1)
        return x, target_t, weight_t


class PersonCropDataset(Dataset[tuple[int, torch.Tensor, torch.Tensor]]):
    """The normalised crop of every person of several videos, from the ground-truth box.

    Item ``i`` belongs to the ``i``-th person of ``videos`` (video after video). The crop is the
    one of :func:`~vatl4pose.data.geometry.crop_person`, the one the active-learning loop caches.

    Args:
        videos: Videos with the same keypoint spec and ``panorama`` setting and at least one
            person in total.

    Raises:
        ValueError: If the videos differ in their keypoint spec or their ``panorama`` setting, or
            hold no person.
    """

    def __init__(self, videos: Sequence[VideoAnnotation]) -> None:
        _, self._people = collect_people(videos)
        self._panorama = videos[0].panorama

    def __len__(self) -> int:
        return len(self._people)

    def __getitem__(self, i: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Return ``(i, x, bbox_crop)``: ``x`` is ``(3, 256, 192)`` float32, ``bbox_crop`` is the
        ``(4,)`` float32 box ``xmin, ymin, xmax, ymax`` of the cropped region."""
        i = int(i)
        if not 0 <= i < len(self):
            raise IndexError(f"index {i} out of range [0, {len(self)})")
        person = self._people[i]
        crop, bbox_crop = crop_person(
            read_image_rgb(person.image_path), person.bbox_xyxy, self._panorama
        )
        x = to_input_tensor(torch.from_numpy(crop).unsqueeze(0))[0]
        assert x.dtype == torch.float32 and tuple(x.shape) == (3, *INPUT_SIZE)
        assert bbox_crop.dtype == np.float32 and bbox_crop.shape == (4,)
        return i, x, torch.from_numpy(bbox_crop)
