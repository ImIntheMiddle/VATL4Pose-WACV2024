"""Torch datasets over a :class:`~vatl4pose.data.cache.VideoCache`.

* :class:`InferenceDataset` yields the normalised cached crop of every sample.
* :class:`FinetuneDataset` yields augmented training items of the labeled samples. It reads the
  original frame of each sample, crops it from the ground-truth box with a random scale, rotation
  and flip, and builds the heatmap target from the labels given by the oracle.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from vatl4pose.keypoints import KeypointSpec

from .augment import AugmentConfig, augment_and_crop
from .cache import VideoCache, read_image_rgb
from .geometry import HEATMAP_SIZE, INPUT_SIZE, make_targets, to_input_tensor


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

    The epoch is set with :meth:`set_epoch`. A :class:`~torch.utils.data.DataLoader` with
    ``num_workers > 0`` copies the dataset into its worker processes when an iteration starts
    (default ``persistent_workers=False``); call :meth:`set_epoch` before the iteration starts.

    Args:
        cache: The cache of the video; the original frames and the ground-truth boxes come from
            it. The cached crops are not used.
        indices: ``(M,)`` int64 sample indices of the training set.
        labels: ``(M, K, 3)`` float array of x, y (image pixels) and visibility, one row per entry
            of ``indices``; the answers of the oracle.
        spec: Keypoint spec, ``K`` joints.
        augment: Augmentation strengths and probabilities.
        seed: Seed of the run; non-negative.
        round_index: Active-learning round; non-negative.

    Raises:
        ValueError: If an argument has a wrong shape, dtype or range.
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
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch whose augmentation draws the following reads use.

        Raises:
            ValueError: If ``epoch`` is negative.
        """
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch = epoch

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
        rng = np.random.default_rng((self._seed, self._round_index, self._epoch, i))
        image = read_image_rgb(pathlib.Path(arrays.image_path[sample]))
        crop, joints, _ = augment_and_crop(
            image,
            arrays.bbox_gt[sample],
            self._labels[i],
            self._spec,
            self._augment,
            rng,
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
