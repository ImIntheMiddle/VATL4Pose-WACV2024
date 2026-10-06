"""The simulated oracle: the human is replaced by the ground truth stored in the cache."""

from __future__ import annotations

import numpy as np

from vatl4pose.data.cache import VideoCache

from .base import OracleResult, check_request


class SimulatedOracle:
    """Answers every request at once with the cached ground truth (paper reproduction).

    Args:
        cache: Cache of the video; its ``keypoints_gt`` are the answers. The cache stores the
            visibility already as 0 or 1.
    """

    def __init__(self, cache: VideoCache) -> None:
        self._arrays = cache.arrays

    def request(
        self, indices: np.ndarray, round_index: int, predictions: np.ndarray
    ) -> OracleResult:
        """Return the ground truth of the people ``indices``, in the order of ``indices``.

        The predictions are not used. The result is a copy, so the caller cannot change the cache.
        """
        check_request(self._arrays, indices, round_index, predictions)
        keypoints = self._arrays.keypoints_gt[indices]
        assert keypoints.dtype == np.float64, keypoints.dtype
        assert keypoints.shape == (len(indices), *self._arrays.keypoints_gt.shape[1:])
        return OracleResult(
            ready=True,
            keypoints=keypoints,
            message=f"round {round_index:02d}: answered {len(indices)} queries "
            "with the cached ground truth",
        )
