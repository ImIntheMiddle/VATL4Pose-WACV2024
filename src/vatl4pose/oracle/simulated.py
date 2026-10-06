"""The simulated oracle: the human is replaced by the ground truth stored in the cache."""

from __future__ import annotations

import numpy as np

from vatl4pose.data.cache import VideoCache

from .base import OracleResult, check_request


_MAX_IDS_SHOWN = 10


class SimulatedOracle:
    """Answers every request at once with the cached ground truth (paper reproduction).

    A person without ground truth (``has_gt`` False, a box without keypoints) cannot be answered:
    its cached keypoints are zeros, not labels. A request for such a person raises.

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

        Raises:
            ValueError: If the request is invalid, or a requested person has no ground truth; the
                message names the annotation ids of those people (at most ten, then the count of
                the rest).
        """
        check_request(self._arrays, indices, round_index, predictions)
        without = indices[~self._arrays.has_gt[indices]]
        if without.size:
            ids = self._arrays.ann_id[without].tolist()
            shown = ", ".join(str(i) for i in ids[:_MAX_IDS_SHOWN])
            if len(ids) > _MAX_IDS_SHOWN:
                shown += f" and {len(ids) - _MAX_IDS_SHOWN} more"
            raise ValueError(
                f"round {round_index:02d}: {len(ids)} of the {len(indices)} requested people have "
                f"no ground-truth keypoints (annotation ids {shown}), so the simulated oracle has "
                "no answer for them. Use oracle=coco_file to annotate a video without ground truth."
            )
        keypoints = self._arrays.keypoints_gt[indices]
        assert keypoints.dtype == np.float64, keypoints.dtype
        assert keypoints.shape == (len(indices), *self._arrays.keypoints_gt.shape[1:])
        return OracleResult(
            ready=True,
            keypoints=keypoints,
            message=f"round {round_index:02d}: answered {len(indices)} queries "
            "with the cached ground truth",
        )
