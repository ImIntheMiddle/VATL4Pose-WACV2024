"""The oracle interface of the active-learning loop and the checks every oracle applies.

Each round the loop selects the people whose annotation it wants and sends them to an oracle. The
simulated oracle answers at once with the cached ground truth; the COCO-file oracle hands the
queries to an annotator through files and has no answer until the annotator has put the answers
file in place. :class:`OracleResult` carries both cases: ``ready=False`` means "ask again later".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vatl4pose.data.cache import CacheArrays


@dataclass(frozen=True)
class OracleResult:
    """The answer of an oracle to one request.

    Attributes:
        ready: ``True`` if the answers are available.
        keypoints: ``(k, K, 3)`` float64 array of x, y and visibility with visibility in
            ``{0, 1}``, row ``i`` being the person ``indices[i]`` of the request; ``None`` while
            ``ready`` is ``False``.
        message: Human-readable status, for example where the annotator has to put the answers.
    """

    ready: bool
    keypoints: np.ndarray | None
    message: str

    def __post_init__(self) -> None:
        if not self.ready:
            if self.keypoints is not None:
                raise ValueError("OracleResult: keypoints must be None while ready is False")
            return
        keypoints = self.keypoints
        if keypoints is None:
            raise ValueError("OracleResult: keypoints are missing although ready is True")
        if (
            not isinstance(keypoints, np.ndarray)
            or keypoints.ndim != 3
            or keypoints.shape[2] != 3
            or keypoints.dtype != np.float64
        ):
            raise ValueError(
                "OracleResult: keypoints must be a float64 array of shape (k, K, 3), got "
                f"{type(keypoints).__name__} with shape {getattr(keypoints, 'shape', None)} "
                f"and dtype {getattr(keypoints, 'dtype', None)}"
            )
        if not np.all((keypoints[..., 2] == 0.0) | (keypoints[..., 2] == 1.0)):
            raise ValueError(
                "OracleResult: keypoints visibility must be 0 or 1, got "
                f"{np.unique(keypoints[..., 2]).tolist()}"
            )


class Oracle(Protocol):
    """Source of annotations for the people a round selects."""

    def request(
        self, indices: np.ndarray, round_index: int, predictions: np.ndarray
    ) -> OracleResult:
        """Ask for the annotation of the people ``indices`` in round ``round_index``.

        Args:
            indices: ``(k,)`` integer array of distinct sample indices of the video cache.
            round_index: Active-learning round, counted from 0.
            predictions: ``(N, K, 3)`` float array of the current predictions (x, y, confidence)
                of all ``N`` people, which an oracle may use to pre-annotate the queries.

        Returns:
            The annotations of the people in the order of ``indices``, or a result with
            ``ready=False`` if the answers are not available yet. Calling again with the same
            arguments is allowed and gives the same answer.
        """
        ...


def check_request(
    arrays: CacheArrays, indices: np.ndarray, round_index: int, predictions: np.ndarray
) -> None:
    """Raise ``ValueError`` unless ``(indices, round_index, predictions)`` is a valid request.

    The indices must be a non-empty 1-D integer array of distinct people of the cache, the round
    a non-negative integer and the predictions a floating-point array of shape ``(N, K, 3)``.
    """
    n, num_joints = arrays.keypoints_gt.shape[:2]
    if (
        not isinstance(indices, np.ndarray)
        or indices.ndim != 1
        or indices.dtype.kind not in "iu"
    ):
        raise ValueError(
            "indices: expected a 1-D integer array, got "
            f"{type(indices).__name__} with dtype {getattr(indices, 'dtype', None)} "
            f"and shape {getattr(indices, 'shape', None)}"
        )
    if indices.size == 0:
        raise ValueError("indices: at least one person must be requested")
    if indices.min() < 0 or indices.max() >= n:
        raise ValueError(
            f"indices: values from {indices.min()} to {indices.max()} are out of range [0, {n})"
        )
    values, counts = np.unique(indices, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"indices: sample indices {values[counts > 1].tolist()} appear more than once"
        )
    if isinstance(round_index, (bool, np.bool_)) or not isinstance(
        round_index, (int, np.integer)
    ) or round_index < 0:
        raise ValueError(
            f"round_index: expected a non-negative integer, got {round_index!r}"
        )
    if (
        not isinstance(predictions, np.ndarray)
        or predictions.dtype.kind != "f"
        or predictions.shape != (n, num_joints, 3)
    ):
        raise ValueError(
            f"predictions: expected a floating-point array of shape ({n}, {num_joints}, 3), got "
            f"{type(predictions).__name__} with dtype {getattr(predictions, 'dtype', None)} "
            f"and shape {getattr(predictions, 'shape', None)}"
        )
