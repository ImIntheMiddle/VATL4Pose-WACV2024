"""The metrics of one active-learning round, with the sigmas of the keypoint spec."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from vatl4pose.keypoints import KeypointSpec

from .coco_ap import keypoint_ap
from .ospa import ospa


def pose_score(peaks: np.ndarray) -> np.ndarray:
    """Score of every predicted pose: the mean plus 1.25 times the maximum of its joint peaks.

    Args:
        peaks: ``(N, K)`` float peak value of every joint, the scores that
            :func:`~vatl4pose.data.geometry.heatmaps_to_coords` returns.

    Returns:
        ``(N,)`` float64 scores, the ``pred_score`` of :func:`evaluate_predictions`.
    """
    assert np.issubdtype(peaks.dtype, np.floating) and peaks.ndim == 2, (
        f"peaks: expected a float array of shape (N, K), got {peaks.dtype} {peaks.shape}"
    )
    values = peaks.astype(np.float64)
    return values.mean(axis=1) + 1.25 * values.max(axis=1)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Boxes ``x1, y1, x2, y2`` to ``x, y, w, h`` with the width and height counting both end pixels.

    This is the box of the 2023 evaluation: the label area is ``w * h`` of it.

    Args:
        boxes: ``(N, 4)`` float boxes.

    Returns:
        ``(N, 4)`` boxes of the same dtype, the ``gt_bbox_xywh`` of :func:`evaluate_predictions`.
    """
    assert np.issubdtype(boxes.dtype, np.floating) and boxes.ndim == 2, (
        f"boxes: expected a float array of shape (N, 4), got {boxes.dtype} {boxes.shape}"
    )
    assert boxes.shape[1] == 4, f"boxes: expected shape (N, 4), got {boxes.shape}"
    return np.hstack((boxes[:, :2], boxes[:, 2:4] - boxes[:, :2] + 1.0))


class RoundMetrics(NamedTuple):
    """Metrics of one evaluation: the ten AP and AR values and the OSPA."""

    ap: dict[str, float]
    ospa: float


def evaluate_predictions(
    pred_kpt: np.ndarray,
    pred_score: np.ndarray,
    gt_kpt: np.ndarray,
    gt_bbox_xywh: np.ndarray,
    image_id: np.ndarray,
    spec: KeypointSpec,
) -> RoundMetrics:
    """Keypoint AP and OSPA of one prediction per labeled person.

    AP and OSPA use the same ``spec.oks_sigmas``.

    Args:
        pred_kpt: ``(N, K, 2)`` predicted joint coordinates (``(N, K, 3)`` is accepted).
        pred_score: ``(N,)`` score of every predicted pose.
        gt_kpt: ``(N, K, 3)`` labeled joints ``x, y, visibility``.
        gt_bbox_xywh: ``(N, 4)`` label boxes ``[x, y, w, h]``.
        image_id: ``(N,)`` integer image id of every person.
        spec: Keypoint spec of the dataset; its ``oks_sigmas`` must have ``K`` entries.

    Returns:
        :class:`RoundMetrics` with the ten values of :data:`vatl4pose.eval.METRIC_KEYS` and the OSPA.
    """
    return RoundMetrics(
        ap=keypoint_ap(
            pred_kpt, pred_score, gt_kpt, gt_bbox_xywh, image_id, spec.oks_sigmas
        ),
        ospa=ospa(pred_kpt, gt_kpt, gt_bbox_xywh, image_id, spec.oks_sigmas),
    )
