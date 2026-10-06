"""The metrics of one active-learning round, with the sigmas of the keypoint specification."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from vatl4pose.keypoints import KeypointSpec

from .coco_ap import keypoint_ap
from .ospa import ospa


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
    """Keypoint AP and OSPA of one prediction per labelled person.

    AP and OSPA use the same ``spec.oks_sigmas``.

    Args:
        pred_kpt: ``(N, K, 2)`` predicted joint coordinates (``(N, K, 3)`` is accepted).
        pred_score: ``(N,)`` score of every predicted pose.
        gt_kpt: ``(N, K, 3)`` labelled joints ``x, y, visibility``.
        gt_bbox_xywh: ``(N, 4)`` label boxes ``[x, y, w, h]``.
        image_id: ``(N,)`` integer image id of every sample.
        spec: Keypoint specification of the dataset; its ``oks_sigmas`` must have ``K`` entries.

    Returns:
        :class:`RoundMetrics` with the ten values of :data:`vatl4pose.eval.METRIC_KEYS` and the OSPA.
    """
    sigmas = np.asarray(spec.oks_sigmas, dtype=np.float64)
    return RoundMetrics(
        ap=keypoint_ap(pred_kpt, pred_score, gt_kpt, gt_bbox_xywh, image_id, sigmas),
        ospa=ospa(pred_kpt, gt_kpt, gt_bbox_xywh, image_id, sigmas),
    )
