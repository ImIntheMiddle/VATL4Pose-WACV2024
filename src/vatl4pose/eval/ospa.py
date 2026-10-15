"""OSPA (optimal sub-pattern assignment) of predicted poses against labeled poses.

For one frame with ``n`` labeled poses and ``n`` predicted poses, the OSPA is the cost of the best
one-to-one assignment, where assigning prediction ``p`` to label ``g`` costs ``1 - OKS(g, p)``,
divided by ``n``. A set of predictions with fewer or more poses than the labels would add the
cardinality difference ``|n_gt - n_pred|`` to the numerator and divide by the larger count. Here
every prediction belongs to exactly one labeled person, so both sets have the same size in every
frame and that term is zero. The cutoff of the OSPA metric is c = 1, because ``1 - OKS`` lies in
[0, 1], and its order is p = 1. The OSPA of a set of frames is the mean over the frames that have
labels; 0 is perfect and 1 is the worst value.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .oks import as_float_array, as_image_id, as_pose_xy, as_sigmas, oks_matrix


def ospa(
    pred_kpt: np.ndarray,
    gt_kpt: np.ndarray,
    gt_bbox_xywh: np.ndarray,
    image_id: np.ndarray,
    sigmas: np.ndarray | Sequence[float],
) -> float:
    """OSPA of one prediction per labeled person, averaged over the frames.

    Args:
        pred_kpt: ``(N, K, 2)`` predicted joint coordinates (``(N, K, 3)`` is accepted; the third
            column is ignored).
        gt_kpt: ``(N, K, 3)`` labeled joints ``x, y, visibility``.
        gt_bbox_xywh: ``(N, 4)`` label boxes ``[x, y, w, h]``; ``w * h`` is the area in the OKS.
        image_id: ``(N,)`` integer image id of every person.
        sigmas: ``(K,)`` OKS sigmas (``KeypointSpec.oks_sigmas``), the same as in :func:`keypoint_ap`.

    Returns:
        The mean over the frames of the per-frame OSPA, in ``[0, 1]``.

    Raises:
        ValueError: If no sample is given.
    """
    sig = as_sigmas(sigmas)
    k = sig.shape[0]
    pred = as_pose_xy("pred_kpt", pred_kpt, k)
    gt = as_float_array("gt_kpt", gt_kpt, (k, 3))
    box = as_float_array("gt_bbox_xywh", gt_bbox_xywh, (4,))
    image = as_image_id("image_id", image_id)
    n = pred.shape[0]
    assert gt.shape[0] == box.shape[0] == image.shape[0] == n, (
        f"pred_kpt, gt_kpt, gt_bbox_xywh and image_id must hold the same number of samples, "
        f"got {n}, {gt.shape[0]}, {box.shape[0]}, {image.shape[0]}"
    )
    if n == 0:
        raise ValueError("ospa: no sample given")

    order = np.argsort(image, kind="stable")
    _, first = np.unique(image[order], return_index=True)
    per_image = []
    for members in np.split(order, first[1:]):
        cost = 1.0 - oks_matrix(pred[members], gt[members], box[members], sig)
        rows, cols = linear_sum_assignment(cost)
        per_image.append(cost[rows, cols].sum() / members.shape[0])
    return float(np.mean(per_image))
