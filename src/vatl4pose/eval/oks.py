"""Object keypoint similarity (OKS) between predicted and labeled poses.

The OKS of a predicted pose against a labeled person is the mean, over the labeled joints, of

``exp(-d_j^2 / (2 * s^2 * k_j^2))``

where ``d_j`` is the distance between the predicted and the labeled joint, ``s^2 = w * h`` is the
area of the person box and ``k_j = 2 * sigma_j`` is the tolerance of joint ``j``. A joint is
labeled when its visibility flag is positive. For a person without any labeled joint the mean
runs over all joints and ``d_j`` is the distance from the predicted joint to the box enlarged to
three times its width and height around its center (zero inside that box).

All functions work on float64 internally. Poses are ``(K, 2)`` or ``(K, 3)`` arrays whose first two
columns are ``x`` and ``y`` (a third column, the joint score, is ignored); labels are ``(K, 3)``
arrays ``x, y, visibility``; boxes are ``[x, y, w, h]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Added to the box area so that an empty box does not divide by zero (the constant of COCO).
_AREA_EPS = float(np.spacing(1))


def as_sigmas(sigmas: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return ``sigmas`` as a float64 vector of positive, finite values."""
    arr = np.asarray(sigmas, dtype=np.float64)
    assert arr.ndim == 1 and arr.shape[0] > 0, (
        f"sigmas: expected a non-empty vector, got shape {arr.shape}"
    )
    assert np.all(np.isfinite(arr)) and np.all(arr > 0), (
        "sigmas: values must be positive and finite"
    )
    return arr


def as_float_array(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return ``value`` as float64 after checking that it is floating and has ``(N, *shape)``."""
    arr = np.asarray(value)
    assert np.issubdtype(arr.dtype, np.floating), (
        f"{name}: expected a float array, got {arr.dtype}"
    )
    assert arr.ndim == 1 + len(shape) and arr.shape[1:] == shape, (
        f"{name}: expected shape (N, {', '.join(str(d) for d in shape)}), got {arr.shape}"
    )
    return arr.astype(np.float64, copy=False)


def as_image_id(name: str, value: np.ndarray) -> np.ndarray:
    """Return ``value`` as an int64 vector after checking that it is an integer array of shape ``(N,)``."""
    arr = np.asarray(value)
    assert np.issubdtype(arr.dtype, np.integer), (
        f"{name}: expected an integer array, got {arr.dtype}"
    )
    assert arr.ndim == 1, f"{name}: expected shape (N,), got {arr.shape}"
    return arr.astype(np.int64, copy=False)


def as_pose_xy(name: str, value: np.ndarray, num_joints: int) -> np.ndarray:
    """Return the ``(N, K, 2)`` float64 coordinates of poses given as ``(N, K, 2)`` or ``(N, K, 3)``."""
    arr = np.asarray(value)
    assert np.issubdtype(arr.dtype, np.floating), (
        f"{name}: expected a float array, got {arr.dtype}"
    )
    assert arr.ndim == 3 and arr.shape[1] == num_joints and arr.shape[2] in (2, 3), (
        f"{name}: expected shape (N, K, 2) or (N, K, 3) with K={num_joints} joints "
        f"(the length of sigmas), got {arr.shape}"
    )
    return arr[:, :, :2].astype(np.float64, copy=False)


def _oks(
    pred_xy: np.ndarray,
    gt_kpt: np.ndarray,
    gt_bbox_xywh: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """OKS over broadcast leading axes.

    Args:
        pred_xy: ``(..., K, 2)`` predicted coordinates.
        gt_kpt: ``(..., K, 3)`` labels ``x, y, visibility``.
        gt_bbox_xywh: ``(..., 4)`` label boxes.
        sigmas: ``(K,)`` joint sigmas.

    Returns:
        The OKS with the broadcast leading shape.
    """
    xd, yd = pred_xy[..., 0], pred_xy[..., 1]
    xg, yg, vg = gt_kpt[..., 0], gt_kpt[..., 1], gt_kpt[..., 2]
    x, y, w, h = (gt_bbox_xywh[..., i, np.newaxis] for i in range(4))

    labeled = vg > 0
    has_label = labeled.any(axis=-1, keepdims=True)
    # Without labels the distance goes to the box enlarged to three times its size.
    dx_box = np.maximum(0.0, (x - w) - xd) + np.maximum(0.0, xd - (x + 2.0 * w))
    dy_box = np.maximum(0.0, (y - h) - yd) + np.maximum(0.0, yd - (y + 2.0 * h))
    dx = np.where(has_label, xd - xg, dx_box)
    dy = np.where(has_label, yd - yg, dy_box)
    used = np.where(has_label, labeled, True)

    variances = (2.0 * sigmas) ** 2
    e = (dx**2 + dy**2) / variances / (w * h + _AREA_EPS) * 0.5
    return (np.exp(-e) * used).sum(axis=-1) / used.sum(axis=-1)


def oks_per_sample(
    pred_xy: np.ndarray,
    gt_kpt: np.ndarray,
    gt_bbox_xywh: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """OKS of every predicted pose against the label of the same person.

    Args:
        pred_xy: ``(N, K, 2)`` predicted joint coordinates (``(N, K, 3)`` is accepted; the third
            column is ignored).
        gt_kpt: ``(N, K, 3)`` labeled joints ``x, y, visibility``.
        gt_bbox_xywh: ``(N, 4)`` label boxes ``x, y, w, h``; ``w * h`` is the area in the OKS.
        sigmas: ``(K,)`` per-joint sigmas, already divided by 10 (``KeypointSpec.oks_sigmas``).

    Returns:
        ``(N,)`` float64 OKS values in ``[0, 1]``.
    """
    sig = as_sigmas(sigmas)
    k = sig.shape[0]
    pred = as_pose_xy("pred_xy", pred_xy, k)
    gt = as_float_array("gt_kpt", gt_kpt, (k, 3))
    box = as_float_array("gt_bbox_xywh", gt_bbox_xywh, (4,))
    assert pred.shape[0] == gt.shape[0] == box.shape[0], (
        f"pred_xy, gt_kpt and gt_bbox_xywh must hold the same number of samples, got "
        f"{pred.shape[0]}, {gt.shape[0]}, {box.shape[0]}"
    )
    out = _oks(pred, gt, box, sig)
    assert out.shape == (pred.shape[0],) and out.dtype == np.float64
    return out


def oks_matrix(
    pred_xy: np.ndarray,
    gt_kpt: np.ndarray,
    gt_bbox_xywh: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """OKS of every predicted pose against every label.

    Args:
        pred_xy: ``(P, K, 2)`` predicted joint coordinates (``(P, K, 3)`` is accepted).
        gt_kpt: ``(G, K, 3)`` labeled joints.
        gt_bbox_xywh: ``(G, 4)`` label boxes; each box belongs to its label.
        sigmas: ``(K,)`` per-joint sigmas.

    Returns:
        ``(G, P)`` float64 matrix; row ``g`` is the label, column ``p`` the prediction.
    """
    sig = as_sigmas(sigmas)
    k = sig.shape[0]
    pred = as_pose_xy("pred_xy", pred_xy, k)
    gt = as_float_array("gt_kpt", gt_kpt, (k, 3))
    box = as_float_array("gt_bbox_xywh", gt_bbox_xywh, (4,))
    assert gt.shape[0] == box.shape[0], (
        f"gt_kpt and gt_bbox_xywh must hold the same number of labels, got {gt.shape[0]}, {box.shape[0]}"
    )
    out = _oks(pred[np.newaxis], gt[:, np.newaxis], box[:, np.newaxis], sig)
    assert out.shape == (gt.shape[0], pred.shape[0]) and out.dtype == np.float64
    return out
