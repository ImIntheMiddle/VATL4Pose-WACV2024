"""Keypoint average precision on the COCO protocol, computed in memory.

The 2023 experiments evaluated with a modified copy of ``pycocotools``. This module reproduces
that evaluation with the stock ``pycocotools`` from PyPI, set up through ``Params``. The 2023 copy
differed from the stock evaluation in three places.

* ``maxDets``. The 2023 copy used 100 detections per image; stock keypoint evaluation uses 20.
  ``params.maxDets`` is set to ``[100]``.
* Detection area. The 2023 copy set the area of a detection that carries a box from the box
  as if it were ``[x1, y1, x2, y2]``, that is ``(w - x) * (h - y)`` for a box ``[x, y, w, h]``.
  The area is negative or wrong for most boxes, and an unmatched detection whose area lies outside
  the area range is ignored instead of counted as a false positive. The experiments of the paper
  carry this behaviour, and :func:`keypoint_ap` keeps it so that the numbers stay comparable.
  Stock ``loadRes`` uses ``w * h``; the detections are therefore built by hand with the area set
  explicitly.
* Summary. The 2023 copy summarised a keypoint evaluation as the ten values of
  :data:`METRIC_KEYS`. Stock ``summarize`` returns ``AP, AP50, AP75, APm, APl, AR, AR50, AR75,
  ARm, ARl`` for 20 detections, so its entries shift against the 2023 key names (the entry at
  the position of ``AP .6`` is AP75). The values are computed here from ``eval["precision"]`` and
  ``eval["recall"]`` and ``summarize`` is never called.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .oks import as_float_array, as_image_id, as_pose_xy, as_sigmas

AP_PROTOCOL = "wacv2024-detection-area"
"""Name of the AP protocol of this package, written next to every AP in ``result.json`` and in the
``vatl4pose-eval`` file. It is the 2023 protocol of the paper, which differs from the standard COCO
keypoint AP in the area of a detection (see the module docstring), so the values are comparable with
the paper and not with another COCO evaluation."""

METRIC_KEYS: tuple[str, ...] = (
    "AP",
    "AP .5",
    "AP .6",
    "AP .7",
    "AP .75",
    "AP .8",
    "AP .95",
    "AP (M)",
    "AP (L)",
    "AR",
)
"""Names of the ten values of one keypoint evaluation, in the order the 2023 code reported them."""

_MAX_DETS = 100
_AREA_RANGES: list[list[float]] = [
    [0.0**2, 1e5**2],
    [32.0**2, 96.0**2],
    [96.0**2, 1e5**2],
]
_AREA_LABELS: list[str] = ["all", "medium", "large"]
_IOU_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("AP .5", 0.5),
    ("AP .6", 0.6),
    ("AP .7", 0.7),
    ("AP .75", 0.75),
    ("AP .8", 0.8),
    ("AP .95", 0.95),
)
_PERSON = 1  # the only category


def _as_score(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    assert np.issubdtype(arr.dtype, np.floating), (
        f"{name}: expected a float array, got {arr.dtype}"
    )
    assert arr.ndim == 1, f"{name}: expected shape (N,), got {arr.shape}"
    return arr.astype(np.float64, copy=False)


def _mean_over_valid(values: np.ndarray) -> float:
    """Mean of the entries above ``-1``; ``-1`` when the setting has no ground truth."""
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else -1.0


def _threshold_index(iou_thresholds: np.ndarray, value: float) -> int:
    hits = np.flatnonzero(np.isclose(iou_thresholds, value, rtol=0.0, atol=1e-9))
    assert hits.size == 1, (
        f"IoU threshold {value} is not on the grid {iou_thresholds.tolist()}"
    )
    return int(hits[0])


def coco_keypoint_ap(
    *,
    det_xy: np.ndarray,
    det_score: np.ndarray,
    det_bbox_xywh: np.ndarray,
    det_image_id: np.ndarray,
    gt_kpt: np.ndarray,
    gt_bbox_xywh: np.ndarray,
    gt_image_id: np.ndarray,
    sigmas: np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Keypoint AP of a set of detections against labelled persons.

    Every labelled person is one ground-truth entry with ``area = w * h`` of its box. Every
    detection carries a box and gets the area ``(w - x) * (h - y)`` of that box read as
    ``[x, y, w, h]`` (the 2023 behaviour, see the module docstring). Detections and labels
    keep their input order; ties in the score are resolved by that order.

    Args:
        det_xy: ``(M, K, 2)`` predicted joint coordinates (``(M, K, 3)`` is accepted; the third
            column is ignored).
        det_score: ``(M,)`` score of every detection.
        det_bbox_xywh: ``(M, 4)`` box of every detection, ``[x, y, w, h]``.
        det_image_id: ``(M,)`` integer image id of every detection.
        gt_kpt: ``(N, K, 3)`` labelled joints ``x, y, visibility``.
        gt_bbox_xywh: ``(N, 4)`` label boxes ``[x, y, w, h]``.
        gt_image_id: ``(N,)`` integer image id of every label.
        sigmas: ``(K,)`` OKS sigmas, already divided by 10 (``KeypointSpec.oks_sigmas``).

    Returns:
        The ten values of :data:`METRIC_KEYS` as floats. ``AP`` and ``AR`` average over the IoU
        thresholds 0.5 to 0.95. ``AP (M)`` and ``AP (L)`` use the labels with an area from 32^2 to
        96^2 and from 96^2. A value is ``-1.0`` when its setting has no label (the COCO convention).

    Raises:
        ValueError: If no detection or no label is given, if a detection refers to an image id
            that no label has, or if a label has no visible joint.
    """
    sig = as_sigmas(sigmas)
    k = sig.shape[0]
    d_xy = as_pose_xy("det_xy", det_xy, k)
    d_score = _as_score("det_score", det_score)
    d_box = as_float_array("det_bbox_xywh", det_bbox_xywh, (4,))
    d_img = as_image_id("det_image_id", det_image_id)
    g_kpt = as_float_array("gt_kpt", gt_kpt, (k, 3))
    g_box = as_float_array("gt_bbox_xywh", gt_bbox_xywh, (4,))
    g_img = as_image_id("gt_image_id", gt_image_id)
    n_det, n_gt = d_xy.shape[0], g_kpt.shape[0]
    assert d_score.shape[0] == d_box.shape[0] == d_img.shape[0] == n_det, (
        f"det_xy, det_score, det_bbox_xywh and det_image_id must hold the same number of "
        f"detections, got {n_det}, {d_score.shape[0]}, {d_box.shape[0]}, {d_img.shape[0]}"
    )
    assert g_box.shape[0] == g_img.shape[0] == n_gt, (
        f"gt_kpt, gt_bbox_xywh and gt_image_id must hold the same number of labels, "
        f"got {n_gt}, {g_box.shape[0]}, {g_img.shape[0]}"
    )

    if n_det == 0:
        raise ValueError("no detection given")
    if n_gt == 0:
        raise ValueError("no ground truth given")
    image_ids = np.unique(g_img)
    foreign = np.setdiff1d(d_img, image_ids)
    if foreign.size:
        raise ValueError(
            f"{foreign.size} detection image id(s) have no ground truth, first ones: "
            f"{foreign[:10].tolist()}"
        )
    n_visible = (g_kpt[:, :, 2] > 0).sum(axis=1)
    unlabelled = np.flatnonzero(n_visible == 0)
    if unlabelled.size:
        raise ValueError(
            f"ground-truth sample {int(unlabelled[0])} has no visible keypoint; the evaluation "
            "of 2023 never ignored a person for that, so such samples must be dropped before"
        )

    images = [{"id": int(i)} for i in image_ids]
    categories = [{"id": _PERSON, "name": "person"}]
    gt_annotations = [
        {
            "id": i + 1,
            "image_id": int(g_img[i]),
            "category_id": _PERSON,
            "iscrowd": 0,
            "bbox": g_box[i].tolist(),
            "area": float(g_box[i, 2] * g_box[i, 3]),
            "num_keypoints": int(n_visible[i]),
            "keypoints": g_kpt[i].reshape(-1).tolist(),
        }
        for i in range(n_gt)
    ]
    # The joint scores are not used by the OKS; the third column only keeps the COCO layout.
    d_kpt = np.concatenate([d_xy, np.ones((n_det, k, 1))], axis=2)
    det_annotations = [
        {
            "id": j + 1,
            "image_id": int(d_img[j]),
            "category_id": _PERSON,
            "iscrowd": 0,
            "score": float(d_score[j]),
            "bbox": d_box[j].tolist(),
            "area": float((d_box[j, 2] - d_box[j, 0]) * (d_box[j, 3] - d_box[j, 1])),
            "keypoints": d_kpt[j].reshape(-1).tolist(),
        }
        for j in range(n_det)
    ]

    # pycocotools prints progress lines; they are not wanted in logs of thousands of rounds.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt, coco_dt = COCO(), COCO()
        coco_gt.dataset = {
            "images": images,
            "categories": categories,
            "annotations": gt_annotations,
        }
        coco_dt.dataset = {
            "images": images,
            "categories": categories,
            "annotations": det_annotations,
        }
        coco_gt.createIndex()
        coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "keypoints")
        evaluator.params.kpt_oks_sigmas = sig
        evaluator.params.maxDets = [_MAX_DETS]
        evaluator.params.areaRng = [list(r) for r in _AREA_RANGES]
        evaluator.params.areaRngLbl = list(_AREA_LABELS)
        evaluator.evaluate()
        evaluator.accumulate()

    # precision is (thresholds, recall points, category, area range, maxDets); recall drops the
    # recall-point axis. There is one category and one maxDets value.
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    iou_thresholds = np.asarray(evaluator.params.iouThrs)
    n_thr, n_rec = len(iou_thresholds), len(evaluator.params.recThrs)
    assert precision.shape == (n_thr, n_rec, 1, 3, 1), precision.shape
    assert recall.shape == (n_thr, 1, 3, 1), recall.shape
    all_areas, medium, large = 0, 1, 2

    stats = {"AP": _mean_over_valid(precision[:, :, :, all_areas, 0])}
    for key, threshold in _IOU_THRESHOLDS:
        t = _threshold_index(iou_thresholds, threshold)
        stats[key] = _mean_over_valid(precision[t, :, :, all_areas, 0])
    stats["AP (M)"] = _mean_over_valid(precision[:, :, :, medium, 0])
    stats["AP (L)"] = _mean_over_valid(precision[:, :, :, large, 0])
    stats["AR"] = _mean_over_valid(recall[:, :, all_areas, 0])
    assert tuple(stats) == METRIC_KEYS
    return stats


def keypoint_ap(
    pred_kpt: np.ndarray,
    pred_score: np.ndarray,
    gt_kpt: np.ndarray,
    gt_bbox_xywh: np.ndarray,
    image_id: np.ndarray,
    sigmas: np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Keypoint AP of one prediction per labelled person (the active-learning evaluation).

    Prediction ``i`` belongs to label ``i``: it lies in the same image and carries the box of
    the label, so its area is ``(w - x) * (h - y)`` of that box (see :func:`coco_keypoint_ap`).
    The labels have the area ``w * h``.

    Args:
        pred_kpt: ``(N, K, 2)`` predicted joint coordinates (``(N, K, 3)`` is accepted; the
            third column is ignored).
        pred_score: ``(N,)`` score of every predicted pose.
        gt_kpt: ``(N, K, 3)`` labelled joints ``x, y, visibility``.
        gt_bbox_xywh: ``(N, 4)`` label boxes ``[x, y, w, h]``.
        image_id: ``(N,)`` integer image id of every sample.
        sigmas: ``(K,)`` OKS sigmas (``KeypointSpec.oks_sigmas``), also used by :func:`ospa`.

    Returns:
        The ten values of :data:`METRIC_KEYS` (``-1.0`` for a setting without labels).

    Raises:
        ValueError: If no detection is given or a label has no visible joint.
    """
    counts = tuple(
        len(a) for a in (pred_kpt, pred_score, gt_kpt, gt_bbox_xywh, image_id)
    )
    assert len(set(counts)) == 1, (
        "pred_kpt, pred_score, gt_kpt, gt_bbox_xywh and image_id must hold the same number of "
        f"samples, got {counts}"
    )
    return coco_keypoint_ap(
        det_xy=pred_kpt,
        det_score=pred_score,
        det_bbox_xywh=gt_bbox_xywh,
        det_image_id=image_id,
        gt_kpt=gt_kpt,
        gt_bbox_xywh=gt_bbox_xywh,
        gt_image_id=image_id,
        sigmas=sigmas,
    )
