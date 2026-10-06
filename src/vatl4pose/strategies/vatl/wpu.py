"""WPU: whole-body pose uncertainty. A pose that the autoencoder reconstructs badly is queried first."""

from __future__ import annotations

import numpy as np

from ...eval import xyxy_to_xywh
from ...models.wholebody_ae import WholeBodyAE, compute_hybrid, wpu_scores
from ..base import RoundContext
from ..registry import register_scorer


def crop_boxes_xywh(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Convert ``(N, 4)`` boxes ``x1, y1, x2, y2`` to ``x, y, w, h`` with ``w = x2 - x1 + 1``.

    The ``+ 1`` is the pixel-inclusive size of the 2023 code; the hybrid feature divides by the
    height. The arithmetic runs in float64 whatever the dtype of the input.
    """
    box = np.asarray(bbox_xyxy, dtype=np.float64)
    assert box.ndim == 2 and box.shape[1] == 4, (
        f"bbox: expected (N, 4), got {box.shape}"
    )
    return xyxy_to_xywh(box)


@register_scorer("wpu")
class WPUScorer:
    """Reconstruction error of the hybrid feature of the predicted pose.

    The feature is built from the crop box ``ctx.bbox_crop`` and the decoded joints
    ``ctx.coords`` weighted by their peak values ``ctx.maxvals``. The autoencoder is
    ``ctx.extras["ae"]``. The score is the mean squared reconstruction error over the columns that
    ``ctx.spec.wpu_keep_index()`` keeps, which leaves out the offsets of the redundant joints of the
    dataset. The scorer takes no arguments: another set of columns is another specification,
    for example ``dataclasses.replace(spec, redundant_joints=())`` for all of them.
    """

    def score(self, ctx: RoundContext) -> np.ndarray:
        ae = ctx.extras.get("ae")
        if not isinstance(ae, WholeBodyAE):
            raise ValueError(
                "scorer 'wpu' needs the whole-body autoencoder in ctx.extras['ae'], got "
                f"{type(ae).__name__}; the loop provides it for strategies that use WPU"
            )
        keypoints = np.concatenate(
            [ctx.coords, ctx.maxvals[..., None]], axis=-1, dtype=np.float64
        )
        hybrid = compute_hybrid(crop_boxes_xywh(ctx.bbox_crop), keypoints, ctx.spec)
        return wpu_scores(ae, hybrid, ctx.spec.wpu_keep_index())
