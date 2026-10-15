"""TPC: Temporal Pose Continuity. A pose that jumps between frames is queried first."""

from __future__ import annotations

import numpy as np
import torch

from ...data.geometry import coords_in_box
from ..base import RoundContext
from ..neighbors import temporal_neighbor_sum
from ..registry import register_scorer

# Pairs decoded at once; bounds the memory of the gathered neighbor heatmaps.
_CHUNK = 256

_THRESHOLD_RATIO = 0.01


def moved_joints(
    ctx: RoundContext, samples: np.ndarray, neighbors: np.ndarray
) -> np.ndarray:
    """Count, for every ``(sample, neighbor)`` pair, the joints that moved.

    The neighbor's heatmaps are decoded against the crop box of the *person*, so the distance
    between the two poses is measured in one coordinate frame. A joint moved when its distance
    exceeds ``0.01 * sqrt(w * h)`` of that box.

    Returns:
        ``(len(samples),)`` float64 counts.
    """
    out = np.empty(samples.shape[0], dtype=np.float64)
    device = ctx.heatmaps.device
    for start in range(0, samples.shape[0], _CHUNK):
        rows = samples[start : start + _CHUNK]
        other = torch.from_numpy(neighbors[start : start + _CHUNK]).to(device)
        boxes = ctx.bbox_crop[rows]
        adjacent = coords_in_box(
            ctx.heatmaps[other], torch.from_numpy(boxes).to(device)
        )
        diff = ctx.coords[rows].astype(np.float64) - adjacent.cpu().numpy().astype(
            np.float64
        )
        distance = np.linalg.norm(diff, axis=-1)
        box = boxes.astype(np.float64)
        threshold = _THRESHOLD_RATIO * np.sqrt(
            (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        )
        out[start : start + _CHUNK] = (distance > threshold[:, None]).sum(axis=1)
    return out


@register_scorer("tpc")
class TPCScorer:
    """Number of joints that moved against the previous and the next frame.

    The counts of both neighbors are added; a person with one neighbor gets twice its count and a
    person without neighbors gets 0.
    """

    def score(self, ctx: RoundContext) -> np.ndarray:
        return temporal_neighbor_sum(
            ctx, lambda samples, neighbors: moved_joints(ctx, samples, neighbors)
        )
