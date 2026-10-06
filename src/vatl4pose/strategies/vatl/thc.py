"""THC: temporal heatmap continuity. A heatmap that changes between frames is queried first."""

from __future__ import annotations

import numpy as np
import torch

from ..base import RoundContext
from ..neighbors import temporal_neighbor_sum
from ..registry import register_scorer

# Pairs compared at once; bounds the memory of the gathered neighbor heatmaps.
_CHUNK = 128


def l1_heatmap_distance(
    heatmaps: torch.Tensor, samples: np.ndarray, neighbors: np.ndarray
) -> np.ndarray:
    """L1 distance between the heatmaps of ``samples[i]`` and ``neighbors[i]``, per joint.

    The distance is the sum of absolute differences over all joints and cells, divided by the
    number of joints. The differences are formed in the dtype of ``heatmaps`` and summed in
    float64.

    Args:
        heatmaps: ``(N, K, H, W)`` float32 heatmaps.
        samples: ``(P,)`` int64 sample indices.
        neighbors: ``(P,)`` int64 neighbor indices.

    Returns:
        ``(P,)`` float64 distances.
    """
    assert heatmaps.ndim == 4 and samples.shape == neighbors.shape, (
        f"heatmaps {tuple(heatmaps.shape)}, samples {samples.shape}, "
        f"neighbors {neighbors.shape}"
    )
    num_joints = heatmaps.shape[1]
    device = heatmaps.device
    out = np.empty(samples.shape[0], dtype=np.float64)
    for start in range(0, samples.shape[0], _CHUNK):
        rows = torch.from_numpy(samples[start : start + _CHUNK]).to(device)
        other = torch.from_numpy(neighbors[start : start + _CHUNK]).to(device)
        diff = (heatmaps[rows] - heatmaps[other]).abs()
        total = diff.sum(dim=(1, 2, 3), dtype=torch.float64)
        out[start : start + _CHUNK] = total.cpu().numpy() / num_joints
    return out


@register_scorer("thc")
class THCScorer:
    """L1 distance between a sample's heatmaps and those of its previous and next frame.

    The distances to both neighbors are added; a sample with one neighbor gets twice its distance
    and a sample without neighbors gets 0. The neighbor heatmaps are the neighbor samples' own
    heatmaps (``ctx.heatmaps``), which the model predicts from each sample's own crop.
    """

    def score(self, ctx: RoundContext) -> np.ndarray:
        return temporal_neighbor_sum(
            ctx,
            lambda samples, neighbors: l1_heatmap_distance(
                ctx.heatmaps, samples, neighbors
            ),
        )
