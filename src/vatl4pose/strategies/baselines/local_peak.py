"""Mean local peak of the heatmaps.

The loop logs the mean of this value over the unlabeled people as ``combine_weight``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ...data.geometry import HEATMAP_SIZE

_FILTER_SIZE = 3
_ORDER = 0.5
_CHUNK = 128


@torch.no_grad()
def local_peak_mean(heatmaps: torch.Tensor, chunk: int = _CHUNK) -> np.ndarray:
    """Mean of the strong local peaks of every person.

    A cell is a local peak when it equals the maximum of the 3 x 3 square around it (the border
    is padded with zeros). A peak is strong when it is at least half of the largest peak of its
    heatmap. The strong peaks of all joints of a person are pooled and averaged.

    Args:
        heatmaps: ``(N, K, 64, 48)`` float32.
        chunk: Samples processed at once.

    Returns:
        ``(N,)`` float64.

    Raises:
        ValueError: If a person has no strong peak in any joint (all heatmaps negative).
    """
    assert heatmaps.ndim == 4 and tuple(heatmaps.shape[2:]) == HEATMAP_SIZE, (
        f"heatmaps: expected shape (N, K, {HEATMAP_SIZE[0]}, {HEATMAP_SIZE[1]}), "
        f"got {tuple(heatmaps.shape)}"
    )
    assert heatmaps.dtype == torch.float32, (
        f"heatmaps: expected float32, got {heatmaps.dtype}"
    )
    n, k, height, width = heatmaps.shape
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        x = heatmaps[start : start + chunk]
        m = x.shape[0]
        padded = F.pad(x.reshape(m * k, 1, height, width), (1, 1, 1, 1), value=0.0)
        local_max = F.max_pool2d(padded, _FILTER_SIZE, stride=1).reshape(
            m, k, height, width
        )
        is_peak = x == local_max
        largest = torch.where(is_peak, x, torch.full_like(x, float("-inf"))).amax(
            dim=(2, 3), keepdim=True
        )
        strong = is_peak & (x >= largest * _ORDER)
        total = (x.to(torch.float64) * strong).sum(dim=(1, 2, 3))
        count = strong.sum(dim=(1, 2, 3))
        empty = (count == 0).nonzero()
        if empty.numel() > 0:
            raise ValueError(
                f"sample {start + int(empty[0])} has no strong local peak in any joint"
            )
        out[start : start + m] = (total / count).cpu().numpy()
    return out
