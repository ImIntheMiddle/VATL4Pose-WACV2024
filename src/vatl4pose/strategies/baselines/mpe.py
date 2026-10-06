"""MPE: multiple peak entropy. A heatmap with several competing peaks is queried first."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ...data.geometry import HEATMAP_SIZE
from ..base import RoundContext
from ..registry import register_scorer

MIN_DISTANCE = 5
NUM_PEAKS = 5

# Samples per step; bounds the memory of the sort over the cells of every heatmap.
_CHUNK = 128


def _local_maximum(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Maximum over the ``(2 * radius + 1)`` square around every cell of ``(B, H, W)``."""
    window = 2 * radius + 1
    y = x.unsqueeze(1)
    y = F.max_pool2d(y, (1, window), stride=1, padding=(0, radius))
    y = F.max_pool2d(y, (window, 1), stride=1, padding=(radius, 0))
    return y.squeeze(1)


def _spaced(
    rows: torch.Tensor,
    cols: torch.Tensor,
    valid: torch.Tensor,
    min_distance: int,
    num_peaks: int,
) -> torch.Tensor:
    """Greedy spacing over candidates that are sorted from the highest value down.

    A candidate is kept when it is valid, no kept candidate before it is closer than
    ``min_distance`` (the larger of the row and the column distance), and fewer than
    ``num_peaks`` candidates are kept so far.
    """
    kept = torch.zeros_like(valid)
    count = torch.zeros(valid.shape[0], dtype=torch.long, device=valid.device)
    for j in range(valid.shape[1]):
        if j == 0:
            blocked = torch.zeros_like(valid[:, 0])
        else:
            distance = torch.maximum(
                (rows[:, :j] - rows[:, j : j + 1]).abs(),
                (cols[:, :j] - cols[:, j : j + 1]).abs(),
            )
            blocked = (kept[:, :j] & (distance < min_distance)).any(dim=1)
        take = valid[:, j] & ~blocked & (count < num_peaks)
        kept[:, j] = take
        count += take
    return kept


def top_local_peak_values(
    heatmaps: torch.Tensor,
    min_distance: int = MIN_DISTANCE,
    num_peaks: int = NUM_PEAKS,
) -> torch.Tensor:
    """Values of the highest local peaks of every heatmap.

    A peak is a cell that equals the maximum of the ``(2 * min_distance + 1)`` square around it,
    is larger than the smallest value of its heatmap and lies at least ``min_distance`` cells away
    from every border. Peaks are taken from the highest value down (row-major order on equal
    values); a peak is dropped when a taken peak is closer than ``min_distance`` cells (largest of
    the row and column distance), and at most ``num_peaks`` peaks are taken.

    Args:
        heatmaps: ``(B, H, W)`` float32.

    Returns:
        ``(B, T)`` float32 with ``T >= num_peaks``: the value of a taken peak, ``-inf`` elsewhere.
        Row ``b`` holds between 0 and ``num_peaks`` finite entries.
    """
    assert heatmaps.ndim == 3 and heatmaps.dtype == torch.float32, (
        f"heatmaps: expected a float32 tensor of shape (B, H, W), got {heatmaps.dtype} "
        f"{tuple(heatmaps.shape)}"
    )
    batch, height, width = heatmaps.shape
    is_peak = (heatmaps == _local_maximum(heatmaps, min_distance)) & (
        heatmaps > heatmaps.amin(dim=(1, 2), keepdim=True)
    )
    is_peak[:, :min_distance, :] = False
    is_peak[:, height - min_distance :, :] = False
    is_peak[:, :, :min_distance] = False
    is_peak[:, :, width - min_distance :] = False

    candidates = torch.where(
        is_peak, heatmaps, torch.full_like(heatmaps, float("-inf"))
    ).reshape(batch, height * width)
    values, cells = candidates.sort(dim=1, descending=True, stable=True)
    available = is_peak.reshape(batch, -1).sum(dim=1)

    # Rejections can leave fewer than num_peaks peaks among the first num_peaks candidates; widen
    # the window of candidates until every row has its peaks or runs out of candidates.
    width_t = num_peaks
    while True:
        top_values, top_cells = values[:, :width_t], cells[:, :width_t]
        kept = _spaced(
            top_cells // width,
            top_cells % width,
            torch.isfinite(top_values),
            min_distance,
            num_peaks,
        )
        unresolved = (kept.sum(dim=1) < num_peaks) & (available > width_t)
        if not bool(unresolved.any()):
            break
        width_t = min(2 * width_t, height * width)
    return torch.where(kept, top_values, torch.full_like(top_values, float("-inf")))


def _softmax_entropy(values: torch.Tensor) -> torch.Tensor:
    """Entropy of the softmax over the finite entries of every row; 0 for a row without any.

    ``values`` is ``(B, T)``; ``-inf`` entries are absent. The computation is float64.
    """
    z = values.to(torch.float64)
    top = z.max(dim=1, keepdim=True).values
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    e = torch.exp(z - top)
    p = e / e.sum(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    return -torch.special.xlogy(p, p).sum(dim=1)


@torch.no_grad()
def multiple_peak_entropy(
    heatmaps: torch.Tensor,
    chunk: int = _CHUNK,
    min_distance: int = MIN_DISTANCE,
    num_peaks: int = NUM_PEAKS,
) -> np.ndarray:
    """Multiple peak entropy of every sample.

    Per joint, the softmax of the values of its (at most ``num_peaks``) highest local peaks gives
    a distribution whose entropy is large when several peaks compete; a joint without a peak adds
    0. The entropies of the joints are added.

    Args:
        heatmaps: ``(N, K, 64, 48)`` float32.
        chunk: Samples processed at once.

    Returns:
        ``(N,)`` float64.
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
        part = heatmaps[start : start + chunk]
        m = part.shape[0]
        peaks = top_local_peak_values(
            part.reshape(m * k, height, width), min_distance, num_peaks
        )
        entropy = _softmax_entropy(peaks).reshape(m, k).sum(dim=1)
        out[start : start + m] = entropy.cpu().numpy()
    return out


@register_scorer("mpe")
class MPEScorer:
    """Multiple peak entropy of the heatmaps (see :func:`multiple_peak_entropy`)."""

    def score(self, ctx: RoundContext) -> np.ndarray:
        return multiple_peak_entropy(ctx.heatmaps)
