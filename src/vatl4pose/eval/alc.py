"""Area under the learning curve (ALC) of an active-learning run.

A run reports a metric after every round, at the percentage of the samples that are labelled by
then. The ALC is the area under that curve with both axes scaled to ``[0, 1]``, so a metric that
is 1 from the start has the ALC 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PERCENT_GRID: np.ndarray = np.linspace(0.0, 100.0, 1001)
"""The grid of label percentages ``0, 0.1, ..., 100`` on which the summaries compare runs."""
PERCENT_GRID.flags.writeable = False


def _as_curve(
    percentages: Sequence[float], performance: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(percentages, dtype=np.float64)
    y = np.asarray(performance, dtype=np.float64)
    assert x.ndim == 1, f"percentages: expected a vector, got shape {x.shape}"
    assert y.shape == x.shape, f"performance: expected shape {x.shape}, got {y.shape}"
    if x.shape[0] < 2:
        raise ValueError(
            f"percentages: a curve needs at least 2 rounds, got {x.shape[0]}"
        )
    if np.any(np.diff(x) < 0):
        raise ValueError(f"percentages must not decrease, got {x.tolist()}")
    return x, y


def compute_alc(percentages: Sequence[float], performance: Sequence[float]) -> float:
    """Area under the learning curve.

    The area is the trapezoid sum over the rounds of ``(0.01 * percentage, performance)``.

    Args:
        percentages: Percentage of labelled samples after every round, from 0 to 100.
        performance: Metric after every round on the scale ``[0, 1]`` (AP, AR, OKS or OSPA).
            Callers that hold percent values divide them by 100 first.

    Returns:
        The area, in ``[0, 1]`` when the percentages span 0 to 100.

    Raises:
        ValueError: If the curve has fewer than 2 rounds, the percentages decrease, or a
            performance value lies outside ``[0, 1]`` (a metric still in percent, or the ``-1``
            of a setting without labels).
    """
    x, y = _as_curve(percentages, performance)
    if np.any(y < 0.0) or np.any(y > 1.0 + 1e-9):
        raise ValueError(
            f"performance must lie in [0, 1], got range [{y.min():.6g}, {y.max():.6g}]; "
            "divide a percent value by 100"
        )
    width = np.diff(0.01 * x)
    return float(np.sum(width * (y[1:] + y[:-1]) * 0.5))


def interpolate_curve(
    percentages: Sequence[float],
    performance: Sequence[float],
    grid: np.ndarray = PERCENT_GRID,
) -> np.ndarray:
    """Linear interpolation of a learning curve onto a grid of label percentages.

    Args:
        percentages: Percentage of labelled samples after every round.
        performance: Metric after every round, on any scale.
        grid: Label percentages to evaluate at; every value must lie inside the range of
            ``percentages``. The default is :data:`PERCENT_GRID`.

    Returns:
        ``(len(grid),)`` float64 values.

    Raises:
        ValueError: If the curve has fewer than 2 rounds, the percentages decrease, or the grid
            reaches outside the percentages.
    """
    x, y = _as_curve(percentages, performance)
    g = np.asarray(grid, dtype=np.float64)
    assert g.ndim == 1, f"grid: expected a vector, got shape {g.shape}"
    if g.size and (g.min() < x[0] or g.max() > x[-1]):
        raise ValueError(
            f"grid [{g.min():.6g}, {g.max():.6g}] lies outside the percentages [{x[0]:.6g}, {x[-1]:.6g}]"
        )
    return np.interp(g, x, y)
