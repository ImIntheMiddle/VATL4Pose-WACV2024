"""Combiners: turn the scores of a strategy's scorers into one total score per sample.

Scores are normalised to ``[0, 1]`` over the *unlabeled* samples only: ``(x - min) / (max - min)``
with ``min`` and ``max`` taken over the unlabeled entries. The total score of a labeled sample is
0.0 and has no meaning.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .base import RoundContext
from .registry import register_combiner


def minmax_over_unlabeled(
    score: np.ndarray, labeled: np.ndarray, label: str
) -> np.ndarray:
    """Min-max normalise ``score`` over the unlabeled samples.

    Args:
        score: ``(N,)`` scores of all samples.
        labeled: ``(N,)`` bool, True for labeled samples.
        label: What the scores belong to, for the error message (e.g. ``"scorer 'thc'"``).

    Returns:
        ``(N,)`` float64 array, 0.0 at labeled samples. With fewer than two unlabeled samples the
        range is undefined and the result is all zeros, as in the legacy code.

    Raises:
        ValueError: If an unlabeled score is not finite, or if all unlabeled scores are equal
            (the legacy code divided by zero there and continued with NaN).
    """
    assert score.shape == labeled.shape and score.ndim == 1, (
        f"{label}: expected one score per sample, got {score.shape} for {labeled.shape} samples"
    )
    out = np.zeros(score.shape, dtype=np.float64)
    unlabeled = ~labeled
    values = score[unlabeled].astype(np.float64)
    if values.shape[0] < 2:
        return out
    if not np.isfinite(values).all():
        raise ValueError(f"{label} has non-finite scores for unlabeled samples")
    low, high = values.min(), values.max()
    if high == low:
        raise ValueError(
            f"{label} is constant ({low:g}) over the {values.shape[0]} unlabeled samples; "
            "min-max normalisation is undefined"
        )
    out[unlabeled] = (values - low) / (high - low)
    return out


@register_combiner("single")
class SingleCombiner:
    """Min-max normalisation of the only score."""

    def combine(
        self, scores: Mapping[str, np.ndarray], ctx: RoundContext
    ) -> np.ndarray:
        if len(scores) != 1:
            raise ValueError(
                f"combiner 'single' needs exactly one score, got {sorted(scores)}"
            )
        ((name, score),) = scores.items()
        return minmax_over_unlabeled(score, ctx.labeled, f"scorer {name!r}")


@register_combiner("minmax_sum")
class MinMaxSumCombiner:
    """Min-max normalise every score, add them, and min-max normalise the sum again."""

    def combine(
        self, scores: Mapping[str, np.ndarray], ctx: RoundContext
    ) -> np.ndarray:
        if not scores:
            raise ValueError("combiner 'minmax_sum' needs at least one score")
        total = np.zeros(ctx.num_samples, dtype=np.float64)
        for name, score in scores.items():
            total += minmax_over_unlabeled(score, ctx.labeled, f"scorer {name!r}")
        names = ", ".join(repr(name) for name in scores)
        return minmax_over_unlabeled(
            total, ctx.labeled, f"the sum of the normalised scores of {names}"
        )


@register_combiner("none")
class NoCombiner:
    """All-zero total score, for strategies without scorers (random and the pure filters)."""

    def combine(
        self, scores: Mapping[str, np.ndarray], ctx: RoundContext
    ) -> np.ndarray:
        if scores:
            raise ValueError(
                f"combiner 'none' takes no scores, got {sorted(scores)}; "
                "use 'single' or 'minmax_sum' to use them"
            )
        return np.zeros(ctx.num_samples, dtype=np.float64)
