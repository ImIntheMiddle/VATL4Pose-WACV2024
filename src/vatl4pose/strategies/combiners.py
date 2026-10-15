"""Combiners: turn the scores of a strategy's scorers into one total score per person.

Scores are normalized to ``[0, 1]`` over the *unlabeled* people only: ``(x - min) / (max - min)``
with ``min`` and ``max`` taken over the unlabeled entries. The total score of a labeled person is
0.0 and has no meaning.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from .base import RoundContext
from .registry import register_combiner

logger = logging.getLogger(__name__)


def minmax_over_unlabeled(
    score: np.ndarray, labeled: np.ndarray, label: str, round_index: int
) -> np.ndarray:
    """Min-max normalize ``score`` over the unlabeled people.

    A score without spread over the unlabeled people (all values equal, or fewer than two
    unlabeled people) normalizes to zeros, and one ``logging.WARNING`` names the scorer and the
    round. (The 2023 code produced NaN here and queried the lowest indices; the stable top-k over
    zeros gives the same order.)

    Args:
        score: ``(N,)`` scores of all people.
        labeled: ``(N,)`` bool, True for labeled people.
        label: What the scores belong to, for the log message (e.g. ``"scorer 'thc'"``).
        round_index: Index of the round, for the log message.

    Returns:
        ``(N,)`` float64 array, 0.0 at labeled people.

    Raises:
        ValueError: If an unlabeled score is not finite (NaN or infinity is an error of the
            scorer, not a property of the data).
    """
    assert score.shape == labeled.shape and score.ndim == 1, (
        f"{label}: expected one score per sample, got {score.shape} for {labeled.shape} samples"
    )
    out = np.zeros(score.shape, dtype=np.float64)
    unlabeled = ~labeled
    values = score[unlabeled].astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} has non-finite scores for unlabeled people")
    if values.shape[0] < 2 or values.max() == values.min():
        logger.warning(
            "%s has no spread over the %d unlabeled people in round %d; "
            "its normalized score is 0",
            label,
            values.shape[0],
            round_index,
        )
        return out
    low, high = values.min(), values.max()
    out[unlabeled] = (values - low) / (high - low)
    return out


@register_combiner("single")
class SingleCombiner:
    """Min-max normalization of the only score."""

    def combine(
        self, scores: Mapping[str, np.ndarray], ctx: RoundContext
    ) -> np.ndarray:
        if len(scores) != 1:
            raise ValueError(
                f"combiner 'single' needs exactly one score, got {sorted(scores)}"
            )
        ((name, score),) = scores.items()
        return minmax_over_unlabeled(
            score, ctx.labeled, f"scorer {name!r}", ctx.round_index
        )


@register_combiner("minmax_sum")
class MinMaxSumCombiner:
    """Min-max normalize every score, add them, and min-max normalize the sum again."""

    def combine(
        self, scores: Mapping[str, np.ndarray], ctx: RoundContext
    ) -> np.ndarray:
        if not scores:
            raise ValueError("combiner 'minmax_sum' needs at least one score")
        total = np.zeros(ctx.num_samples, dtype=np.float64)
        for name, score in scores.items():
            total += minmax_over_unlabeled(
                score, ctx.labeled, f"scorer {name!r}", ctx.round_index
            )
        names = ", ".join(repr(name) for name in scores)
        return minmax_over_unlabeled(
            total,
            ctx.labeled,
            f"the sum of the normalized scores of {names}",
            ctx.round_index,
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
