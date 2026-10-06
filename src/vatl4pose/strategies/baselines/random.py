"""Random: query unlabeled samples drawn uniformly at random."""

from __future__ import annotations

import numpy as np

from ..base import RoundContext, unlabeled_for_query
from ..registry import register_selector
from .topk import rank_unlabeled


@register_selector("random")
class RandomSelector:
    """Draw one uniform score per unlabeled sample and take the ``k`` highest.

    One call ``ctx.rng.random(U)`` serves ``U`` unlabeled samples, in index order, so a context
    whose ``rng`` is the legacy ``numpy`` stream reproduces the legacy ``np.random.rand(U)``
    draw. The total score is ignored.
    """

    def select(self, total: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
        unlabeled = unlabeled_for_query(ctx, k)
        draw = np.asarray(ctx.rng.random(unlabeled.shape[0]), dtype=np.float64)
        assert draw.shape == unlabeled.shape, (
            f"rng.random returned shape {draw.shape} for {unlabeled.shape[0]} samples"
        )
        score = np.zeros(ctx.num_samples, dtype=np.float64)
        score[unlabeled] = draw
        return rank_unlabeled(score, ctx, k)
