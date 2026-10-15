"""Random: query unlabeled people drawn uniformly at random."""

from __future__ import annotations

import numpy as np

from ..base import RoundContext, unlabeled_for_query
from ..registry import register_selector
from .topk import rank_unlabeled


@register_selector("random")
class RandomSelector:
    """Draw one uniform score per unlabeled person and take the ``k`` highest.

    One call ``ctx.rng.random(U)`` draws the scores of the ``U`` unlabeled people, in index order.
    The total score is ignored.
    """

    needs_embeddings = False

    def select(self, total: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
        unlabeled = unlabeled_for_query(ctx, k)
        draw = np.asarray(ctx.rng.random(unlabeled.shape[0]), dtype=np.float64)
        assert draw.shape == unlabeled.shape, (
            f"rng.random returned shape {draw.shape} for {unlabeled.shape[0]} unlabeled people"
        )
        score = np.zeros(ctx.num_samples, dtype=np.float64)
        score[unlabeled] = draw
        return rank_unlabeled(score, ctx, k)
