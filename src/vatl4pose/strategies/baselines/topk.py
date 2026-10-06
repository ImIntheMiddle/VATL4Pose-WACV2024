"""Top-k: query the unlabeled samples with the highest total score."""

from __future__ import annotations

import numpy as np

from ..base import RoundContext, unlabeled_for_query
from ..registry import register_selector


def rank_unlabeled(score: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
    """Return the ``k`` unlabeled samples with the highest ``score``, highest first.

    Samples with equal scores keep their index order (the sort is stable).
    """
    unlabeled = unlabeled_for_query(ctx, k)
    values = score[unlabeled]
    order = np.argsort(-values, kind="stable")[:k]
    return unlabeled[order]


@register_selector("topk")
class TopKSelector:
    """The ``k`` unlabeled samples with the highest total score."""

    needs_embeddings = False

    def select(self, total: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
        unlabeled = unlabeled_for_query(ctx, k)
        assert total.shape == (ctx.num_samples,), (
            f"total: expected shape ({ctx.num_samples},), got {total.shape}"
        )
        if not np.isfinite(total[unlabeled]).all():
            raise ValueError("selector 'topk': the total score has non-finite entries")
        return rank_unlabeled(total, ctx, k)
