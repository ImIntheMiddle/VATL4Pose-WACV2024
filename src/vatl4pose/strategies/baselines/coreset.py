"""Coreset: k-center greedy selection in embedding space."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import pairwise_distances

from ..base import RoundContext, unlabeled_for_query
from ..registry import register_selector

PickRule = Callable[[np.ndarray | None, np.ndarray], int]
"""``rule(min_distance, candidates) -> index`` chooses the next sample of the greedy selection.

``min_distance`` is the ``(N,)`` distance from every sample to the nearest labeled or already
picked sample, or ``None`` while there is no such sample. ``candidates`` holds the ascending
indices of the unlabeled samples not picked yet. The rule returns one of them.
"""


def greedy_k_center(
    embeddings: np.ndarray, labeled: np.ndarray, k: int, pick: PickRule
) -> np.ndarray:
    """Pick ``k`` unlabeled samples one at a time with a pick rule.

    The nearest-center distance of every sample (Euclidean, to the labeled samples and to the
    samples picked so far) is updated after each pick and handed to ``pick``.

    Args:
        embeddings: ``(N, D)`` embeddings.
        labeled: ``(N,)`` bool, True for samples that are centers from the start.
        k: Number of samples to pick, at most the number of unlabeled samples.
        pick: The rule that chooses the next sample.

    Returns:
        ``(k,)`` int64 indices in pick order.
    """
    n = embeddings.shape[0]
    assert embeddings.ndim == 2 and labeled.shape == (n,), (
        f"embeddings {embeddings.shape} and labeled {labeled.shape} disagree"
    )
    available = ~labeled
    min_distance: np.ndarray | None = None
    if labeled.any():
        min_distance = pairwise_distances(
            embeddings, embeddings[labeled], metric="euclidean"
        ).min(axis=1)
    picked: list[int] = []
    for _ in range(k):
        index = int(pick(min_distance, np.flatnonzero(available)))
        if not (0 <= index < n and available[index]):
            raise ValueError(
                f"the pick rule returned sample {index}, which is labeled, already picked "
                "or outside the pool"
            )
        picked.append(index)
        available[index] = False
        distance = pairwise_distances(
            embeddings, embeddings[[index]], metric="euclidean"
        ).reshape(-1)
        min_distance = (
            distance if min_distance is None else np.minimum(min_distance, distance)
        )
    return np.array(picked, dtype=np.int64)


@register_selector("coreset")
class CoresetSelector:
    """Pure k-center greedy: repeatedly pick the sample farthest from the labeled and picked ones.

    With nothing labeled the first pick is ``ctx.rng.choice`` over the unlabeled samples. Labeled
    samples are never picked. The total score is ignored.
    """

    def select(self, total: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
        unlabeled_for_query(ctx, k)
        if ctx.embeddings is None:
            raise ValueError(
                "selector 'coreset' needs ctx.embeddings; the pose model provides none"
            )

        def farthest(min_distance: np.ndarray | None, candidates: np.ndarray) -> int:
            if min_distance is None:
                return int(ctx.rng.choice(candidates))
            return int(candidates[np.argmax(min_distance[candidates])])

        return greedy_k_center(ctx.embeddings, ctx.labeled, k, farthest)
