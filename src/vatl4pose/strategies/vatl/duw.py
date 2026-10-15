"""DUW: Core-Set with Dynamic Uncertainty Weighting, and its fixed-weight ablation."""

from __future__ import annotations

import math

import numpy as np

from ..base import RoundContext, unlabeled_for_query
from ..baselines.coreset import greedy_k_center
from ..registry import register_selector


@register_selector("coreset_duw")
class DUWSelector:
    """Greedy k-center selection with an uncertainty term weighted by the previous round's mean OKS.

    The total score ``u`` (the combiner output, min-max normalized over the unlabeled people) is
    the uncertainty. With ``d`` the Euclidean distance of a person to the nearest labeled or already
    picked person and ``m = ctx.moks_previous`` the mean OKS of the people queried in the previous
    round (0 in round 0), every pick is the unlabeled person that maximizes
    ``(1 - m) * d + lam * m * u``. A small ``m`` (a poor previous round) weights the distance to the
    labeled set, a large ``m`` weights the uncertainty. With nothing labeled the first
    pick is the most uncertain person (``argmax(u)``); that pick is deterministic, where the
    ``coreset`` selector draws it at random. Picked people leave the candidates; the score
    ``total`` is not changed.

    The distance is the unscaled Euclidean distance of the embeddings. With ``lam = 0`` and
    ``m < 1`` the distance alone decides after the first pick; with ``lam = 0`` and ``m = 1``
    every value is 0, so every pick is the lowest candidate index, except the first pick when
    nothing is labeled.

    Args:
        lam: Weight of the uncertainty term (``lam >= 0``); the experiment config sets it.
    """

    needs_embeddings = True

    def __init__(self, lam: float) -> None:
        if not (math.isfinite(lam) and lam >= 0):
            raise ValueError(f"lam must be finite and non-negative, got {lam}")
        self.lam = float(lam)

    def _weights(self, ctx: RoundContext) -> tuple[float, float]:
        """Weights ``(distance, uncertainty)`` of the two terms of a pick in this round."""
        moks = float(ctx.moks_previous)
        return 1 - moks, self.lam * moks

    def select(self, total: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
        unlabeled = unlabeled_for_query(ctx, k)
        if ctx.embeddings is None:
            raise ValueError(
                f"selector {self.name!r} needs ctx.embeddings; the pose estimator provides none"
            )
        assert total.shape == (ctx.num_samples,), (
            f"total: expected shape ({ctx.num_samples},), got {total.shape}"
        )
        if not np.isfinite(total[unlabeled]).all():
            raise ValueError(
                f"selector {self.name!r}: the total score has non-finite entries"
            )
        w_distance, w_uncertainty = self._weights(ctx)

        def pick(min_distance: np.ndarray | None, candidates: np.ndarray) -> int:
            if min_distance is None:  # nothing labeled and nothing picked yet
                return int(candidates[np.argmax(total[candidates])])
            value = w_distance * min_distance[candidates] + (
                w_uncertainty * total[candidates]
            )
            return int(candidates[np.argmax(value)])

        return greedy_k_center(ctx.embeddings, ctx.labeled, k, pick)


@register_selector("coreset_duw_fixed")
class FixedDUWSelector(DUWSelector):
    """DUW without the dynamic weight: greedy k-center with a constant uncertainty weight.

    Every pick is the unlabeled person that maximizes ``d + lam * u``, with ``d`` and ``u`` as in
    :class:`DUWSelector`; the previous round's mean OKS ``m`` does not enter. This is the "fixed"
    row of Table 3 of the paper, the ablation of the dynamic weight ``(1 - m) * d + lam * m * u``.
    The first pick with nothing labeled is that of :class:`DUWSelector`.

    Args:
        lam: Weight of the uncertainty term (``lam >= 0``).
    """

    needs_embeddings = True

    def _weights(self, ctx: RoundContext) -> tuple[float, float]:
        return 1.0, self.lam
