"""DUW: coreset with a dynamic uncertainty weight."""

from __future__ import annotations

import math

import numpy as np

from ..base import RoundContext, unlabeled_for_query
from ..baselines.coreset import greedy_k_center
from ..registry import register_selector


@register_selector("coreset_duw")
class DUWSelector:
    """Greedy k-center selection with an uncertainty term weighted by the previous round's mean OKS.

    The total score ``u`` (the combiner output, min-max normalised over the unlabeled samples) is
    the uncertainty. With ``d`` the Euclidean distance of a sample to the nearest labeled or already
    picked sample and ``m = ctx.moks_previous`` the mean OKS of the samples queried in the previous
    round (0 in round 0), every pick is the unlabeled sample that maximises
    ``(1 - m) * d + lam * m * u``. A small ``m`` (a poor previous round) weights the distance to the
    labeled set, a large ``m`` weights the uncertainty. With nothing labeled the first
    pick is the most uncertain sample (``argmax(u)``); that pick is deterministic, where the
    ``coreset`` selector draws it at random. Picked samples leave the candidates, which has the effect
    of the legacy ``u[pick] = 0``. The score ``total`` is not changed.

    The distance is the unscaled Euclidean distance of the embeddings, as in the legacy code. With
    ``lam = 0`` and ``m < 1`` the distance alone decides after the first pick, which is still the
    most uncertain sample. With ``lam = 0`` and ``m = 1`` every value is 0, so each pick is the
    lowest candidate index. The legacy code switched to its k-center with a random first pick at
    ``lam == 0``; that selection is the ``coreset`` selector here (preset ``_Coresetfilter``).

    Args:
        lam: Weight of the uncertainty term (``lam >= 0``). The experiment configs set it:
            0.01 for PoseTrack21, 1000 for JRDB-Pose.
    """

    needs_embeddings = True

    def __init__(self, lam: float) -> None:
        if not (math.isfinite(lam) and lam >= 0):
            raise ValueError(f"lam must be finite and non-negative, got {lam}")
        self.lam = float(lam)

    def select(self, total: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
        unlabeled = unlabeled_for_query(ctx, k)
        if ctx.embeddings is None:
            raise ValueError(
                "selector 'coreset_duw' needs ctx.embeddings; the pose model provides none"
            )
        assert total.shape == (ctx.num_samples,), (
            f"total: expected shape ({ctx.num_samples},), got {total.shape}"
        )
        if not np.isfinite(total[unlabeled]).all():
            raise ValueError(
                "selector 'coreset_duw': the total score has non-finite entries"
            )
        moks = float(ctx.moks_previous)

        def pick(min_distance: np.ndarray | None, candidates: np.ndarray) -> int:
            if min_distance is None:  # nothing labeled and nothing picked yet
                return int(candidates[np.argmax(total[candidates])])
            value = (1 - moks) * min_distance[candidates] + (
                self.lam * moks * total[candidates]
            )
            return int(candidates[np.argmax(value)])

        return greedy_k_center(ctx.embeddings, ctx.labeled, k, pick)
