"""DUW: coreset with a dynamic uncertainty weight, and its fixed-weight ablation."""

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

    def _weights(self, ctx: RoundContext) -> tuple[float, float]:
        """Weights ``(distance, uncertainty)`` of the two terms of a pick in this round."""
        moks = float(ctx.moks_previous)
        return 1 - moks, self.lam * moks

    def select(self, total: np.ndarray, ctx: RoundContext, k: int) -> np.ndarray:
        unlabeled = unlabeled_for_query(ctx, k)
        if ctx.embeddings is None:
            raise ValueError(
                f"selector {self.name!r} needs ctx.embeddings; the pose model provides none"
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

    Every pick is the unlabeled sample that maximises ``d + lam * u``, with ``d`` and ``u`` as in
    :class:`DUWSelector`; the previous round's mean OKS ``m`` does not enter. This is the
    ``--fixed_lambda`` option of the 2023 code (the "fixed" row of paper Table 3), the ablation of
    the dynamic weight ``(1 - m) * d + lam * m * u``. The first pick with nothing labeled, the
    candidate handling and the unscaled distance are those of :class:`DUWSelector`. The legacy code
    took its pure k-center branch at ``lam == 0`` before it looked at ``--fixed_lambda``; here
    ``lam = 0`` keeps this selector, so that the distance alone decides after the first pick,
    which is still the most uncertain sample.

    Args:
        lam: Weight of the uncertainty term (``lam >= 0``); the experiment configs set it, 0.01
            for the paper's PoseTrack21 runs.
    """

    needs_embeddings = True

    def _weights(self, ctx: RoundContext) -> tuple[float, float]:
        return 1.0, self.lam
