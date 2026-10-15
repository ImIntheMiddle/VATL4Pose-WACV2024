"""Combiners of THC and WPU whose weights follow the share of labeled people.

They are the ablation rows "increase" and "decrease" of Table 3 of the paper.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..base import RoundContext
from ..combiners import minmax_over_unlabeled
from ..registry import register_combiner

WEIGHED_SCORERS = ("thc", "wpu")
"""The two scorers that these combiners weigh; the scores must be exactly these."""


class _ShareWeightedCombiner:
    """Weigh min-max normalized THC and WPU by the share ``r`` of labeled people.

    ``r`` is the number of labeled people at the start of the round divided by the number of all
    people of the video, 0 in round 0. The total score is ``w_thc * thc + w_wpu * wpu`` with
    ``(w_thc, w_wpu) = (r, 1 - r)`` or ``(1 - r, r)`` (class attribute ``thc_weight_rises``),
    min-max normalized over the unlabeled people again. The two subclasses are the rows
    "increase" and "decrease" of Table 3; the combiner ``minmax_sum`` is the row "const"
    (weights 1 and 1).
    """

    name: str  # set by the registry
    thc_weight_rises: bool

    def combine(
        self, scores: Mapping[str, np.ndarray], ctx: RoundContext
    ) -> np.ndarray:
        if set(scores) != set(WEIGHED_SCORERS):
            raise ValueError(
                f"combiner {self.name!r} needs the scores {list(WEIGHED_SCORERS)}, "
                f"got {sorted(scores)}"
            )
        thc, wpu = (
            minmax_over_unlabeled(
                scores[name], ctx.labeled, f"scorer {name!r}", ctx.round_index
            )
            for name in WEIGHED_SCORERS
        )
        share = int(ctx.labeled.sum()) / ctx.num_samples
        w_thc, w_wpu = (
            (share, 1 - share) if self.thc_weight_rises else (1 - share, share)
        )
        return minmax_over_unlabeled(
            w_thc * thc + w_wpu * wpu,
            ctx.labeled,
            "the weighted sum of the normalized scores of 'thc' and 'wpu'",
            ctx.round_index,
        )


@register_combiner("thc_wpu_increase")
class IncreasingThcCombiner(_ShareWeightedCombiner):
    """The weight of THC rises with the share of labeled people, that of WPU falls.

    The weights are ``r`` and ``1 - r``.
    """

    thc_weight_rises = True


@register_combiner("thc_wpu_decrease")
class DecreasingThcCombiner(_ShareWeightedCombiner):
    """The weight of THC falls with the share of labeled people, that of WPU rises.

    The weights are ``1 - r`` and ``r``.
    """

    thc_weight_rises = False
