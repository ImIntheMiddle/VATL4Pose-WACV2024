"""HP: highest probability. The least confident pose is queried first."""

from __future__ import annotations

import numpy as np

from ..base import RoundContext
from ..registry import register_scorer


@register_scorer("hp")
class HPScorer:
    """Score ``-sum_j max(H_j)``: the negative sum of the joints' peak values.

    The sum runs in the dtype of ``ctx.maxvals`` (float32 from the decoder), as in the 2023
    code, so the scores and their ties equal the 2023 ones bit for bit; the result is float64.
    """

    def score(self, ctx: RoundContext) -> np.ndarray:
        return (-ctx.maxvals.sum(axis=1)).astype(np.float64)
