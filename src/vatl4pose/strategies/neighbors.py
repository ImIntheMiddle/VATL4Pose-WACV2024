"""Aggregation of a per-neighbor value over the temporal neighbors of every sample."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import RoundContext

PairValue = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""``f(samples, neighbors) -> values``: one value per ``(sample, neighbor)`` pair, as float64."""


def temporal_neighbor_sum(ctx: RoundContext, pair_value: PairValue) -> np.ndarray:
    """Sum ``pair_value`` over the previous and the next neighbor of every sample.

    A sample with one neighbor gets twice that neighbor's value and a sample without neighbors
    gets 0, so that every sample is on the same scale (the rule of TPC and THC).

    Args:
        ctx: The round; ``prev_index`` and ``next_index`` give the neighbors.
        pair_value: Called once for the previous links and once for the next links, with the
            sample indices that have such a neighbor and their neighbor indices (both int64,
            same length). Returns the float64 value of each pair.

    Returns:
        ``(N,)`` float64 vector.
    """
    n = ctx.num_samples
    total = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    for neighbor in (ctx.prev_index, ctx.next_index):
        samples = np.flatnonzero(neighbor >= 0)
        if samples.shape[0] == 0:
            continue
        values = np.asarray(pair_value(samples, neighbor[samples]), dtype=np.float64)
        assert values.shape == samples.shape, (
            f"pair_value returned shape {values.shape} for {samples.shape[0]} pairs"
        )
        total[samples] += values
        count[samples] += 1
    return np.where(count == 1, 2.0 * total, total)
