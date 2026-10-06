"""Active-learning query strategies.

A strategy is a scorer (or several), a combiner and a selector, named in a
:class:`StrategyConfig` and built with :func:`build_strategy`. Importing this package registers
the baselines: scorers ``hp``, ``tpc``, ``mpe``; combiners ``single``, ``minmax_sum``, ``none``;
selectors ``topk``, ``random``, ``kmeans``, ``coreset``.

Extension point. Another method is a package that registers its components with
:func:`register_scorer`, :func:`register_combiner` and :func:`register_selector` when it is
imported, plus one entry for that package in ``registry.PLUGIN_MODULES``. A config that names one
of its components then works without any import by the caller: the first lookup of an unknown
name imports the listed modules and retries. The baselines do not change.
"""

from . import combiners as combiners  # registers the combiners
from .base import (
    Combiner,
    RoundContext,
    Scorer,
    Selector,
    Strategy,
    StrategyConfig,
)
from .baselines import local_peak_mean  # registers the baselines
from .registry import (
    COMBINERS,
    SCORERS,
    SELECTORS,
    build_strategy,
    register_combiner,
    register_scorer,
    register_selector,
)

__all__ = [
    "COMBINERS",
    "SCORERS",
    "SELECTORS",
    "Combiner",
    "RoundContext",
    "Scorer",
    "Selector",
    "Strategy",
    "StrategyConfig",
    "build_strategy",
    "local_peak_mean",
    "register_combiner",
    "register_scorer",
    "register_selector",
]
