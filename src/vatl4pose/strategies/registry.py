"""Registries of scorers, combiners and selectors, and the builder of a strategy.

Every component registers itself under a name with a decorator::

    @register_scorer("my_score")
    class MyScorer:
        def score(self, ctx: RoundContext) -> np.ndarray: ...

The package registers the baselines when it is imported. Another method (the proposed one,
VL4Pose) is a package of its own that registers its components when it is imported; nothing in
the baselines changes. The module list :data:`PLUGIN_MODULES` names the packages that hold
components. A name that no registry knows yet triggers the import of every listed module that is
not imported yet, and the lookup is retried once, so a :class:`StrategyConfig` can name a
component of a plugin without the caller importing the plugin. :func:`build_strategy` then builds
a :class:`Strategy` from the config.

The registries map a name to a factory (normally the class). A registered class gets its
registry name as the class attribute ``name``.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from typing import Any

from .base import Combiner, Scorer, Selector, Strategy, StrategyConfig

SCORERS: dict[str, Callable[..., Scorer]] = {}
COMBINERS: dict[str, Callable[..., Combiner]] = {}
SELECTORS: dict[str, Callable[..., Selector]] = {}

PLUGIN_MODULES: list[str] = ["vatl4pose.strategies.baselines"]
"""Modules whose import registers components. A method with its own components appends its
package here; the entries are imported on the first lookup of a name that is not registered."""


def _register(
    table: dict[str, Callable[..., Any]], kind: str, name: str
) -> Callable[[type], type]:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{kind} name must be a non-empty string, got {name!r}")

    def decorate(cls: type) -> type:
        if name in table:
            raise ValueError(
                f"{kind} {name!r} is already registered (by {table[name]!r}); "
                f"cannot register {cls!r} under the same name"
            )
        table[name] = cls
        cls.name = name  # type: ignore[attr-defined]
        return cls

    return decorate


def register_scorer(name: str) -> Callable[[type], type]:
    """Class decorator that registers a scorer under ``name``.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    return _register(SCORERS, "scorer", name)


def register_combiner(name: str) -> Callable[[type], type]:
    """Class decorator that registers a combiner under ``name``.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    return _register(COMBINERS, "combiner", name)


def register_selector(name: str) -> Callable[[type], type]:
    """Class decorator that registers a selector under ``name``.

    Raises:
        ValueError: If ``name`` is already registered.
    """
    return _register(SELECTORS, "selector", name)


def _find(
    table: Mapping[str, Callable[..., Any]], name: str
) -> Callable[..., Any] | None:
    """Return the factory registered as ``name``, importing the plugin modules on a miss.

    The first miss imports every module of :data:`PLUGIN_MODULES` that is not imported yet and
    looks once more; ``None`` means the name is still unknown.
    """
    if name not in table:
        for module in PLUGIN_MODULES:
            if module not in sys.modules:
                importlib.import_module(module)
    return table.get(name)


def _lookup(table: Mapping[str, Callable[..., Any]], kind: str, name: str) -> Any:
    factory = _find(table, name)
    if factory is None:
        raise ValueError(
            f"unknown {kind} {name!r}; registered {kind}s: {', '.join(sorted(table))}; "
            f"plugin modules tried: {', '.join(PLUGIN_MODULES)}"
        )
    return factory


def _construct(
    table: Mapping[str, Callable[..., Any]],
    kind: str,
    name: str,
    kwargs: Mapping[str, Any],
) -> Any:
    """Build component ``name`` with ``kwargs``; a refused argument names the component."""
    factory = _lookup(table, kind, name)
    try:
        return factory(**kwargs)
    except TypeError as err:
        raise TypeError(
            f"cannot build {kind} {name!r} with arguments {sorted(kwargs)}: {err}"
        ) from err


def build_strategy(cfg: StrategyConfig) -> Strategy:
    """Build the strategy that ``cfg`` describes.

    Scorer ``s`` is built with the mapping ``cfg.params[s]`` as keyword arguments (none if the
    key is absent). Every other entry of ``cfg.params`` is a keyword argument of the selector.
    The combiner is built without arguments. A name that is not registered yet is looked up in
    the plugin modules (:data:`PLUGIN_MODULES`) first; this includes the keys of ``cfg.params``
    that are not among ``cfg.scorers``, so that a key naming a scorer of a plugin is recognised
    as that scorer's.

    Raises:
        ValueError: If a component name is not registered, or ``cfg.params`` names a registered
            scorer that is not among ``cfg.scorers``, or the arguments of a scorer are not a
            mapping.
        TypeError: If a component does not accept an argument given in ``cfg.params``.
    """
    stray = sorted(
        key
        for key in cfg.params
        if key not in cfg.scorers and _find(SCORERS, key) is not None
    )
    if stray:
        raise ValueError(
            f"strategy {cfg.name!r}: params has arguments for scorer(s) {stray}, "
            f"which are not among its scorers {list(cfg.scorers)}"
        )
    scorers: list[Scorer] = []
    for name in cfg.scorers:
        kwargs = cfg.params.get(name, {})
        if not isinstance(kwargs, Mapping):
            raise ValueError(
                f"strategy {cfg.name!r}: params[{name!r}] must be a mapping of the arguments "
                f"of scorer {name!r}, got {type(kwargs).__name__}"
            )
        scorers.append(_construct(SCORERS, "scorer", name, kwargs))
    combiner = _construct(COMBINERS, "combiner", cfg.combiner, {})
    selector_kwargs = {k: v for k, v in cfg.params.items() if k not in cfg.scorers}
    selector = _construct(SELECTORS, "selector", cfg.selector, selector_kwargs)
    return Strategy(cfg, scorers, combiner, selector)
