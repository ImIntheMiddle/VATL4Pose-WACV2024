"""Strategy presets: the baselines and the proposed combinations of the paper, by name.

Each preset is a :class:`StrategyConfig` for :func:`~vatl4pose.strategies.build_strategy`. The
names are the ones of the paper's tables. This module imports no strategy component: a config
only names them, and the registry imports the plugin that holds a component on first use (see
:data:`vatl4pose.strategies.registry.PLUGIN_MODULES`), so the baseline presets build without the
proposed method being imported.

The presets with a ``coreset_duw`` selector carry the default weight ``lam = 0.01``. An experiment
config overrides it with ``dataclasses.replace(PRESETS[name], params={"lam": ...})`` (PoseTrack21
0.01, JRDB-Pose 1000). The ``VL4Pose`` preset names the scorer ``vl4pose`` of the VL4Pose baseline.
"""

from __future__ import annotations

from .base import StrategyConfig

DEFAULT_LAM = 0.01
"""Default weight of the uncertainty term of the ``coreset_duw`` selector."""

PRESETS: dict[str, StrategyConfig] = {
    cfg.name: cfg
    for cfg in (
        StrategyConfig("Random", (), "none", "random", {}),
        StrategyConfig("HP", ("hp",), "single", "topk", {}),
        StrategyConfig("TPC", ("tpc",), "single", "topk", {}),
        StrategyConfig("MPE", ("mpe",), "single", "topk", {}),
        StrategyConfig("THC", ("thc",), "single", "topk", {}),
        StrategyConfig("WPU", ("wpu",), "single", "topk", {}),
        StrategyConfig("THC+WPU", ("thc", "wpu"), "minmax_sum", "topk", {}),
        StrategyConfig("_K-Meansfilter", (), "none", "kmeans", {}),
        StrategyConfig("_Coresetfilter", (), "none", "coreset", {}),
        StrategyConfig(
            "THC_Coresetfilter",
            ("thc",),
            "single",
            "coreset_duw",
            {"lam": DEFAULT_LAM},
        ),
        StrategyConfig(
            "WPU_Coresetfilter",
            ("wpu",),
            "single",
            "coreset_duw",
            {"lam": DEFAULT_LAM},
        ),
        StrategyConfig(
            "THC+WPU_Coresetfilter",
            ("thc", "wpu"),
            "minmax_sum",
            "coreset_duw",
            {"lam": DEFAULT_LAM},
        ),
        StrategyConfig("VL4Pose", ("vl4pose",), "single", "topk", {}),
    )
}
"""All presets by name."""
