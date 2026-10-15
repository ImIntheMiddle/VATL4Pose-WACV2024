"""The package ``vatl``: the proposed active-learning method with THC, WPU and DUW.

Importing this package registers

* scorer ``thc``, Temporal Heatmap Continuity, THC (:mod:`.thc`),
* scorer ``wpu``, Whole-body Pose Unnaturalness, WPU (:mod:`.wpu`),
* selector ``coreset_duw``, Core-Set with Dynamic Uncertainty Weighting, DUW (:mod:`.duw`),
* selector ``coreset_duw_fixed``, the same Core-Set with a fixed uncertainty weight (:mod:`.duw`),
* combiners ``thc_wpu_increase`` and ``thc_wpu_decrease``, THC and WPU weighted by the share of
  labeled people (:mod:`.combiners`).

The package is a plugin of the strategy framework: ``registry.PLUGIN_MODULES`` lists it, so a
config that names one of these components works without importing the package, and no other module
of the framework refers to it. Deleting this package and that one list entry removes the
proposed method.
"""

from . import combiners, duw, thc, wpu

__all__ = ["combiners", "duw", "thc", "wpu"]
