"""VATL, the proposed active-learning method: the THC and WPU uncertainties and the DUW coreset.

Importing this package registers

* scorer ``thc``, the temporal heatmap continuity (:mod:`.thc`),
* scorer ``wpu``, the whole-body pose uncertainty (:mod:`.wpu`),
* selector ``coreset_duw``, the coreset with a dynamic uncertainty weight (:mod:`.duw`).

The package is a plugin of the strategy framework: ``registry.PLUGIN_MODULES`` lists it, so a
config that names one of these components works without importing the package, and no other module
of the framework refers to it. Deleting this package and that one list entry removes the proposal.
The strategy presets that combine the components are in :mod:`vatl4pose.strategies.presets`.
"""

from . import duw, thc, wpu

__all__ = ["duw", "thc", "wpu"]
