"""The baseline strategies: HP, TPC, MPE, VL4Pose, Random, K-Means and Coreset.

Importing this package registers their components (see :mod:`vatl4pose.strategies.registry`).
"""

from . import coreset, hp, kmeans, mpe, random, topk, tpc, vl4pose
from .local_peak import local_peak_mean

__all__ = [
    "coreset",
    "hp",
    "kmeans",
    "local_peak_mean",
    "mpe",
    "random",
    "topk",
    "tpc",
    "vl4pose",
]
