"""Oracles: the sources of annotations for the people the active-learning loop queries."""

from .base import Oracle, OracleResult
from .coco_file import CocoFileOracle
from .simulated import SimulatedOracle

__all__ = ["CocoFileOracle", "Oracle", "OracleResult", "SimulatedOracle"]
