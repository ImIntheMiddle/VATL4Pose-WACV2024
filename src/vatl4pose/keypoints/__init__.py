"""Keypoint specifications of the supported pose datasets."""

from .spec import (
    COCO17_SIGMAS,
    JRDB_ORDER_SIGMAS,
    KeypointSpec,
    get_spec,
    spec_from_mapping,
)

__all__ = [
    "COCO17_SIGMAS",
    "JRDB_ORDER_SIGMAS",
    "KeypointSpec",
    "get_spec",
    "spec_from_mapping",
]
