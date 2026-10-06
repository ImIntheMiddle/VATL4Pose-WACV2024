"""Dataset annotation readers."""

from .annotations import (
    PersonRecord,
    VideoAnnotation,
    link_neighbors,
    read_coco_custom_video,
    read_jrdb_pose_video,
    read_posetrack21_video,
)

__all__ = [
    "PersonRecord",
    "VideoAnnotation",
    "link_neighbors",
    "read_coco_custom_video",
    "read_jrdb_pose_video",
    "read_posetrack21_video",
]
