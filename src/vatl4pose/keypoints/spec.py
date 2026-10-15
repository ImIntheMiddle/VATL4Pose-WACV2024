"""Dataset-dependent keypoint specifications.

Every component that depends on the joint layout (heatmap heads, horizontal-flip
augmentation, OKS evaluation, the whole-body autoencoder input) derives its sizes
from a :class:`KeypointSpec`. Presets exist for the datasets of the paper
(``posetrack21``, ``jrdb_pose``) and for plain COCO (``coco17``). Other datasets
describe themselves with :func:`spec_from_mapping`.

All joint indices are 0-indexed.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

COCO17_SIGMAS: tuple[float, ...] = (
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
)
"""COCO OKS per-joint sigmas, already divided by 10, in COCO joint order."""

JRDB_ORDER_SIGMAS: tuple[float, ...] = (
    0.079,
    0.025,
    0.025,
    0.079,
    0.026,
    0.079,
    0.072,
    0.072,
    0.107,
    0.062,
    0.107,
    0.107,
    0.062,
    0.087,
    0.087,
    0.089,
    0.089,
)
"""OKS per-joint sigmas, already divided by 10, in the joint order of JRDB-Pose."""

LEGACY_JRDB_FLIP_SWAPS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (0, 4),
    (3, 4),
    (8, 10),
    (5, 7),
    (10, 13),
    (14, 16),
    (4, 5),
    (7, 12),
    (4, 8),
    (3, 6),
    (13, 15),
    (11, 14),
    (6, 9),
    (8, 11),
)
"""The JRDB-Pose flip list of the 2023 code, as ordered swaps.

Joints appear in several pairs, so the swaps do not form a left/right mirror.
"""

LEGACY_COCO_ANGLE_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    (8, 6, 12),
    (6, 8, 10),
    (5, 7, 9),
    (7, 5, 11),
    (11, 12, 14),
    (12, 11, 13),
    (12, 14, 16),
    (11, 13, 15),
)
"""The angle triplets ``(a, b, c)`` of the 2023 code: the angle at joint ``b`` between ``a-b`` and ``b-c``.

They index joints in COCO order. The ``posetrack21`` and ``coco17`` presets use them. The 2023 code
applied the same indices to the JRDB-Pose joint order, where they name other body parts; the
``jrdb_pose`` preset uses triplets at the JRDB-Pose joints of the same body parts instead.
"""

_JRDB_ANGLE_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    (6, 3, 10),
    (3, 6, 9),
    (5, 7, 12),
    (7, 5, 11),
    (11, 10, 13),
    (10, 11, 14),
    (10, 13, 15),
    (11, 14, 16),
)
"""The triplets of :data:`LEGACY_COCO_ANGLE_TRIPLETS` at the JRDB-Pose joints of the same body parts.

Each COCO joint maps to the JRDB-Pose joint of the same name (JRDB-Pose has feet, no ankles), and
the triplets keep their order, so the feature has the same 42 values in the same order.
"""


def _as_sequence(where: str, value: Any) -> Sequence[Any]:
    """Return ``value`` if it is a non-string sequence, else raise ``ValueError``."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{where}: expected a sequence, got {type(value).__name__}")
    return value


def _checked_index(where: str, value: Any, num_joints: int) -> int:
    """Return ``value`` as an ``int`` in ``[0, num_joints)``, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{where}: index {value!r} is not an integer")
    index = int(value)
    if not 0 <= index < num_joints:
        raise ValueError(f"{where}: index {index} is out of range [0, {num_joints})")
    return index


def _checked_index_tuples(
    field: str, value: Any, arity: int, num_joints: int
) -> tuple[tuple[int, ...], ...]:
    """Validate a sequence of ``arity``-tuples of joint indices (flip pairs, edges, triplets)."""
    entries = _as_sequence(field, value)
    checked: list[tuple[int, ...]] = []
    for k, entry in enumerate(entries):
        where = f"{field}[{k}]"
        items = _as_sequence(where, entry)
        if len(items) != arity:
            raise ValueError(f"{where}: expected {arity} indices, got {len(items)}")
        checked.append(tuple(_checked_index(where, item, num_joints) for item in items))
    return tuple(checked)


@dataclass(frozen=True)
class KeypointSpec:
    """Joint layout of one pose dataset.

    Attributes:
        name: Identifier of the layout.
        joint_names: One unique name per joint; the order defines the joint indices.
        flip_pairs: Index swaps applied to the joints on a horizontal flip. The pairs form
            an ordered sequence: the flip helper swaps the first pair, then the second pair
            on the result, and so on. Pairs may overlap, so a joint can be moved more than
            once and the order of the pairs changes the result. The presets list disjoint
            left/right pairs; joints without a partner are not listed.
        skeleton: Joint pairs connected by a limb (the links of the VL4Pose baseline, the skeleton
            of exported COCO files, drawing).
        oks_sigmas: Per-joint OKS sigmas, already divided by 10.
        redundant_joints: Joints that are never annotated. They are excluded from the
            WPU error (see :meth:`wpu_keep_index`).
        angle_triplets: Triplets ``(a, b, c)`` of joints; the hybrid feature holds the
            angle at ``b`` between the segments ``a-b`` and ``b-c``.

    Construction validates every field and stores sequences as tuples, so a spec cannot
    hold an index that is out of range.
    """

    name: str
    joint_names: tuple[str, ...]
    flip_pairs: tuple[tuple[int, int], ...]
    skeleton: tuple[tuple[int, int], ...]
    oks_sigmas: tuple[float, ...]
    redundant_joints: tuple[int, ...]
    angle_triplets: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name: must be a non-empty string")

        names = _as_sequence("joint_names", self.joint_names)
        if len(names) == 0:
            raise ValueError("joint_names: at least one joint is required")
        seen_names: set[str] = set()
        for k, joint_name in enumerate(names):
            if not isinstance(joint_name, str):
                raise ValueError(
                    f"joint_names[{k}]: expected a string, got {type(joint_name).__name__}"
                )
            if joint_name in seen_names:
                raise ValueError(
                    f"joint_names[{k}]: duplicate joint name {joint_name!r}"
                )
            seen_names.add(joint_name)
        num_joints = len(names)

        sigmas = _as_sequence("oks_sigmas", self.oks_sigmas)
        if len(sigmas) != num_joints:
            raise ValueError(
                f"oks_sigmas: expected {num_joints} values (one per joint), got {len(sigmas)}"
            )
        for k, sigma in enumerate(sigmas):
            if isinstance(sigma, bool) or not isinstance(sigma, numbers.Real):
                raise ValueError(f"oks_sigmas[{k}]: sigma {sigma!r} is not a number")
            if not (math.isfinite(sigma) and sigma > 0):
                raise ValueError(
                    f"oks_sigmas[{k}]: sigma must be positive and finite, got {sigma!r}"
                )

        flip_pairs = _checked_index_tuples("flip_pairs", self.flip_pairs, 2, num_joints)
        for k, pair in enumerate(flip_pairs):
            if pair[0] == pair[1]:
                raise ValueError(
                    f"flip_pairs[{k}]: a joint cannot be paired with itself"
                )

        skeleton = _checked_index_tuples("skeleton", self.skeleton, 2, num_joints)
        for k, edge in enumerate(skeleton):
            if edge[0] == edge[1]:
                raise ValueError(f"skeleton[{k}]: an edge needs two different joints")

        redundant = _as_sequence("redundant_joints", self.redundant_joints)
        redundant_joints: list[int] = []
        for k, joint in enumerate(redundant):
            joint = _checked_index(f"redundant_joints[{k}]", joint, num_joints)
            if joint in redundant_joints:
                raise ValueError(
                    f"redundant_joints[{k}]: joint {joint} is listed twice"
                )
            redundant_joints.append(joint)

        triplets = _checked_index_tuples(
            "angle_triplets", self.angle_triplets, 3, num_joints
        )
        for k, triplet in enumerate(triplets):
            if len(set(triplet)) != 3:
                raise ValueError(
                    f"angle_triplets[{k}]: a triplet needs three different joints"
                )

        # The dataclass is frozen, so the canonical (tuple) forms are stored via object.__setattr__.
        object.__setattr__(self, "joint_names", tuple(names))
        object.__setattr__(self, "oks_sigmas", tuple(float(s) for s in sigmas))
        object.__setattr__(self, "flip_pairs", flip_pairs)
        object.__setattr__(self, "skeleton", skeleton)
        object.__setattr__(self, "redundant_joints", tuple(redundant_joints))
        object.__setattr__(self, "angle_triplets", triplets)

    @property
    def num_joints(self) -> int:
        """Number of joints."""
        return len(self.joint_names)

    @property
    def ae_input_dim(self) -> int:
        """Length of the hybrid feature vector: x offsets, y offsets, then the angles."""
        return 2 * self.num_joints + len(self.angle_triplets)

    def wpu_keep_index(self) -> np.ndarray:
        """Indices of the hybrid feature vector that WPU alone compares.

        The hybrid vector holds the x offsets of all joints (indices ``0..J-1``), the
        y offsets (``J..2J-1``) and the angles (``2J..``). WPU drops the x and y offsets
        of the redundant joints and keeps everything else.

        Returns:
            A new sorted ``int64`` array of shape ``(ae_input_dim - 2 * len(redundant_joints),)``.
        """
        drop = np.array(
            [
                offset + joint
                for joint in self.redundant_joints
                for offset in (0, self.num_joints)
            ],
            dtype=np.int64,
        )
        keep = np.setdiff1d(np.arange(self.ae_input_dim, dtype=np.int64), drop)
        assert keep.dtype == np.int64
        assert keep.shape == (self.ae_input_dim - 2 * len(self.redundant_joints),), (
            keep.shape
        )
        return keep


# PoseTrack21 ships its skeleton 1-indexed in the annotation JSON; the preset converts it.
_POSETRACK21_JSON_SKELETON: tuple[tuple[int, int], ...] = (
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
)

_POSETRACK21 = KeypointSpec(
    name="posetrack21",
    joint_names=(
        "nose",
        "head_bottom",
        "head_top",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ),
    flip_pairs=((3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
    skeleton=tuple((a - 1, b - 1) for a, b in _POSETRACK21_JSON_SKELETON),
    oks_sigmas=COCO17_SIGMAS,
    redundant_joints=(3, 4),  # the ears are not annotated in PoseTrack21
    angle_triplets=LEGACY_COCO_ANGLE_TRIPLETS,
)

# The flip pairs join the left and right joints.
_JRDB_POSE = KeypointSpec(
    name="jrdb_pose",
    joint_names=(
        "head",
        "right eye",
        "left eye",
        "right shoulder",
        "center shoulder",
        "left shoulder",
        "right elbow",
        "left elbow",
        "center hip",
        "right wrist",
        "right hip",
        "left hip",
        "left wrist",
        "right knee",
        "left knee",
        "right foot",
        "left foot",
    ),
    flip_pairs=((1, 2), (3, 5), (6, 7), (9, 12), (10, 11), (13, 14), (15, 16)),
    skeleton=(
        (1, 2),
        (0, 4),
        (3, 4),
        (8, 10),
        (5, 7),
        (10, 13),
        (14, 16),
        (4, 5),
        (7, 12),
        (4, 8),
        (3, 6),
        (13, 15),
        (11, 14),
        (6, 9),
        (8, 11),
    ),
    oks_sigmas=JRDB_ORDER_SIGMAS,
    redundant_joints=(),
    angle_triplets=_JRDB_ANGLE_TRIPLETS,
)

_COCO17 = KeypointSpec(
    name="coco17",
    joint_names=(
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ),
    flip_pairs=((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)),
    skeleton=(
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    ),
    oks_sigmas=COCO17_SIGMAS,
    redundant_joints=(),
    angle_triplets=LEGACY_COCO_ANGLE_TRIPLETS,
)

_PRESETS: dict[str, KeypointSpec] = {
    spec.name: spec for spec in (_POSETRACK21, _JRDB_POSE, _COCO17)
}


def get_spec(name: str) -> KeypointSpec:
    """Return the preset specification called ``name``.

    Args:
        name: One of ``"posetrack21"``, ``"jrdb_pose"``, ``"coco17"``.

    Raises:
        ValueError: If ``name`` is not a preset. Custom datasets use :func:`spec_from_mapping`.
    """
    if name not in _PRESETS:
        raise ValueError(
            f"unknown keypoint spec {name!r}; presets: {', '.join(_PRESETS)}"
        )
    return _PRESETS[name]


def spec_from_mapping(m: Mapping[str, Any]) -> KeypointSpec:
    """Build a specification for a custom dataset from a mapping (e.g. a Hydra config node).

    The mapping must hold exactly the fields of :class:`KeypointSpec`: ``name``,
    ``joint_names``, ``flip_pairs``, ``skeleton``, ``oks_sigmas``, ``redundant_joints`` and
    ``angle_triplets``. Sequences may be lists; they are stored as tuples.

    Raises:
        ValueError: If a field is missing or unknown, if ``oks_sigmas`` does not have one
            value per joint, or if an index is out of range or a flip pair joins a joint
            with itself. The message names the offending field and entry.
    """
    field_names = tuple(f.name for f in fields(KeypointSpec))
    missing = [f for f in field_names if f not in m]
    if missing:
        raise ValueError(f"spec_from_mapping: missing field(s) {missing}")
    unknown = sorted(str(k) for k in m if k not in field_names)
    if unknown:
        raise ValueError(
            f"spec_from_mapping: unknown field(s) {unknown}; allowed fields: {list(field_names)}"
        )
    return KeypointSpec(**{f: m[f] for f in field_names})
