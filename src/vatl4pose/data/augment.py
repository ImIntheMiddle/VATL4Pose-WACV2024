"""Training augmentation: random scale, rotation and horizontal flip of a person crop.

:func:`augment_and_crop` draws the augmentation of one training sample and returns the crop with
the joints in crop coordinates. The draws follow the 2023 fine-tuning code, in this order:

1. Scale: the crop size is multiplied by ``clip(1 + sf * N(0, 1), 1 - sf, 1 + sf)``.
2. Rotation: with probability ``rotation_prob`` the angle is ``clip(rf * N(0, 1), -2 rf, 2 rf)``
   degrees, otherwise 0.
3. Flip: with probability ``flip_prob`` the frame is mirrored, the crop centre is mirrored with
   it (``x -> W - 1 - x``) and the joints swap along the flip pairs of the keypoint spec, applied
   as an ordered sequence of swaps (see :func:`flip_joints`). No random number is drawn when
   ``flip`` is off.

The legacy half-body transform is not ported: its probability is -1 in the configs of both
datasets, so it never ran.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vatl4pose.keypoints import KeypointSpec

from .geometry import (
    INPUT_SIZE,
    box_to_center_scale,
    get_affine_transform,
    warp_crop,
)


@dataclass(frozen=True)
class AugmentConfig:
    """Strength and probabilities of the training augmentation.

    Attributes:
        scale_factor: ``sf``; the crop scale is drawn from ``1 +- sf`` (clipped normal draw).
        rotation_factor: ``rf``; the rotation in degrees is a clipped normal draw of std ``rf``.
        rotation_prob: Probability that a sample is rotated at all.
        flip: Whether horizontal flips are applied.
        flip_prob: Probability of a flip when ``flip`` is on.
    """

    scale_factor: float = 0.3
    rotation_factor: float = 40.0
    rotation_prob: float = 0.6
    flip: bool = True
    flip_prob: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.scale_factor < 1.0:
            raise ValueError(f"scale_factor must be in [0, 1), got {self.scale_factor}")
        if self.rotation_factor < 0.0:
            raise ValueError(
                f"rotation_factor must be >= 0, got {self.rotation_factor}"
            )
        if not 0.0 <= self.rotation_prob <= 1.0:
            raise ValueError(
                f"rotation_prob must be in [0, 1], got {self.rotation_prob}"
            )
        if not 0.0 <= self.flip_prob <= 1.0:
            raise ValueError(f"flip_prob must be in [0, 1], got {self.flip_prob}")


def flip_joints(
    joints: np.ndarray, image_width: int, flip_pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Mirror joints horizontally in an image of width ``image_width``.

    Each x becomes ``image_width - x - 1``, the rows of the flip pairs swap (with their
    visibility), and the position of a joint that is not visible becomes ``(0, 0)``.

    The pairs are applied as an ordered sequence of swaps, first to last, each on the result of
    the previous one, like the legacy code. They need not be disjoint: a joint in two pairs is
    moved twice, and the order of the pairs changes the result.

    Args:
        joints: ``(K, 3)`` float array with columns x, y, visibility.
        image_width: Width in pixels of the image the joints refer to.
        flip_pairs: Index pairs ``(a, b)``; each swaps rows ``a`` and ``b`` in the given order.

    Returns:
        A new ``(K, 3)`` array of the dtype of ``joints``.
    """
    assert joints.ndim == 2 and joints.shape[1] == 3, (
        f"joints: expected shape (K, 3), got {joints.shape}"
    )
    assert np.issubdtype(joints.dtype, np.floating), (
        f"joints: expected float, got {joints.dtype}"
    )
    num_joints = joints.shape[0]
    out = joints.copy()
    out[:, 0] = image_width - out[:, 0] - 1
    for left, right in flip_pairs:
        assert 0 <= left < num_joints and 0 <= right < num_joints, (
            f"flip pair ({left}, {right}) is out of range for {num_joints} joints"
        )
        out[[left, right]] = out[[right, left]]
    out[:, :2] *= out[:, 2:3]
    return out


def augment_and_crop(
    image_rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    keypoints: np.ndarray,
    spec: KeypointSpec,
    cfg: AugmentConfig,
    rng: np.random.Generator,
    panorama: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augment and crop one annotated person for training.

    Args:
        image_rgb: ``(H, W, 3)`` uint8 RGB frame.
        bbox_xyxy: ``(4,)`` float box of the person in image pixels.
        keypoints: ``(K, 3)`` float array of x, y (image pixels) and visibility, ``K`` joints of
            ``spec``.
        spec: Keypoint layout; ``spec.flip_pairs`` swap under a flip.
        cfg: Augmentation strengths and probabilities.
        rng: Source of all random draws (``standard_normal()`` and ``random()``).
        panorama: Whether the frame is a 360-degree panorama. The box and the joints are then in
            the unwrapped coordinates of the person (they may lie left of 0 or right of ``W - 1``),
            and the crop reads the frame as a periodic strip (:func:`~.geometry.warp_crop`). A
            flip mirrors the whole strip, so the mirrored person lies at ``W - 1 - x`` in the
            mirrored strip; no random number is drawn differently.

    Returns:
        ``(crop, joints, visibility)``: the ``(256, 192, 3)`` uint8 crop, the ``(K, 3)`` float32
        joints (x, y in crop pixels, visibility; ``(0, 0)`` for joints that are not visible) and
        the ``(K,)`` float32 visibility, equal to ``joints[:, 2]``.
    """
    assert (
        image_rgb.dtype == np.uint8 and image_rgb.ndim == 3 and image_rgb.shape[2] == 3
    ), f"image_rgb: expected uint8 (H, W, 3), got {image_rgb.dtype} {image_rgb.shape}"
    num_joints = spec.num_joints
    assert np.issubdtype(keypoints.dtype, np.floating), (
        f"keypoints: expected float, got {keypoints.dtype}"
    )
    assert keypoints.shape == (num_joints, 3), (
        f"keypoints: expected shape ({num_joints}, 3) for spec {spec.name!r}, got {keypoints.shape}"
    )
    out_h, out_w = INPUT_SIZE
    width = image_rgb.shape[1]

    center, scale = box_to_center_scale(bbox_xyxy)
    sf = cfg.scale_factor
    scale = scale * np.float32(
        np.clip(rng.standard_normal() * sf + 1.0, 1.0 - sf, 1.0 + sf)
    )
    rf = cfg.rotation_factor
    rot = 0.0
    if rng.random() < cfg.rotation_prob:
        rot = float(np.clip(rng.standard_normal() * rf, -2.0 * rf, 2.0 * rf))

    joints = keypoints.astype(np.float32)
    if cfg.flip and rng.random() < cfg.flip_prob:
        image_rgb = np.ascontiguousarray(image_rgb[:, ::-1, :])
        joints = flip_joints(joints, width, spec.flip_pairs)
        center[0] = width - center[0] - 1

    trans = get_affine_transform(center, scale, rot, (out_w, out_h))
    crop = warp_crop(image_rgb, trans, panorama)
    assert crop.shape == (out_h, out_w, 3) and crop.dtype == np.uint8

    visible = joints[:, 2] > 0.0
    homogeneous = np.column_stack(
        [joints[:, :2].astype(np.float64), np.ones(num_joints)]
    )
    mapped = homogeneous @ trans.T
    joints[:, :2] = np.where(visible[:, None], mapped, 0.0).astype(np.float32)
    visibility = joints[:, 2].copy()
    return crop, joints, visibility
