"""Drawing of poses and person crops, shared by the videos and the pick figures."""

from __future__ import annotations

import cv2
import numpy as np

from ..data.geometry import INPUT_SIZE, get_affine_transform, warp_crop
from ..keypoints import KeypointSpec

# RGB colours of the limbs: left side, right side and the middle of the body
LEFT, RIGHT, CENTRE = (255, 138, 61), (61, 165, 255), (155, 225, 93)
SIDE_COLOURS = (LEFT, RIGHT, CENTRE)
PICK = (255, 176, 0)  # the boxes and numbers of the selected people
RECON = (232, 62, 140)  # the skeleton that the autoencoder reconstructs (WPU)
GAP = (230, 40, 40)  # the joints whose reconstruction differs most (WPU)
THC_JOINTS = ((255, 159, 10), (50, 200, 230), (140, 220, 60))  # the joints that changed most (THC)


def limb_sides(spec: KeypointSpec) -> list[tuple[int, int, int]]:
    """The limbs to draw as ``(joint a, joint b, side)``; side 0 is left, 1 right and 2 the middle.

    A limb that touches a redundant joint (one that is never annotated) is left out. The side comes
    from the joint names: a limb with a ``left`` joint is left, one with a ``right`` joint right.
    """
    skip = set(spec.redundant_joints)
    limbs = []
    for a, b in spec.skeleton:
        if a in skip or b in skip:
            continue
        names = spec.joint_names[a] + spec.joint_names[b]
        side = 0 if "left" in names else 1 if "right" in names else 2
        limbs.append((a, b, side))
    return limbs


def draw_pose(
    image: np.ndarray,
    spec: KeypointSpec,
    xy: np.ndarray,
    scale: float,
    colour: tuple[int, int, int] | None = None,
    width_x: int | None = None,
) -> None:
    """Draw the skeleton of one person into an RGB image, in place.

    Args:
        image: ``(H, W, 3)`` uint8 RGB image.
        spec: The keypoint layout.
        xy: ``(K, 2)`` joint positions in pixels.
        scale: Line width factor; 1 is the width for an image 480 pixels high.
        colour: One RGB colour for the whole skeleton; ``None`` colours the limbs by side.
        width_x: The width of a panorama; a limb that crosses its seam is not drawn.
    """
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    assert xy.shape == (spec.num_joints, 2), xy.shape
    skip = set(spec.redundant_joints)
    thickness = max(2, round(2.2 * scale))
    for a, b, side in limb_sides(spec):
        p, q = xy[a], xy[b]
        if width_x is not None and abs(p[0] - q[0]) > width_x / 2:
            continue
        cv2.line(
            image,
            tuple(np.round(p).astype(int).tolist()),
            tuple(np.round(q).astype(int).tolist()),
            colour or SIDE_COLOURS[side],
            thickness,
            cv2.LINE_AA,
        )
    for j in range(spec.num_joints):
        if j in skip:
            continue
        centre = tuple(np.round(xy[j]).astype(int).tolist())
        cv2.circle(image, centre, thickness + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(
            image, centre, max(1, thickness - 1), (255, 255, 255), -1, cv2.LINE_AA
        )


def person_crop(
    image_rgb: np.ndarray, bbox_crop: np.ndarray, panorama: bool
) -> np.ndarray:
    """The ``(256, 192, 3)`` crop that the estimator sees for the box ``bbox_crop``.

    The region of the box is warped the way the cache crops a person
    (:func:`~vatl4pose.data.geometry.warp_crop`), a panorama read as a periodic strip.
    """
    assert bbox_crop.shape == (4,), bbox_crop.shape
    box = bbox_crop.astype(np.float64)
    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
    scale = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float32)
    height, width = INPUT_SIZE
    trans = get_affine_transform(centre, scale, 0.0, (width, height))
    return warp_crop(image_rgb, trans, panorama)


def window(image: np.ndarray, x0: int, x1: int, panorama: bool) -> np.ndarray:
    """The columns ``x0`` to ``x1 - 1`` of an image; a panorama wraps around its seam."""
    assert x1 > x0
    width = image.shape[1]
    if panorama:
        return image[:, np.arange(x0, x1) % width]
    assert x0 >= 0 and x1 <= width, (x0, x1, width)
    return image[:, x0:x1]
