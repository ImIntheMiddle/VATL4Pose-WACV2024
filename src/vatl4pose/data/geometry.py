"""Person crops, heatmap decoding and training targets.

Every pose model of the package reads a ``256 x 192`` (height x width) crop of one person and
predicts one ``64 x 48`` heatmap per joint. This module holds the geometry around that interface:

* box to crop-centre and scale (:func:`box_to_center_scale`, :func:`center_scale_to_box`),
* the affine map between image and crop (:func:`get_affine_transform`),
* cropping and normalising (:func:`crop_person`, :func:`to_input_tensor`),
* decoding heatmaps into image coordinates (:func:`heatmaps_to_coords`, :func:`coords_in_box`),
* Gaussian training targets (:func:`make_targets`).

The numbers follow the 2023 inference code of the paper exactly, including its float32 steps:

* The crop centre and scale are float32. Both come from float64 arithmetic on the box.
* The three reference points of the affine map are rounded to float32 before the 2x3 matrix is
  solved. :func:`heatmaps_to_coords` repeats this rounding, so decoded coordinates of large
  images (JRDB-Pose frames are 3760 pixels wide) equal the legacy ones.
* Both axes of the heatmap-to-image map use the box *width* divided by the heatmap width. A box
  with the crop aspect ratio makes this equal to height divided by heatmap height.
* A crop is not clipped to the image: parts outside the frame are black.

Conventions of this module:

* Boxes are ``(xmin, ymin, xmax, ymax)`` in image pixels.
* ``output_size`` arguments are ``(width, height)``, the order of ``cv2.warpAffine``. The
  constants :data:`INPUT_SIZE` and :data:`HEATMAP_SIZE` are ``(height, width)``.
* Joint arrays for :func:`make_targets` have shape ``(K, 3)`` with columns x, y, visibility.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

INPUT_SIZE: tuple[int, int] = (256, 192)
"""Crop size ``(height, width)`` in pixels."""

HEATMAP_SIZE: tuple[int, int] = (64, 48)
"""Heatmap size ``(height, width)`` in cells."""

PIXEL_MEAN_RGB: tuple[float, float, float] = (0.406, 0.457, 0.480)
"""Per-channel mean (RGB, scale 0 to 1) subtracted from a crop; there is no division by a std."""

_CROP_ASPECT: float = INPUT_SIZE[1] / INPUT_SIZE[0]


def box_to_center_scale(
    bbox_xyxy: np.ndarray, aspect: float = 192 / 256, scale_mult: float = 1.25
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a box into the centre and the size of the region that is cropped.

    The box grows on one side until its width over height equals ``aspect``, then both sides are
    multiplied by ``scale_mult``. The width of the box is ``xmax - xmin`` (no ``+ 1``). A box with
    zero width and positive height is valid: it becomes ``aspect`` times the height wide.

    Args:
        bbox_xyxy: ``(4,)`` float array ``xmin, ymin, xmax, ymax``.
        aspect: Target width over height.
        scale_mult: Margin factor around the aspect-corrected box.

    Returns:
        ``(center, scale)``, both ``(2,)`` float32 arrays holding ``(x, y)`` and ``(width, height)``.

    Raises:
        ValueError: If the box is inverted, has a NaN, or has no extent at all.
    """
    box = np.asarray(bbox_xyxy)
    assert box.shape == (4,), f"bbox_xyxy: expected shape (4,), got {box.shape}"
    assert np.issubdtype(box.dtype, np.floating), (
        f"bbox_xyxy: expected float, got {box.dtype}"
    )
    x0, y0, x1, y1 = (float(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if not (w >= 0.0 and h >= 0.0):
        raise ValueError(f"inverted box (or NaN) {box.tolist()}: width {w}, height {h}")
    if w == 0.0 and h == 0.0:
        raise ValueError(f"empty box {box.tolist()}: it has neither width nor height")
    center = np.array([x0 + w * 0.5, y0 + h * 0.5], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w, h], dtype=np.float32) * np.float32(scale_mult)
    return center, scale


def center_scale_to_box(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return the box ``(xmin, ymin, xmax, ymax)`` of a crop centre and size, in float64."""
    c = np.asarray(center, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    assert c.shape == (2,) and s.shape == (2,), (
        f"expected (2,) centre and scale, got {c.shape}, {s.shape}"
    )
    xmin = c[0] - s[0] * 0.5
    ymin = c[1] - s[1] * 0.5
    return np.array([xmin, ymin, xmin + s[0], ymin + s[1]], dtype=np.float64)


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Point ``b`` plus the vector ``a - b`` turned by 90 degrees (float32 arithmetic)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: tuple[int, int],
    inv: bool = False,
) -> np.ndarray:
    """Affine map between the crop region and the output crop, as a ``(2, 3)`` float64 matrix.

    Three point pairs fix the map. The crop centre goes to the output centre. The point half a
    box width above the centre (turned by ``rot`` degrees around the centre) goes to the point
    half an output width above the output centre. The third pair completes a right angle at the
    first point. The three source points are rounded to float32 first, like the legacy code.

    Args:
        center: ``(2,)`` crop centre ``(x, y)`` in image pixels.
        scale: ``(2,)`` crop size ``(width, height)`` in image pixels; only the width is used.
        rot: Rotation in degrees.
        output_size: ``(width, height)`` of the output crop.
        inv: Return the map from the output crop back to the image.

    Returns:
        ``(2, 3)`` matrix ``[A | t]`` with ``x_out = A @ x_in + t`` (or the inverse map).
    """
    c = np.asarray(center)
    s = np.asarray(scale)
    assert c.shape == (2,), f"center: expected shape (2,), got {c.shape}"
    assert s.shape == (2,), f"scale: expected shape (2,), got {s.shape}"
    dst_w, dst_h = output_size

    rot_rad = np.pi * rot / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    up = float(s[0]) * -0.5
    src_dir = np.array([0.0 * cs - up * sn, 0.0 * sn + up * cs])
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = c.astype(np.float64)
    src[1] = c.astype(np.float64) + src_dir
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2] = _third_point(src[0], src[1])
    dst[2] = _third_point(dst[0], dst[1])

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def crop_person(
    image_rgb: np.ndarray, bbox_xyxy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Crop one person from an RGB frame.

    Args:
        image_rgb: ``(H, W, 3)`` uint8 RGB image.
        bbox_xyxy: ``(4,)`` float box of the person in image pixels.

    Returns:
        ``(crop, bbox_crop)``: the ``(256, 192, 3)`` uint8 crop (bilinear, black outside the
        frame) and the ``(4,)`` float32 box ``xmin, ymin, xmax, ymax`` of the cropped region.
    """
    assert image_rgb.dtype == np.uint8, (
        f"image_rgb: expected uint8, got {image_rgb.dtype}"
    )
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3, (
        f"image_rgb: expected shape (H, W, 3), got {image_rgb.shape}"
    )
    center, scale = box_to_center_scale(bbox_xyxy, aspect=_CROP_ASPECT)
    out_h, out_w = INPUT_SIZE
    trans = get_affine_transform(center, scale, 0.0, (out_w, out_h))
    crop = cv2.warpAffine(image_rgb, trans, (out_w, out_h), flags=cv2.INTER_LINEAR)
    assert crop.shape == (out_h, out_w, 3) and crop.dtype == np.uint8
    bbox_crop = center_scale_to_box(center, scale).astype(np.float32)
    return crop, bbox_crop


def to_input_tensor(crops_uint8: torch.Tensor) -> torch.Tensor:
    """Convert uint8 crops ``(B, 256, 192, 3)`` to model input ``(B, 3, 256, 192)`` float32.

    The values are divided by 255 and the per-channel mean :data:`PIXEL_MEAN_RGB` is subtracted.
    """
    assert crops_uint8.dtype == torch.uint8, f"expected uint8, got {crops_uint8.dtype}"
    assert crops_uint8.ndim == 4 and tuple(crops_uint8.shape[1:]) == (*INPUT_SIZE, 3), (
        f"expected shape (B, {INPUT_SIZE[0]}, {INPUT_SIZE[1]}, 3), got {tuple(crops_uint8.shape)}"
    )
    x = crops_uint8.permute(0, 3, 1, 2).to(torch.float32) / 255.0
    mean = torch.tensor(PIXEL_MEAN_RGB, dtype=torch.float32, device=x.device).view(
        1, 3, 1, 1
    )
    out = (x - mean).contiguous()
    assert out.dtype == torch.float32
    return out


@torch.no_grad()
def heatmaps_to_coords(
    heatmaps: torch.Tensor, bbox_crop: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode heatmaps into image coordinates, for a whole batch at once.

    Per joint: the arg-max cell (first one on ties) gives the position and its value gives the
    score. A joint with a score of 0 or less is placed on cell ``(0, 0)``. The position moves by
    a quarter cell toward the larger neighbour along each axis when the cell is at least two cells
    away from every border. The cell position is then mapped into ``bbox_crop``.

    Args:
        heatmaps: ``(B, K, 64, 48)`` float32.
        bbox_crop: ``(B, 4)`` float32 or float64 boxes ``xmin, ymin, xmax, ymax`` of the crops.

    Returns:
        ``(coords, maxvals)``: ``(B, K, 2)`` float32 image coordinates ``(x, y)`` and ``(B, K)``
        float32 scores.
    """
    assert heatmaps.ndim == 4 and tuple(heatmaps.shape[2:]) == HEATMAP_SIZE, (
        f"heatmaps: expected shape (B, K, {HEATMAP_SIZE[0]}, {HEATMAP_SIZE[1]}), "
        f"got {tuple(heatmaps.shape)}"
    )
    assert heatmaps.dtype == torch.float32, (
        f"heatmaps: expected float32, got {heatmaps.dtype}"
    )
    batch, num_joints, hm_h, hm_w = heatmaps.shape
    assert tuple(bbox_crop.shape) == (batch, 4), (
        f"bbox_crop: expected shape ({batch}, 4), got {tuple(bbox_crop.shape)}"
    )
    assert bbox_crop.dtype in (torch.float32, torch.float64), (
        f"bbox_crop: expected float32 or float64, got {bbox_crop.dtype}"
    )

    flat = heatmaps.reshape(batch, num_joints, hm_h * hm_w)
    index = flat.argmax(dim=-1)
    maxvals = flat.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    positive = maxvals > 0.0
    col = torch.where(positive, index % hm_w, torch.zeros_like(index))
    row = torch.where(positive, index // hm_w, torch.zeros_like(index))

    def at(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        cell = r.clamp(0, hm_h - 1) * hm_w + c.clamp(0, hm_w - 1)
        return flat.gather(-1, cell.unsqueeze(-1)).squeeze(-1)

    inside = (col > 1) & (col < hm_w - 1) & (row > 1) & (row < hm_h - 1)
    shift_x = torch.sign(at(row, col + 1) - at(row, col - 1)) * 0.25
    shift_y = torch.sign(at(row + 1, col) - at(row - 1, col)) * 0.25
    zero = torch.zeros_like(shift_x)
    cell_xy = torch.stack(
        [
            col.to(torch.float32) + torch.where(inside, shift_x, zero),
            row.to(torch.float32) + torch.where(inside, shift_y, zero),
        ],
        dim=-1,
    )

    coords = _cells_to_image(cell_xy, bbox_crop, hm_w, hm_h)
    assert coords.shape == (batch, num_joints, 2) and coords.dtype == torch.float32
    return coords, maxvals


def coords_in_box(heatmaps: torch.Tensor, bbox_crop: torch.Tensor) -> torch.Tensor:
    """Decode heatmaps against another box; the coordinates of :func:`heatmaps_to_coords`.

    Args:
        heatmaps: ``(B, K, 64, 48)`` float32.
        bbox_crop: ``(B, 4)`` float32 or float64 boxes to map the heatmap cells into.

    Returns:
        ``(B, K, 2)`` float32 image coordinates.
    """
    return heatmaps_to_coords(heatmaps, bbox_crop)[0]


def _cells_to_image(
    cell_xy: torch.Tensor, bbox_crop: torch.Tensor, hm_w: int, hm_h: int
) -> torch.Tensor:
    """Map heatmap positions ``(B, K, 2)`` into the boxes ``(B, 4)``; float32 result.

    Closed form of ``get_affine_transform(center, (w, h), 0, (hm_w, hm_h), inv=True)`` applied to
    each position. The reference points are rounded to float32 exactly as in that function, so
    the rounding of the legacy matrix is kept.
    """
    box = bbox_crop.to(torch.float64)
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]
    cx = box[:, 0] + w * 0.5
    cy = box[:, 1] + h * 0.5
    s0 = torch.stack([cx, cy], dim=-1).to(torch.float32)
    s1 = torch.stack([cx, cy + w * -0.5], dim=-1).to(torch.float32)
    direct = s0 - s1
    s2 = s1 + torch.stack([-direct[:, 1], direct[:, 0]], dim=-1)
    s0, s1, s2 = (p.to(torch.float64) for p in (s0, s1, s2))
    # The heatmap points are d0 = (q, hm_h / 2), d1 = d0 - (0, q), d2 = d1 - (q, 0) with
    # q = hm_w / 2, which fixes the linear part and the shift of the inverse map.
    q = hm_w / 2.0
    col_x = (s1 - s2) / q  # image change per heatmap column
    col_y = (s0 - s1) / q  # image change per heatmap row
    shift = s0 - col_x * q - col_y * (hm_h / 2.0)
    out = (
        col_x.unsqueeze(1) * cell_xy[..., 0:1].to(torch.float64)
        + col_y.unsqueeze(1) * cell_xy[..., 1:2].to(torch.float64)
        + shift.unsqueeze(1)
    )
    return out.to(torch.float32)


def make_targets(
    joints: np.ndarray, spec_num_joints: int, sigma: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian heatmap targets and loss weights of one crop.

    A joint at crop position ``(x, y)`` gets the cell ``int(x / 4 + 0.5), int(y / 4 + 0.5)``
    (``int`` truncates toward zero) as the peak, with value 1. The window reaches
    ``int(3 * sigma)`` cells to each side. A joint whose window lies fully outside the map, or
    that is not visible, gets weight 0 and an all-zero heatmap.

    Args:
        joints: ``(K, 3)`` float array with columns x, y (crop pixels) and visibility (0 or 1).
        spec_num_joints: ``K``, the number of joints of the keypoint spec.
        sigma: Gaussian standard deviation in heatmap cells.

    Returns:
        ``(target, weight)``: ``(K, 64, 48)`` and ``(K, 1, 1)`` float32 arrays.
    """
    assert joints.shape == (spec_num_joints, 3), (
        f"joints: expected shape ({spec_num_joints}, 3), got {joints.shape}"
    )
    assert np.issubdtype(joints.dtype, np.floating), (
        f"joints: expected float, got {joints.dtype}"
    )
    hm_h, hm_w = HEATMAP_SIZE
    stride_x = INPUT_SIZE[1] / hm_w
    stride_y = INPUT_SIZE[0] / hm_h
    radius = int(3 * sigma)
    size = 2 * radius + 1
    axis = np.arange(0, size, 1, dtype=np.float32)
    gauss = np.exp(
        -((axis - radius) ** 2 + (axis[:, None] - radius) ** 2) / (2 * sigma**2)
    )

    weight = joints[:, 2].astype(np.float32).copy()
    target = np.zeros((spec_num_joints, hm_h, hm_w), dtype=np.float32)
    for k in range(spec_num_joints):
        mu_x = int(float(joints[k, 0]) / stride_x + 0.5)
        mu_y = int(float(joints[k, 1]) / stride_y + 0.5)
        left, top = mu_x - radius, mu_y - radius
        right, bottom = mu_x + radius + 1, mu_y + radius + 1
        if left >= hm_w or top >= hm_h or right < 0 or bottom < 0:
            weight[k] = 0.0
            continue
        if weight[k] > 0.5:
            gx0, gx1 = max(0, -left), min(right, hm_w) - left
            gy0, gy1 = max(0, -top), min(bottom, hm_h) - top
            x0, x1 = max(0, left), min(right, hm_w)
            y0, y1 = max(0, top), min(bottom, hm_h)
            target[k, y0:y1, x0:x1] = gauss[gy0:gy1, gx0:gx1]
    return target, weight.reshape(spec_num_joints, 1, 1)
