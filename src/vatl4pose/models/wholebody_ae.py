"""Whole-body autoencoder and the hybrid pose feature behind the WPU uncertainty.

The whole-body pose uncertainty (WPU) of a predicted pose is the reconstruction error of a
small autoencoder trained on natural poses. The autoencoder does not see raw coordinates. It sees
the *hybrid feature* of :func:`compute_hybrid`:

``[ (x_j - c_x) / h  for every joint j ,  (y_j - c_y) / h  for every joint j ,  angles ]``

where ``(c_x, c_y)`` is the centre of the joints weighted by their confidence or visibility,
``h`` is the height of the person box, and each angle is the angle at the middle joint of a
triplet from :attr:`KeypointSpec.angle_triplets`. The vector has
:attr:`KeypointSpec.ae_input_dim` entries.

The layer names of :class:`WholeBodyAE` match the released checkpoints, so those load with
``strict=True``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch import nn

from vatl4pose.keypoints import KeypointSpec

# Added to the slope denominators so that vertical segments stay finite.
_SLOPE_EPS = 1e-6

# The first layer of the encoder fixes the length of the input vector.
_FIRST_WEIGHT_KEY = "encoder.0.weight"


class WholeBodyAE(nn.Module):
    """Fully connected autoencoder with a sigmoid output.

    The encoder is ``input_dim -> 24 -> 12 -> 7 -> z_dim`` with a ReLU after every layer except
    the last. The decoder mirrors it, ``z_dim -> 7 -> 12 -> 24 -> input_dim``, and ends with a
    sigmoid. The submodules are called ``encoder`` and ``decoder`` (``nn.Sequential``) as in the
    released checkpoints.

    Args:
        input_dim: Length of the hybrid feature vector (:attr:`KeypointSpec.ae_input_dim`).
        z_dim: Size of the latent code.
    """

    def __init__(self, input_dim: int, z_dim: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if z_dim < 1:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 7),
            nn.ReLU(),
            nn.Linear(7, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 7),
            nn.ReLU(),
            nn.Linear(7, 12),
            nn.ReLU(),
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x``.

        Args:
            x: Floating-point tensor of shape ``(..., input_dim)``.

        Returns:
            Tensor of the same shape with values in ``[0, 1]``.
        """
        assert x.is_floating_point(), x.dtype
        assert x.shape[-1] == self.input_dim, (x.shape, self.input_dim)
        out = self.decoder(self.encoder(x))
        assert out.shape == x.shape, (out.shape, x.shape)
        return out


def _read_state(ckpt: pathlib.Path) -> dict[str, torch.Tensor]:
    """Load the state dict of ``ckpt`` on the CPU, accepting tensors only."""
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or _FIRST_WEIGHT_KEY not in state
        or state[_FIRST_WEIGHT_KEY].ndim != 2
    ):
        raise ValueError(
            f"{ckpt} is not a whole-body autoencoder state dict: "
            f"no 2-D entry {_FIRST_WEIGHT_KEY!r}"
        )
    return state


def read_input_dim(ckpt: pathlib.Path) -> int:
    """Return the input dimension of a checkpoint, the width of ``encoder.0.weight``.

    Raises:
        ValueError: If ``ckpt`` is not a whole-body autoencoder state dict.
    """
    return int(_read_state(ckpt)[_FIRST_WEIGHT_KEY].shape[1])


def load_wholebody_ae(
    ckpt: pathlib.Path, spec: KeypointSpec, z_dim: int, device: torch.device
) -> WholeBodyAE:
    """Load a whole-body autoencoder checkpoint for ``spec``.

    Args:
        ckpt: State dict saved with ``torch.save(model.state_dict(), ...)``.
        spec: Keypoint specification; the checkpoint must have been trained on vectors of
            ``spec.ae_input_dim`` entries.
        z_dim: Latent size the checkpoint was trained with.
        device: Device of the returned model.

    Returns:
        The model in evaluation mode, with every parameter taken from the checkpoint.

    Raises:
        ValueError: If the input dimension of the checkpoint differs from ``spec.ae_input_dim``.
        RuntimeError: If the checkpoint has missing, unexpected or mis-shaped entries
            (for example a different ``z_dim``).
    """
    state = _read_state(ckpt)
    input_dim = int(state[_FIRST_WEIGHT_KEY].shape[1])
    if input_dim != spec.ae_input_dim:
        raise ValueError(
            f"expected input dim {spec.ae_input_dim}, checkpoint {ckpt} has {input_dim}"
        )
    ae = WholeBodyAE(input_dim, z_dim)
    ae.load_state_dict(state, strict=True)
    return ae.to(device).eval()


def compute_hybrid(
    bbox_xywh: np.ndarray, keypoints: np.ndarray, spec: KeypointSpec
) -> np.ndarray:
    """Hybrid feature of ``N`` people (see the module docstring).

    The centre ``c`` of a person is ``sum_j w_j p_j / sum_j w_j``. Every angle is
    ``arctan(|(m1 - m2) / (1 + m1 * m2 + eps)|)`` with the slopes ``m1 = (y_b - y_a) /
    (x_b - x_a + eps)`` and ``m2 = (y_c - y_b) / (x_c - x_b + eps)`` of the triplet ``(a, b, c)``
    and ``eps = 1e-6``, so it lies in ``[0, pi / 2)``.

    Args:
        bbox_xywh: Person boxes ``(x, y, w, h)``, shape ``(N, 4)``. Only the height ``h`` is used.
        keypoints: ``(x, y, weight)`` of every joint, shape ``(N, spec.num_joints, 3)``. The
            weight is the confidence of a predicted joint or the visibility of an annotated one.
        spec: Joint layout that defines the angle triplets.

    Returns:
        Array of shape ``(N, spec.ae_input_dim)`` and dtype ``float64``.

    Raises:
        ValueError: If a shape does not fit, or if a row has a box height that is not positive
            or weights whose sum is not positive. The message names the row.
    """
    box = np.asarray(bbox_xywh, dtype=np.float64)
    kpt = np.asarray(keypoints, dtype=np.float64)
    if box.ndim != 2 or box.shape[1] != 4:
        raise ValueError(f"bbox_xywh: expected shape (N, 4), got {box.shape}")
    n = box.shape[0]
    if kpt.shape != (n, spec.num_joints, 3):
        raise ValueError(
            f"keypoints: expected shape ({n}, {spec.num_joints}, 3) to match bbox_xywh "
            f"({n} rows) and spec {spec.name!r}, got {kpt.shape}"
        )

    height = box[:, 3]
    x, y, weight = kpt[..., 0], kpt[..., 1], kpt[..., 2]
    weight_sum = weight.sum(axis=1)
    # "not (v > 0)" also rejects NaN.
    bad_height = np.flatnonzero(~(height > 0))
    if bad_height.size:
        row = int(bad_height[0])
        raise ValueError(f"row {row}: box height must be positive, got {height[row]}")
    bad_weight = np.flatnonzero(~(weight_sum > 0))
    if bad_weight.size:
        row = int(bad_weight[0])
        raise ValueError(
            f"row {row}: keypoint weights must have a positive sum, got {weight_sum[row]}"
        )

    centre_x = (x * weight).sum(axis=1) / weight_sum
    centre_y = (y * weight).sum(axis=1) / weight_sum
    x_offset = (x - centre_x[:, None]) / height[:, None]
    y_offset = (y - centre_y[:, None]) / height[:, None]

    triplets = np.asarray(spec.angle_triplets, dtype=np.int64).reshape(-1, 3)
    a, b, c = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    slope_ab = (y[:, b] - y[:, a]) / (x[:, b] - x[:, a] + _SLOPE_EPS)
    slope_bc = (y[:, c] - y[:, b]) / (x[:, c] - x[:, b] + _SLOPE_EPS)
    angle = np.arctan(
        np.abs((slope_ab - slope_bc) / (1 + slope_ab * slope_bc + _SLOPE_EPS))
    )

    feature = np.concatenate([x_offset, y_offset, angle], axis=1)
    assert feature.shape == (n, spec.ae_input_dim), feature.shape
    assert feature.dtype == np.float64
    return feature


def wpu_scores(
    ae: WholeBodyAE, hybrid: np.ndarray, keep_index: np.ndarray | None
) -> np.ndarray:
    """Whole-body pose uncertainty: reconstruction error of each hybrid feature.

    Args:
        ae: Autoencoder that reconstructs the feature vectors.
        hybrid: Features of shape ``(N, ae.input_dim)`` from :func:`compute_hybrid`; they are
            converted to ``float32`` for the network.
        keep_index: Columns that enter the error, for example ``spec.wpu_keep_index()``, which
            leaves out the offsets of the redundant joints. ``None`` keeps every column.

    Returns:
        ``float64`` array of shape ``(N,)``, the mean squared difference between the input and the
        reconstruction over the kept columns of each row. The autoencoder is not modified.

    Raises:
        ValueError: If ``hybrid`` or ``keep_index`` has an unexpected shape, dtype or value.
    """
    if hybrid.ndim != 2 or hybrid.shape[1] != ae.input_dim:
        raise ValueError(
            f"hybrid: expected shape (N, {ae.input_dim}), got {hybrid.shape}"
        )
    if keep_index is None:
        keep = np.arange(ae.input_dim, dtype=np.int64)
    else:
        keep = np.asarray(keep_index)
        if keep.ndim != 1 or keep.size == 0 or keep.dtype.kind not in "iu":
            raise ValueError(
                "keep_index: expected a non-empty 1-D integer array, "
                f"got dtype {keep.dtype} and shape {keep.shape}"
            )
        if keep.min() < 0 or keep.max() >= ae.input_dim:
            raise ValueError(
                f"keep_index: entries must lie in [0, {ae.input_dim}), "
                f"got range [{keep.min()}, {keep.max()}]"
            )
        if np.unique(keep).size != keep.size:
            raise ValueError("keep_index: entries must be unique")

    device = next(ae.parameters()).device
    x = torch.tensor(np.asarray(hybrid, dtype=np.float32), device=device)
    columns = torch.as_tensor(keep, dtype=torch.long, device=device)
    with torch.no_grad():
        squared_error = (ae(x) - x)[:, columns] ** 2
        scores = squared_error.mean(dim=1)
    out = scores.cpu().numpy().astype(np.float64)
    assert out.shape == (hybrid.shape[0],), out.shape
    return out
