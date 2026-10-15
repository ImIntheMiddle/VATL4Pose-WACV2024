"""Whole-body autoencoder and the hybrid pose feature behind WPU.

The Whole-body Pose Unnaturalness (WPU) of a predicted pose is the reconstruction error of a
small autoencoder trained on natural poses. The autoencoder does not see raw coordinates. It sees
the *hybrid feature* of :func:`compute_hybrid`:

``[ (x_j - c_x) / h  for every joint j ,  (y_j - c_y) / h  for every joint j ,  angles ]``

where ``(c_x, c_y)`` is the center of the joints weighted by their confidence or visibility,
``h`` is the height of the person box, and each angle is the angle at the middle joint of a
triplet from :attr:`KeypointSpec.angle_triplets`. The vector has
:attr:`KeypointSpec.ae_input_dim` entries.

Three design points differ from the 2023 autoencoder, whose decoder ended in a sigmoid and saw
joints without an annotation at the image corner:

* The autoencoder works on *standardized* features. The mean and the standard deviation of every
  dimension, taken over the valid training entries, are the buffers ``feature_mean`` and
  ``feature_std`` of the model and part of its state dict (:meth:`WholeBodyAE.standardize`).
* The decoder output is linear.
* A joint that is not visible, and an angle that has an invisible joint among its three, is
  *invalid*. Its entry is 0 in the standardized input and takes no part in any loss or score
  (:func:`hybrid_valid_mask`, :func:`masked_mse`, :func:`wpu_scores`).

A checkpoint of the 2023 format has no ``feature_mean`` and ``feature_std``; the loaders refuse it.
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
# The normalization buffers; a checkpoint of the 2023 format has neither.
_NORMALIZATION_KEYS = ("feature_mean", "feature_std")


class WholeBodyAE(nn.Module):
    """Fully connected autoencoder on standardized hybrid features, with a linear output.

    The encoder is ``input_dim -> 24 -> 12 -> 7 -> z_dim`` with a ReLU after every layer except
    the last. The decoder mirrors it, ``z_dim -> 7 -> 12 -> 24 -> input_dim``, and has no output
    activation. The submodules are called ``encoder`` and ``decoder`` (``nn.Sequential``) as in the
    2023 checkpoints.

    The module reads and writes standardized features. The buffers ``feature_mean`` and
    ``feature_std`` (shape ``(input_dim,)``, initially 0 and 1) hold the statistics of the training
    features, and :meth:`standardize` applies them. They are not trained and travel with the state
    dict.

    Args:
        input_dim: Length of the hybrid feature vector (:attr:`KeypointSpec.ae_input_dim`).
        z_dim: Size of the latent code.
    """

    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def __init__(self, input_dim: int, z_dim: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if z_dim < 1:
            raise ValueError(f"z_dim must be positive, got {z_dim}")
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
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
        )

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store the feature statistics; ``std`` must be positive and both must be finite.

        Raises:
            ValueError: If a shape is not ``(input_dim,)`` or a value is not finite, or a
                standard deviation is not positive. The message names the first bad dimension.
        """
        for name, value in (("mean", mean), ("std", std)):
            if value.shape != (self.input_dim,) or not value.is_floating_point():
                raise ValueError(
                    f"{name}: expected a floating-point tensor of shape ({self.input_dim},), "
                    f"got {tuple(value.shape)} of dtype {value.dtype}"
                )
            bad = torch.nonzero(~torch.isfinite(value)).flatten()
            if bad.numel():
                raise ValueError(f"{name}: dimension {int(bad[0])} is not finite")
        bad = torch.nonzero(~(std > 0)).flatten()
        if bad.numel():
            raise ValueError(
                f"std: dimension {int(bad[0])} is {float(std[bad[0]])}; it must be positive"
            )
        with torch.no_grad():
            self.feature_mean.copy_(mean)
            self.feature_std.copy_(std)

    def standardize(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Standardize raw hybrid features with the stored statistics; invalid entries become 0.

        Args:
            features: Raw features of shape ``(N, input_dim)``, floating point.
            valid: Boolean mask of the same shape, see :func:`hybrid_valid_mask`.

        Returns:
            ``(features - feature_mean) / feature_std`` where ``valid``, 0 elsewhere.
        """
        return standardize_features(
            features, valid, self.feature_mean, self.feature_std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct the standardized features ``x``.

        Args:
            x: Floating-point tensor of shape ``(..., input_dim)``, the output of
                :meth:`standardize`.

        Returns:
            Tensor of the same shape; the values at invalid entries carry no meaning.
        """
        assert x.is_floating_point(), x.dtype
        assert x.shape[-1] == self.input_dim, (x.shape, self.input_dim)
        out = self.decoder(self.encoder(x))
        assert out.shape == x.shape, (out.shape, x.shape)
        return out


def standardize_features(
    features: torch.Tensor,
    valid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """``(features - mean) / std`` where ``valid`` and 0 elsewhere.

    Args:
        features: Floating-point tensor ``(N, D)``.
        valid: Boolean tensor ``(N, D)``.
        mean: Per-dimension mean, shape ``(D,)``.
        std: Per-dimension standard deviation, shape ``(D,)``; positive.

    Returns:
        Tensor of the shape and dtype of ``features``. An invalid entry may hold any value (even
        a non-finite one) in ``features``; it is 0 in the result.
    """
    assert features.is_floating_point(), features.dtype
    assert features.ndim == 2, features.shape
    assert valid.dtype == torch.bool and valid.shape == features.shape, (
        valid.dtype,
        valid.shape,
        features.shape,
    )
    assert mean.shape == std.shape == features.shape[1:], (mean.shape, std.shape)
    standardized = torch.where(
        valid, (features - mean) / std, torch.zeros_like(features)
    )
    assert standardized.shape == features.shape
    return standardized


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over the valid entries of all rows.

    Args:
        prediction: ``(N, D)`` floating-point reconstruction.
        target: ``(N, D)`` standardized features.
        valid: ``(N, D)`` boolean mask; an entry that is not valid does not enter the loss and
            passes no gradient.

    Raises:
        ValueError: If no entry is valid.
    """
    assert prediction.shape == target.shape == valid.shape, (
        prediction.shape,
        target.shape,
        valid.shape,
    )
    assert valid.dtype == torch.bool, valid.dtype
    count = int(valid.sum())
    if count == 0:
        raise ValueError("masked_mse: no valid entry; every entry is masked out")
    squared = torch.where(valid, (prediction - target) ** 2, torch.zeros_like(target))
    return squared.sum() / count


def _read_state(ckpt: pathlib.Path) -> dict[str, torch.Tensor]:
    """Load the state dict of ``ckpt`` on the CPU, accepting tensors only.

    Raises:
        ValueError: If ``ckpt`` is no whole-body autoencoder state dict, or one of the 2023
            format (without the normalization buffers), or the buffers do not fit the encoder.
    """
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
    missing = [key for key in _NORMALIZATION_KEYS if key not in state]
    if missing:
        raise ValueError(
            f"{ckpt} is a whole-body autoencoder of the 2023 format (no {', '.join(missing)}: "
            "sigmoid decoder on unscaled features). Use the retrained weights "
            "WholeBodyAE_zdim4_v1.pth or train one with vatl4pose-train train=ae_posetrack21 "
            "or train=ae_jrdb"
        )
    input_dim = int(state[_FIRST_WEIGHT_KEY].shape[1])
    for key in _NORMALIZATION_KEYS:
        if tuple(state[key].shape) != (input_dim,):
            raise ValueError(
                f"{ckpt}: {key} has shape {tuple(state[key].shape)}, expected ({input_dim},) "
                f"like the input of {_FIRST_WEIGHT_KEY}"
            )
    return state


def read_input_dim(ckpt: pathlib.Path) -> int:
    """Return the input dimension of a checkpoint, the width of ``encoder.0.weight``.

    Raises:
        ValueError: If ``ckpt`` is not a whole-body autoencoder state dict of the current format.
    """
    return int(_read_state(ckpt)[_FIRST_WEIGHT_KEY].shape[1])


def load_wholebody_ae(
    ckpt: pathlib.Path, spec: KeypointSpec, z_dim: int, device: torch.device
) -> WholeBodyAE:
    """Load a whole-body autoencoder checkpoint for ``spec``.

    Args:
        ckpt: State dict saved with ``torch.save(model.state_dict(), ...)``.
        spec: Keypoint spec; the checkpoint must have been trained on vectors of
            ``spec.ae_input_dim`` entries.
        z_dim: Latent size the checkpoint was trained with.
        device: Device of the returned model.

    Returns:
        The model in evaluation mode, with every parameter and both normalization buffers taken
        from the checkpoint.

    Raises:
        ValueError: If ``ckpt`` is of the 2023 format or no autoencoder state dict, or its input
            dimension differs from ``spec.ae_input_dim``. The message names the file.
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


def hybrid_valid_mask(keypoint_weights: np.ndarray, spec: KeypointSpec) -> np.ndarray:
    """Which entries of the hybrid feature hold a measured value.

    A joint is usable when its weight is positive (a visible label, a confident prediction) and
    the spec does not list it as redundant, that is never annotated. The x and the y offset of a
    usable joint are valid. An angle is valid when all three joints of its triplet are usable.

    Args:
        keypoint_weights: Weights of the joints, shape ``(N, spec.num_joints)``; the third column
            of the keypoints that :func:`compute_hybrid` takes.
        spec: Keypoint spec.

    Returns:
        Boolean array of shape ``(N, spec.ae_input_dim)``, in the order of the feature:
        x offsets, y offsets, angles.

    Raises:
        ValueError: If ``keypoint_weights`` has the wrong shape.
    """
    weight = np.asarray(keypoint_weights)
    if weight.ndim != 2 or weight.shape[1] != spec.num_joints:
        raise ValueError(
            f"keypoint_weights: expected shape (N, {spec.num_joints}) for spec "
            f"{spec.name!r}, got {weight.shape}"
        )
    usable = weight > 0
    if spec.redundant_joints:
        usable[:, list(spec.redundant_joints)] = False
    triplets = np.asarray(spec.angle_triplets, dtype=np.int64).reshape(-1, 3)
    angle_ok = usable[:, triplets].all(axis=2)
    valid = np.concatenate([usable, usable, angle_ok], axis=1)
    assert valid.shape == (weight.shape[0], spec.ae_input_dim), valid.shape
    assert valid.dtype == np.bool_
    return valid


def feature_dim_name(dim: int, spec: KeypointSpec) -> str:
    """Name a dimension of the hybrid feature for a message, for example ``x offset of nose``."""
    k = spec.num_joints
    if not 0 <= dim < spec.ae_input_dim:
        raise ValueError(f"dimension {dim} is outside [0, {spec.ae_input_dim})")
    if dim < k:
        return f"x offset of {spec.joint_names[dim]}"
    if dim < 2 * k:
        return f"y offset of {spec.joint_names[dim - k]}"
    a, b, c = spec.angle_triplets[dim - 2 * k]
    names = (spec.joint_names[a], spec.joint_names[b], spec.joint_names[c])
    return f"angle {dim - 2 * k} at {names[1]} ({names[0]}, {names[1]}, {names[2]})"


def compute_hybrid(
    bbox_xywh: np.ndarray, keypoints: np.ndarray, spec: KeypointSpec
) -> np.ndarray:
    """Hybrid feature of ``N`` people (see the module docstring).

    The center ``c`` of a person is ``sum_j w_j p_j / sum_j w_j``. Every angle is
    ``arctan(|(m1 - m2) / (1 + m1 * m2 + eps)|)`` with the slopes ``m1 = (y_b - y_a) /
    (x_b - x_a + eps)`` and ``m2 = (y_c - y_b) / (x_c - x_b + eps)`` of the triplet ``(a, b, c)``
    and ``eps = 1e-6``, so it lies in ``[0, pi / 2)``.

    Args:
        bbox_xywh: Person boxes ``(x, y, w, h)``, shape ``(N, 4)``. Only the height ``h`` is used.
        keypoints: ``(x, y, weight)`` of every joint, shape ``(N, spec.num_joints, 3)``. The
            weight is the confidence of a predicted joint or the visibility of an annotated one.
        spec: Keypoint spec that defines the angle triplets.

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
            f"row {row}: joint weights must have a positive sum, got {weight_sum[row]}"
        )

    center_x = (x * weight).sum(axis=1) / weight_sum
    center_y = (y * weight).sum(axis=1) / weight_sum
    x_offset = (x - center_x[:, None]) / height[:, None]
    y_offset = (y - center_y[:, None]) / height[:, None]

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
    ae: WholeBodyAE,
    hybrid: np.ndarray,
    valid: np.ndarray,
    keep_index: np.ndarray | None,
) -> np.ndarray:
    """Whole-body Pose Unnaturalness (WPU): masked reconstruction error of each hybrid feature.

    The features are standardized with the statistics of the autoencoder (invalid entries become
    0). The score of a row is the mean squared difference between the standardized input and the
    reconstruction over the kept columns that are valid in that row.

    Args:
        ae: Autoencoder that reconstructs the feature vectors.
        hybrid: Raw features of shape ``(N, ae.input_dim)`` from :func:`compute_hybrid`; they are
            converted to ``float32`` for the autoencoder.
        valid: Boolean mask of the same shape from :func:`hybrid_valid_mask`.
        keep_index: Columns that enter the error, for example ``spec.wpu_keep_index()``, which
            leaves out the offsets of the redundant joints. ``None`` keeps every column.

    Returns:
        ``float64`` array of shape ``(N,)``. The autoencoder is not modified.

    Raises:
        ValueError: If ``hybrid``, ``valid`` or ``keep_index`` has an unexpected shape, dtype or
            value, or a row has no valid kept column (the message names the row).
    """
    if hybrid.ndim != 2 or hybrid.shape[1] != ae.input_dim:
        raise ValueError(
            f"hybrid: expected shape (N, {ae.input_dim}), got {hybrid.shape}"
        )
    if valid.shape != hybrid.shape or valid.dtype != np.bool_:
        raise ValueError(
            f"valid: expected a bool array of shape {hybrid.shape}, got {valid.dtype} "
            f"of shape {valid.shape}"
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
    counted = valid[:, keep]
    empty = np.flatnonzero(counted.sum(axis=1) == 0)
    if empty.size:
        raise ValueError(
            f"row {int(empty[0])}: no valid entry among the kept columns, so the "
            "reconstruction error is undefined"
        )

    device = next(ae.parameters()).device
    x = ae.standardize(
        torch.tensor(np.asarray(hybrid, dtype=np.float32), device=device),
        torch.tensor(valid, device=device),
    )
    columns = torch.as_tensor(keep, dtype=torch.long, device=device)
    weight = torch.as_tensor(counted, dtype=torch.float32, device=device)
    with torch.no_grad():
        squared_error = ((ae(x) - x)[:, columns]) ** 2
        scores = (squared_error * weight).sum(dim=1) / weight.sum(dim=1)
    out = scores.cpu().numpy().astype(np.float64)
    assert out.shape == (hybrid.shape[0],), out.shape
    return out
