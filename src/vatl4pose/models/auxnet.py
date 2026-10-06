"""Auxiliary network of the VL4Pose baseline (Shukla et al., BMVC 2022).

For every skeleton link ``(u, v)`` the network predicts a Gaussian over the length of the link,
measured as the Euclidean distance between the heatmap peaks of joints ``u`` and ``v`` (in
heatmap cells). The network reads the backbone features of the pose estimator and outputs a mean
and a log-variance per link. It is trained with the Gaussian negative log-likelihood of the
ground-truth link lengths on labeled samples; the pose estimator stays untouched, so the features
that reach :func:`train_auxnet` are detached (stop-gradient). The query score is built from the
same negative log-likelihood, see :mod:`vatl4pose.strategies.baselines.vl4pose`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

LOG_2PI = math.log(2.0 * math.pi)


class AuxNet(nn.Module):
    """Global average pooling followed by a ReLU MLP that outputs a Gaussian per link.

    Args:
        in_channels: Channels ``C`` of the backbone features (2048 for the ResNet-50 backbone).
        num_links: Number of skeleton links ``L``.
        hidden: Widths of the hidden layers. A ReLU follows each of them and none follows the
            output layer, which has ``2 * L`` units.
    """

    def __init__(
        self,
        in_channels: int,
        num_links: int,
        hidden: tuple[int, ...] = (128, 64, 32, 16),
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(
                f"AuxNet: in_channels must be at least 1, got {in_channels}"
            )
        if num_links < 1:
            raise ValueError(f"AuxNet: num_links must be at least 1, got {num_links}")
        if any(width < 1 for width in hidden):
            raise ValueError(f"AuxNet: hidden widths must be at least 1, got {hidden}")
        self.in_channels = in_channels
        self.num_links = num_links
        widths = [in_channels, *hidden]
        layers: list[nn.Module] = []
        for width_in, width_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(width_in, width_out), nn.ReLU()]
        layers.append(nn.Linear(widths[-1], 2 * num_links))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the Gaussians of the link lengths.

        Args:
            features: ``(B, C, H, W)`` backbone features, e.g. ``(B, 2048, 8, 6)``.

        Returns:
            ``(mean, log_var)``, both ``(B, L)``: the mean and the log-variance of the length of
            every link.
        """
        assert (
            features.ndim == 4
            and features.shape[1] == self.in_channels
            and features.dtype.is_floating_point
        ), (
            f"features: expected a float tensor of shape (B, {self.in_channels}, H, W), "
            f"got {features.dtype} {tuple(features.shape)}"
        )
        out = self.mlp(features.mean(dim=(2, 3)))
        out = out.reshape(features.shape[0], self.num_links, 2)
        return out[..., 0], out[..., 1]


def link_lengths(
    coords_hm: torch.Tensor, links: Sequence[tuple[int, int]]
) -> torch.Tensor:
    """Euclidean length of every link.

    Args:
        coords_hm: ``(B, K, 2)`` joint positions in heatmap cells.
        links: ``L`` pairs ``(u, v)`` of joint indices.

    Returns:
        ``(B, L)`` lengths in the dtype of ``coords_hm``.

    Raises:
        ValueError: If ``links`` is empty or names a joint outside ``[0, K)``.
    """
    assert (
        coords_hm.ndim == 3
        and coords_hm.shape[2] == 2
        and coords_hm.dtype.is_floating_point
    ), (
        f"coords_hm: expected a float tensor of shape (B, K, 2), got {coords_hm.dtype} "
        f"{tuple(coords_hm.shape)}"
    )
    if len(links) == 0:
        raise ValueError("links: at least one link is needed")
    joints = torch.as_tensor(links, dtype=torch.long, device=coords_hm.device)
    num_joints = coords_hm.shape[1]
    if joints.ndim != 2 or joints.shape[1] != 2:
        raise ValueError(f"links: expected pairs of joint indices, got {links!r}")
    if int(joints.min()) < 0 or int(joints.max()) >= num_joints:
        raise ValueError(
            f"links: joint indices must lie in [0, {num_joints}) for {num_joints} joints, "
            f"got {links!r}"
        )
    lengths = torch.linalg.vector_norm(
        coords_hm[:, joints[:, 0]] - coords_hm[:, joints[:, 1]], dim=-1
    )
    assert lengths.shape == (coords_hm.shape[0], len(links))
    return lengths


def _link_nll(
    lengths: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood of every link length under its Gaussian, ``(B, L)``."""
    assert lengths.ndim == 2 and lengths.dtype.is_floating_point, (
        f"lengths: expected a float tensor of shape (B, L), got {lengths.dtype} "
        f"{tuple(lengths.shape)}"
    )
    for name, value in (("mean", mean), ("log_var", log_var)):
        assert value.shape == lengths.shape, (
            f"{name}: expected shape {tuple(lengths.shape)} of lengths, got {tuple(value.shape)}"
        )
        assert value.dtype == lengths.dtype, (
            f"{name}: expected dtype {lengths.dtype} of lengths, got {value.dtype}"
        )
    return 0.5 * (LOG_2PI + log_var + (lengths - mean) ** 2 * torch.exp(-log_var))


def gaussian_nll(
    lengths: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """Negative log-likelihood of the link lengths of every sample, summed over the links.

    The link ``l`` has the density ``N(lengths[:, l]; mean[:, l], exp(log_var[:, l]))``, so the
    term of a link is ``0.5 * (log(2 pi) + log_var + (length - mean)^2 * exp(-log_var))``.

    Args:
        lengths: ``(B, L)`` link lengths.
        mean: ``(B, L)`` predicted means, same dtype as ``lengths``.
        log_var: ``(B, L)`` predicted log-variances, same dtype as ``lengths``.

    Returns:
        ``(B,)`` negative log-likelihoods, in the dtype of ``lengths``.
    """
    nll = _link_nll(lengths, mean, log_var).sum(dim=1)
    assert nll.shape == (lengths.shape[0],)
    return nll


def train_auxnet(
    aux: AuxNet,
    features: torch.Tensor,
    gt_coords_hm: torch.Tensor,
    links: Sequence[tuple[int, int]],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    joint_exist: torch.Tensor | None = None,
) -> list[float]:
    """Train ``aux`` on labeled samples with Adam and the Gaussian negative log-likelihood.

    The target of link ``(u, v)`` is the distance between the ground-truth positions of joints
    ``u`` and ``v`` in heatmap cells. The batches are drawn from a private generator seeded with
    ``seed`` (a new permutation every epoch, the last batch may be smaller), so the run depends on
    the seed and on the initial weights of ``aux`` only, and leaves the global random state
    alone. The call puts ``aux`` in training mode.

    Args:
        aux: The network to train.
        features: ``(N, C, H, W)`` float backbone features of the labeled samples, already
            detached from the pose estimator (stop-gradient).
        gt_coords_hm: ``(N, K, 2)`` float32 ground-truth joint positions in heatmap cells.
        links: The ``aux.num_links`` skeleton links ``(u, v)``.
        epochs: Number of passes over the samples.
        lr: Learning rate of Adam.
        batch_size: Samples per step.
        seed: Seed of the batch order.
        joint_exist: ``(N, K)`` bool, False for joints without a ground-truth position. A link
            enters the loss of a sample only when both of its joints exist. ``None`` means that
            every joint exists.

    Returns:
        The mean negative log-likelihood per sample of every epoch (``epochs`` floats), measured
        on the batches while the network trains.

    Raises:
        ValueError: If ``epochs`` or ``batch_size`` is below 1, or ``links`` does not have one
            entry per output link of ``aux``.
    """
    assert not features.requires_grad, (
        "features: the pose network is frozen, so the features must be detached "
        "(stop-gradient), but they require grad"
    )
    assert (
        features.ndim == 4
        and features.shape[1] == aux.in_channels
        and features.dtype.is_floating_point
    ), (
        f"features: expected a float tensor of shape (N, {aux.in_channels}, H, W), "
        f"got {features.dtype} {tuple(features.shape)}"
    )
    n = features.shape[0]
    assert (
        gt_coords_hm.ndim == 3
        and gt_coords_hm.shape[0] == n
        and gt_coords_hm.shape[2] == 2
        and gt_coords_hm.dtype == torch.float32
    ), (
        f"gt_coords_hm: expected a float32 tensor of shape ({n}, K, 2), got "
        f"{gt_coords_hm.dtype} {tuple(gt_coords_hm.shape)}"
    )
    if len(links) != aux.num_links:
        raise ValueError(
            f"links: got {len(links)} links, but the auxiliary network predicts "
            f"{aux.num_links}"
        )
    if epochs < 1:
        raise ValueError(f"epochs: must be at least 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size: must be at least 1, got {batch_size}")

    with torch.no_grad():
        lengths = link_lengths(gt_coords_hm, links)
        if joint_exist is None:
            link_exists = torch.ones_like(lengths)
        else:
            assert (
                joint_exist.shape == gt_coords_hm.shape[:2]
                and joint_exist.dtype == torch.bool
            ), (
                f"joint_exist: expected a bool tensor of shape {tuple(gt_coords_hm.shape[:2])}, "
                f"got {joint_exist.dtype} {tuple(joint_exist.shape)}"
            )
            ends = torch.as_tensor(links, dtype=torch.long)
            link_exists = (
                (joint_exist[:, ends[:, 0]] & joint_exist[:, ends[:, 1]])
                .to(lengths.dtype)
                .to(lengths.device)
            )

    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(aux.parameters(), lr=lr)
    aux.train()
    history: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        epoch_total = 0.0
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            mean, log_var = aux(features[batch])
            nll = (_link_nll(lengths[batch], mean, log_var) * link_exists[batch]).sum(
                dim=1
            )
            optimizer.zero_grad(set_to_none=True)
            nll.mean().backward()
            optimizer.step()
            epoch_total += float(nll.detach().sum())
        history.append(epoch_total / n)
    return history


@torch.no_grad()
def predict_link_gaussians(
    aux: AuxNet, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``aux`` in evaluation mode without gradients.

    The training flag of ``aux`` is restored afterwards.

    Args:
        aux: The trained network.
        features: ``(B, C, H, W)`` backbone features.

    Returns:
        ``(mean, log_var)``, both ``(B, L)``.
    """
    was_training = aux.training
    aux.eval()
    try:
        return aux(features)
    finally:
        aux.train(was_training)
