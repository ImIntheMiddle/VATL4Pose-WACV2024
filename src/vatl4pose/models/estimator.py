"""Pose estimators of the active-learning loop: SimpleBaseline, FastPose and HRNet-W32.

Every estimator maps a ``(B, 3, 256, 192)`` crop to ``(B, K, 64, 48)`` heatmaps. SimpleBaseline and
FastPose expose their backbone, so a pass can also return the ``(B, 2048)`` embedding that the
diversity-based query strategies cluster. HRNet-W32 is a reference estimator without embeddings.

The parameter names equal the keys of the released checkpoints, so those load with ``strict=True``
through :func:`build_estimator` without key renaming.

The head layers start from PyTorch's default initialization; ``build_estimator(init_head=True)``
draws their weights from a normal distribution with a standard deviation of 0.001 instead (see
``initialize_head``).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from third_party.hrnet.pose_hrnet import PoseHighResolutionNet
from vatl4pose.models.backbones import FEATURE_CHANNELS, ResNet50, SEResNet50
from vatl4pose.models.heads import DUC, deconv_stack

INPUT_SIZE = (256, 192)  # crop height and width
HEATMAP_SIZE = (64, 48)  # stride 4
BACKBONE_SIZE = (8, 6)  # stride 32
NO_EMBEDDING_MESSAGE = "embeddings are defined for simplebaseline and fastpose only"

# Learning-rate multipliers of the retraining groups: the new head learns 10x or 5x faster than
# the pretrained backbone.
_OUTPUT_LR_MULT = 10.0
_UPSAMPLE_LR_MULT = 5.0

# HRNet-W32 stages; the field names are those the vendored HRNet reads.
_HRNET_W32_STAGES: dict[str, dict[str, Any]] = {
    "stage2": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "NUM_BLOCKS": [4, 4],
        "NUM_CHANNELS": [32, 64],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    },
    "stage3": {
        "NUM_MODULES": 4,
        "NUM_BRANCHES": 3,
        "NUM_BLOCKS": [4, 4, 4],
        "NUM_CHANNELS": [32, 64, 128],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    },
    "stage4": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 4,
        "NUM_BLOCKS": [4, 4, 4, 4],
        "NUM_CHANNELS": [32, 64, 128, 256],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    },
}


@dataclass(frozen=True)
class EstimatorConfig:
    """Which estimator to build and where its weights come from.

    ``imagenet_init`` initializes the ResNet-50 backbone from the torchvision ImageNet weights, which
    downloads them. Use it only to train from scratch; it is refused together with ``weights``
    (the checkpoint would overwrite it) and for ``hrnet_w32`` (no ImageNet weights are defined).
    """

    arch: Literal["simplebaseline", "fastpose", "hrnet_w32"]
    weights: pathlib.Path | None
    num_joints: int
    imagenet_init: bool = False


class PoseEstimator(nn.Module):
    """Interface shared by the three estimators."""

    num_joints: int
    arch: str
    """Name of the estimator in :class:`EstimatorConfig` and in the result files."""
    has_embedding: bool = False
    """Whether :meth:`forward_with_embedding` and :meth:`backbone_features` are defined."""

    def _check_input(self, x: torch.Tensor) -> None:
        assert x.is_floating_point(), x.dtype
        assert x.ndim == 4 and tuple(x.shape[1:]) == (3, *INPUT_SIZE), tuple(x.shape)

    def _check_heatmaps(self, heatmaps: torch.Tensor, batch: int) -> torch.Tensor:
        assert tuple(heatmaps.shape) == (batch, self.num_joints, *HEATMAP_SIZE), tuple(
            heatmaps.shape
        )
        return heatmaps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Heatmaps ``(B, K, 64, 48)`` of the crops ``x`` ``(B, 3, 256, 192)``."""
        raise NotImplementedError

    def forward_with_embedding(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Heatmaps and the ``(B, 2048)`` global average of the backbone output, from one pass."""
        raise NotImplementedError(NO_EMBEDDING_MESSAGE)

    def backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone output ``(B, 2048, 8, 6)``."""
        raise NotImplementedError(NO_EMBEDDING_MESSAGE)

    def _lr_modules(self) -> list[tuple[str, nn.Module, float]]:
        """``(group name, module, learning-rate multiplier)`` of every retraining group."""
        raise NotImplementedError

    def param_groups(self, base_lr: float) -> list[dict[str, Any]]:
        """Optimizer groups, one per retrained module of ``_lr_modules`` with its multiplier.

        Each parameter appears once.
        """
        groups = [
            {"name": name, "params": list(module.parameters()), "lr": base_lr * mult}
            for name, module, mult in self._lr_modules()
        ]
        grouped = [id(p) for g in groups for p in g["params"]]
        assert len(grouped) == len(set(grouped)), "a parameter belongs to two groups"
        assert set(grouped) == {id(p) for p in self.parameters()}, (
            "a parameter belongs to no group"
        )
        return groups


class _ResNetEstimator(PoseEstimator):
    """Backbone ``preact`` plus an upsampling head; subclasses define the head in ``_decode``."""

    preact: ResNet50
    has_embedding = True

    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def initialize_head(self) -> None:
        """Draw the head layers with small normal weights; the backbone stays as it is."""
        raise NotImplementedError

    def backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        features = self.preact(x)
        assert tuple(features.shape[1:]) == (FEATURE_CHANNELS, *BACKBONE_SIZE), tuple(
            features.shape
        )
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._check_heatmaps(self._decode(self.backbone_features(x)), x.shape[0])

    def forward_with_embedding(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone_features(x)
        heatmaps = self._check_heatmaps(self._decode(features), x.shape[0])
        embedding = F.adaptive_avg_pool2d(features, 1).flatten(1)
        assert tuple(embedding.shape) == (x.shape[0], FEATURE_CHANNELS), tuple(
            embedding.shape
        )
        return heatmaps, embedding


class SimpleBaselineEstimator(_ResNetEstimator):
    """ResNet-50, three transposed-convolution blocks of 256 filters, and a 1x1 output convolution."""

    arch = "simplebaseline"

    def __init__(self, num_joints: int) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.preact = ResNet50()
        self.deconv_layers = deconv_stack(FEATURE_CHANNELS, 256, num_layers=3)
        self.final_layer = nn.Conv2d(256, num_joints, kernel_size=1)

    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.deconv_layers(features))

    def initialize_head(self) -> None:
        """Normal(0, 0.001) weights for the transposed convolutions and the output convolution.

        The batch norms of the upsampling get weight 1 and bias 0, and the output bias is set to 0.
        """
        for module in self.deconv_layers.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.constant_(self.final_layer.bias, 0)

    def _lr_modules(self) -> list[tuple[str, nn.Module, float]]:
        return [
            ("final_layer", self.final_layer, _OUTPUT_LR_MULT),
            ("preact", self.preact, 1.0),
            ("deconv_layers", self.deconv_layers, _UPSAMPLE_LR_MULT),
        ]


class FastPoseEstimator(_ResNetEstimator):
    """SE-ResNet-50, a pixel shuffle, two DUC blocks and a 3x3 output convolution.

    The shuffle turns ``(2048, 8, 6)`` into ``(512, 16, 12)``. ``duc1`` gives ``(256, 32, 24)``,
    ``duc2`` gives ``(128, 64, 48)``, and ``conv_out`` maps the 128 channels to the joints. The
    shuffle has no parameters, so it belongs to no optimizer group.
    """

    arch = "fastpose"

    def __init__(self, num_joints: int) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.preact = SEResNet50()
        self.shuffle = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024)
        self.duc2 = DUC(256, 512)
        self.conv_out = nn.Conv2d(128, num_joints, kernel_size=3, stride=1, padding=1)

    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.duc2(self.duc1(self.shuffle(features))))

    def initialize_head(self) -> None:
        """Normal(0, 0.001) weights and zero bias for the output convolution.

        The DUC blocks keep their default initialization.
        """
        nn.init.normal_(self.conv_out.weight, std=0.001)
        nn.init.constant_(self.conv_out.bias, 0)

    def _lr_modules(self) -> list[tuple[str, nn.Module, float]]:
        return [
            ("conv_out", self.conv_out, _OUTPUT_LR_MULT),
            ("preact", self.preact, 1.0),
            ("duc1", self.duc1, _UPSAMPLE_LR_MULT),
            ("duc2", self.duc2, _UPSAMPLE_LR_MULT),
        ]


class HRNetEstimator(PoseHighResolutionNet, PoseEstimator):
    """HRNet-W32 of Microsoft (vendored under ``third_party/hrnet``) with a 1x1 output convolution."""

    arch = "hrnet_w32"

    def __init__(self, num_joints: int) -> None:
        super().__init__(num_joints, **_HRNET_W32_STAGES, final_conv_kernel=1)
        self.num_joints = num_joints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return self._check_heatmaps(super().forward(x), x.shape[0])

    def _lr_modules(self) -> list[tuple[str, nn.Module, float]]:
        return [("all", self, 1.0)]


_ARCHS: dict[str, type[PoseEstimator]] = {
    "simplebaseline": SimpleBaselineEstimator,
    "fastpose": FastPoseEstimator,
    "hrnet_w32": HRNetEstimator,
}


def _load_checkpoint(model: PoseEstimator, path: pathlib.Path) -> None:
    """Load a state dict into ``model`` strictly; every error names the file and the first bad key."""
    if not path.is_file():
        raise FileNotFoundError(f"pose estimator checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint {path} is not a state dict")
    expected = model.state_dict()
    missing = [k for k in expected if k not in state]
    unexpected = [k for k in state if k not in expected]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {path} does not match {type(model).__name__}: "
            f"{len(missing)} missing keys (first: {missing[:1]}), "
            f"{len(unexpected)} unexpected keys (first: {unexpected[:1]})"
        )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:  # size mismatch of a tensor; add the file name
        raise RuntimeError(
            f"checkpoint {path} does not fit {type(model).__name__}: {exc}"
        ) from exc


def build_estimator(
    cfg: EstimatorConfig, device: torch.device, *, init_head: bool = False
) -> PoseEstimator:
    """Build the estimator of ``cfg`` on ``device``, in the default (training) mode.

    ``cfg.weights`` is loaded with ``torch.load(weights_only=True)`` and ``strict=True``. Without
    weights and without ``imagenet_init`` the estimator keeps its random initial values.

    With ``init_head`` the head layers of an estimator without a checkpoint are drawn by
    ``initialize_head`` (small normal weights, zero biases). The draw happens on the CPU before the
    estimator moves to ``device``, so the global CPU generator decides it. The default keeps
    PyTorch's initialization.

    Raises:
        ValueError: If the arch is unknown, ``num_joints`` is not positive, or ``imagenet_init`` or
            ``init_head`` cannot apply: HRNet-W32 has no ImageNet weights and keeps its own
            initialization, and a checkpoint would overwrite either.
    """
    if cfg.arch not in _ARCHS:
        raise ValueError(f"unknown arch {cfg.arch!r}; expected one of {sorted(_ARCHS)}")
    if cfg.num_joints <= 0:
        raise ValueError(f"num_joints must be positive, got {cfg.num_joints}")
    if cfg.imagenet_init and cfg.arch == "hrnet_w32":
        raise ValueError(
            "imagenet_init is defined for simplebaseline and fastpose only, not hrnet_w32"
        )
    if cfg.imagenet_init and cfg.weights is not None:
        raise ValueError(
            f"imagenet_init would be overwritten by weights {cfg.weights}; set only one"
        )
    if init_head and cfg.arch == "hrnet_w32":
        raise ValueError(
            "init_head is defined for simplebaseline and fastpose only, not hrnet_w32"
        )
    if init_head and cfg.weights is not None:
        raise ValueError(
            f"init_head would be overwritten by weights {cfg.weights}; set only one"
        )
    model = _ARCHS[cfg.arch](cfg.num_joints)
    if cfg.imagenet_init:
        assert isinstance(model, _ResNetEstimator)
        model.preact.load_imagenet_weights()
    if init_head:
        assert isinstance(model, _ResNetEstimator)
        model.initialize_head()
    if cfg.weights is not None:
        _load_checkpoint(model, cfg.weights)
    return model.to(device)
