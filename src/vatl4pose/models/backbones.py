"""ResNet-50 and SE-ResNet-50 trunks that feed the pose heads.

Both trunks reuse the bottleneck block of torchvision. They end at the stride-32 feature map with
2048 channels (``8 x 6`` for a ``256 x 192`` crop) and have no classifier. The layer names
(``conv1``, ``bn1``, ``layer1`` ... ``layer4``, and ``se.fc`` inside a squeeze-excitation block)
follow the released pose checkpoints, so those load with ``strict=True``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import Bottleneck, conv1x1

# ResNet-50 stages as (planes, number of blocks, stride of the first block).
_STAGES = ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2))
FEATURE_CHANNELS = 2048


class SqueezeExcite(nn.Module):
    """Channel gating: average pool, two linear layers with a ReLU between, sigmoid scale.

    The hidden width equals the channel count, which is the width stored in the released
    checkpoints. The gate lives in ``fc`` (``Sequential``) so that the weights are named
    ``fc.0.*`` and ``fc.2.*``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.shape
        squeezed = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, C)
        gate = self.fc(squeezed)
        return x * gate[:, :, None, None]


class SEBottleneck(Bottleneck):
    """Torchvision bottleneck with a squeeze-excitation gate on the residual branch.

    The gate acts after the last batch norm and before the shortcut is added.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__(inplanes, planes, stride=stride, downsample=downsample)
        self.se = SqueezeExcite(planes * self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50(nn.Module):
    """ResNet-50 up to ``layer4``: ``(B, 3, H, W) -> (B, 2048, H/32, W/32)``.

    ``first_block`` is the block class used at the start of every stage. The other blocks are plain
    torchvision bottlenecks.
    """

    def __init__(self, first_block: type[Bottleneck] = Bottleneck) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        inplanes = 64
        for index, (planes, blocks, stride) in enumerate(_STAGES, start=1):
            setattr(
                self,
                f"layer{index}",
                self._make_stage(first_block, inplanes, planes, blocks, stride),
            )
            inplanes = planes * Bottleneck.expansion
        assert inplanes == FEATURE_CHANNELS

    @staticmethod
    def _make_stage(
        first_block: type[Bottleneck],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        outplanes = planes * Bottleneck.expansion
        # The first block of every stage changes the channel count, so it always has a projection.
        downsample = nn.Sequential(
            conv1x1(inplanes, outplanes, stride), nn.BatchNorm2d(outplanes)
        )
        layers: list[nn.Module] = [first_block(inplanes, planes, stride, downsample)]
        layers += [Bottleneck(outplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[1] == 3, x.shape
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        assert x.shape[1] == FEATURE_CHANNELS, x.shape
        return x

    def load_imagenet_weights(self) -> None:
        """Copy the torchvision ImageNet weights into the trunk (downloads them if not cached).

        The classifier of the torchvision model has no counterpart here and is skipped. Layers the
        torchvision model does not have (the squeeze-excitation gates) keep their initial values.
        """
        state = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(
            progress=True, check_hash=True
        )
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"ImageNet ResNet-50 has keys the trunk lacks: {result.unexpected_keys[:3]}"
            )
        gate_free = [k for k in result.missing_keys if ".se." not in k]
        if gate_free:
            raise RuntimeError(
                f"ImageNet ResNet-50 lacks keys the trunk needs: {gate_free[:3]}"
            )


class SEResNet50(ResNet50):
    """ResNet-50 whose first block of each stage carries a squeeze-excitation gate."""

    def __init__(self) -> None:
        super().__init__(first_block=SEBottleneck)

