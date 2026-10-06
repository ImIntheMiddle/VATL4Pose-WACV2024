"""Upsampling heads that turn the stride-32 backbone map into stride-4 heatmaps.

Two designs are used. SimpleBaseline stacks transposed convolutions. FastPose uses sub-pixel
shuffles, each preceded by a convolution (DUC, "dense upsampling convolution"). Layer names match the
released pose checkpoints.
"""

from __future__ import annotations

import torch
from torch import nn


def deconv_stack(
    in_channels: int = 2048, filters: int = 256, num_layers: int = 3
) -> nn.Sequential:
    """Stack of ``ConvTranspose2d(k=4, s=2, p=1) -> BatchNorm2d -> ReLU`` blocks, 2x upsampling each.

    The modules sit at consecutive positions ``3 * i`` (convolution), ``3 * i + 1`` (batch norm) and
    ``3 * i + 2`` (ReLU), which gives the weights the names ``<3 i>.weight`` and ``<3 i + 1>.*``.
    """
    assert num_layers >= 1
    layers: list[nn.Module] = []
    channels = in_channels
    for _ in range(num_layers):
        layers += [
            nn.ConvTranspose2d(
                channels, filters, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        ]
        channels = filters
    return nn.Sequential(*layers)


class DUC(nn.Module):
    """3x3 convolution, batch norm, ReLU, then a pixel shuffle that doubles height and width.

    ``(B, in_channels, H, W) -> (B, out_channels / upscale**2, H * upscale, W * upscale)``.
    """

    def __init__(self, in_channels: int, out_channels: int, upscale: int = 2) -> None:
        super().__init__()
        assert out_channels % (upscale * upscale) == 0, (out_channels, upscale)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.shape
        return self.shuffle(self.relu(self.bn(self.conv(x))))
