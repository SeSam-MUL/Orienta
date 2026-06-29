"""CNN encoder for EBSD diffraction patterns.

A small ResNet-style network that maps a grayscale 128x128 EBSD pattern
to a fixed-length feature vector. The architecture uses residual blocks
with batch normalization, adapted for single-channel scientific imagery.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ebsd_ai.config import DEFAULT_PATTERN_SIZE


class ResidualBlock(nn.Module):
    """Two-conv residual block with optional downsampling.

    Parameters
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    stride : int
        Stride for the first conv (2 to downsample).
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, C_in, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, C_out, H', W')``.
        """
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        result: torch.Tensor = out
        return result


class PatternEncoder(nn.Module):
    """ResNet-style CNN encoder for EBSD patterns.

    Takes a single-channel grayscale pattern of size
    ``(B, 1, pattern_size, pattern_size)`` and produces a feature vector
    of size ``(B, feature_dim)``.

    Architecture (~2.5M parameters):
        Conv7x7/2 -> BN -> ReLU -> MaxPool
        -> 2x ResBlock(32)
        -> 2x ResBlock(64, stride=2)
        -> 2x ResBlock(128, stride=2)
        -> 2x ResBlock(256, stride=2)
        -> AdaptiveAvgPool -> Linear -> feature_dim

    Parameters
    ----------
    feature_dim : int
        Output feature vector dimension. Default 256.
    pattern_size : int
        Expected input spatial size (square). Default 128.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        pattern_size: int = DEFAULT_PATTERN_SIZE,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.pattern_size = pattern_size

        # Initial convolution: 1 channel -> 32 channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, feature_dim)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a sequence of residual blocks."""
        blocks: list[nn.Module] = [
            ResidualBlock(in_channels, out_channels, stride=stride)
        ]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of patterns.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 1, H, W)`` float32 patterns.

        Returns
        -------
        torch.Tensor
            Shape ``(B, feature_dim)`` feature vectors.
        """
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        result: torch.Tensor = out
        return result

    def freeze_early_layers(self) -> None:
        """Freeze stem and first two residual stages for transfer learning."""
        for module in [self.stem, self.layer1, self.layer2]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
