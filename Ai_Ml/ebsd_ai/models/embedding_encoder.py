"""ResNet-18 embedding encoder for EBSD patterns.

Maps grayscale EBSD patterns to L2-normalized embedding vectors on
the unit hypersphere.  The encoder is designed for contrastive learning
where similar crystal orientations should produce similar embeddings.

Architecture: Modified ResNet-18 with 1-channel input, adaptive pooling,
and a projection head to the embedding dimension with L2 normalization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class EmbeddingEncoder(nn.Module):
    """ResNet-18 encoder producing L2-normalized embeddings.

    Parameters
    ----------
    embedding_dim : int
        Output embedding dimension (default 128).
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stem: 1-channel grayscale input
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # ResNet stages: 64 -> 128 -> 256 -> 512
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection head: 512 -> embedding_dim
        self.projection = nn.Linear(512, embedding_dim)

    @staticmethod
    def _make_layer(
        in_ch: int, out_ch: int, n_blocks: int, stride: int,
    ) -> nn.Sequential:
        layers = [_ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(_ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to 512-dim feature vector (pre-projection)."""
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.flatten(1)  # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode patterns to L2-normalized embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, H, W) grayscale patterns.

        Returns
        -------
        torch.Tensor
            Shape (B, embedding_dim) L2-normalized embeddings.
        """
        features = self._encode(x)
        projected = self.projection(features)
        return F.normalize(projected, p=2, dim=1)

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Get pre-projection 512-dim feature vector.

        Useful for auxiliary heads (confidence, strain).

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Shape (B, 512).
        """
        return self._encode(x)

    def get_save_dict(self) -> dict:
        """Build checkpoint dict."""
        return {
            "state_dict": self.state_dict(),
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_save_dict(
        cls, save_dict: dict, device: torch.device | None = None,
    ) -> "EmbeddingEncoder":
        """Restore from checkpoint."""
        model = cls(embedding_dim=save_dict["embedding_dim"])
        model.load_state_dict(save_dict["state_dict"])
        if device is not None:
            model = model.to(device)
        return model
