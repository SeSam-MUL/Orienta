"""Multimodal feature fusion with attention.

Combines pattern encoder features and EDS+metadata encoder features into
a single fused representation. An attention gate learns to weight the
contribution of each modality per sample, enabling graceful degradation
when EDS data is missing (all zeros).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Soft attention over two feature streams.

    Learns a scalar weight ``alpha`` in [0, 1] per sample that controls
    how much the pattern versus EDS feature stream contributes to the
    fused representation.

    ``alpha`` close to 1 → trust pattern more.
    ``alpha`` close to 0 → trust EDS more.

    Parameters
    ----------
    pattern_dim : int
        Dimension of the pattern feature vector.
    eds_dim : int
        Dimension of the EDS feature vector.
    hidden_dim : int
        Hidden size of the attention MLP.
    """

    def __init__(
        self,
        pattern_dim: int,
        eds_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(pattern_dim + eds_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pattern_feat: torch.Tensor,
        eds_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention weight alpha.

        Parameters
        ----------
        pattern_feat : torch.Tensor
            Shape ``(B, pattern_dim)``.
        eds_feat : torch.Tensor
            Shape ``(B, eds_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)`` attention weight alpha (pattern trust).
        """
        combined = torch.cat([pattern_feat, eds_feat], dim=1)
        out: torch.Tensor = self.gate(combined)
        return out


class MultimodalFusion(nn.Module):
    """Attention-gated fusion of pattern and EDS feature streams.

    First projects each modality to a common dimension, then applies a
    learned attention gate to weight contributions, and finally passes
    the attended sum through a fusion MLP.

    The attention weight ``alpha`` is exposed on each forward pass so that
    downstream code can report how much EDS influenced the prediction
    (``eds_contribution = 1 - alpha``).

    Parameters
    ----------
    pattern_dim : int
        Dimension of input pattern features.
    eds_dim : int
        Dimension of input EDS features.
    fused_dim : int
        Output dimension of the fused representation.
    dropout : float
        Dropout probability in the fusion MLP.
    """

    def __init__(
        self,
        pattern_dim: int = 256,
        eds_dim: int = 64,
        fused_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pattern_dim = pattern_dim
        self.eds_dim = eds_dim
        self.fused_dim = fused_dim

        # Project both modalities to the same dimension for gated sum
        self.pattern_proj = nn.Linear(pattern_dim, fused_dim)
        self.eds_proj = nn.Linear(eds_dim, fused_dim)

        # Attention gate
        self.attention = AttentionGate(
            pattern_dim=pattern_dim,
            eds_dim=eds_dim,
            hidden_dim=64,
        )

        # Fusion MLP: refines the attended combination
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        pattern_feat: torch.Tensor,
        eds_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse pattern and EDS features with learned attention.

        Parameters
        ----------
        pattern_feat : torch.Tensor
            Shape ``(B, pattern_dim)`` from :class:`PatternEncoder`.
        eds_feat : torch.Tensor
            Shape ``(B, eds_dim)`` from :class:`EDSEncoder`.

        Returns
        -------
        fused : torch.Tensor
            Shape ``(B, fused_dim)`` fused feature vector.
        alpha : torch.Tensor
            Shape ``(B, 1)`` attention weight. Values near 1 indicate the
            model trusts the pattern stream; values near 0 indicate trust
            in the EDS stream. ``eds_contribution = 1 - alpha``.
        """
        # Attention gate on raw features
        alpha = self.attention(pattern_feat, eds_feat)  # (B, 1)

        # Project to common dimension
        p = self.pattern_proj(pattern_feat)  # (B, fused_dim)
        e = self.eds_proj(eds_feat)  # (B, fused_dim)

        # Weighted combination
        fused = alpha * p + (1.0 - alpha) * e  # (B, fused_dim)

        # Refine through fusion MLP
        fused = self.fusion_mlp(fused)  # (B, fused_dim)

        return fused, alpha
