"""Strain quantification head for embedding encoder.

Auxiliary head that estimates local strain level from pre-projection
512-dim features. Output is non-negative (ReLU) representing
sigma_strain which correlates with KAM.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StrainHead(nn.Module):
    """Strain quantification auxiliary head.

    Parameters
    ----------
    feature_dim : int
        Input feature dimension (default 512).
    """

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.ReLU(),  # strain is non-negative
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict strain level from features.

        Parameters
        ----------
        features : torch.Tensor
            (B, feature_dim) pre-projection features.

        Returns
        -------
        torch.Tensor
            (B, 1) non-negative strain estimates.
        """
        return self.head(features)
