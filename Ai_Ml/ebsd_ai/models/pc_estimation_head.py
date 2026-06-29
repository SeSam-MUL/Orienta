"""Pattern Center estimation head for embedding encoder.

Auxiliary head that predicts PC (x, y, z) from pre-projection
512-dim features. Enables multi-task training:
L_total = L_contrastive + lambda_pc * MSE(PC_pred, PC_true).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PCEstimationHead(nn.Module):
    """Pattern Center estimation auxiliary head.

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
            nn.Linear(128, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict pattern center coordinates.

        Parameters
        ----------
        features : torch.Tensor
            (B, feature_dim) pre-projection features.

        Returns
        -------
        torch.Tensor
            (B, 3) predicted (PC_x, PC_y, PC_z).
        """
        return self.head(features)
