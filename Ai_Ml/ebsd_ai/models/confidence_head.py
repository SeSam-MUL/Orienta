"""Confidence estimation head for embedding encoder.

Auxiliary head that predicts indexing confidence from pre-projection
512-dim features. Target confidence based on actual angular error:
c = exp(-error^2 / (2 * sigma_c^2)).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def confidence_target(
    angular_errors_deg: np.ndarray,
    sigma_c: float = 5.0,
) -> np.ndarray:
    """Compute confidence targets from angular errors.

    Parameters
    ----------
    angular_errors_deg : np.ndarray
        Angular errors in degrees.
    sigma_c : float
        Width parameter for Gaussian confidence mapping.

    Returns
    -------
    np.ndarray
        Confidence values in [0, 1].
    """
    return np.exp(-angular_errors_deg ** 2 / (2 * sigma_c ** 2))


class ConfidenceHead(nn.Module):
    """Auxiliary confidence head.

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
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict confidence from features.

        Parameters
        ----------
        features : torch.Tensor
            (B, feature_dim) pre-projection features.

        Returns
        -------
        torch.Tensor
            (B, 1) confidence values in [0, 1].
        """
        return self.head(features)
