"""L1 + SSIM combined loss for EBSD pattern denoising.

L1 preserves pixel accuracy.  SSIM preserves Kikuchi band edges and
structural similarity -- critical for downstream indexing quality.
SSIM implementation follows Wang et al. (2004) with 11x11 Gaussian window.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2-D Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return g / g.sum()


def compute_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Structural Similarity Index (Wang et al., 2004).

    Parameters
    ----------
    x, y : ``(B, 1, H, W)`` tensors in [0, 1] range.
    window_size : int
        Side length of the Gaussian weighting window.
    sigma : float
        Standard deviation of the Gaussian window.
    c1, c2 : float
        Stabilisation constants (defaults assume data range 1.0).

    Returns
    -------
    Scalar mean SSIM in [-1, 1].
    """
    window = _gaussian_window(window_size, sigma).to(x.device, x.dtype)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator
    return ssim_map.mean()


class DenoisingLoss(nn.Module):
    """Combined L1 + SSIM loss for pattern denoising.

    Parameters
    ----------
    l1_weight : float
        Weight for L1 (mean absolute error) component.
    ssim_weight : float
        Weight for (1 - SSIM) component.
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.5) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Parameters
        ----------
        predicted, target : ``(B, 1, H, W)`` tensors.

        Returns
        -------
        Scalar loss tensor.
        """
        l1 = F.l1_loss(predicted, target)
        ssim_val = compute_ssim(predicted, target)
        return self.l1_weight * l1 + self.ssim_weight * (1.0 - ssim_val)
