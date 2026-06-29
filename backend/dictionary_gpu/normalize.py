"""Per-pattern zero-mean / unit-variance normalization."""
from __future__ import annotations

import torch


def normalize_patterns(patterns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Subtract per-pattern mean, divide by per-pattern std. Operates on (N, H, W)."""
    if patterns.dim() != 3:
        raise ValueError(f"expected (N, H, W), got {tuple(patterns.shape)}")

    flat = patterns.view(patterns.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, unbiased=False, keepdim=True).clamp(min=eps)
    normed = (flat - mean) / std
    return normed.view_as(patterns)
