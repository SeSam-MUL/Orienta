"""Sample master pattern at arbitrary unit directions using Lambert + bilinear."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .lambert import direction_to_lambert


def sample_master_at_directions(
    hemispheres: torch.Tensor,
    rotated_directions: torch.Tensor,
) -> torch.Tensor:
    """Look up master pattern at each rotated direction.

    Parameters
    ----------
    hemispheres : (2, M, M)
        [0] upper hemisphere Lambert square, [1] lower.
    rotated_directions : (N, P, 3)
        Unit directions per orientation per pixel.

    Returns
    -------
    (N, P) tensor of interpolated master pattern intensities.
    """
    if hemispheres.dim() != 3 or hemispheres.shape[0] != 2:
        raise ValueError(f"hemispheres must be (2, M, M), got {tuple(hemispheres.shape)}")
    if rotated_directions.dim() != 3 or rotated_directions.shape[-1] != 3:
        raise ValueError(f"rotated_directions must be (N, P, 3), got {tuple(rotated_directions.shape)}")

    N, P, _ = rotated_directions.shape
    flat = rotated_directions.reshape(N * P, 3)

    # Choose hemisphere by sign of z; flip z for lower so we look up its own square
    is_lower = flat[:, 2] < 0
    flat_for_lambert = flat.clone()
    flat_for_lambert[is_lower, 2] = -flat_for_lambert[is_lower, 2]

    xy = direction_to_lambert(flat_for_lambert)        # (N*P, 2) in [-1, 1]
    # grid_sample expects (B, C, H_in, W_in) input, (B, H_out, W_out, 2) grid in [-1, 1]
    # We'll do two grid_sample calls (upper and lower) and pick per pixel.
    upper = hemispheres[0:1].unsqueeze(0)              # (1, 1, M, M)
    lower = hemispheres[1:2].unsqueeze(0)              # (1, 1, M, M)

    grid = xy.view(1, N * P, 1, 2)
    sampled_upper = F.grid_sample(
        upper, grid, mode="bilinear", padding_mode="border", align_corners=True
    ).view(-1)
    sampled_lower = F.grid_sample(
        lower, grid, mode="bilinear", padding_mode="border", align_corners=True
    ).view(-1)

    sampled = torch.where(is_lower, sampled_lower, sampled_upper)
    return sampled.view(N, P)
