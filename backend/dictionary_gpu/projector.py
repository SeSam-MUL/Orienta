"""Project a master pattern to detector patterns for a batch of orientations."""
from __future__ import annotations

from typing import Tuple

import torch

from .rotations import rotate_directions_by_quaternions
from .sampling import sample_master_at_directions


def project_patterns(
    quats: torch.Tensor,
    pixel_directions: torch.Tensor,
    hemispheres: torch.Tensor,
    detector_shape: Tuple[int, int],
) -> torch.Tensor:
    """Generate (N, H, W) patterns from (N, 4) quats and (H*W, 3) pixel directions.

    All inputs must be on the same device.
    """
    if quats.device != pixel_directions.device or quats.device != hemispheres.device:
        raise ValueError("all inputs must be on the same device")

    H, W = detector_shape
    if pixel_directions.shape[0] != H * W:
        raise ValueError(
            f"pixel_directions has {pixel_directions.shape[0]} rows, "
            f"expected H*W = {H * W}"
        )

    rotated = rotate_directions_by_quaternions(quats, pixel_directions)   # (N, P, 3)
    intensities = sample_master_at_directions(hemispheres, rotated)        # (N, P)
    return intensities.view(quats.shape[0], H, W)
