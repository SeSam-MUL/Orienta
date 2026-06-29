"""Bilinear sampling on Lambert hemispheres.

Translation of kikuchipy's
  - `_get_lambert_interpolation_parameters` (lines 627-725)
  - `_get_pixel_from_master_pattern`        (lines 729-756)
of `kikuchipy/signals/util/_master_pattern.py`, fused into a single
`torch.nn.functional.grid_sample` call per hemisphere.

Critical conventions (see tasks/kikuchipy_projection_reference.md):

  1. Lambert -> pixel coords rescaling:
         xy = scale * _vector2lambert(v) / SQRT_PI_HALF
         pixel_row = xy[:, 1] + scale   (NOTE: Y -> row)
         pixel_col = xy[:, 0] + scale   (NOTE: X -> col)
     With scale = (npx-1)/2 this simplifies to
         pixel_row / (npx-1) * 2 - 1 = lambert_y / SQRT_PI_HALF
         pixel_col / (npx-1) * 2 - 1 = lambert_x / SQRT_PI_HALF
     i.e. the grid_sample coords (align_corners=True) are just
     `(lambert_xy / SQRT_PI_HALF)`. grid_sample's layout is
     `grid[..., 0] = x (column)`, `grid[..., 1] = y (row)`.

  2. Hemisphere dispatch on raw `z >= 0` (NOT abs(z)). Equality goes to
     upper hemisphere. `_vector2lambert` itself ignores the sign of z so
     the Lambert UV is identical for both hemispheres.

  3. align_corners=True replicates kikuchipy's "scale * lambert + scale"
     pixel-center semantics. align_corners=False would shift by half a
     pixel and break parity.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .lambert import vector_to_lambert, SQRT_PI_HALF


def sample_master(
    mp_north: torch.Tensor,
    mp_south: torch.Tensor,
    dirs: torch.Tensor,
) -> torch.Tensor:
    """Bilinear-sample the two hemisphere master patterns at given directions.

    Parameters
    ----------
    mp_north, mp_south
        2D tensors of shape ``(npx, npy)`` on the same CUDA device, same
        dtype (typically float32). Upper / lower hemisphere master
        patterns in square Lambert projection.
    dirs
        Direction vectors of shape ``(N, 3)`` (or any leading shape with
        trailing dim 3). Need not be unit; ``vector_to_lambert``
        normalises internally.

    Returns
    -------
    torch.Tensor
        Sampled intensities of shape ``(N,)`` (or input leading shape),
        same dtype as the master patterns.
    """
    if mp_north.shape != mp_south.shape:
        raise ValueError(
            f"hemisphere shape mismatch: {mp_north.shape} vs {mp_south.shape}"
        )
    if mp_north.ndim != 2:
        raise ValueError(f"master patterns must be 2D, got {mp_north.ndim}D")
    if dirs.shape[-1] != 3:
        raise ValueError(f"dirs last dim must be 3, got {dirs.shape[-1]}")

    lead_shape = dirs.shape[:-1]
    n = int(torch.tensor(lead_shape).prod().item()) if lead_shape else 1
    v = dirs.reshape(n, 3)

    # Lambert UV in [-sqrt(pi)/2, +sqrt(pi)/2]; divide by SQRT_PI_HALF
    # (= sqrt(pi/2)) puts grid_sample coords in [-1/sqrt(2)*sqrt(pi)/(sqrt(pi/2)) ...]
    # which simplifies to the standard [-1, 1] convention align_corners=True
    # expects (see module docstring).
    uv = vector_to_lambert(v) / SQRT_PI_HALF   # (n, 2)

    # grid_sample wants grid as (N=1, H_out, W_out=1, 2), values in [-1, 1],
    # with [..., 0] = x (column), [..., 1] = y (row). Lambert X is the
    # column axis of the master image (kikuchipy quirk: j = xy[:, 0]).
    grid = uv.view(1, n, 1, 2).to(mp_north.dtype)

    # grid_sample needs (N, C, H, W) inputs
    inp_n = mp_north.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    inp_s = mp_south.unsqueeze(0).unsqueeze(0)

    samp_n = F.grid_sample(
        inp_n, grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    ).reshape(n)
    samp_s = F.grid_sample(
        inp_s, grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    ).reshape(n)

    # Hemisphere dispatch on z >= 0 (>= matters — equality goes to upper).
    north_mask = v[:, 2] >= 0
    out = torch.where(north_mask, samp_n, samp_s)

    if lead_shape:
        out = out.reshape(lead_shape)
    return out
