"""Modified-Lambert (Rosca-Lambert) projection on the unit sphere.

Pure-torch port of kikuchipy's reference implementation in
``kikuchipy.signals.util._master_pattern`` (functions ``_vector2lambert``
and ``_lambert2vector``).

Storage convention: this module stores the square Lambert coordinates in
the **unit square** ``[-1, 1]^2`` -- that is, kikuchipy's raw output
(scale factor ``sqrt(pi/2)`` along the axes) divided by ``sqrt(pi/2)``.
This matches kikuchipy's downstream consumer
``_get_lambert_interpolation_parameters``, which divides by
``SQRT_PI_HALF`` exactly to land on the unit square. The inverse
function ``_lambert2vector`` reverses this: it begins by multiplying its
input by ``sqrt(pi/2)``, so it natively consumes unit-square coordinates.

All operations are vectorized torch ops with no Python loops, so they
run on either CPU or CUDA tensors.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

# Same constants as kikuchipy (_master_pattern.py L74-78)
_SQRT_PI = math.sqrt(math.pi)
_SQRT_PI_OVER_2 = _SQRT_PI / 2.0          # = sqrt(pi)/2  ~= 0.8862
_SQRT_PI_HALF = math.sqrt(math.pi / 2.0)  # = sqrt(pi/2)  ~= 1.2533
_TWO_OVER_SQRT_PI = 2.0 / _SQRT_PI


def direction_to_lambert(direction: torch.Tensor) -> torch.Tensor:
    """Project (N, 3) unit directions onto the unit Lambert square.

    Vectorized port of kikuchipy's ``_vector2lambert`` (L577-615 of
    ``_master_pattern.py``), with the trailing scale factor of
    ``1/sqrt(pi/2)`` folded in so the output lies in ``[-1, 1]^2``
    instead of EMsoft's ``[-sqrt(pi/2), sqrt(pi/2)]^2``.

    Parameters
    ----------
    direction
        Tensor of shape ``(N, 3)`` containing unit-length directions in
        Cartesian coordinates. Both hemispheres are accepted; ``|z|`` is
        what determines the radial coordinate, so points on the lower
        hemisphere project onto the same square as their mirror image.

    Returns
    -------
    xy
        Tensor of shape ``(N, 2)`` with values in ``[-1, 1]^2``.
    """
    if direction.dim() != 2 or direction.shape[-1] != 3:
        raise ValueError(f"expected (N, 3), got {tuple(direction.shape)}")

    x, y, z = direction.unbind(-1)

    abs_x = x.abs()
    abs_y = y.abs()
    abs_z = z.abs().clamp(max=1.0)

    # kikuchipy line 603: sqrt(2 * (1 - |z|))
    sqrt_z = torch.sqrt((2.0 * (1.0 - abs_z)).clamp(min=0.0))

    sign_x = torch.sign(x)
    sign_y = torch.sign(y)

    # Avoid division by zero. The result of the branch where the
    # denominator is zero is masked out below by ``branch_x`` /
    # pole_mask, so the placeholder value does not matter.
    eps = torch.finfo(x.dtype).eps
    safe_x = torch.where(abs_x > 0, x, torch.full_like(x, eps))
    safe_y = torch.where(abs_y > 0, y, torch.full_like(y, eps))

    # Branch A (kikuchipy L606-609): |y| <= |x|
    #   raw_X = sign_x * sqrt_z * SQRT_PI_OVER_2
    #   raw_Y = sign_x * sqrt_z * TWO_OVER_SQRT_PI * atan(y / x)
    # Divide by SQRT_PI_HALF to land in [-1, 1]^2:
    #   X = sign_x * sqrt_z * (SQRT_PI_OVER_2 / SQRT_PI_HALF)
    #     = sign_x * sqrt_z / sqrt(2)
    #   Y = sign_x * sqrt_z * (TWO_OVER_SQRT_PI / SQRT_PI_HALF) * atan(y/x)
    #     = sign_x * sqrt_z * (2 / (sqrt(pi) * sqrt(pi/2))) * atan(y/x)
    #     = sign_x * sqrt_z * (sqrt(2)*2/pi) * atan(y/x)
    # We keep the kikuchipy form and just divide once at the end.
    xa_A = sign_x * sqrt_z * _SQRT_PI_OVER_2
    ya_A = sign_x * sqrt_z * _TWO_OVER_SQRT_PI * torch.atan(y / safe_x)

    # Branch B (kikuchipy L610-613): |y| > |x|
    xa_B = sign_y * sqrt_z * _TWO_OVER_SQRT_PI * torch.atan(x / safe_y)
    ya_B = sign_y * sqrt_z * _SQRT_PI_OVER_2

    branch_a = abs_y <= abs_x  # matches kikuchipy "elif np.abs(y) <= np.abs(x)"

    xa = torch.where(branch_a, xa_A, xa_B)
    ya = torch.where(branch_a, ya_A, ya_B)

    # Pole maps to (0, 0) -- kikuchipy L604-605 ("if abs_z == 1: continue")
    pole_mask = abs_z >= 1.0
    xa = torch.where(pole_mask, torch.zeros_like(xa), xa)
    ya = torch.where(pole_mask, torch.zeros_like(ya), ya)

    # EMsoft scale -> unit square (kikuchipy L683: ``/ SQRT_PI_HALF``).
    xy = torch.stack((xa, ya), dim=-1) / _SQRT_PI_HALF
    return xy


def lambert_to_direction(
    xy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert the Lambert square back to a (N, 3) unit direction.

    Vectorized port of kikuchipy's ``_lambert2vector`` (L758-807 of
    ``_master_pattern.py``). Input is expected on the unit square
    ``[-1, 1]^2`` -- kikuchipy multiplies by ``sqrt(pi/2)`` internally
    (L789-790), which restores the EMsoft-scale coordinates the rest of
    the formulae assume.

    Parameters
    ----------
    xy
        Tensor of shape ``(N, 2)`` with values in ``[-1, 1]^2``.

    Returns
    -------
    direction
        Tensor of shape ``(N, 3)``, unit length, on the upper hemisphere
        (``z >= 0``). Callers handle hemisphere flips for lower-square
        lookups.
    hemisphere
        Tensor of shape ``(N,)``, dtype ``int8``. Always 0 here (upper);
        kept for symmetry with future two-sided implementations.
    """
    if xy.dim() != 2 or xy.shape[-1] != 2:
        raise ValueError(f"expected (N, 2), got {tuple(xy.shape)}")

    x_unit, y_unit = xy.unbind(-1)

    # kikuchipy L789-790: scale unit-square -> EMsoft-square coordinates.
    xi = x_unit * _SQRT_PI_HALF
    yi = y_unit * _SQRT_PI_HALF

    xi_abs = xi.abs()
    yi_abs = yi.abs()

    # Pole: max(|xi|, |yi|) == 0  =>  (0, 0, 1)  (kikuchipy L795-796)
    pole_mask = (xi_abs == 0) & (yi_abs == 0)

    # Avoid divide-by-zero in the branches. Masked-out values are
    # overwritten via torch.where below, so the placeholder is harmless.
    eps = torch.finfo(xi.dtype).eps
    safe_xi = torch.where(xi_abs > 0, xi, torch.full_like(xi, eps))
    safe_yi = torch.where(yi_abs > 0, yi, torch.full_like(yi, eps))

    # Branch A (kikuchipy L798-801): xi_abs <= yi_abs
    #   q  = 2*yi*sqrt(pi - yi^2)/pi
    #   qq = xi*pi/(4*yi)
    #   cart = (q*sin(qq), q*cos(qq), 1 - 2*yi^2/pi)
    qA = 2.0 * yi * torch.sqrt((math.pi - yi.pow(2)).clamp(min=0.0)) / math.pi
    qqA = xi * (math.pi * 0.25) / safe_yi
    dx_A = qA * torch.sin(qqA)
    dy_A = qA * torch.cos(qqA)
    dz_A = 1.0 - 2.0 * yi.pow(2) / math.pi

    # Branch B (kikuchipy L802-805): xi_abs >  yi_abs
    qB = 2.0 * xi * torch.sqrt((math.pi - xi.pow(2)).clamp(min=0.0)) / math.pi
    qqB = yi * (math.pi * 0.25) / safe_xi
    dx_B = qB * torch.cos(qqB)
    dy_B = qB * torch.sin(qqB)
    dz_B = 1.0 - 2.0 * xi.pow(2) / math.pi

    branch_a = xi_abs <= yi_abs
    dx = torch.where(branch_a, dx_A, dx_B)
    dy = torch.where(branch_a, dy_A, dy_B)
    dz = torch.where(branch_a, dz_A, dz_B)

    # Apply pole exception last so it overrides any branch result.
    dx = torch.where(pole_mask, torch.zeros_like(dx), dx)
    dy = torch.where(pole_mask, torch.zeros_like(dy), dy)
    dz = torch.where(pole_mask, torch.ones_like(dz), dz)

    direction = torch.stack((dx, dy, dz), dim=-1)
    # kikuchipy's _lambert2vector docstring notes the result "might not
    # be on the unit sphere" due to floating-point drift. Renormalize so
    # the round-trip identity holds tightly.
    norm = direction.norm(dim=-1, keepdim=True).clamp(min=torch.finfo(direction.dtype).eps)
    direction = direction / norm

    hemisphere = torch.zeros(
        direction.shape[0], dtype=torch.int8, device=direction.device
    )
    return direction, hemisphere
