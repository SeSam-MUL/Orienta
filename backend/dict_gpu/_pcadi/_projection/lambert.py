"""Vector <-> square Lambert mapping in PyTorch.

Translation of kikuchipy._vector2lambert (lines 578-615) and
kikuchipy._lambert2vector (lines 764-807) of
kikuchipy/signals/util/_master_pattern.py.

Critical conventions (see tasks/kikuchipy_projection_reference.md):

vector_to_lambert
  * Output range is [-sqrt(pi)/2, +sqrt(pi)/2] (uses SQRT_PI_OVER_2),
    NOT [-1, 1] and NOT [-sqrt(pi/2), +sqrt(pi/2)].
  * Branches on whether |y| <= |x|.
  * Uses arctan (NOT arctan2) with sign(x)/sign(y) supplying the
    quadrant. A naive port to atan2 silently breaks sign on half the
    sphere.
  * Sign of z is ignored (abs(z) used). Hemisphere dispatch is the
    caller's job.

lambert_to_vector
  * Input is normalised square coords in [-1, 1]; multiplied by
    sqrt(pi/2) internally (SQRT_PI_HALF, NOT SQRT_PI_OVER_2).
  * Branches on |x| <= |y|.
  * Returned vectors are NOT unit-normalised; z is always non-negative
    (upper hemisphere only).
"""
from __future__ import annotations
import math
import torch


SQRT_PI = math.sqrt(math.pi)
SQRT_PI_HALF = math.sqrt(math.pi / 2.0)     # ~ 1.2533141373155003
SQRT_PI_OVER_2 = SQRT_PI / 2.0              # ~ 0.8862269254527581
TWO_OVER_SQRT_PI = 2.0 / SQRT_PI            # ~ 1.1283791670955126


def vector_to_lambert(v: torch.Tensor) -> torch.Tensor:
    """Forward Lambert equal-area projection.

    Maps Cartesian 3-vectors onto a square in
    ``[-sqrt(pi)/2, +sqrt(pi)/2]^2``.

    Parameters
    ----------
    v
        Tensor of shape ``(..., 3)``. Need not be unit; the function
        normalises internally.

    Returns
    -------
    torch.Tensor
        Square Lambert coords ``(X, Y)`` of shape ``(..., 2)``,
        preserving the input device and dtype.
    """
    # Normalise (vectorised; kikuchipy also normalises internally)
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w = v / norm

    x = w[..., 0]
    y = w[..., 1]
    z = w[..., 2]
    abs_x = x.abs()
    abs_y = y.abs()
    abs_z = z.abs()

    # sqrt(2 * (1 - |z|)); clamp inside to avoid tiny negatives from FP noise
    sqrt_z = torch.sqrt((2.0 * (1.0 - abs_z)).clamp_min(0.0))

    # Branch: |y| <= |x|
    branch_x = abs_y <= abs_x

    # sign() returns 0 for input 0 — match numpy.sign exactly.
    sign_x = torch.sign(x)
    sign_y = torch.sign(y)

    # Guard divisions against the pole case (x==0 AND y==0, which is the
    # only way `branch_x` selects a divide-by-zero — branch_x requires
    # |y| <= |x|, so x=0 forces y=0). When the denominator is exactly 0
    # replace it with 1.0; sqrt_z will be 0 at the pole and zero out the
    # whole product. Without this guard, `arctan(0/0) = NaN` propagates
    # through `sign(0)*0*NaN` because IEEE NaN is sticky.
    safe_x = torch.where(x == 0, torch.ones_like(x), x)
    safe_y = torch.where(y == 0, torch.ones_like(y), y)

    # Branch |y| <= |x|:
    #   X = sign(x) * sqrt_z * SQRT_PI_OVER_2
    #   Y = sign(x) * sqrt_z * TWO_OVER_SQRT_PI * arctan(y / x)
    X_branch_x = sign_x * sqrt_z * SQRT_PI_OVER_2
    Y_branch_x = sign_x * sqrt_z * TWO_OVER_SQRT_PI * torch.arctan(y / safe_x)

    # Else branch (|y| > |x|):
    #   X = sign(y) * sqrt_z * TWO_OVER_SQRT_PI * arctan(x / y)
    #   Y = sign(y) * sqrt_z * SQRT_PI_OVER_2
    X_branch_y = sign_y * sqrt_z * TWO_OVER_SQRT_PI * torch.arctan(x / safe_y)
    Y_branch_y = sign_y * sqrt_z * SQRT_PI_OVER_2

    X = torch.where(branch_x, X_branch_x, X_branch_y)
    Y = torch.where(branch_x, Y_branch_x, Y_branch_y)

    # Pole case: abs_z == 1 -> sqrt_z = 0 -> X = Y = 0 already (sign*0).
    # No special-case needed.

    out = torch.stack([X, Y], dim=-1)
    return out


def lambert_to_vector(uv: torch.Tensor) -> torch.Tensor:
    """Inverse Lambert: square-grid coords -> upper-hemisphere vector.

    Parameters
    ----------
    uv
        Tensor of shape ``(..., 2)`` of normalised square-grid
        coordinates in ``[-1, 1]``; multiplied internally by
        ``sqrt(pi/2)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(..., 3)``. **NOT unit-normalised.** z is
        always non-negative (upper hemisphere). Preserves input
        device/dtype.
    """
    # Rescale [-1, 1] -> [-sqrt(pi/2), sqrt(pi/2)]
    xi = uv[..., 0] * SQRT_PI_HALF
    yi = uv[..., 1] * SQRT_PI_HALF

    xi_abs = xi.abs()
    yi_abs = yi.abs()

    # Branch: |x| <= |y| (kikuchipy's literal condition)
    branch_y = xi_abs <= yi_abs

    # Guard against division by 0 in either branch (the branch where the
    # denominator is 0 won't be selected, but where() still evaluates both
    # sides). Where the *denominator* would be 0, substitute 1.0 to avoid
    # NaN propagation.
    safe_yi = torch.where(branch_y, yi, torch.ones_like(yi))
    safe_xi = torch.where(branch_y, torch.ones_like(xi), xi)

    pi = math.pi

    # |x| <= |y| branch:
    #   q = 2*yi * sqrt(pi - yi^2) / pi
    #   qq = xi * pi * 0.25 / yi
    #   (x', y', z') = (q*sin(qq), q*cos(qq), 1 - 2*yi^2/pi)
    q_y = 2.0 * yi * torch.sqrt((pi - yi * yi).clamp_min(0.0)) / pi
    qq_y = xi * (pi * 0.25) / safe_yi
    x_y = q_y * torch.sin(qq_y)
    y_y = q_y * torch.cos(qq_y)
    z_y = 1.0 - 2.0 * (yi * yi) / pi

    # |x| > |y| branch:
    #   q = 2*xi * sqrt(pi - xi^2) / pi
    #   qq = yi * pi * 0.25 / xi
    #   (x', y', z') = (q*cos(qq), q*sin(qq), 1 - 2*xi^2/pi)
    q_x = 2.0 * xi * torch.sqrt((pi - xi * xi).clamp_min(0.0)) / pi
    qq_x = yi * (pi * 0.25) / safe_xi
    x_x = q_x * torch.cos(qq_x)
    y_x = q_x * torch.sin(qq_x)
    z_x = 1.0 - 2.0 * (xi * xi) / pi

    out_x = torch.where(branch_y, x_y, x_x)
    out_y = torch.where(branch_y, y_y, y_x)
    out_z = torch.where(branch_y, z_y, z_x)

    # Pole case: max(|xi|, |yi|) == 0 -> (0, 0, 1).
    # In the |x| <= |y| branch with yi == 0: q_y = 0, z_y = 1 -> (0, 0, 1).
    # safe_yi guard prevents NaN. So this falls out for free.

    pole = (xi_abs == 0) & (yi_abs == 0)
    if pole.any():
        out_x = torch.where(pole, torch.zeros_like(out_x), out_x)
        out_y = torch.where(pole, torch.zeros_like(out_y), out_y)
        out_z = torch.where(pole, torch.ones_like(out_z), out_z)

    return torch.stack([out_x, out_y, out_z], dim=-1)
