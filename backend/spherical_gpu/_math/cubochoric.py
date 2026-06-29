"""Cubochoric coordinates for SO(3): equal-volume parameterization that maps
the rotation group to a uniform cube of side π^(2/3).

Vendored from ebsdtorch by Zachary Varley (MIT, see fz_ori.py for full
license text). Source: https://github.com/ZacharyVarley/ebsdtorch/blob/main/
ebsdtorch/s2_and_so3/orientations.py.

Mathematical references:
- Rosca, Morawiec & De Graef. "A new method of constructing a grid in the
  space of 3D rotations and its applications to texture analysis." Modelling
  Simul. Mater. Sci. Eng. 22, 075013 (2014). [The "cubochoric" coordinate.]
- Singh & De Graef. "Orientation sampling for dictionary-based diffraction
  pattern indexing methods." MSMSE 24, 085013 (2016). [Empirical resolution
  fit eq. 12: theta_avg(N) ≈ 131.97049 / (N - 0.038551).]

Pipeline forward:  cu  →  ho  →  ax  →  qu
Pipeline inverse:  qu  →  ho  →  cu

Vendored 2026-05-09 for GPU spherical-indexing perf C1c milestone M1.2.
The functions in this file are bit-for-bit copies of the upstream ebsdtorch
implementations. We re-export `cu2qu` and `qu2cu` as the high-level API the
indexer needs.
"""
from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Forward pipeline: cu → ho → ax → qu
# ---------------------------------------------------------------------------

@torch.jit.script
def cu2ho(cu: torch.Tensor) -> torch.Tensor:
    """Cubochoric → homochoric (Rosca/Morawiec/De Graef 2014, eqs. 25-29).

    Args:
        cu: (..., 3) cubochoric coordinates in [-π^(2/3)/2, π^(2/3)/2]^3.

    Returns:
        (..., 3) homochoric coordinates inside the homochoric ball of
        radius (3π/4)^(1/3).
    """
    indices = torch.argsort(torch.abs(cu), dim=-1, descending=False)
    sorted = torch.gather(cu, -1, indices)
    s, m, b = sorted.unbind(dim=-1)

    trig_arg_xy = s * torch.pi / (12.0 * m)
    trig_arg_xy[torch.isnan(trig_arg_xy)] = 0

    factor_xy = (
        2 ** (1 / 12)
        * 3 ** (1 / 3)
        * m
        * torch.sqrt(
            (
                4 * b ** 2 * (torch.cos(trig_arg_xy) - (2 ** 0.5))
                + (2 ** 0.5) * m ** 2 * (-2 * (2 ** 0.5) * torch.cos(trig_arg_xy) + 3)
            )
            / (torch.cos(trig_arg_xy) - (2 ** 0.5))
        )
        / (torch.pi ** (1 / 3) * b * torch.sqrt(-torch.cos(trig_arg_xy) + (2 ** 0.5)))
    )

    x_s3 = factor_xy * torch.sin(trig_arg_xy)
    y_s3 = factor_xy * 2 ** -0.5 * ((2 ** 0.5) * torch.cos(trig_arg_xy) - 1)
    z_s3 = (
        2 * 6 ** (1 / 3) * b ** 2 * (torch.cos(trig_arg_xy) - (2 ** 0.5))
        + 2 ** (5 / 6)
        * 3 ** (1 / 3)
        * m ** 2
        * (-2 * (2 ** 0.5) * torch.cos(trig_arg_xy) + 3)
    ) / (2 * torch.pi ** (1 / 3) * b * (torch.cos(trig_arg_xy) - (2 ** 0.5)))

    ho = torch.stack((x_s3, y_s3, z_s3), dim=-1)
    ho = torch.scatter(ho, -1, indices, ho)

    ho[torch.isnan(ho)] = 0
    ho.copysign_(cu)
    return ho


@torch.jit.script
def ho2ax(ho: Tensor, fast: bool = True) -> Tensor:
    """Homochoric → axis-angle. Polynomial fit (Varley 2024-05-24).

    The mapping ``f(w) = [(3/4) * (w - sin(w))]^(1/3)`` has no closed-form
    inverse, so the inverse is approximated by a Chebyshev polynomial fit
    in `|ho|^2`. FP32 path uses 9 terms (machine eps); FP64 path uses 11
    terms + 1 Newton iteration when `fast=False`.

    Args:
        ho:   (..., 3) homochoric coordinates.
        fast: skip the Newton refinement step (FP64 only). Default True.

    Returns:
        (..., 4) axis-angle in the format (x, y, z, angle).
    """
    if ho.dtype == torch.float32 or ho.dtype == torch.float16:
        fit_parameters = torch.tensor([
            1.0000000000000009e+00,
            -4.9999943403867775e-01,
            -2.5015165060149020e-02,
            -3.8120131548551729e-03,
            -1.2106188330642162e-03,
             4.9329295993155416e-04,
            -7.0089385526450620e-04,
             3.0979774923589078e-04,
            -7.3023474963298843e-05,
        ], dtype=ho.dtype, device=ho.device)
    else:
        fit_parameters = torch.tensor([
            1.0000000000000000e+00,
            -4.9999997124013285e-01,
            -2.5001181866044025e-02,
            -3.9144209820521038e-03,
            -8.9320268104539483e-04,
             3.1181024286083695e-05,
            -4.3961032788396477e-04,
             3.9657471727506439e-04,
            -2.6379945050586932e-04,
             9.1185355979587159e-05,
            -1.4875867805692529e-05,
        ], dtype=ho.dtype, device=ho.device)

    ho_norm_sq = torch.sum(ho ** 2, dim=-1, keepdim=False)
    s = torch.zeros_like(ho_norm_sq)
    for i in range(len(fit_parameters)):
        s += fit_parameters[i] * ho_norm_sq ** i

    if ho.dtype == torch.float64 and not fast:
        w = 2 * torch.arccos(torch.clamp(s, -1.0, 1.0))
        f_w = ((3 / 4) * (w - torch.sin(w))) ** (1 / 3) - torch.sqrt(ho_norm_sq)
        f_p_w = (1 - torch.cos(w)) / (6 ** (2 / 3) * (w - torch.sin(w)) ** (2 / 3))
        update = f_w / f_p_w
        update[torch.isnan(update)] = 0
        w -= update
    else:
        w = 2.0 * torch.arccos(torch.clamp(s, -1.0, 1.0))

    ax = torch.concat([
        ho * torch.rsqrt(ho_norm_sq).unsqueeze(-1),
        w.unsqueeze(-1),
    ], dim=-1)
    rot_is_identity = torch.abs(ho_norm_sq) < 1e-6
    ax[rot_is_identity] = 0
    ax[rot_is_identity, ..., 2] = 1.0
    return ax


@torch.jit.script
def ax2qu(ax: Tensor) -> Tensor:
    """Axis-angle → quaternion. ZYZ half-angle formula.

    Args:
        ax: (..., 4) axis-angle (x, y, z, angle).

    Returns:
        (..., 4) quaternion (w, x, y, z).
    """
    qu = torch.empty_like(ax)
    cos_half_ang = torch.cos(ax[..., 3] / 2.0)
    sin_half_ang = torch.sin(ax[..., 3:4] / 2.0)
    qu[..., 0] = cos_half_ang
    qu[..., 1:] = ax[..., :3] * sin_half_ang
    return qu


@torch.jit.script
def cu2qu(cu: Tensor) -> Tensor:
    """Cubochoric → quaternion (full forward pipeline cu → ho → ax → qu).

    Args:
        cu: (..., 3) cubochoric coordinates.

    Returns:
        (..., 4) unit quaternion (w, x, y, z), w >= 0 by construction.
    """
    return ax2qu(ho2ax(cu2ho(cu)))


# ---------------------------------------------------------------------------
# Inverse pipeline: qu → ho → cu
# ---------------------------------------------------------------------------

@torch.jit.script
def qu2ho(qu: Tensor) -> Tensor:
    """Quaternion → homochoric coordinate.

    Args:
        qu: (..., 4) unit quaternion (w, x, y, z).

    Returns:
        (..., 3) homochoric coordinates.
    """
    if qu.size(-1) != 4:
        raise ValueError(f"Invalid quaternion shape {qu.shape}.")
    angle = 2 * torch.acos(qu[..., 0:1].clamp_(min=-1.0, max=1.0))
    unit = qu[..., 1:] / torch.norm(qu[..., 1:], dim=-1, keepdim=True)
    ho = unit * (3.0 * (angle - torch.sin(angle)) / 4.0) ** (1 / 3)
    ho[(angle.squeeze(-1) < 1e-8)] = 0.0
    return ho


@torch.jit.script
def ho2cu(ho: Tensor) -> Tensor:
    """Homochoric → cubochoric (closed-form inverse of cu2ho).

    Uses Rosca 2014 inverse Lambert formulas; no polynomial fit needed.
    """
    indices = torch.argsort(torch.abs(ho), dim=-1, descending=False)
    sorted_ho = torch.gather(ho, -1, indices)
    x_s3, y_s3, z_s3 = torch.abs(sorted_ho).unbind(dim=-1)

    r_s = torch.norm(ho, dim=-1, keepdim=False)
    prefactor_xy_s3 = torch.sqrt(2 * r_s / (r_s + z_s3))
    x_s2 = x_s3 * prefactor_xy_s3
    y_s2 = y_s3 * prefactor_xy_s3
    z_s2 = (torch.pi / 6) ** 0.5 * r_s

    prefactor_xy_s2 = (
        (torch.pi / 6) ** 0.5
        * torch.sqrt(
            x_s2 ** 2
            + y_s2 * (torch.sqrt(x_s2 ** 2 + 2 * y_s2 ** 2) + 2 * y_s2)
        )
        / (2 ** 0.5)
    )

    x_s1 = (prefactor_xy_s2 * 12.0 * torch.sign(x_s2) / torch.pi) * (
        torch.arccos(
            (
                (x_s2 ** 2 + y_s2 * torch.sqrt(x_s2 ** 2 + 2 * y_s2 ** 2))
                / ((2 ** 0.5) * (x_s2 ** 2 + y_s2 ** 2))
            ).clamp_(-1.0, 1.0)
        )
    )
    y_s1 = prefactor_xy_s2 * torch.sign(y_s2)

    cu = torch.empty_like(ho)
    cu.scatter_(-1, indices, torch.stack((x_s1, y_s1, z_s2), dim=-1))
    cu /= (torch.pi / 6) ** (1 / 6)

    cu.copysign_(ho)
    cu[torch.isnan(cu)] = 0
    return cu


@torch.jit.script
def qu2cu(qu: Tensor) -> Tensor:
    """Quaternion → cubochoric (full inverse pipeline qu → ho → cu).

    Args:
        qu: (..., 4) unit quaternion (w, x, y, z), w >= 0.

    Returns:
        (..., 3) cubochoric coordinates in [-π^(2/3)/2, π^(2/3)/2]^3.
    """
    return ho2cu(qu2ho(qu))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cube edge length: cubochoric coordinates lie in [-CUBE_HALF_EDGE, +CUBE_HALF_EDGE]^3.
CUBE_HALF_EDGE = (torch.pi ** (2.0 / 3.0)) / 2.0


__all__ = [
    "cu2ho",
    "ho2ax",
    "ax2qu",
    "cu2qu",
    "qu2ho",
    "ho2cu",
    "qu2cu",
    "CUBE_HALF_EDGE",
]
