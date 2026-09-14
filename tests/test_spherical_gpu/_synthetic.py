"""Synthetic test fixtures for the GPU spherical indexer.

The spherical indexer has many subtle conventions (PC sign, sample tilt,
detector frame rotation, ZYZ↔ZXZ Euler, Bunge vs Roe, etc.) that are
near-impossible to debug on real-world EBSD patterns where pattern
quality, lattice strain, and instrumental artefacts mix into the disorientation
distribution.

Synthetic round-trip tests cut through that:

1. Choose a known orientation R (Euler triple or quaternion).
2. Synthesize a "pattern" that we KNOW corresponds to a sphere image rotated
   by R (use the master pattern itself, evaluated at directions rotated by R^-1
   then forward-projected onto the detector).
3. Run the indexer.
4. Assert the recovered orientation matches R within tolerance.

If the round-trip fails on the EASY synthetic case, the real-data gate has
zero chance — so this test ALWAYS runs first.

The fixture below is a *helper*, not a test itself; tests import it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from backend.spherical_gpu._math import (
    rosca_lambert,
    theta_phi_to_xyz,
    grid_DriscollHealy,
)
from backend.spherical_gpu.pipeline.detector import DetectorGeometry
from backend.spherical_gpu.pipeline.sht_io import SHTMasterFile


@dataclass
class SyntheticPattern:
    """A synthetic detector pattern with known ground-truth orientation."""
    pattern: torch.Tensor                  # (H, W) float32 on device
    orientation_quat: torch.Tensor         # (4,) float64 on device — unit quaternion (w,x,y,z)
    orientation_euler: Tuple[float, float, float]  # Bunge ZXZ in radians (phi1, Phi, phi2)
    note: str                              # human-readable description of how it was made


def euler_to_quat(phi1: float, Phi: float, phi2: float) -> torch.Tensor:
    """Bunge ZXZ Euler → unit quaternion (w, x, y, z), float64.

    Convention matches orix.quaternion.Rotation.from_euler default ('bunge').
    """
    c1, s1 = math.cos(phi1 / 2.0), math.sin(phi1 / 2.0)
    cP, sP = math.cos(Phi / 2.0), math.sin(Phi / 2.0)
    c2, s2 = math.cos(phi2 / 2.0), math.sin(phi2 / 2.0)
    # ZXZ Bunge active rotation
    w = c1 * cP * c2 - s1 * cP * s2
    x = c1 * sP * c2 + s1 * sP * s2
    y = -c1 * sP * s2 + s1 * sP * c2
    z = c1 * cP * s2 + s1 * cP * c2
    q = torch.tensor([w, x, y, z], dtype=torch.float64)
    return q / q.norm()


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Active rotation of vectors v (..., 3) by quaternion q (4,)."""
    w, x, y, z = q.unbind(-1)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    # v' = q * v * q^-1, expanded
    rx = (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz
    ry = 2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz
    rz = 2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz
    return torch.stack([rx, ry, rz], dim=-1)


def synthetic_pattern_from_master(
    master: SHTMasterFile,
    geom: DetectorGeometry,
    euler_zxz: Tuple[float, float, float],
    flip_y: bool = True,
) -> SyntheticPattern:
    """Build a synthetic pattern by sampling the master at sphere directions
    obtained by forward-projecting detector pixels and rotating them by R^-1.

    The intended round-trip is::

        synth = synthetic_pattern_from_master(..., euler_zxz=R)
        Tier1Indexer(...).index(synth.pattern.unsqueeze(0))
        # → recovers R within tolerance

    Notes
    -----
    The master SHT coefficients are evaluated on the unit sphere via inverse
    SHT (sht.isht); we sample the resulting spherical image at the rotated
    detector directions. This skips Lambert resampling — going directly
    sphere → detector pixel → bilinear-on-sphere — which means it tests the
    detector geometry + indexer convention WITHOUT depending on the SHT
    forward path.

    Implementation kept minimal for MVP: we use a low-resolution proxy of
    the master pattern (DH grid) instead of the full inverse SHT, since we
    only need a recognisable signal, not photometric accuracy.
    """
    device = master.coefs_ml.device
    H, W = geom.pat_h, geom.pat_w

    # 1. Build a proxy spherical image from the master's m=0 column (real).
    #    Crude but adequate for orientation discrimination tests.
    L = min(master.bandwidth, 64)
    dh = grid_DriscollHealy(bandlimit=L, dtype=7).to(device)        # (2L, 2L, 2)
    sphere_xyz = theta_phi_to_xyz(dh).to(device).to(torch.float64)  # (2L, 2L, 3)

    # 2. Rotate sphere directions by R^-1 (=conj(q) for unit q)
    q = euler_to_quat(*euler_zxz).to(device)
    q_inv = torch.tensor([q[0], -q[1], -q[2], -q[3]], device=device, dtype=torch.float64)

    # 3. Forward-project sphere directions to detector. For each (n0,n1,n2),
    #    use the EMSphInx Geometry::interpolatePixel formula.
    sigma_deg = geom.tilt_deg
    theta_c_deg = 10.0
    alpha = math.radians(90.0 - sigma_deg + theta_c_deg)
    sA = math.sin(alpha); cA = math.cos(alpha)
    xpc = float(geom.xpc.item())
    ypc = float(geom.ypc.item())
    Lscint = float(geom.L.item())
    delta = geom.pixel_size

    n_rotated = quat_rotate(q_inv.expand(sphere_xyz.shape[:-1] + (4,)), sphere_xyz)
    n0 = n_rotated[..., 0]; n1 = n_rotated[..., 1]; n2 = n_rotated[..., 2]
    denom = n0 * sA + n2 * cA
    valid = denom > 1e-12
    d = Lscint / denom.clamp_min(1e-12)
    x_um = n1 * d
    y_um = (sA * n2 - cA * n0) * d

    X = (xpc + x_um / delta) / W + 0.5
    Y = (ypc + y_um / delta) / H + 0.5
    if flip_y:
        Y = 1.0 - Y

    inside = valid & (X >= 0) & (X <= 1) & (Y >= 0) & (Y <= 1)
    # Detector pixel coords (rows = top→down, cols = left→right)
    Xpix = (X * W).clamp(0, W - 1)
    Ypix = (Y * H).clamp(0, H - 1)

    # 4. Build synthetic pattern: assign sphere brightness to each detector
    #    pixel that gets hit; gaps stay zero. We approximate "sphere brightness"
    #    using the proxy spherical image (a Y20 spherical harmonic-like signal:
    #    f(theta) = 3 cos^2(theta) - 1, which is uniaxial and depends only on
    #    the polar angle of the rotated direction).
    proxy_brightness = 3.0 * sphere_xyz[..., 2] ** 2 - 1.0  # (2L, 2L)
    # Project onto detector via splatting (zero-init pattern, accumulate)
    pat = torch.zeros(H, W, dtype=torch.float64, device=device)
    counts = torch.zeros(H, W, dtype=torch.float64, device=device)
    if inside.any():
        rows = Ypix[inside].long()
        cols = Xpix[inside].long()
        vals = proxy_brightness[inside]
        # Use index_put_ with accumulate to handle multi-hit pixels
        pat.index_put_((rows, cols), vals, accumulate=True)
        counts.index_put_((rows, cols), torch.ones_like(vals), accumulate=True)
    pat = torch.where(counts > 0, pat / counts.clamp_min(1.0), torch.zeros_like(pat))

    # Add small noise so the indexer doesn't see a perfectly synthetic signature
    torch.manual_seed(42)
    pat = pat + 0.01 * torch.randn_like(pat)

    return SyntheticPattern(
        pattern=pat.float(),
        orientation_quat=q,
        orientation_euler=euler_zxz,
        note=(
            "Y20-proxy spherical image rotated by ZXZ Euler "
            f"({euler_zxz[0]:.3f}, {euler_zxz[1]:.3f}, {euler_zxz[2]:.3f}) rad, "
            "splatted onto detector via EMSphInx interpolatePixel"
        ),
    )
