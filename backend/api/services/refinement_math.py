"""Pure math primitives for joint R+PC refinement.

- Quaternion <-> so(3) exponential map for unconstrained LM optimization
  around an initial orientation (avoids the unit-norm constraint and the
  q vs -q double-cover singularity).
- Disorientation in degrees for diagnostic layers.
- Sparse 4-neighbour graph Laplacian for the Stage-2 smoothness solve.

See spec: docs/superpowers/specs/2026-05-14-joint-r-pc-refinement-design.md
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.sparse import coo_matrix


def so3_exp_to_quat(omega: torch.Tensor) -> torch.Tensor:
    """Map axis-angle vector ω ∈ ℝ³ to a unit quaternion (w, x, y, z).

    q = exp_q(ω) = (cos(|ω|/2), sin(|ω|/2) * ω̂)

    For small |ω|, returns (1, ω/2, ...) up to floating-point precision.
    Supports unbatched (3,) and batched (B, 3) inputs.
    """
    omega = omega.to(torch.float64)
    if omega.ndim == 1:
        omega = omega.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    norm = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)  # (B, 1)
    half = norm * 0.5
    # Avoid division by zero — use small-angle Taylor expansion sinc(x) ≈ 1 for tiny x
    safe_norm = torch.where(norm < 1e-12, torch.ones_like(norm), norm)
    sin_half_over_norm = torch.where(
        norm < 1e-12,
        torch.full_like(norm, 0.5),                 # lim_{x→0} sin(x/2)/x = 1/2
        torch.sin(half) / safe_norm,
    )
    w = torch.cos(half)                              # (B, 1)
    xyz = omega * sin_half_over_norm                 # (B, 3)
    q = torch.cat([w, xyz], dim=-1)                  # (B, 4)
    if squeeze:
        q = q.squeeze(0)
    return q


def quat_to_so3_log(q: torch.Tensor) -> torch.Tensor:
    """Inverse of so3_exp_to_quat. Maps (w, x, y, z) quaternion to ω ∈ ℝ³.

    ω = 2 * atan2(|xyz|, w) * (xyz / |xyz|)
    For w near ±1 (small rotation), uses Taylor expansion to preserve precision.
    """
    q = q.to(torch.float64)
    if q.ndim == 1:
        q = q.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    # Disambiguate the double cover: if w < 0, flip signs (use q and -q equiv class)
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    q = q * sign
    w = q[..., 0]                                    # (B,)
    xyz = q[..., 1:]                                 # (B, 3)
    sin_half_norm = torch.linalg.vector_norm(xyz, dim=-1)
    cos_half = torch.clamp(w, -1.0, 1.0)
    half_angle = torch.atan2(sin_half_norm, cos_half)  # (B,)
    angle = 2.0 * half_angle
    safe_sin = torch.where(sin_half_norm < 1e-12,
                           torch.ones_like(sin_half_norm),
                           sin_half_norm)
    scale = torch.where(
        sin_half_norm < 1e-12,
        torch.full_like(sin_half_norm, 1.0),         # lim small angle: ω/|xyz| = 2
        angle / safe_sin,
    )
    omega = xyz * scale.unsqueeze(-1)
    if squeeze:
        omega = omega.squeeze(0)
    return omega


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (w, x, y, z) quaternions. Supports broadcasting."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def disorientation_degrees(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Disorientation angle between (w,x,y,z) quaternions in degrees.

    Symmetry-NOT-applied (we want the literal rotation difference between two
    quaternion samples, not the crystal-symmetry-aware minimum). For
    symmetry-aware disorientation, use orix downstream.

    Inputs are normalised internally so that truncated (e.g. float32-rounded)
    quaternions still yield d(q, q) == 0.

    Supports unbatched (4,) and batched (..., 4) inputs.
    """
    # Normalise to unit quaternions so the dot product is a true cos(θ/2).
    n1 = torch.linalg.vector_norm(q1, dim=-1, keepdim=True).clamp_min(1e-30)
    n2 = torch.linalg.vector_norm(q2, dim=-1, keepdim=True).clamp_min(1e-30)
    q1n = q1 / n1
    q2n = q2 / n2
    # cos(θ/2) = |<q1, q2>|
    dot = torch.abs(torch.sum(q1n * q2n, dim=-1))
    dot = torch.clamp(dot, max=1.0)
    half_angle = torch.acos(dot)
    return torch.rad2deg(2.0 * half_angle)


def build_neighbor_laplacian(mask: np.ndarray) -> coo_matrix:
    """Construct the 4-neighbour graph Laplacian over an indexed-pixel mask.

    Parameters
    ----------
    mask : (H, W) bool array — True where pixel is indexed/refined.

    Returns
    -------
    L : (N, N) scipy sparse COO matrix where N = mask.sum().
        L[i, i] = degree of pixel i (count of indexed 4-neighbours)
        L[i, j] = -1 if pixels i, j are 4-neighbours and both indexed
    """
    H, W = mask.shape
    n = int(mask.sum())
    # Map (r, c) → flat index over indexed pixels only
    flat_id = np.full((H, W), -1, dtype=np.int64)
    flat_id[mask] = np.arange(n, dtype=np.int64)

    rows = []
    cols = []
    data = []
    degree = np.zeros(n, dtype=np.int64)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for r in range(H):
            r2 = r + dr
            if r2 < 0 or r2 >= H:
                continue
            for c in range(W):
                c2 = c + dc
                if c2 < 0 or c2 >= W:
                    continue
                if not mask[r, c] or not mask[r2, c2]:
                    continue
                i = flat_id[r, c]
                j = flat_id[r2, c2]
                rows.append(i); cols.append(j); data.append(-1.0)
                degree[i] += 1
    # Diagonal
    for i in range(n):
        rows.append(i); cols.append(i); data.append(float(degree[i]))
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
