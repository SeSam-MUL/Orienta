"""Direct evaluation of cc(R) = sum_{l,m,n} F[l,m,n] * D^l_{mn}(R)
at arbitrary quaternion grid points.

This is the M1.5 reference (Python-loop) implementation. The GPU-batched
production version is M2 work; this reference acts as the oracle for it.

Mathematical core: for unit quaternion R with ZYZ Euler decomposition
(alpha, beta, gamma):

    cc(R) = Re[ sum_{l,m,n} F[l,m,n] * exp(-i*m*alpha)
                                     * d^l_{mn}(beta)
                                     * exp(-i*n*gamma) ]

Wigner-D matrix construction here uses the e3nn-style approach:
``D^l(R) = exp(-i*alpha*Lz) * exp(-i*beta*Ly) * exp(-i*gamma*Lz)``
where ``Lx``, ``Ly``, ``Lz`` are the angular momentum operators (so(3)
generators) at degree ``l``, sized (2l+1) × (2l+1). Built via
``torch.matrix_exp``. Exact and self-contained, slow at large l (~O(l^4)
per matrix_exp). Suitable as an ORACLE for M2 testing — production M2
swaps in batched Fukushima recursion.

Vendored 2026-05-09 for GPU spherical-indexing perf C1c milestone M1.5.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Quaternion -> ZYZ Euler
# ---------------------------------------------------------------------------

def quat_to_zyz_euler(quats: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert unit quaternions (w, x, y, z) → ZYZ Euler (alpha, beta, gamma).

    Convention:  R = R_z(alpha) * R_y(beta) * R_z(gamma).

    This matches Lenthe 2019 eq. 11 and the standard Wigner-D factorization
    ``D^l_{mn} = exp(-i*m*alpha) d^l_{mn}(beta) exp(-i*n*gamma)``.

    Args:
        quats: (..., 4) unit quaternions, w >= 0.

    Returns:
        (alpha, beta, gamma) tuple of (...) tensors.
        - alpha, gamma in [0, 2*pi)
        - beta in [0, pi]
        - Gimbal lock at beta ≈ 0 or beta ≈ pi: alpha takes the full angular
          sum (2*atan2(z, w)), gamma is 0.
    """
    if quats.shape[-1] != 4:
        raise ValueError(f"quaternion last dim must be 4, got {quats.shape}")

    w, x, y, z = quats.unbind(dim=-1)

    # cos(beta) = R_zz = 1 - 2*(x^2 + y^2)
    cos_beta = (w * w - x * x - y * y + z * z).clamp(min=-1.0, max=1.0)
    beta = torch.arccos(cos_beta)

    # alpha = atan2(R_yz, R_xz),  gamma = atan2(R_zy, -R_zx)
    R_xz = 2.0 * (x * z + w * y)
    R_yz = 2.0 * (y * z - w * x)
    R_zx = 2.0 * (x * z - w * y)
    R_zy = 2.0 * (y * z + w * x)

    alpha = torch.atan2(R_yz, R_xz)
    gamma = torch.atan2(R_zy, -R_zx)

    # Gimbal lock at beta=0: pure z-rotation. Pick alpha = total z-angle, gamma = 0.
    # Threshold needs to be looser than 1e-10 because FP32 quats give beta ~1e-3
    # near the pole due to arccos(1 - eps) precision.
    near_pole = beta.abs() < 1e-3
    if near_pole.any():
        z_rot = 2.0 * torch.atan2(z, w)
        alpha = torch.where(near_pole, z_rot, alpha)
        gamma = torch.where(near_pole, torch.zeros_like(gamma), gamma)
        beta = torch.where(near_pole, torch.zeros_like(beta), beta)

    # Wrap alpha and gamma to [0, 2*pi)
    two_pi = 2.0 * torch.pi
    alpha = (alpha + two_pi) % two_pi
    gamma = (gamma + two_pi) % two_pi
    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# SO(3) generators -> Wigner-D via matrix exponential (oracle path)
# ---------------------------------------------------------------------------

def _so3_generators(l: int, dtype=torch.complex128) -> Tuple[Tensor, Tensor, Tensor]:
    """Angular-momentum operators (Lx, Ly, Lz) at degree l.

    Standard quantum-mechanical definitions in the |l, m> basis with
    m running from -l to +l. Returns three (2l+1, 2l+1) complex matrices.

    Reference: Sakurai "Modern Quantum Mechanics" Sec. 3.5; e3nn's
    `o3._wigner.so3_generators` (MIT) follows the same conventions.
    """
    dim = 2 * l + 1
    Lz = torch.zeros((dim, dim), dtype=dtype)
    Lp = torch.zeros((dim, dim), dtype=dtype)  # raising L+
    Lm = torch.zeros((dim, dim), dtype=dtype)  # lowering L-

    for m_idx in range(dim):
        m = m_idx - l
        Lz[m_idx, m_idx] = float(m)
        # L+ |l, m> = sqrt((l-m)(l+m+1)) |l, m+1>  -> matrix element (m+1, m)
        if m_idx + 1 < dim:
            mp = m + 1
            coef = ((l - m) * (l + m + 1)) ** 0.5
            Lp[m_idx + 1, m_idx] = coef
        # L- |l, m> = sqrt((l+m)(l-m+1)) |l, m-1>  -> matrix element (m-1, m)
        if m_idx - 1 >= 0:
            coef = ((l + m) * (l - m + 1)) ** 0.5
            Lm[m_idx - 1, m_idx] = coef

    Lx = 0.5 * (Lp + Lm)
    Ly = -0.5j * (Lp - Lm)
    return Lx, Ly, Lz


def wigner_D_matrix(
    l: int, alpha: float, beta: float, gamma: float,
    dtype=torch.complex128,
) -> Tensor:
    """Wigner-D matrix at degree l for ZYZ Euler triple (alpha, beta, gamma).

    D^l(alpha, beta, gamma) = exp(-i alpha Lz) * exp(-i beta Ly) * exp(-i gamma Lz)

    Returns a (2l+1, 2l+1) complex tensor, indexed by (m_out + l, m_in + l).
    Exact (within float precision) — this is the canonical reference used
    to validate the Fukushima recursion in M2.
    """
    Lx, Ly, Lz = _so3_generators(l, dtype=dtype)
    Da = torch.matrix_exp(-1j * alpha * Lz)
    Db = torch.matrix_exp(-1j * beta * Ly)
    Dc = torch.matrix_exp(-1j * gamma * Lz)
    return Da @ Db @ Dc


def evaluate_wigner_D_at_quaternions(
    spectrum: Tensor,
    quats: Tensor,
    L: int,
    dtype: torch.dtype = torch.complex128,
) -> Tensor:
    """REFERENCE oracle: cc(R) = sum_{l,m,n} F[l,m,n] * D^l_{mn}(R).

    Slow Python-loop over quaternions × bands. Uses matrix_exp generators
    for the Wigner-D matrices — exact within float precision, validated
    against e3nn / spherical.Wigner.D conventions.

    For PRODUCTION USE at L=88 K=30k, see M2 (batched Fukushima recursion).

    Args:
        spectrum: (..., L, 2L-1, 2L-1) complex. F[l, m+L-1, n+L-1] for
            m, n in [-(L-1), L-1]. Bands l > L-1 are not represented;
            entries with |m| > l or |n| > l are implicitly zero (Wigner-D
            kills them via the (2l+1, 2l+1) sub-block).
        quats: (K, 4) unit quaternions, w >= 0.
        L: bandwidth (degrees 0..L-1).
        dtype: complex internal dtype (complex128 default for accuracy).

    Returns:
        (..., K) float32 cc values (real part of the complex sum).
    """
    if spectrum.shape[-3] != L:
        raise ValueError(
            f"spectrum.shape[-3] must equal L={L}, got {spectrum.shape[-3]}"
        )
    if spectrum.shape[-2] != 2 * L - 1 or spectrum.shape[-1] != 2 * L - 1:
        raise ValueError(
            f"spectrum.shape[-2:] must be ({2*L-1}, {2*L-1}), "
            f"got {spectrum.shape[-2:]}"
        )
    if quats.shape[-1] != 4:
        raise ValueError(f"quats.shape[-1] must be 4, got {quats.shape[-1]}")

    device = spectrum.device
    K = quats.shape[0]
    batch_shape = spectrum.shape[:-3]

    # Reshape spectrum to (B, L, 2L-1, 2L-1)
    if batch_shape == ():
        spectrum_flat = spectrum.reshape(1, L, 2 * L - 1, 2 * L - 1)
    else:
        B = int(torch.prod(torch.tensor(batch_shape)).item())
        spectrum_flat = spectrum.reshape(B, L, 2 * L - 1, 2 * L - 1)
    spectrum_flat = spectrum_flat.to(dtype=dtype, device=device)

    alpha, beta, gamma = quat_to_zyz_euler(quats.to(device=device))

    out = torch.zeros((spectrum_flat.shape[0], K), dtype=torch.float64, device=device)

    for k in range(K):
        a_k = float(alpha[k].item())
        b_k = float(beta[k].item())
        g_k = float(gamma[k].item())

        # cc(R) = sum_l Tr_{m,n}( F[l, m, n] * D^l_{m, n}(R) )
        # We compute this band by band.
        cc_k = torch.zeros(spectrum_flat.shape[0], dtype=dtype, device=device)
        for j in range(L):
            D_j = wigner_D_matrix(j, a_k, b_k, g_k, dtype=dtype).to(device)
            # F sub-block at degree j: spectrum[..., j, m+L-1, n+L-1] for
            # m, n in [-j, j]. Slice and contract.
            offset = (L - 1) - j
            F_j = spectrum_flat[
                :, j, offset : offset + 2 * j + 1, offset : offset + 2 * j + 1
            ]  # (B, 2j+1, 2j+1)
            # element-wise sum, reduce both spatial dims
            cc_k = cc_k + (F_j * D_j.unsqueeze(0)).sum(dim=(1, 2))
        out[:, k] = cc_k.real

    out = out.to(dtype=torch.float32)
    if batch_shape == ():
        return out.squeeze(0)
    return out.reshape(batch_shape + (K,))


__all__ = [
    "quat_to_zyz_euler",
    "wigner_D_matrix",
    "evaluate_wigner_D_at_quaternions",
]
