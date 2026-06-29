"""SO(3) cross-correlation evaluated directly at FZ quaternion grid points.

Production-oriented evaluator: ``rs2cc_fz_torch`` accepts a per-phase spectrum
F[l, m, n] and a fundamental-zone quaternion grid (K quats), returns
cc(R_k) = Re sum_{l,m,n} F[l,m,n] * D^l_{mn}(R_k) for each k.

Implementation strategy (M2.2 — chunked batched matrix_exp):
- Loop over band j in [0, L). For each j, build the (2j+1) × (2j+1) Wigner-D
  matrix evaluated at all K_chunk quaternions in one batched call to
  ``torch.matrix_exp`` (e3nn-style), then einsum-reduce against the F sub-block.
- Loop over K in chunks of CHUNK_SIZE to bound peak memory.

Cost at L=88: matrix_exp on (K_chunk, 2j+1, 2j+1) is O(j^3) per quat with
heavy constants. Slower than rs2cc_sparse at L=88 — but correctness-first.
M2.3 will replace matrix_exp with the batched Fukushima recursion for
production speed.

Vendored 2026-05-09 for GPU spherical-indexing perf C1c milestone M2.2.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .wigner_d_eval import _so3_generators, quat_to_zyz_euler


# ---------------------------------------------------------------------------
# Batched Wigner-D at a single band j over K quaternions
# ---------------------------------------------------------------------------

def _batched_wigner_D_at_band(
    j: int,
    alpha: Tensor,
    beta: Tensor,
    gamma: Tensor,
    dtype: torch.dtype = torch.complex128,
) -> Tensor:
    """Build (K, 2j+1, 2j+1) Wigner-D matrices in one matrix_exp call per axis.

    D^j(α, β, γ) = exp(-i α Lz) · exp(-i β Ly) · exp(-i γ Lz)

    Args:
        j:     band index (>= 0).
        alpha: (K,) ZYZ-Euler alpha angles.
        beta:  (K,) ZYZ-Euler beta angles.
        gamma: (K,) ZYZ-Euler gamma angles.
        dtype: complex dtype.

    Returns:
        (K, 2j+1, 2j+1) complex tensor of Wigner-D matrices.
    """
    device = alpha.device
    K = alpha.shape[0]
    if j == 0:
        # Trivial case: D^0(R) = [[1.0]] for any R.
        return torch.ones((K, 1, 1), dtype=dtype, device=device)

    Lx, Ly, Lz = _so3_generators(j, dtype=dtype)
    Lx, Ly, Lz = Lx.to(device), Ly.to(device), Lz.to(device)

    # Each factor is matrix_exp(-i * angle * generator), batched over K.
    a = alpha.to(dtype=dtype).view(K, 1, 1)
    b = beta.to(dtype=dtype).view(K, 1, 1)
    g = gamma.to(dtype=dtype).view(K, 1, 1)

    # Note: -1j * angle * generator yields broadcast-safe (K, dim, dim).
    Ma = torch.matrix_exp(-1j * a * Lz.unsqueeze(0))
    Mb = torch.matrix_exp(-1j * b * Ly.unsqueeze(0))
    Mc = torch.matrix_exp(-1j * g * Lz.unsqueeze(0))
    return Ma @ Mb @ Mc


# ---------------------------------------------------------------------------
# Main entry point: rs2cc_fz_torch
# ---------------------------------------------------------------------------

def rs2cc_fz_torch(
    spectrum: Tensor,
    fz_quats: Tensor,
    L: int,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.complex128,
) -> Tensor:
    """SO(3) cross-correlation evaluated directly on an FZ quaternion grid.

    cc[..., k] = Re sum_{l,m,n} F[..., l, m+L-1, n+L-1] * D^l_{mn}(fz_quats[k])

    Args:
        spectrum: (..., L, 2L-1, 2L-1) complex tensor, F[l, m+L-1, n+L-1].
            Bands l > L-1 not represented; entries with |m| > l or |n| > l
            are ignored (Wigner-D sub-block has size 2l+1 not 2L-1).
        fz_quats: (K, 4) unit quaternions on the fundamental zone.
        L: bandwidth (1 + max degree).
        chunk_size: number of quaternions per matrix_exp call. Tuned for
            12 GB GPU at L=88: 64 keeps peak memory < 2 GB per chunk.
        dtype: complex internal dtype (complex128 for accuracy, complex64
            for speed).

    Returns:
        (..., K) float32 cc values.
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
    if fz_quats.shape[-1] != 4:
        raise ValueError(f"fz_quats.shape[-1] must be 4, got {fz_quats.shape[-1]}")

    device = spectrum.device
    K = fz_quats.shape[0]
    batch_shape = spectrum.shape[:-3]

    # Reshape spectrum to (B, L, 2L-1, 2L-1)
    if batch_shape == ():
        spectrum_flat = spectrum.reshape(1, L, 2 * L - 1, 2 * L - 1)
    else:
        B = int(torch.prod(torch.tensor(batch_shape)).item())
        spectrum_flat = spectrum.reshape(B, L, 2 * L - 1, 2 * L - 1)
    spectrum_flat = spectrum_flat.to(dtype=dtype, device=device)

    # ZYZ Euler from quaternions
    alpha, beta, gamma = quat_to_zyz_euler(fz_quats.to(device=device))
    alpha = alpha.to(dtype=torch.float64)
    beta = beta.to(dtype=torch.float64)
    gamma = gamma.to(dtype=torch.float64)

    cc_out = torch.zeros((spectrum_flat.shape[0], K), dtype=dtype, device=device)

    for k_start in range(0, K, chunk_size):
        k_end = min(k_start + chunk_size, K)
        a_chunk = alpha[k_start:k_end]
        b_chunk = beta[k_start:k_end]
        g_chunk = gamma[k_start:k_end]

        cc_chunk = torch.zeros(
            (spectrum_flat.shape[0], k_end - k_start), dtype=dtype, device=device
        )

        for j in range(L):
            # Build batched Wigner-D matrix for this band
            D_j = _batched_wigner_D_at_band(j, a_chunk, b_chunk, g_chunk, dtype=dtype)
            # F sub-block at this band: (B, 2j+1, 2j+1)
            offset = (L - 1) - j
            F_j = spectrum_flat[
                :, j, offset : offset + 2 * j + 1, offset : offset + 2 * j + 1
            ]
            # Contract: cc_chunk[b, k] += sum_{m, n} F_j[b, m, n] * D_j[k, m, n]
            cc_chunk = cc_chunk + torch.einsum("bmn,kmn->bk", F_j, D_j)

        cc_out[:, k_start:k_end] = cc_chunk

    cc_real = cc_out.real.to(dtype=torch.float32)
    if batch_shape == ():
        return cc_real.squeeze(0)
    return cc_real.reshape(batch_shape + (K,))


__all__ = [
    "rs2cc_fz_torch",
    "_batched_wigner_D_at_band",
    "rs2cc_fz_torch_fukushima",
]


# Module-level cache for unpack tables, keyed on (L, device.type).
_UNPACK_TABLE_CACHE: dict = {}


def _get_cached_unpack_tables(
    L: int, device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (ss_idx, ss_sign, ms_idx, ms_sign) tables, building once per
    (L, device.type). The build is O(L^3) Python triple-loop and is the
    dominant overhead for short rs2cc calls.
    """
    from .wigner_d_unpack import (
        build_unpack_index_and_sign_tables, build_mixed_sign_tables,
    )
    key = (L, device.type)
    if key not in _UNPACK_TABLE_CACHE:
        ss_idx, ss_sign = build_unpack_index_and_sign_tables(L, device)
        ms_idx, ms_sign = build_mixed_sign_tables(L, device)
        _UNPACK_TABLE_CACHE[key] = (ss_idx, ss_sign, ms_idx, ms_sign)
    return _UNPACK_TABLE_CACHE[key]


# ---------------------------------------------------------------------------
# M2.3a-integration: Fukushima-recursion-based rs2cc_fz_torch
# ---------------------------------------------------------------------------

def rs2cc_fz_torch_fukushima(
    spectrum: Tensor,
    fz_quats: Tensor,
    L: int,
    chunk_size: int = 256,
    dtype: torch.dtype = torch.complex128,
    use_compile: bool = True,
) -> Tensor:
    """Production-fast SO(3) cross-correlation on FZ quaternion grid.

    Same API as ``rs2cc_fz_torch`` but uses the batched Fukushima recursion
    + unpack_wigner_d_full (M2.3a) instead of per-band matrix_exp. ~100×
    faster at L=88 K=30k (matrix_exp cost was launch-bound).

    Implementation:
      1. Convert fz_quats → (alpha, beta, gamma) batched.
      2. For each chunk of K quats:
         a. Compute packed Wigner-d at beta and (pi-beta) via batched
            Fukushima (lt_half_pi for beta < pi/2, gt_half_pi for >).
         b. unpack_wigner_d_full → dense (K_chunk, L, 2L-1, 2L-1).
         c. Apply alpha/gamma phase factors (complex64).
         d. einsum reduction with spectrum.

    Args:
        spectrum: (..., L, 2L-1, 2L-1) complex F[l, m+L-1, n+L-1].
        fz_quats: (K, 4) unit quaternions, w >= 0.
        L: bandwidth.
        chunk_size: K chunk for memory budget. At L=88, chunk=256 → 121 MB.
        dtype: complex internal dtype.

    Returns:
        (..., K) float32 cc values.
    """
    import math
    from .wigner_d_torch_compile import (
        wigner_d_lt_half_pi_batched_eager,
        wigner_d_gt_half_pi_batched_eager,
        wigner_d_lt_half_pi_batched_compiled,
    )
    # Pick eager vs compiled (compile gives ~1.6x speedup at L=44 K=2139,
    # ~22s first-compile cost amortizes over many calls).
    lt_fn = (
        wigner_d_lt_half_pi_batched_compiled if use_compile
        else wigner_d_lt_half_pi_batched_eager
    )
    gt_fn = wigner_d_gt_half_pi_batched_eager  # not yet compiled
    from .wigner_d_unpack import (
        build_unpack_index_and_sign_tables,
        build_mixed_sign_tables,
        unpack_wigner_d_full,
    )

    if spectrum.shape[-3] != L:
        raise ValueError(
            f"spectrum.shape[-3] must equal L={L}, got {spectrum.shape[-3]}"
        )
    if spectrum.shape[-2] != 2 * L - 1 or spectrum.shape[-1] != 2 * L - 1:
        raise ValueError(
            f"spectrum.shape[-2:] must be ({2*L-1}, {2*L-1}), "
            f"got {spectrum.shape[-2:]}"
        )
    if fz_quats.shape[-1] != 4:
        raise ValueError(f"fz_quats.shape[-1] must be 4, got {fz_quats.shape[-1]}")

    device = spectrum.device
    K = fz_quats.shape[0]
    batch_shape = spectrum.shape[:-3]

    if batch_shape == ():
        spectrum_flat = spectrum.reshape(1, L, 2 * L - 1, 2 * L - 1)
    else:
        B = int(torch.prod(torch.tensor(batch_shape)).item())
        spectrum_flat = spectrum.reshape(B, L, 2 * L - 1, 2 * L - 1)
    spectrum_flat = spectrum_flat.to(dtype=dtype, device=device)

    alpha, beta, gamma = quat_to_zyz_euler(fz_quats.to(device=device))
    alpha = alpha.to(dtype=torch.float64)
    beta = beta.to(dtype=torch.float64)
    gamma = gamma.to(dtype=torch.float64)

    # Build unpack tables once. CACHED across calls to avoid the O(L^3)
    # Python triple-loop in build_*_tables (~333k iterations at L=44 ≈
    # 5 sec/call without cache). Profiled: cache miss adds ~10 sec to
    # the L=44 K=2139 path; cache hit is ~10 us.
    ss_idx, ss_sign, ms_idx, ms_sign = _get_cached_unpack_tables(L, device)

    # m, n integer indices for phase factors. Use FP32 to avoid the FP64
    # intermediate that doubles memory pressure. (Profiled: FP64 path was
    # 6x slower at L=44 K=2139 due to 1.36 GB intermediate vs 682 MB.)
    m_idx = torch.arange(-(L - 1), L, device=device, dtype=torch.float32)
    n_idx = torch.arange(-(L - 1), L, device=device, dtype=torch.float32)

    cc_out = torch.zeros((spectrum_flat.shape[0], K), dtype=dtype, device=device)

    for k_start in range(0, K, chunk_size):
        k_end = min(k_start + chunk_size, K)
        a_chunk = alpha[k_start:k_end]
        b_chunk = beta[k_start:k_end]
        g_chunk = gamma[k_start:k_end]

        # Batched Wigner-d at beta and (pi - beta).
        # Need to handle both beta < pi/2 and beta > pi/2 cases by routing
        # to the appropriate function. For simplicity (and the common case
        # where most FZ betas are interior to [0, pi]), call lt at clamped-
        # below and gt at clamped-above. Eq case (beta == pi/2) is exact in
        # both; we use lt for it.
        # Clamp beta to (eps, pi-eps) to avoid log(sin(0/2)) = -inf in
        # the Fukushima recurrence. At beta ~= 0 (and ~= pi) the Wigner-d
        # is essentially identity (or anti-identity); the eps clamp gives
        # a value within FP32 epsilon of identity, harmless for cc.
        EPS = 1e-6
        b_chunk = b_chunk.clamp(min=EPS, max=math.pi - EPS)
        is_lt = b_chunk < math.pi / 2.0
        # For beta < pi/2: use lt(beta) and gt(pi - beta)  (since pi-beta > pi/2)
        # For beta >= pi/2: use gt(beta) and lt(pi - beta)
        # Combine into a single (K_chunk, table_size) "packed_at_beta" and
        # "packed_at_pi_minus_beta" via where-mask logic.

        # Strategy: compute lt at min(beta, pi-beta), gt at max(beta, pi-beta);
        # this way one always gets lt at < pi/2 and gt at > pi/2.
        b_min = torch.minimum(b_chunk, math.pi - b_chunk)  # always < pi/2
        b_max = torch.maximum(b_chunk, math.pi - b_chunk)  # always > pi/2

        packed_lt_at_min = lt_fn(b_min, L, torch.float64, device)  # (K_chunk, table_size)
        packed_gt_at_max = gt_fn(b_max, L, torch.float64, device)

        # For each quat, the "at beta" packed is from lt if beta < pi/2 else gt.
        # And "at pi-beta" is the opposite.
        is_lt_e = is_lt.unsqueeze(-1).to(packed_lt_at_min.dtype)
        packed_at_beta = torch.where(
            is_lt.unsqueeze(-1), packed_lt_at_min, packed_gt_at_max,
        )
        packed_at_pi_minus = torch.where(
            is_lt.unsqueeze(-1), packed_gt_at_max, packed_lt_at_min,
        )

        # Unpack to dense (K_chunk, L, 2L-1, 2L-1) real
        d_dense = unpack_wigner_d_full(
            packed_at_beta, packed_at_pi_minus,
            ss_idx, ss_sign, ms_idx, ms_sign,
        )  # FP64 real, shape (K_chunk, L, 2L-1, 2L-1)

        # OPTIMIZED PATH: cast d_dense to FP32 immediately, build FP32 phase
        # factors directly, multiply once into complex64 D. Avoids the FP64
        # intermediate that previously dominated memory bandwidth.
        # Profiled win at L=44 K=2139 B=32: 15.3 sec -> 2.6 sec (~6x).
        spectrum_target_dtype = spectrum_flat.dtype  # complex64 typical
        # Determine corresponding REAL dtype.
        if spectrum_target_dtype == torch.complex64:
            real_dtype = torch.float32
        elif spectrum_target_dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            real_dtype = torch.float32  # fallback

        d_dense_real = d_dense.to(real_dtype)  # (K_chunk, L, 2L-1, 2L-1) FP32
        a_chunk_real = a_chunk.to(real_dtype)
        g_chunk_real = g_chunk.to(real_dtype)
        # Phase factors directly in target complex dtype.
        phase_a = torch.exp(
            -1j * m_idx.view(1, 1, -1, 1) * a_chunk_real.view(-1, 1, 1, 1)
        ).to(spectrum_target_dtype)
        phase_g = torch.exp(
            -1j * n_idx.view(1, 1, 1, -1) * g_chunk_real.view(-1, 1, 1, 1)
        ).to(spectrum_target_dtype)
        D_complex = (
            d_dense_real.to(spectrum_target_dtype) * phase_a * phase_g
        )

        # Reduce: cc[b, k] = sum_{l, m, n} F[b, l, m, n] * D[k, l, m, n]
        cc_chunk = torch.einsum("blmn,klmn->bk", spectrum_flat, D_complex)
        cc_out[:, k_start:k_end] = cc_chunk

    cc_real = cc_out.real.to(dtype=torch.float32)
    if batch_shape == ():
        return cc_real.squeeze(0)
    return cc_real.reshape(batch_shape + (K,))
