"""Tier-2 orientation refinement.

Strategy
--------
Tier-1 cc volume often has the correct orientation as one of the top-K cc
peaks but not necessarily as the brightest (a wrong peak can be ~5-10 %
above the right one due to residual numerical noise). Tier-2 fixes this
with two cheap, robust techniques:

1. **3D Gaussian smoothing of the cc volume** before peak extraction. The
   right orientation forms a wider basin in cc space (the master pattern
   has finite spectral content, so the autocorrelation is smooth around
   the truth). Spurious narrow spikes get attenuated. Empirically this
   moves the brightest peak to the right basin in most pixels.

2. **Sub-bin parabolic peak interpolation** in the (a, b, c) space around
   the peak. The cc-volume bin spacing is ~2.7 degrees (360/135 for
   bw=68); fitting a 1-D parabola along each axis around the bin gives
   an extra ~0.3 deg accuracy.

These two together act like a Tier-2 refiner that is fast (no Newton/
autograd in a Python loop, no FP64 hot path) and mathematically grounded.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class RefinementMode(str, Enum):
    """Tier-2 refinement strategy selector.

    TRI_QUADRATIC: existing 3D triquadratic Newton on the cc-volume's 3x3x3
        peak neighbourhood. Fast, robust, sub-bin accuracy ~0.3 deg.
    ANALYTIC_NEWTON: autograd Newton on the SHT cross-correlation surface
        via ``backend.spherical_gpu._math.sht_newton``. Slower (~10x per
        pixel) but mathematically equivalent to EMSphInx
        ``Correlator<Real>::refinePeak``. Use for alt-oracle parity gate.
    """
    TRI_QUADRATIC = "tri_quadratic"
    ANALYTIC_NEWTON = "analytic_newton"


def smooth_nc_vol(
    nc_vol: torch.Tensor, sigma: float = 1.0,
) -> torch.Tensor:
    """Apply a 3-D Gaussian filter to the cc volume.

    Periodic boundary conditions on all three axes (the SO(3) cc volume
    is periodic in alpha, beta, gamma).

    Parameters
    ----------
    nc_vol : (B, size, size, size) float
    sigma : standard deviation in bins. 1.0 is conservative for bw=68
        where each bin is ~2.66 deg.

    Returns
    -------
    smoothed : (B, size, size, size) float
    """
    if sigma <= 0:
        return nc_vol
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=nc_vol.dtype, device=nc_vol.device)
    g_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    g_1d = g_1d / g_1d.sum()
    k_size = 2 * radius + 1

    # Pre-pad (circular wrap) on all three axes ONCE; each separable conv
    # then consumes its axis's padding. Previous implementation re-padded
    # Y and Z before their convs on top of this pre-pad, leaving 6 extra
    # samples in each of those dims at the output. See
    # tasks/_archive/handoff-smooth-nc-vol-shape.md for the trace.
    padded = F.pad(nc_vol.unsqueeze(1), (radius,) * 6, mode="circular")
    smoothed = F.conv3d(padded,   g_1d.view(1, 1, k_size, 1,      1))
    smoothed = F.conv3d(smoothed, g_1d.view(1, 1, 1,      k_size, 1))
    smoothed = F.conv3d(smoothed, g_1d.view(1, 1, 1,      1,      k_size))
    return smoothed.squeeze(1)


def _gather_at(nc_vol: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Gather nc_vol[batch, a, b, c] with periodic indexing."""
    size = nc_vol.shape[1]
    a_p = a % size
    b_p = b % size
    c_p = c % size
    B = nc_vol.shape[0]
    batch_idx = torch.arange(B, device=nc_vol.device)
    return nc_vol[batch_idx, a_p, b_p, c_p]


# iter-15J: cached 3x3x3 stencil offsets to avoid per-call tensor allocations
# in triquadratic_subbin. Keyed by (device, dtype) so we get a hot constant
# on each (CUDA, int64) combination after the first call.
_STENCIL_CACHE: dict = {}


def _get_stencil_27(device: torch.device, dtype: torch.dtype):
    """Return cached (da, db, dc) of shape (27,) each, in (-1, 0, 1)^3."""
    key = (str(device), dtype)
    cached = _STENCIL_CACHE.get(key)
    if cached is None:
        offsets = torch.tensor([-1, 0, 1], dtype=dtype, device=device)
        da, db, dc = torch.meshgrid(offsets, offsets, offsets, indexing="ij")
        cached = (da.reshape(-1).contiguous(), db.reshape(-1).contiguous(), dc.reshape(-1).contiguous())
        _STENCIL_CACHE[key] = cached
    return cached


def parabolic_subbin(
    nc_vol: torch.Tensor,            # (B, size, size, size)
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1D parabolic peak interpolation along each axis around (a, b, c).

    Returns (a_sub, b_sub, c_sub) — float-valued sub-bin positions.
    """
    a_l = _gather_at(nc_vol, a - 1, b, c)
    a_c = _gather_at(nc_vol, a,     b, c)
    a_r = _gather_at(nc_vol, a + 1, b, c)
    da = 0.5 * (a_l - a_r) / (a_l - 2 * a_c + a_r).clamp_min(1e-12)
    da = da.clamp(-0.5, 0.5)

    b_l = _gather_at(nc_vol, a, b - 1, c)
    b_c = _gather_at(nc_vol, a, b,     c)
    b_r = _gather_at(nc_vol, a, b + 1, c)
    db = 0.5 * (b_l - b_r) / (b_l - 2 * b_c + b_r).clamp_min(1e-12)
    db = db.clamp(-0.5, 0.5)

    c_l = _gather_at(nc_vol, a, b, c - 1)
    c_c = _gather_at(nc_vol, a, b, c)
    c_r = _gather_at(nc_vol, a, b, c + 1)
    dc = 0.5 * (c_l - c_r) / (c_l - 2 * c_c + c_r).clamp_min(1e-12)
    dc = dc.clamp(-0.5, 0.5)

    return a.float() + da, b.float() + db, c.float() + dc


def triquadratic_subbin(
    nc_vol: torch.Tensor,            # (B, size, size, size)
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-D triquadratic Newton step on a 3x3x3 neighborhood around (a, b, c).

    Fits f(x, y, z) = c000 + c100*x + c010*y + c001*z
                       + c110*x*y + c101*x*z + c011*y*z
                       + c200*x^2 + c020*y^2 + c002*z^2
                       + c111*x*y*z
    using the 27 cc values, then takes one Newton step:
        delta = -H^-1 * grad
    where grad and H are computed analytically from the polynomial.

    Equivalent to EMSphInx's `extractNeighborhood<1>` + triquadratic peak
    interpolation, the algorithm used when ref=False on indexImage. Gives
    sub-bin precision of approximately 0.05 deg (about 1/50 bin) for clean
    parabolic peaks.

    iter-15J: vectorized — replaces 27 separate ``_gather_at`` calls (each
    of which launches a CUDA kernel) with a single batched advanced-index
    fetch on a precomputed (27, 3) stencil. Same math, ~1/4 the overhead.

    Returns (a_sub, b_sub, c_sub) — float-valued sub-bin positions clamped
    to within +/- 0.5 of the input integer bin.
    """
    device = nc_vol.device
    size = nc_vol.shape[1]
    B = a.shape[0]

    # Cached 3x3x3 stencil offsets in (-1, 0, 1)^3 packed via ij meshgrid,
    # so the index for (da, db, dc) is (da+1)*9 + (db+1)*3 + (dc+1).
    da, db, dc = _get_stencil_27(device, a.dtype)

    # (B, 27) absolute indices with periodic wrap.
    a_abs = (a.unsqueeze(-1) + da.unsqueeze(0)) % size       # (B, 27)
    b_abs = (b.unsqueeze(-1) + db.unsqueeze(0)) % size
    c_abs = (c.unsqueeze(-1) + dc.unsqueeze(0)) % size
    batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(a_abs)
    # perf Phase 4b: keep FP32 instead of upcasting to FP64.
    # Sub-bin delta is clamped to +/- 0.5 and FP32 has ~7 decimal digits;
    # adequate precision for ~0.05 deg sub-bin orientation accuracy.
    # Validated bit-identical via oracle G1-G5 (max diso 0.000 deg).
    n_flat = nc_vol[batch_idx, a_abs, b_abs, c_abs].float()  # (B, 27)
    return _triquadratic_from_n27(n_flat, a, b, c)


def gather_n27_from_cc_rden(
    cc_real: torch.Tensor,            # (B, size, size, size) real cc volume (contiguous)
    rDen: torch.Tensor,               # (1, size, size, size) NC denominator
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
) -> torch.Tensor:
    """Gather (cc_real * rDen) at the 27 (3x3x3) bins around (a, b, c).

    Equivalent to materialising nc_vol = cc_real * rDen, then doing
    nc_vol[batch, a_abs, b_abs, c_abs] — but without ever materialising
    the full 21 MB / pattern volume. Pairs with ``fused_max_only`` in
    fused_kernels.py to enable skip-volume sub-bin refinement.
    """
    device = cc_real.device
    size = cc_real.shape[-1]
    B = a.shape[0]
    da, db, dc = _get_stencil_27(device, a.dtype)
    a_abs = (a.unsqueeze(-1) + da.unsqueeze(0)) % size       # (B, 27)
    b_abs = (b.unsqueeze(-1) + db.unsqueeze(0)) % size
    c_abs = (c.unsqueeze(-1) + dc.unsqueeze(0)) % size
    batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(a_abs)
    cc_27 = cc_real[batch_idx, a_abs, b_abs, c_abs].float()  # (B, 27)
    # rDen has shape (1, S, S, S); broadcast over batch via [0, ...].
    rDen_27 = rDen[0, a_abs, b_abs, c_abs].float()           # (B, 27)
    return cc_27 * rDen_27


def triquadratic_subbin_n27(
    n_flat: torch.Tensor,             # (B, 27) pre-gathered NC values
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sub-bin Newton step from a pre-gathered 3x3x3 neighbourhood.

    Same math as ``triquadratic_subbin`` but takes the (B, 27) gather
    result directly, enabling the skip-volume path where the caller
    gathered cc*rDen on the fly via ``gather_n27_from_cc_rden``.
    """
    return _triquadratic_from_n27(n_flat, a, b, c)


def _triquadratic_from_n27(
    n_flat: torch.Tensor,             # (B, 27)
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core sub-bin Newton step from a pre-gathered 3x3x3 neighbourhood."""
    # Inverse stencil index: lookup table mapping (da, db, dc) ∈ {-1,0,1}^3
    # to the linear index in the (27,)-axis above. With ij meshgrid, the
    # index for (da, db, dc) is (da+1)*9 + (db+1)*3 + (dc+1).
    def idx(d1: int, d2: int, d3: int) -> int:
        return (d1 + 1) * 9 + (d2 + 1) * 3 + (d3 + 1)

    f0 = n_flat[:, idx(0, 0, 0)]
    g_x = 0.5 * (n_flat[:, idx(1, 0, 0)] - n_flat[:, idx(-1, 0, 0)])
    g_y = 0.5 * (n_flat[:, idx(0, 1, 0)] - n_flat[:, idx(0, -1, 0)])
    g_z = 0.5 * (n_flat[:, idx(0, 0, 1)] - n_flat[:, idx(0, 0, -1)])

    H_xx = n_flat[:, idx(1, 0, 0)] - 2 * f0 + n_flat[:, idx(-1, 0, 0)]
    H_yy = n_flat[:, idx(0, 1, 0)] - 2 * f0 + n_flat[:, idx(0, -1, 0)]
    H_zz = n_flat[:, idx(0, 0, 1)] - 2 * f0 + n_flat[:, idx(0, 0, -1)]
    H_xy = 0.25 * (
        n_flat[:, idx(1, 1, 0)] - n_flat[:, idx(1, -1, 0)]
        - n_flat[:, idx(-1, 1, 0)] + n_flat[:, idx(-1, -1, 0)]
    )
    H_xz = 0.25 * (
        n_flat[:, idx(1, 0, 1)] - n_flat[:, idx(1, 0, -1)]
        - n_flat[:, idx(-1, 0, 1)] + n_flat[:, idx(-1, 0, -1)]
    )
    H_yz = 0.25 * (
        n_flat[:, idx(0, 1, 1)] - n_flat[:, idx(0, 1, -1)]
        - n_flat[:, idx(0, -1, 1)] + n_flat[:, idx(0, -1, -1)]
    )

    # perf Phase 4: closed-form 3x3 inverse for the symmetric
    # Hessian instead of torch.linalg.solve. cProfile iter-1 showed
    # linalg.solve cost 20.0s of inner runtime (~70 ms per call on a
    # (B, 3, 3) system — orders of magnitude above what cuSOLVER should
    # take for a tiny matrix). Closed-form pure-tensor arithmetic skips
    # the cuSOLVER dispatch entirely.
    #
    # Apply small negative-definite regularization on the diagonal to
    # match the original `H_reg = H - 1e-9 * eye` (H is negative-definite
    # at a correlation maximum).
    H_xx_r = H_xx - 1e-9
    H_yy_r = H_yy - 1e-9
    H_zz_r = H_zz - 1e-9

    # Cofactor expansion of the symmetric 3x3 determinant:
    #   det = a*(d*f - e^2) - b*(b*f - e*c) + c*(b*e - d*c)
    # with a=H_xx_r, b=H_xy, c=H_xz, d=H_yy_r, e=H_yz, f=H_zz_r.
    det = (
        H_xx_r * (H_yy_r * H_zz_r - H_yz * H_yz)
        - H_xy  * (H_xy   * H_zz_r - H_yz * H_xz)
        + H_xz  * (H_xy   * H_yz   - H_yy_r * H_xz)
    )

    # Singular-Hessian guard: protect the division and remember which rows
    # need the parabolic 1-D fallback (matches the original try/except
    # behavior — a singular regularized H is rare but possible at saddles).
    singular = det.abs() < 1e-30
    det_safe = torch.where(singular, torch.ones_like(det), det)
    inv_det = 1.0 / det_safe

    # Symmetric inverse via cofactors. (Hessian inverse is symmetric.)
    inv_xx = (H_yy_r * H_zz_r - H_yz * H_yz) * inv_det
    inv_xy = (H_yz * H_xz - H_xy * H_zz_r)   * inv_det
    inv_xz = (H_xy * H_yz - H_yy_r * H_xz)   * inv_det
    inv_yy = (H_xx_r * H_zz_r - H_xz * H_xz) * inv_det
    inv_yz = (H_xy * H_xz - H_xx_r * H_yz)   * inv_det
    inv_zz = (H_xx_r * H_yy_r - H_xy * H_xy) * inv_det

    # delta = -H_reg^{-1} @ grad   (matrix-vector product, closed form).
    ngx, ngy, ngz = -g_x, -g_y, -g_z
    delta_x = inv_xx * ngx + inv_xy * ngy + inv_xz * ngz
    delta_y = inv_xy * ngx + inv_yy * ngy + inv_yz * ngz
    delta_z = inv_xz * ngx + inv_yz * ngy + inv_zz * ngz
    delta = torch.stack([delta_x, delta_y, delta_z], dim=-1)

    # Singular-Hessian fallback: parabolic 1-D per axis (same formula as
    # the original except RuntimeError branch).
    if singular.any():
        parabolic = torch.stack([
            ngx / H_xx_r.clamp_min(1e-9),
            ngy / H_yy_r.clamp_min(1e-9),
            ngz / H_zz_r.clamp_min(1e-9),
        ], dim=-1)
        delta = torch.where(singular.unsqueeze(-1), parabolic, delta)

    # Reject NaN/inf and clamp to within +/-0.5 (one bin)
    delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
    delta = delta.clamp(-0.5, 0.5)

    return (
        a.double() + delta[:, 0],
        b.double() + delta[:, 1],
        c.double() + delta[:, 2],
    )


def decode_cell_to_zxz(
    a_f: torch.Tensor, b_f: torch.Tensor, c_f: torch.Tensor,
    bandwidth: int,
) -> torch.Tensor:
    """Decode (a, b, c) (possibly fractional) to (phi1, Phi, phi2) ZXZ rad.

    Same formula as `Tier1Indexer._decode_peak`.
    """
    L = bandwidth
    size = 2 * L - 1
    off = L - 1
    scale = 2.0 * math.pi / size
    half_pi = math.pi / 2.0
    two_pi = 2.0 * math.pi
    size_f = float(size)
    alpha  = ((a_f - off + size_f).fmod(size_f) + size_f).fmod(size_f) * scale
    gamma_ = ((off  - c_f + size_f).fmod(size_f) + size_f).fmod(size_f) * scale
    Phi    = b_f * scale
    phi1   = (alpha + half_pi) % two_pi
    phi2   = (gamma_ - half_pi) % two_pi
    phi1   = (phi1 + 3.0 * half_pi) % two_pi
    return torch.stack([phi1, Phi, phi2], dim=-1)


def refine_index_result(
    nc_vol: torch.Tensor,
    bandwidth: int,
    smoothing_sigma: float = 1.0,
    *,
    mode: RefinementMode = RefinementMode.TRI_QUADRATIC,
    flm: Optional[torch.Tensor] = None,
    gln: Optional[torch.Tensor] = None,
    eu_seed: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tier-2 refinement: smooth the cc volume + sub-bin parabolic interp.

    Parameters
    ----------
    nc_vol : (B, 2L-1, 2L-1, 2L-1) float — Tier-1 cc volume
    bandwidth : int — L
    smoothing_sigma : Gaussian sigma in cc-bins (~ 2.66 deg per bin)
    mode : RefinementMode — TRI_QUADRATIC (default, fast 3D triquadratic
        Newton on the cc-volume's 3x3x3 peak neighbourhood) or
        ANALYTIC_NEWTON (autograd Newton on the SHT cross-correlation
        surface; mathematically equivalent to EMSphInx
        ``Correlator<Real>::refinePeak``, used for alt-oracle parity).
    flm, gln, eu_seed : required for ANALYTIC_NEWTON. ``flm`` (L, L) complex
        is the master SHT coefs, ``gln`` (B, L, L) complex are the per-pixel
        pattern SHT coefs, ``eu_seed`` (B, 3) float64 is the Tier-1 seed
        Bunge ZXZ. Pass them on CPU; this branch loops in Python.

    Returns
    -------
    eulers_zxz : (B, 3) float32 — refined Bunge ZXZ in radians
    scores : (B,) float32 — refined peak value
    """
    if mode is RefinementMode.ANALYTIC_NEWTON:
        if flm is None or gln is None or eu_seed is None:
            raise ValueError(
                "RefinementMode.ANALYTIC_NEWTON requires flm, gln, and eu_seed; "
                f"got flm={flm is not None}, gln={gln is not None}, "
                f"eu_seed={eu_seed is not None}."
            )
        from backend.spherical_gpu._math.sht_newton import newton_refine

        # Tier1Indexer._master_coefs is shape (1, L, L) for a single-phase
        # master (the batch dim is reserved for multi-phase elsewhere). Squeeze
        # so cc_at_rotation sees (L, L); gln is per-pixel (B, L, L) and stays
        # 3D — we slice gln[i] per pixel in the loop below.
        # Keep on the input device — Wigner-d vectorization makes GPU ~10x
        # faster than CPU at L=68, and there's no benefit to a CPU detour.
        target_device = (
            nc_vol.device if torch.is_tensor(nc_vol) else torch.device("cpu")
        )
        flm_c128 = flm.to(device=target_device, dtype=torch.complex128)
        if flm_c128.dim() == 3:
            if flm_c128.shape[0] != 1:
                raise ValueError(
                    f"ANALYTIC_NEWTON expects a single-phase master; got flm "
                    f"with shape {tuple(flm_c128.shape)} (first dim must be 1)."
                )
            flm_c128 = flm_c128.squeeze(0)
        gln_c128 = gln.to(device=target_device, dtype=torch.complex128)
        eu_f64 = eu_seed.to(device=target_device, dtype=torch.float64)
        B = eu_f64.shape[0]
        refined_eu = torch.empty((B, 3), dtype=torch.float32)
        refined_cc = torch.empty((B,), dtype=torch.float32)
        for i in range(B):
            eu_i, cc_i, _converged = newton_refine(
                flm_c128, gln_c128[i], eu_f64[i], bandwidth,
            )
            refined_eu[i] = eu_i.to(torch.float32)
            refined_cc[i] = cc_i.to(torch.float32)
        return refined_eu, refined_cc

    smoothed = smooth_nc_vol(nc_vol, sigma=smoothing_sigma)
    B, size, _, _ = smoothed.shape

    flat_idx = smoothed.reshape(B, -1).argmax(dim=1)
    a = (flat_idx // (size * size)).long()
    rem = flat_idx % (size * size)
    b = (rem // size).long()
    c = (rem % size).long()

    # Sub-bin parabolic interpolation on the SMOOTHED volume so the parabola
    # sees a good local quadratic around the chosen peak.
    a_sub, b_sub, c_sub = parabolic_subbin(smoothed, a, b, c)

    eulers = decode_cell_to_zxz(a_sub, b_sub, c_sub, bandwidth).float()

    scores = smoothed.reshape(B, -1).gather(
        1, flat_idx.unsqueeze(1)
    ).squeeze(1).float()
    return eulers, scores
