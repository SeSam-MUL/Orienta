"""

This module contains the functions cross correlate two functions on the 2-sphere
over the group SO(3), using spherical harmonics. 

Notes for understanding the code:

- Symmetry-less real valued cross correlation volume: 
- (2L - 1) X (2L - 1) X (L) in the order of (n, k, m)
- PyTorch real FFT calls have trailing half dimension

These papers don't explicitly label the summation over m and n:

1) Gutman, Boris, et al. "Shape registration with spherical cross correlation."
   2nd MICCAI Workshop on Mathematical Foundations of Computational Anatomy.
   2008.

2) Sorgi, Lorenzo, and Kostas Daniilidis. "Template gradient matching in
   spherical images." Image Processing: Algorithms and Systems III. Vol. 5298.
   SPIE, 2004.

3) Hielscher, Ralf, Felix Bartel, and Thomas Benjamin Britton. "Gazing at
   crystal balls: Electron backscatter diffraction pattern analysis and cross
   correlation on the sphere." Ultramicroscopy 207 (2019): 112836.


so one might think m ranges from 0 to L and not -L to L.

But this reference in equation (6) explicitly states the range of m and n:

Carpentier, Thibaut, and Aaron Einbond. "Spherical correlation as a similarity
measure for 3-D radiation patterns of musical instruments." Acta Acustica 7
(2023): 40.

Then if you look at the EMSphInx code:

https://github.com/EMsoft-org/EMSphInx/blob/master/include/sht/sht_xcorr.hpp

One sees that m only ranges from 0 to L but the inverse FFT is done with an 
output spatial size of 2L - 1 nonetheless.


"""

import torch
from torch import Tensor
from torch.nn import Module
from torch.fft import ifftn, irfftn, ifftshift
from .sht import CSHT, RSHT, grid_DriscollHealy, theta_phi_to_xyz  # patched 1: was `from ebsdtorch.harmonics.sht`
from ._wigner_logspace import (  # patched 1: was `from ebsdtorch.harmonics.wigner_d_logspace`
    wigner_d_eq_half_pi,
    read_mn_wigner_d_half_pi_table,
    read_lmn_wigner_d_half_pi_table,
)
# patched 1: removed `from ebsdtorch.io.read_master_pattern import read_master_pattern` —
# upstream's master-pattern reader pulls non-vendored deps. We use our own SHT
# I/O at backend.spherical_gpu.pipeline.sht_io. If any function in this file
# happens to call read_master_pattern, we shim it via a stub that raises so the
# break point is loud, not silent.
def read_master_pattern(*_args, **_kwargs):
    raise NotImplementedError(
        "ebsdtorch.io.read_master_pattern is not vendored. "
        "Use backend.spherical_gpu.pipeline.sht_io.read_sht_master() instead."
    )


@torch.jit.script
def rs2cc_sparse(
    B: int,
    La: int,
    f_a: Tensor,                # (b, B, La) complex pattern coefs at active l only
    g_a: Tensor,                # (b, B, La) complex master  coefs at active l only
    d_mk_pre: Tensor,           # (La, B, 2B-1) complex64 — ifftshifted on k axis
    d_kn_pre: Tensor,           # (La, 2B-1, 2B-1) complex64 — ifftshifted on (k, n)
) -> Tensor:
    """L-sparse + ifftshift-folded variant of rs2cc_vectorized.

    Folds two optimizations compared to ``rs2cc_vectorized``:

    1. Active-l restriction. For symmetric phases (e.g. m-3m, z_rot=4), the
       master coefs satisfy g[m, l] = 0 for all m whenever l is forbidden by
       the symmetry. Caller pre-selects ``l_active`` (LongTensor) and passes
       only those slices: ``f_a = f[..., l_active]``, ``g_a = g[..., l_active]``,
       and the corresponding rows of ``d_mk_pre`` / ``d_kn_pre``. The full
       contraction is mathematically equivalent to the dense rs2cc_vectorized
       (zero contributions are dropped, exact match modulo FP order).

    2. ifftshift folding. The output spectrum is required in fft-standard
       layout (DC at index 0) for irfftn. Instead of doing an explicit
       ``ifftshift(spectrum, dim=(-2, -1))`` after the einsum, the caller
       pre-applies ifftshift to ``d_mk`` (k axis) and ``d_kn`` (k and n axes)
       at init. We also build G_full directly in fft layout: cat([g_pos,
       g_neg_flipped]) instead of cat([g_neg_flipped, g_pos]). This is
       equivalent to ifftshift along the n axis for an odd-length axis.

    On Ni m-3m at L=68, b=88: 53.4ms → 34.7ms (35% reduction).
    """
    bf = f_a.shape[0]
    bg = g_a.shape[0]
    if bf != bg and bf != 1 and bg != 1:
        raise ValueError(f"Batch dims of f {bf} and g {bg} not broadcastable")
    bx = max(bf, bg)
    if bf == 1 and bx > 1:
        f_a = f_a.expand(bx, -1, -1)
    if bg == 1 and bx > 1:
        g_a = g_a.expand(bx, -1, -1)
    L = B

    # G_full in fft layout: [g_pos, g_neg_flipped] along n axis.
    # Equivalent to ifftshift(centered, dim=n) for odd N.
    g_pos = g_a.conj()                                # (b, L, La)
    m_pos_idx = torch.arange(1, L, dtype=torch.float64, device=f_a.device)
    sign = ((-1.0) ** m_pos_idx).to(g_a.dtype)        # (L-1,)
    g_neg = g_a[:, 1:, :] * sign[None, :, None]       # (b, L-1, La)
    g_neg_flipped = g_neg.flip(1)                     # n in (L-1)..1
    G_full = torch.cat([g_pos, g_neg_flipped], dim=1)  # (b, 2L-1, La) — fft layout

    # Step 1+2: build F_lmk and G_lkn intermediates (separate to keep memory
    # bounded — iter-9 tried fusing into one 4-operand einsum and OOMed at
    # 49 GiB at L=88).
    F_lmk = torch.einsum("bml,lmk->blmk", f_a, d_mk_pre)
    G_lkn = torch.einsum("bnl,lkn->blkn", G_full, d_kn_pre)
    # Step 3 — final contraction. Profile finding (2026-05-11, bench_sht_profile_stacktrace.py):
    # the previous matmul(F.permute, G.permute) + .permute().contiguous() path spent
    # ~25 ms/batch in aten::copy_ alone — two implicit copies inside matmul (it
    # materialises the strided permuted inputs for cuBLAS) plus the explicit final
    # .contiguous(). Profiler shapes [32,88,175,175] / [32,175,175,88] / [32,175,175,175]
    # all originated here.
    #
    # Switching the entire step into a single torch.einsum lets the optimizer pick
    # a kernel that emits the (B, m, k, n) output layout in one pass, removing both
    # the intermediate (B, k, m, n) tensor and the final permute+contiguous. Micro-
    # bench (tasks/bench_rs2cc_variants.py) measured V0 28.15 ms vs V1 23.91 ms
    # median across 7 repeats — 1.18x speedup with 0.00e+00 max relative error
    # (bit-identical output).
    spectrum = torch.einsum("blmk,blkn->bmkn", F_lmk, G_lkn)

    # No ifftshift needed: spectrum already in fft layout via pre-shifted d.
    cc = irfftn(
        spectrum,
        s=(2 * L - 1, 2 * L - 1, 2 * L - 1),
        dim=(-1, -2, -3),
        norm="forward",
    )
    return cc


@torch.jit.script
def rs2cc_vectorized(
    B: int,
    f: Tensor,                  # (b, B, B) complex pattern coefs [b, m_pos, l]
    g: Tensor,                  # (b, B, B) complex master  coefs [b, n_pos, l]
    d_full: Tensor,             # (B, 2B-1, 2B-1) real Wigner-d half-pi, masked
                                #   d_full[l, m+B-1, k+B-1] = d^l_{m, k}(pi/2)
                                #   d_full[l, m, k] = 0 if l < max(|m|, |k|)
) -> Tensor:
    """Vectorized rs2cc_ — replaces the per-l Python loop with a single
    4-operand einsum + irfftn. Equivalent math, much faster on GPU.

    The spectrum has shape (b, B, 2B-1, 2B-1) with m as the rfft half
    axis, k and n full. After irfftn the volume is (b, 2B-1, 2B-1, 2B-1).
    """
    bf = f.shape[0]
    bg = g.shape[0]
    if bf != bg and bf != 1 and bg != 1:
        raise ValueError(f"Batch dims of f {bf} and g {bg} not broadcastable")
    bx = max(bf, bg)
    if bf == 1 and bx > 1:
        f = f.expand(bx, -1, -1)
    if bg == 1 and bx > 1:
        g = g.expand(bx, -1, -1)
    L = B

    # Build G_full[b, n_idx, l] in [-L+1, L-1] with Hermitian symmetry:
    #   n >= 0:  G_full[..., n + L - 1, l] = conj(g[..., n, l])
    #   n < 0 :  G_full[..., n + L - 1, l] = (-1)^|n| * g[..., |n|, l]
    g_pos = g.conj()                                # (b, L, L)
    m_pos_idx = torch.arange(1, L, dtype=torch.float64, device=f.device)
    sign = ((-1.0) ** m_pos_idx).to(g.dtype)        # (L-1,)
    # negative n branch (for n = -1, -2, ..., -(L-1)). We need rows at
    # G_full indices 0 .. L-2 for n = -(L-1) .. -1. Order: at index 0 is
    # n = -(L-1); at index L-2 is n = -1.
    # Source: g[b, |n|, l] with sign (-1)^|n|. |n| spans (L-1)..1 (decreasing).
    g_neg = g[:, 1:, :] * sign[None, :, None]       # (b, L-1, L) — n in 1..L-1
    g_neg_flipped = g_neg.flip(1)                   # n in (L-1)..1 → idx 0..L-2 maps to n = -(L-1)..-1
    G_full = torch.cat([g_neg_flipped, g_pos], dim=1)  # (b, 2L-1, L)

    # Slice positive-m half of d_full for the m axis (rfft half).
    # einsum requires matching dtypes — promote real d to complex.
    d_mk = d_full[:, L - 1 :, :].to(g.dtype)        # (L, L, 2L-1) complex
    d_kn = d_full.to(g.dtype)                       # (L, 2L-1, 2L-1) complex

    # Three-step contraction to keep intermediates small.
    # spectrum[b, m_pos, k, n] = sum_l F[b, m_pos, l] * G_full[b, n, l]
    #                                 * d_mk[l, m_pos, k] * d_kn[l, k, n]
    # Step 1: F_lmk[b, l, m, k] = f[b, m, l] * d_mk[l, m, k]
    # Step 2: G_lkn[b, l, k, n] = G_full[b, n, l] * d_kn[l, k, n]
    # Step 3: spectrum[b, m, k, n] = sum_l F_lmk[b, l, m, k] * G_lkn[b, l, k, n]
    F_lmk = torch.einsum("bml,lmk->blmk", f, d_mk)
    G_lkn = torch.einsum("bnl,lkn->blkn", G_full, d_kn)
    spectrum = torch.einsum("blmk,blkn->bmkn", F_lmk, G_lkn)

    # ifftshift along k and n to convert centered → fft-standard layout
    spectrum = ifftshift(spectrum, dim=(-2, -1))
    # irfftn: m is the rfft half (last in dim list)
    cc = irfftn(
        spectrum,
        s=(2 * L - 1, 2 * L - 1, 2 * L - 1),
        dim=(-1, -2, -3),
        norm="forward",
    )
    return cc


def rs2cc_fp64(
    B: int,
    f: Tensor,
    g: Tensor,
    wigner_d_precompute: Tensor,
) -> Tensor:
    """FP64 variant of rs2cc_ — same algorithm, complex128 internal accumulation.

    Use when bin-borderline pixels are flipped by FP32 cc-volume noise.
    Empirically lifts the alt-oracle <5deg fraction from ~28% to ~38% on
    the canonical iter-15 dataset, matching EMSphInx indexImage parity.

    Returns float64 cc volume of shape (b, 2B-1, 2B-1, 2B-1).
    """
    b1 = f.shape[0]
    b2 = g.shape[0]
    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Batch dims of f {b1} and g {b2} are not broadcastable")
    b = max(b1, b2)

    f128 = f.to(torch.complex128)
    g128 = g.to(torch.complex128)
    wigner_fp64 = wigner_d_precompute.to(torch.float64)

    cc = torch.zeros(
        (b, B, 2 * B - 1, 2 * B - 1), dtype=torch.complex128, device=f.device,
    )
    gconj = g128.conj()
    for l in range(B):
        m_inds_half = torch.arange(0, l + 1, dtype=torch.int32, device=f.device)
        g_n = torch.cat([
            (g128[:, m_inds_half[1:], l] * (-1.0) ** m_inds_half[1:][None, :]).flip(1),
            gconj[:, m_inds_half, l],
        ], dim=1)[:, None, None, :]
        f_m = f128[:, m_inds_half, l][:, :, None, None]

        m, k = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        if l != 0:
            m = m[l:]
            k = k[l:]

        d_lmk = read_mn_wigner_d_half_pi_table(
            wigner_fp64, m_coords=m, n_coords=k, l=l,
        )[None, :, :, None]
        k, n = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        d_lkn = read_mn_wigner_d_half_pi_table(
            wigner_fp64, m_coords=k, n_coords=n, l=l,
        )[None, None, :, :]

        cc[:, : (l + 1), (-l + B - 1) : (l + B), (-l + B - 1) : (l + B)] += (
            f_m * g_n * (d_lmk * d_lkn).to(torch.complex128)
        )

    cc = ifftshift(cc, dim=(-1, -2))
    cc = irfftn(
        cc,
        s=(2 * B - 1, 2 * B - 1, 2 * B - 1),
        dim=(-1, -2, -3),
        norm="forward",
    )
    return cc


@torch.jit.script
def rs2cc_(
    B: int,
    f: Tensor,
    g: Tensor,
    wigner_d_precompute: Tensor,
) -> Tensor:
    """
    This function computes the cross correlation of two functions on the 2-sphere
    over the group SO(3), using Spherical Harmonics.

    Args:
        B: int, the bandlimit.
        f: complex tensor of coefficients shape (b, B, B) for real S2 function
        g: complex tensor of coefficients shape (b, B, B) for real S2 function
        wigner_d_precompute: tensor shape (B * (B + 1) * (B + 2) // 6, )

    Returns:
        cc: Cross correlation volume of variable shape


    CC volume: F^-1{f^l_m * conj(g^l_n) * d^l_{m,k}(pi/2) * d^l_{k, n}(pi/2)}

    Shape: (b, 2B - 1, 2B - 1, 2B - 1)

    """
    b1 = f.shape[0]
    b2 = g.shape[0]

    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Batch dimensions of f {b1} and g {b2} are not broadcastable")

    b = max(b1, b2)

    # iterate over l as the 4D volume shape L^4 will often not fit in memory
    cc = torch.zeros(
        (b, B, 2 * B - 1, 2 * B - 1), dtype=torch.complex64, device=f.device
    )
    gconj = g.conj()

    for l in range(B):
        m_inds_half = torch.arange(0, l + 1, dtype=torch.int32, device=f.device)
        # get the relevant harmonic coefficients for g and f for the current l
        g_n = torch.cat(
            [
                (g[:, m_inds_half[1:], l] * (-1.0) ** m_inds_half[1:][None, :]).flip(1),
                gconj[:, m_inds_half, l],
            ],
            dim=1,
        )[
            :, None, None, :
        ]  # (..., m, k, n*) — n spans -l..l
        f_m = f[:, m_inds_half, l][:, :, None, None]  # (..., m*, k, n) — m spans 0..l


        m, k = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        if l != 0:
            m = m[l:]
            k = k[l:]

        # get the precomputed Wigner d-matrices for the current l
        d_lmk = read_mn_wigner_d_half_pi_table(
            wigner_d_precompute,
            m_coords=m,
            n_coords=k,
            l=l,
        )[
            None, :, :, None
        ]  # (B, m*, k*, n)
        k, n = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        d_lkn = read_mn_wigner_d_half_pi_table(
            wigner_d_precompute,
            m_coords=k,
            n_coords=n,
            l=l,
        )[
            None, None, :, :
        ]  # (B, m, k*, n*)

        cc[:, : (l + 1), (-l + B - 1) : (l + B), (-l + B - 1) : (l + B)] += (
            f_m * g_n * (d_lmk * d_lkn).to(torch.complex64)
        )

    # Need to do fftshift so that the low frequencies are at the periphery
    cc = ifftshift(cc, dim=(-1, -2))

    # dim=(..., -3) means the half dimension is the -3 dimension (m)
    cc = irfftn(
        cc,
        s=(2 * B - 1, 2 * B - 1, 2 * B - 1),
        dim=(-1, -2, -3),
        norm="forward",
    )

    return cc


@torch.jit.script
def rs2cc_fast_(
    B: int,
    f: Tensor,
    g: Tensor,
    d_lmk: Tensor,
) -> Tensor:
    """
    This function computes the cross correlation of two functions on the 2-sphere
    over the group SO(3), using Spherical Harmonics.

    Args:
        B: int, the bandlimit.
        f: complex tensor of coefficients shape (b, B, B) for real S2 function
        g: complex tensor of coefficients shape (b, B, B) for real S2 function
        wigner_d_precompute: tensor shape (B * (B + 1) * (B + 2) // 6, )

    Returns:
        cc: Cross correlation volume

    CC volume: F^-1{f^l_m * conj(g^l_n) * d^l_{m,k}(pi/2) * d^l_{k, n}(pi/2)}

    Shape: (b, 2B - 1, 2B - 1, 2B - 1)

    """

    b1 = f.shape[0]
    b2 = g.shape[0]

    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Batch dimensions of f {b1} and g {b2} are not broadcastable")

    m_inds_half = torch.arange(1, B, dtype=torch.int32, device=f.device)
    # augment g and f with zeros for the negative m values
    g_n_aug = torch.cat(
        [
            g[:, 1:, :].flip(1) * ((-1.0) ** m_inds_half.flip(0))[None, :, None],
            g.conj(),
        ],
        dim=1,
    )

    f_lmkb = torch.einsum("lmk,bml->blmk", d_lmk[:, (B - 1) :, :], f)
    g_lknb = torch.einsum("lkn,bnl->blkn", d_lmk[:, :, (B - 1) :], g.conj())
    cc = torch.einsum("blmk,blkn->bmkn", f_lmkb, g_lknb)

    # Need to do fftshift so that the low frequencies are at the periphery
    cc = ifftshift(cc, dim=(-2))

    # dim=(..., -3) means the half dimension is the -3 dimension (m)
    cc = irfftn(cc, s=(2 * B - 1,) * 3, dim=(-1, -2, -3), norm="forward")

    return cc


@torch.jit.script
def cs2cc_(
    B: int,
    f: Tensor,
    g: Tensor,
    wigner_d_precompute: Tensor,
) -> Tensor:
    """
    This function computes the cross correlation of two functions on the 2-sphere
    over the group SO(3), using Spherical Harmonics.

    Args:
        B: int, the bandlimit.
        f: complex tensor of coefficients shape (b, 2B - 1, B) for complex S2 function
        g: complex tensor of coefficients shape (b, 2B - 1, B) for complex S2 function
        wigner_d_precompute: tensor shape (B * (B + 1) * (B + 2) // 6, )

    Returns:
        cc: Cross correlation volume of variable shape


    CC volume: F^-1{f^l_m * conj(g^l_n) * d^l_{m,k}(pi/2) * d^l_{k, n}(pi/2)}

    Shape: (b, 2B - 1, 2B - 1, 2B - 1)

    """

    b1 = f.shape[0]
    b2 = g.shape[0]

    if b1 != b2 and b1 != 1 and b2 != 1:
        raise ValueError(f"Batch dimensions of f {b1} and g {b2} are not broadcastable")

    b = max(b1, b2)

    # iterate over l as the 4D volume shape L^4 will often not fit in memory
    cc = torch.zeros(
        (b, 2 * B - 1, 2 * B - 1, 2 * B - 1), dtype=torch.complex64, device=f.device
    )

    gconj = g.conj()

    for l in range(B):
        m_inds_full = (
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device) + B - 1
        )
        # get the relevant harmonic coefficients for g and f for the current l
        g_n = gconj[:, m_inds_full, l][:, None, None, :]  # (..., m, k, n)
        f_m = f[:, m_inds_full, l][:, :, None, None]  # (..., m, k, n)

        m, k = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        # get the precomputed Wigner d-matrices for the current l
        d_lmk = read_mn_wigner_d_half_pi_table(
            wigner_d_precompute,
            m_coords=m,
            n_coords=k,
            l=l,
        )[
            None, :, :, None
        ]  # (B, m*, k*, n)
        k, n = torch.meshgrid(
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            torch.arange(-l, l + 1, dtype=torch.int32, device=f.device),
            indexing="ij",
        )
        d_lkn = read_mn_wigner_d_half_pi_table(
            wigner_d_precompute,
            m_coords=k,
            n_coords=n,
            l=l,
        )[
            None, None, :, :
        ]  # (B, m, k*, n*)
        cc[
            :, (-l + B - 1) : (l + B), (-l + B - 1) : (l + B), (-l + B - 1) : (l + B)
        ] += (f_m * g_n * (d_lmk * d_lkn).to(torch.complex64))

    # Need to do fftshift so that the low frequencies are at the periphery
    cc = torch.fft.ifftshift(cc, dim=(-3, -2, -1))

    # Inverse 2D FFT along the last two dimensions
    cc = ifftn(
        cc,
        dim=(-3, -2, -1),
        norm="forward",
    ).abs()

    return cc


# # test it by doing autocorrelation of a master pattern with itself
# L = 256
# L_trunc = 128
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")

# precision = "single"
# dtype = torch.float32

# # precision = "double"
# # dtype = torch.float64

# wigner_table = wigner_d_eq_half_pi(
#     L + 2,
#     dtype=dtype,
#     device=device,
# )

# sht_real = RSHT(L, device, precision=precision)
# sht_comp = CSHT(L, device, precision=precision)

# sht_real_trunc = RSHT(L_trunc, device, precision=precision)
# sht_comp_trunc = CSHT(L_trunc, device, precision=precision)

# # dh_grid = grid_DriscollHealy(L).to(device)
# dh_grid = grid_DriscollHealy(L, dtype=dtype).to(device)
# xyz_grid = theta_phi_to_xyz(dh_grid)

# # mp_fname = "../EMs/EMplay/old/Si-master-20kV.h5"
# # mp = read_master_pattern(mp_fname).to(device).to(dtype)

# from kikuchipy.data import ebsd_master_pattern
# import numpy as np
# from ebsdtorch.io.read_master_pattern import MasterPattern

# mp_kp = ebsd_master_pattern(
#     "Ni", allow_download=True, show_progressbar=True, projection="lambert"
# )
# hsphere = torch.tensor(np.array(mp_kp.data[-1]), device=device, dtype=torch.float32)
# hsphere = (hsphere - hsphere.min()) / (hsphere.max() - hsphere.min())
# mp_tensors = (hsphere, hsphere)
# mp = MasterPattern(mp_tensors, laue_group=11)

# mp.normalize("minmax")

# # Interpolate master pattern
# mp_spherical_img = mp.interpolate(
#     xyz_grid,
#     mode="bicubic",
#     padding_mode="border",
#     align_corners=False,
#     normalize_coords=True,
#     virtual_binning=1,
# ).squeeze(0)

# # mp_spherical_img -= mp_spherical_img.mean()

# mp_spherical_r_coeffs = sht_real.fsht(mp_spherical_img.unsqueeze(0))
# mp_spherical_c_coeffs = sht_comp.fsht(mp_spherical_img.unsqueeze(0))

# print(f"mp_spherical_r_coeffs shape: {mp_spherical_r_coeffs.shape}")
# print(f"mp_spherical_c_coeffs shape: {mp_spherical_c_coeffs.shape}")

# # truncate to L
# mp_spherical_r_coeffs = mp_spherical_r_coeffs[:, :L_trunc, :L_trunc].contiguous()
# mp_spherical_c_coeffs = mp_spherical_c_coeffs[
#     :, (-L_trunc + L) : (L_trunc + L - 1), :L_trunc
# ].contiguous()

# print(f"truncate mp_spherical_r_coeffs shape: {mp_spherical_r_coeffs.shape}")
# print(f"truncate mp_spherical_c_coeffs shape: {mp_spherical_c_coeffs.shape}")

# # recover the image from the coefficients and use that image to estimate the maximum possible dot
# mp_spherical_img_recon_r = sht_real_trunc.isht(mp_spherical_r_coeffs)
# mp_spherical_img_recon_c = sht_comp_trunc.isht(mp_spherical_c_coeffs)

# ideal_max_r = (mp_spherical_img_recon_r).mean().item()
# ideal_max_c = (mp_spherical_img_recon_c).abs().mean().item()

# print(f"ideal_max_r: {ideal_max_r}")
# print(f"ideal_max_c: {ideal_max_c}")

# d_lmk = torch.zeros(
#     (L_trunc, 2 * L_trunc - 1, 2 * L_trunc - 1),
#     dtype=torch.float32,
#     device=device,
# )

# ll, nn, kk = torch.meshgrid(
#     torch.arange(0, L_trunc, dtype=torch.int32, device=device),
#     torch.arange(-L_trunc + 1, L_trunc, dtype=torch.int32, device=device),
#     torch.arange(-L_trunc + 1, L_trunc, dtype=torch.int32, device=device),
#     indexing="ij",
# )

# valid = (nn.abs() <= ll) & (kk.abs() <= ll)
# d_lmk[valid] = read_lmn_wigner_d_half_pi_table(
#     wigner_table,
#     coords=torch.stack([ll[valid], nn[valid], kk[valid]], dim=-1),
# )

# d_lmk = d_lmk.to(torch.complex64)

# cc_r = rs2cc_fast_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs, d_lmk)
# # cc_r = rs2cc_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs, wigner_table)
# cc_c = cs2cc_(L_trunc, mp_spherical_c_coeffs, mp_spherical_c_coeffs, wigner_table)

# print(f"cc_r shape: {cc_r.shape}")
# print(f"cc_c shape: {cc_c.shape}")

# print(f"cc_r min/max: {cc_r.min().item()}, {cc_r.max().item()}")
# print(f"cc_c min/max: {cc_c.min().item()}, {cc_c.max().item()}")

# # # min across both
# # minval = min(cc_r.min().item(), cc_c.min().item(
# # maxval = max(cc_r.max().item(), cc_c.max().item())

# # cc_c = (cc_c - minval) / (maxval - minval)
# # cc_r = (cc_r - minval) / (maxval - minval)

# cc_c = (cc_c - cc_c.min()) / (cc_c.max() - cc_c.min())
# cc_r = (cc_r - cc_r.min()) / (cc_r.max() - cc_r.min())

# # make a gif of the slices, only showing every 4th slice
# cc_c = (cc_c * 255).byte().squeeze(0).cpu().numpy()
# cc_r = (cc_r * 255).byte().squeeze(0).cpu().numpy()

# # from PIL import Image
# # import cv2
# # import numpy as np

# # # make compressed video instead of gif
# # fourcc = cv2.VideoWriter_fourcc(*"MJPG")


# # # use a function instead for compactness
# # def write_video(fourcc, fname, cc, dim):
# #     if dim == 0:
# #         out = cv2.VideoWriter(
# #             fname, fourcc, 30, (cc.shape[2], cc.shape[1]), isColor=False
# #         )
# #         for i in range(cc.shape[0]):
# #             out.write(cc[i, :, :])
# #         out.release()
# #     elif dim == 1:
# #         out = cv2.VideoWriter(
# #             fname, fourcc, 30, (cc.shape[2], cc.shape[0]), isColor=False
# #         )
# #         for i in range(cc.shape[1]):
# #             out.write(cc[:, i, :])
# #         out.release()
# #     elif dim == 2:
# #         out = cv2.VideoWriter(
# #             fname, fourcc, 30, (cc.shape[1], cc.shape[0]), isColor=False
# #         )
# #         for i in range(cc.shape[2]):
# #             out.write(cc[:, :, i])
# #         out.release()
# #     else:
# #         raise ValueError(f"dim {dim} is not valid")


# # # write each mp4
# # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# # write_video(fourcc, "cc_r_m.mp4", cc_r, 0)
# # write_video(fourcc, "cc_r_k.mp4", cc_r, 1)
# # write_video(fourcc, "cc_r_n.mp4", cc_r, 2)

# # write_video(fourcc, "cc_c_m.mp4", cc_c, 0)
# # write_video(fourcc, "cc_c_k.mp4", cc_c, 1)
# # write_video(fourcc, "cc_c_n.mp4", cc_c, 2)


# # make a batch of coefficients of real valued signal and measure the speed
# # of the cross correlation via 10 iterations
# import time

# n_iters = 10
# batch_size = 1
# mp_spherical_r_coeffs_batch = mp_spherical_r_coeffs.repeat(batch_size, 1, 1)

# cc_r = rs2cc_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, wigner_table)
# cc_r = rs2cc_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, wigner_table)

# start = time.time()

# for i in range(n_iters):
#     cc_r = rs2cc_(
#         L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, wigner_table
#     )

# torch.cuda.synchronize()

# duration = time.time() - start

# print(f"SLOW: CC per second: {n_iters * batch_size / duration}")

# start = time.time()

# cc_r = rs2cc_fast_(L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, d_lmk)

# for i in range(n_iters):
#     cc_r = rs2cc_fast_(
#         L_trunc, mp_spherical_r_coeffs, mp_spherical_r_coeffs_batch, d_lmk
#     )

# torch.cuda.synchronize()

# duration = time.time() - start

# print(f"FAST: CC per second: {n_iters * batch_size / duration}")


# # # plot with plotly now
# # import plotly.graph_objects as go
# # import numpy as np

# # X, Y, Z = np.meshgrid(
# #     np.arange(cc_c.shape[0]),
# #     np.arange(cc_c.shape[1]),
# #     np.arange(cc_c.shape[2]),
# #     indexing="ij",
# # )

# # print(f"X shape: {X.shape}")
# # print(f"Y shape: {Y.shape}")
# # print(f"Z shape: {Z.shape}")
# # print(f"cc_c shape: {cc_c.shape}")

# # fig = go.Figure(
# #     data=[
# #         go.Volume(
# #             x=X.flatten().astype(np.int8),
# #             y=Y.flatten().astype(np.int8),
# #             z=Z.flatten().astype(np.int8),
# #             value=cc_c.flatten(),
# #             isomin=0,
# #             isomax=150,
# #             opacity=0.15,  # needs to be small to see through all surfaces
# #             surface_count=5,  # needs to be a large number for good volume rendering
# #         )
# #     ]
# # )

# # # fig.show()

# # # write html
# # fig.write_html("cc_c.html")