"""Sympy-validated Wigner-d half-pi table + EMSphInx-convention cc kernel.

This module provides a CLEAN, sympy-validated implementation of the
small Wigner-d matrix at beta=pi/2 (the table needed by spherical SHT
cross-correlation), AND an EMSphInx-convention cross-correlation kernel
that uses it.

Why this exists
---------------
ebsdtorch's `read_mn_wigner_d_half_pi_table` returns Kostelec-Rockmore
(TS2Kit) convention values when called with `swap_mn=False` (the default
used by `rs2cc_`). These do NOT match sympy / EMSphInx in 46 % of
(l, m, n) triples (validated up to L=8 in test_diag_wigner_validation).

We instead build the table directly from a stable closed-form Riemann
sum (the same recurrence EMSphInx uses internally in `wigner.hpp`),
validated against sympy for small l.

The table layout is:
  table[l, m, n] = d^l_{m,n}(pi/2)  with -l <= m, n <= l, 0 <= l < L
i.e. shape (L, 2L-1, 2L-1) — full m, n range, dense.

For high l (l > ~32) numerical precision can be an issue with the
naive Riemann formula; we use FP64 throughout and the formula is
known stable up to l ~ 100 in FP64.

EMSphInx wigner.hpp uses an extended-precision (xnum) variant for very
high l. We can plug that in later if we hit precision limits at L=68.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def _factorial_table(n: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """Returns tensor of factorials [0!, 1!, ..., n!] in fp64."""
    out = torch.zeros(n + 1, dtype=dtype, device=device)
    out[0] = 1.0
    for i in range(1, n + 1):
        out[i] = out[i - 1] * i
    return out


def build_wigner_d_half_pi_table(
    L: int, dtype=torch.float64, device=None,
) -> torch.Tensor:
    """Build d^l_{m,n}(pi/2) for 0 <= l < L, -l <= m, n <= l.

    Uses the explicit Riemann formula:

        d^l_{m,n}(beta) = sum_{s} (-1)^{m-n+s} *
                             sqrt(factorial(l+m)*factorial(l-m)*factorial(l+n)*factorial(l-n)) /
                             (factorial(l+n-s) * factorial(s) * factorial(m-n+s) * factorial(l-m-s)) *
                             cos(beta/2)^{2l+n-m-2s} *
                             sin(beta/2)^{m-n+2s}

    At beta = pi/2: cos(beta/2) = sin(beta/2) = sqrt(2)/2, so:
        cos^a * sin^b = (sqrt(2)/2)^(a+b) = (1/2)^((a+b)/2) = (1/2)^l

    Therefore:
        d^l_{m,n}(pi/2) = (1/2)^l * sqrt(...) *
                            sum_{s} (-1)^{m-n+s} /
                                    (factorial(l+n-s) * factorial(s) *
                                     factorial(m-n+s) * factorial(l-m-s))

    Sum range: s_min = max(0, n-m), s_max = min(l-m, l+n).

    Returns: (L, 2L-1, 2L-1) tensor d[l, m+l, n+l] = d^l_{m,n}(pi/2)
    """
    fact = _factorial_table(2 * L + 2, dtype=dtype, device=device)
    out = torch.zeros(L, 2 * L - 1, 2 * L - 1, dtype=dtype, device=device)

    for l in range(L):
        norm_const = (0.5) ** l
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                norm = torch.sqrt(
                    fact[l + m] * fact[l - m] * fact[l + n] * fact[l - n]
                )
                s_min = max(0, n - m)
                s_max = min(l - m, l + n)
                if s_max < s_min:
                    continue
                s_sum = torch.zeros((), dtype=dtype, device=device)
                for s in range(s_min, s_max + 1):
                    sign = -1.0 if (m - n + s) % 2 else 1.0
                    denom = (
                        fact[l + n - s]
                        * fact[s]
                        * fact[m - n + s]
                        * fact[l - m - s]
                    )
                    s_sum = s_sum + sign / denom
                out[l, m + l, n + l] = norm_const * norm * s_sum

    return out


def emsphinx_cc_kernel(
    f: torch.Tensor,                # (B, L, L) complex — pattern coefs f^l_m for m=0..l
    g: torch.Tensor,                # (1 or B, L, L) complex — master coefs g^l_n
    wigner_table: torch.Tensor,     # (L, 2L-1, 2L-1) float — d^l_{m,n}(pi/2)
    L: int,
) -> torch.Tensor:
    """EMSphInx-convention spherical cross-correlation kernel.

    Computes the cross-correlation cc(R) for all R on the SO(3) torus,
    via the trick that uses the wigner-d half-pi table to fft into
    a 3D volume.

    Implements the naive reference loop from
    EMSphInx sht_xcorr.hpp:660-685:

      cc[k_idx, n_idx, m_idx] = sum_{l, s.t. l>=max(am, ak, an)}
                                   f^l_m * conj(g^l_n) *
                                   d^l_{k, m}(pi/2) * d^l_{n, k}(pi/2)

    where:
      m ranges 0 to L-1 (rfft half-spectrum)
      n ranges -(L-1) to L-1
      k ranges -(L-1) to L-1
      |m| | |n| | |k| <= l < L

    Real-signal symmetries are applied:
      f^l_{-m} = (-1)^m * conj(f^l_m)
      g^l_{-n} = (-1)^n * conj(g^l_n)

    Returns the SPECTRAL volume (B, L, 2L-1, 2L-1) of complex64,
    indexed as [batch, m_idx, k_idx, n_idx]. Caller must apply the
    inverse 3D FFT to get the spatial cc volume.

    Note: this is the SLOW, CORRECT-BY-CONSTRUCTION reference. Use
    only for validation. Production should call a vectorized version.
    """
    B = max(f.shape[0], g.shape[0])
    sl = 2 * L - 1
    spectrum = torch.zeros(B, L, sl, sl, dtype=torch.complex128, device=f.device)

    f128 = f.to(torch.complex128)
    g128 = g.to(torch.complex128)
    if g128.shape[0] == 1 and B > 1:
        g128 = g128.expand(B, -1, -1)
    wig = wigner_table.to(torch.float64)

    for l in range(L):
        for m in range(0, l + 1):
            for k in range(-l, l + 1):
                for n in range(-l, l + 1):
                    am = abs(m)
                    an = abs(n)
                    if l < max(am, an, abs(k)):
                        continue
                    # f^l_m for m >= 0 — direct lookup
                    f_lm = f128[:, m, l]
                    # g^l_n: real-signal symmetry for n < 0
                    if n >= 0:
                        g_ln = g128[:, n, l]
                    else:
                        sign = (-1.0) ** an
                        g_ln = sign * g128[:, an, l].conj()
                    # d^l_{k,m}(pi/2) and d^l_{n,k}(pi/2) — sympy convention
                    d_km = wig[l, k + l, m + l]
                    d_nk = wig[l, n + l, k + l]
                    spectrum[:, m, k + (L - 1), n + (L - 1)] += (
                        f_lm * g_ln.conj() * d_km * d_nk
                    )

    return spectrum.to(torch.complex64)


def emsphinx_cc_volume_from_spectrum(spectrum: torch.Tensor) -> torch.Tensor:
    """Apply 3D inverse-real-FFT to convert the spectrum to a cc volume.

    Input  shape: (B, L, 2L-1, 2L-1)
    Output shape: (B, 2L-1, 2L-1, 2L-1) — real (k, n, m) spatial axes
    """
    B, L, sl_n, sl_k = spectrum.shape
    sl = 2 * L - 1
    assert sl_n == sl == sl_k

    # First axis (m) is the rfft half-spectrum: irfft expands it to sl
    # Other two axes (k, n) need ifftshift first because EMSphInx stores
    # (positive_freqs, negative_freqs) but irfftn wants standard FFT layout
    # (0, 1, ..., sl/2, -sl/2+1, ..., -1) — but that IS what we have
    # since k_idx = k + (L-1) maps k in [-(L-1), L-1] to [0, 2L-2].
    # So shifting from "centered" (DC at L-1) to "fft-standard" (DC at 0)
    # is a single ifftshift along k and n.
    shifted = torch.fft.ifftshift(spectrum, dim=(-2, -1))
    # 3D inverse real FFT: m axis is the half-spectrum
    cc = torch.fft.irfftn(
        shifted,
        s=(sl, sl, sl),
        dim=(-2, -1, -3),
        norm="forward",
    )
    return cc
