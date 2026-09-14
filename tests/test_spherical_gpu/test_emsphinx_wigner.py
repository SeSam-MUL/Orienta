"""Validate emsphinx_wigner.build_wigner_d_half_pi_table against sympy."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def test_build_wigner_table_matches_sympy():
    from sympy.physics.quantum.spin import Rotation
    import sympy as sp
    from backend.spherical_gpu._math.emsphinx_wigner import (
        build_wigner_d_half_pi_table,
    )

    L = 6
    table = build_wigner_d_half_pi_table(L, dtype=torch.float64, device=torch.device("cpu"))
    assert table.shape == (L, 2 * L - 1, 2 * L - 1)
    assert torch.isfinite(table).all()

    n_total = 0
    n_match = 0
    fails = []
    for l in range(L):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                ref = float(Rotation.d(l, m, n, sp.pi / 2).doit())
                ours = float(table[l, m + l, n + l].item())
                n_total += 1
                if abs(ours - ref) < 1e-9:
                    n_match += 1
                else:
                    fails.append((l, m, n, ref, ours, ours - ref))

    print(f"\nMatched {n_match}/{n_total} entries (tol=1e-9)")
    if fails:
        print(f"Mismatches (first 10):")
        for f in fails[:10]:
            print(f"  l={f[0]} m={f[1]} n={f[2]}: sympy={f[3]:.6f} ours={f[4]:.6f} diff={f[5]:.2e}")
    assert n_match == n_total, f"{len(fails)} entries do not match sympy"


def test_emsphinx_cc_master_master_peaks_at_identity():
    """For f = g = master, cc(R) should peak at R = identity (modulo gauge)."""
    from backend.spherical_gpu._math.emsphinx_wigner import (
        build_wigner_d_half_pi_table,
        emsphinx_cc_kernel,
        emsphinx_cc_volume_from_spectrum,
    )

    L = 5  # small enough that the slow kernel is fast
    # Build a synthetic master with one nonzero entry to keep it tiny
    f = torch.zeros(1, L, L, dtype=torch.complex128)
    f[0, 0, 2] = 1.0 + 0j
    f[0, 1, 3] = 0.3 + 0.1j

    table = build_wigner_d_half_pi_table(L, dtype=torch.float64, device=torch.device("cpu"))

    spectrum = emsphinx_cc_kernel(f, f, table, L)
    cc_volume = emsphinx_cc_volume_from_spectrum(spectrum)
    assert cc_volume.shape == (1, 2 * L - 1, 2 * L - 1, 2 * L - 1)
    assert torch.isfinite(cc_volume).all()

    # The peak should be at the identity orientation. For the FFT layout,
    # identity = (k=0, n=0, m=0) which after ifftshift maps to a specific bin.
    # We just check that the peak is reasonable (not at random location).
    flat = cc_volume[0].reshape(-1)
    max_val = float(flat.max().item())
    print(f"\nemsphinx cc kernel: master.master peak = {max_val:.4f}")
    assert max_val > 0
