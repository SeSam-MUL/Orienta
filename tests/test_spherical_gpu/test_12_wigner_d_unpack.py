"""Tests for `wigner_d_unpack`: packed-to-dense Wigner-d converter.

Validates that the dense `(K, L, 2L-1, 2L-1)` produced by unpacking the
packed Fukushima recursion output matches the matrix-exp oracle bit-
identically on the SAME-SIGN quadrants (m, n both ≥ 0 or both ≤ 0). The
mixed-sign quadrants are masked out and not tested here — they require
the `wigner_d_gt_half_pi` complementary recursion.
"""
from __future__ import annotations

import pytest
import torch

from backend.spherical_gpu._math.wigner_d_torch_compile import (
    wigner_d_lt_half_pi_batched_eager,
)
from backend.spherical_gpu._math.wigner_d_eval import wigner_D_matrix
from backend.spherical_gpu._math.wigner_d_unpack import (
    build_unpack_index_and_sign_tables,
    unpack_wigner_d_to_dense,
    supported_mn_mask,
)


@pytest.mark.parametrize("L", [4, 6, 8])
@pytest.mark.parametrize("beta_val", [0.1, 0.3, 0.5, 1.0, 1.4])
def test_unpack_matches_matrix_exp_oracle(L: int, beta_val: float) -> None:
    """At every supported (l, m, n), unpacked dense Wigner-d must match the
    matrix-exp oracle bit-identically (within FP64 epsilon).
    """
    device = torch.device("cpu")
    idx_table, sign_table = build_unpack_index_and_sign_tables(L, device)
    mask = supported_mn_mask(L, device)

    betas = torch.tensor([beta_val], dtype=torch.float64, device=device)
    packed = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)
    dense = unpack_wigner_d_to_dense(packed, idx_table, sign_table)

    assert dense.shape == (1, L, 2 * L - 1, 2 * L - 1)

    for l in range(L):
        D_oracle = wigner_D_matrix(
            l, 0.0, beta_val, 0.0, dtype=torch.complex128
        ).real
        D_unpack = dense[0, l, (L - 1) - l: (L - 1) + l + 1, (L - 1) - l: (L - 1) + l + 1].to(torch.float64)
        mask_l = mask[l, (L - 1) - l: (L - 1) + l + 1, (L - 1) - l: (L - 1) + l + 1]
        if mask_l.sum() > 0:
            diff = (D_oracle - D_unpack)[mask_l].abs().max().item()
            assert diff < 1e-12, (
                f"L={L} beta={beta_val} l={l}: max err {diff:.2e} > 1e-12"
            )


def test_unpack_mask_correctness() -> None:
    """The mask should be True iff (m, n) are same-sign and |m|, |n| ≤ l."""
    device = torch.device("cpu")
    L = 6
    mask = supported_mn_mask(L, device)
    for l in range(L):
        for m_dense in range(2 * L - 1):
            m = m_dense - (L - 1)
            for n_dense in range(2 * L - 1):
                n = n_dense - (L - 1)
                want = (
                    abs(m) <= l and abs(n) <= l
                    and ((m >= 0 and n >= 0) or (m < 0 and n < 0))
                )
                got = bool(mask[l, m_dense, n_dense].item())
                assert got == want, (
                    f"l={l} m={m} n={n}: mask {got}, expected {want}"
                )


def test_unpack_diagonal_entries_match() -> None:
    """For (m=n) entries (always same-sign, supported), unpack must match
    the standard d^l_{m,m}(beta) formula.
    """
    device = torch.device("cpu")
    L = 5
    beta_val = 0.4
    idx_table, sign_table = build_unpack_index_and_sign_tables(L, device)
    betas = torch.tensor([beta_val], dtype=torch.float64, device=device)
    packed = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)
    dense = unpack_wigner_d_to_dense(packed, idx_table, sign_table)

    # d^l_{0,0}(beta) = P_l(cos(beta))  (Legendre polynomial)
    import math
    for l in range(L):
        cos_b = math.cos(beta_val)
        # Legendre P_l(cos_b) — closed forms for small l
        if l == 0:
            expected = 1.0
        elif l == 1:
            expected = cos_b
        elif l == 2:
            expected = 0.5 * (3 * cos_b**2 - 1)
        elif l == 3:
            expected = 0.5 * (5 * cos_b**3 - 3 * cos_b)
        elif l == 4:
            expected = (1 / 8) * (35 * cos_b**4 - 30 * cos_b**2 + 3)
        else:
            continue  # skip
        actual = dense[0, l, L - 1, L - 1].item()
        assert abs(actual - expected) < 1e-12, (
            f"l={l}: d^l_{{0,0}} = {actual} vs P_l(cos(beta))={expected}"
        )


def test_unpack_batched() -> None:
    """Unpack for multiple betas at once gives correct result for each."""
    device = torch.device("cpu")
    L = 4
    idx_table, sign_table = build_unpack_index_and_sign_tables(L, device)
    mask = supported_mn_mask(L, device)
    betas = torch.tensor([0.1, 0.5, 1.0], dtype=torch.float64, device=device)
    packed = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)
    dense = unpack_wigner_d_to_dense(packed, idx_table, sign_table)

    assert dense.shape == (3, L, 2 * L - 1, 2 * L - 1)

    for k, beta_val in enumerate([0.1, 0.5, 1.0]):
        for l in range(L):
            D_oracle = wigner_D_matrix(
                l, 0.0, beta_val, 0.0, dtype=torch.complex128
            ).real
            D_unpack = dense[k, l, (L - 1) - l: (L - 1) + l + 1, (L - 1) - l: (L - 1) + l + 1].to(torch.float64)
            mask_l = mask[l, (L - 1) - l: (L - 1) + l + 1, (L - 1) - l: (L - 1) + l + 1]
            if mask_l.sum() > 0:
                diff = (D_oracle - D_unpack)[mask_l].abs().max().item()
                assert diff < 1e-12, (
                    f"k={k} beta={beta_val} l={l}: err {diff:.2e}"
                )


# ---------------------------------------------------------------------------
# FULL unpack (same-sign + mixed-sign via wigner_d_gt_half_pi)
# ---------------------------------------------------------------------------

import math
from backend.spherical_gpu._math.wigner_d_torch_compile import (
    wigner_d_gt_half_pi_batched_eager,
)
from backend.spherical_gpu._math.wigner_d_unpack import (
    build_mixed_sign_tables, unpack_wigner_d_full,
)


@pytest.mark.parametrize("L", [4, 6, 8, 10])
@pytest.mark.parametrize("beta_val", [0.1, 0.5, 1.0, 1.4])
def test_unpack_full_matches_oracle_at_all_mn(L: int, beta_val: float) -> None:
    """unpack_wigner_d_full should match matrix_exp oracle on EVERY (l, m, n)
    including mixed-sign entries, to machine eps in FP64.
    """
    device = torch.device("cpu")
    pi_minus = math.pi - beta_val

    ss_idx, ss_sign = build_unpack_index_and_sign_tables(L, device)
    ms_idx, ms_sign = build_mixed_sign_tables(L, device)

    betas = torch.tensor([beta_val], dtype=torch.float64, device=device)
    betas_pi = torch.tensor([pi_minus], dtype=torch.float64, device=device)
    packed_lt = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)
    packed_gt = wigner_d_gt_half_pi_batched_eager(betas_pi, L, torch.float64, device)

    dense = unpack_wigner_d_full(packed_lt, packed_gt, ss_idx, ss_sign, ms_idx, ms_sign)
    assert dense.shape == (1, L, 2 * L - 1, 2 * L - 1)

    max_err = 0.0
    for l in range(L):
        D_oracle = wigner_D_matrix(
            l, 0.0, beta_val, 0.0, dtype=torch.complex128
        ).real
        D_unpack = dense[0, l, (L - 1) - l: (L - 1) + l + 1, (L - 1) - l: (L - 1) + l + 1].to(torch.float64)
        err = (D_oracle - D_unpack).abs().max().item()
        max_err = max(max_err, err)
    assert max_err < 1e-12, (
        f"L={L} beta={beta_val}: full unpack max err {max_err:.2e}"
    )


def test_unpack_full_handles_batched_betas() -> None:
    """Batched (K=4) unpack must give correct dense per quat."""
    device = torch.device("cpu")
    L = 6
    betas_list = [0.3, 0.7, 1.0, 1.3]
    betas = torch.tensor(betas_list, dtype=torch.float64, device=device)
    pi_minus = torch.tensor([math.pi - b for b in betas_list], dtype=torch.float64, device=device)

    ss_idx, ss_sign = build_unpack_index_and_sign_tables(L, device)
    ms_idx, ms_sign = build_mixed_sign_tables(L, device)
    packed_lt = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)
    packed_gt = wigner_d_gt_half_pi_batched_eager(pi_minus, L, torch.float64, device)
    dense = unpack_wigner_d_full(packed_lt, packed_gt, ss_idx, ss_sign, ms_idx, ms_sign)

    for k, beta_val in enumerate(betas_list):
        for l in range(L):
            D_oracle = wigner_D_matrix(
                l, 0.0, beta_val, 0.0, dtype=torch.complex128
            ).real
            D_unpack = dense[k, l, (L - 1) - l: (L - 1) + l + 1, (L - 1) - l: (L - 1) + l + 1].to(torch.float64)
            err = (D_oracle - D_unpack).abs().max().item()
            assert err < 1e-12, (
                f"k={k} beta={beta_val} l={l}: err {err:.2e}"
            )
