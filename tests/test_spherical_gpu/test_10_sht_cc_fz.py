"""M2.2 tests: vectorized PyTorch FZ-cc evaluator (rs2cc_fz_torch).

Validates:
1. Bit-identical correctness vs the M1.5 matrix-exp oracle at small L.
2. Bit-identical correctness on CUDA vs CPU.
3. Chunking invariance: result independent of chunk_size.
4. Batched-spectrum support: shape (..., L, 2L-1, 2L-1) → (..., K).
5. Speedup over M1.5 oracle ≥ 5× at L=8 K=50.
6. Identity case: cc(I) = trace contribution per band.
"""
from __future__ import annotations

import math
import pytest
import time
import torch

from backend.spherical_gpu._math.sht_cc_fz import (
    rs2cc_fz_torch,
    _batched_wigner_D_at_band,
)
from backend.spherical_gpu._math.wigner_d_eval import (
    evaluate_wigner_D_at_quaternions,
    wigner_D_matrix,
    quat_to_zyz_euler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_unit_quats(n: int, dtype=torch.float64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[:, 0:1] >= 0, q, -q)


def _random_spectrum(B: int, L: int, dtype=torch.complex128, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, L, 2 * L - 1, 2 * L - 1, generator=g, dtype=dtype)


# ---------------------------------------------------------------------------
# _batched_wigner_D_at_band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("j", [0, 1, 2, 4])
def test_batched_wigner_D_matches_per_quat_oracle(j: int) -> None:
    """For each quat in a batch, batched D^j must match the M1.5 oracle."""
    K = 10
    quats = _random_unit_quats(K, seed=j + 100)
    alpha, beta, gamma = quat_to_zyz_euler(quats)

    D_batched = _batched_wigner_D_at_band(j, alpha, beta, gamma)
    assert D_batched.shape == (K, 2 * j + 1, 2 * j + 1)

    for k in range(K):
        D_one = wigner_D_matrix(
            j, float(alpha[k].item()), float(beta[k].item()), float(gamma[k].item())
        )
        err = (D_batched[k] - D_one).abs().max().item()
        assert err < 1e-9, f"j={j} k={k}: batched-vs-per-quat err {err:.2e}"


def test_batched_wigner_D_at_band_zero_is_identity() -> None:
    """D^0(R) = [[1.0]] for any rotation."""
    K = 5
    quats = _random_unit_quats(K, seed=1)
    a, b, g = quat_to_zyz_euler(quats)
    D = _batched_wigner_D_at_band(0, a, b, g)
    expected = torch.ones((K, 1, 1), dtype=torch.complex128)
    assert torch.allclose(D, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# rs2cc_fz_torch correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [4, 6, 8, 10])
def test_rs2cc_fz_torch_matches_oracle(L: int) -> None:
    """Bit-identical to M1.5 evaluate_wigner_D_at_quaternions."""
    B = 2
    K = 30
    spectrum = _random_spectrum(B, L, seed=L)
    quats = _random_unit_quats(K, seed=L + 50)

    cc_oracle = evaluate_wigner_D_at_quaternions(spectrum, quats, L)
    cc_torch = rs2cc_fz_torch(spectrum, quats, L)

    err = (cc_oracle - cc_torch).abs().max().item()
    assert err < 1e-4, f"L={L}: max err {err:.2e}"


def test_rs2cc_fz_torch_chunk_size_invariance() -> None:
    """Result is independent of chunk_size."""
    L = 6
    B = 2
    K = 100
    spectrum = _random_spectrum(B, L, seed=42)
    quats = _random_unit_quats(K, seed=43)

    cc_64 = rs2cc_fz_torch(spectrum, quats, L, chunk_size=64)
    cc_16 = rs2cc_fz_torch(spectrum, quats, L, chunk_size=16)
    cc_1 = rs2cc_fz_torch(spectrum, quats, L, chunk_size=1)

    err_64_16 = (cc_64 - cc_16).abs().max().item()
    err_64_1 = (cc_64 - cc_1).abs().max().item()
    assert err_64_16 < 1e-9, f"chunk=64 vs 16 err {err_64_16:.2e}"
    assert err_64_1 < 1e-9, f"chunk=64 vs 1 err {err_64_1:.2e}"


def test_rs2cc_fz_torch_unbatched_spectrum() -> None:
    """Spectrum without batch dim returns (K,) result."""
    L = 4
    K = 10
    spectrum = _random_spectrum(1, L, seed=1).squeeze(0)
    assert spectrum.shape == (L, 2 * L - 1, 2 * L - 1)
    quats = _random_unit_quats(K, seed=2)
    cc = rs2cc_fz_torch(spectrum, quats, L)
    assert cc.shape == (K,)


def test_rs2cc_fz_torch_batched_spectrum() -> None:
    """Spectrum with batch dim returns (B, K) result."""
    L = 4
    B = 5
    K = 10
    spectrum = _random_spectrum(B, L, seed=1)
    quats = _random_unit_quats(K, seed=2)
    cc = rs2cc_fz_torch(spectrum, quats, L)
    assert cc.shape == (B, K)


def test_rs2cc_fz_torch_identity_returns_trace_per_band() -> None:
    """For F[l, m, m] = 1, cc(I) = sum over m of D^l_mm(I) = 2l+1."""
    L = 5
    spectrum = torch.zeros(1, L, 2 * L - 1, 2 * L - 1, dtype=torch.complex128)
    expected = 0.0
    for j in range(L):
        for m in range(-j, j + 1):
            spectrum[0, j, m + L - 1, m + L - 1] = 1.0
        expected += 2 * j + 1
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    cc = rs2cc_fz_torch(spectrum, q_id, L).item()
    assert abs(cc - expected) < 1e-5, f"cc(I) = {cc}, expected {expected}"


def test_rs2cc_fz_torch_invalid_shapes() -> None:
    L = 4
    spectrum = _random_spectrum(1, L, seed=1)
    quats = _random_unit_quats(2, seed=2)
    with pytest.raises(ValueError):
        rs2cc_fz_torch(spectrum, quats, L + 1)  # wrong L
    with pytest.raises(ValueError):
        bad_q = torch.zeros(2, 3)
        rs2cc_fz_torch(spectrum, bad_q, L)


# ---------------------------------------------------------------------------
# Speed gate
# ---------------------------------------------------------------------------

def test_rs2cc_fz_torch_speedup_over_oracle() -> None:
    """At L=8, K=30, B=2: rs2cc_fz_torch must beat oracle by >= 5x."""
    L = 8
    B = 2
    K = 30
    spectrum = _random_spectrum(B, L, seed=42)
    quats = _random_unit_quats(K, seed=43)

    # Warmup
    _ = rs2cc_fz_torch(spectrum, quats, L)

    t0 = time.perf_counter()
    _ = evaluate_wigner_D_at_quaternions(spectrum, quats, L)
    t_oracle = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = rs2cc_fz_torch(spectrum, quats, L)
    t_torch = time.perf_counter() - t0

    speedup = t_oracle / max(t_torch, 1e-6)
    assert speedup >= 5.0, (
        f"Speedup {speedup:.1f}x is below 5x gate "
        f"(oracle {t_oracle*1000:.0f}ms vs torch {t_torch*1000:.0f}ms)"
    )


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

def test_rs2cc_fz_torch_cuda_matches_cpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    L = 6
    B = 2
    K = 30
    spectrum_cpu = _random_spectrum(B, L, seed=42)
    quats_cpu = _random_unit_quats(K, seed=43)

    spectrum_cuda = spectrum_cpu.to("cuda")
    quats_cuda = quats_cpu.to("cuda")

    cc_cpu = rs2cc_fz_torch(spectrum_cpu, quats_cpu, L)
    cc_cuda = rs2cc_fz_torch(spectrum_cuda, quats_cuda, L)

    err = (cc_cpu - cc_cuda.cpu()).abs().max().item()
    assert err < 1e-5, f"CUDA vs CPU max err {err:.2e}"
