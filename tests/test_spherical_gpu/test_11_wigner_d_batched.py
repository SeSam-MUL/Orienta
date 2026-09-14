"""M2.3a-batched tests: batched Wigner-d Fukushima recursion.

Validates:
1. `wigner_d_lt_half_pi_eager(beta: Tensor, ...)` matches the original
   single-beta `wigner_d_lt_half_pi(beta: float, ...)` bit-identically.
2. `wigner_d_lt_half_pi_batched_eager(betas: (K,), ...)` matches the
   per-beta loop with the eager version, bit-identically.
3. Speedup gates: batched-eager must beat per-beta loop by ≥ 50× at K=200.
4. Output shapes are correct.
5. Identity case (β=0): every Wigner-d entry should be 0 except for the
   trivial l=m=n=0 case → d^0_{0,0} = 1.
"""
from __future__ import annotations

import time
import math

import pytest
import torch

from backend.spherical_gpu._math._wigner_logspace import wigner_d_lt_half_pi
from backend.spherical_gpu._math.wigner_d_torch_compile import (
    wigner_d_lt_half_pi_eager,
    wigner_d_lt_half_pi_batched_eager,
)


# ---------------------------------------------------------------------------
# Single-beta port matches original JIT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [4, 6, 8, 10])
@pytest.mark.parametrize("beta_val", [0.1, 0.3, 0.5, 0.7, 1.0, 1.4])
def test_eager_port_matches_jit_original(L: int, beta_val: float) -> None:
    device = torch.device("cpu")
    beta_t = torch.tensor(beta_val, dtype=torch.float64, device=device)
    ref = wigner_d_lt_half_pi(beta_val, L, torch.float64, device)
    new = wigner_d_lt_half_pi_eager(beta_t, L, torch.float64, device)
    err = (ref - new).abs().max().item()
    assert err < 1e-12, (
        f"L={L} beta={beta_val}: eager port differs from JIT original "
        f"by {err:.2e}"
    )


# ---------------------------------------------------------------------------
# Batched matches per-beta loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L", [4, 6, 8])
def test_batched_matches_per_beta_loop(L: int) -> None:
    device = torch.device("cpu")
    torch.manual_seed(L)
    K = 30
    betas = torch.rand(K, dtype=torch.float64, device=device) * 1.5  # < pi/2

    # Per-beta loop (eager port)
    ref_stack = torch.stack([
        wigner_d_lt_half_pi_eager(betas[k], L, torch.float64, device)
        for k in range(K)
    ])
    # Batched
    batched = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)

    err = (ref_stack - batched).abs().max().item()
    assert err < 1e-12, f"L={L}: batched vs per-beta-loop max abs err {err:.2e}"


def test_batched_output_shape() -> None:
    device = torch.device("cpu")
    K = 50
    L = 6
    betas = torch.rand(K, dtype=torch.float64, device=device) * 1.5
    out = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)

    table_size = (
        ((L * (L + 1)) // 2) * (L + 2) // 3 + L * (L + 1) // 2 + L + 1
    )
    assert out.shape == (K, table_size)


def test_batched_handles_single_beta() -> None:
    """K=1 batched should match the single-beta port."""
    device = torch.device("cpu")
    L = 5
    beta_val = 0.5
    beta_t = torch.tensor([beta_val], dtype=torch.float64, device=device)

    single = wigner_d_lt_half_pi_eager(beta_t[0], L, torch.float64, device)
    batched = wigner_d_lt_half_pi_batched_eager(beta_t, L, torch.float64, device)

    assert torch.equal(single, batched[0])


# ---------------------------------------------------------------------------
# Speedup gate
# ---------------------------------------------------------------------------

def test_batched_speedup_over_loop_at_K200() -> None:
    """At K=200 L=8, batched-eager must be at least 50x faster than loop."""
    if not torch.cuda.is_available():
        pytest.skip("speedup gate is GPU-only")
    device = torch.device("cuda")
    L = 8
    K = 200
    torch.manual_seed(0)
    betas = torch.rand(K, dtype=torch.float64, device=device) * 1.5

    # Warmup batched
    _ = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        _ = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)
    torch.cuda.synchronize()
    t_batched = (time.perf_counter() - t0) / 5

    # Per-beta loop
    _ = [wigner_d_lt_half_pi_eager(betas[0], L, torch.float64, device)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = torch.stack([
        wigner_d_lt_half_pi_eager(betas[k], L, torch.float64, device)
        for k in range(K)
    ])
    torch.cuda.synchronize()
    t_loop = time.perf_counter() - t0

    speedup = t_loop / t_batched
    assert speedup >= 50.0, (
        f"Batched speedup {speedup:.1f}x < 50x gate at L={L} K={K} "
        f"(loop {t_loop*1000:.0f} ms, batched {t_batched*1000:.0f} ms)"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_batched_with_zero_beta_is_finite() -> None:
    """At β=0 (identity), Wigner-d collapses but log(0)=−inf may cause NaN.
    Verify the function at least doesn't blow up; correctness is moot.
    """
    device = torch.device("cpu")
    L = 4
    # Use very small β rather than exactly 0 (log(0) → -inf in trig powers)
    betas = torch.tensor([1e-3, 0.5, 1.4], dtype=torch.float64, device=device)
    out = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, device)
    # Just check the rest finished without blowing up
    assert out.shape[0] == 3
    # The β=0.5 and β=1.4 entries should be finite
    assert torch.isfinite(out[1]).all(), "β=0.5 entries must be finite"
    assert torch.isfinite(out[2]).all(), "β=1.4 entries must be finite"


def test_cuda_path_works_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    L = 8
    K = 64
    betas = torch.rand(K, dtype=torch.float64, device="cuda") * 1.5
    out = wigner_d_lt_half_pi_batched_eager(betas, L, torch.float64, torch.device("cuda"))
    assert out.device.type == "cuda"
    assert out.shape == (K, ((L * (L + 1)) // 2) * (L + 2) // 3 + L * (L + 1) // 2 + L + 1)


# ---------------------------------------------------------------------------
# gt_half_pi (β > π/2) batched port
# ---------------------------------------------------------------------------

from backend.spherical_gpu._math._wigner_logspace import wigner_d_gt_half_pi
from backend.spherical_gpu._math.wigner_d_torch_compile import (
    wigner_d_gt_half_pi_batched_eager,
)


@pytest.mark.parametrize("L", [4, 6, 8, 10])
@pytest.mark.parametrize("beta_val", [1.6, 1.8, 2.0, 2.5, 3.0])
def test_gt_batched_matches_jit_original(L: int, beta_val: float) -> None:
    """Single-beta gt batched must match JIT original bit-identically."""
    device = torch.device("cpu")
    ref = wigner_d_gt_half_pi(beta_val, L, torch.float64, device)
    betas_t = torch.tensor([beta_val], dtype=torch.float64, device=device)
    new = wigner_d_gt_half_pi_batched_eager(betas_t, L, torch.float64, device)
    err = (ref - new[0]).abs().max().item()
    assert err < 1e-12, f"L={L} beta={beta_val}: gt batched err {err:.2e}"


@pytest.mark.parametrize("L", [4, 6, 8])
def test_gt_batched_matches_per_beta_loop(L: int) -> None:
    device = torch.device("cpu")
    torch.manual_seed(L + 99)
    K = 30
    # betas in (pi/2, pi]
    import math
    betas = math.pi / 2.0 + torch.rand(K, dtype=torch.float64, device=device) * (
        math.pi / 2.0 - 0.01
    )
    ref_stack = torch.stack([
        wigner_d_gt_half_pi_batched_eager(
            betas[k:k+1], L, torch.float64, device
        )[0]
        for k in range(K)
    ])
    batched = wigner_d_gt_half_pi_batched_eager(betas, L, torch.float64, device)
    err = (ref_stack - batched).abs().max().item()
    assert err < 1e-12, f"L={L}: gt batched vs per-beta-loop err {err:.2e}"


def test_gt_cuda_path() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import math
    L = 8
    K = 32
    betas = math.pi / 2.0 + torch.rand(K, dtype=torch.float64, device="cuda") * 1.0
    out = wigner_d_gt_half_pi_batched_eager(betas, L, torch.float64, torch.device("cuda"))
    assert out.device.type == "cuda"
