"""M1.2 + M1.6 tests: cubochoric coordinate pipeline (cu ↔ qu).

Validates the forward `cu2qu` and inverse `qu2cu` mappings (Roca/Morawiec/
De Graef 2014) by:
1. Round-trip identity at the cubochoric origin (cu=0 → identity quaternion).
2. Round-trip error on 10000 random quaternions: max FP64 error < 1e-6.
3. Round-trip error on 10000 random cubochoric points: max < 1e-6.
4. Forward map preserves unit-quaternion norm.
5. cu lies in the cube [-π^(2/3)/2, +π^(2/3)/2]^3 for valid inputs.
"""
from __future__ import annotations

import torch
import pytest

from backend.spherical_gpu._math.cubochoric import (
    cu2qu, qu2cu, cu2ho, qu2ho, ho2cu, ho2ax, ax2qu, CUBE_HALF_EDGE,
)
from backend.spherical_gpu._math.fz_ori import qu_std


def _random_unit_quats(n: int, dtype=torch.float64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    return qu_std(q)


def _random_cubochoric_points(
    n: int, dtype=torch.float64, seed: int = 0, fill: float = 0.95,
) -> torch.Tensor:
    """Random points uniformly in [-fill * half_edge, +fill * half_edge]^3.

    `fill < 1` keeps us strictly inside the cube to avoid edge instability.
    """
    g = torch.Generator().manual_seed(seed)
    cu = (torch.rand(n, 3, generator=g, dtype=dtype) - 0.5) * (2.0 * fill * CUBE_HALF_EDGE)
    return cu


def test_cube_half_edge_constant() -> None:
    expected = (torch.pi ** (2.0 / 3.0)) / 2.0
    assert abs(CUBE_HALF_EDGE - expected) < 1e-12


def test_cu_origin_is_identity() -> None:
    """cu = (0, 0, 0) must map to the identity quaternion (1, 0, 0, 0)."""
    cu0 = torch.zeros(1, 3, dtype=torch.float64)
    q = cu2qu(cu0)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(q, expected, atol=1e-12)


def test_qu_identity_is_cu_origin() -> None:
    """Identity quaternion (1, 0, 0, 0) must map to cu = (0, 0, 0)."""
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    cu = qu2cu(q_id)
    assert torch.allclose(cu, torch.zeros(1, 3, dtype=torch.float64), atol=1e-12)


def test_cu2qu_preserves_unit_norm() -> None:
    cu = _random_cubochoric_points(1000, seed=42)
    q = cu2qu(cu)
    norms = q.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
        f"cu2qu produced non-unit quats: norm range "
        f"[{norms.min().item()}, {norms.max().item()}]"
    )


def test_round_trip_quaternion_to_cu_and_back() -> None:
    """quat → cu → quat must round-trip with max error < 1e-6 (FP64)."""
    q = _random_unit_quats(10000, dtype=torch.float64, seed=42)
    cu = qu2cu(q)
    q_round = qu_std(cu2qu(cu))

    # A quaternion q and its negation -q represent the SAME rotation.
    # Compare both directions, take the smaller.
    diff_pos = (q - q_round).norm(dim=-1)
    diff_neg = (q + q_round).norm(dim=-1)
    err = torch.minimum(diff_pos, diff_neg)

    max_err = err.max().item()
    mean_err = err.mean().item()
    assert max_err < 1e-6, (
        f"qu round-trip max err {max_err:.2e} exceeds 1e-6 (mean {mean_err:.2e})"
    )


def test_round_trip_cu_to_quat_and_back() -> None:
    """cu → quat → cu must round-trip with max error < 1e-6 (FP64).

    Excludes cube-corner regions where the map is anisotropic and
    numerical precision degrades; use `fill=0.95` to stay inside.
    """
    cu = _random_cubochoric_points(10000, dtype=torch.float64, seed=99, fill=0.95)
    q = cu2qu(cu)
    cu_round = qu2cu(q)
    err = (cu - cu_round).norm(dim=-1)

    max_err = err.max().item()
    mean_err = err.mean().item()
    assert max_err < 1e-6, (
        f"cu round-trip max err {max_err:.2e} exceeds 1e-6 (mean {mean_err:.2e})"
    )


def test_qu2cu_lies_in_cube() -> None:
    """qu2cu output must lie in [-CUBE_HALF_EDGE, +CUBE_HALF_EDGE]^3.

    Allows a tiny float-edge slack (1e-9) for boundary cases.
    """
    q = _random_unit_quats(1000, dtype=torch.float64, seed=200)
    cu = qu2cu(q)
    slack = 1e-9
    assert torch.all(cu.abs() <= CUBE_HALF_EDGE + slack), (
        f"qu2cu produced cu outside cube; max abs {cu.abs().max().item()} "
        f"vs CUBE_HALF_EDGE = {CUBE_HALF_EDGE}"
    )


def test_intermediate_homochoric_is_inside_ball() -> None:
    """Homochoric coordinates must lie inside the homochoric ball of
    radius (3π/4)^(1/3)."""
    R_HO = (3.0 * torch.pi / 4.0) ** (1.0 / 3.0)
    cu = _random_cubochoric_points(1000, dtype=torch.float64, seed=300, fill=0.95)
    ho = cu2ho(cu)
    radii = ho.norm(dim=-1)
    slack = 1e-9
    assert torch.all(radii <= R_HO + slack), (
        f"cu2ho produced ho outside the ball of radius {R_HO}; "
        f"max radius {radii.max().item()}"
    )


def test_fp32_round_trip_within_fp32_eps() -> None:
    """FP32 round-trip should still be reasonable (looser tolerance)."""
    q = _random_unit_quats(1000, dtype=torch.float32, seed=400)
    cu = qu2cu(q)
    q_round = qu_std(cu2qu(cu))
    diff_pos = (q - q_round).norm(dim=-1)
    diff_neg = (q + q_round).norm(dim=-1)
    err = torch.minimum(diff_pos, diff_neg)
    max_err = err.max().item()
    # FP32 polynomial fit (9 terms) is good to ~1e-4 (machine eps amplified
    # through the cu2ho square-root chain).
    assert max_err < 5e-3, f"FP32 round-trip max err {max_err:.2e} > 5e-3"


def test_cuda_path_works_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    q = _random_unit_quats(64, dtype=torch.float64, seed=500).cuda()
    cu = qu2cu(q)
    assert cu.device.type == "cuda"
    q_round = cu2qu(cu)
    assert q_round.device.type == "cuda"
    assert q_round.shape == q.shape
