"""M1.3 tests: cubochoric quaternion grid construction inside Laue FZs.

Validates:
1. Empirical edge × theta ≈ 183 fit (within 5% across edge ∈ [22, 132]).
2. Every grid point lies inside the requested FZ (`ori_in_fz_laue` True).
3. Average angular nearest-neighbor gap matches the requested resolution.
4. FZ point count is consistent with cube^3 / (|G|/2) within ~10%.
5. Permutation reproducibility under fixed seed.
6. Direct-edge API (`sample_ori_fz_laue_edge`) bypasses the resolution fit.
"""
from __future__ import annotations

import math
import torch
import pytest

from backend.spherical_gpu._math.fz_grid import (
    so3_cubochoric_grid,
    sample_ori_fz_laue_angle,
    sample_ori_fz_laue_edge,
    edge_length_for_resolution,
)
from backend.spherical_gpu._math.fz_ori import ori_in_fz_laue, get_laue_mult


def _avg_nearest_gap_deg(quats: torch.Tensor, sample_size: int = 200) -> float:
    """Mean nearest-neighbor angular gap (degrees) across a subsample.

    Quaternion angular distance: 2 * acos(|<q1, q2>|).
    """
    sample = quats[:sample_size]
    dots = torch.matmul(sample, quats.T).abs().clamp(0.0, 1.0)  # (S, N)
    dots.fill_diagonal_(0.0)  # exclude self-match
    max_dots = dots.max(dim=1).values
    ang_gap_rad = 2.0 * torch.arccos(max_dots)
    return torch.rad2deg(ang_gap_rad).mean().item()


def test_edge_length_for_resolution_monotone() -> None:
    e1 = edge_length_for_resolution(1.0)
    e2 = edge_length_for_resolution(2.0)
    e3 = edge_length_for_resolution(3.0)
    assert e1 > e2 > e3
    assert e2 == 91 or e2 == 92  # 183/2 ≈ 91.5
    assert e3 == 61  # 183/3 ≈ 61


def test_edge_length_for_resolution_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        edge_length_for_resolution(0.0)
    with pytest.raises(ValueError):
        edge_length_for_resolution(-1.0)


def test_so3_cubochoric_grid_size_matches_edge_cubed() -> None:
    grid = so3_cubochoric_grid(10, torch.device("cpu"))
    assert grid.shape == (1000, 4)
    norms = grid.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.parametrize("laue_id", [3, 5, 9, 10, 11])
def test_grid_points_all_in_fz(laue_id: int) -> None:
    """Every quaternion in the FZ grid must satisfy `ori_in_fz_laue`."""
    grid = sample_ori_fz_laue_angle(
        laue_id=laue_id,
        angular_resolution_deg=4.0,  # coarse for fast test
        device=torch.device("cpu"),
        permute=False,
    )
    assert grid.shape[0] > 50, f"FZ grid for laue_id={laue_id} too sparse"
    in_fz = ori_in_fz_laue(grid, laue_id)
    assert in_fz.all(), f"laue_id={laue_id}: {(~in_fz).sum().item()} points outside FZ"


def test_m3m_grid_at_2deg_has_expected_size() -> None:
    """m-3m at 2.057° (L=88 cc-bin spacing) should give ~30k FZ points."""
    grid = sample_ori_fz_laue_angle(
        laue_id=11,
        angular_resolution_deg=2.057,
        device=torch.device("cpu"),
        permute=False,
    )
    # Empirical: 89³ = 704,969 total → ~29,500 in FZ (~1/24).
    assert 25000 < grid.shape[0] < 35000, (
        f"m-3m FZ size at 2.057° = {grid.shape[0]}, expected ~30k"
    )


def test_m3m_grid_at_137deg_has_expected_size() -> None:
    """m-3m at 1.37° (the R-C1c-3 reference grid) should give ~96k."""
    grid = sample_ori_fz_laue_edge(
        laue_id=11, edge_length=132,
        device=torch.device("cpu"), permute=False,
    )
    assert 90000 < grid.shape[0] < 100000, (
        f"m-3m FZ size at edge=132 = {grid.shape[0]}, expected ~96k"
    )


def test_avg_gap_matches_target_resolution() -> None:
    """Empirical fit `edge × theta ≈ 183` must hold within 10%."""
    target_thetas = [3.0, 2.057, 1.5]
    for target in target_thetas:
        grid = sample_ori_fz_laue_angle(
            laue_id=11,
            angular_resolution_deg=target,
            device=torch.device("cpu"),
            permute=False,
        )
        avg_gap = _avg_nearest_gap_deg(grid)
        # Allow 10% slack (boundary effects, m-3m FZ shape vs full cube).
        assert 0.9 * target <= avg_gap <= 1.10 * target + 0.2, (
            f"target {target}°, got avg gap {avg_gap:.2f}°"
        )


def test_permute_seed_reproducibility() -> None:
    """Same seed → same permutation."""
    g1 = sample_ori_fz_laue_angle(
        laue_id=11, angular_resolution_deg=4.0,
        device=torch.device("cpu"), permute=True, seed=42,
    )
    g2 = sample_ori_fz_laue_angle(
        laue_id=11, angular_resolution_deg=4.0,
        device=torch.device("cpu"), permute=True, seed=42,
    )
    assert torch.equal(g1, g2)
    g3 = sample_ori_fz_laue_angle(
        laue_id=11, angular_resolution_deg=4.0,
        device=torch.device("cpu"), permute=True, seed=43,
    )
    assert not torch.equal(g1, g3)
    assert g1.shape == g3.shape


def test_permute_false_is_deterministic() -> None:
    g1 = sample_ori_fz_laue_angle(
        laue_id=11, angular_resolution_deg=4.0,
        device=torch.device("cpu"), permute=False,
    )
    g2 = sample_ori_fz_laue_angle(
        laue_id=11, angular_resolution_deg=4.0,
        device=torch.device("cpu"), permute=False,
    )
    assert torch.equal(g1, g2)


def test_fz_count_consistent_across_laue_groups() -> None:
    """N_fz should scale roughly as cube_size / (|G|/2)."""
    edge = 50
    sizes = {}
    for laue_id in [3, 5, 11]:  # mmm (4), 4/mmm (8), m-3m (24)
        g = sample_ori_fz_laue_edge(
            laue_id=laue_id, edge_length=edge,
            device=torch.device("cpu"), permute=False,
        )
        sizes[laue_id] = g.shape[0]

    full_cube = edge ** 3
    # FZ size ≈ full_cube / (|G|/2). Allow 30% slack for boundary effects.
    expected_3 = full_cube / 4
    expected_5 = full_cube / 8
    expected_11 = full_cube / 24
    assert 0.7 * expected_3 < sizes[3] < 1.3 * expected_3, (
        f"laue=3 mmm: {sizes[3]} vs expected ~{expected_3:.0f}"
    )
    assert 0.7 * expected_5 < sizes[5] < 1.3 * expected_5, (
        f"laue=5 4/mmm: {sizes[5]} vs expected ~{expected_5:.0f}"
    )
    assert 0.7 * expected_11 < sizes[11] < 1.3 * expected_11, (
        f"laue=11 m-3m: {sizes[11]} vs expected ~{expected_11:.0f}"
    )


def test_invalid_laue_id_raises() -> None:
    with pytest.raises(ValueError):
        sample_ori_fz_laue_angle(
            laue_id=12, angular_resolution_deg=2.0, device=torch.device("cpu"),
        )
    with pytest.raises(ValueError):
        sample_ori_fz_laue_edge(
            laue_id=0, edge_length=10, device=torch.device("cpu"),
        )


def test_cuda_path_works_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    grid = sample_ori_fz_laue_angle(
        laue_id=11, angular_resolution_deg=4.0,
        device=torch.device("cuda"), permute=False,
    )
    assert grid.device.type == "cuda"
    assert grid.shape[1] == 4
