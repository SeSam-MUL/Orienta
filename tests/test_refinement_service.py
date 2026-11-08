"""Tests for backend/api/services/refinement.py — service-layer refinement."""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _clear_refinement_registry():
    """Prevent cross-test pollution of JOB_REGISTRY."""
    try:
        from backend.api.services.refinement import JOB_REGISTRY
        JOB_REGISTRY.clear()
        yield
        JOB_REGISTRY.clear()
    except ImportError:
        yield


@pytest.fixture(autouse=True)
def _private_result_registry(monkeypatch):
    """The orchestrator stores results in the process-wide registry and makes
    the last one active; without this, later test files in the same run find a
    stored result they did not expect."""
    from collections import OrderedDict
    from backend.api.routes import indexing
    monkeypatch.setattr(indexing, "_result_registry", OrderedDict())
    monkeypatch.setattr(indexing, "_active_result_id", None)


def _mock_renderer_returns(I):
    """Build a mock renderer that mirrors the (B, H, W) batch as per-index renders.

    The refinement service was refactored to call ``renderer.render`` once
    per pixel (``render_batch`` cannot do per-pixel PC). The mock
    therefore exposes a ``render`` side_effect that returns ``I[i]`` on the
    i-th call, cycling. ``render_batch`` is kept too because the orchestrator
    path still calls it where the test for compute_full_refinement patches a
    different mock entirely.
    """
    r = MagicMock()
    r.render_batch.return_value = I

    counter = {"i": 0}
    if I.ndim == 3:
        slices = [I[i] for i in range(I.shape[0])]
    else:
        slices = [I]

    def _render(**kwargs):
        idx = counter["i"] % len(slices)
        counter["i"] += 1
        return slices[idx]

    r.render.side_effect = _render
    return r


def test_refine_per_pixel_converges_on_identity():
    """If I_exp == I_sim(q0, p0), refinement converges to delta omega approx 0, Delta p approx 0, NCC approx 1."""
    from backend.api.services.refinement import refine_per_pixel
    # Synthetic batch: 2 pixels, 8x8 patterns
    B, H, W = 2, 8, 8
    I_exp = torch.randn(B, H, W, dtype=torch.float64) + 1.0
    # Renderer returns exactly I_exp - perfect match at init
    grid = MagicMock(); grid.sht_path = "/fake"
    renderer = _mock_renderer_returns(I_exp.to(torch.float32))
    q0 = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=torch.float64)
    p0 = torch.tensor([[50.0, 50.0, 15000.0], [50.0, 50.0, 15000.0]], dtype=torch.float64)
    out = refine_per_pixel(
        renderer=renderer, grid=grid,
        I_exp=I_exp,
        q0=q0, pc0=p0,
        detector_shape=(H, W),
        pixel_size_um=70.0, tilt_deg=70.0, det_tilt_deg=0.0,
        max_iter=20,
    )
    # Each pixel returns dict-like or tuple: omega, dpc, ncc, iters, flag, J_p
    assert out["omega"].shape == (B, 3)
    assert out["dpc"].shape == (B, 3)
    assert out["ncc_final"].shape == (B,)
    np.testing.assert_allclose(out["omega"].numpy(), 0.0, atol=1e-3)
    np.testing.assert_allclose(out["dpc"].numpy(), 0.0, atol=1e-3)
    np.testing.assert_allclose(out["ncc_final"].numpy(), 1.0, atol=1e-3)
    assert (out["convergence_flag"] == 1).all()   # all converged


def test_refine_per_pixel_respects_max_iter():
    """If renderer returns un-matching patterns, max_iter is the upper bound."""
    from backend.api.services.refinement import refine_per_pixel
    B, H, W = 1, 8, 8
    I_exp = torch.randn(B, H, W, dtype=torch.float64)
    I_sim = torch.randn(B, H, W, dtype=torch.float64)  # different!
    grid = MagicMock()
    renderer = _mock_renderer_returns(I_sim.to(torch.float32))
    q0 = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
    p0 = torch.tensor([[50.0, 50.0, 15000.0]], dtype=torch.float64)
    out = refine_per_pixel(
        renderer=renderer, grid=grid,
        I_exp=I_exp, q0=q0, pc0=p0,
        detector_shape=(H, W), pixel_size_um=70.0, tilt_deg=70.0,
        det_tilt_deg=0.0,
        max_iter=5,
    )
    # iter_used <= max_iter
    assert (out["iter_used"] <= 5).all()
    # Either converged or max_iter (flag in {1, 2}) - not failed (0)
    assert ((out["convergence_flag"] == 1) | (out["convergence_flag"] == 2)).all()


def test_refine_per_pixel_returns_jp_for_stage2():
    """Stage 1 must return per-pixel Jacobian d r/d PC for Stage 2."""
    from backend.api.services.refinement import refine_per_pixel
    B, H, W = 2, 8, 8
    I_exp = torch.randn(B, H, W, dtype=torch.float64)
    grid = MagicMock()
    renderer = _mock_renderer_returns(I_exp.to(torch.float32))
    q0 = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=torch.float64)
    p0 = torch.tensor([[50.0, 50.0, 15000.0], [50.0, 50.0, 15000.0]], dtype=torch.float64)
    out = refine_per_pixel(
        renderer=renderer, grid=grid,
        I_exp=I_exp, q0=q0, pc0=p0,
        detector_shape=(H, W), pixel_size_um=70.0, tilt_deg=70.0,
        det_tilt_deg=0.0,
        max_iter=3,
    )
    # J_p: (B, 3) - partial derivative of residual r w.r.t. each PC component
    assert out["J_p"].shape == (B, 3)
    assert out["J_p"].dtype == torch.float64


def test_solve_smooth_pc_lambda_zero_reproduces_stage1():
    """With λ=0 (no smoothness), the Stage 2 solve returns Stage-1 PC values."""
    from backend.api.services.refinement import solve_smooth_pc
    H, W = 3, 3
    n = H * W
    np.random.seed(42)
    pc_stage1 = np.random.randn(n, 3) * 0.5 + np.array([50.0, 50.0, 15000.0])
    # Jacobian: arbitrary non-zero (otherwise data term is degenerate)
    J_p = np.full((n, 3), 0.1, dtype=np.float64)
    r_stage1 = np.zeros(n, dtype=np.float64)   # residuals already small
    mask = np.ones((H, W), dtype=bool)
    pc_refined = solve_smooth_pc(
        pc_stage1=pc_stage1, r_stage1=r_stage1, J_p=J_p,
        mask=mask, lambda_=0.0,
    )
    # With λ=0 and r=0, the optimum is pc = pc_stage1 (data term minimized at init)
    np.testing.assert_allclose(pc_refined, pc_stage1, atol=1e-6)


def test_solve_smooth_pc_lambda_large_returns_constant():
    """As λ → ∞, the smoothness term dominates and PC becomes spatially constant."""
    from backend.api.services.refinement import solve_smooth_pc
    H, W = 3, 3
    n = H * W
    pc_stage1 = np.random.RandomState(0).randn(n, 3) * 0.5 + np.array([50.0, 50.0, 15000.0])
    J_p = np.full((n, 3), 0.1, dtype=np.float64)
    r_stage1 = np.zeros(n, dtype=np.float64)
    mask = np.ones((H, W), dtype=bool)
    pc_refined = solve_smooth_pc(
        pc_stage1=pc_stage1, r_stage1=r_stage1, J_p=J_p,
        mask=mask, lambda_=1e6,
    )
    # Each component should be approximately constant across pixels
    for axis in range(3):
        spread = pc_refined[:, axis].max() - pc_refined[:, axis].min()
        rng = pc_stage1[:, axis].max() - pc_stage1[:, axis].min()
        assert spread < 0.1 * rng, f"axis {axis}: spread {spread} not << rng {rng}"


def test_solve_smooth_pc_shape():
    """Output shape matches pc_stage1 shape (N, 3)."""
    from backend.api.services.refinement import solve_smooth_pc
    H, W = 4, 5
    n = H * W
    pc = np.zeros((n, 3))
    pc_out = solve_smooth_pc(
        pc_stage1=pc, r_stage1=np.zeros(n), J_p=np.ones((n, 3)) * 0.1,
        mask=np.ones((H, W), dtype=bool), lambda_=0.01,
    )
    assert pc_out.shape == (n, 3)


def test_refine_r_only_converges_on_identity():
    """R-only refinement matches Stage 1 behavior with PC frozen."""
    from backend.api.services.refinement import refine_r_only
    B, H, W = 2, 8, 8
    I_exp = torch.randn(B, H, W, dtype=torch.float64) + 1.0
    grid = MagicMock()
    renderer = _mock_renderer_returns(I_exp.to(torch.float32))
    q0 = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=torch.float64)
    pc_fixed = torch.tensor([[50.0, 50.0, 15000.0], [50.0, 50.0, 15000.0]], dtype=torch.float64)
    out = refine_r_only(
        renderer=renderer, grid=grid,
        I_exp=I_exp, q0=q0, pc_fixed=pc_fixed,
        detector_shape=(H, W), pixel_size_um=70.0, tilt_deg=70.0,
        det_tilt_deg=0.0,
        max_iter=10,
    )
    assert out["omega"].shape == (B, 3)
    assert "dpc" not in out                # R-only: no PC delta in output
    np.testing.assert_allclose(out["omega"].numpy(), 0.0, atol=1e-3)
    assert (out["convergence_flag"] == 1).all()


def test_refinement_job_lifecycle():
    from backend.api.services.refinement import RefinementJob, JOB_REGISTRY
    job = RefinementJob(result_id="r1", smoothness_lambda=0.01)
    assert job.state == "running"
    job.set_progress(0.5, current_stage="stage_2", outer_iter=1)
    assert job.progress == 0.5
    assert job.current_stage == "stage_2"
    job.mark_completed()
    assert job.state == "completed"
    assert JOB_REGISTRY[job.job_id] is job


def test_refinement_job_cancel_sets_event():
    from backend.api.services.refinement import RefinementJob
    job = RefinementJob(result_id="r1", smoothness_lambda=0.01)
    job.cancel()
    assert job.cancel_event.is_set()
    assert job.state == "cancelled"


def test_refinement_job_lru_eviction():
    from backend.api.services.refinement import RefinementJob, JOB_REGISTRY, JOB_REGISTRY_CAPACITY
    JOB_REGISTRY.clear()
    ids = [RefinementJob(result_id=f"r{i}", smoothness_lambda=0.01).job_id
           for i in range(JOB_REGISTRY_CAPACITY + 2)]
    assert len(JOB_REGISTRY) == JOB_REGISTRY_CAPACITY
    assert ids[0] not in JOB_REGISTRY
    assert ids[-1] in JOB_REGISTRY


def test_refinement_job_mark_completed_respects_cancel():
    """If cancel was called before mark_completed, state stays 'cancelled'."""
    from backend.api.services.refinement import RefinementJob
    job = RefinementJob(result_id="r1", smoothness_lambda=0.01)
    job.cancel()
    job.mark_completed()
    assert job.state == "cancelled"


def _make_fake_indexed_result(n_rows=4, n_cols=4):
    """Tiny IndexingResult fully indexed to phase 0.

    A real ``IndexingResult`` on a real orix ``CrystalMap``: the refinement
    builds its own result from the parent's map, and a MagicMock parent hid
    for months that the result it registered was not one (see
    ``tests/test_refined_result_listing.py``).
    """
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    from indexing_controller import IndexingMethod, IndexingResult

    n_pix = n_rows * n_cols
    xmap = CrystalMap(
        rotations=Rotation(np.tile([1.0, 0.0, 0.0, 0.0], (n_pix, 1))),
        phase_id=np.zeros(n_pix, dtype=np.int32),
        x=np.tile(np.arange(n_cols, dtype=np.float64), n_rows),
        y=np.repeat(np.arange(n_rows, dtype=np.float64), n_cols),
        phase_list=PhaseList(phases=[
            Phase(name="Al", space_group=225, point_group="m-3m")]),
    )
    return IndexingResult(
        xmap=xmap,
        selection_mask=np.ones((n_rows, n_cols), dtype=bool),
        original_shape=(n_rows, n_cols),
        method=IndexingMethod.SPHERICAL,
        confidence_scores=np.full(n_pix, 0.8, dtype=np.float32),
        metadata={
            "indexing_method": "spherical",
            "sht_paths_by_phase": {0: "/fake.sht"},
            "detector_geometry": {
                "pat_height": 8, "pat_width": 8,
                "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.6,
                "vendor": "Bruker", "pixel_size": 70.0,
                "sample_tilt": 70.0, "binning": 1,
            },
        },
    )


def test_compute_full_refinement_writes_metadata():
    """End-to-end refinement on a tiny fake result writes refinement block."""
    from backend.api.services import refinement as rf
    result = _make_fake_indexed_result(n_rows=4, n_cols=4)
    fake_grid = MagicMock(); fake_grid.sht_path = "/fake.sht"
    fake_renderer = MagicMock()
    # Per-pixel render path: provide both batch + single-pattern
    # mocks. Both return uniform ones (NCC will be 0/degenerate or 1).
    fake_renderer.render_batch.return_value = torch.ones(16, 8, 8, dtype=torch.float32)
    fake_renderer.render.return_value = torch.ones(8, 8, dtype=torch.float32)

    with patch.object(rf, "load_or_get_phase", return_value=fake_grid), \
         patch.object(rf, "get_renderer", return_value=fake_renderer), \
         patch.object(rf, "_get_experimental_pattern",
                      return_value=np.ones((8, 8), dtype=np.float32)):
        rf.compute_full_refinement(result, smoothness_lambda=0.01)

    block = result.metadata["refinement"]
    assert "refined_result_id" in block
    assert block["smoothness_lambda"] == 0.01
    assert block["refined_ncc_map"].shape == (4, 4)
    assert block["convergence_status_map"].shape == (4, 4)
    assert block["orientation_delta_map"].shape == (4, 4)
    assert block["pc_delta_x_map"].shape == (4, 4)
    assert block["pc_delta_y_map"].shape == (4, 4)
    assert block["pc_delta_l_map"].shape == (4, 4)
    assert "summary" in block


def test_compute_full_refinement_registers_refined_result():
    """The orchestrator adds a separate refined IndexingResult to _result_registry."""
    from backend.api.services import refinement as rf
    from backend.api.routes import indexing as ind
    result = _make_fake_indexed_result()
    rid = ind._store_result(result, "spherical")
    fake_grid = MagicMock(); fake_renderer = MagicMock()
    fake_renderer.render_batch.return_value = torch.ones(16, 8, 8, dtype=torch.float32)
    fake_renderer.render.return_value = torch.ones(8, 8, dtype=torch.float32)

    with patch.object(rf, "load_or_get_phase", return_value=fake_grid), \
         patch.object(rf, "get_renderer", return_value=fake_renderer), \
         patch.object(rf, "_get_experimental_pattern",
                      return_value=np.ones((8, 8), dtype=np.float32)):
        rf.compute_full_refinement(result, smoothness_lambda=0.01)

    refined_id = result.metadata["refinement"]["refined_result_id"]
    refined = ind._result_registry[refined_id]
    assert refined is not None
    assert refined.metadata["is_refined"] is True
    assert refined.metadata["refinement_of"] == rid
    assert refined.metadata["indexing_method"] == "spherical"   # preserved
    assert refined.metadata["pc_per_pixel"].shape == (4, 4, 3)


def test_compute_full_refinement_rejects_hough():
    from backend.api.services import refinement as rf
    result = _make_fake_indexed_result()
    result.metadata["indexing_method"] = "hough"
    with pytest.raises(Exception, match="spherical or dictionary"):
        rf.compute_full_refinement(result, smoothness_lambda=0.01)
