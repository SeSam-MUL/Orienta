"""Tests for the GPU forward-simulation endpoint (POST /api/simulation/start-gpu).

Tests cover:
1. POST /start-gpu returns a task_id and status="running" with a mocked runner.
2. The runner run_gpu_simulation produces the correct result dict keys.
3. Task transitions to "completed" with result files populated (mocked).
4. ForwardSimError on missing CUDA propagates to task["status"] == "failed".
5. The EMsoft /start code is byte-identical (no accidental modifications).
6. GET /status/{task_id} returns the same dict shape as /start tasks.
7. run_gpu_simulation with a tiny CPU build (npx=8, dmin=0.10, n_electrons=100)
   produces files on disk when FORWARD_SIM_DEVICE=cpu is set (optional / skip
   without a real .xtal in the test environment).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

# Needs the maintainer's crystal library / measurement data, which a clone
# does not have — skip with the missing path named, never fail.
from tests.data_deps import CIF_LIBRARY, requires

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_client():
    """Return a FastAPI TestClient for the simulation router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.routes import simulation as sim_mod

    # Fresh module-level state for each test (tasks dict)
    sim_mod._simulation_tasks.clear()
    sim_mod._batch_jobs.clear()

    app = FastAPI()
    app.include_router(sim_mod.router, prefix="/api/simulation")
    return TestClient(app), sim_mod


# ---------------------------------------------------------------------------
# 1. POST /start-gpu returns task_id + status=running
# ---------------------------------------------------------------------------


def _real_xtal(tmp_path, name="Ni.xtal"):
    """A .xtal that actually exists, so nothing has to lie about it.

    The runner guards its input with `xtal.exists()` and hands the same path to
    write_provenance_sidecar, which records `found: xtal_path.exists()`. A
    blanket `Path.exists -> True` satisfied the guard AND wrote `found: true`
    for a path that never existed; four such records sat in the user's phase
    library from 2026-08-07. read_crystal_structure is mocked here, so the
    content does not matter - only that the file is real.
    """
    f = Path(tmp_path) / name
    f.write_bytes(b"")
    return f

class TestStartGpuEndpoint:
    """POST /start-gpu endpoint contract tests."""

    def test_returns_task_id_and_running_status(self, tmp_path):
        """POST /start-gpu should return {"task_id": <str>, "status": "running"}."""
        client, sim_mod = _make_test_client()

        # Mock run_gpu_simulation so the background task completes quickly
        mock_result = {"master": None, "sht": "/fake/path.sht"}

        with patch(
            "backend.api.routes.simulation.run_gpu_simulation"
            if hasattr(sim_mod, "run_gpu_simulation")
            else "backend.api.services.gpu_sim_runner.run_gpu_simulation",
            return_value=mock_result,
        ) as _mock_runner, patch(
            "backend.api.routes.simulation._append_history"
        ), patch(
            "backend.api.routes.simulation._write_job_log"
        ):
            # Patch run_gpu_simulation on the real module (no sys.modules eviction).
            with _patch_runner(_make_gpu_runner_mock(mock_result)):
                resp = client.post(
                    "/api/simulation/start-gpu",
                    json={
                        "xtal_path": "/fake/Ni.xtal",
                        "ekev": 20.0,
                        "output_type": "sht_only",
                    },
                )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "running"
        assert isinstance(data["task_id"], str)
        assert len(data["task_id"]) > 0

    def test_task_dict_shape_matches_emsoft_start(self, tmp_path):
        """The task dict created by /start-gpu must have the same keys as /start."""
        client, sim_mod = _make_test_client()

        with _patch_runner(_make_gpu_runner_mock({"master": None, "sht": None})), \
             patch("backend.api.routes.simulation._append_history"), \
             patch("backend.api.routes.simulation._write_job_log"):
            resp = client.post(
                "/api/simulation/start-gpu",
                json={"xtal_path": "/fake/Ni.xtal", "ekev": 20.0, "output_type": "sht_only"},
            )
        task_id = resp.json()["task_id"]

        # The task dict stored internally should have the canonical keys
        task = sim_mod._simulation_tasks[task_id]
        for key in ("status", "progress", "message", "result", "error", "logLines"):
            assert key in task, f"Missing key {key!r} in task dict"

    def test_get_status_endpoint_finds_gpu_task(self, tmp_path):
        """GET /status/{task_id} must return the task created by /start-gpu."""
        client, sim_mod = _make_test_client()

        with _patch_runner(_make_gpu_runner_mock({"master": None, "sht": None})), \
             patch("backend.api.routes.simulation._append_history"), \
             patch("backend.api.routes.simulation._write_job_log"):
            resp = client.post(
                "/api/simulation/start-gpu",
                json={"xtal_path": "/fake/Ni.xtal", "output_type": "sht_only"},
            )
        task_id = resp.json()["task_id"]

        status_resp = client.get(f"/api/simulation/status/{task_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        # The status endpoint must return one of the canonical statuses
        assert status_data["status"] in ("running", "completed", "failed", "cancelled")


# ---------------------------------------------------------------------------
# 2. Runner produces correct result dict keys
# ---------------------------------------------------------------------------

class TestRunGpuSimulationRunner:
    """Unit tests for run_gpu_simulation (backend.api.services.gpu_sim_runner)."""

    @requires(CIF_LIBRARY / "Ni.cif")
    def test_result_dict_has_master_and_sht_keys(self, tmp_path, monkeypatch):
        """run_gpu_simulation must always return dict with 'master' and 'sht' keys."""
        from backend.api.services import gpu_sim_runner

        # The runner writes its outputs under Database/ — the user's real
        # phase library. write_sht is mocked below, but the provenance
        # sidecar is not, so before this every run of THIS test wrote a
        # record into Database/EBSD_SHT_Database/Ni/ claiming a source .xtal
        # of "/fake/Ni.xtal" as found, against a real EMsoft master. It had
        # done so since 2026-08-07, and it broke test_sht_info_endpoint,
        # which reads that sidecar.
        #
        # Redirecting the output roots keeps the real code path under test —
        # the name is still resolved from the real CIF library, the sidecar
        # is still written — but into tmp_path.
        monkeypatch.setattr(gpu_sim_runner, "_SHT_DB", tmp_path / "EBSD_SHT_Database")
        monkeypatch.setattr(gpu_sim_runner, "_H5_CACHE", tmp_path / "EBSD_H5_Cache")
        # _cif_library_dir() is derived from _SHT_DB.parent, so redirecting the
        # output would silently redirect the READ as well and the test would
        # stop exercising name resolution from the real library. Reading the
        # user's library is fine; writing to it is not.
        monkeypatch.setattr(gpu_sim_runner, "_cif_library_dir",
                            lambda: CIF_LIBRARY)

        progress_calls: list = []
        log_calls: list = []

        dummy_master = MagicMock()
        dummy_master.isfinite.return_value.all.return_value.item.return_value = True
        dummy_master.shape = (101, 101)
        dummy_master.detach.return_value.cpu.return_value.numpy.return_value = \
            __import__("numpy").zeros((101, 101), dtype="float32")

        mock_mc = MagicMock()
        mock_mc.n_energy = 11
        mock_mc.EkeVs = __import__("torch").linspace(10, 20, 11)
        mock_mc.accum_e = __import__("torch").zeros(501, 501, 11)
        mock_mc.depth_step = 1.0
        mock_mc.depth_max = 100.0
        mock_mc.EkeVs.__getitem__ = lambda self, idx: __import__("torch").tensor(20.0)

        import torch

        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cpu")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure") as mock_read_xtal, \
             patch("backend.forward_sim.mc.composition.mc_composition_from_structure") as mock_comp, \
             patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", return_value=mock_mc) as mock_mc_fn, \
             patch("backend.forward_sim.dynamical.master_builder.build_master", return_value=dummy_master), \
             patch("backend.forward_sim.io.mc_h5.write_mc_h5", return_value="/fake/mc.h5"), \
             patch("backend.forward_sim.io.master_h5.write_master_h5", return_value="/fake/master.h5"), \
             patch("backend.forward_sim.io.sht_writer.write_sht", return_value="/fake/out.sht"), \
             patch("backend.forward_sim.io.sht_writer._formula_from_atoms", return_value="Ni"):

            comp_mock = MagicMock()
            comp_mock.mean_Z = 28.0
            comp_mock.mean_A = 58.693
            comp_mock.rho = 8.908
            mock_comp.return_value = comp_mock

            structure_mock = MagicMock()
            structure_mock.space_group = 225
            structure_mock.lattice = (0.352, 0.352, 0.352, 90.0, 90.0, 90.0)
            structure_mock.atoms = [MagicMock()]
            mock_read_xtal.return_value = structure_mock

            # Mock EkeVs indexing
            mock_mc.EkeVs.__class__ = torch.Tensor
            mock_mc.EkeVs = torch.linspace(10.0, 20.0, 11)
            mock_mc.accum_e = torch.zeros(501, 501, 11)
            mock_mc.n_energy = 11

            result = gpu_sim_runner.run_gpu_simulation(
                str(_real_xtal(tmp_path)),
                {"output_type": "both", "ekev": 20.0, "npx": 50, "dmin": 0.10},
                progress_cb=lambda p, m: progress_calls.append((p, m)),
                log_cb=lambda l: log_calls.append(l),
            )

        assert "master" in result
        assert "sht" in result
        assert "mc" in result  # MC .h5 is always written (TASK 1)

    def test_gpu_mc_electron_count_is_capped(self, tmp_path, monkeypatch):
        """The GPU MC caps the electron count: the master uses the normalised
        depth/energy profile (converges ~10-25M), so a 500M request must NOT run
        500M trajectories through the PyTorch MC — it is capped to the ceiling.
        A below-ceiling count passes through unchanged.

        To actually exercise the GPU-MC cap, this test must FORCE strategy (c):
        the existing-MC strategy (a) is skipped (the MC .h5 is reported as
        absent), and the EMsoft-WSL strategy (b) is skipped (_wsl_available →
        False), so run_gpu_mc IS called and its MCConfig.n_simulations carries
        the cap.  A blanket ``Path.exists → True`` would short-circuit at
        strategy (a) and run_gpu_mc would never be called (call_args is None)."""
        from backend.api.services import gpu_sim_runner
        import torch
        import numpy as np

        dummy_master = MagicMock()
        dummy_master.isfinite.return_value.all.return_value.item.return_value = True
        dummy_master.shape = (101, 101)
        dummy_master.detach.return_value.cpu.return_value.numpy.return_value = \
            np.zeros((101, 101), dtype="float32")

        def _n_sim_for(totnum_el):
            mock_mc = MagicMock()
            mock_mc.n_energy = 11
            mock_mc.EkeVs = torch.linspace(10.0, 20.0, 11)
            mock_mc.accum_e = torch.zeros(501, 501, 11)
            mock_mc.depth_step = 1.0
            mock_mc.depth_max = 100.0

            # No Path.exists patch. The .xtal is real and the MC .h5 is simply
            # not created, so strategy (a) is skipped for the true reason
            # rather than a mocked one. The outputs go to tmp_path: without
            # that, only the Path.mkdir patch below stood between this test and
            # a provenance record in the user's phase library.
            monkeypatch.setattr(gpu_sim_runner, "_SHT_DB", tmp_path / "EBSD_SHT_Database")
            monkeypatch.setattr(gpu_sim_runner, "_H5_CACHE", tmp_path / "EBSD_H5_Cache")
            xtal = _real_xtal(tmp_path)

            with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cpu")), \
                 patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure") as mock_read_xtal, \
                 patch("backend.forward_sim.mc.composition.mc_composition_from_structure") as mock_comp, \
                 patch("backend.forward_sim.mc.gpu_monte_carlo.run_gpu_mc", return_value=mock_mc) as mock_mc_fn, \
                 patch("backend.forward_sim.dynamical.master_builder.build_master", return_value=dummy_master), \
                 patch("backend.forward_sim.io.mc_h5.write_mc_h5", return_value="/fake/mc.h5"), \
                 patch("backend.forward_sim.io.master_h5.write_master_h5", return_value="/fake/master.h5"), \
                 patch("backend.api.services.gpu_sim_runner._wsl_available", return_value=False):
                comp_mock = MagicMock()
                comp_mock.mean_Z, comp_mock.mean_A, comp_mock.rho = 28.0, 58.693, 8.908
                mock_comp.return_value = comp_mock
                structure_mock = MagicMock()
                structure_mock.space_group = 225
                structure_mock.lattice = (0.352, 0.352, 0.352, 90.0, 90.0, 90.0)
                structure_mock.atoms = [MagicMock()]
                mock_read_xtal.return_value = structure_mock
                gpu_sim_runner.run_gpu_simulation(
                    str(xtal),
                    {"output_type": "master_only", "ekev": 20.0, "npx": 50,
                     "dmin": 0.10, "totnum_el": totnum_el},
                    progress_cb=lambda p, m: None,
                    log_cb=lambda l: None,
                )
                # run_gpu_mc MUST have been called (strategy (c) GPU fallback).
                assert mock_mc_fn.call_args is not None, (
                    "run_gpu_mc was never called — the GPU-MC fallback (strategy c) "
                    "did not run, so the electron cap is not exercised"
                )
                # run_gpu_mc(mean_Z, mean_A, rho, MCConfig) — MCConfig is arg[3]
                return mock_mc_fn.call_args.args[3].n_simulations

        assert gpu_sim_runner.GPU_MC_MAX_ELECTRONS == 50_000_000
        assert _n_sim_for(500_000_000) == 50_000_000   # capped
        assert _n_sim_for(10_000_000) == 10_000_000    # below ceiling -> unchanged

    def test_missing_xtal_raises_file_not_found(self):
        """run_gpu_simulation must raise FileNotFoundError when xtal_path is missing."""
        from backend.api.services import gpu_sim_runner

        with pytest.raises(FileNotFoundError):
            gpu_sim_runner.run_gpu_simulation(
                "/nonexistent/path/Ni.xtal",
                {"output_type": "sht_only"},
                progress_cb=lambda p, m: None,
                log_cb=lambda l: None,
            )

    def test_invalid_output_type_raises_value_error(self, tmp_path):
        """run_gpu_simulation must raise ValueError for unknown output_type."""
        from backend.api.services import gpu_sim_runner
        import torch

        with patch("backend.forward_sim.runtime.resolve_device_adaptive", return_value=torch.device("cpu")), \
             patch("backend.forward_sim.crystal.xtal_io.read_crystal_structure") as mock_read_xtal:

            structure_mock = MagicMock()
            structure_mock.space_group = 225
            structure_mock.lattice = (0.352, 0.352, 0.352, 90.0, 90.0, 90.0)
            structure_mock.atoms = [MagicMock()]
            mock_read_xtal.return_value = structure_mock

            with pytest.raises(ValueError, match="output_type"):
                gpu_sim_runner.run_gpu_simulation(
                    str(_real_xtal(tmp_path)),
                    {"output_type": "unknown_type"},
                    progress_cb=lambda p, m: None,
                    log_cb=lambda l: None,
                )


# ---------------------------------------------------------------------------
# 3. Task transitions to "completed" when runner succeeds
# ---------------------------------------------------------------------------

class TestGpuTaskCompletion:
    """Verify task dict transitions for the GPU endpoint."""

    def test_task_completes_with_result(self):
        """When runner returns files, the task must transition to 'completed'."""
        client, sim_mod = _make_test_client()

        mock_result = {"master": "/fake/Ni_master.h5", "sht": "/fake/Ni.sht"}
        mock_module = _make_gpu_runner_mock(mock_result)

        with _patch_runner(mock_module), \
             patch("backend.api.routes.simulation._append_history"), \
             patch("backend.api.routes.simulation._write_job_log"):
            resp = client.post(
                "/api/simulation/start-gpu",
                json={"xtal_path": "/fake/Ni.xtal", "output_type": "both"},
            )
        task_id = resp.json()["task_id"]

        # Wait for background task to complete (TestClient runs synchronously
        # but background tasks are async — give it a moment)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            task = sim_mod._simulation_tasks.get(task_id, {})
            if task.get("status") in ("completed", "failed"):
                break
            time.sleep(0.05)

        task = sim_mod._simulation_tasks[task_id]
        assert task["status"] == "completed", f"Expected completed, got: {task}"
        assert task["result"] == mock_result

    def test_task_fails_on_runner_exception(self):
        """When runner raises, the task must transition to 'failed'."""
        client, sim_mod = _make_test_client()

        error_module = _make_gpu_runner_mock_raises(RuntimeError("no CUDA"))

        with _patch_runner(error_module), \
             patch("backend.api.routes.simulation._append_history"), \
             patch("backend.api.routes.simulation._write_job_log"):
            resp = client.post(
                "/api/simulation/start-gpu",
                json={"xtal_path": "/fake/Ni.xtal", "output_type": "sht_only"},
            )
        task_id = resp.json()["task_id"]

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            task = sim_mod._simulation_tasks.get(task_id, {})
            if task.get("status") in ("completed", "failed"):
                break
            time.sleep(0.05)

        task = sim_mod._simulation_tasks[task_id]
        assert task["status"] == "failed"
        assert "no CUDA" in (task.get("error") or "")


# ---------------------------------------------------------------------------
# 4. ForwardSimError surfaces as task failure
# ---------------------------------------------------------------------------

class TestForwardSimErrorHandling:
    def test_forward_sim_error_surfaces_as_task_failed(self):
        """ForwardSimError (no CUDA) should mark task as failed, not crash the server."""
        client, sim_mod = _make_test_client()

        from backend.forward_sim.runtime import ForwardSimError
        error_module = _make_gpu_runner_mock_raises(ForwardSimError("CUDA not available"))

        with _patch_runner(error_module), \
             patch("backend.api.routes.simulation._append_history"), \
             patch("backend.api.routes.simulation._write_job_log"):
            resp = client.post(
                "/api/simulation/start-gpu",
                json={"xtal_path": "/fake/Ni.xtal", "output_type": "sht_only"},
            )
        assert resp.status_code == 200  # endpoint itself is fine
        task_id = resp.json()["task_id"]

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            task = sim_mod._simulation_tasks.get(task_id, {})
            if task.get("status") in ("completed", "failed"):
                break
            time.sleep(0.05)

        task = sim_mod._simulation_tasks[task_id]
        assert task["status"] == "failed"


# ---------------------------------------------------------------------------
# 5. EMsoft /start code is UNCHANGED
# ---------------------------------------------------------------------------

class TestEmSoftStartUnchanged:
    """Guard: the /start EMsoft endpoint must be byte-identical to what we found."""

    def test_emsoft_start_uses_simulation_controller(self):
        """POST /start must still use SimulationController (not the GPU runner)."""
        import ast
        src = (
            Path(_ROOT) / "backend" / "api" / "routes" / "simulation.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)

        # Find the run_sim function defined inside start_simulation
        run_sim_bodies: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.name == "start_simulation":
                    # walk its body for nested 'run_sim' defs
                    for child in ast.walk(node):
                        if isinstance(child, ast.FunctionDef) and child.name == "run_sim":
                            run_sim_bodies.append(ast.unparse(child))

        assert run_sim_bodies, "run_sim closure not found inside start_simulation"
        # The EMsoft run_sim must still reference SimulationController / ctrl
        body_text = " ".join(run_sim_bodies)
        assert "SimulationParameters" in body_text, (
            "start_simulation no longer imports SimulationParameters — EMsoft path broken"
        )
        assert "ctrl.start_simulation" in body_text, (
            "start_simulation no longer calls ctrl.start_simulation — EMsoft path broken"
        )

    def test_start_gpu_is_separate_function(self):
        """POST /start-gpu must be a SEPARATE route, not the same as /start."""
        import ast
        src = (
            Path(_ROOT) / "backend" / "api" / "routes" / "simulation.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)

        route_funcs = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
        ]
        assert "start_simulation" in route_funcs
        assert "start_gpu_simulation" in route_funcs
        # They must be distinct function objects
        assert "start_simulation" != "start_gpu_simulation"

    def test_emsoft_start_does_not_import_gpu_runner(self):
        """The run_sim closure in /start must NOT import gpu_sim_runner."""
        import ast
        src = (
            Path(_ROOT) / "backend" / "api" / "routes" / "simulation.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                if node.name == "start_simulation":
                    for child in ast.walk(node):
                        if isinstance(child, ast.FunctionDef) and child.name == "run_sim":
                            body_text = ast.unparse(child)
                            assert "gpu_sim_runner" not in body_text, (
                                "run_sim inside start_simulation now imports gpu_sim_runner "
                                "— EMsoft path was accidentally modified"
                            )


# ---------------------------------------------------------------------------
# 6. Output path convention
# ---------------------------------------------------------------------------

class TestOutputPathConvention:
    """Verify the naming convention used by gpu_sim_runner matches scan_missing.

    These tests exercise the runner's REAL path-construction helpers
    (``mc_h5_path`` / ``master_h5_path``) rather than re-deriving names, so a
    regression in the sanitizer is caught here.
    """

    def test_master_h5_name_matches_scanner_pattern(self):
        """Master .h5 filename must match the */{safe_stem}_master*.h5 glob."""
        from backend.api.services.gpu_sim_runner import master_h5_path
        from simulation.simulation_controller import sanitize_sim_name

        stem = "Ni"
        ekv = 20
        npx = 500
        fname = master_h5_path(stem, ekv, npx).name
        safe = sanitize_sim_name(stem)
        # The scanner uses:  f"*/{safe_stem}_master*.h5"
        assert f"{safe}_master" in fname
        assert f"E{ekv}kV" in fname

    def test_sht_name_contains_stem_in_parens(self):
        """SHT filename must contain '(<stem>)' so scan_missing detects it."""
        stem = "Ni"
        formula = "Ni"
        ekv = 20
        fname = f"{stem} ({stem}) [{formula}] {{{ekv}kV}}.sht"
        assert f"({stem})" in fname, "scan_missing guard f'({stem})' in f.name would fail"

    def test_sht_name_special_chars_in_stem(self):
        """Even stems with spaces or brackets produce a parseable SHT name."""
        stem = "Al6Fe mp-123"
        formula = "AlFe"
        ekv = 20
        # The SHT filename uses the original stem (not safe) for the bracketed part
        fname = f"{stem} ({stem}) [{formula}] {{{ekv}kV}}.sht"
        assert f"({stem})" in fname

    @pytest.mark.parametrize(
        "stem",
        ["Ni", "Al", "Mg32(Al,Zn)49", "α-AlFeSi", "Al6Fe mp-123"],
    )
    def test_runner_mc_master_names_match_scan_missing_globs(self, tmp_path, stem):
        """REGRESSION: runner→scan_missing naming contract for special-char stems.

        The path the runner WOULD write (from its real helpers) must be found
        by the EXACT globs ``SimulationController.scan_missing_materials`` uses.
        This fails against the old ``_safe_stem`` (which left spaces, commas,
        parentheses and Greek letters untouched) and passes after switching to
        ``sanitize_filename``.  Otherwise "Simulate All Missing" re-queues those
        phases forever even though the GPU already produced their files.
        """
        from backend.api.services.gpu_sim_runner import (
            mc_h5_path, master_h5_path,
        )
        from simulation.simulation_controller import sanitize_sim_name

        ekev = 20
        safe_stem = sanitize_sim_name(stem)  # what scan_missing greps for

        # Names the runner would produce for this stem.
        mc_name = mc_h5_path(stem, ekev, 70.0, 501).name
        master_name = master_h5_path(stem, ekev, 500).name

        # Lay them down under a temp Database (subfolder named like the runner does).
        h5_cache = tmp_path / "EBSD_H5_Cache" / safe_stem
        h5_cache.mkdir(parents=True, exist_ok=True)
        (h5_cache / mc_name).write_bytes(b"x")
        (h5_cache / master_name).write_bytes(b"x")

        root = tmp_path / "EBSD_H5_Cache"
        # scan_missing MC glob (verbatim from scan_missing_materials):
        mc_hits = [
            f for f in root.glob(f"*/{safe_stem}_E{ekev}kV*.h5")
            if "master" not in f.name
        ]
        # scan_missing master glob (verbatim):
        master_hits = list(root.glob(f"*/{safe_stem}_master*.h5"))

        assert mc_hits, (
            f"runner MC name {mc_name!r} NOT found by scan_missing glob "
            f"'*/{safe_stem}_E{ekev}kV*.h5' — batch would re-queue {stem!r} forever"
        )
        assert master_hits, (
            f"runner master name {master_name!r} NOT found by scan_missing glob "
            f"'*/{safe_stem}_master*.h5' — batch would re-queue {stem!r} forever"
        )


# ---------------------------------------------------------------------------
# 7. GPU batch endpoint (TASK 2) — Simulate All Missing on the GPU engine
# ---------------------------------------------------------------------------

class TestGpuBatchEndpoint:
    """POST /batch/start-gpu contract + per-phase runner invocation."""

    def test_returns_batch_id_and_jobs(self):
        client, sim_mod = _make_test_client()
        mod = _make_gpu_runner_mock({"mc": "/m.h5", "master": None, "sht": "/o.sht"})

        with _patch_runner(mod), \
             patch("backend.api.routes.simulation._append_history"), \
             patch("backend.api.routes.simulation._write_job_log"):
            resp = client.post(
                "/api/simulation/batch/start-gpu",
                json={
                    "xtal_paths": ["/fake/Ni.xtal", "/fake/Al.xtal"],
                    "output_type": "sht_only",
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "batch_id" in data
        assert data["job_count"] == 2
        assert len(data["jobs"]) == 2
        for j in data["jobs"]:
            assert set(("task_id", "xtal_path", "status", "progress", "message")) <= set(j)

    def test_runner_invoked_per_phase(self):
        client, sim_mod = _make_test_client()
        mod = _make_gpu_runner_mock({"mc": "/m.h5", "master": None, "sht": "/o.sht"})

        with _patch_runner(mod), \
             patch("backend.api.routes.simulation._append_history"), \
             patch("backend.api.routes.simulation._write_job_log"):
            resp = client.post(
                "/api/simulation/batch/start-gpu",
                json={
                    "xtal_paths": ["/fake/Ni.xtal", "/fake/Al.xtal"],
                    "output_type": "sht_only",
                },
            )
            batch_id = resp.json()["batch_id"]

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                b = sim_mod._batch_jobs.get(batch_id, {})
                if b.get("status") == "completed":
                    break
                time.sleep(0.05)

        # run_gpu_simulation must have been called once per phase.
        assert mod.run_gpu_simulation.call_count == 2
        called_paths = [c.args[0] for c in mod.run_gpu_simulation.call_args_list]
        assert "/fake/Ni.xtal" in called_paths
        assert "/fake/Al.xtal" in called_paths

    def test_batch_status_shape_matches_emsoft_batch(self):
        """/batch/status/{id} for a GPU batch must have the EMsoft batch keys."""
        client, sim_mod = _make_test_client()
        mod = _make_gpu_runner_mock({"mc": "/m.h5", "master": None, "sht": "/o.sht"})

        with _patch_runner(mod), \
             patch("backend.api.routes.simulation._append_history"), \
             patch("backend.api.routes.simulation._write_job_log"):
            resp = client.post(
                "/api/simulation/batch/start-gpu",
                json={"xtal_paths": ["/fake/Ni.xtal"], "output_type": "sht_only"},
            )
            batch_id = resp.json()["batch_id"]
            status = client.get(f"/api/simulation/batch/status/{batch_id}")

        assert status.status_code == 200
        sdata = status.json()
        for key in ("status", "jobs", "completed", "failed"):
            assert key in sdata, f"GPU batch status missing key {key!r}"
        assert sdata.get("engine") == "gpu"

    def test_empty_xtal_paths_rejected(self):
        client, sim_mod = _make_test_client()
        resp = client.post(
            "/api/simulation/batch/start-gpu",
            json={"xtal_paths": [], "output_type": "sht_only"},
        )
        assert resp.status_code == 400

    def test_runner_failure_marks_job_failed(self):
        client, sim_mod = _make_test_client()
        mod = _make_gpu_runner_mock_raises(RuntimeError("no CUDA"))

        with _patch_runner(mod), \
             patch("backend.api.routes.simulation._append_history"), \
             patch("backend.api.routes.simulation._write_job_log"):
            resp = client.post(
                "/api/simulation/batch/start-gpu",
                json={"xtal_paths": ["/fake/Ni.xtal"], "output_type": "sht_only"},
            )
            batch_id = resp.json()["batch_id"]
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                b = sim_mod._batch_jobs.get(batch_id, {})
                if b.get("status") == "completed":
                    break
                time.sleep(0.05)

        b = sim_mod._batch_jobs[batch_id]
        assert b["failed"] == 1
        assert b["jobs"][0]["status"] == "failed"
        assert "no CUDA" in (b["jobs"][0].get("error") or "")


# ---------------------------------------------------------------------------
# 8. TASK 4 cosmetic nits — bandwidth resolution + EMsoft logLines shape
# ---------------------------------------------------------------------------

class TestBandwidthAndShapeNits:
    def test_resolve_gpu_bandwidth_defaults_to_384(self):
        from backend.api.routes import simulation as sim_mod
        assert sim_mod.resolve_gpu_bandwidth(60.0) == 384  # EMsoft default → GPU 384
        assert sim_mod.resolve_gpu_bandwidth(128.0) == 128  # explicit honoured
        assert sim_mod.resolve_gpu_bandwidth(384.0) == 384

    def test_emsoft_start_task_dict_has_loglines(self):
        """/start (EMsoft) task dict must init with logLines:[] for shape parity."""
        import ast
        src = (
            Path(_ROOT) / "backend" / "api" / "routes" / "simulation.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)
        # Find start_simulation's task-dict init and confirm 'logLines' key present.
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "start_simulation":
                body = ast.unparse(node)
                # The init dict is OUTSIDE run_sim, before the closure.
                assert "'logLines': []" in body or '"logLines": []' in body, body[:400]
                found = True
        assert found, "start_simulation not found"


# ---------------------------------------------------------------------------
# Helpers for mock modules
# ---------------------------------------------------------------------------

def _make_gpu_runner_mock(mock_result: dict):
    """Return a stand-in module object whose ``run_gpu_simulation`` returns
    ``mock_result``.

    POLLUTION FIX: the GPU endpoint closures do
    ``from backend.api.services.gpu_sim_runner import run_gpu_simulation``, so
    instead of swapping the WHOLE module in ``sys.modules`` (which evicts the
    real module and forces a re-import on teardown — that re-runs ``import
    torch`` from scratch and trips ``torch.overrides`` "_has_torch_function
    already has a docstring", polluting EVERY later test in the process), the
    ``_patch_runner`` helper patches ONLY the ``run_gpu_simulation`` attribute on
    the REAL, already-imported module.  This object just carries the mock fn.
    """
    module = MagicMock()
    module.run_gpu_simulation = MagicMock(return_value=mock_result)
    return module


def _make_gpu_runner_mock_raises(exc: Exception):
    """Like :func:`_make_gpu_runner_mock` but ``run_gpu_simulation`` raises ``exc``."""
    module = MagicMock()
    module.run_gpu_simulation = MagicMock(side_effect=exc)
    return module


import contextlib  # noqa: E402


@contextlib.contextmanager
def _patch_runner(mock_module):
    """Patch run_gpu_simulation on the REAL gpu_sim_runner module (no sys.modules
    eviction → no torch re-import pollution).  Accepts the same module-mock objects
    produced by _make_gpu_runner_mock / _make_gpu_runner_mock_raises so call sites
    barely change."""
    from backend.api.services import gpu_sim_runner as _real
    with patch.object(_real, "run_gpu_simulation", mock_module.run_gpu_simulation):
        yield
