"""Loading a phase, a grid calibration and a detector Apply drop the simulated results.

``_sim_cache`` holds, per calibration pattern, the Kikuchi bands and the CI the page
shows when the pattern is selected; ``GET /pattern/{idx}/result`` answers from it
before it indexes. Each of the three actions changed what those results depend on
(the phase, the PC map, the detector) and cleared the controller's own cache, but
not this one, so after them selecting a pattern still showed the old bands and CI.
(``/phase/add``, ``/phase/remove``, ``/detector/update-pc`` and a refine already
cleared it.)
"""
import asyncio

import numpy as np
import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient
from kikuchipy.detectors import EBSDDetector

SHAPE = (60, 60)
STALE = {"ci": 0.77, "phase_name": "old", "segments": [[[1.0, 2.0], [3.0, 4.0]]], "n_bands": 1}


def _cif(tmp_path):
    cif = tmp_path / "Ni.cif"
    cif.write_text(
        "data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n", encoding="utf-8")
    return str(cif)


class _Signal:
    def __init__(self):
        self.data = np.zeros((4, 4) + SHAPE, dtype=np.uint8)


@pytest.fixture
def env(monkeypatch, tmp_path):
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from backend.api.services.calibration_store import CalibrationStore
    monkeypatch.setattr(R, "_sessions", {})
    monkeypatch.setattr(R, "_parked_sessions", {})
    monkeypatch.setattr(R, "calibration_store", CalibrationStore())
    monkeypatch.setattr(R, "_optimization_active", False)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", None, raising=False)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: _Signal())
    ctrl = R._get_controller()
    ctrl.load_phase(_cif(tmp_path))
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0))
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.uint8))
    R._sim_cache[0] = dict(STALE)
    app = FastAPI()
    app.include_router(R.router, prefix="/api/pc")
    n_tasks = len(R._optimization_tasks)
    yield TestClient(app), R, ctrl, tmp_path
    R._optimization_active = False
    while len(R._optimization_tasks) > n_tasks:
        R._optimization_tasks.popitem()


def test_loading_a_phase_drops_the_simulated_results(env):
    c, R, ctrl, tmp_path = env
    assert 0 in R._sim_cache
    r = c.post("/api/pc/phase/load", json={"cif_path": _cif(tmp_path)})
    assert r.status_code == 200, r.text
    assert 0 not in R._sim_cache and len(R._sim_cache) == 0
    assert ctrl.cache == {}                        # as the controller's own cache already was


def test_a_failed_phase_load_keeps_them(env):
    c, R, ctrl, tmp_path = env
    r = c.post("/api/pc/phase/load", json={"cif_path": str(tmp_path / "missing.cif")})
    assert r.status_code == 400
    assert 0 in R._sim_cache                       # nothing changed, so nothing is stale


def test_a_detector_apply_drops_the_simulated_results(env):
    c, R, ctrl, tmp_path = env
    body = {"shape": list(SHAPE), "pc": [0.52, 0.41, 0.61], "sample_tilt": 68.0}
    r = c.post("/api/pc/detector/set", json=body)
    assert r.status_code == 200, r.text
    assert 0 not in R._sim_cache and len(R._sim_cache) == 0
    assert ctrl.cache == {}


def _run_grid(R):
    bt = BackgroundTasks()
    out = asyncio.run(R.calibrate_grid(R.GridCalibrationRequest(grid_step=3, mode="plane"), bt))
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    return out


def _fake_refine(signal, points, method="PSO", search_limit=0.05, progress_callback=None):
    return np.array([[0.5 + 1e-3 * c, 0.4 + 1e-3 * r, 0.6] for r, c in points])


def test_a_grid_calibration_drops_the_simulated_results(env, monkeypatch):
    c, R, ctrl, tmp_path = env
    monkeypatch.setattr(ctrl, "refine_pc_at_points", _fake_refine)
    out = _run_grid(R)
    assert R._optimization_tasks[out["task_id"]]["status"] == "completed"
    assert 0 not in R._sim_cache and len(R._sim_cache) == 0


def test_a_grid_calibration_clears_the_session_it_ran_on_not_the_active_one(env, monkeypatch):
    """The run takes minutes; the viewer may show another dataset when it ends."""
    from backend.api.routes import ebsd_viewer
    c, R, ctrl, tmp_path = env
    started_on = R._sessions["ds"]
    other = R._PCSession(ctrl=type(ctrl)())
    R._sessions["other"] = other
    other.sim_cache[0] = dict(STALE)

    def refine_then_switch(signal, points, method="PSO", search_limit=0.05, progress_callback=None):
        monkeypatch.setattr(ebsd_viewer, "_active_dataset", "other")   # the user switches
        return _fake_refine(signal, points)

    monkeypatch.setattr(ctrl, "refine_pc_at_points", refine_then_switch)
    out = _run_grid(R)
    assert R._optimization_tasks[out["task_id"]]["status"] == "completed"
    assert started_on.sim_cache == {}              # the dataset it was run for is cleared
    assert 0 in other.sim_cache                    # the one that became active is not touched
