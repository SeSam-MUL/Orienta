"""A PC job writes its result to the dataset it was started on.

A Global PC Refine or a grid calibration takes minutes. Both read ``_active_dataset``
when they WROTE their result, so a user who switched to another file meanwhile had
the refined PC (or the whole PC map) land on that other file. The dataset is taken
when the job starts.
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import BackgroundTasks
from kikuchipy.detectors import EBSDDetector

from pc_controller import PCController

SHAPE = (60, 60)
START_PC = (0.5, 0.4, 0.6)
REFINED = (0.4301, 0.2141, 0.5009)


class _Signal:
    def __init__(self):
        self.data = np.zeros((4, 4) + SHAPE, dtype=np.uint8)
        self.axes_manager = SimpleNamespace(navigation_shape=(4, 4))


def _cif(tmp_path):
    cif = tmp_path / "Ni.cif"
    cif.write_text(
        "data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n", encoding="utf-8")
    return str(cif)


@pytest.fixture
def route(monkeypatch, tmp_path):
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from backend.api.services.calibration_store import CalibrationStore
    store = CalibrationStore()
    for name in ("ds", "other"):
        store.register(name, SimpleNamespace(
            detector=EBSDDetector(shape=SHAPE, pc=START_PC, sample_tilt=70.0)))
    ctrl = PCController()
    ctrl.load_phase(_cif(tmp_path))
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=START_PC, sample_tilt=70.0))
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: _Signal())
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    monkeypatch.setattr(R, "_optimization_active", False)
    monkeypatch.setattr(R, "_sessions", {"ds": SimpleNamespace(ctrl=ctrl, sim_cache={})})
    n_tasks = len(R._optimization_tasks)
    yield R, ctrl, store, ebsd_viewer
    R._optimization_active = False
    while len(R._optimization_tasks) > n_tasks:
        R._optimization_tasks.popitem()


def _run_tasks(bt):
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)


def test_a_global_refine_lands_on_the_dataset_it_was_started_on(route, monkeypatch):
    R, ctrl, store, ebsd_viewer = route
    from tests.pc_engine_stub import stub_search

    def switch_file(call):
        ebsd_viewer._active_dataset = "other"           # the user switches file mid-run

    stub_search(monkeypatch, [REFINED], on_call=switch_file)
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: {
        "ci": 0.5, "phase_name": "Ni", "segments": []})
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    R._sessions["ds"].sim_cache = {9: "line segments at the old PC"}
    R._sessions["other"] = SimpleNamespace(ctrl=PCController(), sim_cache={7: "the other file's"})
    bt = BackgroundTasks()
    out = asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[5], method="PSO", search_limit=0.05), bt))
    _run_tasks(bt)
    state = R._optimization_tasks[out["task_id"]]
    assert state["status"] == "completed", state.get("error")
    assert R._sessions["ds"].sim_cache == {}                                  # stale lines dropped here
    assert R._sessions["other"].sim_cache == {7: "the other file's"}          # not there
    np.testing.assert_allclose(store.get_pc("ds"), REFINED, atol=1e-9)        # started here: refined
    np.testing.assert_allclose(store.get_pc("other"), START_PC, atol=1e-9)    # not touched
    assert store.get_entry("ds").pc_source == "refined"
    assert store.get_entry("other").pc_source != "refined"
    # and the result says so: it is for "ds", and the active dataset is another by now
    assert state["result"]["dataset"] == "ds" and state["result"]["dataset_changed"] is True


def test_the_refine_route_hands_the_started_dataset_to_the_job(route):
    """The call site: what the route gives the background job."""
    R, ctrl, store, ebsd_viewer = route
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    bt = BackgroundTasks()
    asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[5], method="PSO", search_limit=0.05), bt))
    task = bt.tasks[0]
    assert task.func is R._run_optimization
    assert "ds" in task.args and ctrl in task.args


def test_a_grid_calibration_lands_on_the_dataset_it_was_started_on(route, monkeypatch):
    R, ctrl, store, ebsd_viewer = route

    def fake_refine(signal, points, method="PSO", search_limit=0.05, progress_callback=None):
        ebsd_viewer._active_dataset = "other"           # the user switches file mid-run
        return np.array([[0.5 + 1e-3 * c, 0.4 + 1e-3 * r, 0.6] for r, c in points])

    monkeypatch.setattr(ctrl, "refine_pc_at_points", fake_refine)
    bt = BackgroundTasks()
    out = asyncio.run(R.calibrate_grid(R.GridCalibrationRequest(grid_step=3, mode="plane"), bt))
    _run_tasks(bt)
    state = R._optimization_tasks[out["task_id"]]
    assert state["status"] == "completed", state.get("error")
    assert store.has_pc_map("ds") and not store.has_pc_map("other")
    assert state["result"]["dataset"] == "ds" and state["result"]["dataset_changed"] is True


def test_a_job_on_the_file_that_stays_active_is_not_marked_as_changed(route, monkeypatch):
    R, ctrl, store, ebsd_viewer = route

    def fake_refine(signal, points, method="PSO", search_limit=0.05, progress_callback=None):
        return np.array([[0.5 + 1e-3 * c, 0.4 + 1e-3 * r, 0.6] for r, c in points])

    monkeypatch.setattr(ctrl, "refine_pc_at_points", fake_refine)
    bt = BackgroundTasks()
    out = asyncio.run(R.calibrate_grid(R.GridCalibrationRequest(grid_step=3, mode="plane"), bt))
    _run_tasks(bt)
    result = R._optimization_tasks[out["task_id"]]["result"]
    assert result["dataset"] == "ds" and result["dataset_changed"] is False
