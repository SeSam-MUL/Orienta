"""A PC search or grid calibration does not survive a load into another file.

Two files of the same name in two folders share one dataset name, so the name alone cannot say whose
entry a finished search is about. The job takes the ``file_key`` of the dataset's store entry when it
starts and compares it with the entry under the lock before it writes: another file (or no entry) means
nothing is written and the job ends ``failed`` with the code ``fileChanged``. The search itself is
stubbed (tests/pc_engine_stub.py); the store is the real one.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import BackgroundTasks
from kikuchipy.detectors import EBSDDetector

from pc_controller import PCController
from tests.pc_engine_stub import stub_search

SHAPE = (60, 60)
PC_A = (0.5, 0.4, 0.6)
PC_B = (0.45, 0.35, 0.55)
FOUND = (0.4301, 0.2141, 0.5009)
KEY_A = "/folder_a/scan.h5"
KEY_B = "/folder_b/scan.h5"


class _Signal:
    def __init__(self):
        self.data = np.zeros((6, 6) + SHAPE, dtype=np.uint8)
        self.axes_manager = SimpleNamespace(navigation_shape=(6, 6))


def _file(pc):
    return SimpleNamespace(detector=EBSDDetector(shape=SHAPE, pc=pc, sample_tilt=70.0))


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
    store.register("scan", _file(PC_A), file_key=KEY_A)
    ctrl = PCController()
    ctrl.load_phase(_cif(tmp_path))
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=PC_A, sample_tilt=70.0))
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: _Signal())
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "scan")
    monkeypatch.setattr(R, "_optimization_active", False)
    monkeypatch.setattr(R, "_sessions", {"scan": SimpleNamespace(ctrl=ctrl, sim_cache={})})
    monkeypatch.setattr(R, "_pc_before", {})
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: {
        "ci": 0.5, "phase_name": "Ni", "segments": []})
    n_tasks = len(R._optimization_tasks)
    yield R, ctrl, store
    R._optimization_active = False
    while len(R._optimization_tasks) > n_tasks:
        R._optimization_tasks.popitem()


def _optimize(R, ctrl):
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    bt = BackgroundTasks()
    out = asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[7], method="PSO", search_limit=0.05), bt))
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    return R._optimization_tasks[out["task_id"]]


def test_a_search_whose_file_was_replaced_by_a_file_of_the_same_name_writes_nothing(route, monkeypatch):
    R, ctrl, store = route
    stub_search(monkeypatch, [FOUND],
                on_call=lambda call: store.register("scan", _file(PC_B), file_key=KEY_B))   # file B loaded
    task = _optimize(R, ctrl)
    assert task["status"] == "failed"
    assert task["code"] == "fileChanged"
    assert task["error"] == "Another file was opened during the search. Nothing was written."
    assert task["result"] is None
    np.testing.assert_allclose(store.get_pc("scan"), PC_B, atol=1e-9)        # B keeps its own PC
    assert store.get_entry("scan").pc_source != "refined"
    assert R._pc_before == {}                                                 # no step to put back either
    np.testing.assert_allclose(np.asarray(ctrl.pc_at(None)).reshape(-1)[:3], PC_A, atol=1e-9)   # the page's detector too


def test_a_search_whose_entry_is_gone_writes_nothing(route, monkeypatch):
    R, ctrl, store = route
    stub_search(monkeypatch, [FOUND], on_call=lambda call: store.remove("scan"))
    task = _optimize(R, ctrl)
    assert task["status"] == "failed" and task["code"] == "fileChanged"
    assert store.get_entry("scan") is None


def test_a_search_on_the_file_that_stays_is_written(route, monkeypatch):
    R, ctrl, store = route
    stub_search(monkeypatch, [FOUND])
    task = _optimize(R, ctrl)
    assert task["status"] == "completed", task.get("error")
    assert "code" not in task
    np.testing.assert_allclose(store.get_pc("scan"), FOUND, atol=1e-9)


def test_the_route_hands_the_file_it_started_on_to_the_job(route):
    """The call site: the key is taken when the route starts the job (not when the job gets its turn)."""
    R, ctrl, store = route
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    bt = BackgroundTasks()
    asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[7], method="PSO", search_limit=0.05), bt))
    store.register("scan", _file(PC_B), file_key=KEY_B)           # B is loaded before the job runs
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    state = list(R._optimization_tasks.values())[-1]
    assert state["status"] == "failed" and state["code"] == "fileChanged"


def test_a_grid_calibration_whose_file_was_replaced_writes_nothing(route, monkeypatch):
    R, ctrl, store = route

    def fake_refine(signal, points, method="PSO", search_limit=0.05, progress_callback=None):
        store.register("scan", _file(PC_B), file_key=KEY_B)       # file B loaded mid-run
        return np.array([[0.5 + 1e-3 * c, 0.4 + 1e-3 * r, 0.6] for r, c in points])

    monkeypatch.setattr(ctrl, "refine_pc_at_points", fake_refine)
    bt = BackgroundTasks()
    out = asyncio.run(R.calibrate_grid(R.GridCalibrationRequest(grid_step=3, mode="plane"), bt))
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    state = R._optimization_tasks[out["task_id"]]
    assert state["status"] == "failed" and state["code"] == "fileChanged"
    assert not store.has_pc_map("scan")


def _start_with_a_load_during_the_read(R, ctrl, store, monkeypatch, during):
    """POST /optimize for a position the page does not hold, so the route reads it from the signal off the event
    loop (an await); ``during`` runs in the middle of that read."""
    from backend.api.services import load_choices

    def slow_read(signal, row, col):
        during()
        return np.zeros(SHAPE, np.uint8)

    monkeypatch.setattr(load_choices, "signal_pattern", slow_read)
    bt = BackgroundTasks()
    out = asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[7], method="PSO", search_limit=0.05), bt))
    return out, bt


def test_a_switch_of_the_active_file_while_the_patterns_are_read_does_not_change_the_file_of_the_job(route, monkeypatch):
    from backend.api.routes import ebsd_viewer
    R, ctrl, store = route
    store.register("other", _file(PC_B), file_key="/folder_c/other.h5")

    def switch():
        ebsd_viewer._active_dataset = "other"

    out, bt = _start_with_a_load_during_the_read(R, ctrl, store, monkeypatch, switch)
    task = bt.tasks[0]
    assert task.args[5] == "scan"                                  # the dataset the request was made for


def test_a_file_loaded_under_the_same_name_while_the_patterns_are_read_fails_the_job(route, monkeypatch):
    R, ctrl, store = route
    stub_search(monkeypatch, [FOUND])
    out, bt = _start_with_a_load_during_the_read(
        R, ctrl, store, monkeypatch, lambda: store.register("scan", _file(PC_B), file_key=KEY_B))
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    state = R._optimization_tasks[out["task_id"]]
    assert state["status"] == "failed" and state["code"] == "fileChanged"
    np.testing.assert_allclose(store.get_pc("scan"), PC_B, atol=1e-9)
