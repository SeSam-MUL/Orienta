"""A pixel-wise PC grid calibration does not run beside the other PC jobs.

``/calibrate/grid`` ran without the ``_optimization_active`` guard that
``/optimize`` has, so a Global PC Refine or a second grid run could start beside
it on the same controller, and its optimiser was handed ``self.indexer`` afresh
for every point: a request that clears the controller's cache meanwhile
(``/index-pattern``, a PC edit) sets it to None, and the next point's optimiser got
None. The refine, too, sat on its own indexer only as long as nothing cleared it.
"""
import asyncio

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException
from kikuchipy.detectors import EBSDDetector

from pc_controller import PCController

SHAPE = (60, 60)


class _Signal:
    """The active signal as the grid route reads it: a 4 x 4 scan of 60 x 60 patterns."""
    def __init__(self):
        self.data = np.zeros((4, 4) + SHAPE, dtype=np.uint8)


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
    ctrl = PCController()
    ctrl.load_phase(_cif(tmp_path))
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0))
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "calibration_store", CalibrationStore())
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: _Signal())
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    monkeypatch.setattr(R, "_optimization_active", False)
    n_tasks = len(R._optimization_tasks)
    yield R, ctrl
    R._optimization_active = False
    while len(R._optimization_tasks) > n_tasks:
        R._optimization_tasks.popitem()


def _start_grid(R, mode="plane"):
    bt = BackgroundTasks()
    out = asyncio.run(R.calibrate_grid(R.GridCalibrationRequest(grid_step=3, mode=mode), bt))
    return out, bt


def test_a_grid_run_is_refused_while_another_optimisation_runs(route):
    R, ctrl = route
    R._optimization_active = True                     # a Global PC Refine is on
    before = len(R._optimization_tasks)
    with pytest.raises(HTTPException) as exc:
        _start_grid(R)
    assert exc.value.status_code == 409
    assert len(R._optimization_tasks) == before       # and no ghost task was created
    assert R._optimization_active is True             # the running job's flag is not touched


def test_a_grid_run_holds_the_flag_while_it_runs_and_clears_it_after(route, monkeypatch):
    R, ctrl = route
    seen = []

    def fake_refine(signal, points, method="PSO", search_limit=0.05, progress_callback=None):
        seen.append(R._optimization_active)           # what a second request would see
        return np.array([[0.5 + 1e-3 * c, 0.4 + 1e-3 * r, 0.6] for r, c in points])

    monkeypatch.setattr(ctrl, "refine_pc_at_points", fake_refine)
    out, bt = _start_grid(R)
    assert R._optimization_active is True             # from the moment it is accepted
    with pytest.raises(HTTPException):                # so a second one is refused, not interleaved
        _start_grid(R)
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    state = R._optimization_tasks[out["task_id"]]
    assert state["status"] == "completed", state.get("error")
    assert seen == [True]
    assert R._optimization_active is False
    # the fit's own sample tilt estimate is reported, the page's tilt is what stays
    assert state["result"]["sample_tilt"] == 70.0
    assert isinstance(state["result"]["fitted_sample_tilt"], float)
    assert state["result"]["fitted_sample_tilt"] != 70.0


def test_a_failing_grid_run_clears_the_flag_too(route, monkeypatch):
    R, ctrl = route

    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ctrl, "refine_pc_at_points", broken)
    out, bt = _start_grid(R)
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    assert R._optimization_tasks[out["task_id"]]["status"] == "failed"
    assert R._optimization_active is False


def test_no_signal_leaves_no_ghost_task(route, monkeypatch):
    from backend.api.routes import ebsd_viewer
    R, ctrl = route
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: None)
    before = len(R._optimization_tasks)
    with pytest.raises(HTTPException) as exc:
        _start_grid(R)
    assert exc.value.status_code == 400
    assert len(R._optimization_tasks) == before
    assert R._optimization_active is False


def test_the_optimiser_keeps_its_indexer_when_the_cache_is_cleared_meanwhile(tmp_path, monkeypatch):
    """/index-pattern and a PC edit call clear_cache(), which sets self.indexer to None."""
    import pc_controller

    ctrl = PCController()
    ctrl.load_phase(_cif(tmp_path))
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0))
    handed = []

    def fake_optimize(detector, indexer, pattern, method="PSO", search_limit=0.05, batch=True):
        handed.append(indexer)
        if len(handed) == 1:
            ctrl.clear_cache()                        # another request, mid-run
        return (0.5, 0.4, 0.6), None

    monkeypatch.setattr(pc_controller, "optimize_pc", fake_optimize)
    ctrl.refine_pc_at_points(_Signal(), [(0, 0), (1, 1), (2, 2)])
    assert len(handed) == 3 and all(i is not None for i in handed)
    assert handed[0] is handed[1] is handed[2]        # the same indexer, built once

    handed.clear()
    ctrl.indexer = None
    for i in range(2):
        ctrl.add_pattern((i, i), np.zeros(SHAPE, np.uint8))
    ctrl.refine_global_pc()
    assert len(handed) == 2 and all(i is not None for i in handed)


def test_the_grid_replaces_the_controllers_detector_under_the_state_lock(route, monkeypatch):
    """A request that holds the state lock (/index-pattern indexes under it) must
    not see the detector swapped under it."""
    import threading
    import time
    R, ctrl = route

    def fake_refine(signal, points, method="PSO", search_limit=0.05, progress_callback=None):
        return np.array([[0.5 + 1e-3 * c, 0.4 + 1e-3 * r, 0.6] for r, c in points])

    monkeypatch.setattr(ctrl, "refine_pc_at_points", fake_refine)
    out, bt = _start_grid(R)
    worker = threading.Thread(target=lambda: [t.func(*t.args, **t.kwargs) for t in bt.tasks])
    R._state_lock.acquire()
    try:
        worker.start()
        time.sleep(1.5)                       # the fit is done long before; it waits for the lock
        from pc_controller import pc_is_per_pixel
        assert not pc_is_per_pixel(ctrl.detector), "the detector was replaced while the lock was held"
    finally:
        R._state_lock.release()
    worker.join(timeout=20)
    assert not worker.is_alive()
    from pc_controller import pc_is_per_pixel
    assert pc_is_per_pixel(ctrl.detector)
