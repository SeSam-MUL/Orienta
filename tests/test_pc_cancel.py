"""Cancel stops a Global PC Refine or a grid calibration, and nothing is written.

The Cancel button of the PC page used to stop only the page's polling: the run
went on in the background, later wrote its PC to the controller and the store
unseen, and until it ended every new run or phase change was refused with 409.
Now the route asks the run to stop. A refine is one search over all patterns: it
stops at the search's next progress call, or at the check before it writes; a grid
calibration stops between two grid points. Either way the status becomes
"cancelled" and the PC, the store and the caches stay as they were.
"""
import asyncio

import numpy as np
import pytest
from fastapi import BackgroundTasks

from tests.test_pc_job_dataset_capture import REFINED, SHAPE, START_PC, _run_tasks, route  # noqa: F401


def _start_refine(R, n_patterns=3):
    bt = BackgroundTasks()
    out = asyncio.run(R.optimize_pc(
        R.OptimizePCRequest(patterns=list(range(n_patterns)), method="PSO", search_limit=0.05), bt))
    return out["task_id"], bt


def test_cancel_mid_run_writes_nothing(route, monkeypatch):
    R, ctrl, store, ebsd_viewer = route
    from tests.pc_engine_stub import stub_search
    task = {}
    after_progress = []

    def press_cancel(call):                          # the user presses Cancel during the search
        assert asyncio.run(R.cancel_optimization(task["id"]))["status"] == "cancelling"
        call["progress"](0.5)                        # the search's next progress call ends the run
        after_progress.append(1)

    calls = stub_search(monkeypatch, [REFINED] * 3, on_call=press_cancel)
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: pytest.fail("no re-index after Cancel"))
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    R._sessions["ds"].sim_cache = {0: "lines at the PC before the run"}
    task["id"], bt = _start_refine(R)
    _run_tasks(bt)

    state = R._optimization_tasks[task["id"]]
    assert state["status"] == "cancelled"
    assert len(calls) == 1                                           # one search, and it was left
    assert after_progress == []                                      # at its progress call
    np.testing.assert_allclose(np.asarray(ctrl.detector.pc).reshape(-1, 3)[0], START_PC)
    np.testing.assert_allclose(store.get_pc("ds"), START_PC, atol=1e-9)
    assert store.get_entry("ds").pc_source != "refined"
    assert R._sessions["ds"].sim_cache == {0: "lines at the PC before the run"}
    assert R._optimization_active is False                           # a new run may start
    assert task["id"] not in R._cancel_requested


def test_cancel_after_the_last_progress_call_still_writes_nothing(route, monkeypatch):
    """Pressed with no progress call after it: the check before the write-back catches it."""
    R, ctrl, store, ebsd_viewer = route
    from tests.pc_engine_stub import stub_search
    task = {}

    def press_cancel(call):
        asyncio.run(R.cancel_optimization(task["id"]))      # and the search ends without another progress call

    stub_search(monkeypatch, [REFINED] * 2, on_call=press_cancel)
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: pytest.fail("no re-index after Cancel"))
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    task["id"], bt = _start_refine(R, n_patterns=2)
    _run_tasks(bt)
    assert R._optimization_tasks[task["id"]]["status"] == "cancelled"
    np.testing.assert_allclose(store.get_pc("ds"), START_PC, atol=1e-9)


def test_cancel_of_a_finished_run_keeps_its_result(route, monkeypatch):
    R, ctrl, store, ebsd_viewer = route
    from tests.pc_engine_stub import stub_search
    stub_search(monkeypatch, [REFINED])
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: {"ci": 0.5, "phase_name": "Ni", "segments": []})
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    tid, bt = _start_refine(R, n_patterns=1)
    _run_tasks(bt)
    assert asyncio.run(R.cancel_optimization(tid))["status"] == "completed"
    assert R._optimization_tasks[tid]["status"] == "completed"
    assert tid not in R._cancel_requested
    np.testing.assert_allclose(store.get_pc("ds"), REFINED, atol=1e-9)


def test_cancel_of_an_unknown_task_is_a_404(route):
    from fastapi import HTTPException
    R, *_ = route
    with pytest.raises(HTTPException) as e:
        asyncio.run(R.cancel_optimization("no-such-task"))
    assert e.value.status_code == 404


def test_the_grid_calibration_stops_between_grid_points(route, monkeypatch):
    R, ctrl, store, ebsd_viewer = route
    applied = []
    monkeypatch.setattr(ctrl, "generate_calibration_grid", lambda nr, nc, step: [(0, 0), (1, 1), (2, 2)])
    task = {}

    def fake_refine(signal, grid, method, search_limit, progress_callback):
        for i in range(len(grid)):
            if i == 1:
                asyncio.run(R.cancel_optimization(task["id"]))
            progress_callback(i + 1, len(grid))
        return [[0.5, 0.4, 0.6]] * len(grid)

    monkeypatch.setattr(ctrl, "refine_pc_at_points", fake_refine)
    monkeypatch.setattr(ctrl, "apply_pixelwise_pc", lambda det: applied.append(det))
    bt = BackgroundTasks()
    out = asyncio.run(R.calibrate_grid(R.GridCalibrationRequest(grid_step=2), bt))
    task["id"] = out["task_id"]
    _run_tasks(bt)
    assert R._optimization_tasks[task["id"]]["status"] == "cancelled"
    assert applied == []
    assert R._optimization_active is False
