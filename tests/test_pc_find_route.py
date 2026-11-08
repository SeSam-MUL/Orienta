"""POST /api/pc/optimize with ``mode``: "Find PC" and the joint "Global PC Refine", through the
route and the background job, with the search itself stubbed (tests/pc_engine_stub.py)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException
from kikuchipy.detectors import EBSDDetector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pc_controller import PCController  # noqa: E402
from tests.pc_engine_stub import (  # noqa: E402
    clear_confidence, not_judged_confidence, stub_search, unclear_confidence)

SHAPE = (60, 60)
START_PC = (0.5, 0.4, 0.6)
FOUND = (0.4301, 0.2141, 0.5009)


class _Signal:
    def __init__(self):
        rng = np.random.default_rng(0)
        self.data = rng.integers(0, 255, (6, 6) + SHAPE).astype(np.uint8)
        self.axes_manager = SimpleNamespace(navigation_shape=(6, 6))


def _cif(tmp_path):
    cif = tmp_path / "Ni.cif"
    cif.write_text(
        "data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n", encoding="utf-8")
    return str(cif)


def _register(store, name, source="header", pc=START_PC):
    store.register(name, SimpleNamespace(
        detector=EBSDDetector(shape=SHAPE, pc=pc, sample_tilt=70.0),
        metadata=SimpleNamespace(get_item=lambda key, default=None: ("default" if source == "missing"
                                                                       and key == "Signal.pc_source" else default))))


@pytest.fixture
def route(monkeypatch, tmp_path):
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from backend.api.services.calibration_store import CalibrationStore
    store = CalibrationStore()
    _register(store, "ds")
    ctrl = PCController()
    ctrl.load_phase(_cif(tmp_path))
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=START_PC, sample_tilt=70.0))
    signal = _Signal()
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: signal)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    monkeypatch.setattr(R, "_optimization_active", False)
    monkeypatch.setattr(R, "_sessions", {"ds": SimpleNamespace(ctrl=ctrl, sim_cache={})})
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: {"ci": 0.5, "phase_name": "Ni", "segments": []})
    n_tasks = len(R._optimization_tasks)
    yield R, ctrl, store, signal
    R._optimization_active = False
    while len(R._optimization_tasks) > n_tasks:
        R._optimization_tasks.popitem()


def _start(R, ctrl, positions, **req):
    for (r, c) in positions:
        ctrl.add_pattern((r, c), np.full(SHAPE, r * 10 + c, np.uint8))
    bt = BackgroundTasks()
    flat = [r * 6 + c for r, c in positions]
    out = asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=flat, **req), bt))
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    return R._optimization_tasks[out["task_id"]], bt


FIVE = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


# ------------------------------------------------------------------ the mode reaches the search

def test_find_runs_the_search_in_find_mode_without_a_start(route, monkeypatch):
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM")
    task, bt = _start(R, ctrl, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    assert (calls[0]["mode"], calls[0]["start"]) == ("find", None)
    assert task["result"]["mode"] == "find" and task["result"]["search"]["mode"] == "find"
    np.testing.assert_allclose(store.get_pc("ds"), FOUND, atol=1e-9)
    assert store.get_entry("ds").pc_refinement == {"n_patterns": 5, "method": "PSO", "mode": "find"}   # swarm credited
    assert bt.tasks[0].kwargs["mode"] == "find"


def test_refine_is_the_default_and_starts_at_the_pc_the_page_holds(route, monkeypatch):
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND] * 2)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["status"] == "completed", task.get("error")
    assert calls[0]["mode"] == "refine"
    np.testing.assert_allclose(calls[0]["start"], START_PC, atol=1e-9)
    assert store.get_entry("ds").pc_refinement == {"n_patterns": 2, "method": "Nelder-Mead", "mode": "refine"}


def test_find_with_fewer_patterns_than_it_needs_is_refused_before_anything_starts(route):
    R, ctrl, store, _ = route
    for r, c in FIVE[:4]:
        ctrl.add_pattern((r, c), np.zeros(SHAPE, np.uint8))
    n_tasks = len(R._optimization_tasks)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[0, 7, 14, 21], mode="find"), BackgroundTasks()))
    assert exc.value.status_code == 400 and exc.value.headers["X-Orienta-Code"] == "tooFewPatterns"
    assert R._optimization_active is False                  # nothing was started
    assert len(R._optimization_tasks) == n_tasks


def test_an_unknown_mode_is_not_accepted():
    from backend.api.routes import pcrefinement as R
    with pytest.raises(Exception):
        R.OptimizePCRequest(patterns=[1], mode="guess")


def test_the_search_sees_the_calibration_pattern_the_page_shows_not_the_live_signal(route, monkeypatch):
    R, ctrl, store, signal = route
    calls = stub_search(monkeypatch, [FOUND] * 5)
    _start(R, ctrl, FIVE, mode="find")
    for k, (r, c) in enumerate(FIVE):
        np.testing.assert_array_equal(calls[0]["patterns"][k], np.full(SHAPE, r * 10 + c))
        assert not np.array_equal(calls[0]["patterns"][k], signal.data[r, c])


def test_a_requested_position_that_is_not_stored_is_read_from_the_signal(route, monkeypatch):
    R, ctrl, store, signal = route
    calls = stub_search(monkeypatch, [FOUND] * 2)
    ctrl.add_pattern((0, 0), np.full(SHAPE, 7, np.uint8))
    bt = BackgroundTasks()
    asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[0, 6 * 2 + 3]), bt))      # (0, 0) and (2, 3)
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    np.testing.assert_array_equal(calls[0]["patterns"][0], np.full(SHAPE, 7))
    np.testing.assert_array_equal(calls[0]["patterns"][1], signal.data[2, 3])


# ------------------------------------------------------------------ what the route refuses, before a job exists

def _refused(R, **req):
    n_tasks = len(R._optimization_tasks)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[0, 7], **req), BackgroundTasks()))
    assert exc.value.status_code == 400
    assert len(R._optimization_tasks) == n_tasks             # no task was made
    assert R._optimization_active is False                   # nothing is running
    return exc.value.headers["X-Orienta-Code"]


def test_a_refine_with_an_unknown_method_is_refused_with_its_code(route, monkeypatch):
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND] * 2)
    assert _refused(R, method="Simplex") == "badMethod"
    assert calls == []


def test_a_refine_with_a_search_limit_outside_what_a_box_can_be_is_refused_with_its_code(route, monkeypatch):
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND] * 2)
    assert _refused(R, search_limit=2.0) == "badSearchLimit"
    assert calls == []


def test_a_refine_whose_start_is_outside_the_physical_box_is_refused_with_its_code(route, monkeypatch):
    """The start is the PC the page holds: the detector's."""
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND] * 2)
    ctrl.detector.pc = (0.5, 0.4, 3.0)                       # z far beyond a detector (0.2 - 1.6)
    assert _refused(R, method="Nelder-Mead") == "badStart"
    assert calls == []


def test_a_refine_whose_start_comes_from_the_store_outside_the_box_is_refused_with_its_code(route, monkeypatch):
    """No detector on the page yet: the route attaches the store's, and that PC is outside the box."""
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND] * 2)
    _register(store, "ds", pc=(0.5, 0.4, 3.0))
    ctrl.detector = None
    assert _refused(R, method="Nelder-Mead") == "badStart"
    assert calls == []


def test_a_refine_without_any_pattern_is_refused_with_its_code(route, monkeypatch):
    """An empty list used to run a job that averaged nothing; now it never starts."""
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND])
    n_tasks = len(R._optimization_tasks)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[]), BackgroundTasks()))
    assert exc.value.status_code == 400 and exc.value.headers["X-Orienta-Code"] == "noPatterns"
    assert len(R._optimization_tasks) == n_tasks and R._optimization_active is False
    assert calls == []


def test_a_refine_that_is_not_refused_still_runs(route, monkeypatch):
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND] * 2)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)], method="Nelder-Mead", search_limit=0.05)
    assert task["status"] == "completed" and len(calls) == 1


def test_a_find_is_not_refused_for_a_method_or_a_limit_it_does_not_use(route, monkeypatch):
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find", method="Simplex", search_limit=2.0)
    assert task["status"] == "completed", task.get("error")
    assert len(calls) == 1 and calls[0]["mode"] == "find"


def test_a_find_is_not_refused_for_a_start_outside_the_box(route, monkeypatch):
    """A find needs no start: the page's PC is irrelevant, even when it is outside the box."""
    R, ctrl, store, _ = route
    ctrl.detector.pc = (0.5, 0.4, 3.0)
    calls = stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    assert calls[0]["start"] is None


# ------------------------------------------------------------------ Cancel reaches the search

def test_cancel_during_a_find_ends_the_job_through_the_searchs_progress_call(route, monkeypatch):
    R, ctrl, store, _ = route
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: pytest.fail("no re-index after Cancel"))
    reached = []

    def press_cancel(call):
        task_id = next(reversed(R._optimization_tasks))
        assert asyncio.run(R.cancel_optimization(task_id))["status"] == "cancelling"
        call["progress"](0.5)                               # the search's next progress call ends the run
        reached.append("after the progress call")

    stub_search(monkeypatch, [FOUND] * 5, on_call=press_cancel, method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert task["status"] == "cancelled" and task["result"] is None
    assert reached == []
    np.testing.assert_allclose(store.get_pc("ds"), START_PC, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ctrl.detector.pc).reshape(-1, 3)[0], START_PC)
    assert store.get_entry("ds").pc_source != "refined"
    assert R._optimization_active is False and not R._cancel_requested


def test_cancel_pressed_before_the_job_gets_to_start_the_search_never_runs_it(route, monkeypatch):
    R, ctrl, store, _ = route
    calls = stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM")
    for (r, c) in FIVE:
        ctrl.add_pattern((r, c), np.full(SHAPE, r * 10 + c, np.uint8))
    bt = BackgroundTasks()
    out = asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[r * 6 + c for r, c in FIVE], mode="find"), bt))
    assert asyncio.run(R.cancel_optimization(out["task_id"]))["status"] == "cancelling"
    for task in bt.tasks:                                    # the job starts only now
        task.func(*task.args, **task.kwargs)
    assert R._optimization_tasks[out["task_id"]]["status"] == "cancelled"
    assert calls == []
    np.testing.assert_allclose(store.get_pc("ds"), START_PC, atol=1e-9)


def test_progress_of_a_job_that_was_not_cancelled_does_not_raise(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, on_call=lambda call: call["progress"](0.5), method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")


# ------------------------------------------------------------------ the warnings

def _codes(task):
    return [c["code"] for c in task["result"]["pc_warning_codes"]]


def test_a_file_without_a_pc_has_no_drift_to_report_against_the_placeholder(route, monkeypatch):
    """A4: the store holds the placeholder (0.5, 0.4, 0.6) as pc_source 'missing'. A refine from it that
    ends far away is not 'drift from the starting PC'; it is said to have started at the placeholder."""
    R, ctrl, store, _ = route
    _register(store, "ds", source="missing")
    assert store.get_entry("ds").pc_source == "missing"
    stub_search(monkeypatch, [(0.7, 0.2, 0.9)] * 2)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["status"] == "completed", task.get("error")
    assert task["result"]["pc_deviation"] is None and task["result"]["file_pc"] is None
    assert "pcDrift" not in _codes(task) and "refineFromPlaceholder" in _codes(task)


def test_a_file_with_a_pc_still_reports_drift_of_a_refine(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [(0.7, 0.2, 0.9)] * 2)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert "pcDrift" in _codes(task) and "refineFromPlaceholder" not in _codes(task)
    assert task["result"]["pc_deviation"] == pytest.approx(0.3)
    assert task["result"]["file_pc"] == pytest.approx(START_PC)
    assert task["result"]["file_pc_deviation"] == pytest.approx(0.3)


def test_a_find_is_not_compared_with_a_start_but_reports_its_distance_from_the_file_pc(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [(0.52, 0.41, 0.62)] * 5, method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert "pcDrift" not in _codes(task) and task["result"]["pc_deviation"] is None
    assert task["result"]["file_pc_deviation"] == pytest.approx(0.02)


def test_an_unclear_result_carries_the_warning_with_its_number(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, confidence=unclear_confidence())
    task, _ = _start(R, ctrl, FIVE, mode="find")
    unclear = [c for c in task["result"]["pc_warning_codes"] if c["code"] == "pcUnclear"]
    assert unclear and unclear[0]["params"] == {"contrast": 0.1}
    assert "10%" in task["result"]["pc_warning"]
    assert task["result"]["search"]["confidence"]["state"] == "unclear"


def test_an_unclear_result_is_written_with_its_note(route, monkeypatch):
    """Decision: an unclear result is written, with the note (not a proposal the user must accept)."""
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, confidence=unclear_confidence(), method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert task["status"] == "completed"
    np.testing.assert_allclose(store.get_pc("ds"), FOUND, atol=1e-9)
    assert store.get_entry("ds").pc_source == "refined"


def test_a_clear_result_carries_no_such_warning(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, confidence=clear_confidence())
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert "pcUnclear" not in _codes(task) and "pcOnBoxEdge" not in _codes(task)


def test_a_result_the_search_did_not_judge_carries_neither_note(route, monkeypatch):
    """While the search has no threshold its state is "notJudged": a note on every result would be noise."""
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, confidence=not_judged_confidence(), method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    assert "pcUnclear" not in _codes(task) and "pcNotAMinimum" not in _codes(task)
    assert task["result"]["pc_warning"] is None
    assert task["result"]["search"]["confidence"]["state"] == "notJudged"


def test_a_not_judged_result_with_a_cheaper_neighbour_says_it_is_not_a_minimum(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM",
                confidence=not_judged_confidence(reasons=["notAMinimum"]))
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert "pcNotAMinimum" in _codes(task) and "pcUnclear" not in _codes(task)
    assert "lower cost" in task["result"]["pc_warning"]


def test_an_unclear_result_that_is_also_not_a_minimum_carries_both(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM",
                confidence=unclear_confidence(reasons=["notAMinimum", "noContrast"]))
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert {"pcUnclear", "pcNotAMinimum"} <= set(_codes(task))


def test_a_result_on_the_edge_of_its_box_says_which_edges(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, on_edge=["y", "z"])
    task, _ = _start(R, ctrl, FIVE, mode="find")
    edge = [c for c in task["result"]["pc_warning_codes"] if c["code"] == "pcOnBoxEdge"]
    assert edge and edge[0]["params"] == {"axes": "y, z"}


OTHER = {"pc": (0.40, 0.30, 1.30), "cost": 5.0}


def test_a_second_position_is_a_warning_with_its_numbers_and_is_in_the_quality(route, monkeypatch):
    """The stand-in search finds FOUND at cost 4.7 with another position at (0.40, 0.30, 1.30), cost 5.0."""
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM", second_position=OTHER, cost=4.7)
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    second = [c for c in task["result"]["pc_warning_codes"] if c["code"] == "pcSecondPosition"]
    assert second == [{"code": "pcSecondPosition",
                       "params": {"pc": "0.400, 0.300, 1.300", "cost": 5.0, "best_cost": 4.7}}]
    assert "Two about equally good positions. These patterns may not fix the PC well. Check the bands." in task["result"]["pc_warning"]
    quality = task["result"]["quality"]
    assert quality["warning"] == "secondPosition" and quality["best_cost"] == 4.7
    assert quality["second_position"] == {"pc": [0.40, 0.30, 1.30], "cost": 5.0}
    assert task["result"]["search"]["second_position"] == {"pc": [0.40, 0.30, 1.30], "cost": 5.0}
    np.testing.assert_allclose(store.get_pc("ds"), FOUND, atol=1e-9)             # the result is what is written


def test_a_result_without_a_second_position_has_no_such_warning(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert "pcSecondPosition" not in _codes(task) and task["result"]["search"]["second_position"] is None
    assert "second_position" not in task["result"]["quality"]


def test_the_old_small_pattern_warning_is_gone(route, monkeypatch):
    """Patterns of 60 x 60 px used to be called unreliable for any PC refine; the measurement found a
    60 px nickel scan within 0.007 of the file's PC."""
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 2)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert "smallPatterns" not in _codes(task)


def test_the_pc_applied_is_the_joint_minimum_not_the_mean_of_what_the_patterns_prefer(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [(0.43, 0.21, 0.50), (0.45, 0.23, 0.52)], joint=FOUND)   # their mean is (0.44, 0.22, 0.51)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["status"] == "completed", task.get("error")
    np.testing.assert_allclose(task["result"]["mean_pc"], FOUND, atol=1e-9)
    np.testing.assert_allclose(store.get_pc("ds"), FOUND, atol=1e-9)
    assert task["result"]["pc_values"] == [[0.43, 0.21, 0.5], [0.45, 0.23, 0.52]]      # kept for the spread only


def test_progress_reaches_the_task_and_the_per_pattern_values_are_kept(route, monkeypatch):
    R, ctrl, store, _ = route
    seen = []

    def watch(call):
        call["progress"](0.5)
        seen.append(R._optimization_tasks[next(reversed(R._optimization_tasks))]["progress"])

    stub_search(monkeypatch, [(0.43, 0.21, 0.5), (0.44, 0.22, 0.51)], on_call=watch)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert seen == [0.5]
    assert task["result"]["pc_values"] == [[0.43, 0.21, 0.5], [0.44, 0.22, 0.51]]
    np.testing.assert_allclose(task["result"]["mean_pc"], [0.435, 0.215, 0.505])


# ------------------------------------------------------------------ a refine at a member of a calibration

@pytest.fixture
def calibrations(tmp_path):
    from backend.api.services import pc_calibration_apply as use
    from backend.api.services import pc_calibrations as cal
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    use.forget_all()
    yield cal, use
    use.forget_all()
    cal.set_calibrations_dir_for_test(None)


def _two_members(cal, use, store):
    _register(store, "other")
    made = cal.create(name="Cal", pc=START_PC, features=store.get_entry("ds").features, origin={})
    # the stand-in signals state no settings, so the file counts as unverified: the second naming accepts it
    done = use.apply(made["id"], ["ds", "other"], accept_mismatch=["ds", "other"], store=store)
    assert [d["status"] for d in done] == ["applied", "applied"]
    return made


def test_a_refine_at_a_member_reports_the_other_members_and_passes_nothing_on(route, calibrations, monkeypatch):
    R, ctrl, store, _ = route
    cal, use = calibrations
    _two_members(cal, use, store)
    stub_search(monkeypatch, [FOUND] * 2)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["status"] == "completed", task.get("error")
    got = task["result"]["calibration"]
    assert got["calibration"]["version"] == 1 and got["diverged"] is True
    assert got["followers"] == [{"dataset": "other", "per_pixel": False, "map_kind": None}]
    np.testing.assert_allclose(store.get_pc("other"), START_PC, atol=1e-12)        # nothing was passed on
    np.testing.assert_allclose(store.get_pc("ds"), FOUND, atol=1e-9)
    assert use.search_of("ds")["mode"] == "refine"                                   # and the run is remembered


EMPTY_BLOCK = {"calibration": None, "diverged": False, "followers": []}


@pytest.mark.parametrize("restart", [False, True])
def test_a_search_at_a_file_that_had_already_left_the_calibration_asks_nothing(route, calibrations, monkeypatch, restart):
    """ds and other are on version 1. A PC is typed at ds: it left the calibration then, not by the search that follows.
    The search must not ask about ``other``, and neither after a restart (the memory is empty, the calibration file's
    member list still names both)."""
    R, ctrl, store, _ = route
    cal, use = calibrations
    _two_members(cal, use, store)
    store.update_pc("ds", (0.45, 0.35, 0.55), source="manual")             # typed: ds is off the calibration
    assert use.current_ref("ds", store) is None and use.current_ref("other", store) is not None
    if restart:
        use.forget_all()
    stub_search(monkeypatch, [FOUND] * 2)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["status"] == "completed", task.get("error")
    assert task["result"]["calibration"] == EMPTY_BLOCK
    np.testing.assert_allclose(store.get_pc("ds"), FOUND, atol=1e-9)        # the search itself ran
    np.testing.assert_allclose(store.get_pc("other"), START_PC, atol=1e-12)
    assert use.search_of("ds")["mode"] == "refine"                           # and it is remembered all the same


def test_the_second_search_after_leaving_the_calibration_does_not_ask_either(route, calibrations, monkeypatch):
    """The first search at a member asks (it leaves the calibration). The next search at the same file does not:
    it was off the calibration when it started."""
    R, ctrl, store, _ = route
    cal, use = calibrations
    _two_members(cal, use, store)
    stub_search(monkeypatch, [FOUND] * 2)
    first, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert [f["dataset"] for f in first["result"]["calibration"]["followers"]] == ["other"]
    second, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert second["status"] == "completed", second.get("error")
    assert second["result"]["calibration"] == EMPTY_BLOCK


def test_a_search_at_a_file_that_was_on_a_calibration_asks_about_its_followers_with_their_maps(route, calibrations, monkeypatch):
    R, ctrl, store, _ = route
    cal, use = calibrations
    _two_members(cal, use, store)
    entry = store.get_entry("other")
    entry.pc_map = np.tile(np.asarray(START_PC, dtype=float), (2, 2, 1))        # a map the file brought; its mean is START_PC
    stub_search(monkeypatch, [FOUND] * 2)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["result"]["calibration"]["followers"] == [{"dataset": "other", "per_pixel": True, "map_kind": "header"}]


def test_a_search_report_that_was_not_judged_is_remembered_unchanged(route, calibrations, monkeypatch):
    R, ctrl, store, _ = route
    cal, use = calibrations
    _two_members(cal, use, store)
    stub_search(monkeypatch, [FOUND] * 2, confidence=not_judged_confidence())
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["status"] == "completed", task.get("error")
    assert use.search_of("ds")["confidence"]["state"] == "notJudged"
    assert use.search_of("ds") == task["result"]["search"]


@pytest.mark.parametrize("when", ["mid_search", "after_search"])
def test_a_cancelled_refine_at_a_member_leaves_no_question_and_no_note(route, calibrations, monkeypatch, when):
    R, ctrl, store, _ = route
    cal, use = calibrations
    _two_members(cal, use, store)
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: pytest.fail("no re-index after Cancel"))
    asked = []
    monkeypatch.setattr(use, "followers", lambda *a, **k: asked.append(a) or pytest.fail("nobody is asked"))

    def press_cancel(call):
        task_id = next(reversed(R._optimization_tasks))
        assert asyncio.run(R.cancel_optimization(task_id))["status"] == "cancelling"
        if when == "mid_search":
            call["progress"](0.5)                        # the search ends here
        # "after_search": the search returns, the run notices the Cancel before it writes anything

    stub_search(monkeypatch, [FOUND] * 2, on_call=press_cancel)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["status"] == "cancelled" and task["result"] is None
    assert use.search_of("ds") is None and asked == []
    np.testing.assert_allclose(store.get_pc("ds"), START_PC, atol=1e-9)
    np.testing.assert_allclose(store.get_pc("other"), START_PC, atol=1e-12)


@pytest.mark.parametrize("how", ["file_unreadable", "read_raises"])
def test_a_followers_read_that_fails_never_fails_a_finished_run(route, calibrations, monkeypatch, how):
    R, ctrl, store, _ = route
    cal, use = calibrations
    made = _two_members(cal, use, store)
    if how == "file_unreadable":
        (cal.calibrations_dir() / f"{made['id']}.json").write_text("{ not json", encoding="utf-8")
    else:
        def broken(*a, **k):
            raise OSError("the calibration folder is locked")
        monkeypatch.setattr(use, "followers", broken)
    stub_search(monkeypatch, [FOUND] * 2)
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["status"] == "completed", task.get("error")
    assert task["result"]["calibration"] == {"calibration": None, "diverged": False, "followers": []}
    np.testing.assert_allclose(store.get_pc("ds"), FOUND, atol=1e-9)                # the PC stays written
    np.testing.assert_allclose(task["result"]["mean_pc"], FOUND, atol=1e-9)


# ------------------------------------------------------------------ a PC the store did not take is said

NOT_STORED_TEXT = "This PC is on this page only: indexing will not use it. Load the file again."


def _unknown_to_the_store(R, ctrl, monkeypatch, name="ghost"):
    """The page is on ``name``, a dataset the calibration store never registered."""
    from backend.api.routes import ebsd_viewer
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", name)
    monkeypatch.setattr(R, "_sessions", {name: R._PCSession(ctrl=ctrl)})


def test_a_found_pc_for_a_dataset_the_store_does_not_know_is_not_stored_and_says_so(route, monkeypatch):
    R, ctrl, store, _ = route
    _unknown_to_the_store(R, ctrl, monkeypatch)
    assert store.get_entry("ghost") is None
    stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    assert task["result"]["pc_stored"] is False
    assert "pcNotStored" in _codes(task)
    assert NOT_STORED_TEXT in task["result"]["pc_warning"]
    np.testing.assert_allclose(store.get_pc("ds"), START_PC, atol=1e-12)        # nothing leaked to another entry


def test_a_found_pc_for_a_known_dataset_is_stored_and_carries_no_such_warning(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    assert task["result"]["pc_stored"] is True
    assert "pcNotStored" not in _codes(task)
    assert "indexing will not use it" not in (task["result"]["pc_warning"] or "")


def test_a_typed_pc_for_a_dataset_the_store_does_not_know_is_not_stored_and_says_so(route, monkeypatch):
    R, ctrl, store, _ = route
    _unknown_to_the_store(R, ctrl, monkeypatch)
    answer = asyncio.run(R.update_pc(R.UpdatePCRequest(pcx=0.45, pcy=0.35, pcz=0.55)))
    assert answer["success"] is True and answer["pc"] == [0.45, 0.35, 0.55]
    assert answer["pc_stored"] is False
    np.testing.assert_allclose(store.get_pc("ds"), START_PC, atol=1e-12)


def test_a_typed_pc_for_a_known_dataset_is_stored(route, monkeypatch):
    R, ctrl, store, _ = route
    monkeypatch.setattr(R, "_sessions", {"ds": R._PCSession(ctrl=ctrl)})
    answer = asyncio.run(R.update_pc(R.UpdatePCRequest(pcx=0.45, pcy=0.35, pcz=0.55)))
    assert answer["pc_stored"] is True
    np.testing.assert_allclose(store.get_pc("ds"), [0.45, 0.35, 0.55], atol=1e-12)


# ------------------------------------------------------------------ quality of the result

def test_the_result_carries_its_quality_and_a_find_has_no_contrast_before(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM")
    task, _ = _start(R, ctrl, FIVE, mode="find")
    assert task["result"]["quality"] == {"contrast": 0.95, "warning": None, "axes": [], "n_patterns": 5,
                                         "contrast_before": None}


def test_a_refine_result_carries_the_contrast_of_its_start(route, monkeypatch):
    R, ctrl, store, _ = route
    stub_search(monkeypatch, [FOUND] * 2, contrast_at_start=0.55, on_edge=["z"])
    task, _ = _start(R, ctrl, [(0, 0), (1, 1)])
    assert task["result"]["quality"] == {"contrast": 0.95, "warning": "edge", "axes": ["z"], "n_patterns": 2,
                                         "contrast_before": 0.55}
