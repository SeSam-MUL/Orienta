"""Race guards in the PC refinement routes.

1. A Global PC Refine takes the cache of the session it was started on from the
   route; looked up later, a file switch in between sent it to another dataset.
2. /pattern/{idx}/result looks into the cache once (.get): worker threads clear
   it concurrently, and "is it there?" followed by "give it to me" could 500.
3. /pattern/{idx}/result checks the pattern again once it holds the lock: a
   Remove that ran while it waited turned into an IndexError (500).
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import BackgroundTasks, HTTPException

from tests.test_pc_job_dataset_capture import SHAPE, route  # noqa: F401


def test_the_refine_route_hands_over_its_sessions_cache(route):
    R, ctrl, store, ebsd_viewer = route
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    bt = BackgroundTasks()
    asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=[5], method="PSO", search_limit=0.05), bt))
    assert bt.tasks[0].kwargs["sim_cache"] is R._sessions["ds"].sim_cache


def test_the_job_writes_into_the_cache_it_was_given(route, monkeypatch):
    """Even when its session is gone from _sessions by the time it runs."""
    R, ctrl, store, ebsd_viewer = route
    from tests.pc_engine_stub import stub_search
    stub_search(monkeypatch, [(0.5, 0.4, 0.6)])
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    mine = {3: "old lines"}
    other = {9: "the other file's"}
    monkeypatch.setattr(R, "_sessions", {})                       # parked meanwhile
    monkeypatch.setattr(R, "_get_session", lambda: SimpleNamespace(sim_cache=other))
    written = []
    monkeypatch.setattr(R, "_index_and_simulate",
                        lambda c, idx, cache: written.append(cache) or {"ci": 0.5, "phase_name": "Ni", "segments": []})
    R._optimization_tasks["t-cache"] = {"status": "running", "progress": 0.0, "result": None, "error": None}
    R._run_optimization("t-cache", [np.zeros(SHAPE, np.uint8)], "PSO", 0.05,
                        [(1, 1)], "ds", ctrl, sim_cache=mine)
    assert R._optimization_tasks["t-cache"]["status"] == "completed"
    assert mine == {} and other == {9: "the other file's"}
    assert all(c is mine for c in written)


@pytest.fixture
def result_route(monkeypatch):
    import backend.api.routes.pcrefinement as R
    ctrl = SimpleNamespace(patterns=[((0, 0), np.zeros(SHAPE)), ((1, 1), np.zeros(SHAPE))],
                           detector=object(), phase=object(), cache={})
    cache = {}
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "_auto_attach_detector", lambda c: None)
    monkeypatch.setattr(R, "_get_session", lambda: SimpleNamespace(sim_cache=cache))
    return R, ctrl, cache


def test_a_pattern_removed_while_waiting_for_the_lock_is_a_400(result_route, monkeypatch):
    R, ctrl, cache = result_route

    class _RemoveWhileWaiting:
        def __enter__(self):
            ctrl.patterns.pop()            # a Remove ran while this request waited
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(R, "_state_lock", _RemoveWhileWaiting())
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: pytest.fail("must not index a gone pattern"))
    with pytest.raises(HTTPException) as e:
        asyncio.run(R.get_pattern_result(1))
    assert e.value.status_code == 400


def test_a_pattern_indexed_by_another_request_meanwhile_is_not_indexed_twice(result_route, monkeypatch):
    R, ctrl, cache = result_route

    class _OtherRequestFilledIt:
        def __enter__(self):
            cache[1] = {"ci": 0.7, "segments": []}
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(R, "_state_lock", _OtherRequestFilledIt())
    monkeypatch.setattr(R, "_index_and_simulate", lambda *a, **k: pytest.fail("indexed twice"))
    out = asyncio.run(R.get_pattern_result(1))
    assert out["cached"] is True and out["ci"] == 0.7


def test_a_long_job_holding_the_lock_makes_an_edit_answer_busy(result_route, monkeypatch):
    """Index All holds _state_lock for minutes without being a refine. A page edit
    (here: Remove) waited silently up to the client's 5-minute timeout; now it
    answers 409 after a short wait."""
    R, ctrl, cache = result_route
    monkeypatch.setattr(R, "_BUSY_WAIT_S", 0.1)
    monkeypatch.setattr(R, "_optimization_active", False)
    R._state_lock.acquire()
    try:
        with pytest.raises(HTTPException) as e:
            asyncio.run(R.remove_pattern(1))
    finally:
        R._state_lock.release()
    assert e.value.status_code == 409
    assert e.value.detail["code"] == "pc_job_running"
    assert len(ctrl.patterns) == 2                     # nothing was removed


def test_a_pattern_result_waits_for_the_lock_instead_of_failing(result_route, monkeypatch):
    """Indexing a pattern on demand IS the long work (a cold indexer takes a
    minute): a second click queues behind it, it is not refused."""
    import threading
    R, ctrl, cache = result_route
    monkeypatch.setattr(R, "_BUSY_WAIT_S", 0.05)
    monkeypatch.setattr(R, "_index_and_simulate",
                        lambda c, idx, sim_cache=None: {"ci": 0.3, "segments": []})
    R._state_lock.acquire()
    threading.Timer(0.4, R._state_lock.release).start()   # the other index finishes later
    out = asyncio.run(R.get_pattern_result(1))
    assert out["success"] is True and out["ci"] == 0.3

def test_the_cache_is_looked_into_once(result_route, monkeypatch):
    """A cache whose "in" says yes but whose entry is gone by the time [] reads it
    (a worker cleared it in between). One .get() cannot fall into that gap."""
    R, ctrl, _ = result_route

    class _ClearedBetween(dict):
        def __contains__(self, k):
            return True                    # "it is there" ...
        def __getitem__(self, k):
            raise KeyError(k)              # ... "no, it was just cleared"

    monkeypatch.setattr(R, "_get_session", lambda: SimpleNamespace(sim_cache=_ClearedBetween()))
    called = []
    monkeypatch.setattr(R, "_index_and_simulate",
                        lambda c, idx, sim_cache=None: called.append(idx) or {"ci": 0.1, "segments": []})
    out = asyncio.run(R.get_pattern_result(0))
    assert out["success"] is True and called == [0]
