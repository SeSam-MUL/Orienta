"""Adding or removing a pattern, or typing a PC or tilt, is refused while a refine runs.

A Global PC Refine and a grid calibration iterate the controller's calibration
patterns for minutes. ``/pattern/remove`` popped one (and ``/pattern/add`` appended
one) meanwhile, so the indices shifted under the running job; a PC or tilt typed on
the page replaced the detector it works on. The routes answer 409 with a coded
message instead, and change the pattern list under ``_state_lock``.
"""
import asyncio

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from kikuchipy.detectors import EBSDDetector

SHAPE = (60, 60)


class _Axes:
    navigation_shape = (4, 4)


class _Signal:
    axes_manager = _Axes()

    def __init__(self):
        self.data = np.zeros((4, 4) + SHAPE, dtype=np.uint8)


@pytest.fixture
def env(monkeypatch):
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
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0))
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.uint8))
    ctrl.add_pattern((1, 1), np.zeros(SHAPE, np.uint8))
    app = FastAPI()
    app.include_router(R.router, prefix="/api/pc")
    return app, R, ctrl


def _request(app, method, url, body=None):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.request(method, url, json=body)
    return asyncio.run(go())


EDITS = [
    ("POST", "/api/pc/pattern/add", {"row": 2, "col": 2}),
    ("POST", "/api/pc/pattern/remove?idx=0", None),
    ("POST", "/api/pc/detector/update-pc", {"pcx": 0.6, "pcy": 0.5, "pcz": 0.7}),
    ("POST", "/api/pc/detector/update-tilt", {"sample_tilt": 65.0}),
]
IDS = ["add", "remove", "update-pc", "update-tilt"]


@pytest.mark.parametrize("method,url,body", EDITS, ids=IDS)
def test_an_edit_is_refused_while_a_refine_runs(env, method, url, body):
    app, R, ctrl = env
    R._optimization_active = True
    pc_before = np.asarray(ctrl.detector.pc, dtype=float).copy()
    tilt_before = float(ctrl.detector.sample_tilt)
    resp = _request(app, method, url, body)
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "pc_job_running"
    assert "optimization is running" in detail["message"]
    # nothing was changed behind the running job
    assert [c for c, _ in ctrl.patterns] == [(0, 0), (1, 1)]
    assert np.array_equal(np.asarray(ctrl.detector.pc, dtype=float), pc_before)
    assert float(ctrl.detector.sample_tilt) == tilt_before
    assert R._optimization_active is True              # the running job's flag is not touched


def test_without_a_refine_the_edits_still_work(env):
    app, R, ctrl = env
    r = _request(app, "POST", "/api/pc/pattern/add", {"row": 2, "col": 2})
    assert r.status_code == 200 and r.json()["n_patterns"] == 3
    r = _request(app, "POST", "/api/pc/pattern/remove?idx=0")
    assert r.status_code == 200 and r.json()["n_patterns"] == 2
    assert [c for c, _ in ctrl.patterns] == [(1, 1), (2, 2)]
    r = _request(app, "POST", "/api/pc/detector/update-pc", {"pcx": 0.6, "pcy": 0.5, "pcz": 0.7})
    assert r.status_code == 200
    assert np.allclose(np.asarray(ctrl.detector.pc, dtype=float).ravel(), [0.6, 0.5, 0.7])
    r = _request(app, "POST", "/api/pc/detector/update-tilt", {"sample_tilt": 65.0})
    assert r.status_code == 200 and float(ctrl.detector.sample_tilt) == 65.0
    r = _request(app, "POST", "/api/pc/pattern/remove?idx=7")
    assert r.status_code == 400                         # an index out of range is still a 400


@pytest.mark.parametrize("method,url,body", EDITS[:2], ids=IDS[:2])
def test_the_pattern_list_is_changed_under_the_state_lock(env, method, url, body):
    """An Index All holds the lock while it iterates the list: add and remove wait
    for it, and the event loop does not wait with them."""
    app, R, ctrl = env

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            R._state_lock.acquire()
            try:
                task = asyncio.create_task(ac.request(method, url, json=body))
                await asyncio.sleep(0.5)              # a free loop gets here; a blocked one does not
                assert not task.done(), "the list was changed while the lock was held"
                assert [c for c, _ in ctrl.patterns] == [(0, 0), (1, 1)]
            finally:
                R._state_lock.release()
            return await asyncio.wait_for(task, 10)

    resp = asyncio.run(go())
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("method,url,body", EDITS, ids=IDS)
def test_a_refine_that_starts_while_the_edit_waits_for_the_lock_still_refuses_it(env, method, url, body):
    app, R, ctrl = env

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            R._state_lock.acquire()
            try:
                task = asyncio.create_task(ac.request(method, url, json=body))
                await asyncio.sleep(0.5)
                R._optimization_active = True          # the refine is accepted meanwhile
            finally:
                R._state_lock.release()
            return await asyncio.wait_for(task, 10)

    resp = asyncio.run(go())
    assert resp.status_code == 409, resp.text
    assert [c for c, _ in ctrl.patterns] == [(0, 0), (1, 1)]
