"""``GET /api/pc/status`` is ``async def`` and answers ``pc_in_use``, which asks whether the dataset still runs on
a saved calibration (``current_ref``: reads the calibration files under a lock with retries and, after a restart,
hashes the head of the data file). That work runs off the event loop, as the other PC routes' does."""
import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from kikuchipy.detectors import EBSDDetector

SHAPE = (60, 60)
SLOW_S = 1.5


@pytest.fixture
def env(monkeypatch):
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from backend.api.services import pc_calibration_apply as use
    from backend.api.services.calibration_store import CalibrationStore
    store = CalibrationStore()
    store.register("ds", SimpleNamespace(detector=EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0)),
                   file_key="/folder/ds.h5")
    monkeypatch.setattr(R, "_sessions", {})
    monkeypatch.setattr(R, "_parked_sessions", {})
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(R, "_optimization_active", False)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", None, raising=False)
    seen = {"threads": []}
    real = use.current_ref

    def slow_current_ref(*args, **kwargs):
        seen["threads"].append(threading.current_thread())
        time.sleep(SLOW_S)
        return real(*args, **kwargs)

    monkeypatch.setattr(use, "current_ref", slow_current_ref)
    app = FastAPI()
    app.include_router(R.router, prefix="/api/pc")
    return app, seen


def _get_with_ticker(app, url):
    async def scenario():
        ticks = 0
        stop = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not stop.is_set():
                await asyncio.sleep(0.05)
                ticks += 1

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            tick_task = asyncio.create_task(ticker())
            resp = await ac.get(url)
            stop.set()
            await tick_task
        return resp, ticks

    return asyncio.run(scenario())


def test_status_asks_for_the_calibration_off_the_event_loop(env):
    app, seen = env
    resp, ticks = _get_with_ticker(app, "/api/pc/status")
    assert resp.status_code == 200, resp.text
    assert seen["threads"], "current_ref was not asked: the test does not reach the call"
    assert all(t is not threading.main_thread() for t in seen["threads"])
    assert ticks >= 10, f"the event loop was blocked: {ticks} ticks in {SLOW_S} s of waiting"
    body = resp.json()
    assert body["pc_in_use"]["kind"] == "file" and body["pc_in_use"]["pc"] == pytest.approx([0.5, 0.4, 0.6])
