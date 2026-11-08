"""The PC routes that index a pattern do it off the event loop and under the state lock.

Selecting a calibration pattern whose result is not cached (``GET /pattern/{idx}/result``)
and editing the PC or the tilts (``POST /detector/update-pc``, ``/detector/update-tilt``)
index the pattern with Hough: building the indexer takes seconds to minutes. The three
routes are ``async def`` and called that work directly, so the whole backend stopped
answering (viewer, progress polls) meanwhile, and they changed the controller outside
``_state_lock``, which ``/index-pattern`` and a running refine rely on.
"""
import asyncio
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from kikuchipy.detectors import EBSDDetector

SHAPE = (60, 60)
SLOW_S = 1.5


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
    ctrl = R._get_controller()
    ctrl.load_phase(_cif(tmp_path))
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0))
    ctrl.add_pattern((0, 0), np.zeros(SHAPE, np.uint8))
    seen = {}

    def slow_index(c, idx, sim_cache=None):
        seen["thread"] = threading.current_thread()
        seen["lock_held"] = R._state_lock.locked()
        time.sleep(SLOW_S)
        result = {"ci": 0.5, "phase_name": "Ni", "segments": [], "n_bands": 0,
                  "pc": [0.5, 0.4, 0.6]}
        (R._sim_cache if sim_cache is None else sim_cache)[idx] = result
        return result

    monkeypatch.setattr(R, "_index_and_simulate", slow_index)
    app = FastAPI()
    app.include_router(R.router, prefix="/api/pc")
    return app, R, ctrl, seen


def _call(app, method, url, body=None):
    """Run the request while a ticker runs on the same loop; return (response, ticks)."""
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
            if method == "GET":
                resp = await ac.get(url)
            else:
                resp = await ac.post(url, json=body)
            stop.set()
            await tick_task
        return resp, ticks

    return asyncio.run(scenario())


ROUTES = [
    ("GET", "/api/pc/pattern/0/result", None),
    ("POST", "/api/pc/detector/update-pc",
     {"pcx": 0.51, "pcy": 0.41, "pcz": 0.61, "current_pattern_idx": 0}),
    ("POST", "/api/pc/detector/update-tilt",
     {"sample_tilt": 69.0, "current_pattern_idx": 0}),
]


@pytest.mark.parametrize("method,url,body", ROUTES, ids=["result", "update-pc", "update-tilt"])
def test_indexing_a_pattern_leaves_the_event_loop_free(env, method, url, body):
    app, R, ctrl, seen = env
    resp, ticks = _call(app, method, url, body)
    assert resp.status_code == 200, resp.text
    # Over SLOW_S a free loop ticks ~30 times; one that is blocked while the
    # indexing runs ticks (almost) not at all.
    assert ticks >= 10, f"the event loop was blocked: {ticks} ticks in {SLOW_S} s of indexing"
    assert seen["thread"] is not threading.main_thread()


@pytest.mark.parametrize("method,url,body", ROUTES, ids=["result", "update-pc", "update-tilt"])
def test_indexing_a_pattern_holds_the_state_lock(env, method, url, body):
    app, R, ctrl, seen = env
    resp, _ = _call(app, method, url, body)
    assert resp.status_code == 200, resp.text
    assert seen["lock_held"] is True, "the controller was indexed outside _state_lock"


def test_a_cached_result_is_still_answered_without_indexing(env):
    app, R, ctrl, seen = env
    R._sim_cache[0] = {"ci": 0.9, "phase_name": "Ni", "segments": [], "n_bands": 0}
    resp, _ = _call(app, "GET", "/api/pc/pattern/0/result")
    body = resp.json()
    assert body["cached"] is True and body["ci"] == 0.9
    assert "thread" not in seen
