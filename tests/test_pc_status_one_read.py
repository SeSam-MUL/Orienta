"""``GET /api/pc/status`` reads the PC the page shows and ``pc_in_use`` in one go.

The route is ``async def``. ``pc_in_use`` is asked off the event loop (it reads the calibration files), and the PC of
the page used to be read on the loop before that, so a PC written while the worker thread worked left the header of
the page with a PC and an "in use" that did not belong together. Both are read in the same worker thread call now.
"""
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_pc_status_off_loop import env  # noqa: F401  (the fixture of the sibling test)


def test_the_pc_and_the_pc_in_use_are_read_in_the_same_worker_thread(env, monkeypatch):
    app, _seen = env
    from backend.api.routes import pcrefinement as R
    from backend.api.services import pc_in_use
    seen = {}
    real_attach, real_report, real_in_use = R._auto_attach_detector, R._pc_report, pc_in_use.pc_in_use_of

    def attach(*args, **kwargs):
        seen["loop"] = threading.get_ident()                   # runs on the event loop, before any worker call
        return real_attach(*args, **kwargs)

    def report(*args, **kwargs):
        seen["pc"] = threading.get_ident()
        return real_report(*args, **kwargs)

    def in_use(*args, **kwargs):
        seen["in_use"] = threading.get_ident()
        return real_in_use(*args, **kwargs)

    monkeypatch.setattr(R, "_auto_attach_detector", attach)
    monkeypatch.setattr(R, "_pc_report", report)
    monkeypatch.setattr(pc_in_use, "pc_in_use_of", in_use)
    r = TestClient(app).get("/api/pc/status")
    assert r.status_code == 200, r.text
    assert set(seen) == {"loop", "pc", "in_use"}, f"the test does not reach all three reads: {sorted(seen)}"
    assert r.json()["pc"], "the premise: the page has a PC"
    assert seen["pc"] == seen["in_use"]
    assert seen["in_use"] != seen["loop"]
