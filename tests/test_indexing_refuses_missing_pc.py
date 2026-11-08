"""No indexing run starts on a file whose pattern centre nobody supplied.

A file without a PC (an EDAX UP1/UP2 without its .osc) is registered as
``missing``, but ``/api/indexing/start``, both batch starts and the batch job
fetched ``get_detector()`` without looking, and indexed at kikuchipy's placeholder
(0.5, 0.5, 0.5): confident, wrong orientations. They now answer a coded 400
(``X-Orienta-Code: pc_missing``) that tells the user to calibrate the PC or provide
the .osc, and files with a header, .osc, typed or refined PC are untouched.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import indexing as I
from backend.api.services.calibration_store import CalibrationStore


class _Started(Exception):
    """Raised by a stub placed right after the guard: the request got past it."""


@pytest.fixture(scope="module")
def nickel():
    import kikuchipy as kp
    return kp.data.nickel_ebsd_small()


def _store(nickel, name, source):
    store = CalibrationStore()
    sig = nickel.deepcopy()
    if source == "missing":
        sig.metadata.set_item("Signal.pc_source", "default")
    elif source == "osc":
        sig.metadata.set_item("Signal.pc_source", "osc")
    # registered as the viewer registers a load: with the canonical path of its file
    from backend.api.routes.ebsd_viewer import _canonical_path
    store.register(name, sig, file_key=_canonical_path(f"C:/data/{name}.up1"))
    if source in ("manual", "refined"):
        store.update_pc(name, [0.51, 0.4, 0.62], source=source)
    return store


@pytest.fixture
def start(monkeypatch, nickel):
    """POST /start for the active dataset 'ds' with a calibration of the given kind."""
    from backend.api.routes import ebsd_viewer

    def run(source, method="hough"):
        monkeypatch.setattr(I, "calibration_store", _store(nickel, "ds", source))
        monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
        monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: nickel)

        def started(*a, **k):
            raise _Started()

        monkeypatch.setattr(I, "_track_task", started)       # the next line after the guard
        return TestClient(app, raise_server_exceptions=True).post("/api/indexing/start", json={"method": method})
    return run


@pytest.mark.parametrize("method", ["hough", "dictionary", "spherical"])
def test_start_refuses_every_method_without_a_pc(start, method):
    resp = start("missing", method)
    assert resp.status_code == 400
    assert resp.headers["x-orienta-code"] == "pc_missing"
    assert "Calibrate the PC" in resp.json()["detail"] and ".osc" in resp.json()["detail"]


@pytest.mark.parametrize("source", ["header", "osc", "manual", "refined"])
def test_start_is_untouched_for_a_file_with_a_pc(start, source):
    with pytest.raises(_Started):                 # got past the guard to the task
        start(source)


def test_batch_start_refuses_a_registered_file_without_a_pc(monkeypatch, nickel):
    monkeypatch.setattr(I, "calibration_store", _store(nickel, "scan", "missing"))
    resp = TestClient(app).post("/api/indexing/batch/start", json={"datasets": [
        {"file_path": "C:/data/scan.up1", "method": "hough"}]})
    assert resp.status_code == 400 and resp.headers["x-orienta-code"] == "pc_missing"
    assert I._batch_state["running"] is False


def test_batch_start_lets_a_request_that_brings_its_own_pc_through(monkeypatch, nickel):
    monkeypatch.setattr(I, "calibration_store", _store(nickel, "scan", "missing"))

    def started(*a, **k):
        raise _Started()

    monkeypatch.setattr(I, "_submit_run_guarded", started)
    saved = dict(I._batch_state)
    try:
        with pytest.raises(_Started):
            TestClient(app).post("/api/indexing/batch/start", json={"datasets": [
                {"file_path": "C:/data/scan.up1", "method": "hough", "pc": [0.5, 0.4, 0.6]}]})
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved)


def test_batch_job_fails_a_file_that_turns_out_to_have_no_pc(monkeypatch, nickel):
    """Not registered when the batch is started: registered as missing when it is loaded."""
    from backend.api.routes import ebsd_viewer
    store = CalibrationStore()
    monkeypatch.setattr(I, "calibration_store", store)

    def load(path):
        sig = nickel.deepcopy()
        sig.metadata.set_item("Signal.pc_source", "default")
        from backend.api.routes.ebsd_viewer import _canonical_path
        store.register("late", sig, file_key=_canonical_path(path))
        return {"ok": True}

    monkeypatch.setattr(ebsd_viewer, "load_ebsd_file", load)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: nickel)
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())
    saved = dict(I._batch_state)
    try:
        resp = TestClient(app).post("/api/indexing/batch/start", json={"datasets": [
            {"file_path": "C:/data/late.up1", "method": "hough"}]})
        assert resp.status_code == 200
        entry = I._batch_state["results"][-1]
        assert entry["dataset"] == "late" and "carries no pattern centre" in entry["error"]
        assert entry["n_indexed"] == 0
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved)


def test_the_backend_batch_refuses_at_start_and_in_the_job(monkeypatch, nickel):
    from backend.api.routes import batch_v2 as B
    from backend.api.services import batch_manager as M

    store = _store(nickel, "scan", "missing")
    monkeypatch.setattr(B, "calibration_store", store)
    mgr = SimpleNamespace(
        get_status=lambda bid: {"status": "pending"},
        get_jobs=lambda bid: [{"file_path": "C:/data/scan.up1"}],
    )
    monkeypatch.setattr(B, "_get_manager", lambda: mgr)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        asyncio.run(B.start_batch("b1"))
    assert exc.value.status_code == 400
    assert exc.value.headers["X-Orienta-Code"] == "pc_missing"

    # and the job itself, for a file registered after the batch was started
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    with pytest.raises(RuntimeError, match="carries no pattern centre"):
        M._get_detector_for_file("C:/data/scan.up1", nickel)
    ok = _store(nickel, "scan", "header")
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", ok)
    assert M._get_detector_for_file("C:/data/scan.up1", nickel) is not None


UP1 = os.environ.get("ORIENTA_TEST_UP1")        # a real .up1 whose .osc sidecar is NOT next to it


@pytest.mark.skipif(not UP1 or not Path(UP1).is_file(), reason="ORIENTA_TEST_UP1 not set")
def test_a_real_up1_without_its_osc_is_refused(monkeypatch):
    from backend.api.routes import ebsd_viewer
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(UP1, verbose=False)
    store = CalibrationStore()
    store.register("up1", sig)
    monkeypatch.setattr(I, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "up1")
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: sig)
    resp = TestClient(app).post("/api/indexing/start", json={"method": "hough"})
    assert resp.status_code == 400 and resp.headers["x-orienta-code"] == "pc_missing"
