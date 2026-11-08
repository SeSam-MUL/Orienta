"""The phase test and the orientation refine refuse a file with no PC with a code.

``/single-pixel-phase-test``, ``/single-pixel-phase-test/start`` and ``/refine`` said
"Pattern Center is not set" in English prose without ``X-Orienta-Code``, so the dialog
showed an untranslated sentence in a German window and the other starts answered with
the code. All three now go through ``pc_guard.require_pc``: the same coded 400, the same
text. A file with a PC of any origin is untouched.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import indexing as I
from backend.api.services.calibration_store import CalibrationStore


@pytest.fixture(scope="module")
def nickel():
    import kikuchipy as kp
    return kp.data.nickel_ebsd_small()


@pytest.fixture
def client(monkeypatch, nickel):
    from backend.api.routes import ebsd_viewer

    def use(source):
        store = CalibrationStore()
        sig = nickel.deepcopy()
        if source == "missing":
            sig.metadata.set_item("Signal.pc_source", "default")
        store.register("ds", sig)
        if source in ("manual", "refined"):
            store.update_pc("ds", [0.51, 0.4, 0.62], source=source)
        monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
        monkeypatch.setattr(I, "calibration_store", store)
        monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
        monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: nickel)
        monkeypatch.setattr(I, "_get_active_signal_for_phase_test", lambda: nickel)
        return TestClient(app, raise_server_exceptions=True)
    return use


PHASE_TEST_ROUTES = ["/api/indexing/single-pixel-phase-test",
                     "/api/indexing/single-pixel-phase-test/start"]


@pytest.mark.parametrize("url", PHASE_TEST_ROUTES)
def test_a_phase_test_without_a_pc_is_refused_with_the_code(client, url):
    resp = client("missing").post(url, json={"pixel_index": 3})
    assert resp.status_code == 400
    assert resp.headers["x-orienta-code"] == "pc_missing"
    assert "Calibrate the PC" in resp.json()["detail"]


@pytest.mark.parametrize("url", PHASE_TEST_ROUTES)
@pytest.mark.parametrize("source", ["header", "manual", "refined"])
def test_a_phase_test_with_a_pc_gets_past_the_guard(client, monkeypatch, url, source):
    class _Past(Exception):
        pass

    def stop(*a, **k):
        raise _Past()

    # the first thing after the guard on both routes reads the measured pattern's size
    monkeypatch.setattr(I, "_phase_test_library_entries", stop)
    monkeypatch.setattr(I, "_bg_remove_phase_test", stop)
    c = client(source)
    try:
        resp = c.post(url, json={"pixel_index": 3, "bg_remove": True})
    except _Past:
        return
    assert resp.headers.get("x-orienta-code") != "pc_missing", resp.text


def test_the_orientation_refine_is_refused_with_the_code(client, monkeypatch, tmp_path):
    master = tmp_path / "master.h5"
    master.write_bytes(b"x")
    monkeypatch.setattr(I, "_get_result", lambda: SimpleNamespace(method=SimpleNamespace(value="dictionary")))
    resp = client("missing").post("/api/indexing/refine", json={"master_h5_path": str(master)})
    assert resp.status_code == 400
    assert resp.headers["x-orienta-code"] == "pc_missing"
