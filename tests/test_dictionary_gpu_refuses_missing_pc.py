"""POST /api/dictionary-gpu/generate does not bake the placeholder PC into a dictionary.

The pattern centre comes from the page. For a file that carries none the page holds
kikuchipy's placeholder (0.5, 0.5, 0.5); a dictionary simulated at it is wrong for
every run that later reads it. The route answers the coded 400 (pc_missing) of the
indexing starts when the PC is that placeholder and the active dataset has no PC.
A PC typed by hand (any other value), a dataset with a PC, and a stand-alone use
with no dataset open all go through.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.services.calibration_store import CalibrationStore

PLACEHOLDER = [0.5, 0.5, 0.5]
TYPED = [0.5026, 0.3273, 0.8458]


class _Det:
    shape = (60, 60)
    pc = [0.5, 0.5, 0.5]
    sample_tilt, tilt, azimuthal = 70.0, 0.0, 0.0


@pytest.fixture
def post(monkeypatch, tmp_path):
    from backend.api.routes import ebsd_viewer
    master = tmp_path / "master.h5"
    master.write_bytes(b"\x89HDF\r\n\x1a\n")

    def run(pc, dataset_source):
        store = CalibrationStore()
        if dataset_source is not None:
            sig = SimpleNamespace(detector=_Det())
            if dataset_source == "missing":
                sig.metadata = type("M", (), {"get_item": staticmethod(lambda k, d=None: "default")})()
            store.register("ds", sig)
        monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
        monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds" if dataset_source else "")
        payload = {"master_path": str(master), "detector_shape": [60, 60], "pc": pc,
                   "sample_tilt": 70.0, "resolution_deg": 5.0}
        with patch("backend.api.routes.dictionary_gpu.threading.Thread") as thread:
            resp = TestClient(app).post("/api/dictionary-gpu/generate", json=payload)
        return resp, thread
    return run


def test_the_placeholder_pc_of_a_file_without_one_is_refused(post):
    resp, thread = post(PLACEHOLDER, "missing")
    assert resp.status_code == 400
    assert resp.headers["x-orienta-code"] == "pc_missing"
    assert not thread.called                                  # nothing was started


@pytest.mark.parametrize("pc,source", [
    (TYPED, "missing"),            # the user typed a PC: theirs to choose
    ([0.5, 0.5, 0.5 + 1e-3], "missing"),   # near the placeholder but not the stored one
    (PLACEHOLDER, "header"),       # a file whose header really says 0.5
    (PLACEHOLDER, None),           # stand-alone: no dataset open
])
def test_everything_else_goes_through(post, pc, source):
    resp, thread = post(pc, source)
    assert resp.status_code == 200, resp.text
    assert thread.called


def test_the_comparison_is_with_the_stored_placeholder_not_with_0_5(monkeypatch, tmp_path):
    """The entry says what the placeholder is: a different stored value is what is refused."""
    from backend.api.routes import ebsd_viewer
    store = CalibrationStore()
    det = _Det()
    det.pc = [0.43, 0.61, 0.52]
    sig = SimpleNamespace(detector=det,
                          metadata=type("M", (), {"get_item": staticmethod(lambda k, d=None: "default")})())
    store.register("ds", sig)
    assert store.get_entry("ds").pc_source == "missing"
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    master = tmp_path / "master.h5"
    master.write_bytes(b"HDF")
    payload = {"master_path": str(master), "detector_shape": [60, 60], "pc": [0.43, 0.61, 0.52],
               "sample_tilt": 70.0, "resolution_deg": 5.0}
    with patch("backend.api.routes.dictionary_gpu.threading.Thread") as thread:
        resp = TestClient(app).post("/api/dictionary-gpu/generate", json=payload)
    assert resp.status_code == 400 and resp.headers["x-orienta-code"] == "pc_missing"
    assert not thread.called
