"""POST /api/batch-v2/quick-load does not reset a calibration the user made.

Quick-load registered the file unconditionally, so the active file's typed PC became
"missing" again (and the PC page, which keeps its own copy, still showed it). What the
user made stays; what the file gave is refreshed. Real route, real UP1 files without an
.osc (see test_pc_restore_policy for the layout).
"""
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import batch_v2 as B
from backend.api.services.calibration_store import CalibrationStore

SIDE = 24
TYPED = [0.48, 0.52, 0.60]


def _write_up1(path, n_side=4):
    rng = np.random.default_rng(3)
    pats = rng.integers(0, 250, (n_side * n_side, SIDE, SIDE)).astype("u1")
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, SIDE, SIDE, 16))
        f.write(pats.tobytes())
    return str(path)


@pytest.fixture
def store(monkeypatch):
    s = CalibrationStore()
    monkeypatch.setattr(B, "calibration_store", s)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", s)
    return s


def _quick_load(path):
    resp = TestClient(app).post("/api/batch-v2/quick-load", json={"file_path": path})
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.parametrize("source", ["manual", "refined"])
def test_a_calibration_the_user_made_survives_a_quick_load(store, tmp_path, source):
    path = _write_up1(tmp_path / "scan_a.up1")
    first = _quick_load(path)
    assert first["pc"] is None and first["pc_source"] == "none"        # no PC in the file
    store.update_pc("scan_a", TYPED, source=source)
    again = _quick_load(path)                                          # e.g. the batch page is opened
    assert store.get_entry("scan_a").pc_source == source               # not "missing" again
    np.testing.assert_allclose(store.get_pc("scan_a"), TYPED, atol=1e-12)
    np.testing.assert_allclose(again["pc"], TYPED, atol=1e-12)
    assert again["pc_source"] == source


def test_what_the_file_gave_is_refreshed(store, tmp_path):
    """No calibration of the user's: the file is read afresh, as before."""
    path = _write_up1(tmp_path / "scan_b.up1")
    _quick_load(path)
    entry_before = store.get_entry("scan_b")
    _quick_load(path)
    assert store.get_entry("scan_b") is not entry_before               # registered anew
    assert store.get_entry("scan_b").pc_source == "missing"
