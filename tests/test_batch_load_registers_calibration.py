"""The batch's own file loader registers the calibration, and the guard fails closed.

``load_ebsd_file`` (the loader of ``/api/indexing/batch/start``) put the signal in the
viewer's registry but never registered a calibration entry. ``is_pc_missing`` answered
False when there was no entry, so a file that carries no PC (EDAX UP1 without its
.osc) was indexed at kikuchipy's placeholder (0.5, 0.5, 0.5) with no error; and a PC
the user had typed on another file was consumed from the stash with nothing to put it
back into. Everything here runs the REAL loader, the real functions and real files: a
synthetic UP1 in the v1 layout (16-byte header, uint8 patterns) with no .osc beside it.
"""
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import indexing as I
from backend.api.services.calibration_store import CalibrationStore
from backend.api.services.pc_guard import is_pc_missing

SIDE = 24                      # pattern edge, px
TYPED = [0.48, 0.52, 0.60]


def _write_up1(path, n_side=4):
    """A version-1 UP1: n_side x n_side patterns of SIDE x SIDE uint8, no grid in the header."""
    rng = np.random.default_rng(3)
    pats = rng.integers(0, 250, (n_side * n_side, SIDE, SIDE)).astype("u1")
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, SIDE, SIDE, 16))
        f.write(pats.tobytes())
    return str(path)


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    from backend.api.routes import ebsd_viewer as ev
    store = CalibrationStore()
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(I, "calibration_store", store)
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    monkeypatch.setattr(ev, "_active_dataset", "")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    files = {n: _write_up1(tmp_path / f"{n}.up1") for n in ("scan_a", "scan_b")}
    return ev, store, files


def test_the_batch_loader_registers_a_file_without_a_pc_as_missing(viewer):
    ev, store, files = viewer
    assert ev.load_ebsd_file(files["scan_a"]) is True
    entry = store.get_entry("scan_a")
    assert entry is not None and entry.pc_source == "missing"
    assert store.get_pc("scan_a") is None


def test_the_legacy_batch_refuses_a_file_without_a_pc_and_nothing_is_indexed(viewer, monkeypatch):
    ev, store, files = viewer
    monkeypatch.setattr(I, "_submit_run_guarded", lambda fn, kind="": fn())    # run the job inline
    saved = dict(I._batch_state)
    try:
        resp = TestClient(app).post("/api/indexing/batch/start", json={"datasets": [
            {"file_path": files["scan_a"], "method": "hough"}]})
        assert resp.status_code == 200, resp.text      # unregistered when started: checked at load
        entry = I._batch_state["results"][-1]
        assert entry["dataset"] == "scan_a" and "carries no pattern centre" in entry["error"]
        assert entry["n_indexed"] == 0
        # the file the batch left active is not indexable either
        resp = TestClient(app).post("/api/indexing/start", json={"method": "hough"})
        assert resp.status_code == 400 and resp.headers["x-orienta-code"] == "pc_missing"
    finally:
        I._batch_state.clear()
        I._batch_state.update(saved)


def test_a_pc_typed_on_a_file_survives_the_batch_loading_other_files(viewer):
    ev, store, files = viewer
    ev.load_ebsd_file(files["scan_a"])
    store.update_pc("scan_a", TYPED, source="manual")
    ev.load_ebsd_file(files["scan_b"])                 # the batch moves on to another file
    assert store.get_entry("scan_b").pc_source == "missing"
    ev.load_ebsd_file(files["scan_a"])                 # and back to the first
    entry = store.get_entry("scan_a")
    assert entry.pc_source == "manual"
    np.testing.assert_allclose(store.get_pc("scan_a"), TYPED, atol=1e-12)
    assert is_pc_missing("scan_a", store) is False     # so the batch may index it


def test_the_guard_fails_closed_without_a_calibration_entry(viewer):
    """No entry: the signal's own record decides, a PC nobody supplied is missing."""
    from safe_loader import load_ebsd_safe
    ev, store, files = viewer
    sig = load_ebsd_safe(files["scan_a"], verbose=False)
    assert store.get_entry("scan_a") is None
    assert is_pc_missing("scan_a", store, sig) is True
    # a signal whose loader recorded a PC, or one that recorded nothing, is not missing
    sig.metadata.set_item("Signal.pc_source", "osc")
    assert is_pc_missing("scan_a", store, sig) is False
    sig.metadata.set_item("Signal.pc_source", None)
    assert is_pc_missing("scan_a", store, sig) is False
    assert is_pc_missing("never_loaded", store) is False


def test_the_guard_finds_the_viewers_signal_of_that_name(viewer):
    from safe_loader import load_ebsd_safe
    ev, store, files = viewer
    ev._raw_signals["scan_a"] = load_ebsd_safe(files["scan_a"], verbose=False)
    assert store.get_entry("scan_a") is None
    assert is_pc_missing("scan_a", store) is True        # the caller passed no signal


def test_the_guard_takes_the_viewers_signal_only_for_the_viewers_file(viewer):
    """The viewer's signal of a NAME is the loaded file's: asked about another file of the
    same stem it is no evidence (it would refuse a file that has a PC)."""
    from safe_loader import load_ebsd_safe
    ev, store, files = viewer
    ev._raw_signals["scan_a"] = load_ebsd_safe(files["scan_a"], verbose=False)    # tagged: no PC
    ev._ebsd_file_path = files["scan_a"]
    assert is_pc_missing("scan_a", store, file_key=ev._canonical_path(files["scan_a"])) is True
    other = ev._canonical_path(files["scan_b"])                                    # another file, same name asked
    assert is_pc_missing("scan_a", store, file_key=other) is False


def test_an_entry_without_a_file_key_is_no_evidence_about_a_file(viewer):
    """Asked about a file, the guard ignores an entry registered without a path: it says
    nothing about that file (it was the foreign Oxford entry import-h5 left)."""
    ev, store, files = viewer
    sig = ev_signal_default = None
    from safe_loader import load_ebsd_safe
    sig = load_ebsd_safe(files["scan_a"], verbose=False)               # tagged: no PC
    store.register("scan_a", sig)                                      # ... registered without a path
    assert store.get_entry("scan_a").pc_source == "missing"
    key = ev._canonical_path(files["scan_a"])
    assert is_pc_missing("scan_a", store) is True                      # asked by name only: as before
    assert is_pc_missing("scan_a", store, file_key=key) is False       # asked about the FILE: not its entry
