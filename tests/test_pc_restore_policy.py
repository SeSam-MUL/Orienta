"""What a restored calibration and a freshly loaded file decide about the PC, end to end.

Real files and real functions: synthetic EDAX UP1 files in the v1 layout (16-byte
header, uint8 patterns) loaded through ``POST /api/ebsd/load`` and ``load_ebsd_file``,
with an ``.osc`` sidecar written beside one of them for the case "the file now
carries a PC". By hand: the TSL PC of the .osc below is (0.55, 0.50, 0.70).

* a PC the user typed on a COPY reaches the raw dataset as "propagated" and must come
  back with it after a file switch (it used to be lost: not on the list of user-made
  sources);
* the load response names what the STORE holds: a file without a PC whose typed PC
  was restored is not "defaulted" (the viewer would warn that indexing uses the
  placeholder while it uses the typed PC), and says so once ("user_pc_kept");
* an .osc put beside the file afterwards (what the error message advises) makes the
  FILE's PC the newer evidence: it replaces the restored typed PC, and the response
  says so ("file_pc_wins"); a file that did not change keeps the user's PC.
"""
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.services.calibration_store import CalibrationStore

SIDE = 24
TYPED = [0.48, 0.52, 0.60]
OSC_TSL = (0.55, 0.50, 0.70)
OSC_TSL_2 = (0.60, 0.45, 0.75)


def _write_up1(path, n_side=4):
    rng = np.random.default_rng(3)
    pats = rng.integers(0, 250, (n_side * n_side, SIDE, SIDE)).astype("u1")
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, SIDE, SIDE, 16))
        f.write(pats.tobytes())
    return str(path)


def _write_osc(path, pc_tsl, n_side=4):
    """An .osc the real reader takes a grid and a PC from: grid at @16/@20/@24, PC at @1860."""
    from edax_up1 import _OSC_PC_OFFSET
    raw = bytearray(_OSC_PC_OFFSET + 64)
    struct.pack_into("<3I", raw, 16, n_side - 1, n_side - 1, n_side * n_side)
    struct.pack_into("<3f", raw, _OSC_PC_OFFSET, *pc_tsl)
    with open(path, "wb") as f:
        f.write(bytes(raw))


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import indexing as I
    store = CalibrationStore()
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(I, "calibration_store", store)
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file", "_pc_load_notice"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    monkeypatch.setattr(ev, "_active_dataset", "")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    files = {n: _write_up1(tmp_path / f"{n}.up1") for n in ("scan_a", "scan_b")}
    return ev, store, files, tmp_path


def _load(path):
    resp = TestClient(app).post("/api/ebsd/load", json={"path": path})
    assert resp.status_code == 200, resp.text
    return resp.json()


# --- a PC typed on a copy reaches the raw dataset and survives a switch ---------

def test_a_pc_typed_on_a_copy_survives_a_file_switch(viewer):
    from backend.api.routes.pcrefinement import _writeback_refined_pc
    ev, store, files, _ = viewer
    ev.load_ebsd_file(files["scan_a"])
    ev._raw_signals["scan_a_bg"] = ev._raw_signals["scan_a"]            # a copy the user made
    ev._dirty_datasets.add("scan_a_bg")
    store.register_derived("scan_a_bg", "scan_a")
    _writeback_refined_pc("scan_a_bg", TYPED, store=store, source="manual")
    assert store.get_entry("scan_a").pc_source == "propagated"
    ev.load_ebsd_file(files["scan_b"])
    ev.load_ebsd_file(files["scan_a"])
    entry = store.get_entry("scan_a")
    assert entry.pc_source == "propagated"                              # not "missing" again
    np.testing.assert_allclose(store.get_pc("scan_a"), TYPED, atol=1e-12)


# --- the load response names the store's state ----------------------------------

def test_the_response_after_a_restore_reports_the_typed_pc_not_the_placeholder(viewer):
    ev, store, files, _ = viewer
    first = _load(files["scan_a"])
    assert first["pc_defaulted"] is True and first["pc_source"] == "default" and first["pc"] is None
    store.update_pc("scan_a", TYPED, source="manual")
    _load(files["scan_b"])
    back = _load(files["scan_a"])                                       # switch back
    assert back["pc_defaulted"] is False                                # no "indexing uses the default PC" warning
    assert back["pc_source"] == "manual" and back["pc_store_source"] == "manual"
    np.testing.assert_allclose(back["pc"], TYPED, atol=1e-12)
    assert back["pc_notice"] == "user_pc_kept"
    other = _load(files["scan_b"])                                      # said once: it does not follow to the next file
    assert other["pc_notice"] is None and other["pc_defaulted"] is True


def test_an_osc_put_beside_the_file_replaces_the_restored_typed_pc(viewer):
    from safe_loader import load_ebsd_safe
    ev, store, files, tmp = viewer
    _load(files["scan_a"])
    store.update_pc("scan_a", TYPED, source="manual")
    _load(files["scan_b"])
    _write_osc(tmp / "scan_a.osc", OSC_TSL)                             # what the error message advises
    back = _load(files["scan_a"])
    entry = store.get_entry("scan_a")
    assert entry.pc_source == "header"                                  # the file's PC, not the typed one
    file_pc = np.asarray(load_ebsd_safe(files["scan_a"], verbose=False).detector.pc).ravel()[:3]
    np.testing.assert_allclose(store.get_pc("scan_a"), file_pc, atol=1e-9)
    assert not np.allclose(store.get_pc("scan_a"), TYPED, atol=1e-3)
    assert back["pc_notice"] == "file_pc_wins" and back["pc_source"] == "osc"
    np.testing.assert_allclose(back["pc_discarded"], TYPED, atol=1e-12)   # the typed PC, so it can be typed again
    np.testing.assert_allclose(back["pc"], file_pc, atol=1e-9)
    assert back["pc_defaulted"] is False


def test_a_file_that_did_not_change_keeps_the_pc_the_user_set(viewer):
    ev, store, files, tmp = viewer
    _write_osc(tmp / "scan_a.osc", OSC_TSL)
    first = _load(files["scan_a"])
    assert first["pc_source"] == "osc"
    store.update_pc("scan_a", TYPED, source="manual")                   # the user overrides the header
    _load(files["scan_b"])
    back = _load(files["scan_a"])
    assert back["pc_notice"] == "user_pc_kept" and back["pc_source"] == "manual"
    np.testing.assert_allclose(store.get_pc("scan_a"), TYPED, atol=1e-12)


def test_a_file_whose_own_pc_changed_takes_precedence(viewer):
    ev, store, files, tmp = viewer
    _write_osc(tmp / "scan_a.osc", OSC_TSL)
    _load(files["scan_a"])
    store.update_pc("scan_a", TYPED, source="manual")
    _load(files["scan_b"])
    _write_osc(tmp / "scan_a.osc", OSC_TSL_2)                           # the file says something else now
    back = _load(files["scan_a"])
    assert back["pc_notice"] == "file_pc_wins"
    assert store.get_entry("scan_a").pc_source == "header"


def test_the_notice_is_read_once(viewer):
    ev, store, files, _ = viewer
    _load(files["scan_a"])
    store.update_pc("scan_a", TYPED, source="manual")
    _load(files["scan_b"])
    _load(files["scan_a"])                                              # the response consumed the notice
    assert ev._pc_state_for_response("scan_a", "default")["pc_notice"] is None


# --- an Orienta export whose /Detector/pc differs from its header ---------------

def test_the_response_for_an_export_with_a_differing_detector_pc_names_both(viewer):
    import h5py
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    from backend.api.routes.indexing import _write_fresh_edax_h5
    ev, store, files, tmp = viewer
    shape, header_pc, typed = (60, 40), (0.5241, 0.4133, 0.6317), [0.51, 0.33, 0.84]
    rng = np.random.default_rng(3)
    sig = kp.signals.EBSD(rng.integers(0, 255, (3, 4) + shape).astype(np.uint8))
    sig.detector = EBSDDetector(shape=shape, pc=header_pc, sample_tilt=70.0)
    out = str(tmp / "export.h5")
    _write_fresh_edax_h5(out, sig, pc_bruker=header_pc)
    with h5py.File(out, "a") as f:
        f.create_group("Detector").create_dataset("pc", data=np.array(typed))
    resp = _load(out)
    assert resp["pc_notice"] == "export_pc_ignored"
    np.testing.assert_allclose(resp["pc_export_ignored"], typed, atol=1e-9)
    np.testing.assert_allclose(resp["pc"], header_pc, atol=1e-6)          # the header is what is used
    np.testing.assert_allclose(store.get_pc("export"), header_pc, atol=1e-6)
