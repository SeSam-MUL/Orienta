"""The PC Refinement page and the calibration store say the same, and two files with one name stay apart.

The page works on a controller per dataset name; the store is what indexing and exports
read. After a load the store can hold another PC than the controller was built with (a
restored typed PC, a header PC that now wins because an .osc was put beside the file, or
none any more), and a session kept by NAME served a second file of the same stem (another
folder) with the first one's calibration patterns and PC, a refine on which was then written
to the second file's entry. Real UP1 files (v1 layout, synthetic, see test_pc_restore_policy),
real routes, real controllers.
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


def _write_up1(path, n_side=4, seed=3):
    rng = np.random.default_rng(seed)
    pats = rng.integers(0, 250, (n_side * n_side, SIDE, SIDE)).astype("u1")
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, SIDE, SIDE, 16))
        f.write(pats.tobytes())
    return str(path)


def _write_osc(path, pc_tsl, n_side=4):
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
    from backend.api.routes import pcrefinement as R
    store = CalibrationStore()
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(I, "calibration_store", store)
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file", "_pc_load_notice"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_loaded_files", [])
    monkeypatch.setattr(ev, "_active_dataset", "")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    monkeypatch.setattr(R, "_sessions", {})
    monkeypatch.setattr(R, "_parked_sessions", {})
    monkeypatch.setattr(R, "_pc_before", {})
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    return ev, R, store, tmp_path


def _load(path):
    resp = TestClient(app).post("/api/ebsd/load", json={"path": path})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _status():
    resp = TestClient(app).get("/api/pc/status")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _typed_on_page(R, store, name):
    """What the page does when the user types a PC: controller and store both written."""
    from backend.api.routes.pcrefinement import _writeback_refined_pc
    ctrl = R._get_controller()
    ctrl.set_global_pc(TYPED)
    _writeback_refined_pc(name, TYPED, store=store, source="manual")


# --- the page follows the store after a load ---------------------------------------

def test_an_osc_beside_a_typed_file_is_what_the_page_shows_after_the_reload(viewer):
    ev, R, store, tmp = viewer
    path = _write_up1(tmp / "a" / "scan.up1")
    _load(path)
    _status()                                                            # the page opens: session built
    _typed_on_page(R, store, "scan")
    np.testing.assert_allclose(_status()["pc"], TYPED, atol=1e-9)
    _write_osc(tmp / "a" / "scan.osc", OSC_TSL)                          # what the error message advises
    resp = _load(path)
    assert resp["pc_notice"] == "file_pc_wins"
    np.testing.assert_allclose(resp["pc_discarded"], TYPED, atol=1e-9)   # the typed PC, to type again
    st = _status()
    np.testing.assert_allclose(st["pc"], store.get_pc("scan"), atol=1e-9)    # page == store
    assert st["pc_source"] == store.get_entry("scan").pc_source == "header"
    assert not np.allclose(st["pc"], TYPED, atol=1e-3)


def test_the_page_follows_the_store_when_the_osc_is_gone_again(viewer):
    ev, R, store, tmp = viewer
    path = _write_up1(tmp / "a" / "scan.up1")
    _write_osc(tmp / "a" / "scan.osc", OSC_TSL)
    _load(path)
    _status()
    (tmp / "a" / "scan.osc").unlink()
    _load(path)
    st = _status()
    assert store.get_entry("scan").pc_source == "missing"
    assert st["pc_source"] == "missing"
    np.testing.assert_allclose(st["pc"], [0.5, 0.5, 0.5], atol=1e-9)     # what the hint says the fields show


def test_a_restored_typed_pc_is_what_the_page_shows_after_a_switch_back(viewer):
    ev, R, store, tmp = viewer
    a = _write_up1(tmp / "a" / "scan_a.up1")
    b = _write_up1(tmp / "b" / "scan_b.up1")
    _load(a)
    _status()
    _typed_on_page(R, store, "scan_a")
    _load(b)
    _load(a)
    np.testing.assert_allclose(_status()["pc"], TYPED, atol=1e-9)
    np.testing.assert_allclose(np.asarray(R._get_controller().detector.pc).ravel(),
                               store.get_pc("scan_a"), atol=1e-12)


# --- two files with the same name ---------------------------------------------------

def test_two_files_with_the_same_stem_do_not_share_a_session(viewer):
    ev, R, store, tmp = viewer
    a = _write_up1(tmp / "a" / "Scan5.up1", seed=1)
    b = _write_up1(tmp / "b" / "Scan5.up1", seed=2)
    _write_osc(tmp / "b" / "Scan5.osc", OSC_TSL)
    _load(a)
    _status()
    ctrl_a = R._get_controller()
    _typed_on_page(R, store, "Scan5")
    ctrl_a.add_pattern((0, 0), np.zeros((SIDE, SIDE), np.uint8))
    ctrl_a.add_pattern((1, 1), np.zeros((SIDE, SIDE), np.uint8))
    _load(b)
    assert R._get_controller() is not ctrl_a
    st = _status()
    assert st["patterns"] == []                                          # A's calibration patterns are not B's
    assert st["pc_source"] == "header"
    np.testing.assert_allclose(st["pc"], store.get_pc("Scan5"), atol=1e-9)
    assert not np.allclose(st["pc"], TYPED, atol=1e-3)
    _load(a)                                                             # and A gets its own session back
    assert R._get_controller() is ctrl_a
    assert len(_status()["patterns"]) == 2
    np.testing.assert_allclose(_status()["pc"], TYPED, atol=1e-9)


def test_the_batch_loader_brings_the_page_in_line_with_the_store_too(viewer):
    ev, R, store, tmp = viewer
    path = _write_up1(tmp / "a" / "scan.up1")
    _load(path)
    _status()
    _typed_on_page(R, store, "scan")
    _write_osc(tmp / "a" / "scan.osc", OSC_TSL)
    assert ev.load_ebsd_file(path) is True                              # what /api/indexing/batch/start calls
    np.testing.assert_allclose(_status()["pc"], store.get_pc("scan"), atol=1e-9)
    assert not np.allclose(_status()["pc"], TYPED, atol=1e-3)


# --- a session that was parked while another file had its name ----------------------

def test_a_parked_session_follows_the_store_when_its_file_comes_back_changed(viewer):
    """A (typed PC) -> B with the same stem (A's session is parked) -> A again after an
    .osc was put beside it: the store changed while the session was away, and the page
    must show the store's PC, not the typed one it kept."""
    ev, R, store, tmp = viewer
    a = _write_up1(tmp / "a" / "scan.up1", seed=1)
    b = _write_up1(tmp / "b" / "scan.up1", seed=2)
    _load(a)
    _status()
    _typed_on_page(R, store, "scan")
    _load(b)                                                            # parks A's session
    assert R._parked_sessions, "the setup must have parked a session"
    _write_osc(tmp / "a" / "scan.osc", OSC_TSL)                         # while A is away
    resp = _load(a)
    assert resp["pc_notice"] == "file_pc_wins"
    st = _status()
    np.testing.assert_allclose(st["pc"], store.get_pc("scan"), atol=1e-9)
    assert not np.allclose(st["pc"], TYPED, atol=1e-3)
    assert st["pc_source"] == "header"


def test_a_parked_session_follows_the_store_when_the_osc_is_gone(viewer):
    ev, R, store, tmp = viewer
    a = _write_up1(tmp / "a" / "scan.up1", seed=1)
    _write_osc(tmp / "a" / "scan.osc", OSC_TSL)
    b = _write_up1(tmp / "b" / "scan.up1", seed=2)
    _load(a)
    _status()
    _load(b)
    (tmp / "a" / "scan.osc").unlink()                                   # while A is away
    _load(a)
    st = _status()
    assert store.get_entry("scan").pc_source == "missing" and st["pc_source"] == "missing"
    np.testing.assert_allclose(st["pc"], [0.5, 0.5, 0.5], atol=1e-9)


# --- one step back ends with a load ------------------------------------------------

def _put_back(name):
    return TestClient(app).post("/api/pc/put-back", json={"dataset": name})


def test_a_step_does_not_survive_a_reload_that_changed_the_files_pc(viewer):
    """A file without an .osc, a typed PC (a step: "missing" before), another file, an .osc put beside the
    first, a reload: the store holds the .osc PC. Putting the step back would set "missing" and the
    placeholder while the file's own PC is the .osc one."""
    ev, R, store, tmp = viewer
    a = _write_up1(tmp / "a" / "Scan5.up1", seed=1)
    other = _write_up1(tmp / "b" / "Other.up1", seed=2)
    _load(a)
    _status()
    assert store.get_entry("Scan5").pc_source == "missing"
    _typed_on_page(R, store, "Scan5")
    assert R._previous_of("Scan5") == {"pc": None, "had_pc": False}              # the step exists
    _load(other)
    _write_osc(tmp / "a" / "Scan5.osc", OSC_TSL)
    assert _load(a)["pc_notice"] == "file_pc_wins"
    entry = store.get_entry("Scan5")
    assert entry.pc_source == "header"
    np.testing.assert_allclose(store.get_pc("Scan5"), entry.file_pc, atol=1e-12)
    assert R._pc_before == {}
    resp = _put_back("Scan5")
    assert resp.status_code == 409 and resp.headers.get("X-Orienta-Code") == "nothingToPutBack"
    assert TestClient(app).get("/api/pc/previous", params={"dataset": "Scan5"}).json()["previous"] is None
    assert store.get_entry("Scan5").pc_source == "header"                        # still the .osc PC
    np.testing.assert_allclose(_status()["pc"], store.get_pc("Scan5"), atol=1e-9)


def test_a_to_b_to_a_has_nothing_to_put_back(viewer):
    ev, R, store, tmp = viewer
    a = _write_up1(tmp / "a" / "scan_a.up1", seed=1)
    b = _write_up1(tmp / "b" / "scan_b.up1", seed=2)
    _load(a)
    _status()
    _typed_on_page(R, store, "scan_a")
    _load(b)
    _load(a)
    np.testing.assert_allclose(store.get_pc("scan_a"), TYPED, atol=1e-9)         # the typed PC was restored
    resp = _put_back("scan_a")
    assert resp.status_code == 409 and resp.headers.get("X-Orienta-Code") == "nothingToPutBack"
    np.testing.assert_allclose(store.get_pc("scan_a"), TYPED, atol=1e-9)


def test_a_switch_between_the_datasets_of_one_file_keeps_the_step(viewer):
    """A copy of the dataset and a switch to it and back are no load: the step stays, and the copy
    made after the write is the one the answer says stays as it is."""
    ev, R, store, tmp = viewer
    path = _write_up1(tmp / "a" / "scan.up1")
    _load(path)
    _status()
    _typed_on_page(R, store, "scan")
    client = TestClient(app)
    copy = client.post("/api/ebsd/deepcopy", json={"name": "late"})
    assert copy.status_code == 200, copy.text
    assert client.post("/api/ebsd/select-dataset", json={"name": "late"}).status_code == 200
    assert client.post("/api/ebsd/select-dataset", json={"name": "scan"}).status_code == 200
    assert R._previous_of("scan") == {"pc": None, "had_pc": False}
    resp = _put_back("scan")
    assert resp.status_code == 200, resp.text
    assert resp.json()["restored"] == ["scan"] and resp.json()["not_restored"] == ["late"]
    assert store.get_entry("scan").pc_source == "missing"
    np.testing.assert_allclose(store.get_pc("late"), TYPED, atol=1e-9)
