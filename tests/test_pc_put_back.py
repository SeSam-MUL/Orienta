"""One step back: "Put it back" restores the PC a PC write replaced, "previous" says what it was.

Real calibration store, real routes (``TestClient(app)`` without ``with``), real controllers; only the
PC search itself is the stand-in of tests/pc_engine_stub.py. Every expected number is chosen here.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from kikuchipy.detectors import EBSDDetector

from backend.api.main import app
from backend.api.services import pc_calibration_apply as use
from backend.api.services import pc_calibrations as cal
from backend.api.services import pc_guard
from backend.api.services.calibration_store import CalibrationStore
from pc_controller import PCController

SHAPE = (60, 60)
HEADER = [0.5, 0.4, 0.6]
FOUND = [0.4301, 0.2141, 0.5009]
TYPED = [0.45, 0.35, 0.55]
FIVE = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
REFINEMENT = {"n_patterns": 5, "method": "PSO"}
#: a 2 x 2 map whose mean is HEADER: PCx 0.49 / 0.51 in a checker, PCy and PCz constant
MAP = np.array([[[0.49, 0.4, 0.6], [0.51, 0.4, 0.6]],
                [[0.49, 0.4, 0.6], [0.51, 0.4, 0.6]]])


def _cif(tmp_path):
    cif = tmp_path / "Ni.cif"
    cif.write_text(
        "data_Ni\n_cell_length_a 3.52\n_cell_length_b 3.52\n_cell_length_c 3.52\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'F m -3 m'\n_symmetry_Int_Tables_number 225\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\nNi1 Ni 0 0 0\n", encoding="utf-8")
    return str(cif)


class _Signal:
    def __init__(self):
        self.data = np.zeros((6, 6) + SHAPE, dtype=np.uint8)
        self.axes_manager = SimpleNamespace(navigation_shape=(6, 6))


def _kp_signal():
    import kikuchipy as kp
    det = EBSDDetector(shape=SHAPE, pc=tuple(HEADER), sample_tilt=70.0)
    return kp.signals.EBSD(np.zeros((2, 2) + SHAPE, np.uint8), detector=det)


class World:
    """A calibration store, the PC page's sessions, and the active dataset, wired into the routes."""

    def __init__(self, R, ev, store, tmp_path):
        self.R, self.ev, self.store, self.tmp = R, ev, store, tmp_path

    def add_file(self, name, pc=HEADER, missing=False, pc_map=None, file_key=None, page_pc=None):
        """A file loaded as ``name``: its store entry (as a load registers it) and its PC page."""
        tag = "default" if missing else None
        if missing:
            pc = [0.5, 0.5, 0.5]                    # what a loader without a PC leaves: kikuchipy's default
        signal = SimpleNamespace(
            detector=EBSDDetector(shape=SHAPE, pc=pc if pc_map is None else pc_map, sample_tilt=70.0),
            metadata=SimpleNamespace(get_item=lambda key, default=None: tag
                                     if key == "Signal.pc_source" and tag else default))
        self.store.register(name, signal, file_key=file_key if file_key is not None else f"key:{name}")
        self.page(name, page_pc if page_pc is not None else ([0.5, 0.5, 0.5] if missing else pc))

    def add_copy(self, name, parent):
        self.store.register_derived(name, parent)
        self.page(name, self.store.get_entry(name).pc_single)

    def page(self, name, pc):
        ctrl = PCController()
        ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=tuple(float(v) for v in pc), sample_tilt=70.0))
        self.R._sessions[name] = self.R._PCSession(ctrl=ctrl, file_key="")
        return ctrl

    def ctrl_pc(self, name):
        return np.asarray(self.R._sessions[name].ctrl.detector.pc, dtype=float).reshape(-1, 3)


@pytest.fixture
def world(monkeypatch, tmp_path):
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.routes import pcrefinement as R
    store = CalibrationStore()
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    use.forget_all()
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    monkeypatch.setattr(ev, "calibration_store", store)
    monkeypatch.setattr(ev, "_active_dataset", "ds")
    monkeypatch.setattr(ev, "_ebsd_file_path", None)
    monkeypatch.setattr(ev, "_get_active_signal", lambda *a, **k: _Signal())
    monkeypatch.setattr(R, "_optimization_active", False)
    monkeypatch.setattr(R, "_sessions", {})
    monkeypatch.setattr(R, "_parked_sessions", {})
    monkeypatch.setattr(R, "_pc_before", {})
    n_tasks = len(R._optimization_tasks)
    yield World(R, ev, store, tmp_path)
    R._optimization_active = False
    while len(R._optimization_tasks) > n_tasks:
        R._optimization_tasks.popitem()
    use.forget_all()
    cal.set_calibrations_dir_for_test(None)


def _client():
    return TestClient(app)


def _put_back(body=None):
    return _client().post("/api/pc/put-back", json=body if body is not None else {})


def _write(w, name, pc, **kw):
    kw.setdefault("source", "refined")
    return w.R._writeback_refined_pc(name, pc, store=w.store, **kw)


def _code(resp):
    return resp.headers.get("X-Orienta-Code")


# ------------------------------------------------------------------ the store

def test_restore_pc_sets_only_the_pc_fields_and_skips_entries_that_are_gone_or_of_another_file(world):
    w = world
    w.add_file("a", file_key="fileA")
    w.add_file("b", file_key="fileB")
    w.add_file("c", file_key="fileC")
    before = w.store.snapshot(["a", "b", "c"])
    for name in ("a", "b", "c"):
        w.store.update_pc_detail(name, FOUND, source="refined", refinement=REFINEMENT)
        w.store.update_geometry(name, sample_tilt=65.0, tilt=3.0, azimuthal=1.5)
    w.store.remove("c")                                                       # gone
    w.store.register("b", SimpleNamespace(detector=EBSDDetector(shape=SHAPE, pc=TYPED)), file_key="other")
    assert w.store.restore_pc(before) == ["a"]
    a = w.store.get_entry("a")
    np.testing.assert_allclose(a.pc_single, HEADER, atol=1e-12)
    assert a.pc_source == "header" and a.pc_refinement is None and a.pc_map is None
    assert (a.sample_tilt, a.tilt, a.azimuthal) == (65.0, 3.0, 1.5)           # tilts stay
    np.testing.assert_allclose(a.file_pc, HEADER, atol=1e-12)                 # file_pc stays
    np.testing.assert_allclose(w.store.get_pc("b"), TYPED, atol=1e-12)        # another file's entry: untouched
    assert "c" not in w.store


def test_restore_pc_does_not_alias_the_snapshot(world):
    w = world
    w.add_file("a", pc_map=MAP)
    snap = w.store.snapshot(["a"])
    w.store.restore_pc(snap)
    w.store.get_entry("a").pc_map[0, 0, 0] = 9.0
    assert snap["a"].pc_map[0, 0, 0] == pytest.approx(0.49)


# ------------------------------------------------------------------ put it back

def test_a_found_pc_on_a_file_with_a_header_pc_goes_back_to_the_header_pc(world):
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND, refinement=REFINEMENT)
    entry = w.store.get_entry("ds")
    assert entry.pc_source == "refined" and entry.pc_refinement == REFINEMENT
    resp = _put_back()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"dataset": "ds", "pc": HEADER, "had_pc": True, "restored": ["ds"],
                    "not_restored": [], "pc_per_pixel": False}
    np.testing.assert_allclose(w.store.get_pc("ds"), HEADER, atol=1e-12)
    assert entry.pc_source == "header" and entry.pc_refinement is None
    assert w.R._pc_before == {}                                               # the step is used up


def test_the_map_the_file_had_comes_back(world):
    w = world
    w.add_file("ds", pc_map=MAP)
    _write(w, "ds", FOUND, refinement=REFINEMENT, on_refined_map="replace")   # a header map is shifted
    shifted = w.store.get_entry("ds").pc_map
    np.testing.assert_allclose(shifted[..., 0], [[0.4201, 0.4401], [0.4201, 0.4401]], atol=1e-12)
    body = _put_back().json()
    assert body["pc_per_pixel"] is True and body["pc"] == pytest.approx(HEADER)
    assert "pc_map_action" not in body                                        # "replaced" was a wrong word for a restore
    entry = w.store.get_entry("ds")
    np.testing.assert_allclose(entry.pc_map, MAP, atol=1e-12)
    np.testing.assert_allclose(entry.pc_single, HEADER, atol=1e-12)
    assert entry.pc_source == "header" and entry.pc_refinement is None


def test_a_grid_calibrated_map_that_a_global_pc_dropped_comes_back(world):
    w = world
    w.add_file("ds")
    w.store.update_pc_map("ds", MAP, refinement=REFINEMENT)                   # source "refined_map"
    _write(w, "ds", FOUND, on_refined_map="replace")
    assert w.store.get_entry("ds").pc_map is None
    body = _put_back().json()
    assert body["pc_per_pixel"] is True
    entry = w.store.get_entry("ds")
    np.testing.assert_allclose(entry.pc_map, MAP, atol=1e-12)
    assert entry.pc_source == "refined_map" and entry.pc_refinement == REFINEMENT


def test_a_file_without_a_pc_goes_back_to_missing_and_the_guard_refuses_again(world):
    w = world
    w.add_file("ds", missing=True)
    assert pc_guard.is_pc_missing("ds", w.store) is True
    _write(w, "ds", FOUND, refinement=REFINEMENT)
    assert pc_guard.is_pc_missing("ds", w.store) is False
    assert w.R._previous_of("ds") == {"pc": None, "had_pc": False}
    body = _put_back().json()
    assert body == {"dataset": "ds", "pc": None, "had_pc": False, "restored": ["ds"],
                    "not_restored": [], "pc_per_pixel": False}
    entry = w.store.get_entry("ds")
    assert entry.pc_source == "missing"
    np.testing.assert_allclose(entry.pc_single, [0.5, 0.5, 0.5], atol=1e-12)    # the placeholder
    assert w.store.get_pc("ds") is None
    assert pc_guard.is_pc_missing("ds", w.store) is True
    with pytest.raises(Exception) as exc:
        pc_guard.require_pc("ds", w.store)
    assert exc.value.status_code == 400 and exc.value.headers["X-Orienta-Code"] == "pc_missing"


def test_the_parent_and_the_copies_that_followed_go_back_too(world):
    w = world
    w.add_file("raw")
    w.add_copy("raw_a", "raw")
    w.add_copy("raw_b", "raw")                                                # a sibling that never had a PC of its own
    w.add_copy("raw_own", "raw")
    w.store.update_pc_detail("raw_own", TYPED, source="manual")               # one that did
    w.ev._active_dataset = "raw_a"
    _write(w, "raw_a", FOUND, source="manual", on_refined_map="shift")
    for name in ("raw", "raw_a", "raw_b"):
        np.testing.assert_allclose(w.store.get_pc(name), FOUND, atol=1e-12)
    body = _put_back().json()
    assert sorted(body["restored"]) == ["raw", "raw_a", "raw_b"]              # raw_own was not touched
    assert body["not_restored"] == []                                         # raw_own has a PC of its own, no copy is late
    np.testing.assert_allclose(w.store.get_pc("raw"), HEADER, atol=1e-12)
    assert w.store.get_entry("raw").pc_source == "header"
    for name in ("raw_a", "raw_b"):
        np.testing.assert_allclose(w.store.get_pc(name), HEADER, atol=1e-12)
        assert w.store.get_entry(name).pc_source == "inherited"
    np.testing.assert_allclose(w.store.get_pc("raw_own"), TYPED, atol=1e-12)
    assert w.store.get_entry("raw_own").pc_source == "manual"


def test_copies_and_crops_made_after_the_write_keep_the_pc_and_are_listed(world):
    """A copy made after the write took the new PC from its parent and is in no step: the root goes
    back, the copy does not. The answer names them (``not_restored``)."""
    from backend.api.services.crop_window import CropWindow
    w = world
    w.add_file("raw")
    w.add_copy("early", "raw")                                                # made before the write: in the step
    w.ev._active_dataset = "raw"
    _write(w, "raw", FOUND)
    w.add_copy("late_copy", "raw")
    w.add_copy("late_of_copy", "late_copy")                                   # a copy of a late copy
    w.store.register_cropped("late_crop", "raw", CropWindow("raw.up1", 0, 0, 1, 1, (2, 2)))
    w.add_copy("late_own", "raw")
    w.store.update_pc_detail("late_own", TYPED, source="manual")              # a late copy with a PC of its own
    late = ("late_copy", "late_of_copy", "late_crop")
    for name in late:
        np.testing.assert_allclose(w.store.get_pc(name), FOUND, atol=1e-12)
        assert w.store.get_entry(name).pc_source == "inherited"
    body = _put_back({"dataset": "raw"}).json()
    assert sorted(body["restored"]) == ["early", "raw"]
    assert body["not_restored"] == ["late_copy", "late_crop", "late_of_copy"]
    np.testing.assert_allclose(w.store.get_pc("raw"), HEADER, atol=1e-12)
    np.testing.assert_allclose(w.store.get_pc("early"), HEADER, atol=1e-12)
    for name in late:                                                         # unchanged: not touched by the put-back
        np.testing.assert_allclose(w.store.get_pc(name), FOUND, atol=1e-12)
        assert w.store.get_entry(name).pc_source == "inherited"
    np.testing.assert_allclose(w.store.get_pc("late_own"), TYPED, atol=1e-12)


def test_an_inherited_copy_of_another_family_is_not_listed(world):
    w = world
    w.add_file("raw")
    _write(w, "raw", FOUND)
    w.add_file("other")
    w.add_copy("other_copy", "other")                                         # inherited, but another family
    assert _put_back({"dataset": "raw"}).json()["not_restored"] == []


def test_a_copy_found_only_after_the_first_writes_is_in_the_step_too(world):
    """``y`` follows only because its parent ``p`` (typed, so on its own before) was just made "propagated"
    by this very write. A step taken before the write would not know ``y``."""
    w = world
    w.add_file("raw")
    w.add_copy("p", "raw")
    w.store.update_pc_detail("p", TYPED, source="manual")
    w.add_copy("x", "p")
    w.add_copy("y", "p")
    w.ev._active_dataset = "x"
    _write(w, "x", FOUND, source="manual")
    np.testing.assert_allclose(w.store.get_pc("y"), FOUND, atol=1e-12)         # it followed
    body = _put_back().json()
    assert sorted(body["restored"]) == ["p", "raw", "x", "y"]
    np.testing.assert_allclose(w.store.get_pc("y"), TYPED, atol=1e-12)         # p's own PC, which y had inherited
    np.testing.assert_allclose(w.store.get_pc("p"), TYPED, atol=1e-12)
    assert w.store.get_entry("p").pc_source == "manual" and w.store.get_entry("y").pc_source == "inherited"
    np.testing.assert_allclose(w.store.get_pc("raw"), HEADER, atol=1e-12)


def test_the_page_session_follows_the_store_with_an_empty_simulation_cache(world):
    w = world
    w.add_file("raw")
    w.add_copy("ds", "raw")
    resp = _client().post("/api/pc/detector/update-pc", json={"pcx": FOUND[0], "pcy": FOUND[1], "pcz": FOUND[2]})
    assert resp.status_code == 200, resp.text
    np.testing.assert_allclose(w.ctrl_pc("ds"), [FOUND], atol=1e-12)
    np.testing.assert_allclose(w.ctrl_pc("raw"), [FOUND], atol=1e-12)           # the parent's page followed the write
    for name in ("ds", "raw"):
        w.R._sessions[name].sim_cache.update({3: "lines at the old PC"})
        w.R._sessions[name].ctrl.cache[3] = "result at the old PC"
        w.R._sessions[name].ctrl.indexer = object()
    assert _put_back().status_code == 200
    for name in ("ds", "raw"):
        np.testing.assert_allclose(w.ctrl_pc(name), [HEADER], atol=1e-12)
        assert w.R._sessions[name].sim_cache == {}
        assert w.R._sessions[name].ctrl.cache == {} and w.R._sessions[name].ctrl.indexer is None


def test_the_caches_of_a_page_that_already_shows_the_restored_pc_are_emptied_too(world):
    """A page the write never reached holds the header PC already; its cached results are still
    the ones computed under the PC that was put back from."""
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND)                                                    # the store only, not the page
    np.testing.assert_allclose(w.ctrl_pc("ds"), [HEADER], atol=1e-12)
    w.R._sessions["ds"].sim_cache[1] = "lines"
    w.R._sessions["ds"].ctrl.cache[1] = "result"
    _put_back()
    assert w.R._sessions["ds"].sim_cache == {} and w.R._sessions["ds"].ctrl.cache == {}


def test_what_a_map_served_before_is_its_mean_not_the_single_pc_kept_beside_it(world):
    w = world
    w.add_file("ds")
    w.store.update_pc_map("ds", MAP)                                          # a grid-calibrated map
    _write(w, "ds", FOUND, on_refined_map="keep")                             # the map stays, pc_single moves
    entry = w.store.get_entry("ds")
    np.testing.assert_allclose(entry.pc_single, FOUND, atol=1e-12)
    _write(w, "ds", TYPED, source="manual", on_refined_map="keep")
    assert w.R._previous_of("ds") == {"pc": pytest.approx(HEADER), "had_pc": True}   # the map's mean


def test_a_page_on_a_per_pixel_map_gets_the_map_back(world):
    w = world
    w.add_file("ds", pc_map=MAP)
    det = EBSDDetector(shape=SHAPE, pc=MAP, sample_tilt=70.0)
    w.R._sessions["ds"].ctrl.attach_detector(det)
    _write(w, "ds", FOUND, on_refined_map="replace")
    w.R._session_takes_pc("ds", FOUND, "shifted")
    assert _put_back().status_code == 200
    np.testing.assert_allclose(w.ctrl_pc("ds").reshape(2, 2, 3), MAP, atol=1e-12)


def test_a_second_put_back_is_refused(world):
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND)
    assert _put_back().status_code == 200
    again = _put_back()
    assert again.status_code == 409 and _code(again) == "nothingToPutBack"
    np.testing.assert_allclose(w.store.get_pc("ds"), HEADER, atol=1e-12)


def test_nothing_stored_is_refused(world):
    w = world
    w.add_file("ds")
    resp = _put_back()
    assert resp.status_code == 409 and _code(resp) == "nothingToPutBack"
    assert _client().get("/api/pc/previous").json() == {"dataset": "ds", "previous": None}


def test_a_new_write_replaces_the_step_there_is_only_one(world):
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND)
    _write(w, "ds", TYPED, source="manual")
    body = _put_back().json()
    np.testing.assert_allclose(w.store.get_pc("ds"), FOUND, atol=1e-12)        # back one step, not two
    assert w.store.get_entry("ds").pc_source == "refined" and body["pc"] == pytest.approx(FOUND)
    assert _put_back().status_code == 409


def test_a_step_does_not_reach_another_file_with_the_same_dataset_name(world):
    w = world
    w.add_file("scan", file_key="folderA/scan.up1")
    _write(w, "scan", FOUND)
    # another file with the same stem is loaded: the store entry under the name is now its own
    w.add_file("scan", pc=TYPED, file_key="folderB/scan.up1")
    w.ev._active_dataset = "scan"
    resp = _put_back()
    assert resp.status_code == 409 and _code(resp) == "nothingToPutBack"
    assert _client().get("/api/pc/previous").json()["previous"] is None
    np.testing.assert_allclose(w.store.get_pc("scan"), TYPED, atol=1e-12)
    # and a write on the second file makes its own step, which does work
    _write(w, "scan", FOUND)
    assert _put_back().json()["pc"] == pytest.approx(TYPED)


def test_a_dataset_can_be_named_and_the_active_one_is_the_default(world):
    w = world
    w.add_file("ds")
    w.add_file("other")
    _write(w, "other", FOUND)
    assert _put_back().status_code == 409                                      # the active "ds" has no step
    body = _put_back({"dataset": "other"}).json()
    assert body["dataset"] == "other" and body["pc"] == pytest.approx(HEADER)
    assert _put_back(None).status_code == 409
    assert _client().post("/api/pc/put-back").status_code == 409               # no body at all is the default too


def test_the_put_back_writes_nothing_to_the_file_on_disk(world):
    w = world
    path = w.tmp / "ds.h5oina"
    path.write_bytes(b"patterns" * 100)
    w.add_file("ds", file_key=str(path))
    stamp = (path.stat().st_size, path.stat().st_mtime_ns)
    _write(w, "ds", FOUND)
    _put_back()
    assert (path.stat().st_size, path.stat().st_mtime_ns) == stamp
    assert sorted(p.name for p in w.tmp.iterdir() if p.is_file()) == ["ds.h5oina"]


# ------------------------------------------------------------------ refusals while a job runs

def test_refused_while_a_search_runs_and_nothing_changes(world):
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND)
    w.R._optimization_active = True
    resp = _put_back()
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "pc_job_running"
    np.testing.assert_allclose(w.store.get_pc("ds"), FOUND, atol=1e-12)
    assert "ds" in w.R._pc_before.get("ds", {})                                # the step is still there
    w.R._optimization_active = False
    assert _put_back().status_code == 200


def test_refused_while_a_search_runs_without_waiting_for_the_lock(world, monkeypatch):
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND)
    w.R._optimization_active = True

    def must_not_be_asked(what):
        raise AssertionError("the request waited for the state lock")

    monkeypatch.setattr(w.R, "_state_lock_or_busy", must_not_be_asked)
    resp = _put_back()
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "pc_job_running"


def test_refused_a_second_time_inside_the_lock_when_a_search_started_meanwhile(world, monkeypatch):
    """The first check passes; a search starts while the request waits for the lock."""
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND)
    import contextlib
    real = w.R._state_lock_or_busy

    @contextlib.contextmanager
    def lock_then_search_starts(what):
        with real(what):
            w.R._optimization_active = True
            yield

    monkeypatch.setattr(w.R, "_state_lock_or_busy", lock_then_search_starts)
    resp = _put_back()
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "pc_job_running"
    np.testing.assert_allclose(w.store.get_pc("ds"), FOUND, atol=1e-12)
    assert w.R._pc_before


def test_refused_as_busy_while_another_job_holds_the_state_lock(world, monkeypatch):
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND)
    monkeypatch.setattr(w.R, "_BUSY_WAIT_S", 0.05)
    held = threading.Event()
    release = threading.Event()

    def hold():
        with w.R._state_lock:
            held.set()
            release.wait(10)

    t = threading.Thread(target=hold)
    t.start()
    try:
        assert held.wait(5)
        resp = _put_back()
    finally:
        release.set()
        t.join()
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "pc_job_running"
    np.testing.assert_allclose(w.store.get_pc("ds"), FOUND, atol=1e-12)
    assert _put_back().status_code == 200                                      # and works once the lock is free


# ------------------------------------------------------------------ the routes that write a PC

def _start(w, positions, **req):
    """Run Global PC Refine / Find PC through the route and its background job."""
    R = w.R
    ctrl = R._sessions["ds"].ctrl
    for (r, c) in positions:
        ctrl.add_pattern((r, c), np.full(SHAPE, r * 10 + c, np.uint8))
    bt = BackgroundTasks()
    flat = [r * 6 + c for r, c in positions]
    out = asyncio.run(R.optimize_pc(R.OptimizePCRequest(patterns=flat, **req), bt))
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    return R._optimization_tasks[out["task_id"]]


@pytest.fixture
def search(world, monkeypatch):
    from tests.pc_engine_stub import stub_search
    w = world
    w.add_file("ds")
    w.R._sessions["ds"].ctrl.load_phase(_cif(w.tmp))
    monkeypatch.setattr(w.R, "_index_and_simulate",
                        lambda *a, **k: {"ci": 0.5, "phase_name": "Ni", "segments": []})
    return w, lambda pcs, **kw: stub_search(monkeypatch, pcs, **kw)


def test_a_search_result_says_what_the_pc_was_and_the_step_restores_it(search):
    w, stub = search
    stub([FOUND] * 5)
    task = _start(w, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    assert task["result"]["previous"] == {"pc": HEADER, "had_pc": True}
    np.testing.assert_allclose(w.store.get_pc("ds"), FOUND, atol=1e-9)
    assert _client().get("/api/pc/previous").json() == {
        "dataset": "ds", "previous": {"pc": HEADER, "had_pc": True}}
    body = _put_back().json()
    assert body["pc"] == pytest.approx(HEADER) and body["restored"] == ["ds"]
    np.testing.assert_allclose(w.store.get_pc("ds"), HEADER, atol=1e-12)
    assert w.store.get_entry("ds").pc_refinement is None
    np.testing.assert_allclose(w.ctrl_pc("ds"), [HEADER], atol=1e-12)          # the page too


def test_a_search_on_a_file_without_a_pc_says_there_was_none(world, monkeypatch):
    from tests.pc_engine_stub import stub_search
    w = world
    w.add_file("ds", missing=True)
    w.R._sessions["ds"].ctrl.load_phase(_cif(w.tmp))
    monkeypatch.setattr(w.R, "_index_and_simulate",
                        lambda *a, **k: {"ci": 0.5, "phase_name": "Ni", "segments": []})
    stub_search(monkeypatch, [FOUND] * 5)
    task = _start(w, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    assert task["result"]["previous"] == {"pc": None, "had_pc": False}
    assert pc_guard.is_pc_missing("ds", w.store) is False
    _put_back()
    assert pc_guard.is_pc_missing("ds", w.store) is True


def test_a_cancelled_search_leaves_no_step(search):
    w, stub = search
    task = {}

    def press_cancel(call):
        assert asyncio.run(w.R.cancel_optimization(task["id"]))["status"] == "cancelling"
        call["progress"](0.5)

    stub([FOUND] * 5, on_call=press_cancel)
    ctrl = w.R._sessions["ds"].ctrl
    for (r, c) in FIVE:
        ctrl.add_pattern((r, c), np.zeros(SHAPE, np.uint8))
    bt = BackgroundTasks()
    out = asyncio.run(w.R.optimize_pc(w.R.OptimizePCRequest(patterns=[0, 7, 14, 21, 28], mode="find"), bt))
    task["id"] = out["task_id"]
    for t in bt.tasks:
        t.func(*t.args, **t.kwargs)
    assert w.R._optimization_tasks[out["task_id"]]["status"] == "cancelled"
    assert w.R._pc_before == {}                                                # nothing was written, nothing to undo
    resp = _put_back()
    assert resp.status_code == 409 and _code(resp) == "nothingToPutBack"
    assert w.store.get_entry("ds").pc_source == "header"


def test_a_cancelled_search_keeps_the_step_of_the_write_before(search):
    w, stub = search
    _write(w, "ds", TYPED, source="manual")
    task = {}

    def press_cancel(call):
        asyncio.run(w.R.cancel_optimization(task["id"]))
        call["progress"](0.5)

    stub([FOUND] * 5, on_call=press_cancel)
    ctrl = w.R._sessions["ds"].ctrl
    for (r, c) in FIVE:
        ctrl.add_pattern((r, c), np.zeros(SHAPE, np.uint8))
    bt = BackgroundTasks()
    out = asyncio.run(w.R.optimize_pc(w.R.OptimizePCRequest(patterns=[0, 7, 14, 21, 28], mode="find"), bt))
    task["id"] = out["task_id"]
    for t in bt.tasks:
        t.func(*t.args, **t.kwargs)
    assert _put_back().json()["pc"] == pytest.approx(HEADER)


def test_a_typed_pc_says_what_it_replaced_and_goes_back(world):
    w = world
    w.add_file("ds")
    w.R._sessions["ds"].ctrl.load_phase(_cif(w.tmp))
    resp = _client().post("/api/pc/detector/update-pc", json={"pcx": TYPED[0], "pcy": TYPED[1], "pcz": TYPED[2]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["previous"] == {"pc": HEADER, "had_pc": True}
    np.testing.assert_allclose(w.store.get_pc("ds"), TYPED, atol=1e-12)
    assert _put_back().status_code == 200
    np.testing.assert_allclose(w.store.get_pc("ds"), HEADER, atol=1e-12)
    # typing again after a put-back is a new step from the header PC again
    again = _client().post("/api/pc/detector/update-pc", json={"pcx": FOUND[0], "pcy": FOUND[1], "pcz": FOUND[2]})
    assert again.json()["previous"] == {"pc": HEADER, "had_pc": True}


def test_apply_detector_writes_with_the_same_writer_so_it_has_a_step_too(world):
    """Pin: ``/detector/set`` (the page's Apply) goes through ``_writeback_refined_pc``."""
    w = world
    w.add_file("ds")
    body = {"shape": list(SHAPE), "pc": TYPED, "sample_tilt": 70.0, "camera_tilt": 0.0, "binning": 1,
            "azimuthal": 0.0, "convention": "bruker"}
    resp = _client().post("/api/pc/detector/set", json=body)
    assert resp.status_code == 200, resp.text
    np.testing.assert_allclose(w.store.get_pc("ds"), TYPED, atol=1e-12)
    assert _client().get("/api/pc/previous").json() == {
        "dataset": "ds", "previous": {"pc": HEADER, "had_pc": True}}
    assert _put_back().status_code == 200
    np.testing.assert_allclose(w.store.get_pc("ds"), HEADER, atol=1e-12)
    assert _put_back().status_code == 409                                      # used up


def test_a_stand_in_store_without_a_snapshot_still_takes_a_write(world):
    w = world

    class Plain:
        def __init__(self):
            self.calls = []

        def update_pc_detail(self, name, pc, **kw):
            self.calls.append((name, list(pc)))
            return "single"

        def get_entry(self, name):
            return SimpleNamespace(parent_name=None)

        def following_family(self, name, exclude=()):
            return []

    plain = Plain()
    assert w.R._writeback_refined_pc("ds", FOUND, store=plain) == "single"
    assert plain.calls == [("ds", FOUND)] and w.R._pc_before == {}


def test_a_write_that_wrote_nothing_keeps_the_step_there_was(world):
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND)
    assert _write(w, "ghost", TYPED) is None                                  # not in the store: nothing written
    assert w.R._previous_of("ds") == {"pc": HEADER, "had_pc": True}
    assert "ghost" not in w.R._pc_before


def test_a_write_that_fails_keeps_the_step_there_was(world):
    w = world
    w.add_file("ds")
    _write(w, "ds", FOUND)

    class Failing:
        """The real store, but its write raises."""
        def __getattr__(self, attr):
            return getattr(w.store, attr)

        def update_pc_detail(self, *a, **k):
            raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        w.R._writeback_refined_pc("ds", TYPED, store=Failing())
    assert w.R._previous_of("ds") == {"pc": HEADER, "had_pc": True}
    np.testing.assert_allclose(w.store.get_pc("ds"), FOUND, atol=1e-12)


# ------------------------------------------------------------------ a calibration, a deleted dataset

def test_putting_back_ends_the_calibration_the_dataset_was_on(world):
    w = world
    path = w.tmp / "ds.h5oina"
    path.write_bytes(b"ds" * 100)
    w.add_file("ds", file_key=str(path))
    made = cal.create(name="REM 2026-10-14", pc=[0.5026, 0.3272, 0.8458], features=cal.features_of_signal(_kp_signal()))
    out = use.apply(made["id"], ["ds"], accept_mismatch=["ds"], store=w.store)
    assert out[0]["status"] == "applied" and out[0]["previous"] == {"pc": HEADER, "had_pc": True}
    assert use.current_ref("ds", w.store) == {"id": made["id"], "name": "REM 2026-10-14", "version": 1}
    body = _put_back().json()
    assert body["pc"] == pytest.approx(HEADER)
    assert w.store.get_entry("ds").pc_source == "header"
    assert use.current_ref("ds", w.store) is None


def test_putting_back_a_refine_that_left_a_calibration_puts_the_dataset_on_it_again(world):
    w = world
    path = w.tmp / "ds.h5oina"
    path.write_bytes(b"ds" * 100)
    w.add_file("ds", file_key=str(path))
    made = cal.create(name="REM", pc=[0.5026, 0.3272, 0.8458], features=cal.features_of_signal(_kp_signal()))
    use.apply(made["id"], ["ds"], accept_mismatch=["ds"], store=w.store)
    _write(w, "ds", FOUND, refinement=REFINEMENT)
    assert use.current_ref("ds", w.store) is None                              # the refine left it
    _put_back()
    assert use.current_ref("ds", w.store) is not None                          # and the step brings it back
    assert w.store.get_entry("ds").pc_source == "calibration"


def test_putting_back_a_second_calibration_puts_the_dataset_on_the_first_again(world):
    """Apply A, apply B, put back: the PC is A's again, so the file runs on A again (current_ref), not on nothing."""
    w = world
    path = w.tmp / "ds.h5oina"
    path.write_bytes(b"ds" * 100)
    w.add_file("ds", file_key=str(path))
    features = cal.features_of_signal(_kp_signal())
    # names chosen so that the member lists alone (same second, name order) would name the SECOND one
    first = cal.create(name="Z first", pc=[0.5026, 0.3272, 0.8458], features=features)
    second = cal.create(name="B second", pc=[0.4801, 0.3012, 0.7919], features=features)
    use.apply(first["id"], ["ds"], accept_mismatch=["ds"], store=w.store)
    use.apply(second["id"], ["ds"], accept_mismatch=["ds"], store=w.store)
    assert use.current_ref("ds", w.store)["id"] == second["id"]
    body = _put_back().json()
    assert body["pc"] == pytest.approx([0.5026, 0.3272, 0.8458])
    assert use.current_ref("ds", w.store) == {"id": first["id"], "name": "Z first", "version": 1}


def test_putting_back_a_calibration_that_was_the_first_leaves_the_memory_empty(world):
    w = world
    path = w.tmp / "ds.h5oina"
    path.write_bytes(b"ds" * 100)
    w.add_file("ds", file_key=str(path))
    made = cal.create(name="A", pc=[0.5026, 0.3272, 0.8458], features=cal.features_of_signal(_kp_signal()))
    use.apply(made["id"], ["ds"], accept_mismatch=["ds"], store=w.store)
    assert use.memory_usage_of("ds", w.store) == {"id": made["id"], "version": 1}
    _put_back()
    assert use.memory_usage_of("ds", w.store) is None


def test_deleting_a_dataset_drops_its_step(world):
    w = world
    w.add_file("ds")
    w.add_file("other")
    _write(w, "ds", FOUND)
    _write(w, "other", FOUND)
    w.R.drop_session("ds")
    assert "ds" not in w.R._pc_before and "other" in w.R._pc_before


# ------------------------------------------------------------------ what a search left behind, and its word to the page

REFINED = [0.44, 0.22, 0.51]
GOOD_FIND = {"contrast": 0.95, "warning": None, "axes": [], "n_patterns": 5}
UNCLEAR_REFINE = {"contrast": 0.1, "warning": None, "axes": [], "n_patterns": 5}


def _start_on(w, name, positions, **req):
    """Like ``_start`` for the dataset ``name`` (made active), whose page already holds a phase."""
    w.ev._active_dataset = name
    ctrl = w.R._sessions[name].ctrl
    if ctrl.phase is None:
        ctrl.load_phase(_cif(w.tmp))
    for (r, c) in positions:
        ctrl.add_pattern((r, c), np.full(SHAPE, r * 10 + c, np.uint8))
    bt = BackgroundTasks()
    out = asyncio.run(w.R.optimize_pc(w.R.OptimizePCRequest(patterns=[r * 6 + c for r, c in positions], **req), bt))
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)
    return w.R._optimization_tasks[out["task_id"]]


@pytest.fixture
def runs(world, monkeypatch):
    from tests.pc_engine_stub import stub_search, unclear_confidence
    monkeypatch.setattr(world.R, "_index_and_simulate",
                        lambda *a, **k: {"ci": 0.5, "phase_name": "Ni", "segments": []})

    def stub(pc, unclear=False):
        stub_search(monkeypatch, [pc] * 5, **({"confidence": unclear_confidence()} if unclear else {}))

    return world, stub


def _type(pc):
    resp = _client().post("/api/pc/detector/update-pc", json={"pcx": pc[0], "pcy": pc[1], "pcz": pc[2]})
    assert resp.status_code == 200, resp.text


def test_a_find_on_a_file_map_after_a_put_back_is_told_as_shifted_not_replaced(runs):
    """An Oxford-style file: a header map. Typing a PC and putting it back leaves the page's controller holding the
    file's per-pixel map again; the store shifts that map for a find, and the store is what indexing reads."""
    from backend.api.services import pc_in_use
    w, stub = runs
    w.add_file("ds", pc_map=MAP)
    w.R._sessions["ds"].ctrl.load_phase(_cif(w.tmp))
    _type(TYPED)
    assert _put_back().status_code == 200
    assert w.ctrl_pc("ds").shape == (4, 3)                                     # the page holds the map again
    stub(FOUND)
    task = _start_on(w, "ds", FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    assert task["result"]["pc_map_action"] == "shifted"
    assert task["result"]["pc_map_actions"] == {"ds": "shifted"}
    got = pc_in_use.pc_in_use_of("ds", w.store)
    assert got["per_pixel"] is True and got["kind"] == "find"
    np.testing.assert_allclose(w.store.get_entry("ds").pc_map.reshape(-1, 3).mean(axis=0), FOUND, atol=1e-9)


def test_a_find_on_a_copy_leaves_its_quality_with_the_raw_file(runs):
    from backend.api.routes import pc_calibrations as routes
    from backend.api.services import pc_in_use
    from backend.api.services.pc_record import pc_run_record
    w, stub = runs
    w.add_file("ds")
    w.add_copy("copy", "ds")
    stub(FOUND)
    task = _start_on(w, "copy", FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    assert pc_in_use.pc_quality_of("ds", w.store) == GOOD_FIND
    assert pc_in_use.pc_quality_of("copy", w.store) == GOOD_FIND
    assert pc_run_record("ds", w.store)["pc_quality"] == GOOD_FIND
    assert routes._origin_of("ds")["quality"] == GOOD_FIND                      # what a calibration saved from the raw file carries


def test_a_find_a_refine_and_a_put_back_is_a_find_with_its_own_quality_again(runs):
    from backend.api.services import pc_in_use
    w, stub = runs
    w.add_file("ds")
    stub(FOUND)
    assert _start_on(w, "ds", FIVE, mode="find")["status"] == "completed"
    stub(REFINED, unclear=True)
    assert _start_on(w, "ds", FIVE, mode="refine")["status"] == "completed"
    now = pc_in_use.pc_in_use_of("ds", w.store)
    assert now["kind"] == "refine" and pc_in_use.pc_quality_of("ds", w.store) == UNCLEAR_REFINE
    assert _put_back().status_code == 200
    np.testing.assert_allclose(w.store.get_pc("ds"), FOUND, atol=1e-9)
    assert pc_in_use.pc_in_use_of("ds", w.store)["kind"] == "find"
    assert pc_in_use.pc_quality_of("ds", w.store) == GOOD_FIND


def test_a_find_a_typed_pc_and_a_put_back_gives_the_finds_quality_again(runs):
    from backend.api.services import pc_in_use
    w, stub = runs
    w.add_file("ds")
    stub(FOUND)
    assert _start_on(w, "ds", FIVE, mode="find")["status"] == "completed"
    _type(TYPED)
    assert pc_in_use.pc_quality_of("ds", w.store) is None
    assert _put_back().status_code == 200
    assert pc_in_use.pc_in_use_of("ds", w.store)["kind"] == "find"
    assert pc_in_use.pc_quality_of("ds", w.store) == GOOD_FIND


def test_putting_back_a_find_on_a_file_that_had_no_search_forgets_the_search(runs):
    from backend.api.services import pc_in_use
    w, stub = runs
    w.add_file("ds")
    stub(FOUND)
    assert _start_on(w, "ds", FIVE, mode="find")["status"] == "completed"
    assert use.search_of("ds", w.store) is not None
    assert _put_back().status_code == 200
    assert use.search_of("ds", w.store) is None
    _type(FOUND)                      # the same numbers typed by hand are not that search's result
    assert pc_in_use.pc_in_use_of("ds", w.store)["kind"] == "typed"
    assert pc_in_use.pc_quality_of("ds", w.store) is None


# ------------------------------------------------------------------ a second position of a find

OTHER = [0.40, 0.30, 1.30]
FOUND_COST, OTHER_COST = 4.7, 5.0


def _pos(pc, cost):
    """A second position as the answers give it; the PC of a search is a mean, so equal to rounding."""
    return {"pc": pytest.approx(pc, abs=1e-12), "cost": cost}


def _find_with_second(w, monkeypatch, name="ds", **extra):
    """A find whose stand-in search reports FOUND at cost 4.7 and OTHER as its second position at cost 5.0."""
    from tests.pc_engine_stub import stub_search
    stub_search(monkeypatch, [FOUND] * 5, method="PSO+NM", cost=FOUND_COST,
                second_position={"pc": tuple(OTHER), "cost": OTHER_COST}, **extra)
    task = _start_on(w, name, FIVE, mode="find")
    assert task["status"] == "completed", task.get("error")
    return task


def test_a_calibration_saved_after_a_find_carries_the_second_position(runs, monkeypatch):
    from backend.api.routes import pc_calibrations as routes
    w, _ = runs
    w.add_file("ds")
    _find_with_second(w, monkeypatch)
    first = routes._origin_of("ds")["quality"]
    assert first["warning"] == "secondPosition" and first["second_position"] == _pos(OTHER, OTHER_COST)
    assert first["best_cost"] == FOUND_COST
    assert routes._origin_of("ds")["method"] == "find"
