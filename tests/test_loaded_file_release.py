"""Removing a file from the loaded-files list lets go of its pattern file.

A native UP scan is opened as a memory map (``edax_apex._MappedPatterns``). When the user leaves a
file that has derived datasets (a crop), the file's datasets are stashed so a switch back restores
them, and the stash keeps the lazy signal and with it the map. On Windows a mapped file can be
neither replaced nor moved (WinError 32) until the map is gone, so a file the user removed from
the list stayed locked until the backend was restarted.

Hand-written expectations: after the removal no stash of that file exists, the stashed signals are
no longer referenced (a weak reference to each is dead after a garbage collection), and the pattern
file can be deleted. The active file is out of scope: it stays open.
"""
from __future__ import annotations

import asyncio
import gc
import os
import threading
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.apex_helpers import make_apex_h5, make_up, measurement_spec, wipe_backend  # noqa: E402
from tests.test_calibration_survives_file_switch import (  # noqa: E402,F401  (viewer is a fixture)
    HEADER, MAP, _calibrated_scan5, _reload, _sig, viewer,
)

ROWS, COLS = 6, 5
CROP = dict(shape="rect", row0=1, col0=0, rows=3, cols=4, name="odd")


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture(autouse=True)
def lazy_and_clean(monkeypatch):
    monkeypatch.setenv("KIKUCHIPY_LAZY_THRESHOLD_BYTES", "0")        # every file is opened lazily
    wipe_backend()
    yield
    wipe_backend()


@pytest.fixture
def files(tmp_path):
    """Two hexagonal APEX scans (so that either can be the one left behind) and a square file."""
    paths = []
    for name in ("one", "two"):
        d = tmp_path / name
        d.mkdir()
        make_apex_h5(d / "s.h5", [measurement_spec(n_rows=ROWS, n_cols=COLS, step_x=0.5)])
        paths.append(make_up(d / f"scan_{name}.up1", n=ROWS * COLS, nx=COLS, ny=ROWS))
    d = tmp_path / "sq"
    d.mkdir()
    paths.append(make_up(d / "o.up1", guid_bytes=None, version=3, hex_flag=0, n=6, nx=3, ny=2))
    return paths


def _load(client, path):
    r = client.post("/api/ebsd/load", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _left_behind(client, path, then_load):
    """``path`` loaded, cropped (a derived dataset, so the file is stashed), then ``then_load`` loaded."""
    _load(client, path)
    r = client.post("/api/ebsd/crop", json=CROP)
    assert r.status_code == 200, r.text
    _load(client, then_load)


def _stashed_signals(path):
    from backend.api.routes import ebsd_viewer
    stash = ebsd_viewer._registry_by_file.get(ebsd_viewer._canonical_path(str(path)))
    assert stash is not None, "the premise: leaving a file with a crop stashes it"
    refs = [weakref.ref(s) for s in stash["raw_signals"].values()]
    assert refs, "the premise: the stash holds the file's signals"
    return refs


def _collect():
    for _ in range(3):
        gc.collect()


def _can_be_deleted(path):
    try:
        os.remove(path)
    except PermissionError:
        return False
    return True


def test_removing_a_file_from_the_list_drops_its_stash_and_frees_the_pattern_file(client, files):
    one, _, sq = files
    _left_behind(client, one, sq)
    refs = _stashed_signals(one)
    r = client.post("/api/ebsd/loaded-files/remove", json={"path": str(one)})
    assert r.status_code == 200, r.text
    from backend.api.routes import ebsd_viewer
    assert ebsd_viewer._canonical_path(str(one)) not in ebsd_viewer._registry_by_file
    _collect()
    assert [r() for r in refs] == [None] * len(refs)
    assert _can_be_deleted(one), "the pattern file is still mapped (WinError 32 on Windows)"


def test_clearing_the_list_drops_the_stash_of_every_file_but_the_active_one(client, files):
    one, two, sq = files
    _left_behind(client, one, two)
    _left_behind(client, two, sq)
    one_refs, two_refs = _stashed_signals(one), _stashed_signals(two)
    assert client.post("/api/ebsd/loaded-files/clear").status_code == 200
    _collect()
    assert [r() for r in one_refs + two_refs] == [None] * (len(one_refs) + len(two_refs))
    assert _can_be_deleted(one) and _can_be_deleted(two)
    assert client.get("/api/ebsd/info").json()["loaded"] is True       # the active file stays open


def test_clearing_everything_drops_the_stashes_too(client, files):
    one, _, sq = files
    _left_behind(client, one, sq)
    refs = _stashed_signals(one)
    assert client.post("/api/ebsd/loaded-files/clear-all").status_code == 200
    _collect()
    assert [r() for r in refs] == [None] * len(refs)
    assert _can_be_deleted(one)


def test_the_active_file_cannot_be_removed_and_keeps_its_data(client, files):
    """Out of scope and unchanged: the refusal comes first and nothing is dropped."""
    one, _, sq = files
    _left_behind(client, one, sq)
    r = client.post("/api/ebsd/loaded-files/remove", json={"path": str(sq)})
    assert r.status_code == 400
    from backend.api.routes import ebsd_viewer
    assert ebsd_viewer._canonical_path(str(one)) in ebsd_viewer._registry_by_file     # the other file's stash is untouched


def test_removing_a_file_that_is_not_in_the_list_leaves_other_stashes_alone(client, files, tmp_path):
    one, _, sq = files
    _left_behind(client, one, sq)
    r = client.post("/api/ebsd/loaded-files/remove", json={"path": str(tmp_path / "never_loaded.up1")})
    assert r.status_code == 404
    from backend.api.routes import ebsd_viewer
    assert ebsd_viewer._canonical_path(str(one)) in ebsd_viewer._registry_by_file


def test_removing_one_file_keeps_the_stash_of_another_file_that_stays_in_the_list(client, files):
    one, two, sq = files
    _left_behind(client, one, two)
    _left_behind(client, two, sq)
    two_refs = _stashed_signals(two)
    assert client.post("/api/ebsd/loaded-files/remove", json={"path": str(one)}).status_code == 200
    _collect()
    from backend.api.routes import ebsd_viewer
    assert ebsd_viewer._canonical_path(str(two)) in ebsd_viewer._registry_by_file
    assert all(r() is not None for r in two_refs)
    assert _can_be_deleted(one)
    assert client.post("/api/ebsd/load", json={"path": str(two)}).json()["dataset_name"] == "odd"


def test_a_file_that_stays_in_the_list_is_restored_with_its_crop(client, files):
    """The stash is for switching back: dropping it only for removed files must not touch this."""
    one, _, sq = files
    _left_behind(client, one, sq)
    again = _load(client, one)
    assert again["dataset_name"] == "odd"


# --- a PC the user made outlives the removal --------------------------------------------------
#
# A raw-only file whose PC was typed, refined or grid-calibrated is stashed without any signal
# (only ``calibration`` and ``geometries``, see test_calibration_survives_file_switch): dropping
# that frees nothing, and the user guide promises that a PC you set survives a file switch. What
# removal discards is every signal reference and every derived dataset; the user's PC stays.

USER_PC = [0.4301, 0.2141, 0.5009]
REFINED_BG = [0.45, 0.22, 0.51]


class _Weakly(SimpleNamespace):
    """A stand-in signal that a weak reference can point at (SimpleNamespace itself cannot)."""
GRID_MAP_SHAPE = (3, 2)


@pytest.fixture
def squares(tmp_path):
    """Three square UP1 scans without a PC (6x5 patterns each, so a crop window fits)."""
    paths = []
    for name in ("a", "b", "c"):
        d = tmp_path / name
        d.mkdir()
        paths.append(make_up(d / f"{name}.up1", guid_bytes=None, version=3, hex_flag=0,
                             n=ROWS * COLS, nx=COLS, ny=ROWS))
    return paths


def _make_pc(how):
    """Type or calibrate a PC on the active dataset; return the ``pc_source`` that results."""
    import numpy as np
    from backend.api.routes import ebsd_viewer
    from backend.api.services.calibration_store import calibration_store
    name = ebsd_viewer._active_dataset
    assert calibration_store.get_entry(name).pc_source == "missing", "the premise: the file has no PC"
    if how == "manual":
        assert calibration_store.update_pc(name, USER_PC, source="manual")
        return "manual"
    pc_map = np.tile(np.array(USER_PC), GRID_MAP_SHAPE + (1,))
    pc_map[0, 0] = (0.40, 0.30, 0.70)
    assert calibration_store.update_pc_map(name, pc_map, refinement={"n_patterns": 9, "method": "PSO"})
    return "refined_map"


def _remove(client, path):
    return client.post("/api/ebsd/loaded-files/remove", json={"path": str(path)})


def _clear(client, path):
    return client.post("/api/ebsd/loaded-files/clear")


def _clear_all(client, path):
    return client.post("/api/ebsd/loaded-files/clear-all")


@pytest.mark.parametrize("how", ["manual", "map"])
@pytest.mark.parametrize("leave", [_remove, _clear, _clear_all], ids=["remove", "clear", "clear-all"])
def test_a_pc_the_user_made_survives_the_removal_of_its_file(client, squares, how, leave):
    from backend.api.routes import ebsd_viewer
    from backend.api.services.calibration_store import calibration_store
    a, b, _ = squares
    _load(client, a)
    want = _make_pc(how)
    _load(client, b)                                  # leaves a: a calibration-only stash
    key = ebsd_viewer._canonical_path(str(a))
    assert ebsd_viewer._registry_by_file[key]["raw_signals"] == {}, "the premise: no signal in the stash"
    assert leave(client, a).status_code == 200
    stash = ebsd_viewer._registry_by_file.get(key)
    assert stash is not None, "the PC the user made was thrown away with the file"
    assert stash["raw_signals"] == {} and stash["crops"] == {} and stash["dirty"] == set()
    assert [e.pc_source for e in stash["calibration"].values()] == [want]
    again = _load(client, a)                          # by path: it is not in the list any more
    assert calibration_store.get_entry(ebsd_viewer._active_dataset).pc_source == want
    assert again["pc_source"] == want
    assert again["pc_defaulted"] is False
    if how == "manual":
        assert again["pc"] == pytest.approx(USER_PC, abs=1e-9)


def test_a_file_without_a_pc_of_the_user_leaves_no_stash_behind(client, squares):
    from backend.api.routes import ebsd_viewer
    a, b, _ = squares
    _load(client, a)
    _load(client, b)
    # the premise: a raw-only file is left with its calibration (the PC page lists it) and no signal
    (stash,) = ebsd_viewer._registry_by_file.values()
    assert stash["raw_signals"] == {} and stash["dirty"] == set() and stash["crops"] == {}
    assert _remove(client, a).status_code == 200
    assert ebsd_viewer._registry_by_file == {}


def test_a_crop_stash_with_a_typed_pc_lets_go_of_its_signals_and_keeps_the_pc(client, squares):
    from backend.api.routes import ebsd_viewer
    from backend.api.services.calibration_store import calibration_store
    a, b, _ = squares
    _load(client, a)
    raw_name = ebsd_viewer._active_dataset
    _make_pc("manual")
    r = client.post("/api/ebsd/crop", json=CROP)
    assert r.status_code == 200, r.text
    assert ebsd_viewer._active_dataset == "odd"
    assert calibration_store.get_entry("odd") is not None, "the premise: the crop has its own entry"
    _load(client, b)
    key = ebsd_viewer._canonical_path(str(a))
    refs = [weakref.ref(s) for s in ebsd_viewer._registry_by_file[key]["raw_signals"].values()]
    assert len(refs) == 2, "the premise: the stash holds the raw and the cropped signal"
    assert _remove(client, a).status_code == 200
    _collect()
    assert [r() for r in refs] == [None] * len(refs)
    assert _can_be_deleted(a), "the pattern file is still mapped (WinError 32 on Windows)"
    stash = ebsd_viewer._registry_by_file[key]
    assert stash["raw_signals"] == {} and stash["crops"] == {} and stash["crop_masks"] == {}
    assert list(stash["calibration"]) == [raw_name]    # the crop's own entry is gone with the crop
    assert stash["calibration"][raw_name].pc_source == "manual"


def test_the_derived_datasets_are_released_and_the_pc_stays(viewer):
    """A background-removed dataset (materialised, minutes of compute) is dropped with the removal."""
    ev, store = viewer
    _calibrated_scan5(ev, store, "manual")
    derived = _Weakly(**vars(_sig(HEADER)))
    ev._raw_signals["scan5_bg"] = derived
    ev._dirty_datasets.add("scan5_bg")
    store.register_derived("scan5_bg", "scan5")
    store.update_pc("scan5_bg", REFINED_BG, source="propagated")
    from backend.api.services import scan_geometry
    scan_geometry.set_geometry("scan5", scan_geometry.ScanGeometry.square(4, 4, 0.5))
    scan_geometry.set_geometry("scan5_bg", scan_geometry.ScanGeometry.square(4, 4, 0.5))
    ev._stash_registry_for_file("/data/scan5.up1")
    (snap,) = ev._registry_by_file.values()
    assert "scan5_bg" in snap["raw_signals"] and snap["dirty"] == {"scan5_bg"}, "the premise"
    assert set(snap["geometries"]) == {"scan5", "scan5_bg"}, "the premise"
    ref = weakref.ref(derived)
    ev._raw_signals.clear()                           # what loading the next file does
    ev._dirty_datasets.clear()
    del derived, snap
    ev._drop_stash("/data/scan5.up1")
    (snap,) = ev._registry_by_file.values()
    assert snap["raw_signals"] == {} and snap["dirty"] == set() and snap["positions"] == {}
    assert list(snap["calibration"]) == ["scan5"] and snap["calibration"]["scan5"].pc_source == "manual"
    assert set(snap["geometries"]) == {"scan5"}        # the derived dataset's geometry goes with it
    _collect()
    assert ref() is None


@pytest.mark.parametrize("how", ["map", "refined", "manual"])
def test_after_the_removal_a_reload_of_the_file_brings_the_pc_back(viewer, how):
    """A calibrated raw-only file, left, removed from the list, loaded again."""
    ev, store = viewer
    _calibrated_scan5(ev, store, how)
    want = store.get_entry("scan5").pc_source
    _reload(ev, store, "/data/scan5.up1", "/data/other.h5", "other", _sig(HEADER))
    ev._drop_stash("/data/scan5.up1")                  # what removing the file from the list does
    assert _reload(ev, store, "/data/other.h5", "/data/scan5.up1", "scan5",
                   _sig((0.5, 0.5, 0.5), tag="default")) is True
    entry = store.get_entry("scan5")
    assert entry.pc_source == want
    if how == "map":
        assert entry.pc_map is not None and entry.pc_map.shape == MAP.shape
    assert ev._pc_load_notice["scan5"][0] == "user_pc_kept"


# --- a clear while a switch is running -----------------------------------------------------------
#
# ``_ebsd_file_path`` changes only when the load worker finishes. A clear during a switch from A to B
# therefore sees A as the active file, and B - the file the load is about to make active, whose
# stash the worker is about to restore - looks like any other file that leaves the list.


@pytest.mark.parametrize("which", ["clear", "clear-all"])
def test_a_clear_during_a_switch_keeps_the_stash_of_the_file_being_loaded(monkeypatch, files, which):
    from backend.api.routes import ebsd_viewer as ev
    one, two, sq = files
    from backend.api.main import app
    client = TestClient(app, client=("127.0.0.1", 5000))
    _left_behind(client, one, two)
    _left_behind(client, two, sq)                       # sq is active; one and two are stashed
    assert {ev._canonical_path(str(one)), ev._canonical_path(str(two))} <= set(ev._registry_by_file)
    started, release = threading.Event(), threading.Event()
    real = ev._load_ebsd_blocking

    def slow(*args, **kwargs):
        started.set()
        assert release.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(ev, "_load_ebsd_blocking", slow)
    clear = ev.clear_loaded_files if which == "clear" else ev.clear_all_loaded_files

    async def scenario():
        task = asyncio.create_task(ev.load_ebsd(ev.LoadEBSDRequest(path=str(two))))
        while not started.is_set():
            await asyncio.sleep(0.01)
        await clear()                                   # the switch to two is running
        release.set()
        return await task

    answer = asyncio.run(scenario())
    assert answer["dataset_name"] == "odd", "two's crop was dropped with its stash during the switch"
    assert ev._canonical_path(str(one)) not in ev._registry_by_file       # a file that only sat in the list
    assert ev._loading_paths == []                       # nothing is marked as loading afterwards


def test_removing_the_file_a_switch_is_loading_keeps_its_stash(monkeypatch, files):
    """The panel hides the x only for its own switch; the header switcher and the Indexing dropdown
    switch too. ``remove`` is the third route that drops a stash, and the file being loaded is not
    "a file that left the list" yet: the worker is about to restore its crop."""
    from backend.api.routes import ebsd_viewer as ev
    one, two, sq = files
    from backend.api.main import app
    client = TestClient(app, client=("127.0.0.1", 5000))
    _left_behind(client, one, two)
    _left_behind(client, two, sq)                       # sq is active; one and two are stashed
    started, release = threading.Event(), threading.Event()
    real = ev._load_ebsd_blocking

    def slow(*args, **kwargs):
        started.set()
        assert release.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(ev, "_load_ebsd_blocking", slow)

    async def scenario():
        task = asyncio.create_task(ev.load_ebsd(ev.LoadEBSDRequest(path=str(two))))
        while not started.is_set():
            await asyncio.sleep(0.01)
        await ev.remove_loaded_file(ev.SwitchFileRequest(path=str(two)))      # the x of the file being loaded
        release.set()
        return await task

    answer = asyncio.run(scenario())
    assert answer["dataset_name"] == "odd", "two's crop was dropped with its stash during the switch"
    assert ev._loading_paths == []


def test_removing_a_file_that_is_not_being_loaded_still_drops_its_stash(files):
    from backend.api.routes import ebsd_viewer as ev
    one, two, sq = files
    from backend.api.main import app
    client = TestClient(app, client=("127.0.0.1", 5000))
    _left_behind(client, one, two)
    _left_behind(client, two, sq)
    assert ev._canonical_path(str(one)) in ev._registry_by_file
    asyncio.run(ev.remove_loaded_file(ev.SwitchFileRequest(path=str(one))))
    assert ev._canonical_path(str(one)) not in ev._registry_by_file


def test_clear_all_during_a_switch_leaves_no_stash_of_the_file_that_was_left(monkeypatch, files):
    """``clear-all`` empties the list while a switch from A to B runs. A is still the active file at
    that moment, so its stash is kept; the worker then stashes A (it holds a crop) on its way out. A is
    no longer in the list by then, so nothing would ever release that stash and it would hold A's
    pattern file open (on Windows a mapped file can be neither replaced nor moved)."""
    from backend.api.routes import ebsd_viewer as ev
    one, two, sq = files
    from backend.api.main import app
    client = TestClient(app, client=("127.0.0.1", 5000))
    _load(client, one)
    assert client.post("/api/ebsd/crop", json=CROP).status_code == 200      # one is active, with a crop
    started, release = threading.Event(), threading.Event()
    real = ev._load_ebsd_blocking

    def slow(*args, **kwargs):
        started.set()
        assert release.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(ev, "_load_ebsd_blocking", slow)

    async def scenario():
        task = asyncio.create_task(ev.load_ebsd(ev.LoadEBSDRequest(path=str(sq))))
        while not started.is_set():
            await asyncio.sleep(0.01)
        await ev.clear_all_loaded_files()                # the switch to sq is running
        release.set()
        return await task

    asyncio.run(scenario())
    assert [e["path"] for e in ev._loaded_files] == [str(sq)]               # premise: only sq is listed
    assert ev._canonical_path(str(one)) not in ev._registry_by_file, "one left the list and still holds a stash"
