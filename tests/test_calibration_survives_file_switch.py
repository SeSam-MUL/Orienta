"""A PC the user calibrated on a raw dataset survives a file switch and a reload.

A file that carries no PC (UP1 without .osc) is calibrated on the PC page: a grid
calibration leaves a per-pixel map, a refine a PC. Loading another file and coming
back (or loading the same file again) re-registers the raw dataset from the file,
which has no PC - and the stash skipped a file whose only dataset is the raw one,
and restore skipped the raw entry anyway. The store said "missing" again while the
PC page still showed the map, and indexing refused "calibrate in PC refinement":
unresolvable. The user's own calibration is now carried over; what the file gives
(a header PC) is not overwritten.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from backend.api.services.calibration_store import CalibrationStore

SHAPE = (60, 60)
HEADER = np.array([0.5026, 0.3273, 0.8458])
MAP = np.stack([np.full((4, 5), 0.47), np.full((4, 5), 0.41), np.full((4, 5), 0.77)], axis=-1)
MAP[0, 0] = (0.40, 0.30, 0.70)            # the map is not flat, so shape and mean differ
REFINED = np.array([0.4301, 0.2141, 0.5009])


class _Det:
    def __init__(self, pc):
        self.shape = SHAPE
        self.pc = np.asarray(pc, dtype=float)
        self.sample_tilt, self.tilt, self.azimuthal = 70.0, 3.0, 0.0


def _sig(pc, tag=None):
    sig = SimpleNamespace(detector=_Det(pc))
    if tag is not None:
        sig.metadata = type("M", (), {"get_item": staticmethod(lambda k, d=None: tag)})()
    return sig


@pytest.fixture
def viewer(monkeypatch):
    from backend.api.routes import ebsd_viewer as ev
    store = CalibrationStore()
    monkeypatch.setattr(ev, "calibration_store", store)
    for attr in ("_raw_signals", "_positions", "_signal_masks", "_registry_by_file"):
        monkeypatch.setattr(ev, attr, {})
    monkeypatch.setattr(ev, "_dirty_datasets", set())
    monkeypatch.setattr(ev, "_active_dataset", "scan5")
    return ev, store


def _reload(ev, store, leaving, loading, name, sig):
    """What loading a file does: stash the registry of the file being left, clear,
    register the fresh raw dataset of the file being loaded, restore its stash."""
    ev._stash_registry_for_file(leaving)
    ev._raw_signals.clear(); ev._positions.clear(); ev._dirty_datasets.clear()
    store.clear()
    ev._raw_signals[name] = sig
    store.register(name, sig)
    ev._active_dataset = name
    return ev._restore_registry_for_file(loading, name)


def _calibrated_scan5(ev, store, how):
    sig = _sig((0.5, 0.5, 0.5), tag="default")
    ev._raw_signals["scan5"] = sig
    store.register("scan5", sig)
    assert store.get_entry("scan5").pc_source == "missing"
    if how == "map":
        store.update_pc_map("scan5", MAP, refinement={"n_patterns": 9, "method": "PSO"})
    elif how == "refined":
        store.update_pc("scan5", REFINED, source="refined")
    else:
        store.update_pc("scan5", REFINED, source="manual")


@pytest.mark.parametrize("how", ["map", "refined", "manual"])
def test_the_calibration_survives_a_switch_to_another_file_and_back(viewer, how):
    ev, store = viewer
    _calibrated_scan5(ev, store, how)
    want = store.get_entry("scan5").pc_source
    _reload(ev, store, "/data/scan5.up1", "/data/other.h5", "other", _sig(HEADER))
    assert _reload(ev, store, "/data/other.h5", "/data/scan5.up1", "scan5", _sig((0.5, 0.5, 0.5), tag="default")) is True
    e = store.get_entry("scan5")
    assert e.pc_source == want                                   # not "missing"
    assert store.get_pc("scan5") is not None
    if how == "map":
        assert e.pc_map is not None and e.pc_map.shape == MAP.shape
        np.testing.assert_allclose(e.pc_single, MAP.reshape(-1, 3).mean(axis=0), atol=1e-12)
        assert e.pc_refinement == {"n_patterns": 9, "method": "PSO"}
    else:
        np.testing.assert_allclose(store.get_pc("scan5"), REFINED, atol=1e-12)


def test_the_calibration_survives_loading_the_same_file_again(viewer):
    ev, store = viewer
    _calibrated_scan5(ev, store, "map")
    assert _reload(ev, store, "/data/scan5.up1", "/data/scan5.up1", "scan5", _sig((0.5, 0.5, 0.5), tag="default")) is True
    assert store.get_entry("scan5").pc_source == "refined_map"
    assert store.get_entry("scan5").pc_map.shape == MAP.shape


def test_a_header_pc_is_not_overwritten_by_the_stash(viewer):
    ev, store = viewer
    sig = _sig(HEADER)
    ev._raw_signals["scan5"] = sig
    store.register("scan5", sig)
    ev._raw_signals["scan5_bg"] = _sig(HEADER); ev._dirty_datasets.add("scan5_bg")
    store.register_derived("scan5_bg", "scan5")          # a derived dataset: the registry IS stashed
    _reload(ev, store, "/data/scan5.h5", "/data/other.h5", "other", _sig(HEADER))
    new_header = np.array([0.51, 0.33, 0.84])            # what the file says now
    _reload(ev, store, "/data/other.h5", "/data/scan5.h5", "scan5", _sig(new_header))
    np.testing.assert_allclose(store.get_pc("scan5"), new_header, atol=1e-12)
    assert store.get_entry("scan5").pc_source == "header"


def test_a_raw_only_file_leaves_its_calibration_and_no_signal(viewer):
    """The PC page lists the files that were left (to give them a calibration), so even a file with the PC
    its header states keeps its entry; the signal is not kept (that is what batch indexing must not pile up)."""
    ev, store = viewer
    sig = _sig(HEADER)
    ev._raw_signals["scan5"] = sig
    store.register("scan5", sig)
    ev._stash_registry_for_file("/data/scan5.h5")
    (snap,) = ev._registry_by_file.values()
    assert snap["raw_signals"] == {} and snap["dirty"] == set() and snap["crops"] == {}
    assert list(snap["calibration"]) == ["scan5"] and snap["calibration"]["scan5"].pc_source == "header"


def test_a_raw_file_the_store_does_not_know_is_not_stashed(viewer):
    ev, store = viewer
    ev._raw_signals["scan5"] = _sig(HEADER)               # nothing registered: no calibration to keep
    ev._stash_registry_for_file("/data/scan5.h5")
    assert ev._registry_by_file == {}


def test_a_header_pc_comes_back_as_the_files_own_after_a_round_trip(viewer):
    ev, store = viewer
    sig = _sig(HEADER)
    ev._raw_signals["scan5"] = sig
    store.register("scan5", sig)
    _reload(ev, store, "/data/scan5.h5", "/data/other.h5", "other", _sig(HEADER))
    new_header = np.array([0.51, 0.33, 0.84])             # what the file says now
    assert _reload(ev, store, "/data/other.h5", "/data/scan5.h5", "scan5", _sig(new_header)) is True
    np.testing.assert_allclose(store.get_pc("scan5"), new_header, atol=1e-12)
    assert store.get_entry("scan5").pc_source == "header"


def test_a_calibration_stash_holds_no_signal(viewer):
    ev, store = viewer
    _calibrated_scan5(ev, store, "refined")
    ev._stash_registry_for_file("/data/scan5.up1")
    (snap,) = ev._registry_by_file.values()
    assert snap["raw_signals"] == {}                      # no h5 handle is kept alive
