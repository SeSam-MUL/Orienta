"""A refined PC reaches the copies of its raw dataset, not only the copy and its parent.

A Global PC Refine usually runs on a derived copy (``..._bg_dyn_avg``). The PC went
to that copy and to its parent. Another copy made BEFORE the refine (a second
background removal, a frame average) kept the old PC, and so did the PC page of any
dataset that had already been opened, although the user refined "this sample's"
PC. Copies that never had a PC of their own now follow; one whose PC was typed,
refined or read from its file does not.
"""
import numpy as np
import pytest

from backend.api.services.calibration_store import CalibrationStore

OLD = np.array([0.50, 0.40, 0.60])
NEW = np.array([0.5321, 0.3987, 0.7123])


class _Det:
    def __init__(self, pc=OLD):
        self.shape = (60, 60)
        self.pc = np.asarray(pc, dtype=float)
        self.sample_tilt, self.tilt, self.azimuthal = 70.0, 0.0, 0.0


class _Sig:
    def __init__(self, det):
        self.detector = det


def _family():
    store = CalibrationStore()
    store.register("raw", _Sig(_Det()))
    store.register_derived("a", "raw")            # a copy made before the refine
    store.register_derived("b", "raw")            # the copy the refine runs on
    store.register_derived("a2", "a")             # a copy of a copy
    store.register("other", _Sig(_Det()))         # another file entirely
    return store


def _write(store, name="b", pc=NEW, **kw):
    from backend.api.routes.pcrefinement import _writeback_refined_pc
    return _writeback_refined_pc(name, pc, store=store, source="refined", **kw)


def test_the_whole_family_that_inherits_follows(monkeypatch):
    store = _family()
    _write(store)
    for name in ("b", "raw", "a", "a2"):
        np.testing.assert_allclose(store.get_pc(name), NEW, atol=1e-12, err_msg=name)


def test_the_raw_dataset_gets_it_through_a_chain_of_copies():
    store = _family()
    _write(store, name="a2")                  # refined on a copy of a copy of raw
    for name in ("a2", "a", "raw", "b"):
        np.testing.assert_allclose(store.get_pc(name), NEW, atol=1e-12, err_msg=name)


def test_another_file_is_not_touched():
    store = _family()
    _write(store)
    np.testing.assert_allclose(store.get_pc("other"), OLD, atol=1e-12)


def test_a_copy_with_a_pc_of_its_own_keeps_it_and_its_copies_follow_it():
    store = _family()
    store.update_pc("a", [0.47, 0.41, 0.58], source="manual")        # typed on that copy
    _write(store)
    np.testing.assert_allclose(store.get_pc("a"), [0.47, 0.41, 0.58], atol=1e-12)
    assert store.get_entry("a").pc_source == "manual"
    # a2 is a copy of `a`, which has its own PC: it does not follow the family
    np.testing.assert_allclose(store.get_pc("a2"), OLD, atol=1e-12)
    np.testing.assert_allclose(store.get_pc("raw"), NEW, atol=1e-12)


def test_a_second_refine_reaches_them_again():
    store = _family()
    _write(store)
    later = np.array([0.5400, 0.4100, 0.7000])
    _write(store, pc=later)
    for name in ("b", "raw", "a", "a2"):
        np.testing.assert_allclose(store.get_pc(name), later, atol=1e-12, err_msg=name)


def test_the_opened_pc_page_of_a_followed_dataset_follows_too(monkeypatch):
    from kikuchipy.detectors import EBSDDetector
    from backend.api.routes import pcrefinement as R
    from pc_controller import PCController

    store = _family()
    ctrl = PCController()
    ctrl.attach_detector(EBSDDetector(shape=(60, 60), pc=tuple(OLD), sample_tilt=70.0))
    sess = R._PCSession(ctrl=ctrl, sim_cache={0: {"ci": 0.3}})
    monkeypatch.setitem(R._sessions, "a", sess)
    _write(store)
    np.testing.assert_allclose(np.asarray(ctrl.detector.pc).ravel(), NEW, atol=1e-12)
    assert sess.sim_cache == {}                           # the overlay of the old PC is gone


def test_following_family_names_the_copies_that_follow():
    store = _family()
    assert store.following_family("other") == []
    assert sorted(store.following_family("b")) == ["a", "a2"]            # raw is the root, not a follower
    assert sorted(store.following_family("b", exclude=["a"])) == ["a2"]


def test_copies_of_a_file_without_a_pc_follow_the_first_refine():
    """A copy of a "missing" dataset is "missing" too (inherited); the refine that
    gives the family a PC must reach it, or it stays unindexable beside its raw."""
    store = CalibrationStore()
    sig = _Sig(_Det())
    sig.metadata = type("M", (), {"get_item": staticmethod(lambda k, d=None: "default")})()
    store.register("raw", sig)
    store.register_derived("a", "raw")
    store.register_derived("b", "raw")
    assert store.get_entry("a").pc_source == "missing"
    _write(store, name="b")
    for name in ("raw", "a", "b"):
        np.testing.assert_allclose(store.get_pc(name), NEW, atol=1e-12, err_msg=name)
        assert store.get_entry(name).pc_source != "missing"


def _missing_family():
    """raw (no PC) -> c1 -> c11 -> c111, and a second copy c2 of raw."""
    store = CalibrationStore()
    sig = _Sig(_Det())
    sig.metadata = type("M", (), {"get_item": staticmethod(lambda k, d=None: "default")})()
    store.register("raw", sig)
    store.register_derived("c1", "raw")
    store.register_derived("c11", "c1")
    store.register_derived("c111", "c11")
    store.register_derived("c2", "raw")
    return store


def test_the_copies_of_the_refined_dataset_follow_it():
    """c1 is refined and now holds a PC of its own; its own copies are measured
    from it, not from the root, so they follow (they stayed "missing" before)."""
    store = _missing_family()
    _write(store, name="c1")
    for name in ("c1", "c11", "c111", "raw", "c2"):
        np.testing.assert_allclose(store.get_pc(name), NEW, atol=1e-12, err_msg=name)
        assert store.get_entry(name).pc_source != "missing", name


def test_a_copy_of_the_refined_dataset_with_its_own_pc_stops_the_chain():
    store = _missing_family()
    store.update_pc("c11", [0.47, 0.41, 0.58], source="manual")
    _write(store, name="c1")
    np.testing.assert_allclose(store.get_pc("c11"), [0.47, 0.41, 0.58], atol=1e-12)
    assert store.get_entry("c111").pc_source == "missing"       # follows c11, which is its own
    np.testing.assert_allclose(store.get_pc("c1"), NEW, atol=1e-12)


def test_following_family_of_a_refined_copy_names_its_descendants():
    store = _missing_family()
    _write(store, name="c1")
    assert sorted(store.following_family("c1")) == ["c11", "c111", "c2"]
