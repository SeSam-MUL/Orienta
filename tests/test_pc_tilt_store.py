"""Tilts changed on the PC Refinement page reach the calibration store.

The page edits the controller's own detector copy. Indexing builds its detector
from the calibration store (``get_detector``), so a sample or camera tilt changed
on the page - by typing it, or by Apply - never reached indexing: the page showed
one geometry and every run used another. The PC already went to the store (and to
the dataset's parent); the tilts did not.
"""
import asyncio

import numpy as np
import pytest
from kikuchipy.detectors import EBSDDetector

from backend.api.services.calibration_store import CalibrationStore
from pc_controller import PCController

SHAPE = (60, 80)


class _Det:
    def __init__(self, pc=(0.5, 0.4, 0.6), sample_tilt=70.0, tilt=0.0, azimuthal=0.0):
        self.shape = SHAPE
        self.pc = np.asarray(pc, dtype=float)
        self.sample_tilt, self.tilt, self.azimuthal = sample_tilt, tilt, azimuthal


class _Sig:
    def __init__(self, det):
        self.detector = det


@pytest.fixture
def page(monkeypatch):
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    store = CalibrationStore()
    store.register("raw", _Sig(_Det()))
    store.register_derived("raw_avg", "raw")
    ctrl = PCController()
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.5, 0.4, 0.6), sample_tilt=70.0))
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "raw_avg")
    return R, ctrl, store


def test_a_tilt_edit_reaches_the_store_and_the_parent(page):
    R, ctrl, store = page
    asyncio.run(R.update_tilt(R.UpdateTiltRequest(sample_tilt=72.5, detector_tilt=5.0, azimuthal=1.5)))
    for name in ("raw_avg", "raw"):                         # the active copy and its parent
        e = store.get_entry(name)
        assert (e.sample_tilt, e.tilt, e.azimuthal) == (72.5, 5.0, 1.5)
    # and what indexing reads, the store's detector, carries them
    det = store.get_detector("raw")
    assert (det.sample_tilt, det.tilt, det.azimuthal) == (72.5, 5.0, 1.5)


def test_a_partial_edit_leaves_the_other_tilts_alone(page):
    R, ctrl, store = page
    asyncio.run(R.update_tilt(R.UpdateTiltRequest(sample_tilt=68.0)))
    e = store.get_entry("raw_avg")
    assert (e.sample_tilt, e.tilt, e.azimuthal) == (68.0, 0.0, 0.0)


def test_apply_detector_writes_pc_and_tilts_to_the_store(page):
    R, ctrl, store = page
    res = asyncio.run(R.set_detector(R.SetDetectorRequest(
        shape=list(SHAPE), pc=[0.52, 0.41, 0.63], sample_tilt=71.0, camera_tilt=3.5, azimuthal=0.5)))
    e = store.get_entry("raw_avg")
    assert (e.sample_tilt, e.tilt, e.azimuthal) == (71.0, 3.5, 0.5)
    np.testing.assert_allclose(e.pc_single, [0.52, 0.41, 0.63], atol=1e-12)
    assert e.pc_source == "manual"
    assert res["pc_map_action"] == "none"
    # the parent has the same tilts and the PC
    p = store.get_entry("raw")
    assert (p.sample_tilt, p.tilt) == (71.0, 3.5)
    np.testing.assert_allclose(p.pc_single, [0.52, 0.41, 0.63], atol=1e-12)


def test_apply_on_the_placeholder_pc_does_not_make_it_a_real_one(monkeypatch):
    """A file without a PC stays "missing" when Apply is pressed to set a tilt."""
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    store = CalibrationStore()
    sig = _Sig(_Det(pc=(0.5, 0.5, 0.5)))
    sig.metadata = type("M", (), {"get_item": staticmethod(lambda k, d=None: "default")})()
    store.register("up1", sig)
    assert store.get_entry("up1").pc_source == "missing"
    ctrl = PCController()
    monkeypatch.setattr(R, "_get_controller", lambda: ctrl)
    monkeypatch.setattr(R, "calibration_store", store)
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "up1")

    asyncio.run(R.set_detector(R.SetDetectorRequest(
        shape=list(SHAPE), pc=[0.5, 0.5, 0.5], sample_tilt=72.0)))
    e = store.get_entry("up1")
    assert e.pc_source == "missing" and e.sample_tilt == 72.0       # tilt written, PC still nobody's
    # a PC that IS different ends the missing state
    asyncio.run(R.set_detector(R.SetDetectorRequest(
        shape=list(SHAPE), pc=[0.51, 0.49, 0.61], sample_tilt=72.0)))
    assert store.get_entry("up1").pc_source == "manual"


def test_apply_over_a_grid_map_replaces_it_in_the_controller_and_the_store(page):
    R, ctrl, store = page
    rows, cols = np.mgrid[0:3, 0:4]
    plane = np.stack([0.5 + 1e-3 * cols, 0.4 + 0 * rows, 0.6 + 0 * rows], axis=-1)
    store.update_pc_map("raw_avg", plane)
    ctrl.apply_pixelwise_pc(EBSDDetector(shape=SHAPE, pc=plane, sample_tilt=70.0))
    res = asyncio.run(R.set_detector(R.SetDetectorRequest(shape=list(SHAPE), pc=[0.52, 0.41, 0.63])))
    assert res["pc_map_action"] == "replaced"
    assert store.get_entry("raw_avg").pc_map is None
    assert np.asarray(ctrl.detector.pc).reshape(-1, 3).shape[0] == 1


def test_update_geometry_of_an_unknown_dataset_is_refused_quietly():
    assert CalibrationStore().update_geometry("ghost", sample_tilt=70.0) is False


def _grid_page(page):
    R, ctrl, store = page
    rows, cols = np.mgrid[0:3, 0:4]
    plane = np.stack([0.5 + 1e-3 * cols, 0.4 + 2e-3 * rows, 0.6 + 0 * rows], axis=-1)
    store.update_pc_map("raw_avg", plane)
    ctrl.apply_pixelwise_pc(EBSDDetector(shape=SHAPE, pc=plane, sample_tilt=70.0))
    return R, ctrl, store, plane, plane.reshape(-1, 3).mean(axis=0)


def test_apply_for_a_tilt_keeps_a_pc_map_when_the_pc_is_the_one_shown(page):
    """The PC fields show the MEAN of the map; Apply with that same PC (pressed to
    set a tilt) used to turn the whole map into that one value."""
    R, ctrl, store, plane, mean = _grid_page(page)
    res = asyncio.run(R.set_detector(R.SetDetectorRequest(
        shape=list(SHAPE), pc=[float(v) for v in mean], sample_tilt=72.5, camera_tilt=3.0)))
    assert res["pc_map_action"] == "none"
    moved = np.asarray(ctrl.detector.pc)
    assert moved.shape == plane.shape
    np.testing.assert_allclose(moved, plane, atol=1e-12)           # the map, untouched
    assert (ctrl.detector.sample_tilt, ctrl.detector.tilt) == (72.5, 3.0)
    e = store.get_entry("raw_avg")
    np.testing.assert_allclose(e.pc_map, plane, atol=1e-12)
    assert e.pc_source == "refined_map"
    assert (e.sample_tilt, e.tilt) == (72.5, 3.0)


def test_apply_for_the_binning_keeps_the_map_too(page):
    R, ctrl, store, plane, mean = _grid_page(page)
    res = asyncio.run(R.set_detector(R.SetDetectorRequest(
        shape=list(SHAPE), pc=[float(v) for v in mean], binning=4, pixel_size=70.0)))
    assert res["pc_map_action"] == "none"
    np.testing.assert_allclose(np.asarray(ctrl.detector.pc), plane, atol=1e-12)
    assert ctrl.detector.binning == 4
    assert ctrl.detector.px_size == pytest.approx(17.5)


def test_apply_with_a_changed_pc_still_replaces_the_map(page):
    R, ctrl, store, plane, mean = _grid_page(page)
    res = asyncio.run(R.set_detector(R.SetDetectorRequest(
        shape=list(SHAPE), pc=[float(mean[0]) + 0.01, float(mean[1]), float(mean[2])])))
    assert res["pc_map_action"] == "replaced"
    assert store.get_entry("raw_avg").pc_map is None
    assert np.asarray(ctrl.detector.pc).reshape(-1, 3).shape[0] == 1


def test_apply_on_the_pc_a_header_entry_already_has_does_not_make_it_manual(page):
    R, ctrl, store = page
    asyncio.run(R.set_detector(R.SetDetectorRequest(
        shape=list(SHAPE), pc=[0.5, 0.4, 0.6], sample_tilt=71.0)))
    assert store.get_entry("raw_avg").pc_source == "inherited"
