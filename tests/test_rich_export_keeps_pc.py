"""A rich export of a file without an HDF5 source keeps its pattern centre.

The rich export of an EDAX UP1/UP2 source builds a fresh EDAX-format .h5
(``_write_fresh_edax_h5``). It wrote the patterns, the step and the sample tilt
but no ``Pattern Center Calibration`` and no camera tilt, so kikuchipy's EDAX
reader, loading the file again, set the PC to (0.5, 0.5, 0.5) and the camera
tilt to 0 without a word, although the export's own ``/Detector/pc`` carried the
right PC. Two fixes, one test group each:

* the header carries the PC (EDAX TSL convention) and the camera tilt;
* a file whose manufacturer header has no PC but whose ``/Detector/pc`` does (any
  export made before the first fix) is loaded with that PC.

The expected PC is the Bruker triple the detector was built with, on a NON-square
detector, so the TSL conversion (y* = 1 - PCy, z* scaled by the aspect) is on the
path both ways.
"""
import h5py
import numpy as np
import pytest

SHAPE = (60, 40)                    # rows, cols: not square, taller than wide
PC = (0.5241, 0.4133, 0.6317)       # Bruker, as the detector holds it
TILT = 7.5
SAMPLE_TILT = 71.0


def _signal(pc=PC):
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    rng = np.random.default_rng(3)
    sig = kp.signals.EBSD(rng.integers(0, 255, (3, 4) + SHAPE).astype(np.uint8))
    sig.detector = EBSDDetector(shape=SHAPE, pc=pc, sample_tilt=SAMPLE_TILT, tilt=TILT)
    return sig


def _load(path):
    from safe_loader import load_ebsd_safe
    return load_ebsd_safe(str(path), verbose=False)


def test_the_exported_header_carries_the_pc_and_the_camera_tilt(tmp_path):
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "fresh.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=PC)

    # the header, by hand: x* = PCx, y* = 1 - PCy, z* = PCz * rows / min(rows, cols)
    with h5py.File(out, "r") as f:
        g = f["Scan1/EBSD/Header/Pattern Center Calibration"]
        assert g["x-star"][0] == pytest.approx(PC[0])
        assert g["y-star"][0] == pytest.approx(1 - PC[1])
        assert g["z-star"][0] == pytest.approx(PC[2] * 60 / 40)
        assert f["Scan1/EBSD/Header/Camera Elevation Angle"][0] == pytest.approx(TILT)

    back = _load(out)
    np.testing.assert_allclose(np.asarray(back.detector.pc).ravel(), PC, atol=1e-6)
    assert back.detector.tilt == pytest.approx(TILT)
    assert back.detector.sample_tilt == pytest.approx(SAMPLE_TILT)


def test_the_exported_pc_is_the_one_given_not_the_signals(tmp_path):
    """A refined PC (calibration store) wins over what the signal's detector holds."""
    from backend.api.routes.indexing import _write_fresh_edax_h5
    refined = (0.5011, 0.3802, 0.6704)
    out = tmp_path / "fresh.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=refined)
    np.testing.assert_allclose(np.asarray(_load(out).detector.pc).ravel(), refined, atol=1e-6)


def test_no_pc_is_invented_when_none_is_known(tmp_path):
    """A file whose PC nobody supplied gets no header PC (a made-up one would be
    read back as a measurement)."""
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "fresh.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=None)
    with h5py.File(out, "r") as f:
        assert "Pattern Center Calibration" not in f["Scan1/EBSD/Header"]


# --- a file exported before: the header has no PC, /Detector/pc has ----------------

def _old_export(tmp_path, detector_pc=PC, detector_shape=SHAPE):
    """What the rich export wrote before the fix: no header PC, /Detector/pc present."""
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "old.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=None)
    with h5py.File(out, "a") as f:
        det = f.create_group("Detector")
        det.create_dataset("pc", data=np.asarray(detector_pc, dtype=float))
        det.attrs["sample_tilt"] = SAMPLE_TILT
        det.attrs["shape"] = list(detector_shape)
    return out


def _reader_default(tmp_path):
    """What kikuchipy's reader makes of an EDAX header without a PC: a file with no
    /Detector group at all (its (0.5, 0.5, 0.5) in the TSL convention, which is NOT
    0.5 everywhere in Bruker numbers for a detector that is not square)."""
    from backend.api.routes.indexing import _write_fresh_edax_h5
    plain = tmp_path / "plain.h5"
    _write_fresh_edax_h5(str(plain), _signal(), pc_bruker=None)
    return np.asarray(_load(plain).detector.pc).ravel()


def test_an_export_without_a_header_pc_is_loaded_with_its_detector_pc(tmp_path):
    back = _load(_old_export(tmp_path))
    np.testing.assert_allclose(np.asarray(back.detector.pc).ravel(), PC, atol=1e-6)
    assert back.metadata.get_item("Signal.pc_source") == "orienta_export"
    # the rest of the geometry is the file's own
    assert tuple(back.detector.shape) == SHAPE
    assert back.detector.sample_tilt == pytest.approx(SAMPLE_TILT)


def test_a_header_pc_wins_over_the_detector_group(tmp_path):
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "both.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=PC)
    with h5py.File(out, "a") as f:
        f.create_group("Detector").create_dataset("pc", data=np.array([0.1, 0.2, 0.3]))
    back = _load(out)
    np.testing.assert_allclose(np.asarray(back.detector.pc).ravel(), PC, atol=1e-6)


def test_a_detector_pc_of_another_pattern_size_is_not_applied(tmp_path):
    back = _load(_old_export(tmp_path, detector_shape=(60, 80)))
    # the shape in /Detector says it belongs to another detector: kikuchipy's default stays
    np.testing.assert_allclose(np.asarray(back.detector.pc).ravel(), _reader_default(tmp_path), atol=1e-9)


def test_an_unreadable_detector_pc_does_not_break_the_load(tmp_path):
    back = _load(_old_export(tmp_path, detector_pc=[np.nan, 0.4, 0.6]))
    np.testing.assert_allclose(np.asarray(back.detector.pc).ravel(), _reader_default(tmp_path), atol=1e-9)


def test_the_same_when_the_scan_group_sorts_before_detector(tmp_path):
    """kikuchipy's first attempt reads the alphabetically first group as the scan:
    it only succeeds when the scan group sorts before /Detector, a second route
    through the loader that must restore the PC as well."""
    out = _old_export(tmp_path)
    with h5py.File(out, "a") as f:
        f.move("Scan1", "Abc")
    back = _load(out)
    np.testing.assert_allclose(np.asarray(back.detector.pc).ravel(), PC, atol=1e-6)
    assert back.metadata.get_item("Signal.pc_source") == "orienta_export"


# --- "no PC" must not be laundered into a header PC by an export and a reload -------

def test_a_file_with_a_header_pc_is_not_tagged(tmp_path):
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "fresh.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=PC)
    assert _load(out).metadata.get_item("Signal.pc_source", None) is None


def test_an_export_of_a_file_without_a_pc_loads_as_having_none(tmp_path):
    """The export says pc_missing; the reload used to register kikuchipy's placeholder
    as a header PC."""
    from backend.api.routes.indexing import _write_fresh_edax_h5
    from backend.api.services.calibration_store import CalibrationStore
    out = tmp_path / "fresh.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=None)
    with h5py.File(out, "a") as f:
        f.create_group("Detector").attrs["pc_missing"] = True
    back = _load(out)
    assert back.metadata.get_item("Signal.pc_source") == "default"
    entry = CalibrationStore().register("again", back)
    assert entry.pc_source == "missing"


def test_a_foreign_edax_h5_without_a_pc_loads_as_having_none(tmp_path):
    """No /Detector group at all: an EDAX file whose header never had a PC."""
    from backend.api.services.calibration_store import CalibrationStore
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "foreign.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=None)
    back = _load(out)
    assert back.metadata.get_item("Signal.pc_source") == "default"
    assert CalibrationStore().register("f", back).pc_source == "missing"


def test_a_detector_pc_of_another_detector_leaves_the_file_without_a_pc(tmp_path):
    back = _load(_old_export(tmp_path, detector_shape=(60, 80)))
    assert back.metadata.get_item("Signal.pc_source") == "default"


def test_a_file_that_is_not_edax_is_left_alone(tmp_path):
    """Other formats keep their own way of carrying a PC: no EDAX manufacturer, no tag."""
    import safe_loader
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "other.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=None)
    with h5py.File(out, "a") as f:
        del f["Manufacturer"]
        f.create_dataset("Manufacturer", data=np.bytes_("Bruker"))
    sig = _signal()
    safe_loader._restore_pc_from_orienta_export(sig, str(out), verbose=False)
    assert sig.metadata.get_item("Signal.pc_source", None) is None


def test_pc_missing_wins_over_a_pc_dataset_beside_it(tmp_path):
    out = _old_export(tmp_path)                       # has /Detector/pc = PC
    with h5py.File(out, "a") as f:
        f["Detector"].attrs["pc_missing"] = True
    back = _load(out)
    assert back.metadata.get_item("Signal.pc_source") == "default"
    np.testing.assert_allclose(np.asarray(back.detector.pc).ravel(), _reader_default(tmp_path), atol=1e-9)


# --- an export whose /Detector/pc differs from its header: the header wins, and it is said ---

def test_a_detector_pc_that_differs_from_the_header_is_reported_not_applied(tmp_path):
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "differs.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=PC)
    with h5py.File(out, "a") as f:
        f.create_group("Detector").create_dataset("pc", data=np.array([0.51, 0.33, 0.84]))   # typed later
    back = _load(out)
    np.testing.assert_allclose(np.asarray(back.detector.pc).ravel(), PC, atol=1e-6)         # header wins
    assert back.metadata.get_item("Signal.export_pc_ignored") == pytest.approx([0.51, 0.33, 0.84])


def test_a_detector_pc_equal_to_the_header_says_nothing(tmp_path):
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "same.h5"
    _write_fresh_edax_h5(str(out), _signal(), pc_bruker=PC)
    with h5py.File(out, "a") as f:
        f.create_group("Detector").create_dataset("pc", data=np.array(PC))
    assert _load(out).metadata.get_item("Signal.export_pc_ignored", None) is None
