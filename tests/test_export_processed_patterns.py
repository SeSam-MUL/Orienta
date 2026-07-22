"""Rich-.h5 export must store the PROCESSED patterns the user indexed on.

Before this, the rich export always shutil.copy2'd the raw source file, so
BG-removal / CLAHE / frame-average never made it into the saved patterns (and a
non-HDF5 EDAX .up1 source produced a corrupt .h5). Two helpers now fix this:

  * _overwrite_patterns_with_processed — swaps the pixel values of an HDF5
    export's pattern dataset for the processed signal (structure preserved).
  * _write_fresh_edax_h5 — builds a fresh EDAX-format .h5 from the signal for a
    non-HDF5 (.up1/.up2) source, so its rich export works at all.

These round-trip tests write, reload via the app's loader, and assert the
patterns that come back are the processed ones (bit-exact).
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
UP1 = ROOT / "Test_data" / "new test" / "Scan57.up1"
RICH = ROOT / "Test_data" / "result_Scan1_rich_2026-07-15.h5"


def _bg_removed(path):
    from safe_loader import load_ebsd_safe
    s = load_ebsd_safe(str(path), verbose=False)
    s.data = np.ascontiguousarray(s.data)
    s.remove_dynamic_background()
    return s


@pytest.mark.skipif(not RICH.exists(), reason="rich .h5 not present")
def test_overwrite_hdf5_patterns_with_processed(tmp_path):
    from backend.api.routes.indexing import _overwrite_patterns_with_processed
    from safe_loader import load_ebsd_safe

    out = tmp_path / "ov.h5"
    shutil.copy2(str(RICH), str(out))

    raw = load_ebsd_safe(str(RICH), verbose=False)
    raw_pat = np.asarray(raw.data[5, 5]).copy()
    proc = _bg_removed(RICH)
    proc_pat = np.asarray(proc.data[5, 5]).copy()

    assert _overwrite_patterns_with_processed(str(out), proc) is True

    back = load_ebsd_safe(str(out), verbose=False)
    got = np.asarray(back.data[5, 5])
    assert np.array_equal(got, proc_pat)      # the processed pattern
    assert not np.array_equal(got, raw_pat)   # not the raw one


@pytest.mark.skipif(not RICH.exists(), reason="rich .h5 not present")
def test_overwrite_skips_on_shape_mismatch(tmp_path):
    """Fail-soft: if the signal's patterns don't match the dataset shape, keep
    the raw copy rather than writing a wrong-shaped dataset."""
    from backend.api.routes.indexing import _overwrite_patterns_with_processed
    from safe_loader import load_ebsd_safe
    import kikuchipy as kp

    out = tmp_path / "ov2.h5"
    shutil.copy2(str(RICH), str(out))
    wrong = kp.signals.EBSD(np.zeros((3, 3, 10, 10), dtype=np.uint8))  # wrong shape
    assert _overwrite_patterns_with_processed(str(out), wrong) is False
    # file still loads (raw patterns intact)
    assert load_ebsd_safe(str(out), verbose=False) is not None


@pytest.mark.skipif(not UP1.exists(), reason="Scan57.up1 not present")
def test_fresh_edax_h5_from_up1_roundtrips_processed(tmp_path):
    from backend.api.routes.indexing import _write_fresh_edax_h5
    from safe_loader import load_ebsd_safe

    out = tmp_path / "fresh.h5"
    proc = _bg_removed(UP1)
    _write_fresh_edax_h5(str(out), proc)

    back = load_ebsd_safe(str(out), verbose=False)
    assert back.axes_manager.navigation_shape == (151, 151)
    assert back.axes_manager.signal_shape == (79, 79)
    assert back.data.dtype == np.uint8
    assert np.array_equal(np.asarray(proc.data[7, 7]), np.asarray(back.data[7, 7]))


def test_is_hdf5_file(tmp_path):
    from backend.api.routes.indexing import _is_hdf5_file
    if UP1.exists():
        assert _is_hdf5_file(str(UP1)) is False   # .up1 is not HDF5
    if RICH.exists():
        assert _is_hdf5_file(str(RICH)) is True
    assert _is_hdf5_file(str(tmp_path / "nope.h5")) is False


def test_dirty_flag_importable_from_ebsd_viewer():
    """The export wired is_active_signal_dirty from the wrong module
    (indexing_controller) → ImportError → it silently never wrote processed
    patterns. Guard the correct import location."""
    from backend.api.routes.ebsd_viewer import is_active_signal_dirty
    assert callable(is_active_signal_dirty)


def test_overwrite_targets_oxford_processed_patterns(tmp_path):
    """Oxford files name the dataset 'Processed Patterns' (+ 'Unprocessed
    Patterns'). The finder must hit 'Processed Patterns' (startswith('pattern')
    missed it) and leave the raw 'Unprocessed Patterns' untouched. Synthetic so
    it never depends on a real file that a user might overwrite."""
    import h5py
    import kikuchipy as kp
    from backend.api.routes.indexing import _overwrite_patterns_with_processed

    ny, nx, sy, sx = 4, 5, 8, 10          # 20 patterns
    proc = kp.signals.EBSD(np.random.default_rng(1).integers(
        0, 255, size=(ny, nx, sy, sx), dtype=np.uint8))

    out = tmp_path / "ox.h5"
    unproc_raw = np.arange(ny * nx * sy * sx, dtype=np.int16).reshape(ny * nx, sy, sx)
    with h5py.File(str(out), "w") as h:
        d = h.create_group("1/EBSD/Data")
        d.create_dataset("Processed Patterns",
                         data=np.zeros((ny * nx, sy, sx), np.uint8))
        d.create_dataset("Unprocessed Patterns", data=unproc_raw)  # raw, int16

    assert _overwrite_patterns_with_processed(str(out), proc) is True

    want = np.asarray(proc.data).reshape(ny * nx, sy, sx)
    with h5py.File(str(out), "r") as h:
        pp = h["1/EBSD/Data/Processed Patterns"]
        assert pp.attrs.get("patterns_processed")
        assert np.array_equal(np.asarray(pp), want)         # processed written
        # raw 'Unprocessed Patterns' untouched (and still int16)
        up = h["1/EBSD/Data/Unprocessed Patterns"]
        assert up.dtype == np.int16
        assert np.array_equal(np.asarray(up), unproc_raw)


INSERT = ROOT / "Test_data" / "Insert_Test_PatternSafe.h5"


def test_render_geometry_roundtrips(tmp_path):
    """A re-imported spherical result must carry detector_geometry +
    sht_paths_by_phase so the Pattern Match dialog can render simulated
    patterns. Export writes them as JSON on /Indexing; import reads them back
    (with int phase-id keys, and numpy scalars coerced to plain floats)."""
    import h5py
    import numpy as np
    from backend.api.routes.indexing import (
        _write_render_geometry_attrs, _restore_render_geometry)

    md = {
        "detector_geometry": {
            "pc_x": 0.55, "pc_y": np.float32(0.497), "pc_z": 0.702,
            "pat_width": 79, "pat_height": 79, "sample_tilt": 70.0,
            "tilt": 0.0, "vendor": "EDAX", "source_vendor": "edax",
            "pixel_size": 70.0, "binning": 1,
        },
        "sht_paths_by_phase": {1: "E:/x/Al (Al) [cF4] {20kV}.sht"},
    }
    out = tmp_path / "geom.h5"
    with h5py.File(str(out), "w") as f:
        _write_render_geometry_attrs(f.create_group("Indexing"), md)

    got = _restore_render_geometry(str(out))
    assert got["detector_geometry"]["pc_x"] == 0.55
    assert abs(got["detector_geometry"]["pc_y"] - 0.497) < 1e-4
    assert got["detector_geometry"]["vendor"] == "EDAX"
    assert got["sht_paths_by_phase"] == {1: "E:/x/Al (Al) [cF4] {20kV}.sht"}
    assert list(got["sht_paths_by_phase"].keys())[0] == 1  # int key, not "1"


def test_restore_render_geometry_missing_is_empty(tmp_path):
    """Older files without the attrs restore to {} (no crash)."""
    import h5py
    from backend.api.routes.indexing import _restore_render_geometry
    out = tmp_path / "noattr.h5"
    with h5py.File(str(out), "w") as f:
        f.create_group("Indexing")
    assert _restore_render_geometry(str(out)) == {}
    assert _restore_render_geometry(str(tmp_path / "nope.h5")) == {}


@pytest.mark.skipif(not INSERT.exists(), reason="insert export file not present")
def test_safe_loader_reads_export_with_extra_root_groups():
    """A rich .h5 export adds /Detector, /Documentation, /Indexing at root.
    kikuchipy's EDAX reader picks the alphabetically-first non-metadata group
    ('Detector') as the scan → KeyError. safe_loader must retry with the real
    EBSD scan group and load the patterns."""
    from safe_loader import load_ebsd_safe
    s = load_ebsd_safe(str(INSERT), verbose=False)
    assert len(s.axes_manager.navigation_shape) == 2
    assert s.data.dtype in (np.uint8, np.uint16)
    _ = np.asarray(s.data[0, 0])  # a pattern actually reads
