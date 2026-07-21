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
