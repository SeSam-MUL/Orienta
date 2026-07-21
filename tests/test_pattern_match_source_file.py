"""Regression test for the cross-file Pattern Match bug.

With two results from two different files in the gallery (e.g. an EDAX .up1
indexed live + an older rich .h5 result), the Pattern Match dialog used to
show the CURRENTLY-ACTIVE file's experimental patterns for *both* results —
because get_experimental_pattern fell straight back to the active signal for
Spherical/Hough results (which don't stash their own 'signal'). Symptoms:
wrong patterns + R=0.0, and the circular aperture silently dropped (the active
file's pattern shape didn't match the result's detector-derived aperture mask).

The fix reads a result's patterns from its OWN source_file when that differs
from the active file. These tests assert the correct pattern *shape* comes back
per result, which is exactly what distinguishes the two files here (79x79 vs
118x118) and what the aperture mask keys on.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
UP1 = ROOT / "Test_data" / "new test" / "Scan57.up1"          # 79x79
RICH = ROOT / "Test_data" / "result_Scan1_rich_2026-07-15.h5"  # 118x118


@pytest.fixture
def active_up1():
    """Make the .up1 the active signal (the bug context), restore after."""
    from safe_loader import load_ebsd_safe
    import backend.api.routes.ebsd_viewer as ev
    from tools import pattern_comparison as pc

    saved = (ev._ebsd_signal, ev._ebsd_file_path, ev._active_dataset,
             dict(ev._raw_signals))
    pc._source_signal_cache.clear()
    up1 = load_ebsd_safe(str(UP1), verbose=False)
    ev._ebsd_signal = up1
    ev._ebsd_file_path = str(UP1)
    ev._active_dataset = None
    ev._raw_signals.clear()
    try:
        yield ev
    finally:
        ev._ebsd_signal, ev._ebsd_file_path, ev._active_dataset, raw = saved
        ev._raw_signals.clear()
        ev._raw_signals.update(raw)
        pc._source_signal_cache.clear()


@pytest.mark.skipif(not (UP1.exists() and RICH.exists()),
                    reason="needs Scan57.up1 + rich .h5 in Test_data/")
def test_cross_file_result_reads_its_own_patterns(active_up1):
    from tools.pattern_comparison import get_experimental_pattern
    # Spherical/Hough result → no 'signal' in metadata, source is the rich .h5.
    res = types.SimpleNamespace(metadata={"source_file": str(RICH)})
    pat = get_experimental_pattern(res, 10, 10)
    assert pat is not None
    # MUST be the rich file's shape (118x118), not the active .up1 (79x79).
    assert pat.shape == (118, 118)


@pytest.mark.skipif(not UP1.exists(), reason="needs Scan57.up1")
def test_same_file_result_uses_active_signal(active_up1):
    from tools.pattern_comparison import get_experimental_pattern
    res = types.SimpleNamespace(metadata={"source_file": str(UP1)})
    pat = get_experimental_pattern(res, 10, 10)
    assert pat.shape == (79, 79)


@pytest.mark.skipif(not UP1.exists(), reason="needs Scan57.up1")
def test_legacy_result_without_source_file_falls_back_to_active(active_up1):
    from tools.pattern_comparison import get_experimental_pattern
    res = types.SimpleNamespace(metadata={})
    pat = get_experimental_pattern(res, 10, 10)
    assert pat.shape == (79, 79)


@pytest.mark.skipif(not UP1.exists(), reason="needs Scan57.up1")
def test_missing_cross_file_source_returns_none_not_active(active_up1):
    """Fail loud: a result from a DIFFERENT file whose source can't be loaded
    must return None — never the active file's (wrong) patterns."""
    from tools.pattern_comparison import get_experimental_pattern
    res = types.SimpleNamespace(
        metadata={"source_file": "Z:/does/not/exist/deleted_scan.h5oina"})
    assert get_experimental_pattern(res, 10, 10) is None


def test_store_result_preserves_preset_source_file():
    """An imported result tags its OWN source_file BEFORE _store_result; that
    must not be clobbered with whatever file is active at import time (the root
    cause of the cross-file Pattern Match bug)."""
    import types as _t
    import backend.api.routes.ebsd_viewer as ev
    import backend.api.routes.indexing as idx

    saved = ev._ebsd_file_path
    ev._ebsd_file_path = "C:/active/scan.up1"
    try:
        # imported result already carries its own file
        imp = _t.SimpleNamespace(metadata={"source_file": "D:/saved/result.h5",
                                            "imported_from_h5": True})
        idx._store_result(imp, "spherical")
        assert imp.metadata["source_file"] == "D:/saved/result.h5"

        # freshly-indexed result (no source_file) gets tagged with active file
        fresh = _t.SimpleNamespace(metadata={})
        idx._store_result(fresh, "hough")
        assert fresh.metadata["source_file"] == "C:/active/scan.up1"
    finally:
        ev._ebsd_file_path = saved
