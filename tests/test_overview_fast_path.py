"""Tests for the /overview stored-quality-array fast path + subsampling cap.

Root cause (session 2026-06-01): on a lazy-loaded multi-GB H5OINA the overview
map reduced over EVERY pattern (one HDF5 chunk per pattern), taking minutes and
appearing to hang — the lazy-load work moved all pattern I/O from load-time to
first-overview-time. Fix: read Aztec's precomputed per-pixel quality arrays
(Band Contrast etc., ~3 ms) instead of the patterns, and cap the lazy compute
path for pattern-derived modes via navigation subsampling.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import ebsd_viewer as ev  # noqa: E402

SMALL_OXFORD_FILE = (
    Path(__file__).resolve().parents[1]
    / "Test_data"
    / "EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 "
      "Elementverteilungsdaten 1.h5oina"
)

# SampleB is a 90 (rows) x 120 (cols) scan.
N_ROWS, N_COLS = 90, 120


@pytest.mark.skipif(not SMALL_OXFORD_FILE.exists(), reason="SampleB H5OINA not present")
def test_stored_quality_map_band_contrast_shape_and_range():
    """Band Contrast must come back shaped to the nav grid with sane values —
    proving the fast path reads Aztec's stored array, not the patterns."""
    m = ev._read_stored_quality_map(str(SMALL_OXFORD_FILE), "bc", N_ROWS, N_COLS)
    assert m is not None, "expected stored Band Contrast array"
    assert m.shape == (N_ROWS, N_COLS)
    # Band Contrast is a uint8 metric → values within 0..255, non-constant.
    assert 0 <= m.min() <= m.max() <= 255
    assert m.max() > m.min()


@pytest.mark.skipif(not SMALL_OXFORD_FILE.exists(), reason="SampleB H5OINA not present")
def test_stored_quality_map_std_and_snr_have_arrays():
    """std and snr modes map to stored quality datasets too (Band Slope /
    Pattern Quality), so they also skip reading any patterns."""
    for mode in ("std", "snr"):
        m = ev._read_stored_quality_map(str(SMALL_OXFORD_FILE), mode, N_ROWS, N_COLS)
        assert m is not None and m.shape == (N_ROWS, N_COLS), f"mode {mode}"


def test_stored_quality_map_returns_none_for_pattern_modes():
    """mean/max/sharpness/entropy have no stored equivalent → None so the
    caller falls back to computing from patterns. (No file needed.)"""
    for mode in ("mean", "max", "sharpness", "entropy", "ncc"):
        assert ev._read_stored_quality_map("nonexistent.h5oina", mode, 4, 4) is None


def test_stored_quality_map_shape_mismatch_returns_none():
    """A grid that doesn't match the stored array size must yield None, not a
    reshape crash — the caller then computes from patterns instead."""
    if not SMALL_OXFORD_FILE.exists():
        pytest.skip("SampleB H5OINA not present")
    assert ev._read_stored_quality_map(str(SMALL_OXFORD_FILE), "bc", 7, 7) is None


def test_upscale_nearest_dims_and_corners():
    small = np.arange(12, dtype=float).reshape(3, 4)
    up = ev._upscale_nearest(small, 9, 12)
    assert up.shape == (9, 12)
    assert up[0, 0] == small[0, 0]
    assert up[-1, -1] == small[-1, -1]


def test_reduce_nav_modes_on_numpy_stack():
    """_reduce_nav collapses the signal axes to one value per scan pixel for
    each mode, on a plain numpy stack (the subsample path materialises numpy).

    'bc' is deliberately excluded: it is no longer a dedicated _reduce_nav
    branch (the overview handler serves it via the pattern-quality service)."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(5, 6, 8, 8)).astype(np.uint8)
    for mode in ("mean", "std", "max", "sharpness", "snr", "entropy"):
        nav = np.asarray(ev._reduce_nav(data, mode)).astype(float)
        assert nav.shape == (5, 6), f"mode {mode} produced {nav.shape}"
        assert np.isfinite(nav).all(), f"mode {mode} produced non-finite values"


def test_reduce_nav_has_no_bc_cov_mode():
    """The std/mean 'coefficient of variation' fake is retired: 'bc' is served
    by the pattern-quality service, not _reduce_nav. mean/std/max/sharpness/
    snr/entropy remain. 'bc' now falls through to the default (mean) inside
    _reduce_nav because the overview handler intercepts bc BEFORE calling it."""
    rng = np.random.default_rng(1)
    data = rng.integers(1, 255, size=(4, 3, 3, 3)).astype(np.float32)
    for m in ("mean", "std", "max", "sharpness", "snr", "entropy"):
        assert ev._reduce_nav(data, m) is not None
    # No dedicated CoV branch anymore → bc == the default mean reduction.
    bc = np.asarray(ev._reduce_nav(data, "bc")).astype(float)
    mean = np.asarray(ev._reduce_nav(data, "mean")).astype(float)
    np.testing.assert_allclose(bc, mean)
