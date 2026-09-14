"""Tests for the streaming-path (index_h5) pseudo-symmetry resolution fix in
indexing_controller.

Bug: on a full-map spherical index of a clean signal, patterns are streamed from
the H5 via ``backend.index_h5`` and never materialised in RAM, so the Hough
pseudo-symmetry resolver + variant unification were silently skipped — a z_rot==2
intermetallic (alpha-AlFeSi m-3, S-phase mmm, ...) kept its WRONG spherical
pseudo-variant on EVERY pixel. The user could only trigger the fix by doing a
viewer edit (dirty signal) or indexing an ROI.

Fix: ``_materialize_patterns_for_resolution`` reads the pattern stack from the H5
on the streaming path IFF a spherical-unreliable phase actually won pixels, so the
auto-correction runs on a plain full-map index. These tests exercise the read
helper + the gating/fail-safe logic without a GPU or real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from indexing_controller import (  # noqa: E402
    _PatternStackTooLarge,
    _materialize_patterns_for_resolution,
    _read_h5_pattern_stack,
    _resolve_read_budget_bytes,
)


def _make_h5(path, ds_path, n=6, h=5, w=4, dtype=np.uint8):
    import h5py
    data = (np.arange(n * h * w).reshape(n, h, w) % 251).astype(dtype)
    with h5py.File(path, "w") as f:
        f.create_dataset(ds_path, data=data)
    return data


# ----------------------------------------------------------------------
# _read_h5_pattern_stack
# ----------------------------------------------------------------------
def test_read_full_stack_oxford(tmp_path):
    p = tmp_path / "oxford.h5oina"
    data = _make_h5(p, "1/EBSD/Data/Processed Patterns")
    out = _read_h5_pattern_stack(str(p))
    assert out.shape == data.shape
    assert np.array_equal(out, data)


def test_read_full_stack_edax_scan_path(tmp_path):
    """EDAX H5 stores patterns at <Scan>/EBSD/Data/Pattern — discovery must find
    it via the top-level-group walk, not just the fixed Oxford candidates."""
    p = tmp_path / "edax.h5"
    data = _make_h5(p, "Scan1/EBSD/Data/Pattern")
    out = _read_h5_pattern_stack(str(p))
    assert np.array_equal(out, data)


def test_read_by_indices_subset(tmp_path):
    p = tmp_path / "oxford.h5oina"
    data = _make_h5(p, "1/EBSD/Data/Processed Patterns", n=8)
    idx = np.array([0, 3, 5])
    out = _read_h5_pattern_stack(str(p), indices=idx)
    assert out.shape == (3, data.shape[1], data.shape[2])
    assert np.array_equal(out, data[idx])


def test_read_missing_dataset_raises(tmp_path):
    p = tmp_path / "nopatterns.h5"
    import h5py
    with h5py.File(p, "w") as f:
        f.create_dataset("1/EBSD/Data/SomethingElse", data=np.zeros((3, 2)))
    with pytest.raises(FileNotFoundError):
        _read_h5_pattern_stack(str(p))


def test_read_full_stack_too_large_raises(tmp_path):
    p = tmp_path / "oxford.h5oina"
    _make_h5(p, "1/EBSD/Data/Processed Patterns", n=6, h=5, w=4)  # 120 bytes uint8
    with pytest.raises(_PatternStackTooLarge) as ei:
        _read_h5_pattern_stack(str(p), max_bytes=10)
    assert ei.value.nbytes == 6 * 5 * 4  # reported before allocation


def test_read_by_indices_ignores_max_bytes(tmp_path):
    """The size cap is a full-read safety valve; an explicit (user-bounded) ROI
    read is never capped."""
    p = tmp_path / "oxford.h5oina"
    data = _make_h5(p, "1/EBSD/Data/Processed Patterns")
    out = _read_h5_pattern_stack(str(p), indices=np.array([1, 2]), max_bytes=1)
    assert np.array_equal(out, data[[1, 2]])


# ----------------------------------------------------------------------
# _materialize_patterns_for_resolution — gating + fail-safe
# ----------------------------------------------------------------------
def test_materialize_reads_when_zrot2_phase_present(tmp_path):
    p = tmp_path / "oxford.h5oina"
    data = _make_h5(p, "1/EBSD/Data/Processed Patterns", n=6)
    masters = [{"z_rot": 2, "point_group": "m-3", "formula": "AlMnFe"}]
    phase_id = np.ones(6, dtype=int)  # all pixels -> phase id 1 (the m-3 phase)
    out = _materialize_patterns_for_resolution(str(p), masters, phase_id)
    assert out is not None and np.array_equal(out, data)


def test_materialize_none_when_all_high_symmetry(tmp_path):
    """No z_rot==2 phase -> return None WITHOUT reading (a bad path must not even
    be opened, proving the gate short-circuits before any I/O)."""
    masters = [{"z_rot": 4, "point_group": "m-3m"}]
    phase_id = np.ones(6, dtype=int)
    out = _materialize_patterns_for_resolution(
        "this/path/does/not/exist.h5oina", masters, phase_id)
    assert out is None


def test_materialize_none_when_zrot2_phase_won_no_pixels(tmp_path):
    p = tmp_path / "oxford.h5oina"
    _make_h5(p, "1/EBSD/Data/Processed Patterns", n=6)
    # phase 1 = high-sym (won all pixels); phase 2 = m-3 (z_rot 2) but 0 pixels.
    masters = [{"z_rot": 4, "point_group": "m-3m"},
               {"z_rot": 2, "point_group": "m-3"}]
    phase_id = np.ones(6, dtype=int)   # only id 1 present
    out = _materialize_patterns_for_resolution(str(p), masters, phase_id)
    assert out is None


def test_materialize_multiphase_reads_when_one_zrot2_has_pixels(tmp_path):
    p = tmp_path / "oxford.h5oina"
    data = _make_h5(p, "1/EBSD/Data/Processed Patterns", n=6)
    masters = [{"z_rot": 4, "point_group": "m-3m"},
               {"z_rot": 2, "point_group": "m-3"}]
    phase_id = np.array([1, 1, 1, 2, 2, 2])  # phase 2 (m-3) won pixels 3..5
    out = _materialize_patterns_for_resolution(str(p), masters, phase_id)
    assert out is not None and np.array_equal(out, data)


def test_materialize_failsafe_on_bad_h5(tmp_path):
    """z_rot==2 phase present but the H5 read fails -> None (never raises, so a
    resolver problem can't break the indexing run)."""
    masters = [{"z_rot": 2, "point_group": "m-3"}]
    phase_id = np.ones(6, dtype=int)
    out = _materialize_patterns_for_resolution(
        str(tmp_path / "missing.h5oina"), masters, phase_id)
    assert out is None


def test_materialize_too_large_returns_none(tmp_path, monkeypatch):
    import indexing_controller as IC
    p = tmp_path / "oxford.h5oina"
    _make_h5(p, "1/EBSD/Data/Processed Patterns", n=6)
    masters = [{"z_rot": 2, "point_group": "m-3"}]
    phase_id = np.ones(6, dtype=int)
    monkeypatch.setattr(IC, "_resolve_read_budget_bytes", lambda: 1)
    msgs = []
    out = _materialize_patterns_for_resolution(
        str(p), masters, phase_id, progress=lambda m, pct=None: msgs.append(m))
    assert out is None
    assert any("exceeds" in m for m in msgs)  # honest skip message, not silent


def test_budget_is_positive():
    assert _resolve_read_budget_bytes() >= 1 * 1024 ** 3
