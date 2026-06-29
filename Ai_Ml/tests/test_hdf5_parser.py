"""Tests for ebsd_ai.data.hdf5_parser."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ebsd_ai.data.hdf5_parser import (
    _detect_subformat,
    _quaternion_to_euler,
    parse_emsoft,
    parse_h5oina,
    parse_hdf5,
    parse_kikuchipy,
)
from ebsd_ai.data.scan_import import ScanFormat

# ---------------------------------------------------------------------------
# Helpers — synthetic HDF5 file writers
# ---------------------------------------------------------------------------

_N_ROWS = 5
_N_COLS = 10
_N = _N_ROWS * _N_COLS


def _make_euler(n: int = _N) -> np.ndarray:
    """Generate random Euler angles in radians (N, 3)."""
    rng = np.random.default_rng(42)
    phi1 = rng.uniform(0, 2 * np.pi, n)
    big_phi = rng.uniform(0, np.pi, n)
    phi2 = rng.uniform(0, 2 * np.pi, n)
    return np.stack([phi1, big_phi, phi2], axis=-1).astype(np.float32)


def _make_phase_ids(n: int = _N, n_phases: int = 2) -> np.ndarray:
    """Generate random phase IDs (0 = not indexed, 1..n_phases)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, n_phases + 1, size=n).astype(np.uint8)


def _write_h5oina(
    path: Path,
    *,
    n_rows: int = _N_ROWS,
    n_cols: int = _N_COLS,
    with_patterns: bool = False,
    with_mad: bool = True,
    with_bc: bool = True,
    phase_names: list[str] | None = None,
) -> None:
    """Create a synthetic H5OINA file."""
    n = n_rows * n_cols
    euler = _make_euler(n)
    phase_ids = _make_phase_ids(n)
    if phase_names is None:
        phase_names = ["Ferrite", "Austenite"]

    with h5py.File(path, "w") as f:
        ebsd = f.create_group("1/EBSD")
        header = ebsd.create_group("Header")
        data = ebsd.create_group("Data")

        header.create_dataset("X Cells", data=n_cols)
        header.create_dataset("Y Cells", data=n_rows)
        header.create_dataset("X Step", data=0.5)
        header.create_dataset("Y Step", data=0.5)

        phases_grp = header.create_group("Phases")
        for i, name in enumerate(phase_names, start=1):
            pg = phases_grp.create_group(str(i))
            pg.create_dataset("Name", data=name.encode())

        data.create_dataset("Euler", data=euler)
        data.create_dataset("Phase", data=phase_ids)

        if with_mad:
            rng = np.random.default_rng(99)
            mad = rng.uniform(0.1, 2.0, n).astype(np.float32)
            data.create_dataset("Mean Angular Deviation", data=mad)

        if with_bc:
            rng = np.random.default_rng(99)
            bc = rng.integers(50, 250, n).astype(np.uint8)
            data.create_dataset("Band Contrast", data=bc)

        if with_patterns:
            rng = np.random.default_rng(99)
            pats = rng.integers(0, 255, (n_rows, n_cols, 60, 60), dtype=np.uint8)
            data.create_dataset("Processed Patterns", data=pats)


def _write_kikuchipy(
    path: Path,
    *,
    n_rows: int = _N_ROWS,
    n_cols: int = _N_COLS,
    with_patterns: bool = True,
    with_crystal_map: bool = False,
    phase_names: list[str] | None = None,
) -> None:
    """Create a synthetic kikuchipy file."""
    n = n_rows * n_cols
    if phase_names is None:
        phase_names = ["Silicon"]

    with h5py.File(path, "w") as f:
        scan = f.create_group("Scan 1")
        ebsd = scan.create_group("EBSD")
        header = ebsd.create_group("Header")
        data = ebsd.create_group("Data")

        header.create_dataset("Number of patterns", data=[n_rows, n_cols])
        header.create_dataset("Step sizes", data=[0.3, 0.3])
        header.create_dataset("Pattern height", data=64)
        header.create_dataset("Pattern width", data=64)

        phases_grp = header.create_group("Phases")
        for i, name in enumerate(phase_names, start=1):
            pg = phases_grp.create_group(str(i))
            pg.create_dataset("name", data=name.encode())

        if with_patterns:
            rng = np.random.default_rng(42)
            pats = rng.integers(0, 255, (n_rows, n_cols, 64, 64), dtype=np.uint8)
            data.create_dataset("patterns", data=pats)

        if with_crystal_map:
            cm = scan.create_group("crystal_map")
            # Create quaternions from Euler angles via forward transform
            euler = _make_euler(n)
            quats = _euler_to_quat_simple(euler)
            cm.create_dataset("rotations", data=quats)
            cm.create_dataset("phase_id", data=_make_phase_ids(n, len(phase_names)))

            props = cm.create_group("properties")
            rng = np.random.default_rng(42)
            props.create_dataset("dp", data=rng.uniform(0.5, 1.0, n).astype(np.float32))
            iq = rng.uniform(100, 1000, n).astype(np.float32)
            props.create_dataset("iq", data=iq)


def _write_emsoft(
    path: Path,
    *,
    n_rows: int = _N_ROWS,
    n_cols: int = _N_COLS,
    with_patterns: bool = False,
    with_ci: bool = True,
    master_only: bool = False,
) -> None:
    """Create a synthetic EMsoft file."""
    n = n_rows * n_cols

    with h5py.File(path, "w") as f:
        # Crystal data
        crystal = f.create_group("CrystalData")
        crystal.create_dataset("SpaceGroupNumber", data=225)
        crystal.create_dataset(
            "LatticeParameters",
            data=[3.56, 3.56, 3.56, 90.0, 90.0, 90.0],
        )
        crystal.create_dataset("PhaseName", data=b"Copper")

        emdata = f.create_group("EMData")

        if master_only:
            master = emdata.create_group("EBSDmaster")
            rng = np.random.default_rng(42)
            master.create_dataset(
                "masterSPNH", data=rng.random((201, 201), dtype=np.float32)
            )
            master.create_dataset(
                "masterSPSH", data=rng.random((201, 201), dtype=np.float32)
            )
        else:
            ebsd = emdata.create_group("EBSD")
            euler = _make_euler(n).reshape(n_rows, n_cols, 3)
            ebsd.create_dataset("EulerAngles", data=euler)

            phase_ids = _make_phase_ids(n).reshape(n_rows, n_cols)
            ebsd.create_dataset("Phase", data=phase_ids)

            if with_ci:
                rng = np.random.default_rng(42)
                ci = rng.uniform(0.1, 0.9, (n_rows, n_cols)).astype(np.float32)
                ebsd.create_dataset("CI", data=ci)

            if with_patterns:
                rng = np.random.default_rng(42)
                pats = rng.integers(0, 255, (n, 60, 60), dtype=np.uint8)
                ebsd.create_dataset("EBSDPatterns", data=pats)

        # NML parameters
        nml = f.create_group("NMLparameters")
        idx_nml = nml.create_group("EBSDIndexingNameListType")
        idx_nml.create_dataset("step_x", data=0.4)
        idx_nml.create_dataset("step_y", data=0.4)
        idx_nml.create_dataset("ipf_wd", data=n_cols)
        idx_nml.create_dataset("ipf_ht", data=n_rows)


def _euler_to_quat_simple(euler: np.ndarray) -> np.ndarray:
    """Simple Euler → quaternion for test data generation."""
    phi1 = euler[:, 0].astype(np.float64)
    big_phi = euler[:, 1].astype(np.float64)
    phi2 = euler[:, 2].astype(np.float64)

    sigma = (phi1 + phi2) / 2.0
    delta = (phi1 - phi2) / 2.0
    c_phi = np.cos(big_phi / 2.0)
    s_phi = np.sin(big_phi / 2.0)

    w = c_phi * np.cos(sigma)
    x = s_phi * np.cos(delta)
    y = s_phi * np.sin(delta)
    z = c_phi * np.sin(sigma)

    quats = np.stack([w, x, y, z], axis=-1)
    neg = quats[:, 0] < 0
    quats[neg] *= -1.0
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (quats / norms).astype(np.float64)


# ===========================================================================
# Tests: format detection
# ===========================================================================


class TestDetectSubformat:
    """Tests for _detect_subformat (in-memory detection)."""

    def test_detects_h5oina(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p)
        with h5py.File(p, "r") as f:
            assert _detect_subformat(f) == ScanFormat.H5OINA

    def test_detects_kikuchipy(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p)
        with h5py.File(p, "r") as f:
            assert _detect_subformat(f) == ScanFormat.KIKUCHIPY

    def test_detects_emsoft(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_emsoft(p)
        with h5py.File(p, "r") as f:
            assert _detect_subformat(f) == ScanFormat.EMSOFT

    def test_unknown_for_empty_hdf5(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.h5"
        with h5py.File(p, "w"):
            pass
        with h5py.File(p, "r") as f:
            assert _detect_subformat(f) == ScanFormat.UNKNOWN


# ===========================================================================
# Tests: parse_hdf5 (auto-detect)
# ===========================================================================


class TestParseHdf5:
    """Tests for the top-level parse_hdf5 auto-detect function."""

    def test_auto_detects_h5oina(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p)
        sd = parse_hdf5(p)
        assert sd.source_format == ScanFormat.H5OINA

    def test_auto_detects_kikuchipy(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p)
        sd = parse_hdf5(p)
        assert sd.source_format == ScanFormat.KIKUCHIPY

    def test_auto_detects_emsoft(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_emsoft(p)
        sd = parse_hdf5(p)
        assert sd.source_format == ScanFormat.EMSOFT

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_hdf5("/nonexistent/scan.h5")

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "weird.h5"
        with h5py.File(p, "w") as f:
            f.create_group("RandomGroup")
        with pytest.raises(ValueError, match="Cannot determine"):
            parse_hdf5(p)


# ===========================================================================
# Tests: H5OINA parser
# ===========================================================================


class TestParseH5oina:
    """Tests for the H5OINA parser."""

    def test_basic_parsing(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p)
        sd = parse_h5oina(p)

        assert sd.source_format == ScanFormat.H5OINA
        assert sd.n_points == _N
        assert sd.euler_angles.shape == (_N, 3)
        assert sd.phase_ids.shape == (_N,)
        assert sd.confidence_scores.shape == (_N,)
        assert sd.source_file == str(p)

    def test_phase_names(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p, phase_names=["Iron", "Nickel", "Chromium"])
        sd = parse_h5oina(p)

        assert sd.phase_names == [
            "Not indexed", "Iron", "Nickel", "Chromium"
        ]

    def test_euler_angles_radians(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p)
        sd = parse_h5oina(p)

        # Euler angles should be in radians (original data is already radians)
        assert sd.euler_angles.dtype == np.float64
        assert np.all(sd.euler_angles[:, 0] >= 0)
        assert np.all(sd.euler_angles[:, 0] < 2 * np.pi + 0.1)

    def test_mad_to_confidence(self, tmp_path: Path) -> None:
        """MAD should be converted to confidence via exp(-MAD)."""
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p, with_mad=True, with_bc=False)
        sd = parse_h5oina(p)

        assert sd.confidence_scores.dtype == np.float32
        assert np.all(sd.confidence_scores >= 0)
        assert np.all(sd.confidence_scores <= 1)
        assert "mad" in sd.metadata

    def test_bc_fallback_confidence(self, tmp_path: Path) -> None:
        """Without MAD, Band Contrast / 255 should be used."""
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p, with_mad=False, with_bc=True)
        sd = parse_h5oina(p)

        assert sd.confidence_scores.dtype == np.float32
        assert np.all(sd.confidence_scores >= 0)
        assert np.all(sd.confidence_scores <= 1)
        assert "band_contrast" in sd.metadata

    def test_grid_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p, n_rows=8, n_cols=12)
        sd = parse_h5oina(p)

        assert sd.scan_shape == (8, 12)

    def test_step_sizes(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p)
        sd = parse_h5oina(p)

        assert sd.step_sizes == (0.5, 0.5)

    def test_with_patterns(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p, with_patterns=True)
        sd = parse_h5oina(p)

        assert sd.has_patterns
        assert sd.patterns is not None
        assert sd.patterns.shape == (_N, 60, 60)
        assert sd.patterns.dtype == np.uint8

    def test_without_patterns(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p, with_patterns=False)
        sd = parse_h5oina(p)

        assert not sd.has_patterns
        assert sd.patterns is None

    def test_validate_passes(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p)
        sd = parse_h5oina(p)
        assert sd.validate() == []

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_h5oina("/nonexistent/scan.h5oina")

    def test_no_data_group(self, tmp_path: Path) -> None:
        """File with EBSD group but no Data should raise."""
        p = tmp_path / "bad.h5oina"
        with h5py.File(p, "w") as f:
            ebsd = f.create_group("1/EBSD")
            ebsd.create_group("Header")
            # No Data group
        with pytest.raises(ValueError, match="No EBSD/Data"):
            parse_h5oina(p)


# ===========================================================================
# Tests: kikuchipy parser
# ===========================================================================


class TestParseKikuchipy:
    """Tests for the kikuchipy parser."""

    def test_basic_with_patterns(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p, with_patterns=True)
        sd = parse_kikuchipy(p)

        assert sd.source_format == ScanFormat.KIKUCHIPY
        assert sd.n_points == _N
        assert sd.has_patterns
        assert sd.patterns is not None
        assert sd.patterns.shape == (_N, 64, 64)

    def test_phase_names(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p, phase_names=["GaAs", "InP"])
        sd = parse_kikuchipy(p)

        assert sd.phase_names == ["Not indexed", "GaAs", "InP"]

    def test_crystal_map_orientations(self, tmp_path: Path) -> None:
        """Crystal map with quaternions should be converted to Euler."""
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p, with_crystal_map=True)
        sd = parse_kikuchipy(p)

        assert sd.euler_angles.shape == (_N, 3)
        assert sd.euler_angles.dtype == np.float64
        # Euler angles should be non-negative
        assert np.all(sd.euler_angles >= 0)

    def test_crystal_map_confidence(self, tmp_path: Path) -> None:
        """Dot product from crystal map should become confidence."""
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p, with_crystal_map=True)
        sd = parse_kikuchipy(p)

        assert sd.confidence_scores.dtype == np.float32
        assert np.all(sd.confidence_scores >= 0)
        assert np.all(sd.confidence_scores <= 1)

    def test_grid_shape_from_header(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p, n_rows=4, n_cols=8)
        sd = parse_kikuchipy(p)

        assert sd.scan_shape == (4, 8)

    def test_step_sizes(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p)
        sd = parse_kikuchipy(p)

        assert sd.step_sizes == (0.3, 0.3)

    def test_without_crystal_map(self, tmp_path: Path) -> None:
        """Without crystal_map, Euler should be zeros."""
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p, with_crystal_map=False)
        sd = parse_kikuchipy(p)

        assert np.all(sd.euler_angles == 0)

    def test_validate_passes(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p, with_crystal_map=True)
        sd = parse_kikuchipy(p)
        assert sd.validate() == []

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_kikuchipy("/nonexistent/scan.h5")

    def test_no_scan_group(self, tmp_path: Path) -> None:
        """File without Scan N group should raise."""
        p = tmp_path / "bad.h5"
        with h5py.File(p, "w") as f:
            f.create_group("Something")
        with pytest.raises(ValueError, match="No 'Scan N' group"):
            parse_kikuchipy(p)


# ===========================================================================
# Tests: EMsoft parser
# ===========================================================================


class TestParseEmsoft:
    """Tests for the EMsoft parser."""

    def test_basic_indexing(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_emsoft(p)
        sd = parse_emsoft(p)

        assert sd.source_format == ScanFormat.EMSOFT
        assert sd.n_points == _N
        assert sd.euler_angles.shape == (_N, 3)
        assert sd.phase_ids.shape == (_N,)

    def test_phase_names_from_crystal(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_emsoft(p)
        sd = parse_emsoft(p)

        assert "Copper" in sd.phase_names
        assert sd.phase_names[0] == "Not indexed"

    def test_euler_angles_from_3d_reshape(self, tmp_path: Path) -> None:
        """EMsoft stores Euler as (rows, cols, 3) → should reshape to (N, 3)."""
        p = tmp_path / "scan.h5"
        _write_emsoft(p)
        sd = parse_emsoft(p)

        assert sd.euler_angles.dtype == np.float64
        assert sd.euler_angles.shape == (_N, 3)

    def test_ci_confidence(self, tmp_path: Path) -> None:
        """CI should become confidence scores."""
        p = tmp_path / "scan.h5"
        _write_emsoft(p, with_ci=True)
        sd = parse_emsoft(p)

        assert sd.confidence_scores.dtype == np.float32
        assert np.all(sd.confidence_scores >= 0)
        assert np.all(sd.confidence_scores <= 1)
        assert "ci" in sd.metadata

    def test_grid_shape_from_nml(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_emsoft(p, n_rows=6, n_cols=8)
        sd = parse_emsoft(p)

        assert sd.scan_shape == (6, 8)

    def test_step_sizes(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_emsoft(p)
        sd = parse_emsoft(p)

        assert sd.step_sizes == (0.4, 0.4)

    def test_with_patterns(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_emsoft(p, with_patterns=True)
        sd = parse_emsoft(p)

        assert sd.has_patterns
        assert sd.patterns is not None
        assert sd.patterns.shape == (_N, 60, 60)

    def test_master_pattern_only(self, tmp_path: Path) -> None:
        """Master-pattern-only file should parse without errors."""
        p = tmp_path / "master.h5"
        _write_emsoft(p, master_only=True)
        sd = parse_emsoft(p)

        assert sd.n_points == 0
        assert sd.metadata.get("file_type") == "master_pattern"

    def test_validate_passes(self, tmp_path: Path) -> None:
        p = tmp_path / "scan.h5"
        _write_emsoft(p)
        sd = parse_emsoft(p)
        assert sd.validate() == []

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_emsoft("/nonexistent/scan.h5")

    def test_no_emdata_group(self, tmp_path: Path) -> None:
        """File without EMData should raise."""
        p = tmp_path / "bad.h5"
        with h5py.File(p, "w") as f:
            f.create_group("NMLparameters")
        with pytest.raises(ValueError, match="No EMData"):
            parse_emsoft(p)

    def test_scan_shape_from_phase_2d(self, tmp_path: Path) -> None:
        """Grid shape inferred from 2D Phase array when NML is absent."""
        p = tmp_path / "scan.h5"
        n_rows, n_cols = 3, 7
        n = n_rows * n_cols
        with h5py.File(p, "w") as f:
            f.create_group("CrystalData")
            ebsd = f.create_group("EMData/EBSD")
            ebsd.create_dataset(
                "EulerAngles",
                data=_make_euler(n).reshape(n_rows, n_cols, 3),
            )
            ebsd.create_dataset(
                "Phase",
                data=np.ones((n_rows, n_cols), dtype=np.uint8),
            )
        sd = parse_emsoft(p)
        assert sd.scan_shape == (n_rows, n_cols)


# ===========================================================================
# Tests: quaternion ↔ euler roundtrip
# ===========================================================================


class TestQuaternionToEuler:
    """Tests for the internal _quaternion_to_euler function."""

    def test_identity_quaternion(self) -> None:
        """Identity quaternion should give Euler (0, 0, 0)."""
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        euler = _quaternion_to_euler(q)
        assert euler.shape == (1, 3)
        np.testing.assert_allclose(euler[0], [0.0, 0.0, 0.0], atol=1e-10)

    def test_roundtrip_single(self) -> None:
        """Euler → quat → Euler should recover original angles."""
        from ebsd_ai.data.scan_import import euler_to_quaternion

        original = np.array([1.0, 0.8, 2.0])
        quat = euler_to_quaternion(original)
        recovered = _quaternion_to_euler(quat.reshape(1, 4))[0]

        # Angles wrap modulo 2π
        np.testing.assert_allclose(
            np.cos(original), np.cos(recovered), atol=1e-10
        )
        np.testing.assert_allclose(
            np.sin(original), np.sin(recovered), atol=1e-10
        )

    def test_roundtrip_batch(self) -> None:
        """Batch Euler → quat → Euler roundtrip."""
        from ebsd_ai.data.scan_import import euler_to_quaternion

        rng = np.random.default_rng(123)
        original = np.column_stack([
            rng.uniform(0, 2 * np.pi, 100),
            rng.uniform(0, np.pi, 100),
            rng.uniform(0, 2 * np.pi, 100),
        ])

        quats = euler_to_quaternion(original)
        recovered = _quaternion_to_euler(quats)

        np.testing.assert_allclose(
            np.cos(original), np.cos(recovered), atol=1e-10
        )

    def test_output_shape(self) -> None:
        quats = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ])
        euler = _quaternion_to_euler(quats)
        assert euler.shape == (2, 3)

    def test_single_input_still_2d(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        euler = _quaternion_to_euler(q)
        assert euler.shape == (1, 3)


# ===========================================================================
# Tests: edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for HDF5 parsers."""

    def test_h5oina_no_quality_metrics(self, tmp_path: Path) -> None:
        """H5OINA with neither MAD nor BC should give zero confidence."""
        p = tmp_path / "scan.h5oina"
        _write_h5oina(p, with_mad=False, with_bc=False)
        sd = parse_h5oina(p)

        assert np.all(sd.confidence_scores == 0)

    def test_h5oina_flat_euler(self, tmp_path: Path) -> None:
        """H5OINA with flat (1D) Euler array should still parse."""
        p = tmp_path / "scan.h5oina"
        with h5py.File(p, "w") as f:
            ebsd = f.create_group("1/EBSD")
            ebsd.create_group("Header")
            data = ebsd.create_group("Data")
            # Flat 1D array: 6 values → reshape to (2, 3)
            data.create_dataset(
                "Euler", data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            )
            data.create_dataset("Phase", data=np.array([1, 1]))

        sd = parse_h5oina(p)
        assert sd.euler_angles.shape == (2, 3)

    def test_emsoft_without_ci(self, tmp_path: Path) -> None:
        """EMsoft without CI should give zero confidence."""
        p = tmp_path / "scan.h5"
        _write_emsoft(p, with_ci=False)
        sd = parse_emsoft(p)
        assert np.all(sd.confidence_scores == 0)

    def test_emsoft_with_iq_fallback(self, tmp_path: Path) -> None:
        """EMsoft with IQ but no CI should normalize IQ as confidence."""
        p = tmp_path / "scan.h5"
        n_rows, n_cols = 3, 4
        n = n_rows * n_cols
        with h5py.File(p, "w") as f:
            f.create_group("CrystalData")
            ebsd = f.create_group("EMData/EBSD")
            ebsd.create_dataset(
                "EulerAngles", data=_make_euler(n).reshape(n_rows, n_cols, 3)
            )
            ebsd.create_dataset(
                "Phase", data=np.ones((n_rows, n_cols), dtype=np.uint8)
            )
            iq_values = np.linspace(100, 500, n).reshape(n_rows, n_cols)
            ebsd.create_dataset("IQ", data=iq_values.astype(np.float32))

        sd = parse_emsoft(p)
        assert sd.confidence_scores.dtype == np.float32
        # Max IQ → confidence = 1.0
        assert np.isclose(sd.confidence_scores.max(), 1.0, atol=0.01)
        assert "image_quality" in sd.metadata

    def test_kikuchipy_patterns_only(self, tmp_path: Path) -> None:
        """kikuchipy file with only patterns (no crystal_map)."""
        p = tmp_path / "scan.h5"
        _write_kikuchipy(p, with_patterns=True, with_crystal_map=False)
        sd = parse_kikuchipy(p)

        assert sd.has_patterns
        assert np.all(sd.euler_angles == 0)
        assert np.all(sd.phase_ids == 0)

    def test_kikuchipy_crystal_map_iq_fallback(self, tmp_path: Path) -> None:
        """IQ used as confidence when dp is not available."""
        p = tmp_path / "scan.h5"
        n = _N
        with h5py.File(p, "w") as f:
            scan = f.create_group("Scan 1")
            ebsd = scan.create_group("EBSD")
            header = ebsd.create_group("Header")
            header.create_dataset("Number of patterns", data=[_N_ROWS, _N_COLS])
            header.create_dataset("Step sizes", data=[0.5, 0.5])
            ebsd.create_group("Data")
            phases_grp = header.create_group("Phases")
            pg = phases_grp.create_group("1")
            pg.create_dataset("name", data=b"Test")

            cm = scan.create_group("crystal_map")
            cm.create_dataset(
                "rotations",
                data=_euler_to_quat_simple(_make_euler(n)),
            )
            cm.create_dataset(
                "phase_id", data=np.ones(n, dtype=np.uint8)
            )
            props = cm.create_group("properties")
            rng = np.random.default_rng(42)
            props.create_dataset(
                "iq", data=rng.uniform(100, 1000, n).astype(np.float32)
            )

        sd = parse_kikuchipy(p)
        # IQ normalized to [0, 1]
        assert sd.confidence_scores.max() <= 1.0
        assert sd.confidence_scores.min() >= 0.0
        assert "image_quality" in sd.metadata
