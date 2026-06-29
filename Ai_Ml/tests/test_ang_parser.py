"""Tests for the ANG file parser."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from ebsd_ai.data.ang_parser import parse_ang
from ebsd_ai.data.scan_import import ScanFormat


# ---------------------------------------------------------------------------
# Helpers to write synthetic ANG files
# ---------------------------------------------------------------------------


def _write_ang(
    path: Path,
    *,
    n_rows: int = 4,
    n_cols: int = 5,
    x_step: float = 1.0,
    y_step: float = 1.0,
    phases: list[str] | None = None,
    kv: float | None = 20.0,
    wd: float | None = 15.0,
    tilt: float | None = 70.0,
    grid: str = "SqrGrid",
    extra_header_lines: list[str] | None = None,
    data_override: str | None = None,
    n_data_cols: int = 10,
    rng: np.random.Generator | None = None,
) -> Path:
    """Create a synthetic ANG file and return its path."""
    if rng is None:
        rng = np.random.default_rng(42)
    if phases is None:
        phases = ["Ferrite", "Austenite"]

    lines: list[str] = []

    # Header
    lines.append(f"# GRID: {grid}")
    lines.append(f"# XSTEP: {x_step:.6f}")
    lines.append(f"# YSTEP: {y_step:.6f}")
    lines.append(f"# NCOLS_ODD: {n_cols}")
    lines.append(f"# NCOLS_EVEN: {n_cols}")
    lines.append(f"# NROWS: {n_rows}")

    if kv is not None:
        lines.append(f"# KV: {kv:.1f}")
    if wd is not None:
        lines.append(f"# WORKING_DISTANCE: {wd:.1f}")
    if tilt is not None:
        lines.append(f"# SampleTiltAngle: {tilt:.1f}")

    for i, phase_name in enumerate(phases, 1):
        lines.append(f"# Phase {i}")
        lines.append(f"# MaterialName \t{phase_name}")
        lines.append(f"# Formula \t{phase_name}")
        lines.append("# Symmetry \t43")

    if extra_header_lines:
        lines.extend(extra_header_lines)

    lines.append("#")

    # Data rows
    if data_override is not None:
        lines.append(data_override)
    else:
        n_points = n_rows * n_cols
        n_phases = len(phases)
        for i in range(n_points):
            row_idx = i // n_cols
            col_idx = i % n_cols
            phi1 = rng.uniform(0, 2 * np.pi)
            big_phi = rng.uniform(0, np.pi)
            phi2 = rng.uniform(0, 2 * np.pi)
            x = col_idx * x_step
            y = row_idx * y_step
            iq = rng.uniform(100, 1000)
            ci = rng.uniform(0.0, 1.0)
            phase_id = rng.integers(1, n_phases + 1)
            sem = rng.uniform(50, 200)
            fit = rng.uniform(0.5, 3.0)

            cols = [phi1, big_phi, phi2, x, y, iq, ci, phase_id, sem, fit]
            row_str = "  ".join(f"{v:.6f}" for v in cols[:n_data_cols])
            lines.append(row_str)

    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseAng:
    """Tests for parse_ang()."""

    def test_basic_parse(self, tmp_path: Path) -> None:
        """Parse a standard ANG file with 2 phases."""
        fpath = _write_ang(tmp_path / "test.ang", n_rows=4, n_cols=5)
        scan = parse_ang(fpath)

        assert scan.source_format == ScanFormat.ANG
        assert scan.n_points == 20
        assert scan.patterns is None
        assert scan.has_patterns is False
        assert scan.has_eds is False

    def test_phase_names(self, tmp_path: Path) -> None:
        """Phase names include 'Not indexed' at position 0."""
        fpath = _write_ang(
            tmp_path / "test.ang",
            phases=["Ferrite", "Austenite", "Martensite"],
        )
        scan = parse_ang(fpath)
        assert scan.phase_names == [
            "Not indexed", "Ferrite", "Austenite", "Martensite"
        ]

    def test_euler_angles_in_radians(self, tmp_path: Path) -> None:
        """Euler angles are stored in radians."""
        fpath = _write_ang(tmp_path / "test.ang")
        scan = parse_ang(fpath)
        assert scan.euler_angles.shape == (20, 3)
        assert scan.euler_angles.dtype == np.float64
        # All Euler angles should be in valid range [0, 2*pi]
        assert np.all(scan.euler_angles >= 0)
        assert np.all(scan.euler_angles <= 2 * np.pi + 0.01)

    def test_confidence_scores(self, tmp_path: Path) -> None:
        """CI values parsed correctly as float32."""
        fpath = _write_ang(tmp_path / "test.ang")
        scan = parse_ang(fpath)
        assert scan.confidence_scores.shape == (20,)
        assert scan.confidence_scores.dtype == np.float32
        assert np.all(scan.confidence_scores >= 0)
        assert np.all(scan.confidence_scores <= 1.0)

    def test_phase_ids_zero_based(self, tmp_path: Path) -> None:
        """Phase IDs are 0-based (0 = not indexed)."""
        fpath = _write_ang(tmp_path / "test.ang", phases=["A", "B"])
        scan = parse_ang(fpath)
        assert scan.phase_ids.dtype == np.int32
        # Original ANG uses 1-based; after parse they should be [0..N]
        assert int(scan.phase_ids.min()) >= 0
        assert int(scan.phase_ids.max()) <= 2  # 0=not indexed, 1=A, 2=B

    def test_scan_shape(self, tmp_path: Path) -> None:
        """Grid shape inferred from header."""
        fpath = _write_ang(
            tmp_path / "test.ang", n_rows=10, n_cols=8
        )
        scan = parse_ang(fpath)
        assert scan.scan_shape == (10, 8)

    def test_step_sizes(self, tmp_path: Path) -> None:
        """Step sizes parsed from header."""
        fpath = _write_ang(
            tmp_path / "test.ang", x_step=0.5, y_step=0.75
        )
        scan = parse_ang(fpath)
        assert scan.step_sizes is not None
        assert abs(scan.step_sizes[0] - 0.5) < 1e-6
        assert abs(scan.step_sizes[1] - 0.75) < 1e-6

    def test_metadata_kv(self, tmp_path: Path) -> None:
        """kV parsed from header."""
        fpath = _write_ang(tmp_path / "test.ang", kv=25.0)
        scan = parse_ang(fpath)
        assert abs(float(scan.metadata["kv"]) - 25.0) < 1e-6  # type: ignore[arg-type]

    def test_metadata_working_distance(self, tmp_path: Path) -> None:
        """Working distance parsed from header."""
        fpath = _write_ang(tmp_path / "test.ang", wd=12.5)
        scan = parse_ang(fpath)
        assert abs(float(scan.metadata["working_distance"]) - 12.5) < 1e-6  # type: ignore[arg-type]

    def test_metadata_sample_tilt(self, tmp_path: Path) -> None:
        """Sample tilt parsed from header."""
        fpath = _write_ang(tmp_path / "test.ang", tilt=65.0)
        scan = parse_ang(fpath)
        assert abs(float(scan.metadata["sample_tilt"]) - 65.0) < 1e-6  # type: ignore[arg-type]

    def test_image_quality_in_metadata(self, tmp_path: Path) -> None:
        """IQ values stored in metadata."""
        fpath = _write_ang(tmp_path / "test.ang")
        scan = parse_ang(fpath)
        iq = scan.metadata["image_quality"]
        assert isinstance(iq, np.ndarray)
        assert iq.shape == (scan.n_points,)  # type: ignore[union-attr]

    def test_source_file(self, tmp_path: Path) -> None:
        """Source file path recorded."""
        fpath = _write_ang(tmp_path / "scan.ang")
        scan = parse_ang(fpath)
        assert "scan.ang" in scan.source_file

    def test_validate_passes(self, tmp_path: Path) -> None:
        """Parsed ScanData passes validation."""
        fpath = _write_ang(tmp_path / "test.ang")
        scan = parse_ang(fpath)
        issues = scan.validate()
        assert issues == [], f"Validation issues: {issues}"

    def test_orientations_property(self, tmp_path: Path) -> None:
        """Orientations auto-convert Euler → quaternion."""
        fpath = _write_ang(tmp_path / "test.ang")
        scan = parse_ang(fpath)
        quats = scan.orientations
        assert quats.shape == (scan.n_points, 4)
        norms = np.linalg.norm(quats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_ang(tmp_path / "nonexistent.ang")

    def test_empty_data(self, tmp_path: Path) -> None:
        """File with header but no data raises ValueError."""
        fpath = tmp_path / "empty.ang"
        fpath.write_text(
            "# GRID: SqrGrid\n"
            "# XSTEP: 1.0\n"
            "# YSTEP: 1.0\n"
            "# NCOLS_ODD: 5\n"
            "# NROWS: 5\n"
        )
        with pytest.raises(ValueError, match="No data rows"):
            parse_ang(fpath)

    def test_single_phase(self, tmp_path: Path) -> None:
        """File with a single phase."""
        fpath = _write_ang(
            tmp_path / "test.ang", phases=["Iron"]
        )
        scan = parse_ang(fpath)
        assert scan.phase_names == ["Not indexed", "Iron"]

    def test_no_phase_in_header(self, tmp_path: Path) -> None:
        """File with no phase definitions."""
        fpath = tmp_path / "nophase.ang"
        # Write minimal header + data
        lines = [
            "# GRID: SqrGrid",
            "# XSTEP: 1.0",
            "# YSTEP: 1.0",
            "# NCOLS_ODD: 2",
            "# NROWS: 2",
            "#",
            "0.1 0.2 0.3 0.0 0.0 500 0.8 1 100 1.5",
            "0.4 0.5 0.6 1.0 0.0 600 0.7 1 110 1.2",
            "0.7 0.8 0.9 0.0 1.0 700 0.9 1 120 1.1",
            "1.0 1.1 1.2 1.0 1.0 800 0.6 1 130 1.0",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ang(fpath)
        assert scan.n_points == 4
        # Only "Not indexed" as phase_names (no phases defined)
        assert scan.phase_names == ["Not indexed"]

    def test_grid_type_in_metadata(self, tmp_path: Path) -> None:
        """Grid type stored in metadata."""
        fpath = _write_ang(
            tmp_path / "test.ang", grid="HexGrid"
        )
        scan = parse_ang(fpath)
        assert scan.metadata.get("grid") == "HexGrid"

    def test_phase_id_zero_not_indexed(self, tmp_path: Path) -> None:
        """Phase ID 0 in data means 'Not indexed'."""
        fpath = tmp_path / "zero_phase.ang"
        lines = [
            "# GRID: SqrGrid",
            "# XSTEP: 1.0",
            "# YSTEP: 1.0",
            "# NCOLS_ODD: 2",
            "# NROWS: 1",
            "# Phase 1",
            "# MaterialName \tFerrite",
            "#",
            "0.1 0.2 0.3 0.0 0.0 500 0.8 0 100 1.5",
            "0.4 0.5 0.6 1.0 0.0 600 0.7 1 110 1.2",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ang(fpath)
        assert scan.phase_ids[0] == 0  # Not indexed
        assert scan.phase_ids[1] == 1  # Ferrite
        assert scan.phase_names[0] == "Not indexed"
        assert scan.phase_names[1] == "Ferrite"

    def test_large_scan(self, tmp_path: Path) -> None:
        """Parse a larger scan (100x100 = 10000 points)."""
        fpath = _write_ang(
            tmp_path / "large.ang", n_rows=100, n_cols=100
        )
        scan = parse_ang(fpath)
        assert scan.n_points == 10000
        assert scan.scan_shape == (100, 100)
        assert scan.validate() == []

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        """Blank lines in the file are skipped."""
        fpath = tmp_path / "blanks.ang"
        lines = [
            "# GRID: SqrGrid",
            "# XSTEP: 1.0",
            "# YSTEP: 1.0",
            "# NCOLS_ODD: 2",
            "# NROWS: 1",
            "#",
            "",
            "0.1 0.2 0.3 0.0 0.0 500 0.8 0 100 1.5",
            "",
            "0.4 0.5 0.6 1.0 0.0 600 0.7 0 110 1.2",
            "",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ang(fpath)
        assert scan.n_points == 2

    def test_formula_as_fallback_name(self, tmp_path: Path) -> None:
        """Phase formula used when MaterialName is missing."""
        fpath = tmp_path / "formula.ang"
        lines = [
            "# GRID: SqrGrid",
            "# XSTEP: 1.0",
            "# YSTEP: 1.0",
            "# NCOLS_ODD: 2",
            "# NROWS: 1",
            "# Phase 1",
            "# Formula \tFe3C",
            "#",
            "0.1 0.2 0.3 0.0 0.0 500 0.8 1 100 1.5",
            "0.4 0.5 0.6 1.0 0.0 600 0.7 1 110 1.2",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ang(fpath)
        assert "Fe3C" in scan.phase_names

    def test_no_header_grid_info(self, tmp_path: Path) -> None:
        """Missing grid info results in None scan_shape."""
        fpath = tmp_path / "minimal.ang"
        lines = [
            "#",
            "0.1 0.2 0.3 0.0 0.0 500 0.8 0 100 1.5",
            "0.4 0.5 0.6 1.0 0.0 600 0.7 0 110 1.2",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ang(fpath)
        assert scan.n_points == 2
        assert scan.scan_shape is None

    def test_infer_nrows_from_ncols(self, tmp_path: Path) -> None:
        """If nrows missing but ncols known, infer from point count."""
        fpath = tmp_path / "infer.ang"
        lines = [
            "# NCOLS_ODD: 3",
            "#",
        ]
        for i in range(6):
            lines.append(
                f"{i * 0.1:.6f} 0.200000 0.300000 "
                f"{(i % 3) * 1.0:.6f} {(i // 3) * 1.0:.6f} "
                f"500.0 0.8 1 100.0 1.5"
            )
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ang(fpath)
        assert scan.scan_shape == (2, 3)

    def test_inconsistent_columns_skipped(self, tmp_path: Path) -> None:
        """Rows with wrong column count are skipped."""
        fpath = tmp_path / "inconsistent.ang"
        lines = [
            "# NCOLS_ODD: 2",
            "# NROWS: 1",
            "#",
            "0.1 0.2 0.3 0.0 0.0 500 0.8 1 100 1.5",
            "0.4 0.5",  # too few columns — skipped
            "0.7 0.8 0.9 1.0 0.0 600 0.7 1 110 1.2",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ang(fpath)
        assert scan.n_points == 2

    def test_only_comment_lines(self, tmp_path: Path) -> None:
        """File with only header/comment lines raises ValueError."""
        fpath = tmp_path / "comments.ang"
        fpath.write_text(
            "# GRID: SqrGrid\n"
            "# Just a comment file\n"
        )
        with pytest.raises(ValueError, match="No data rows"):
            parse_ang(fpath)

    def test_no_kv_wd_tilt(self, tmp_path: Path) -> None:
        """Missing kV/WD/tilt doesn't crash."""
        fpath = _write_ang(
            tmp_path / "test.ang",
            kv=None,
            wd=None,
            tilt=None,
        )
        scan = parse_ang(fpath)
        assert "kv" not in scan.metadata
        assert "working_distance" not in scan.metadata
        assert "sample_tilt" not in scan.metadata

    def test_multiple_phases_numbering(self, tmp_path: Path) -> None:
        """Multiple phases properly numbered."""
        phases = ["Alpha", "Beta", "Gamma", "Delta"]
        fpath = _write_ang(tmp_path / "test.ang", phases=phases)
        scan = parse_ang(fpath)
        assert len(scan.phase_names) == 5  # Not indexed + 4
        assert scan.phase_names[1:] == phases

    def test_encoding_tolerant(self, tmp_path: Path) -> None:
        """File with non-UTF8 chars doesn't crash (errors='replace')."""
        fpath = tmp_path / "encoding.ang"
        content = (
            b"# GRID: SqrGrid\n"
            b"# MaterialName \tFerrit\xe9\n"  # non-UTF8 byte
            b"#\n"
            b"0.1 0.2 0.3 0.0 0.0 500 0.8 0 100 1.5\n"
        )
        fpath.write_bytes(content)
        scan = parse_ang(fpath)
        assert scan.n_points == 1
