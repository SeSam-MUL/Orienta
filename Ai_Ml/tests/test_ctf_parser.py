"""Tests for the CTF file parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ebsd_ai.data.ctf_parser import parse_ctf
from ebsd_ai.data.scan_import import ScanFormat


# ---------------------------------------------------------------------------
# Helpers to write synthetic CTF files
# ---------------------------------------------------------------------------


def _write_ctf(
    path: Path,
    *,
    n_rows: int = 4,
    n_cols: int = 5,
    x_step: float = 1.0,
    y_step: float = 1.0,
    phases: list[str] | None = None,
    kv: float | None = 20.0,
    extra_header_lines: list[str] | None = None,
    include_channel_header: bool = True,
    data_override: str | None = None,
    rng: np.random.Generator | None = None,
) -> Path:
    """Create a synthetic CTF file and return its path."""
    if rng is None:
        rng = np.random.default_rng(42)
    if phases is None:
        phases = ["Ferrite", "Austenite"]

    lines: list[str] = []

    # Header
    if include_channel_header:
        lines.append("Channel Text File")
        lines.append("Prj\ttest_project")
        lines.append("Author\t[Unknown]")
        lines.append("JobMode\tGrid")
    lines.append(f"XCells\t{n_cols}")
    lines.append(f"YCells\t{n_rows}")
    lines.append(f"XStep\t{x_step}")
    lines.append(f"YStep\t{y_step}")
    if kv is not None:
        lines.append(f"AcqE1\t{kv}")

    # Phase definitions
    lines.append(f"Phases\t{len(phases)}")
    for phase_name in phases:
        # Lattice params; angles; name; group; space group; ...
        lines.append(
            f"3.24;3.24;5.18\t90;90;120\t{phase_name}\t11\t225"
        )

    if extra_header_lines:
        lines.extend(extra_header_lines)

    # Data rows
    if data_override is not None:
        lines.append(data_override)
    else:
        n_points = n_rows * n_cols
        n_ph = len(phases)
        for i in range(n_points):
            row_idx = i // n_cols
            col_idx = i % n_cols
            phase_id = rng.integers(1, n_ph + 1)
            x = col_idx * x_step
            y = row_idx * y_step
            bands = rng.integers(3, 12)
            error = 0
            e1 = rng.uniform(0, 360)
            e2 = rng.uniform(0, 180)
            e3 = rng.uniform(0, 360)
            mad = rng.uniform(0.1, 2.5)
            bc = rng.integers(50, 255)
            bs = rng.integers(30, 200)

            row = (
                f"{phase_id}\t{x:.4f}\t{y:.4f}\t{bands}\t{error}\t"
                f"{e1:.4f}\t{e2:.4f}\t{e3:.4f}\t{mad:.4f}\t{bc}\t{bs}"
            )
            lines.append(row)

    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseCTF:
    """Tests for parse_ctf()."""

    def test_basic_parse(self, tmp_path: Path) -> None:
        """Parse a standard CTF file with 2 phases."""
        fpath = _write_ctf(tmp_path / "test.ctf", n_rows=4, n_cols=5)
        scan = parse_ctf(fpath)

        assert scan.source_format == ScanFormat.CTF
        assert scan.n_points == 20
        assert scan.patterns is None
        assert scan.has_patterns is False
        assert scan.has_eds is False

    def test_phase_names(self, tmp_path: Path) -> None:
        """Phase names include 'Not indexed' at position 0."""
        fpath = _write_ctf(
            tmp_path / "test.ctf",
            phases=["Ferrite", "Austenite", "Martensite"],
        )
        scan = parse_ctf(fpath)
        assert scan.phase_names == [
            "Not indexed", "Ferrite", "Austenite", "Martensite"
        ]

    def test_euler_angles_in_radians(self, tmp_path: Path) -> None:
        """Euler angles are converted from degrees to radians."""
        fpath = _write_ctf(tmp_path / "test.ctf")
        scan = parse_ctf(fpath)
        assert scan.euler_angles.shape == (20, 3)
        assert scan.euler_angles.dtype == np.float64
        # Original values are 0-360 degrees → should be 0 to 2*pi radians
        assert np.all(scan.euler_angles >= 0)
        assert np.all(scan.euler_angles <= 2 * np.pi + 0.01)

    def test_confidence_from_mad(self, tmp_path: Path) -> None:
        """Confidence scores derived from MAD (lower MAD = higher CI)."""
        fpath = _write_ctf(tmp_path / "test.ctf")
        scan = parse_ctf(fpath)
        assert scan.confidence_scores.shape == (20,)
        assert scan.confidence_scores.dtype == np.float32
        # MAD in range [0.1, 2.5] → confidence should be in (0, 1)
        assert np.all(scan.confidence_scores > 0)
        assert np.all(scan.confidence_scores <= 1.0)

    def test_phase_ids_zero_based(self, tmp_path: Path) -> None:
        """Phase IDs are 0-based (0 = not indexed)."""
        fpath = _write_ctf(tmp_path / "test.ctf", phases=["A", "B"])
        scan = parse_ctf(fpath)
        assert scan.phase_ids.dtype == np.int32
        assert int(scan.phase_ids.min()) >= 0
        assert int(scan.phase_ids.max()) <= 2

    def test_scan_shape(self, tmp_path: Path) -> None:
        """Grid shape from header."""
        fpath = _write_ctf(
            tmp_path / "test.ctf", n_rows=10, n_cols=8
        )
        scan = parse_ctf(fpath)
        assert scan.scan_shape == (10, 8)

    def test_step_sizes(self, tmp_path: Path) -> None:
        """Step sizes parsed from header."""
        fpath = _write_ctf(
            tmp_path / "test.ctf", x_step=0.5, y_step=0.75
        )
        scan = parse_ctf(fpath)
        assert scan.step_sizes is not None
        assert abs(scan.step_sizes[0] - 0.5) < 1e-6
        assert abs(scan.step_sizes[1] - 0.75) < 1e-6

    def test_metadata_kv(self, tmp_path: Path) -> None:
        """kV parsed from AcqE1 header."""
        fpath = _write_ctf(tmp_path / "test.ctf", kv=25.0)
        scan = parse_ctf(fpath)
        assert abs(float(scan.metadata["kv"]) - 25.0) < 1e-6  # type: ignore[arg-type]

    def test_band_contrast_in_metadata(self, tmp_path: Path) -> None:
        """BC values stored in metadata."""
        fpath = _write_ctf(tmp_path / "test.ctf")
        scan = parse_ctf(fpath)
        bc = scan.metadata["band_contrast"]
        assert isinstance(bc, np.ndarray)
        assert bc.shape == (scan.n_points,)  # type: ignore[union-attr]

    def test_mad_in_metadata(self, tmp_path: Path) -> None:
        """MAD values stored in metadata."""
        fpath = _write_ctf(tmp_path / "test.ctf")
        scan = parse_ctf(fpath)
        mad = scan.metadata["mad"]
        assert isinstance(mad, np.ndarray)
        assert mad.shape == (scan.n_points,)  # type: ignore[union-attr]

    def test_source_file(self, tmp_path: Path) -> None:
        """Source file path recorded."""
        fpath = _write_ctf(tmp_path / "scan.ctf")
        scan = parse_ctf(fpath)
        assert "scan.ctf" in scan.source_file

    def test_validate_passes(self, tmp_path: Path) -> None:
        """Parsed ScanData passes validation."""
        fpath = _write_ctf(tmp_path / "test.ctf")
        scan = parse_ctf(fpath)
        issues = scan.validate()
        assert issues == [], f"Validation issues: {issues}"

    def test_orientations_property(self, tmp_path: Path) -> None:
        """Orientations auto-convert Euler → quaternion."""
        fpath = _write_ctf(tmp_path / "test.ctf")
        scan = parse_ctf(fpath)
        quats = scan.orientations
        assert quats.shape == (scan.n_points, 4)
        norms = np.linalg.norm(quats, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_ctf(tmp_path / "nonexistent.ctf")

    def test_empty_data(self, tmp_path: Path) -> None:
        """File with header but no data raises ValueError."""
        fpath = tmp_path / "empty.ctf"
        fpath.write_text(
            "Channel Text File\n"
            "XCells\t5\n"
            "YCells\t5\n"
            "Phases\t1\n"
            "3.24;3.24;5.18\t90;90;120\tIron\t11\t225\n"
        )
        with pytest.raises(ValueError, match="No data rows"):
            parse_ctf(fpath)

    def test_single_phase(self, tmp_path: Path) -> None:
        """File with single phase."""
        fpath = _write_ctf(tmp_path / "test.ctf", phases=["Iron"])
        scan = parse_ctf(fpath)
        assert scan.phase_names == ["Not indexed", "Iron"]

    def test_phase_id_zero_not_indexed(self, tmp_path: Path) -> None:
        """Phase ID 0 means 'Not indexed'."""
        fpath = tmp_path / "zero.ctf"
        lines = [
            "Channel Text File",
            "XCells\t2",
            "YCells\t1",
            "XStep\t1.0",
            "YStep\t1.0",
            "Phases\t1",
            "3.24;3.24;5.18\t90;90;120\tFerrite\t11\t225",
            "0\t0.0\t0.0\t5\t0\t45.0\t30.0\t60.0\t1.5\t120\t80",
            "1\t1.0\t0.0\t8\t0\t90.0\t45.0\t120.0\t0.5\t200\t150",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ctf(fpath)
        assert scan.phase_ids[0] == 0  # Not indexed
        assert scan.phase_ids[1] == 1  # Ferrite

    def test_euler_degrees_to_radians(self, tmp_path: Path) -> None:
        """Known Euler angles in degrees are correctly converted."""
        fpath = tmp_path / "known.ctf"
        lines = [
            "Channel Text File",
            "XCells\t1",
            "YCells\t1",
            "XStep\t1.0",
            "YStep\t1.0",
            "Phases\t1",
            "3.24;3.24;5.18\t90;90;120\tFerrite\t11\t225",
            # Euler: 90, 45, 180 degrees
            "1\t0.0\t0.0\t6\t0\t90.0\t45.0\t180.0\t0.5\t150\t100",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ctf(fpath)
        np.testing.assert_allclose(
            scan.euler_angles[0],
            [np.pi / 2, np.pi / 4, np.pi],
            atol=1e-10,
        )

    def test_mad_to_confidence_known(self, tmp_path: Path) -> None:
        """MAD=0 → confidence≈1, MAD large → confidence≈0."""
        fpath = tmp_path / "mad.ctf"
        lines = [
            "Channel Text File",
            "XCells\t2",
            "YCells\t1",
            "XStep\t1.0",
            "YStep\t1.0",
            "Phases\t1",
            "3.24;3.24;5.18\t90;90;120\tFerrite\t11\t225",
            # MAD = 0.0 → high confidence
            "1\t0.0\t0.0\t6\t0\t0.0\t0.0\t0.0\t0.0\t150\t100",
            # MAD = 5.0 → very low confidence
            "1\t1.0\t0.0\t6\t0\t0.0\t0.0\t0.0\t5.0\t150\t100",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ctf(fpath)
        # MAD=0 → exp(0) = 1.0
        assert abs(scan.confidence_scores[0] - 1.0) < 1e-5
        # MAD=5 → exp(-5) ≈ 0.0067
        assert scan.confidence_scores[1] < 0.01

    def test_large_scan(self, tmp_path: Path) -> None:
        """Parse a larger scan (50x40 = 2000 points)."""
        fpath = _write_ctf(
            tmp_path / "large.ctf", n_rows=50, n_cols=40
        )
        scan = parse_ctf(fpath)
        assert scan.n_points == 2000
        assert scan.scan_shape == (50, 40)
        assert scan.validate() == []

    def test_no_channel_header(self, tmp_path: Path) -> None:
        """File without 'Channel Text File' line still parses."""
        fpath = _write_ctf(
            tmp_path / "test.ctf",
            include_channel_header=False,
        )
        scan = parse_ctf(fpath)
        assert scan.n_points == 20

    def test_multiple_phases(self, tmp_path: Path) -> None:
        """Multiple phases properly numbered."""
        phases = ["Alpha", "Beta", "Gamma", "Delta"]
        fpath = _write_ctf(tmp_path / "test.ctf", phases=phases)
        scan = parse_ctf(fpath)
        assert len(scan.phase_names) == 5
        assert scan.phase_names[1:] == phases

    def test_no_kv(self, tmp_path: Path) -> None:
        """Missing kV doesn't crash."""
        fpath = _write_ctf(tmp_path / "test.ctf", kv=None)
        scan = parse_ctf(fpath)
        assert "kv" not in scan.metadata

    def test_space_delimited_data(self, tmp_path: Path) -> None:
        """CTF files with space-delimited data instead of tabs."""
        fpath = tmp_path / "space.ctf"
        lines = [
            "Channel Text File",
            "XCells\t2",
            "YCells\t1",
            "XStep\t1.0",
            "YStep\t1.0",
            "Phases\t1",
            "3.24;3.24;5.18\t90;90;120\tFerrite\t11\t225",
            # Space-delimited data rows
            "1 0.0 0.0 6 0 45.0 30.0 60.0 0.5 150 100",
            "1 1.0 0.0 8 0 90.0 45.0 120.0 0.3 200 150",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ctf(fpath)
        assert scan.n_points == 2

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        """Blank lines are skipped."""
        fpath = tmp_path / "blanks.ctf"
        lines = [
            "Channel Text File",
            "",
            "XCells\t2",
            "YCells\t1",
            "XStep\t1.0",
            "YStep\t1.0",
            "Phases\t1",
            "3.24;3.24;5.18\t90;90;120\tFerrite\t11\t225",
            "",
            "1\t0.0\t0.0\t6\t0\t45.0\t30.0\t60.0\t0.5\t150\t100",
            "",
            "1\t1.0\t0.0\t8\t0\t90.0\t45.0\t120.0\t0.3\t200\t150",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ctf(fpath)
        assert scan.n_points == 2

    def test_encoding_tolerant(self, tmp_path: Path) -> None:
        """Non-UTF8 bytes don't crash the parser."""
        fpath = tmp_path / "enc.ctf"
        content = (
            b"Channel Text File\n"
            b"XCells\t1\n"
            b"YCells\t1\n"
            b"Phases\t1\n"
            b"3.24;3.24;5.18\t90;90;120\tFerrit\xe9\t11\t225\n"
            b"1\t0.0\t0.0\t6\t0\t0.0\t0.0\t0.0\t0.5\t150\t100\n"
        )
        fpath.write_bytes(content)
        scan = parse_ctf(fpath)
        assert scan.n_points == 1

    def test_phase_name_from_lattice_fallback(
        self, tmp_path: Path
    ) -> None:
        """Phase definition with empty name gets a fallback name."""
        fpath = tmp_path / "noname.ctf"
        lines = [
            "Channel Text File",
            "XCells\t1",
            "YCells\t1",
            "Phases\t1",
            # Phase with no name (empty third field)
            "3.24;3.24;5.18\t90;90;120\t\t11\t225",
            "1\t0.0\t0.0\t6\t0\t0.0\t0.0\t0.0\t0.5\t150\t100",
        ]
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ctf(fpath)
        # Should get a fallback name
        assert len(scan.phase_names) == 2
        assert scan.phase_names[1] == "Phase_1"

    def test_only_header(self, tmp_path: Path) -> None:
        """File with only header raises ValueError."""
        fpath = tmp_path / "header.ctf"
        fpath.write_text(
            "Channel Text File\n"
            "XCells\t5\n"
            "YCells\t5\n"
        )
        with pytest.raises(ValueError, match="No data rows"):
            parse_ctf(fpath)

    def test_author_and_job_mode(self, tmp_path: Path) -> None:
        """Author and JobMode parsed from header."""
        fpath = _write_ctf(tmp_path / "test.ctf")
        scan = parse_ctf(fpath)
        assert scan.metadata.get("author") == "[Unknown]"
        assert scan.metadata.get("job_mode") == "Grid"

    def test_infer_nrows_from_xcells(self, tmp_path: Path) -> None:
        """If ycells missing, infer from xcells + point count."""
        fpath = tmp_path / "infer.ctf"
        lines = [
            "Channel Text File",
            "XCells\t3",
            "XStep\t1.0",
            "YStep\t1.0",
            "Phases\t1",
            "3.24;3.24;5.18\t90;90;120\tIron\t11\t225",
        ]
        for i in range(6):
            x = (i % 3) * 1.0
            y = (i // 3) * 1.0
            lines.append(
                f"1\t{x:.4f}\t{y:.4f}\t6\t0\t0.0\t0.0\t0.0\t0.5\t150\t100"
            )
        fpath.write_text("\n".join(lines) + "\n")
        scan = parse_ctf(fpath)
        assert scan.scan_shape == (2, 3)
