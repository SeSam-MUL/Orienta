"""Tests for the import pipeline (parse_scan, import_to_store, CLI).

Tests verify:
- parse_scan() dispatches to the correct parser by format
- import_to_store() filters by CI, handles no-pattern files, builds DetectorInfo
- ImportResult contains correct counts
- _build_detector_info() maps formats to manufacturers/conventions
- CLI script (scripts/import_data.py) handles normal and error cases
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ebsd_ai.data.scan_import import (
    ImportResult,
    ScanData,
    ScanFormat,
    _build_detector_info,
    import_to_store,
    parse_scan,
)
from ebsd_ai.data.training_store import TrainingStore


# ---------------------------------------------------------------------------
# Helpers: create synthetic scan files
# ---------------------------------------------------------------------------


def _write_ang_file(path: Path, n_points: int = 20, n_phases: int = 2) -> Path:
    """Write a minimal ANG file with synthetic data."""
    filepath = path / "test_scan.ang"
    rng = np.random.default_rng(42)

    lines = [
        "# TEM_PIXperUM          1.000000",
        "# x-star                0.500000",
        "# y-star                0.200000",
        "# z-star                0.600000",
        "# GRID: SqrGrid",
        "# XSTEP: 0.500000",
        "# YSTEP: 0.500000",
        f"# NCOLS_ODD: {n_points // 2}",
        f"# NROWS: 2",
    ]
    for i in range(n_phases):
        lines.append(f"# Phase {i + 1}")
        lines.append(f"# MaterialName  SyntheticPhase_{i}")
        lines.append(f"# Formula  Fe{i}")
        lines.append(f"# Symmetry  43")

    for _ in range(n_points):
        phi1 = rng.uniform(0, 2 * np.pi)
        phi_big = rng.uniform(0, np.pi)
        phi2 = rng.uniform(0, 2 * np.pi)
        x = rng.uniform(0, 10)
        y = rng.uniform(0, 10)
        iq = rng.uniform(100, 1000)
        ci = rng.uniform(0.0, 1.0)
        phase = rng.integers(1, n_phases + 1)
        sem = 100.0
        fit = rng.uniform(0, 3)
        lines.append(
            f"  {phi1:.6f}  {phi_big:.6f}  {phi2:.6f}  "
            f"{x:.6f}  {y:.6f}  {iq:.1f}  {ci:.4f}  "
            f"{phase}  {sem:.1f}  {fit:.4f}"
        )

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


def _write_ctf_file(path: Path, n_points: int = 20, n_phases: int = 2) -> Path:
    """Write a minimal CTF file with synthetic data."""
    filepath = path / "test_scan.ctf"
    rng = np.random.default_rng(42)

    ncols = n_points // 2
    nrows = 2

    lines = [
        "Channel Text File",
        f"Prj\ttest_project",
        f"Author\ttest_author",
        f"JobMode\tGrid",
        f"XCells\t{ncols}",
        f"YCells\t{nrows}",
        f"XStep\t0.5",
        f"YStep\t0.5",
        f"AcqE1\t20.0",
        f"Phases\t{n_phases}",
    ]
    for i in range(n_phases):
        lines.append(
            f"3.24;3.24;5.18\t90;90;120\tTestPhase_{i}\t11\t225\t"
        )

    for _ in range(n_points):
        phase = rng.integers(1, n_phases + 1)
        x = rng.uniform(0, 10)
        y = rng.uniform(0, 10)
        bands = rng.integers(3, 12)
        error = 0
        e1 = rng.uniform(0, 360)
        e2 = rng.uniform(0, 180)
        e3 = rng.uniform(0, 360)
        mad = rng.uniform(0.1, 3.0)
        bc = rng.integers(50, 255)
        bs = rng.integers(10, 100)
        lines.append(
            f"{phase}\t{x:.4f}\t{y:.4f}\t{bands}\t{error}\t"
            f"{e1:.4f}\t{e2:.4f}\t{e3:.4f}\t{mad:.4f}\t{bc}\t{bs}"
        )

    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


def _write_h5oina_file(
    path: Path,
    n_points: int = 20,
    n_phases: int = 2,
    include_patterns: bool = True,
) -> Path:
    """Write a minimal H5OINA-like HDF5 file."""
    filepath = path / "test_scan.h5oina"
    rng = np.random.default_rng(42)

    nrows = 2
    ncols = n_points // 2

    with h5py.File(filepath, "w") as f:
        grp = f.create_group("1/EBSD")
        header = grp.create_group("Header")
        data = grp.create_group("Data")

        header.create_dataset("X Cells", data=ncols)
        header.create_dataset("Y Cells", data=nrows)
        header.create_dataset("X Step", data=0.5)
        header.create_dataset("Y Step", data=0.5)

        phases_grp = header.create_group("Phases")
        for i in range(n_phases):
            pg = phases_grp.create_group(str(i + 1))
            pg.create_dataset("Name", data=f"H5Phase_{i}")

        euler = rng.uniform(0, 2 * np.pi, size=(n_points, 3)).astype(np.float32)
        data.create_dataset("Euler", data=euler)

        phase_ids = rng.integers(0, n_phases + 1, size=n_points).astype(np.int32)
        data.create_dataset("Phase", data=phase_ids)

        bc = rng.integers(50, 255, size=n_points).astype(np.uint8)
        data.create_dataset("Band Contrast", data=bc)

        if include_patterns:
            patterns = rng.integers(0, 255, size=(nrows, ncols, 60, 80), dtype=np.uint8)
            data.create_dataset("Processed Patterns", data=patterns)

    return filepath


# ---------------------------------------------------------------------------
# Tests: parse_scan()
# ---------------------------------------------------------------------------


class TestParseScan:
    """Tests for the parse_scan() dispatcher."""

    def test_parse_ang(self, tmp_path: Path) -> None:
        filepath = _write_ang_file(tmp_path)
        scan = parse_scan(filepath)
        assert scan.source_format == ScanFormat.ANG
        assert scan.n_points == 20
        assert len(scan.phase_names) > 1

    def test_parse_ctf(self, tmp_path: Path) -> None:
        filepath = _write_ctf_file(tmp_path)
        scan = parse_scan(filepath)
        assert scan.source_format == ScanFormat.CTF
        assert scan.n_points == 20

    def test_parse_h5oina(self, tmp_path: Path) -> None:
        filepath = _write_h5oina_file(tmp_path)
        scan = parse_scan(filepath)
        assert scan.source_format == ScanFormat.H5OINA
        assert scan.n_points == 20
        assert scan.has_patterns

    def test_parse_unknown_extension(self, tmp_path: Path) -> None:
        filepath = tmp_path / "test.xyz"
        filepath.write_text("garbage", encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown or unsupported"):
            parse_scan(filepath)

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        filepath = tmp_path / "nonexistent.ang"
        with pytest.raises(FileNotFoundError):
            parse_scan(filepath)


# ---------------------------------------------------------------------------
# Tests: ImportResult
# ---------------------------------------------------------------------------


class TestImportResult:
    """Tests for the ImportResult dataclass."""

    def test_defaults(self) -> None:
        r = ImportResult()
        assert r.source_file == ""
        assert r.format == ScanFormat.UNKNOWN
        assert r.n_points_total == 0
        assert r.n_points_imported == 0
        assert r.n_points_skipped_no_pattern == 0
        assert r.phase_counts == {}
        assert r.ci_threshold == 0.3
        assert r.has_eds is False
        assert r.warnings == []

    def test_custom_values(self) -> None:
        r = ImportResult(
            source_file="test.ang",
            format=ScanFormat.ANG,
            n_points_total=100,
            n_points_imported=80,
            phase_counts={"Ferrit": 50, "Austenit": 30},
            ci_threshold=0.4,
            has_eds=True,
        )
        assert r.n_points_imported == 80
        assert r.phase_counts["Ferrit"] == 50


# ---------------------------------------------------------------------------
# Tests: _build_detector_info()
# ---------------------------------------------------------------------------


class TestBuildDetectorInfo:
    """Tests for detector info inference from ScanData metadata."""

    def test_ang_format(self) -> None:
        scan = ScanData(source_format=ScanFormat.ANG)
        det = _build_detector_info(scan)
        assert det.manufacturer.value == "EDAX"
        assert det.pc_convention.value == "EDAX"

    def test_ctf_format(self) -> None:
        scan = ScanData(source_format=ScanFormat.CTF)
        det = _build_detector_info(scan)
        assert det.manufacturer.value == "OXFORD"
        assert det.pc_convention.value == "OXFORD"

    def test_h5oina_format(self) -> None:
        scan = ScanData(source_format=ScanFormat.H5OINA)
        det = _build_detector_info(scan)
        assert det.manufacturer.value == "OXFORD"
        assert det.pc_convention.value == "OXFORD"

    def test_kikuchipy_format(self) -> None:
        scan = ScanData(source_format=ScanFormat.KIKUCHIPY)
        det = _build_detector_info(scan)
        assert det.manufacturer.value == "OTHER"
        assert det.pc_convention.value == "KIKUCHIPY"

    def test_emsoft_format(self) -> None:
        scan = ScanData(source_format=ScanFormat.EMSOFT)
        det = _build_detector_info(scan)
        assert det.manufacturer.value == "OTHER"
        assert det.pc_convention.value == "EMSOFT"

    def test_unknown_format_defaults(self) -> None:
        scan = ScanData(source_format=ScanFormat.UNKNOWN)
        det = _build_detector_info(scan)
        assert det.manufacturer.value == "OTHER"
        assert det.pc_convention.value == "KIKUCHIPY"  # default

    def test_metadata_kv(self) -> None:
        scan = ScanData(
            source_format=ScanFormat.ANG,
            metadata={"kv": 25.0},
        )
        det = _build_detector_info(scan)
        assert det.kv == 25.0

    def test_metadata_working_distance(self) -> None:
        scan = ScanData(
            source_format=ScanFormat.CTF,
            metadata={"working_distance": 12.0},
        )
        det = _build_detector_info(scan)
        assert det.working_distance == 12.0

    def test_metadata_sample_tilt(self) -> None:
        scan = ScanData(
            source_format=ScanFormat.H5OINA,
            metadata={"sample_tilt": 65.0},
        )
        det = _build_detector_info(scan)
        assert det.sample_tilt == 65.0

    def test_missing_metadata_uses_defaults(self) -> None:
        scan = ScanData(source_format=ScanFormat.ANG, metadata={})
        det = _build_detector_info(scan)
        assert det.kv == 20.0
        assert det.working_distance == 15.0
        assert det.sample_tilt == 70.0


# ---------------------------------------------------------------------------
# Tests: import_to_store()
# ---------------------------------------------------------------------------


class TestImportToStore:
    """Tests for the import_to_store() pipeline."""

    def test_import_h5oina_with_patterns(self, tmp_path: Path) -> None:
        filepath = _write_h5oina_file(tmp_path, n_points=20, n_phases=2)
        store = TrainingStore(local_path=tmp_path / "store")

        result = import_to_store(filepath, store, ci_threshold=0.0)

        assert result.format == ScanFormat.H5OINA
        assert result.n_points_total == 20
        assert result.n_points_imported == 20  # CI=0 means all pass
        assert result.source_file == str(filepath)
        assert len(result.phase_counts) > 0

    def test_import_ci_filtering(self, tmp_path: Path) -> None:
        filepath = _write_h5oina_file(tmp_path, n_points=20)
        store = TrainingStore(local_path=tmp_path / "store")

        # With high CI threshold, fewer points should pass
        result = import_to_store(filepath, store, ci_threshold=0.99)

        assert result.n_points_imported <= result.n_points_total
        # Band contrast / 255 rarely reaches 0.99
        assert result.n_points_imported < 20

    def test_import_no_patterns(self, tmp_path: Path) -> None:
        """ANG files have no patterns — import should skip."""
        filepath = _write_ang_file(tmp_path)
        store = TrainingStore(local_path=tmp_path / "store")

        result = import_to_store(filepath, store)

        assert result.n_points_imported == 0
        assert result.n_points_skipped_no_pattern == 20
        assert any("No patterns" in w for w in result.warnings)

    def test_import_h5oina_no_patterns(self, tmp_path: Path) -> None:
        """H5OINA without patterns dataset."""
        filepath = _write_h5oina_file(
            tmp_path, n_points=20, include_patterns=False
        )
        store = TrainingStore(local_path=tmp_path / "store")

        result = import_to_store(filepath, store)

        assert result.n_points_imported == 0
        assert result.n_points_skipped_no_pattern == 20

    def test_import_pre_parsed_scan(self, tmp_path: Path) -> None:
        """Using a pre-parsed ScanData."""
        rng = np.random.default_rng(42)
        n = 10

        scan = ScanData(
            patterns=rng.integers(0, 255, (n, 60, 80), dtype=np.uint8),
            phase_ids=np.array([1, 1, 1, 2, 2, 0, 1, 2, 1, 0], dtype=np.int32),
            phase_names=["Not indexed", "Alpha", "Beta"],
            euler_angles=rng.uniform(0, 2 * np.pi, (n, 3)),
            confidence_scores=np.array(
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.4, 0.3, 0.2, 0.05],
                dtype=np.float32,
            ),
            source_format=ScanFormat.H5OINA,
            source_file="preloaded.h5oina",
        )

        store = TrainingStore(local_path=tmp_path / "store")
        # Dummy filepath — not actually opened since scan_data is provided
        dummy = tmp_path / "dummy.h5oina"
        dummy.touch()

        result = import_to_store(
            dummy, store, ci_threshold=0.3, scan_data=scan
        )

        assert result.n_points_total == 10
        # CI >= 0.3: indices 0(0.9), 1(0.8), 2(0.7), 3(0.6),
        # 4(0.5), 6(0.4), 7(0.3) → 7 points
        assert result.n_points_imported == 7
        assert "Alpha" in result.phase_counts
        assert "Beta" in result.phase_counts

    def test_import_empty_scan(self, tmp_path: Path) -> None:
        """Scan with 0 points returns early."""
        scan = ScanData(
            source_format=ScanFormat.ANG,
        )
        store = TrainingStore(local_path=tmp_path / "store")
        dummy = tmp_path / "empty.ang"
        dummy.touch()

        result = import_to_store(dummy, store, scan_data=scan)

        assert result.n_points_imported == 0
        assert any("0 points" in w for w in result.warnings)

    def test_import_writes_to_store(self, tmp_path: Path) -> None:
        """Verify data actually ends up in the TrainingStore."""
        filepath = _write_h5oina_file(tmp_path, n_points=10, n_phases=2)
        store = TrainingStore(local_path=tmp_path / "store")

        result = import_to_store(filepath, store, ci_threshold=0.0)

        stats = store.get_dataset_stats()
        assert stats["total_samples"] == result.n_points_imported
        assert stats["total_samples"] > 0

    def test_import_result_has_eds_false(self, tmp_path: Path) -> None:
        """H5OINA files in our test don't have EDS."""
        filepath = _write_h5oina_file(tmp_path)
        store = TrainingStore(local_path=tmp_path / "store")

        result = import_to_store(filepath, store, ci_threshold=0.0)

        assert result.has_eds is False

    def test_import_with_custom_target_size(self, tmp_path: Path) -> None:
        """Custom target_size is passed through."""
        filepath = _write_h5oina_file(tmp_path, n_points=4, n_phases=1)
        store = TrainingStore(local_path=tmp_path / "store")

        result = import_to_store(
            filepath, store, ci_threshold=0.0, target_size=64
        )

        assert result.n_points_imported > 0


# ---------------------------------------------------------------------------
# Tests: CLI (scripts/import_data.py)
# ---------------------------------------------------------------------------


class TestImportCLI:
    """Tests for the import_data.py CLI script."""

    def test_import_single_file(self, tmp_path: Path) -> None:
        from scripts.import_data import main

        filepath = _write_h5oina_file(tmp_path, n_points=10)
        store_dir = tmp_path / "store"

        code = main([str(filepath), "--store", str(store_dir), "--ci", "0.0"])
        assert code == 0

        store = TrainingStore(local_path=store_dir)
        assert store.get_dataset_stats()["total_samples"] > 0

    def test_import_multiple_files(self, tmp_path: Path) -> None:
        from scripts.import_data import main

        (tmp_path / "d1").mkdir()
        (tmp_path / "d2").mkdir()
        f1 = _write_h5oina_file(tmp_path / "d1", n_points=6)
        f2 = _write_h5oina_file(tmp_path / "d2", n_points=8)
        store_dir = tmp_path / "store"

        code = main([
            str(f1), str(f2),
            "--store", str(store_dir),
            "--ci", "0.0",
        ])
        assert code == 0

    def test_import_dry_run(self, tmp_path: Path) -> None:
        from scripts.import_data import main

        filepath = _write_h5oina_file(tmp_path, n_points=10)
        store_dir = tmp_path / "store"

        code = main([
            str(filepath), "--store", str(store_dir), "--dry-run",
        ])
        assert code == 0
        # Store should NOT be created in dry-run mode
        assert not store_dir.exists()

    def test_import_verbose(self, tmp_path: Path) -> None:
        from scripts.import_data import main

        filepath = _write_h5oina_file(tmp_path, n_points=10)
        store_dir = tmp_path / "store"

        code = main([
            str(filepath), "--store", str(store_dir),
            "--ci", "0.0", "--verbose",
        ])
        assert code == 0

    def test_import_missing_file(self, tmp_path: Path) -> None:
        from scripts.import_data import main

        store_dir = tmp_path / "store"
        code = main([
            str(tmp_path / "nonexistent.ang"),
            "--store", str(store_dir),
        ])
        assert code == 1

    def test_import_no_pattern_file(self, tmp_path: Path) -> None:
        """ANG files have no patterns — CLI should handle gracefully."""
        from scripts.import_data import main

        filepath = _write_ang_file(tmp_path)
        store_dir = tmp_path / "store"

        code = main([
            str(filepath), "--store", str(store_dir),
        ])
        # Should succeed (no error), just 0 imported
        assert code == 0

    def test_import_custom_target_size(self, tmp_path: Path) -> None:
        from scripts.import_data import main

        filepath = _write_h5oina_file(tmp_path, n_points=6)
        store_dir = tmp_path / "store"

        code = main([
            str(filepath), "--store", str(store_dir),
            "--ci", "0.0", "--target-size", "64",
        ])
        assert code == 0

    def test_parse_args_defaults(self) -> None:
        from scripts.import_data import parse_args

        args = parse_args(["file.ang", "--store", "./data"])
        assert args.files == ["file.ang"]
        assert args.store == "./data"
        assert args.ci == 0.3
        assert args.target_size == 128
        assert args.dry_run is False
        assert args.verbose is False

    def test_parse_args_all_options(self) -> None:
        from scripts.import_data import parse_args

        args = parse_args([
            "a.ang", "b.ctf",
            "--store", "/tmp/store",
            "--ci", "0.5",
            "--target-size", "64",
            "--dry-run",
            "--verbose",
        ])
        assert args.files == ["a.ang", "b.ctf"]
        assert args.store == "/tmp/store"
        assert args.ci == 0.5
        assert args.target_size == 64
        assert args.dry_run is True
        assert args.verbose is True
