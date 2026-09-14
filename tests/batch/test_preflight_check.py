"""Tests for pre-flight validation before batch start."""
import os
import pytest
from unittest.mock import patch, MagicMock
from backend.api.services.preflight_check import PreFlightChecker, PreFlightReport


class TestPreFlightChecker:

    def test_all_pass_with_valid_files(self, tmp_path):
        f1 = tmp_path / "a.h5oina"
        f1.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 1000)  # Fake HDF5 header
        phase = tmp_path / "Al.sht"
        phase.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 500)
        checker = PreFlightChecker()
        report = checker.run(
            files=[str(f1)],
            phases=[{"name": "Al", "path": str(phase), "method": "spherical"}],
            export_dir=str(tmp_path),
        )
        assert isinstance(report, PreFlightReport)
        # At minimum, file existence checks should pass
        file_check = [c for c in report.checks if c["name"] == "source_files_exist"]
        assert len(file_check) == 1
        assert file_check[0]["status"] == "pass"

    def test_fail_missing_source_file(self, tmp_path):
        checker = PreFlightChecker()
        report = checker.run(
            files=[str(tmp_path / "nonexistent.h5oina")],
            phases=[{"name": "Al", "path": str(tmp_path / "Al.sht"), "method": "spherical"}],
            export_dir=str(tmp_path),
        )
        assert report.can_start is False
        file_check = [c for c in report.checks if c["name"] == "source_files_exist"]
        assert file_check[0]["status"] == "fail"

    def test_fail_missing_phase_file(self, tmp_path):
        f1 = tmp_path / "a.h5oina"
        f1.write_bytes(b"\x00" * 100)
        checker = PreFlightChecker()
        report = checker.run(
            files=[str(f1)],
            phases=[{"name": "Al", "path": "/nonexistent/Al.sht", "method": "spherical"}],
            export_dir=str(tmp_path),
        )
        assert report.can_start is False

    def test_warn_on_battery(self, tmp_path):
        f1 = tmp_path / "a.h5oina"
        f1.write_bytes(b"\x00" * 100)
        phase = tmp_path / "Al.sht"
        phase.write_bytes(b"\x00" * 100)
        with patch("psutil.sensors_battery") as mock_bat:
            mock_bat.return_value = MagicMock(power_plugged=False, percent=50)
            checker = PreFlightChecker()
            report = checker.run(
                files=[str(f1)],
                phases=[{"name": "Al", "path": str(phase), "method": "spherical"}],
                export_dir=str(tmp_path),
            )
        power_check = [c for c in report.checks if c["name"] == "power_source"]
        assert power_check[0]["status"] == "warn"

    def test_fail_export_dir_not_writable(self, tmp_path):
        f1 = tmp_path / "a.h5oina"
        f1.write_bytes(b"\x00" * 100)
        phase = tmp_path / "Al.sht"
        phase.write_bytes(b"\x00" * 100)
        checker = PreFlightChecker()
        report = checker.run(
            files=[str(f1)],
            phases=[{"name": "Al", "path": str(phase), "method": "spherical"}],
            export_dir="/nonexistent/path/that/does/not/exist",
        )
        dir_check = [c for c in report.checks if c["name"] == "export_dir_writable"]
        assert dir_check[0]["status"] == "fail"

    def test_disk_space_check(self, tmp_path):
        f1 = tmp_path / "a.h5oina"
        f1.write_bytes(b"\x00" * 100)
        phase = tmp_path / "Al.sht"
        phase.write_bytes(b"\x00" * 100)
        checker = PreFlightChecker()
        report = checker.run(
            files=[str(f1)],
            phases=[{"name": "Al", "path": str(phase), "method": "spherical"}],
            export_dir=str(tmp_path),
        )
        disk_check = [c for c in report.checks if c["name"] == "disk_space"]
        assert len(disk_check) == 1
