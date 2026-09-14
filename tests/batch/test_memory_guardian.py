"""Tests for Memory Guardian — RAM monitoring and OOM prevention."""
import pytest
from unittest.mock import patch, MagicMock
from backend.api.services.memory_guardian import MemoryGuardian, MemoryStatus


class TestMemoryGuardian:

    def test_get_current_usage_returns_dict(self):
        mg = MemoryGuardian()
        usage = mg.get_current_usage()
        assert "total_mb" in usage
        assert "available_mb" in usage
        assert "percent" in usage
        assert "process_rss_mb" in usage
        assert usage["total_mb"] > 0

    def test_check_can_proceed_returns_status(self):
        mg = MemoryGuardian(min_free_gb=0.001)  # Very low threshold
        status = mg.check_can_proceed()
        assert status in (MemoryStatus.OK, MemoryStatus.WARNING, MemoryStatus.CRITICAL)

    def test_check_can_proceed_critical_when_low_ram(self):
        mg = MemoryGuardian(min_free_gb=99999)  # Impossibly high
        status = mg.check_can_proceed()
        assert status == MemoryStatus.CRITICAL

    def test_estimate_job_memory_from_file_size(self, tmp_path):
        # Create a fake 10MB file
        fake = tmp_path / "test.h5"
        fake.write_bytes(b"\x00" * (10 * 1024 * 1024))
        mg = MemoryGuardian()
        est = mg.estimate_job_memory(str(fake))
        # Should be > file size (decompression factor + overhead)
        assert est > 10

    def test_estimate_job_memory_with_dict(self, tmp_path):
        fake_data = tmp_path / "data.h5"
        fake_dict = tmp_path / "dict.h5"
        fake_data.write_bytes(b"\x00" * (10 * 1024 * 1024))
        fake_dict.write_bytes(b"\x00" * (50 * 1024 * 1024))
        mg = MemoryGuardian()
        est = mg.estimate_job_memory(str(fake_data), dict_path=str(fake_dict))
        assert est > 60  # data + dict + overhead

    def test_pre_flight_budget(self, tmp_path):
        f1 = tmp_path / "a.h5"
        f2 = tmp_path / "b.h5"
        f1.write_bytes(b"\x00" * (5 * 1024 * 1024))
        f2.write_bytes(b"\x00" * (10 * 1024 * 1024))
        mg = MemoryGuardian()
        report = mg.pre_flight_budget([str(f1), str(f2)], [])
        assert "fits_in_ram" in report
        assert "largest_file_mb" in report
        assert report["largest_file_mb"] >= 10

    def test_recommend_n_per_iteration(self):
        mg = MemoryGuardian()
        # With plenty of RAM, should return None (no chunking needed)
        n = mg.recommend_n_per_iteration(available_mb=8000, n_patterns=5000)
        assert n is None or n >= 5000
        # With very low RAM, should recommend chunking
        n = mg.recommend_n_per_iteration(available_mb=100, n_patterns=50000)
        assert n is not None
        assert n < 50000

    def test_force_cleanup_runs_gc(self):
        mg = MemoryGuardian()
        # Should not raise
        mg.force_cleanup()

    @patch("psutil.sensors_battery")
    def test_is_on_battery(self, mock_bat):
        mock_bat.return_value = MagicMock(power_plugged=False, percent=50)
        mg = MemoryGuardian()
        assert mg.is_on_battery() is True

    @patch("psutil.sensors_battery")
    def test_is_on_ac_power(self, mock_bat):
        mock_bat.return_value = MagicMock(power_plugged=True, percent=100)
        mg = MemoryGuardian()
        assert mg.is_on_battery() is False

    @patch("psutil.sensors_battery")
    def test_no_battery_desktop(self, mock_bat):
        mock_bat.return_value = None  # Desktop PC
        mg = MemoryGuardian()
        assert mg.is_on_battery() is False
