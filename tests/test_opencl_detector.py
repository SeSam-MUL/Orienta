"""
Test Suite for simulation/opencl_detector.py

Tests OpenCL device detection, EMsoft binary discovery, and compute
recommendation logic without invoking real subprocess calls.

Covered functions:
- parse_clinfo(raw_output)
- parse_clinfo_list(raw_output)
- detect_emsoft_binaries(bin_dir)
- compute_emsoft_devid(devices)
- recommend_device(status)
- auto_detect_nthreads()
- auto_detect_nthreads_4n3()
- validate_globalworkgrpsz(requested, device)
- detect_opencl(emsoft_bin_dir)
- validate_icd_files()
"""

import os
import sys
import stat
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest

IS_WINDOWS = sys.platform == "win32"

from simulation.opencl_detector import (
    OpenCLDevice,
    OpenCLStatus,
    parse_clinfo,
    parse_clinfo_list,
    detect_emsoft_binaries,
    compute_emsoft_devid,
    recommend_device,
    auto_detect_nthreads,
    auto_detect_nthreads_4n3,
    validate_globalworkgrpsz,
    detect_opencl,
    validate_icd_files,
    EMSOFT_BINARIES,
    _run_command as _real_run_command,
)


def _run_command_for_test_isolation(tmp_path: Path):
    """Return a _run_command replacement that isolates from real $HOME.

    The returned callable intercepts 'echo $HOME' to return a fake
    nonexistent path, preventing detect_emsoft_binaries from finding
    real binaries on the developer's machine. All other commands
    (test -x, which) are passed through to the real _run_command.
    """
    fake_home = str(tmp_path / "fake_home_nonexistent")

    def _fake_run_command(cmd: str, timeout: int = 10):
        if cmd.strip() == "echo $HOME":
            return (fake_home, True)
        if cmd.strip().startswith("which "):
            # Block 'which' from finding real binaries on PATH
            return ("", False)
        return _real_run_command(cmd, timeout)

    return _fake_run_command


# ---------------------------------------------------------------------------
# Sample clinfo output strings used across tests
# ---------------------------------------------------------------------------

# Full NVIDIA + PoCL system (1 platform, 2 devices: CPU + GPU)
CLINFO_NVIDIA_POCL = """Number of platforms                               1
  Platform Name                                   Portable Computing Language
  Platform Vendor                                 The pocl project
  Platform Version                                OpenCL 3.0 PoCL 7.2-pre

  Platform Name                                   Portable Computing Language
Number of devices                                 2
  Device Name                                     cpu-znver3-AMD Ryzen 9 5900X 12-Core Processor
  Device Vendor                                   AuthenticAMD
  Device Type                                     CPU
  Max compute units                               24
  Max clock frequency                             3700MHz
  Max work group size                             4096
  Global memory size                              29318336512
  Driver Version                                  7.2-pre

  Device Name                                     NVIDIA GeForce RTX 4070
  Device Vendor                                   NVIDIA Corporation
  Device Type                                     GPU
  Max compute units                               46
  Max clock frequency                             2475MHz
  Max work group size                             1024
  Global memory size                              12878086144
  Driver Version                                  560.35.03

NULL platform behavior
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_DEFAULT)  Success (1)
    Platform Name                                 Portable Computing Language
    Device Name                                   cpu-znver3-AMD Ryzen 9 5900X 12-Core Processor
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU)  Success (1)
    Platform Name                                 Portable Computing Language
    Device Name                                   NVIDIA GeForce RTX 4070
"""

# CPU-only setup (Intel CPU, no GPU)
CLINFO_CPU_ONLY = """Number of platforms                               1
  Platform Name                                   Portable Computing Language

  Platform Name                                   Portable Computing Language
Number of devices                                 1
  Device Name                                     Intel Core i7-1185G7
  Device Vendor                                   GenuineIntel
  Device Type                                     CPU
  Max compute units                               8
  Max clock frequency                             3000MHz
  Max work group size                             4096
  Global memory size                              16777216000
  Driver Version                                  4.0

NULL platform behavior
  clCreateContextFromType(NULL, CL_DEVICE_TYPE_DEFAULT)  Success (1)
    Platform Name                                 Portable Computing Language
    Device Name                                   Intel Core i7-1185G7
"""

# AMD GPU only (1 platform, 1 GPU)
CLINFO_AMD_GPU = """Number of platforms                               1
  Platform Name                                   AMD Accelerated Parallel Processing

  Platform Name                                   AMD Accelerated Parallel Processing
Number of devices                                 1
  Device Name                                     AMD Radeon RX 6800 XT
  Device Vendor                                   Advanced Micro Devices, Inc.
  Device Type                                     GPU
  Max compute units                               72
  Max clock frequency                             2310MHz
  Max work group size                             256
  Global memory size                              17179869184
  Driver Version                                  3451.5

NULL platform behavior
"""

# Multiple platforms: NVIDIA (GPU) + Intel (CPU), one platform each
CLINFO_MULTI_PLATFORM = """Number of platforms                               2
  Platform Name                                   NVIDIA CUDA
  Platform Vendor                                 NVIDIA Corporation

  Platform Name                                   NVIDIA CUDA
Number of devices                                 1
  Device Name                                     NVIDIA GeForce RTX 3090
  Device Vendor                                   NVIDIA Corporation
  Device Type                                     GPU
  Max compute units                               82
  Max clock frequency                             1695MHz
  Max work group size                             1024
  Global memory size                              25769803776
  Driver Version                                  520.61.05

  Platform Name                                   Intel(R) OpenCL HD Graphics
Number of devices                                 1
  Device Name                                     Intel(R) UHD Graphics 630
  Device Vendor                                   Intel(R) Corporation
  Device Type                                     GPU
  Max compute units                               24
  Max clock frequency                             1200MHz
  Max work group size                             256
  Global memory size                              12884901888
  Driver Version                                  22.43.24595

NULL platform behavior
"""

# Empty / no platforms
CLINFO_EMPTY = ""

# clinfo -l short output (NVIDIA + PoCL)
CLINFO_LIST = """Platform #0: Portable Computing Language
 +-- Device #0: cpu-znver3-AMD Ryzen 9 5900X 12-Core Processor
 `-- Device #1: NVIDIA GeForce RTX 4070"""

# clinfo -l with multiple platforms
CLINFO_LIST_MULTI = """Platform #0: NVIDIA CUDA
 `-- Device #0: NVIDIA GeForce RTX 3090
Platform #1: Portable Computing Language
 +-- Device #0: cpu-znver3-AMD Ryzen 9 5900X 12-Core Processor
 `-- Device #1: Some other device"""


# ---------------------------------------------------------------------------
# TestOpenCLDevice: dataclass properties
# ---------------------------------------------------------------------------

class TestOpenCLDevice:
    """Test OpenCLDevice dataclass and its computed properties."""

    def _make_gpu(self, **kwargs) -> OpenCLDevice:
        defaults = dict(
            platform_index=0,
            device_index=0,
            name="NVIDIA GeForce RTX 4070",
            device_type="GPU",
            vendor="NVIDIA Corporation",
            global_memory_bytes=12_878_086_144,  # ~12 GB
            max_work_group_size=1024,
            max_compute_units=46,
        )
        defaults.update(kwargs)
        return OpenCLDevice(**defaults)

    def _make_cpu(self, **kwargs) -> OpenCLDevice:
        defaults = dict(
            platform_index=0,
            device_index=0,
            name="AMD Ryzen 9 5900X 12-Core Processor",
            device_type="CPU",
            max_work_group_size=4096,
        )
        defaults.update(kwargs)
        return OpenCLDevice(**defaults)

    def test_is_gpu_true_for_gpu_device(self):
        """Device with type GPU should report is_gpu=True."""
        gpu = self._make_gpu()
        assert gpu.is_gpu is True

    def test_is_gpu_false_for_cpu_device(self):
        """Device with type CPU should report is_gpu=False."""
        cpu = self._make_cpu()
        assert cpu.is_gpu is False

    def test_is_gpu_case_insensitive(self):
        """is_gpu should work regardless of device_type casing."""
        dev_lower = OpenCLDevice(0, 0, "TestGPU", device_type="gpu")
        dev_mixed = OpenCLDevice(0, 0, "TestGPU", device_type="Gpu")
        assert dev_lower.is_gpu is True
        assert dev_mixed.is_gpu is True

    def test_global_memory_gb_calculation(self):
        """global_memory_gb should convert bytes to GiB correctly."""
        gpu = self._make_gpu(global_memory_bytes=12_884_901_888)  # 12 GiB exactly
        assert gpu.global_memory_gb == pytest.approx(12.0, rel=0.01)

    def test_global_memory_gb_zero_when_unset(self):
        """Unset memory (0 bytes) should give 0.0 GB."""
        dev = OpenCLDevice(0, 0, "NullDev", device_type="GPU", global_memory_bytes=0)
        assert dev.global_memory_gb == 0.0

    def test_summary_includes_name_and_memory(self):
        """summary property should include device name and GB figure."""
        gpu = self._make_gpu(global_memory_bytes=12_884_901_888, max_compute_units=46)
        summary = gpu.summary
        assert "NVIDIA GeForce RTX 4070" in summary
        assert "GB" in summary
        assert "46" in summary

    def test_summary_shows_unknown_memory_when_zero(self):
        """summary should show '? GB' when memory is 0."""
        dev = OpenCLDevice(0, 0, "TestDev", device_type="GPU", global_memory_bytes=0)
        assert "? GB" in dev.summary


# ---------------------------------------------------------------------------
# TestOpenCLStatus: status aggregation properties
# ---------------------------------------------------------------------------

class TestOpenCLStatus:
    """Test OpenCLStatus dataclass and its properties."""

    def test_has_gpu_true_with_gpu_devices(self):
        """has_gpu should be True when gpu_devices is non-empty."""
        status = OpenCLStatus()
        status.gpu_devices.append(
            OpenCLDevice(0, 0, "RTX 4070", device_type="GPU")
        )
        assert status.has_gpu is True

    def test_has_gpu_false_when_empty(self):
        """has_gpu should be False on a fresh status."""
        status = OpenCLStatus()
        assert status.has_gpu is False

    def test_recommended_mode_gpu_when_gpu_present(self):
        """recommended_mode should be 'gpu' when gpu_devices populated."""
        status = OpenCLStatus()
        status.gpu_devices.append(OpenCLDevice(0, 0, "RTX", device_type="GPU"))
        assert status.recommended_mode == "gpu"

    def test_recommended_mode_cpu_when_only_cpu(self):
        """recommended_mode should be 'cpu' when only cpu_devices."""
        status = OpenCLStatus()
        status.cpu_devices.append(OpenCLDevice(0, 0, "i7", device_type="CPU"))
        assert status.recommended_mode == "cpu"

    def test_recommended_mode_none_when_empty(self):
        """recommended_mode should be 'none' when no devices."""
        status = OpenCLStatus()
        assert status.recommended_mode == "none"


# ---------------------------------------------------------------------------
# TestParseClinfo: full clinfo output parser
# ---------------------------------------------------------------------------

class TestParseClinfo:
    """Test parse_clinfo() with various clinfo output strings."""

    def test_nvidia_pocl_returns_one_platform(self):
        """NVIDIA+PoCL output should yield exactly one platform."""
        platforms, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        assert len(platforms) == 1
        assert "Portable Computing Language" in platforms[0]

    def test_nvidia_pocl_returns_two_devices(self):
        """NVIDIA+PoCL output should yield 2 devices."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        assert len(devices) == 2

    def test_nvidia_pocl_first_device_is_cpu(self):
        """First device should be the AMD Ryzen CPU."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        cpu = devices[0]
        assert cpu.device_type == "CPU"
        assert "5900X" in cpu.name

    def test_nvidia_pocl_second_device_is_gpu(self):
        """Second device should be the NVIDIA RTX 4070."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        gpu = devices[1]
        assert gpu.device_type == "GPU"
        assert "RTX 4070" in gpu.name

    def test_nvidia_pocl_gpu_memory(self):
        """RTX 4070 global memory should parse to ~12 GB."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        gpu = next(d for d in devices if d.is_gpu)
        assert gpu.global_memory_bytes == 12_878_086_144

    def test_nvidia_pocl_gpu_work_group_size(self):
        """RTX 4070 max_work_group_size should be 1024."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        gpu = next(d for d in devices if d.is_gpu)
        assert gpu.max_work_group_size == 1024

    def test_nvidia_pocl_gpu_compute_units(self):
        """RTX 4070 max_compute_units should be 46."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        gpu = next(d for d in devices if d.is_gpu)
        assert gpu.max_compute_units == 46

    def test_nvidia_pocl_gpu_clock_mhz(self):
        """RTX 4070 max_clock_mhz should be 2475."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        gpu = next(d for d in devices if d.is_gpu)
        assert gpu.max_clock_mhz == 2475

    def test_nvidia_pocl_gpu_driver_version(self):
        """RTX 4070 driver version should parse correctly."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        gpu = next(d for d in devices if d.is_gpu)
        assert "560" in gpu.driver_version

    def test_nvidia_pocl_gpu_vendor(self):
        """RTX 4070 vendor should be NVIDIA Corporation."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        gpu = next(d for d in devices if d.is_gpu)
        assert "NVIDIA" in gpu.vendor

    def test_null_platform_section_not_parsed_as_devices(self):
        """Devices listed under 'NULL platform behavior' must not be counted."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        # Only 2 real devices — no extras from the NULL section
        assert len(devices) == 2

    def test_cpu_only_output(self):
        """CPU-only clinfo output should yield 1 CPU device and no GPU."""
        platforms, devices = parse_clinfo(CLINFO_CPU_ONLY)
        assert len(devices) == 1
        assert devices[0].device_type == "CPU"
        assert not any(d.is_gpu for d in devices)

    def test_cpu_only_platform_name(self):
        """CPU-only platform should be 'Portable Computing Language'."""
        platforms, _ = parse_clinfo(CLINFO_CPU_ONLY)
        assert len(platforms) == 1
        assert "Portable Computing Language" in platforms[0]

    def test_amd_gpu_parsed_correctly(self):
        """AMD GPU output: 1 platform, 1 GPU device."""
        platforms, devices = parse_clinfo(CLINFO_AMD_GPU)
        assert len(platforms) == 1
        assert len(devices) == 1
        assert devices[0].is_gpu
        assert "Radeon RX 6800 XT" in devices[0].name

    def test_amd_gpu_memory(self):
        """AMD RX 6800 XT global memory should be 16 GiB."""
        _, devices = parse_clinfo(CLINFO_AMD_GPU)
        assert devices[0].global_memory_bytes == 17_179_869_184

    def test_multi_platform_yields_two_platforms(self):
        """Multi-platform output should yield 2 distinct platform names."""
        platforms, devices = parse_clinfo(CLINFO_MULTI_PLATFORM)
        assert len(platforms) == 2

    def test_multi_platform_yields_two_devices(self):
        """Multi-platform output should yield 2 devices total."""
        _, devices = parse_clinfo(CLINFO_MULTI_PLATFORM)
        assert len(devices) == 2

    def test_multi_platform_second_platform_index(self):
        """Device from second platform should have platform_index=1."""
        _, devices = parse_clinfo(CLINFO_MULTI_PLATFORM)
        # Intel UHD is listed under 'Intel(R) OpenCL HD Graphics' (second)
        intel_dev = next(d for d in devices if "UHD" in d.name)
        assert intel_dev.platform_index == 1

    def test_empty_output_returns_empty_lists(self):
        """Empty clinfo output should return ([], [])."""
        platforms, devices = parse_clinfo(CLINFO_EMPTY)
        assert platforms == []
        assert devices == []

    def test_whitespace_only_returns_empty_lists(self):
        """Whitespace-only clinfo output should return ([], [])."""
        platforms, devices = parse_clinfo("   \n  \t  \n")
        assert platforms == []
        assert devices == []

    def test_device_platform_index_assigned_to_first_platform(self):
        """Both devices in NVIDIA_POCL should have platform_index=0."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        for dev in devices:
            assert dev.platform_index == 0

    def test_device_index_within_platform(self):
        """Devices should be numbered 0, 1 within their platform."""
        _, devices = parse_clinfo(CLINFO_NVIDIA_POCL)
        assert devices[0].device_index == 0
        assert devices[1].device_index == 1


# ---------------------------------------------------------------------------
# TestParseClinfoList: clinfo -l short output parser
# ---------------------------------------------------------------------------

class TestParseClinfoList:
    """Test parse_clinfo_list() with clinfo -l output."""

    def test_pocl_list_returns_one_platform(self):
        """clinfo -l output should yield 1 platform."""
        platforms, devices = parse_clinfo_list(CLINFO_LIST)
        assert len(platforms) == 1
        assert "Portable Computing Language" in platforms[0]

    def test_pocl_list_returns_two_devices(self):
        """clinfo -l output should yield 2 devices."""
        _, devices = parse_clinfo_list(CLINFO_LIST)
        assert len(devices) == 2

    def test_pocl_list_nvidia_detected_as_gpu(self):
        """NVIDIA GeForce in device name should be inferred as GPU."""
        _, devices = parse_clinfo_list(CLINFO_LIST)
        gpu = next(d for d in devices if "RTX" in d.name)
        assert gpu.device_type == "GPU"

    def test_pocl_list_amd_ryzen_detected_as_cpu(self):
        """CPU processor in device name should be inferred as CPU."""
        _, devices = parse_clinfo_list(CLINFO_LIST)
        cpu = next(d for d in devices if "5900X" in d.name)
        assert cpu.device_type == "CPU"

    def test_pocl_list_device_indices(self):
        """Device indices should match the numbers in clinfo -l output."""
        _, devices = parse_clinfo_list(CLINFO_LIST)
        names = {d.device_index: d.name for d in devices}
        assert 0 in names  # first device
        assert 1 in names  # second device

    def test_multi_platform_list_yields_two_platforms(self):
        """Multi-platform clinfo -l should yield 2 platforms."""
        platforms, devices = parse_clinfo_list(CLINFO_LIST_MULTI)
        assert len(platforms) == 2

    def test_multi_platform_list_device_count(self):
        """Multi-platform clinfo -l should yield 3 devices total."""
        _, devices = parse_clinfo_list(CLINFO_LIST_MULTI)
        assert len(devices) == 3

    def test_multi_platform_list_rtx3090_is_gpu(self):
        """RTX 3090 in multi-platform list should be GPU."""
        _, devices = parse_clinfo_list(CLINFO_LIST_MULTI)
        gpu = next(d for d in devices if "RTX 3090" in d.name)
        assert gpu.device_type == "GPU"

    def test_empty_list_returns_empty(self):
        """Empty clinfo -l output should return ([], [])."""
        platforms, devices = parse_clinfo_list("")
        assert platforms == []
        assert devices == []

    def test_unknown_device_type_fallback(self):
        """Unrecognized device names should get UNKNOWN type, not crash."""
        raw = """Platform #0: SomeOpenCL
 `-- Device #0: XYZ Accelerator 9000"""
        _, devices = parse_clinfo_list(raw)
        assert len(devices) == 1
        assert devices[0].device_type == "UNKNOWN"


# ---------------------------------------------------------------------------
# TestDetectEmsoftBinaries: filesystem binary detection
# ---------------------------------------------------------------------------

class TestDetectEmsoftBinaries:
    """Test detect_emsoft_binaries() using tmp_path for isolation."""

    def _make_executable(self, path: Path) -> None:
        """Create a file and mark it executable."""
        path.touch()
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    @pytest.mark.skipif(
        IS_WINDOWS,
        reason="detect_emsoft_binaries uses Unix executable-bit semantics; "
        "Windows chmod(S_IXUSR) is a no-op and os.access(X_OK) keys off the "
        ".exe extension, not the test's mode bits. The companion "
        "test_missing_binaries_return_empty_string already skips on Windows "
        "for the same reason — this one was missed when the marker was added.",
    )
    def test_all_binaries_found_when_all_present(self, tmp_path):
        """All 5 EMsoft binaries present and executable should all be found."""
        bin_dir = tmp_path / "Bin"
        bin_dir.mkdir()
        for name in EMSOFT_BINARIES:
            self._make_executable(bin_dir / name)

        result = detect_emsoft_binaries(bin_dir=bin_dir)

        assert len(result) == 5
        for name in EMSOFT_BINARIES:
            assert result[name] != "", f"{name} should be found"
            assert name in result[name]  # path contains the binary name

    @pytest.mark.skipif(IS_WINDOWS, reason="WSL binary detection uses Unix paths not available in Windows tmp")
    def test_missing_binaries_return_empty_string(self, tmp_path):
        """Missing binaries should map to empty string when default dirs are patched away."""
        bin_dir = tmp_path / "Bin"
        bin_dir.mkdir()
        # Only create EMMCOpenCL
        self._make_executable(bin_dir / "EMMCOpenCL")

        # Patch _run_command so echo $HOME returns a fake nonexistent path,
        # preventing the function from finding real binaries via $HOME fallback.
        # Also let the 'test -x' and 'which' calls work normally for our tmp dir.
        original_run = _run_command_for_test_isolation(tmp_path)
        with patch("simulation.opencl_detector._run_command", side_effect=original_run):
            result = detect_emsoft_binaries(bin_dir=bin_dir)

        assert result["EMMCOpenCL"] != ""
        assert result["EMMC"] == ""
        assert result["EMEBSDmaster"] == ""
        assert result["EMEBSDmasterOpenCL"] == ""
        assert result["EMEBSDmasterSHT"] == ""

    def test_non_executable_file_not_detected(self, tmp_path):
        """A non-executable file should not be returned as found."""
        bin_dir = tmp_path / "Bin"
        bin_dir.mkdir()
        # Create file but do NOT mark it executable
        non_exec = bin_dir / "EMMCOpenCL"
        non_exec.touch()
        # Remove execute bit to be sure
        non_exec.chmod(0o644)

        # Patch _run_command so echo $HOME returns a fake path,
        # preventing fallback to real $HOME directories.
        original_run = _run_command_for_test_isolation(tmp_path)
        with patch("simulation.opencl_detector._run_command", side_effect=original_run):
            result = detect_emsoft_binaries(bin_dir=bin_dir)

        assert result["EMMCOpenCL"] == ""

    def test_result_keys_match_emsoft_binaries_dict(self, tmp_path):
        """detect_emsoft_binaries should always return all 5 keys."""
        result = detect_emsoft_binaries(bin_dir=tmp_path / "nonexistent")
        assert set(result.keys()) == set(EMSOFT_BINARIES.keys())

    @pytest.mark.skipif(IS_WINDOWS, reason="WSL binary paths not recognized as absolute on Windows")
    def test_returns_correct_path_string(self, tmp_path):
        """Found binary path should be a valid absolute path string."""
        bin_dir = tmp_path / "Bin"
        bin_dir.mkdir()
        self._make_executable(bin_dir / "EMMC")

        result = detect_emsoft_binaries(bin_dir=bin_dir)

        path_str = result["EMMC"]
        assert Path(path_str).is_absolute()
        assert Path(path_str).exists()

    def test_none_bin_dir_does_not_crash(self):
        """Passing None as bin_dir should not raise (falls back to defaults)."""
        # This may return empty results if default dirs don't exist, but must not crash
        result = detect_emsoft_binaries(bin_dir=None)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(EMSOFT_BINARIES.keys())


# ---------------------------------------------------------------------------
# TestComputeEmsoftDevid: GPU device ID for EMsoft
# ---------------------------------------------------------------------------

class TestComputeEmsoftDevid:
    """Test compute_emsoft_devid() — EMsoft counts only GPU devices, 1-based."""

    def _cpu(self, idx=0) -> OpenCLDevice:
        return OpenCLDevice(0, idx, f"CPU-{idx}", device_type="CPU")

    def _gpu(self, idx=0) -> OpenCLDevice:
        return OpenCLDevice(0, idx, f"GPU-{idx}", device_type="GPU")

    def test_single_gpu_returns_1(self):
        """Single GPU in device list should yield emsoft_devid=1."""
        devices = [self._gpu(0)]
        assert compute_emsoft_devid(devices) == 1

    def test_cpu_then_gpu_returns_1(self):
        """CPU before GPU: EMsoft still sees GPU as devid=1."""
        devices = [self._cpu(0), self._gpu(1)]
        assert compute_emsoft_devid(devices) == 1

    def test_gpu_then_cpu_returns_1(self):
        """GPU before CPU: first GPU is still devid=1."""
        devices = [self._gpu(0), self._cpu(1)]
        assert compute_emsoft_devid(devices) == 1

    def test_two_gpus_returns_1_for_first(self):
        """With two GPUs, the first one gets devid=1."""
        devices = [self._gpu(0), self._gpu(1)]
        assert compute_emsoft_devid(devices) == 1

    def test_no_gpu_returns_0(self):
        """No GPU devices should return 0 (invalid devid signal)."""
        devices = [self._cpu(0)]
        assert compute_emsoft_devid(devices) == 0

    def test_empty_device_list_returns_0(self):
        """Empty device list should return 0."""
        assert compute_emsoft_devid([]) == 0

    def test_cpu_only_list_returns_0(self):
        """All-CPU list should return 0."""
        devices = [self._cpu(0), self._cpu(1)]
        assert compute_emsoft_devid(devices) == 0


# ---------------------------------------------------------------------------
# TestAutoDetectNthreads: CPU thread count calculations
# ---------------------------------------------------------------------------

class TestAutoDetectNthreads:
    """Test auto_detect_nthreads() and auto_detect_nthreads_4n3()."""

    def test_nthreads_12_core_system(self):
        """12-core CPU: min(12-2, 12) = 10."""
        with patch("os.cpu_count", return_value=12):
            assert auto_detect_nthreads() == 10

    def test_nthreads_24_core_system(self):
        """24-core CPU: min(24-2, 12) = 12 (capped at 12)."""
        with patch("os.cpu_count", return_value=24):
            assert auto_detect_nthreads() == 12

    def test_nthreads_4_core_system(self):
        """4-core CPU: min(4-2, 12) = 2."""
        with patch("os.cpu_count", return_value=4):
            assert auto_detect_nthreads() == 2

    def test_nthreads_2_core_system(self):
        """2-core CPU: min(2-2, 12) = 0, but clamped to 1."""
        with patch("os.cpu_count", return_value=2):
            assert auto_detect_nthreads() == 1

    def test_nthreads_1_core_system(self):
        """1-core CPU: min(1-2, 12) = -1, clamped to 1."""
        with patch("os.cpu_count", return_value=1):
            assert auto_detect_nthreads() == 1

    def test_nthreads_none_cpu_count(self):
        """When os.cpu_count() returns None, fallback to 4."""
        with patch("os.cpu_count", return_value=None):
            result = auto_detect_nthreads()
            # min(4-2, 12) = 2
            assert result == 2

    def test_nthreads_result_always_at_least_1(self):
        """Result should never be less than 1."""
        for cores in (1, 2, 3):
            with patch("os.cpu_count", return_value=cores):
                assert auto_detect_nthreads() >= 1

    def test_nthreads_result_never_exceeds_12(self):
        """Result should never exceed 12."""
        for cores in (14, 32, 64, 128):
            with patch("os.cpu_count", return_value=cores):
                assert auto_detect_nthreads() <= 12

    def test_4n3_12_core_system(self):
        """12-core CPU target=10, nearest 4N+3 <= 10 is 7."""
        with patch("os.cpu_count", return_value=12):
            result = auto_detect_nthreads_4n3()
            assert result == 7

    def test_4n3_is_valid_form(self):
        """Result must satisfy (n - 3) % 4 == 0."""
        for cores in (8, 12, 16, 24, 32):
            with patch("os.cpu_count", return_value=cores):
                result = auto_detect_nthreads_4n3()
                assert (result - 3) % 4 == 0, f"Result {result} is not 4N+3"

    def test_4n3_minimum_is_7(self):
        """Result should never be less than 7."""
        for cores in (1, 2, 4, 8):
            with patch("os.cpu_count", return_value=cores):
                assert auto_detect_nthreads_4n3() >= 7

    def test_4n3_none_cpu_count_fallback(self):
        """None cpu_count should fall back gracefully, returning >= 7."""
        with patch("os.cpu_count", return_value=None):
            result = auto_detect_nthreads_4n3()
            assert result >= 7
            assert (result - 3) % 4 == 0


# ---------------------------------------------------------------------------
# TestValidateGlobalworkgrpsz: workgroup size clamping
# ---------------------------------------------------------------------------

class TestValidateGlobalworkgrpsz:
    """Test validate_globalworkgrpsz() clamping behavior."""

    def _gpu(self, max_wgs: int) -> OpenCLDevice:
        return OpenCLDevice(
            0, 0, "Test GPU", device_type="GPU", max_work_group_size=max_wgs
        )

    def test_requested_within_limit_returned_unchanged(self):
        """Requested value within device max should pass through."""
        gpu = self._gpu(max_wgs=1024)
        assert validate_globalworkgrpsz(150, gpu) == 150

    def test_requested_over_limit_clamped_to_max(self):
        """Requested value exceeding device max should be clamped."""
        gpu = self._gpu(max_wgs=256)
        assert validate_globalworkgrpsz(512, gpu) == 256

    def test_requested_at_exact_limit_passes(self):
        """Requested value exactly at device max should be returned."""
        gpu = self._gpu(max_wgs=1024)
        assert validate_globalworkgrpsz(1024, gpu) == 1024

    def test_no_device_returns_requested_value(self):
        """With no device, requested value should be returned as-is."""
        assert validate_globalworkgrpsz(200, None) == 200

    def test_device_with_zero_max_wgs_not_limiting(self):
        """Device with max_work_group_size=0 should not limit the value."""
        gpu = self._gpu(max_wgs=0)
        assert validate_globalworkgrpsz(300, gpu) == 300

    def test_negative_requested_clamped_to_1(self):
        """Negative requested value should be clamped to minimum 1."""
        assert validate_globalworkgrpsz(-5, None) == 1

    def test_zero_requested_clamped_to_1(self):
        """Zero requested value should be clamped to 1."""
        gpu = self._gpu(max_wgs=1024)
        assert validate_globalworkgrpsz(0, gpu) == 1


# ---------------------------------------------------------------------------
# TestRecommendDevice: compute recommendation logic
# ---------------------------------------------------------------------------

class TestRecommendDevice:
    """Test recommend_device() output for various hardware configurations."""

    def _make_status_with_gpu(
        self,
        gpu_memory_bytes: int = 12_878_086_144,
        gpu_max_wgs: int = 1024,
        has_emmc: bool = True,
        has_emebsdmaster: bool = True,
        has_emmcopencl: bool = True,
    ) -> OpenCLStatus:
        status = OpenCLStatus(available=True)
        cpu = OpenCLDevice(0, 0, "AMD Ryzen 5900X", device_type="CPU")
        gpu = OpenCLDevice(
            0, 1, "NVIDIA RTX 4070", device_type="GPU",
            global_memory_bytes=gpu_memory_bytes,
            max_work_group_size=gpu_max_wgs,
        )
        status.devices = [cpu, gpu]
        status.cpu_devices = [cpu]
        status.gpu_devices = [gpu]
        status.binaries = {
            "EMMCOpenCL": "/bin/EMMCOpenCL" if has_emmcopencl else "",
            "EMMC": "/bin/EMMC" if has_emmc else "",
            "EMEBSDmaster": "/bin/EMEBSDmaster" if has_emebsdmaster else "",
            "EMEBSDmasterOpenCL": "",
            "EMEBSDmasterSHT": "/bin/EMEBSDmasterSHT",
        }
        return status

    def _make_status_cpu_only(self) -> OpenCLStatus:
        status = OpenCLStatus(available=True)
        cpu = OpenCLDevice(0, 0, "Intel i7-1185G7", device_type="CPU")
        status.devices = [cpu]
        status.cpu_devices = [cpu]
        status.binaries = {
            "EMMCOpenCL": "",
            "EMMC": "/bin/EMMC",
            "EMEBSDmaster": "/bin/EMEBSDmaster",
            "EMEBSDmasterOpenCL": "",
            "EMEBSDmasterSHT": "/bin/EMEBSDmasterSHT",
        }
        return status

    def _make_status_unavailable(self) -> OpenCLStatus:
        status = OpenCLStatus(available=False)
        status.binaries = {k: "" for k in EMSOFT_BINARIES}
        return status

    # --- mode selection ---

    def test_gpu_status_recommends_gpu_mode(self):
        """GPU present → mode should be 'gpu'."""
        status = self._make_status_with_gpu()
        rec = recommend_device(status)
        assert rec["mode"] == "gpu"

    def test_cpu_only_recommends_cpu_mode(self):
        """No GPU → mode should be 'cpu'."""
        status = self._make_status_cpu_only()
        rec = recommend_device(status)
        assert rec["mode"] == "cpu"

    def test_unavailable_recommends_cpu_mode_with_warning(self):
        """Unavailable OpenCL → mode 'cpu' and warning present."""
        status = self._make_status_unavailable()
        rec = recommend_device(status)
        assert rec["mode"] == "cpu"
        assert len(rec["warnings"]) > 0

    # --- program selection ---

    def test_gpu_status_selects_emmcopencl_for_mc(self):
        """GPU present → mc_program should be EMMCOpenCL."""
        status = self._make_status_with_gpu()
        rec = recommend_device(status)
        assert rec["mc_program"] == "EMMCOpenCL"

    def test_cpu_only_selects_emmc_for_mc(self):
        """No GPU → mc_program should be EMMC."""
        status = self._make_status_cpu_only()
        rec = recommend_device(status)
        assert rec["mc_program"] == "EMMC"

    def test_gpu_below_16gb_uses_cpu_master(self):
        """GPU with < 16 GB RAM → master_program = EMEBSDmaster (CPU fallback)."""
        status = self._make_status_with_gpu(gpu_memory_bytes=12_878_086_144)  # ~12 GB
        rec = recommend_device(status)
        assert rec["master_program"] == "EMEBSDmaster"

    def test_gpu_above_16gb_uses_gpu_master(self):
        """GPU with >= 16 GB RAM → master_program = EMEBSDmasterOpenCL."""
        # 17 GiB
        status = self._make_status_with_gpu(gpu_memory_bytes=18_253_611_008)
        # But EMEBSDmasterOpenCL is empty in our helper, so override
        status.binaries["EMEBSDmasterOpenCL"] = "/bin/EMEBSDmasterOpenCL"
        rec = recommend_device(status)
        assert rec["master_program"] == "EMEBSDmasterOpenCL"

    def test_gpu_below_16gb_adds_warning(self):
        """GPU with < 16 GB RAM → warning about EMEBSDmasterOpenCL."""
        status = self._make_status_with_gpu(gpu_memory_bytes=8_589_934_592)  # 8 GB
        rec = recommend_device(status)
        warning_text = " ".join(rec["warnings"])
        assert "16 GB" in warning_text or "GB" in warning_text

    def test_cpu_only_adds_no_gpu_warning(self):
        """CPU-only setup should warn that no GPU was found."""
        status = self._make_status_cpu_only()
        rec = recommend_device(status)
        warning_text = " ".join(rec["warnings"])
        assert "GPU" in warning_text or "cpu" in warning_text.lower()

    # --- emsoft_devid ---

    def test_gpu_status_emsoft_devid_is_1(self):
        """First GPU should get emsoft_devid=1."""
        status = self._make_status_with_gpu()
        rec = recommend_device(status)
        assert rec["emsoft_devid"] == 1

    def test_cpu_only_emsoft_devid_is_0(self):
        """No GPU → emsoft_devid should be 0."""
        status = self._make_status_cpu_only()
        rec = recommend_device(status)
        assert rec["emsoft_devid"] == 0

    # --- workgroup size ---

    def test_gpu_with_small_wgs_limit_applied(self):
        """GPU with max_work_group_size=128 should clamp globalworkgrpsz to 128."""
        status = self._make_status_with_gpu(gpu_max_wgs=128)
        rec = recommend_device(status)
        assert rec["globalworkgrpsz"] <= 128

    def test_gpu_with_large_wgs_uses_default_150(self):
        """GPU with max_work_group_size=2048 → default 150 should be kept."""
        status = self._make_status_with_gpu(gpu_max_wgs=2048)
        rec = recommend_device(status)
        assert rec["globalworkgrpsz"] == 150

    # --- binary fallback ---

    def test_missing_emmcopencl_falls_back_to_emmc(self):
        """If EMMCOpenCL is missing and EMMC exists, fall back to EMMC."""
        status = self._make_status_with_gpu(has_emmcopencl=False, has_emmc=True)
        rec = recommend_device(status)
        # Should warn and fall back
        assert rec["mc_program"] == "EMMC"
        assert rec["mode"] == "cpu"

    def test_all_required_keys_present(self):
        """recommend_device must always return all expected keys."""
        expected_keys = {
            "mode", "platid", "devid", "emsoft_devid", "globalworkgrpsz",
            "nthreads", "nthreads_master_opencl", "mc_program",
            "master_program", "sht_program", "warnings",
        }
        status = self._make_status_cpu_only()
        rec = recommend_device(status)
        assert expected_keys.issubset(rec.keys())

    def test_nthreads_is_positive_integer(self):
        """nthreads in recommendation should be a positive integer."""
        status = self._make_status_cpu_only()
        rec = recommend_device(status)
        assert isinstance(rec["nthreads"], int)
        assert rec["nthreads"] >= 1

    def test_nthreads_4n3_is_valid(self):
        """nthreads_master_opencl must be 4N+3 form and >= 7."""
        status = self._make_status_cpu_only()
        rec = recommend_device(status)
        n = rec["nthreads_master_opencl"]
        assert n >= 7
        assert (n - 3) % 4 == 0


# ---------------------------------------------------------------------------
# TestDetectOpencl: main entry point with mocked subprocess
# ---------------------------------------------------------------------------

class TestDetectOpencl:
    """Test detect_opencl() mocking the subprocess calls."""

    @patch("simulation.opencl_detector._run_command")
    def test_detect_opencl_nvidia_pocl_full(self, mock_run, tmp_path):
        """Full clinfo output from NVIDIA+PoCL should produce valid status."""
        mock_run.return_value = (CLINFO_NVIDIA_POCL, True)

        status = detect_opencl(emsoft_bin_dir=tmp_path)

        assert status.available is True
        assert len(status.platforms) == 1
        assert len(status.devices) == 2
        assert len(status.gpu_devices) == 1
        assert len(status.cpu_devices) == 1

    @patch("simulation.opencl_detector._run_command")
    def test_detect_opencl_cpu_only(self, mock_run, tmp_path):
        """CPU-only clinfo should produce status with no GPU."""
        mock_run.return_value = (CLINFO_CPU_ONLY, True)

        status = detect_opencl(emsoft_bin_dir=tmp_path)

        assert status.available is True
        assert status.has_gpu is False
        assert len(status.cpu_devices) == 1

    @patch("simulation.opencl_detector.detect_emsoft_binaries")
    @patch("simulation.opencl_detector._run_command")
    def test_detect_opencl_fallback_to_clinfo_list(self, mock_run, mock_binaries, tmp_path):
        """When full clinfo fails, clinfo -l should be tried as fallback."""
        # First call (clinfo) fails, second call (clinfo -l) succeeds
        mock_run.side_effect = [
            ("", False),           # clinfo fails
            (CLINFO_LIST, True),   # clinfo -l succeeds
        ]
        mock_binaries.return_value = {name: "" for name in EMSOFT_BINARIES}

        status = detect_opencl(emsoft_bin_dir=tmp_path)

        assert status.available is True
        assert len(status.devices) == 2

    @patch("simulation.opencl_detector._run_command")
    def test_detect_opencl_no_opencl_available(self, mock_run, tmp_path):
        """Both clinfo calls failing should result in status.available=False."""
        mock_run.return_value = ("", False)

        status = detect_opencl(emsoft_bin_dir=tmp_path)

        assert status.available is False
        assert status.error != ""

    @patch("simulation.opencl_detector._run_command")
    def test_detect_opencl_sets_cpu_count(self, mock_run, tmp_path):
        """detect_opencl should record os.cpu_count() in status.cpu_count."""
        mock_run.return_value = (CLINFO_CPU_ONLY, True)

        with patch("os.cpu_count", return_value=8):
            status = detect_opencl(emsoft_bin_dir=tmp_path)

        assert status.cpu_count == 8

    @pytest.mark.skipif(IS_WINDOWS, reason="WSL path translation not available on Windows")
    @patch("simulation.opencl_detector._run_command")
    def test_detect_opencl_records_emsoft_bin_dir(self, mock_run, tmp_path):
        """If a binary is found, emsoft_bin_dir should be recorded."""
        # Create an executable binary in tmp_path so detect_emsoft_binaries finds it
        import stat as _stat
        bin_path = tmp_path / "EMMC"
        bin_path.touch()
        bin_path.chmod(bin_path.stat().st_mode | _stat.S_IXUSR)

        fake_home = str(tmp_path / "fake_home_nonexistent")

        def side_effect(cmd, timeout=10):
            if cmd.strip() == "clinfo":
                return (CLINFO_CPU_ONLY, True)
            if cmd.strip() == "echo $HOME":
                return (fake_home, True)
            if cmd.strip().startswith("which "):
                return ("", False)
            # Let 'test -x' calls run for real
            return _real_run_command(cmd, timeout)

        mock_run.side_effect = side_effect

        status = detect_opencl(emsoft_bin_dir=tmp_path)

        assert status.emsoft_bin_dir == str(tmp_path)

    @patch("simulation.opencl_detector.detect_emsoft_binaries")
    @patch("simulation.opencl_detector._run_command")
    def test_detect_opencl_empty_output_not_available(self, mock_run, mock_binaries, tmp_path):
        """Empty string from both clinfo calls → available=False."""
        mock_run.side_effect = [
            ("", False),
            ("", False),
        ]
        mock_binaries.return_value = {name: "" for name in EMSOFT_BINARIES}
        status = detect_opencl(emsoft_bin_dir=tmp_path)
        assert status.available is False

    @patch("simulation.opencl_detector._run_command")
    def test_detect_opencl_gpu_classified_correctly(self, mock_run, tmp_path):
        """GPU device from NVIDIA+PoCL output should land in gpu_devices."""
        mock_run.return_value = (CLINFO_NVIDIA_POCL, True)

        status = detect_opencl(emsoft_bin_dir=tmp_path)

        assert any("RTX 4070" in d.name for d in status.gpu_devices)
        assert not any("RTX 4070" in d.name for d in status.cpu_devices)

    @patch("simulation.opencl_detector._run_command")
    def test_detect_opencl_amd_gpu(self, mock_run, tmp_path):
        """AMD GPU clinfo output should yield 1 GPU and no CPU."""
        mock_run.return_value = (CLINFO_AMD_GPU, True)

        status = detect_opencl(emsoft_bin_dir=tmp_path)

        assert status.has_gpu is True
        assert len(status.cpu_devices) == 0


# ---------------------------------------------------------------------------
# TestValidateIcdFiles: /etc/OpenCL/vendors/*.icd validation
# ---------------------------------------------------------------------------

class TestValidateIcdFiles:
    """Test validate_icd_files() by patching the filesystem path."""

    def test_valid_nvidia_and_pocl_icd(self, tmp_path):
        """Valid nvidia.icd and pocl.icd should both report valid=True."""
        icd_dir = tmp_path / "vendors"
        icd_dir.mkdir()
        (icd_dir / "nvidia.icd").write_text("libnvidia-opencl.so.1\n")
        (icd_dir / "pocl.icd").write_text("libpocl.so\n")

        with patch("simulation.opencl_detector.Path") as MockPath:
            # Make Path("/etc/OpenCL/vendors") return our temp dir
            def path_side_effect(arg):
                if arg == "/etc/OpenCL/vendors":
                    real = type(Path())  # actual Path class
                    p = real(icd_dir)
                    return p
                return Path(arg)
            MockPath.side_effect = path_side_effect

            # Call directly with patched filesystem using real Path but different dir
            # We'll use a simpler approach: patch Path inside the module
            pass

        # Simpler: monkey-patch the icd_dir used in validate_icd_files
        with patch("simulation.opencl_detector.Path") as MockPath:
            mock_icd_path = MagicMock()
            mock_icd_path.exists.return_value = True
            mock_icd_path.glob.return_value = [
                icd_dir / "nvidia.icd",
                icd_dir / "pocl.icd",
            ]
            MockPath.return_value = mock_icd_path

            results = validate_icd_files()

        assert len(results) == 2
        for r in results:
            assert r["valid"] is True

    def test_missing_icd_dir_returns_empty_list(self):
        """If /etc/OpenCL/vendors doesn't exist, result should be []."""
        with patch("simulation.opencl_detector.Path") as MockPath:
            mock_icd_path = MagicMock()
            mock_icd_path.exists.return_value = False
            MockPath.return_value = mock_icd_path

            results = validate_icd_files()

        assert results == []

    def test_broken_icd_content_reports_invalid(self, tmp_path):
        """ICD file containing garbage (no .so) should report valid=False."""
        icd_dir = tmp_path / "vendors"
        icd_dir.mkdir()
        broken = icd_dir / "nvidia.icd"
        broken.write_text("not-a-library-path\n")  # no .so path, so the file is invalid

        with patch("simulation.opencl_detector.Path") as MockPath:
            mock_icd_path = MagicMock()
            mock_icd_path.exists.return_value = True
            mock_icd_path.glob.return_value = [broken]
            MockPath.return_value = mock_icd_path

            results = validate_icd_files()

        assert len(results) == 1
        assert results[0]["valid"] is False

    def test_valid_result_structure(self, tmp_path):
        """Result dicts should have all required keys."""
        icd_dir = tmp_path / "vendors"
        icd_dir.mkdir()
        nvidia_icd = icd_dir / "nvidia.icd"
        nvidia_icd.write_text("libnvidia-opencl.so.1\n")

        with patch("simulation.opencl_detector.Path") as MockPath:
            mock_icd_path = MagicMock()
            mock_icd_path.exists.return_value = True
            mock_icd_path.glob.return_value = [nvidia_icd]
            MockPath.return_value = mock_icd_path

            results = validate_icd_files()

        assert len(results) == 1
        r = results[0]
        assert "file" in r
        assert "name" in r
        assert "content" in r
        assert "valid" in r
        assert "expected" in r

    def test_icd_with_so_in_content_is_valid(self, tmp_path):
        """.icd content containing .so should be marked valid."""
        icd_dir = tmp_path / "vendors"
        icd_dir.mkdir()
        custom_icd = icd_dir / "custom.icd"
        custom_icd.write_text("libcustom-opencl.so.2\n")

        with patch("simulation.opencl_detector.Path") as MockPath:
            mock_icd_path = MagicMock()
            mock_icd_path.exists.return_value = True
            mock_icd_path.glob.return_value = [custom_icd]
            MockPath.return_value = mock_icd_path

            results = validate_icd_files()

        assert results[0]["valid"] is True

    def test_icd_with_very_long_content_is_invalid(self, tmp_path):
        """ICD content longer than 200 chars should not be considered valid."""
        icd_dir = tmp_path / "vendors"
        icd_dir.mkdir()
        bad_icd = icd_dir / "nvidia.icd"
        # Even if it contains .so, very long content is suspicious
        bad_icd.write_text("libnvidia-opencl.so.1" + "x" * 200 + "\n")

        with patch("simulation.opencl_detector.Path") as MockPath:
            mock_icd_path = MagicMock()
            mock_icd_path.exists.return_value = True
            mock_icd_path.glob.return_value = [bad_icd]
            MockPath.return_value = mock_icd_path

            results = validate_icd_files()

        assert results[0]["valid"] is False


# ---------------------------------------------------------------------------
# TestParseCliinfoEdgeCases: parser robustness
# ---------------------------------------------------------------------------

class TestParseCliinfoEdgeCases:
    """Test parse_clinfo() with unusual or malformed input."""

    def test_device_type_cl_prefix_stored_verbatim(self):
        """Device Type 'CL_DEVICE_TYPE_GPU' is stored verbatim; is_gpu requires exact 'GPU'."""
        raw = """  Platform Name                                   Test Platform

  Device Name                                     Test GPU
  Device Type                                     CL_DEVICE_TYPE_GPU
"""
        _, devices = parse_clinfo(raw)
        # Parser stores the raw type token exactly as found
        assert len(devices) == 1
        # Real clinfo outputs bare "GPU" not the CL_ prefix form, so this is an
        # edge-case input. The stored type should be the captured token.
        assert "GPU" in devices[0].device_type.upper()

    def test_device_type_bare_gpu_is_recognized(self):
        """Device Type 'GPU' (as real clinfo emits) should make is_gpu=True."""
        raw = """  Platform Name                                   Test Platform

  Device Name                                     Test GPU
  Device Type                                     GPU
"""
        _, devices = parse_clinfo(raw)
        assert len(devices) == 1
        assert devices[0].is_gpu is True

    def test_single_device_no_platform_header(self):
        """Device block without a Platform Name header should still be parsed."""
        raw = """  Device Name                                     Orphan GPU
  Device Type                                     GPU
  Max compute units                               8
"""
        _, devices = parse_clinfo(raw)
        assert len(devices) == 1
        assert devices[0].name == "Orphan GPU"

    def test_multiple_platforms_deduplicated(self):
        """Duplicate platform names should only appear once in the list."""
        # The NVIDIA_POCL fixture mentions 'Portable Computing Language' twice
        platforms, _ = parse_clinfo(CLINFO_NVIDIA_POCL)
        assert platforms.count("Portable Computing Language") == 1

    def test_no_crash_on_partial_device_block(self):
        """Incomplete device block (no type) should not crash the parser."""
        raw = """  Platform Name                                   Test Platform

  Device Name                                     IncompleteDevice
"""
        platforms, devices = parse_clinfo(raw)
        assert len(devices) == 1
        assert devices[0].name == "IncompleteDevice"
        assert devices[0].device_type == "UNKNOWN"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ---------------------------------------------------------------------------

class TestBlocksizeThresholdMatchesTheMessage:
    """The printed memory and the branch it chose must agree.

    Reported from the 0.4.5 laptop test: "GPU has 8.0 GB ... using blocksize=8".
    `global_memory_gb` is bytes / 1024**3 and OpenCL reports usable memory, so a
    card whose true value was just under 8 printed "8.0" through `.1f` while
    `>= 8` was False. The message denied the decision it was explaining. (The
    card's exact value was never measured; all that follows from the message is
    7.95 <= v < 8.0.)

    The fix FLOORS to one decimal, so no card changes branch -- see the comment
    at the call site for why rounding would have been a tuning decision.
    """

    def _rec_for(self, memory_bytes: int) -> dict:
        helper = TestRecommendDevice()
        status = helper._make_status_with_gpu(gpu_memory_bytes=memory_bytes)
        return recommend_device(status)

    def test_a_card_just_under_8_says_what_it_did(self):
        """The reported case: prints 8.0, takes the sub-8 branch -- now prints 7.9."""
        rec = self._rec_for(int(7.96 * 1024 ** 3))
        assert rec["blocksize"] == 8          # unchanged by this commit
        text = " ".join(rec["warnings"])
        assert "7.9 GB" in text, text         # and no longer claims 8.0
        assert "8.0 GB" not in text
        assert "blocksize=8" in text

    def test_flooring_never_moves_a_card_into_another_branch(self):
        """floor1(v) >= 8 exactly when v >= 8, so the boundary stays put."""
        assert self._rec_for(int(8.0 * 1024 ** 3))["blocksize"] == 16
        assert self._rec_for(8 * 1024 ** 3 - 1)["blocksize"] == 8
        assert self._rec_for(int(16.0 * 1024 ** 3))["blocksize"] == 32
        assert self._rec_for(16 * 1024 ** 3 - 1)["blocksize"] == 16

    @pytest.mark.parametrize("gib, blocksize", [
        (24.0, 32), (16.0, 32),
        (15.94, 16), (12.0, 16), (8.0, 16),
        (7.96, 8), (7.5, 8), (4.0, 8), (2.0, 8),
    ])
    def test_every_branch_reports_the_number_it_compared(self, gib, blocksize):
        rec = self._rec_for(int(gib * 1024 ** 3))
        assert rec["blocksize"] == blocksize
        if blocksize == 32:
            return   # the big branch says nothing, and has nothing to contradict
        shown = math.floor(gib * 10) / 10
        text = " ".join(rec["warnings"])
        assert f"{shown:.1f} GB" in text, text
        assert f"blocksize={blocksize}" in text, text

    def test_a_big_card_warns_about_nothing(self):
        rec = self._rec_for(24 * 1024 ** 3)
        assert rec["blocksize"] == 32
        assert not [w for w in rec["warnings"] if "blocksize" in w]

    def test_the_prose_is_unchanged_including_the_dash(self):
        """Every other warning in this function uses an em dash; so must these."""
        rec = self._rec_for(int(7.5 * 1024 ** 3))
        assert "GB — using blocksize=8" in " ".join(rec["warnings"])

    # --- the structured half, for the interface ---

    def test_the_blocksize_warning_carries_a_code_and_values(self):
        """Settings -> System Status showed this sentence in English because
        prose was all the backend sent."""
        rec = self._rec_for(int(7.5 * 1024 ** 3))
        items = [i for i in rec["warning_items"] if i["code"] == "gpuBlocksizeMinimal"]
        assert len(items) == 1
        assert items[0]["values"] == {"gb": "7.5", "blocksize": 8}
        assert items[0]["message"] in rec["warnings"]

    def test_the_reduced_warning_too(self):
        rec = self._rec_for(int(12.0 * 1024 ** 3))
        items = [i for i in rec["warning_items"] if i["code"] == "gpuBlocksizeReduced"]
        assert len(items) == 1
        assert items[0]["values"] == {"gb": "12.0", "blocksize": 16}

    def test_a_whole_number_keeps_its_decimal(self):
        """A JSON 8.0 reaches i18next as `8` and would print a different number
        than the English prose one line away."""
        rec = self._rec_for(int(8.0 * 1024 ** 3))
        item = [i for i in rec["warning_items"] if i["code"] == "gpuBlocksizeReduced"][0]
        assert item["values"]["gb"] == "8.0"
        assert "8.0 GB" in item["message"]

    def test_prose_only_warnings_are_untouched(self):
        """Every other producer keeps working exactly as before."""
        helper = TestRecommendDevice()
        rec = recommend_device(helper._make_status_unavailable())
        assert rec["warnings"]
        assert rec["warning_items"] == []
