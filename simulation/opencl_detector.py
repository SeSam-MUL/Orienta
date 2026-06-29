"""
OpenCL Device Detection and EMsoft Binary Discovery.

Parses `clinfo` output to detect OpenCL platforms/devices, checks which EMsoft
binaries are available, and recommends GPU vs CPU execution paths.

Key concept: EMsoft's CLinit_PDCCQ (CLsupport.f90:557) queries ONLY
CL_DEVICE_TYPE_GPU, so emsoft_devid counts GPU devices separately from
the overall clinfo device index.
"""

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Import path validation to guard against WSL error messages
try:
    from path_utils import _is_valid_unix_path
except ImportError:
    # Standalone usage fallback
    def _is_valid_unix_path(path: str) -> bool:
        return bool(path and path.startswith('/') and re.match(r'^[/\w.\-]+$', path))

logger = logging.getLogger(__name__)

# Default EMsoft binary directory
_DEFAULT_EMSOFT_BIN = Path.home() / "emsoft" / "builds" / "EMsoft-Release" / "Bin"

# All 5 EMsoft programs in the EBSD pipeline
EMSOFT_BINARIES = {
    "EMMCOpenCL": "Monte Carlo (GPU)",
    "EMMC": "Monte Carlo (CPU)",
    "EMEBSDmaster": "Master Pattern (CPU)",
    "EMEBSDmasterOpenCL": "Master Pattern (GPU)",
    "EMEBSDmasterSHT": "SHT (CPU)",
}


@dataclass
class OpenCLDevice:
    """A single OpenCL device."""
    platform_index: int
    device_index: int  # index within the platform
    name: str
    device_type: str  # "GPU" or "CPU"
    vendor: str = ""
    global_memory_bytes: int = 0
    max_work_group_size: int = 0
    max_compute_units: int = 0
    max_clock_mhz: int = 0
    driver_version: str = ""

    @property
    def global_memory_gb(self) -> float:
        return self.global_memory_bytes / (1024 ** 3)

    @property
    def is_gpu(self) -> bool:
        return self.device_type.upper() == "GPU"

    @property
    def summary(self) -> str:
        mem = f"{self.global_memory_gb:.1f} GB" if self.global_memory_bytes else "? GB"
        return f"{self.name} ({mem}, {self.max_compute_units} CU)"


@dataclass
class OpenCLStatus:
    """Overall OpenCL detection result."""
    available: bool = False
    platforms: List[str] = field(default_factory=list)
    devices: List[OpenCLDevice] = field(default_factory=list)
    gpu_devices: List[OpenCLDevice] = field(default_factory=list)
    cpu_devices: List[OpenCLDevice] = field(default_factory=list)
    raw_output: str = ""
    error: str = ""

    # EMsoft binary availability
    binaries: dict = field(default_factory=dict)  # name -> path or ""
    emsoft_bin_dir: str = ""

    # CPU info
    cpu_count: int = 0

    @property
    def has_gpu(self) -> bool:
        return len(self.gpu_devices) > 0

    @property
    def recommended_mode(self) -> str:
        """Return 'gpu', 'cpu', or 'none'."""
        if self.gpu_devices:
            return "gpu"
        if self.cpu_devices:
            return "cpu"
        return "none"


def _run_command(cmd: str, timeout: int = 15) -> Tuple[str, bool]:
    """Run a shell command and return (stdout, success).

    Uses bash -lc (login shell) so that ~/.profile PATH additions
    (set by install_emsoft.sh) are available.  Does NOT use -i
    (interactive) because many .bashrc files guard with
    '[ -z "$PS1" ] && return' which skips PATH in subprocesses.
    """
    try:
        if sys.platform == "win32":
            cmd_list = ['wsl', 'bash', '-lc', cmd]
        else:
            cmd_list = ['bash', '-lc', cmd]
        result = subprocess.run(
            cmd_list, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "", False


def parse_clinfo(raw_output: str) -> Tuple[List[str], List[OpenCLDevice]]:
    """Parse full `clinfo` output into platforms and devices.

    Args:
        raw_output: Full text output from `clinfo` (not `clinfo -l`).

    Returns:
        Tuple of (platform_names, devices).
    """
    platforms: List[str] = []
    devices: List[OpenCLDevice] = []

    if not raw_output.strip():
        return platforms, devices

    # Split into platform sections
    # clinfo outputs "Platform Name" followed by device blocks
    lines = raw_output.splitlines()

    current_platform_idx = -1
    current_device_idx = -1
    current_device: Optional[OpenCLDevice] = None
    in_device_block = False
    device_counter_in_platform = 0

    for line in lines:
        stripped = line.strip()

        # Stop at "NULL platform behavior" section — entries below are
        # context tests, not real device definitions.
        if stripped.startswith("NULL platform behavior"):
            break

        # Detect platform name (2-space indent in main section)
        # "  Platform Name                                   Portable Computing Language"
        match = re.match(r'^  Platform Name\s+(.+)$', line)
        if match:
            pname = match.group(1).strip()
            if pname not in platforms:
                platforms.append(pname)
                current_platform_idx = len(platforms) - 1
                device_counter_in_platform = 0
            continue

        # Detect device name - start of a new device block (2-space indent)
        match = re.match(r'^  Device Name\s+(.+)$', line)
        if match:
            # Save previous device
            if current_device is not None:
                devices.append(current_device)

            current_device = OpenCLDevice(
                platform_index=max(current_platform_idx, 0),
                device_index=device_counter_in_platform,
                name=match.group(1).strip(),
                device_type="UNKNOWN",
            )
            device_counter_in_platform += 1
            in_device_block = True
            continue

        if not in_device_block or current_device is None:
            continue

        # Parse device properties
        match = re.match(r'^\s*Device Type\s+(\S+)', line)
        if match:
            raw = match.group(1).upper()
            # Normalize "CL_DEVICE_TYPE_GPU" → "GPU", "CL_DEVICE_TYPE_CPU" → "CPU"
            if "GPU" in raw:
                current_device.device_type = "GPU"
            elif "CPU" in raw:
                current_device.device_type = "CPU"
            else:
                current_device.device_type = raw
            continue

        match = re.match(r'^\s*Device Vendor\s+(.+)$', line)
        if match and not current_device.vendor:
            current_device.vendor = match.group(1).strip()
            continue

        match = re.match(r'^\s*Global memory size\s+(\d+)', line)
        if match:
            current_device.global_memory_bytes = int(match.group(1))
            continue

        match = re.match(r'^\s*Max work group size\s+(\d+)', line)
        if match:
            current_device.max_work_group_size = int(match.group(1))
            continue

        match = re.match(r'^\s*Max compute units\s+(\d+)', line)
        if match:
            current_device.max_compute_units = int(match.group(1))
            continue

        match = re.match(r'^\s*Max clock frequency\s+(\d+)', line)
        if match:
            current_device.max_clock_mhz = int(match.group(1))
            continue

        match = re.match(r'^\s*Driver Version\s+(.+)$', line)
        if match:
            current_device.driver_version = match.group(1).strip()
            continue

    # Don't forget the last device
    if current_device is not None:
        devices.append(current_device)

    return platforms, devices


def parse_clinfo_list(raw_output: str) -> Tuple[List[str], List[OpenCLDevice]]:
    """Parse `clinfo -l` (short list) output as a fallback.

    Example output:
        Platform #0: Portable Computing Language
         +-- Device #0: cpu-znver3-AMD Ryzen 9 5900X 12-Core Processor
         `-- Device #1: NVIDIA GeForce RTX 4070

    Returns:
        Tuple of (platform_names, devices) with minimal info.
    """
    platforms: List[str] = []
    devices: List[OpenCLDevice] = []

    current_platform_idx = -1
    for line in raw_output.splitlines():
        # Platform line
        match = re.match(r'\s*Platform #(\d+):\s*(.+)', line)
        if match:
            platforms.append(match.group(2).strip())
            current_platform_idx = int(match.group(1))
            continue

        # Device line
        match = re.match(r'\s*[+`]-+\s*Device #(\d+):\s*(.+)', line)
        if match:
            dev_name = match.group(2).strip()
            dev_idx = int(match.group(1))

            # Infer type from name heuristics.
            # IMPORTANT: PoCL names its CPU device with a "cpu-" prefix
            # (e.g. "cpu-znver4-AMD Ryzen 9 8945HS w/ Radeon 780M Graphics").
            # Check this prefix FIRST — it takes priority over keyword matching,
            # otherwise "RADEON" in the CPU device name would cause it to be
            # mis-classified as a GPU, leading to wrong platid/devid for EMsoft.
            name_upper = dev_name.upper()
            if dev_name.lower().startswith('cpu-'):
                dev_type = "CPU"
            elif any(kw in name_upper for kw in ("NVIDIA", "GEFORCE", "RADEON", "RTX", "GTX")):
                dev_type = "GPU"
            elif any(kw in name_upper for kw in ("CPU", "INTEL", "AMD RYZEN", "XEON")):
                dev_type = "CPU"
            else:
                dev_type = "UNKNOWN"

            devices.append(OpenCLDevice(
                platform_index=max(current_platform_idx, 0),
                device_index=dev_idx,
                name=dev_name,
                device_type=dev_type,
            ))

    return platforms, devices


def detect_emsoft_binaries(bin_dir: Optional[Path] = None) -> dict:
    """Check which EMsoft binaries exist via WSL/bash subprocess.

    Uses _run_command() so this works on both Windows (via WSL) and
    native Linux/WSL. Avoids Path.exists() which cannot see the Linux
    filesystem when called from Windows.

    Args:
        bin_dir: Directory to search first (WSL or Linux path).

    Returns:
        Dict mapping binary name to full path (or "" if not found).
    """
    result = {name: "" for name in EMSOFT_BINARIES}

    # Get the actual Linux/WSL home directory via $HOME env var.
    # On Windows, Path.home() gives C:\Users\... which WSL can't use.
    # $HOME inside the WSL shell always gives the correct Linux path.
    home_out, _ = _run_command('echo $HOME', timeout=3)
    wsl_home = home_out.strip() if (home_out and home_out.strip().startswith('/')) else None

    # Build list of directories to probe
    dirs_to_check = []
    if bin_dir:
        dirs_to_check.append(str(bin_dir).replace('\\', '/'))
    if wsl_home:
        dirs_to_check.extend([
            f"{wsl_home}/EMsoft_Dev/EMsoftBuild/Release/Bin",
            f"{wsl_home}/emsoft/builds/EMsoft-Release/Bin",
            f"{wsl_home}/EMsoft/builds/EMsoft-Release/Bin",
        ])

    for name in EMSOFT_BINARIES:
        # Check each candidate directory via WSL/bash
        for d in dirs_to_check:
            full = f"{d}/{name}"
            out, ok = _run_command(f'test -x "{full}" && echo "{full}"', timeout=5)
            if ok and out.strip() and _is_valid_unix_path(out.strip()):
                result[name] = out.strip()
                break
        # Fallback: check PATH via 'which' (.profile sourced via -lc flag)
        if not result[name]:
            out, ok = _run_command(f'which {name} 2>/dev/null', timeout=5)
            if ok and out.strip() and _is_valid_unix_path(out.strip()):
                result[name] = out.strip()

    return result


def compute_emsoft_devid(devices: List[OpenCLDevice], target_gpu: Optional[OpenCLDevice] = None) -> int:
    """Compute the EMsoft-compatible device ID for a specific GPU.

    EMsoft's CLinit_PDCCQ queries ONLY CL_DEVICE_TYPE_GPU on the selected
    platform, numbering them starting at 1. So if a platform has
    [CPU, GPU1, GPU2], EMsoft sees devid=1 → GPU1, devid=2 → GPU2.

    Args:
        devices: All detected OpenCL devices.
        target_gpu: The GPU to compute the EMsoft device ID for.
            If None, returns the ID of the first GPU found.

    Returns:
        1-based GPU device index within the target platform for EMsoft,
        or 0 if no GPU found.
    """
    if target_gpu is None:
        # Legacy behavior: find first GPU across all platforms
        for dev in devices:
            if dev.is_gpu:
                target_gpu = dev
                break
        if target_gpu is None:
            return 0

    gpu_count = 0
    for dev in devices:
        if dev.platform_index == target_gpu.platform_index and dev.is_gpu:
            gpu_count += 1
            if dev.device_index == target_gpu.device_index:
                return gpu_count
    return 0


def recommend_device(status: OpenCLStatus) -> dict:
    """Recommend compute settings based on detected hardware.

    Returns:
        Dict with keys: mode ("gpu"/"cpu"), platid, devid, emsoft_devid,
        globalworkgrpsz, nthreads, mc_program, master_program, sht_program,
        warnings.
    """
    rec = {
        "mode": "cpu",
        "platid": 1,
        "devid": 1,
        "emsoft_devid": 0,
        "globalworkgrpsz": 150,
        "blocksize": 16,
        "nthreads": auto_detect_nthreads(),
        "nthreads_master_opencl": auto_detect_nthreads_4n3(),
        "mc_program": "EMMC",
        "master_program": "EMEBSDmaster",
        "sht_program": "EMEBSDmasterSHT",
        "warnings": [],
    }

    if not status.available:
        rec["warnings"].append("No OpenCL detected — CPU-only mode")
        return rec

    if status.has_gpu:
        gpu = status.gpu_devices[0]
        rec["mode"] = "gpu"
        # platid is 1-based platform index
        rec["platid"] = gpu.platform_index + 1
        # EMsoft's CLinit_PDCCQ counts only GPU devices on the platform,
        # so devid must be the GPU-only index, not the clinfo device index
        emsoft_devid = compute_emsoft_devid(status.devices, gpu)
        rec["emsoft_devid"] = emsoft_devid
        rec["devid"] = emsoft_devid if emsoft_devid > 0 else gpu.device_index + 1
        rec["mc_program"] = "EMMCOpenCL"

        # Validate workgroup size against device max
        if gpu.max_work_group_size > 0:
            rec["globalworkgrpsz"] = min(150, gpu.max_work_group_size)

        # GPU master pattern: blocksize determines VRAM usage
        # blocksize=32 -> ~8-16 GB, blocksize=16 -> ~2-4 GB, blocksize=8 -> ~0.5-1 GB
        rec["master_program"] = "EMEBSDmasterOpenCL"
        if gpu.global_memory_gb >= 16:
            rec["blocksize"] = 32
        elif gpu.global_memory_gb >= 8:
            rec["blocksize"] = 16
            rec["warnings"].append(
                f"GPU has {gpu.global_memory_gb:.1f} GB — "
                f"using blocksize=16 for EMEBSDmasterOpenCL (reduced from 32)."
            )
        else:
            rec["blocksize"] = 8
            rec["warnings"].append(
                f"GPU has {gpu.global_memory_gb:.1f} GB — "
                f"using blocksize=8 for EMEBSDmasterOpenCL. "
                f"Complex crystals (>20 atoms) may still fail."
            )
    else:
        rec["warnings"].append("No GPU found — using CPU-only pipeline")

    # Check binary availability
    for prog_key in ("mc_program", "master_program", "sht_program"):
        prog_name = rec[prog_key]
        if prog_name and status.binaries.get(prog_name) == "":
            rec["warnings"].append(f"{prog_name} not found at {status.emsoft_bin_dir}")
            # Fall back to alternative
            if prog_name == "EMMCOpenCL" and status.binaries.get("EMMC"):
                rec["mc_program"] = "EMMC"
                rec["mode"] = "cpu"
            elif prog_name == "EMEBSDmasterOpenCL" and status.binaries.get("EMEBSDmaster"):
                rec["master_program"] = "EMEBSDmaster"

    return rec


def auto_detect_nthreads() -> int:
    """Auto-detect optimal nthreads for CPU-based EMsoft programs.

    Formula: min(cpu_count - 2, 12) — leaves headroom for OS/GUI and
    caps at 12 (diminishing returns above that for EMsoft OpenMP loops).
    The user has a spinbox in the UI to override.
    """
    count = os.cpu_count() or 4
    return max(1, min(count - 2, 12))


def auto_detect_nthreads_4n3() -> int:
    """Auto-detect nthreads for EMEBSDmasterOpenCL (must be >= 7 and form 4N+3).

    Valid values: 7, 11, 15, 19, 23, ...
    """
    count = os.cpu_count() or 8
    target = max(7, count - 2)
    # Find the nearest valid 4N+3 value <= target
    for n in range(target, 6, -1):
        if (n - 3) % 4 == 0:
            return n
    return 7  # minimum valid value


def validate_globalworkgrpsz(requested: int, device: Optional[OpenCLDevice] = None) -> int:
    """Validate and clamp globalworkgrpsz to device maximum.

    Args:
        requested: Requested work group size.
        device: OpenCL device to validate against (optional).

    Returns:
        Valid work group size.
    """
    if device and device.max_work_group_size > 0:
        return min(max(1, requested), device.max_work_group_size)
    return max(1, requested)


def detect_opencl(emsoft_bin_dir: Optional[Path] = None) -> OpenCLStatus:
    """Main entry point: detect OpenCL devices and EMsoft binaries.

    Args:
        emsoft_bin_dir: Override path to EMsoft Bin directory.

    Returns:
        OpenCLStatus with full detection results.
    """
    status = OpenCLStatus()
    status.cpu_count = os.cpu_count() or 0

    # 1. Try full clinfo
    raw, ok = _run_command("clinfo", timeout=15)
    if ok and raw:
        status.raw_output = raw
        status.platforms, status.devices = parse_clinfo(raw)
    else:
        # Fallback to clinfo -l
        raw_list, ok_list = _run_command("clinfo -l", timeout=10)
        if ok_list and raw_list:
            status.raw_output = raw_list
            status.platforms, status.devices = parse_clinfo_list(raw_list)
        else:
            status.error = "clinfo not available or returned no output"
            status.available = False

    # Classify devices
    for dev in status.devices:
        if dev.is_gpu:
            status.gpu_devices.append(dev)
        elif dev.device_type.upper() == "CPU":
            status.cpu_devices.append(dev)

    status.available = len(status.devices) > 0

    # 2. Detect EMsoft binaries
    status.binaries = detect_emsoft_binaries(emsoft_bin_dir)
    # Record the directory where binaries were found
    for path in status.binaries.values():
        if path:
            status.emsoft_bin_dir = str(Path(path).parent)
            break

    return status


def validate_icd_files() -> List[dict]:
    """Check /etc/OpenCL/vendors/*.icd files for valid library paths.

    Returns:
        List of dicts with keys: file, content, valid, expected.
    """
    results = []
    icd_dir = Path("/etc/OpenCL/vendors")
    if not icd_dir.exists():
        return results

    # Known valid ICD contents — both short names and WSL2 full paths are valid.
    # In WSL2, the NVIDIA library lives at /usr/lib/wsl/lib/ (passed through from
    # the Windows host driver).  The ICD may contain either the bare library name
    # or the absolute path — both are correct for the OpenCL ICD loader.
    known_valid_patterns = {
        "nvidia.icd": ["libnvidia-opencl.so", "/usr/lib/wsl/lib/libnvidia-opencl.so"],
        "pocl.icd": ["libpocl.so"],
    }

    for icd_file in sorted(icd_dir.glob("*.icd")):
        try:
            content = icd_file.read_text().strip()
        except PermissionError:
            content = "(permission denied)"

        patterns = known_valid_patterns.get(icd_file.name, [])
        # Valid if content is reasonably short AND either looks like a library
        # path (.so) or matches a known valid pattern. Over-long content is
        # always rejected — a real ICD file holds just one short library path.
        is_valid = len(content) < 200 and (
            ".so" in content
            or any(pat in content for pat in patterns)
        )

        # Expected value: pick the first pattern as suggestion
        expected = patterns[0] if patterns else ""

        results.append({
            "file": str(icd_file),
            "name": icd_file.name,
            "content": content,
            "valid": is_valid,
            "expected": expected,
        })

    return results


def fix_icd_files() -> Tuple[bool, List[str]]:
    """Attempt to fix broken ICD files using ldconfig -p to find correct library paths.

    Requires sudo. Returns (success, messages).
    """
    messages = []
    icd_results = validate_icd_files()
    broken = [r for r in icd_results if not r["valid"]]

    if not broken:
        messages.append("All ICD files are valid — nothing to fix")
        return True, messages

    # Build library lookup from ldconfig
    lib_lookup = {}
    raw, ok = _run_command("ldconfig -p", timeout=5)
    if ok:
        for line in raw.splitlines():
            # "  libnvidia-opencl.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1"
            match = re.search(r'(lib\S+\.so\S*)\s.*=>\s*(\S+)', line)
            if match:
                lib_lookup[match.group(1)] = match.group(2)

    # Known fixes — try WSL2 full path first for NVIDIA, then bare library name.
    known_fixes_candidates = {
        "nvidia.icd": [
            "/usr/lib/wsl/lib/libnvidia-opencl.so.1",  # WSL2 (Windows driver passthrough)
            "libnvidia-opencl.so.1",                     # Native Linux
        ],
        "pocl.icd": ["libpocl.so"],
    }

    all_fixed = True
    for entry in broken:
        candidates = known_fixes_candidates.get(entry["name"], [])
        fix_content = ""

        # Try each candidate in order — pick the first one that actually exists
        for candidate in candidates:
            if candidate.startswith("/"):
                # Absolute path — check file directly
                if Path(candidate).exists():
                    fix_content = candidate
                    break
            else:
                # Bare library name — check ldconfig or /usr/lib
                if candidate in lib_lookup or Path(f"/usr/lib/{candidate}").exists():
                    fix_content = candidate
                    break

        if not fix_content:
            # Try to guess from the filename via ldconfig
            base = entry["name"].replace(".icd", "")
            for lib_name in lib_lookup:
                if base.lower() in lib_name.lower():
                    fix_content = lib_name
                    break

        if not fix_content:
            messages.append(f"Cannot determine fix for {entry['name']}")
            all_fixed = False
            continue

        try:
            result = subprocess.run(
                ['sudo', 'tee', entry['file']],
                input=fix_content + "\n",
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                messages.append(f"Fixed {entry['name']}: '{entry['content']}' → '{fix_content}'")
            else:
                messages.append(f"Failed to write {entry['name']}: {result.stderr}")
                all_fixed = False
        except Exception as e:
            messages.append(f"Error fixing {entry['name']}: {e}")
            all_fixed = False

    return all_fixed, messages
