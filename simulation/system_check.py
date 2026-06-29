"""
System status checker for WSL, EMsoft, and EMSphInx availability.
Runs subprocess commands with short timeouts to verify installation status.
"""

import logging
import subprocess
import json
import sys
from dataclasses import dataclass, field
from path_utils import is_wsl

logger = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """Result of the system environment check."""
    wsl_installed: bool = False
    wsl_distro: str = ""
    emsoft_available: bool = False
    emmc_path: str = ""
    emsht_path: str = ""
    emsphinx_available: bool = False
    emsphinx_path: str = ""
    config_valid: bool = False
    config_data: dict = field(default_factory=dict)
    opencl_available: bool = False
    opencl_devices: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # Enriched OpenCL/EMsoft info from opencl_detector
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    cpu_count: int = 0
    emsoft_binaries: dict = field(default_factory=dict)  # name -> path
    recommended_mode: str = ""  # "gpu", "cpu", or "none"
    recommended_settings: dict = field(default_factory=dict)

    @property
    def overall_status(self):
        """Return 'ok', 'partial', or 'missing'."""
        if self.emsoft_available and self.wsl_installed:
            if self.emsphinx_available and self.opencl_available:
                return "ok"
            return "partial"
        if self.wsl_installed:
            return "partial"
        return "missing"


def _run_wsl_cmd(cmd, timeout=15):
    """Run a WSL/Linux command and return (stdout, success).

    On Windows: runs via 'wsl bash -lc' to reach WSL.
    On Linux/WSL: runs directly via 'bash -lc'.
    Uses login shell (-l) so that ~/.profile PATH additions (set by
    install_emsoft.sh) are available.  Does NOT use -i (interactive)
    because many .bashrc files guard with '[ -z "$PS1" ] && return'.
    Default timeout is 15s to handle WSL cold-start after reboot.
    """
    try:
        if sys.platform == "win32":
            cmd_list = ['wsl', 'bash', '-lc', cmd]
        else:
            cmd_list = ['bash', '-lc', cmd]
        result = subprocess.run(
            cmd_list,
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", False
    except FileNotFoundError:
        return "", False
    except Exception as e:
        logger.debug("Linux command failed: %s", e)
        return "", False


def _run_cmd(cmd_list, timeout=5):
    """Run a Windows command and return (stdout, success)."""
    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return "", False


def check_wsl_installed():
    """Check if WSL is installed and get the distro name.

    On Linux/WSL: We ARE the WSL environment, so return True directly.
    On Windows: Check via 'wsl --list --quiet'.
    """
    if sys.platform != "win32":
        # We're running on Linux — if it's WSL, report the distro
        if is_wsl():
            import platform
            return True, platform.node()
        # Native Linux — WSL concept doesn't apply, but tools work natively
        return True, "native-linux"

    output, ok = _run_cmd(['wsl', '--list', '--quiet'], timeout=5)
    if not ok or not output:
        return False, ""

    # Parse distro names (may have BOM or encoding issues)
    lines = [l.strip().replace('\x00', '') for l in output.splitlines() if l.strip().replace('\x00', '')]
    if lines:
        return True, lines[0]  # First distro is the default
    return False, ""


def check_wsl_responsive():
    """Check if WSL actually responds to commands."""
    if sys.platform != "win32":
        # On Linux/WSL, we can always run bash directly
        return True
    output, ok = _run_wsl_cmd('echo ok', timeout=10)
    return ok and 'ok' in output


def check_executable(name):
    """Check if an executable is available in PATH (WSL or native Linux)."""
    output, ok = _run_wsl_cmd(f'which {name} 2>/dev/null', timeout=5)
    if ok and output and '/' in output:
        return True, output
    return False, ""


def check_executable_at_path(directory, name):
    """Check if a specific executable exists at the given directory path."""
    full_path = f"{directory.rstrip('/')}/{name}"
    output, ok = _run_wsl_cmd(f'test -x "{full_path}" && echo "{full_path}"', timeout=5)
    if ok and output and '/' in output:
        return True, output.strip()
    return False, ""


def check_emsoft_config():
    """Read and validate EMsoft config from WSL."""
    output, ok = _run_wsl_cmd('cat ~/.config/EMsoft/EMsoftConfig.json 2>/dev/null', timeout=5)
    if not ok or not output:
        return False, {}

    try:
        data = json.loads(output)
        # Check required fields
        required = ['EMsoftpathname', 'EMXtalFolderpathname', 'EMdatapathname']
        for field_name in required:
            if field_name not in data:
                return False, data
        return True, data
    except json.JSONDecodeError:
        return False, {}


def check_opencl():
    """Check if OpenCL is available in WSL."""
    output, ok = _run_wsl_cmd('clinfo -l 2>/dev/null', timeout=10)
    if not ok or not output:
        return False, []

    devices = []
    for line in output.splitlines():
        line = line.strip()
        if line and ('Platform' in line or 'Device' in line or 'POCL' in line.upper()):
            devices.append(line)
    return len(devices) > 0, devices


def check_system_status(manual_paths=None):
    """Run all checks and return a SystemStatus object.

    Args:
        manual_paths: Optional dict with keys 'emsoft_bin_dir' and/or 'emsphinx_dir'
                      containing WSL Linux paths to search before falling back to $PATH.
    """
    status = SystemStatus()

    # 1. WSL installed?
    status.wsl_installed, status.wsl_distro = check_wsl_installed()
    if not status.wsl_installed:
        status.errors.append("WSL is not installed. Install with: wsl --install")
        return status

    # 2. WSL responsive?
    if not check_wsl_responsive():
        status.errors.append("WSL is installed but not responding. Try: wsl --shutdown && wsl")
        return status

    emsoft_dir = manual_paths.get('emsoft_bin_dir', '') if manual_paths else ''
    sphinx_dir = manual_paths.get('emsphinx_dir', '') if manual_paths else ''

    # --- Auto-discover paths via candidate directories if not provided ---
    wsl_user = None
    if not emsoft_dir or not sphinx_dir:
        wsl_user_out, wsl_user_ok = _run_wsl_cmd('whoami', timeout=5)
        wsl_user = wsl_user_out.strip() if wsl_user_ok and wsl_user_out.strip() else None

    if not emsoft_dir and wsl_user:
        candidates = [
            f"/home/{wsl_user}/EMsoft_Dev/EMsoftBuild/Release/Bin",
            f"/home/{wsl_user}/emsoft/builds/EMsoft-Release/Bin",
            f"/home/{wsl_user}/EMsoft/builds/EMsoft-Release/Bin",
        ]
        for candidate in candidates:
            found_cand, _ = _run_wsl_cmd(f'test -x "{candidate}/EMMCOpenCL" && echo yes', timeout=5)
            if 'yes' in found_cand:
                emsoft_dir = candidate
                logger.info(f"Auto-discovered EMsoft bin dir: {emsoft_dir}")
                break

    if not sphinx_dir and wsl_user:
        sphinx_candidates = [
            f"/home/{wsl_user}/emsoft/builds/EMSphInx-Release",
            f"/home/{wsl_user}/EMSphInx/build",
        ]
        for candidate in sphinx_candidates:
            found_cand, _ = _run_wsl_cmd(f'test -x "{candidate}/IndexEBSD" && echo yes', timeout=5)
            if 'yes' in found_cand:
                sphinx_dir = candidate
                logger.info(f"Auto-discovered EMSphInx dir: {sphinx_dir}")
                break

    # 3. EMMCOpenCL (try manual/discovered path first, then PATH)
    found, path = False, ""
    if emsoft_dir:
        found, path = check_executable_at_path(emsoft_dir, 'EMMCOpenCL')
    if not found:
        found, path = check_executable('EMMCOpenCL')
    status.emmc_path = path
    if not found:
        status.warnings.append("EMMCOpenCL not found")

    # 4. EMEBSDmasterSHT (try manual/discovered path first, then PATH)
    found_sht, path_sht = False, ""
    if emsoft_dir:
        found_sht, path_sht = check_executable_at_path(emsoft_dir, 'EMEBSDmasterSHT')
    if not found_sht:
        found_sht, path_sht = check_executable('EMEBSDmasterSHT')
    status.emsht_path = path_sht
    if not found_sht:
        status.warnings.append("EMEBSDmasterSHT not found")

    status.emsoft_available = bool(status.emmc_path and status.emsht_path)
    if not status.emsoft_available:
        status.errors.append("EMsoft executables not found. Install EMsoft in WSL.")

    # 5. EMSphInx (try manual/discovered path first, then PATH)
    found_sphinx, path_sphinx = False, ""
    if sphinx_dir:
        found_sphinx, path_sphinx = check_executable_at_path(sphinx_dir, 'IndexEBSD')
    if not found_sphinx:
        found_sphinx, path_sphinx = check_executable('IndexEBSD')
    status.emsphinx_path = path_sphinx
    status.emsphinx_available = found_sphinx
    if not found_sphinx:
        status.warnings.append("EMSphInx not found (optional for Spherical Indexing)")

    # 6. EMsoft config
    status.config_valid, status.config_data = check_emsoft_config()
    if not status.config_valid:
        status.warnings.append("EMsoft config (~/.config/EMsoft/EMsoftConfig.json) missing or invalid")

    # 7. OpenCL — use enriched detector for GPU/CPU info and binary discovery
    try:
        from simulation.opencl_detector import detect_opencl, recommend_device
        from pathlib import Path

        emsoft_bin_path = Path(emsoft_dir) if emsoft_dir else None
        ocl_status = detect_opencl(emsoft_bin_dir=emsoft_bin_path)

        status.opencl_available = ocl_status.available
        status.opencl_devices = [d.summary for d in ocl_status.devices]
        status.has_gpu = ocl_status.has_gpu
        status.cpu_count = ocl_status.cpu_count
        status.emsoft_binaries = ocl_status.binaries

        if ocl_status.gpu_devices:
            gpu = ocl_status.gpu_devices[0]
            status.gpu_name = gpu.name
            status.gpu_memory_gb = gpu.global_memory_gb

        rec = recommend_device(ocl_status)
        status.recommended_mode = rec["mode"]
        status.recommended_settings = rec

        if not status.opencl_available:
            status.warnings.append("OpenCL not available in WSL (needed for EMMCOpenCL GPU acceleration)")
        for w in rec.get("warnings", []):
            status.warnings.append(w)

    except Exception:
        # Fallback to old simple check if detector fails
        status.opencl_available, status.opencl_devices = check_opencl()
        if not status.opencl_available:
            status.warnings.append("OpenCL not available in WSL (needed for EMMCOpenCL GPU acceleration)")

    return status
