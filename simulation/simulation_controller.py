"""
Simulation Controller - Business Logic Layer for EMsoft Simulations

Manages simulation lifecycle, configuration, and interaction with automation script.
Provides clean API for GUI layer.
"""

import logging
import configparser
import os
import re
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List
import queue
import threading
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# Import canonical sanitize function from path_utils (single source of truth)
from path_utils import sanitize_filename, ekev_to_kv_label, _is_valid_unix_path


def sanitize_sim_name(text: str) -> str:
    """Sanitize xtal stem to the ASCII filename used by EMsoft for H5/NML files.

    Delegates to path_utils.sanitize_filename() — the single source of truth.
    Example: "α-(AlMnSi)" → "alpha-_AlMnSi"
    """
    return sanitize_filename(text)


def _has_usable_master(master_files) -> bool:
    """True if at least one given master ``.h5`` is usable (not disc-masked).

    A disc-masked master (the pre-2026-06-22 corner bug — zeroed Lambert-square
    corners) is treated as ABSENT so ``scan_missing_materials`` reports the phase as
    needing regeneration rather than silently keeping a corrupt file. A present file
    we cannot judge (unreadable / not a master) counts as usable, so we never force a
    needless re-simulation on an I/O hiccup.
    """
    try:
        from backend.forward_sim.io.master_validation import h5_master_is_disc_masked
    except Exception:
        # If the detector can't be imported, fall back to "present == usable".
        return bool(list(master_files))
    for mf in master_files:
        try:
            if not h5_master_is_disc_masked(str(mf)):
                return True
        except Exception:
            return True
    return False


class SimulationStatus(Enum):
    """Status states for a simulation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutputType(Enum):
    """Output file types for simulation."""
    SHT_ONLY = "sht_only"           # EMEBSDmasterSHT → .sht (for EMSphinx)
    MASTER_ONLY = "master_only"     # EMEBSDmaster → .h5 master (for kikuchipy DI)
    BOTH = "both"                   # Both SHT and master pattern files


@dataclass
class SimulationJob:
    """Represents a single simulation job."""
    job_id: str
    xtal_path: str
    status: SimulationStatus
    progress_message: str = ""
    progress_pct: float = 0.0
    error_message: str = ""
    log_lines: Optional[List[str]] = None
    worker_thread: Optional[threading.Thread] = None
    result_files: Optional[Dict[str, str]] = None  # {"h5": path, "sht": path, "master": path}


@dataclass
class SimulationParameters:
    """Parameters for EMsoft simulation."""
    # Monte Carlo parameters
    ekev: float = 20.0          # Accelerating voltage (kV)
    sig: float = 70.0           # Sample tilt (degrees)
    omega: float = 0.0          # Sample rotation (degrees)
    numsx: int = 501            # Pattern resolution (pixels, must be odd)

    # Electron parameters
    totnum_el: int = 500000000  # Total electrons (500M default)
    num_el: int = 10            # Electrons per workitem
    multiplier: int = 1         # Depth multiplier

    # SHT parameters
    dmin: float = 0.05          # Minimum d-spacing (nm) - CRITICAL!
    bandwidth: float = 60.0     # Bandwidth (degrees)

    # Master pattern parameters (EMEBSDmaster)
    npx: int = 500              # Number of pixels along x (master pattern half-size)
    nthreads: int = 10          # OpenMP threads for master pattern computation

    # Output type selection
    output_type: str = "sht_only"  # "sht_only", "master_only", "both"

    # System parameters
    platid: int = 1             # OpenCL platform ID
    devid: int = 1              # OpenCL device ID
    globalworkgrpsz: int = 150  # OpenCL work group size
    blocksize: int = 16             # k-vector block size for EMEBSDmasterOpenCL (power of 2)

    # Compute mode: "auto", "gpu", "cpu"
    compute_mode: str = "auto"

    # Program selection (filled by auto_select_programs)
    mc_program: str = ""        # EMMCOpenCL or EMMC
    master_program: str = ""    # EMEBSDmaster, EMEBSDmasterOpenCL
    sht_program: str = ""       # EMEBSDmasterSHT

    # Advanced: EMMCOpenCL parameters
    depthmax: float = 100.0     # Maximum depth for exit depth statistics (nm)
    depthstep: float = 1.0      # Depth step size (nm)

    # Advanced: EMEBSDmaster parameters
    combinesites: bool = False      # Combine all atom sites into one BSE yield
    useEnergyWeighting: bool = False  # Use MC depth histogram to scale intensities
    doLegendre: bool = False        # Use Legendre latitudinal grid
    Esel: int = -1                  # Energy selection (-1 = all energies)
    uniform: bool = False           # Uniform master pattern (all 1.0)

    def validate_and_fix(self) -> List[str]:
        """Validate parameters and auto-fix where possible.

        Returns list of warnings/fixes applied.
        """
        warnings = []

        # numsx must be odd (EMsoft requirement)
        if self.numsx % 2 == 0:
            self.numsx += 1
            warnings.append(f"numsx must be odd — adjusted to {self.numsx}")

        # npx should be positive
        if self.npx < 1:
            self.npx = 500
            warnings.append(f"npx must be positive — reset to {self.npx}")

        # dmin must be positive
        if self.dmin <= 0:
            self.dmin = 0.05
            warnings.append(f"dmin must be > 0 — reset to {self.dmin}")

        # nthreads must be >= 1
        if self.nthreads < 1:
            self.nthreads = 1
            warnings.append("nthreads must be >= 1 — set to 1")

        # For GPU master pattern (EMEBSDmasterOpenCL): nthreads must be 4N+3
        # form (7, 11, 15, 19, 23, ...) and >= 7
        if self.master_program and "OpenCL" in self.master_program:
            if self.nthreads < 7:
                self.nthreads = 7
                warnings.append("GPU master pattern requires nthreads >= 7 — set to 7")
            elif (self.nthreads - 3) % 4 != 0:
                # Round up to next valid 4N+3 value
                n = (self.nthreads - 3 + 3) // 4  # ceiling division
                self.nthreads = 4 * n + 3
                if self.nthreads < 7:
                    self.nthreads = 7
                warnings.append(
                    f"GPU master pattern requires nthreads in 4N+3 form — adjusted to {self.nthreads}"
                )

        # blocksize must be power of 2
        if self.blocksize > 0 and (self.blocksize & (self.blocksize - 1)) != 0:
            # Round up to next power of 2
            self.blocksize = 1 << (self.blocksize - 1).bit_length()
            warnings.append(f"blocksize must be power of 2 — adjusted to {self.blocksize}")

        # totnum_el should be reasonable
        if self.totnum_el < 1000:
            self.totnum_el = 1000000
            warnings.append(f"totnum_el too small — reset to {self.totnum_el}")

        return warnings


class SimulationController:
    """
    Controller for managing EMsoft simulations.

    Responsibilities:
    - Start/stop/monitor simulations
    - Manage worker threads
    - Handle configuration
    - Coordinate with automation script
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize simulation controller.

        Args:
            config_path: Path to emsphinx_config.ini (optional)
        """
        self.jobs: Dict[str, SimulationJob] = {}
        self.upload_queue: queue.Queue = queue.Queue()
        self.job_counter = 0
        self._needs_initial_setup = False

        # Locate config file
        if config_path is None:
            # Try to find it in the crystal-structures folder
            script_dir = Path(__file__).parent.parent
            config_path = script_dir / "crystal-structures-for-ebsd-main" / "_Phyton_Automization" / "windwos_to_WSL" / "emsphinx_config.ini"

        self.config_path = Path(config_path)
        self.config: Optional[configparser.ConfigParser] = None

        if self.config_path.exists():
            self._load_config()
            self._auto_fix_paths()
        else:
            # Auto-copy from template if available
            template_path = self.config_path.with_suffix('.ini.template')
            if template_path.exists():
                try:
                    import shutil
                    shutil.copy2(template_path, self.config_path)
                    logger.info(f"Created config from template: {self.config_path}")
                    self._load_config()
                    self._auto_fix_paths()
                    self._needs_initial_setup = True
                except Exception as e:
                    logger.error(f"Failed to copy config template: {e}")
            else:
                logger.warning(f"Config file not found at {self.config_path}")

    def _load_config(self):
        """Load configuration from emsphinx_config.ini"""
        try:
            self.config = configparser.ConfigParser()
            self.config.read(self.config_path, encoding='utf-8')
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = None

    def validate_config(self) -> Dict[str, str]:
        """Validate simulation configuration and return any issues found.

        Checks for missing or misconfigured sections and paths.

        Returns:
            Dict of {issue_key: description}. Empty dict means config is valid.
        """
        issues: Dict[str, str] = {}

        if self.config is None:
            issues["no_config"] = (
                f"Configuration file not found at {self.config_path}. "
                "Create one from emsphinx_config.ini.template or use Settings."
            )
            return issues

        # Check required sections
        required_sections = ["EMsoftPaths", "WSLEnvironment"]
        for section in required_sections:
            if not self.config.has_section(section):
                issues[f"missing_{section}"] = (
                    f"Section [{section}] missing from config. "
                    "Re-run auto-fix or open Settings to configure."
                )

        # Validate EMsoft binary paths if section exists
        if self.config.has_section("EMsoftPaths"):
            mc_path = self.config.get("EMsoftPaths", "emmcopencl_executable_wsl", fallback="")
            if not mc_path:
                issues["no_emmc_path"] = (
                    "EMsoft Monte Carlo executable path not configured. "
                    "Set emmcopencl_executable_wsl in [EMsoftPaths]."
                )

        # Validate WSL data path
        if self.config.has_section("WSLEnvironment"):
            data_path = self.config.get("WSLEnvironment", "EMsoftData_WSL_Base_Path", fallback="")
            if not data_path:
                issues["no_data_path"] = (
                    "EMsoftData base path not configured. "
                    "Set EMsoftData_WSL_Base_Path in [WSLEnvironment]."
                )

        return issues

    @staticmethod
    def _run_linux_cmd(cmd: str, timeout: int = 15) -> subprocess.CompletedProcess:
        """Platform-aware bash command runner.

        On Windows: routes through WSL ('wsl bash -lc <cmd>').
        On Linux/WSL native: runs directly ('bash -lc <cmd>').

        Uses login shell (-l) so that ~/.profile PATH entries (set by
        install_emsoft.sh) are available. Interactive (-i) is NOT used
        because many .bashrc files guard with '[ -z "$PS1" ] && return'
        which would skip PATH exports in non-interactive subprocesses.

        Default timeout is 15s to handle WSL cold-start after reboot.
        """
        if sys.platform == "win32":
            cmd_list = ['wsl', 'bash', '-lc', cmd]
        else:
            cmd_list = ['bash', '-lc', cmd]
        return subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout)

    def _get_wsl_username(self) -> Optional[str]:
        """Detect the Linux/WSL username.

        On native Linux/WSL: returns the current OS user directly.
        On Windows: runs 'wsl whoami' to find the WSL username.

        Returns:
            Username string (e.g. "wsluser"), or None if unavailable.
        """
        if sys.platform != "win32":
            # Running natively on Linux/WSL — current user IS the Linux user
            username = os.environ.get('USER') or os.environ.get('LOGNAME') or ''
            if username:
                logger.info(f"Detected Linux username: {username}")
                return username

        # Windows: ask WSL
        try:
            result = subprocess.run(
                ['wsl', 'bash', '-lc', 'whoami'],
                capture_output=True, text=True, timeout=5
            )
            username = result.stdout.strip().replace('\x00', '')
            # Validate: username must be a simple alphanumeric string
            if username and re.match(r'^[a-z_][a-z0-9_-]*$', username):
                logger.info(f"Detected WSL username: {username}")
                return username
            elif username:
                logger.warning(f"Ignoring invalid WSL username: {username!r}")
        except Exception as e:
            logger.warning(f"Could not detect WSL username: {e}")
        return None

    def _auto_fix_paths(self):
        """Auto-detect and fix EMsoft/WSL paths for the current system.

        Works on both Windows (via WSL subprocess) and native Linux/WSL.
        Uses the Linux username to replace stale /home/<old_user>/ prefixes,
        discovers the correct EMsoft binary directory, and normalizes any
        backslash-corrupted paths left by Windows pathlib misuse.
        Saves changes back to the config file.
        """
        if self.config is None:
            return

        changed = False

        # --- Step 0: Normalize backslash-corrupted WSL paths (Windows pathlib bug) ---
        # str(Path("/home/wsluser/...").parent) on Windows yields \home\wsluser\...
        for section in ["EMsoftPaths", "WSLEnvironment"]:
            if self.config.has_section(section):
                for key, value in self.config.items(section):
                    if '\\' in value:
                        normalized = value.replace('\\', '/')
                        # Restore leading slash if dropped (e.g. "home/..." → "/home/...")
                        if not normalized.startswith('/') and 'home' in normalized:
                            normalized = '/' + normalized.lstrip('/')
                        if normalized != value:
                            self.config.set(section, key, normalized)
                            changed = True
                            logger.info(f"Normalized backslash path [{section}] {key}: {value!r} -> {normalized!r}")

        # --- Step 1: Get real Linux/WSL username (never hardcoded) ---
        wsl_user = self._get_wsl_username()

        if wsl_user:
            # Replace /home/<any_user>/ with /home/<wsl_user>/ in all WSL path sections
            for section in ["EMsoftPaths", "WSLEnvironment"]:
                if self.config.has_section(section):
                    for key, value in self.config.items(section):
                        new_value = re.sub(r'/home/[^/]+/', f'/home/{wsl_user}/', value)
                        if new_value != value:
                            self.config.set(section, key, new_value)
                            changed = True
                            logger.info(f"Auto-fixed [{section}] {key}: {value!r} -> {new_value!r}")

        # --- Step 2: Discover EMsoft binary directory ---
        if wsl_user and self.config.has_section("EMsoftPaths"):
            bin_dir = None

            # Priority 1: Manual path from user settings JSON
            try:
                settings_file = Path(__file__).resolve().parents[1] / "data" / "user_settings.json"
                if settings_file.exists():
                    import json
                    with open(settings_file, "r", encoding="utf-8") as f:
                        user_settings = json.load(f)
                    manual = (user_settings.get("manual_paths", {}).get("emsoft_bin_dir", "") or "").strip()
                    if manual:
                        r = self._run_linux_cmd(f'test -f "{manual}/EMMCOpenCL" && echo yes')
                        if r.stdout.strip() == 'yes':
                            bin_dir = manual
                            logger.info(f"Using EMsoft bin from user_settings.json: {bin_dir}")
            except Exception as e:
                logger.debug("User settings EMsoft path check failed: %s", e)

            # Priority 2: Verify current config value (may already be correct)
            if bin_dir is None:
                current_mc = self.config.get("EMsoftPaths", "emmcopencl_executable_wsl", fallback="")
                if current_mc:
                    try:
                        r = self._run_linux_cmd(f'test -f "{current_mc}" && echo yes')
                        if r.stdout.strip() == 'yes':
                            bin_dir = current_mc.rsplit('/', 1)[0]
                    except Exception as e:
                        logger.debug("Config EMsoft path verify failed: %s", e)

            # Priority 3: Known candidate paths (most common install locations)
            if bin_dir is None:
                wsl_home_str = f"/home/{wsl_user}"
                candidates = [
                    f"{wsl_home_str}/EMsoft_Dev/EMsoftBuild/Release/Bin",
                    f"{wsl_home_str}/emsoft/builds/EMsoft-Release/Bin",
                    f"{wsl_home_str}/EMsoft/builds/EMsoft-Release/Bin",
                ]
                for candidate in candidates:
                    try:
                        r = self._run_linux_cmd(f'test -f "{candidate}/EMMCOpenCL" && echo yes')
                        if r.stdout.strip() == 'yes':
                            bin_dir = candidate
                            break
                    except Exception as e:
                        logger.debug("Candidate path %s check failed: %s", candidate, e)

            # Priority 4: 'which EMMCOpenCL' — respects ~/.bashrc PATH
            if bin_dir is None:
                try:
                    r = self._run_linux_cmd('which EMMCOpenCL 2>/dev/null', timeout=10)
                    exe = r.stdout.strip()
                    if exe and _is_valid_unix_path(exe):
                        bin_dir = exe.rsplit('/', 1)[0]
                        logger.info(f"Discovered EMsoft bin via 'which': {bin_dir}")
                    elif exe:
                        logger.warning(f"Ignoring invalid 'which' output: {exe!r}")
                except Exception as e:
                    logger.debug("'which EMMCOpenCL' failed: %s", e)

            # Apply discovered bin_dir to all three binary keys
            if bin_dir:
                for key, exe in [
                    ("emmcopencl_executable_wsl", "EMMCOpenCL"),
                    ("emebsdmastersht_executable_wsl", "EMEBSDmasterSHT"),
                    ("emebsdmaster_executable_wsl", "EMEBSDmaster"),
                    ("emebsdmasteropencl_executable_wsl", "EMEBSDmasterOpenCL"),
                ]:
                    new_val = f"{bin_dir}/{exe}"
                    if self.config.get("EMsoftPaths", key, fallback="") != new_val:
                        self.config.set("EMsoftPaths", key, new_val)
                        changed = True
                        logger.info(f"Auto-fixed binary path: {key} = {new_val}")

        # --- Step 3: Fix WSLEnvironment data path ---
        if wsl_user and self.config.has_section("WSLEnvironment"):
            expected_data = f"/home/{wsl_user}/EMsoftData"
            try:
                check = self._run_linux_cmd(f'test -d "{expected_data}" && echo yes')
                if check.stdout.strip() == 'yes':
                    current = self.config.get("WSLEnvironment", "EMsoftData_WSL_Base_Path", fallback="")
                    if current != expected_data:
                        self.config.set("WSLEnvironment", "EMsoftData_WSL_Base_Path", expected_data)
                        changed = True
                        logger.info(f"Auto-fixed EMsoftData path: {expected_data}")
            except Exception as e:
                logger.debug("EMsoftData path auto-fix failed: %s", e)

        # --- Step 4: Fix Windows temp directory (Windows-only) ---
        if sys.platform == "win32" and self.config.has_section("TemporaryDirectories"):
            win_user = os.environ.get("USERNAME") or os.environ.get("USER", "")
            temp_base = self.config.get("TemporaryDirectories", "Temp_Base_Folder_Windows", fallback="")
            if not temp_base.strip() and win_user:
                # Empty from template — set sensible default using project Database/temp_sim
                project_temp = Path(__file__).parent.parent / "Database" / "temp_sim"
                new_temp = str(project_temp).replace("\\", "/")
                self.config.set("TemporaryDirectories", "Temp_Base_Folder_Windows", new_temp)
                changed = True
                logger.info(f"Auto-set temp path (from template): {new_temp}")
            elif win_user and temp_base:
                new_temp = re.sub(
                    r'(C:/Users/)[^/]+(/.+)',
                    rf'\g<1>{win_user}\g<2>',
                    temp_base
                )
                if new_temp != temp_base:
                    self.config.set("TemporaryDirectories", "Temp_Base_Folder_Windows", new_temp)
                    changed = True
                    logger.info(f"Auto-fixed temp path: {new_temp}")

        # Save changes
        if changed:
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    self.config.write(f)
                logger.info(f"Config auto-fixed and saved to {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to save auto-fixed config: {e}")

    def get_default_parameters(self) -> SimulationParameters:
        """
        Get default simulation parameters from config file.

        Returns:
            SimulationParameters with defaults from config or hardcoded defaults.
            nthreads uses auto-detection as fallback (min(cpu_count-2, 12)).
        """
        from simulation.opencl_detector import auto_detect_nthreads

        params = SimulationParameters()
        auto_threads = auto_detect_nthreads()

        if self.config is None:
            logger.warning("No config loaded, using hardcoded defaults")
            params.nthreads = auto_threads
            return params

        try:
            # Load from config if available
            if self.config.has_section("DefaultSimulationParametersEMMCOpenCL"):
                section = "DefaultSimulationParametersEMMCOpenCL"
                params.ekev = self.config.getfloat(section, "EkeV", fallback=20.0)
                params.sig = self.config.getfloat(section, "sig", fallback=70.0)
                params.omega = self.config.getfloat(section, "omega", fallback=0.0)
                params.numsx = self.config.getint(section, "numsx", fallback=501)
                params.totnum_el = self.config.getint(section, "totnum_el", fallback=500000000)
                params.num_el = self.config.getint(section, "num_el", fallback=10)
                params.multiplier = self.config.getint(section, "multiplier", fallback=1)
                params.platid = self.config.getint(section, "platid", fallback=1)
                params.devid = self.config.getint(section, "devid", fallback=1)
                params.globalworkgrpsz = self.config.getint(section, "globalworkgrpsz", fallback=150)

            if self.config.has_section("DefaultSimulationParametersEMEBSDmasterSHT"):
                section = "DefaultSimulationParametersEMEBSDmasterSHT"
                params.dmin = self.config.getfloat(section, "dmin", fallback=0.05)
                params.bandwidth = self.config.getfloat(section, "Butterfly", fallback=60.0)

            if self.config.has_section("DefaultSimulationParametersEMEBSDmaster"):
                section = "DefaultSimulationParametersEMEBSDmaster"
                params.npx = self.config.getint(section, "npx", fallback=500)
                params.nthreads = self.config.getint(section, "nthreads", fallback=auto_threads)

        except Exception as e:
            logger.error(f"Error reading config parameters: {e}")

        return params

    def auto_select_programs(self, params: SimulationParameters) -> SimulationParameters:
        """Auto-select EMsoft programs and OpenCL settings based on detection.

        Modifies params in-place with recommended programs, platid, devid,
        and globalworkgrpsz. Respects params.compute_mode ("auto"/"gpu"/"cpu").

        Returns:
            The modified SimulationParameters.
        """
        try:
            from simulation.opencl_detector import (
                detect_opencl, recommend_device, validate_globalworkgrpsz,
                auto_detect_nthreads, auto_detect_nthreads_4n3,
            )

            ocl_status = detect_opencl()
            rec = recommend_device(ocl_status)

            # Apply compute_mode override
            mode = params.compute_mode
            if mode == "auto":
                mode = rec["mode"]
            elif mode == "gpu" and not ocl_status.has_gpu:
                logger.warning("GPU requested but not available — falling back to CPU")
                mode = "cpu"
            elif mode == "gpu" and rec.get("emsoft_devid", 0) == 0:
                logger.warning(
                    "GPU detected by clinfo but emsoft_devid=0 (no usable GPU "
                    "for EMsoft) — falling back to CPU"
                )
                mode = "cpu"

            if mode == "gpu":
                params.mc_program = rec["mc_program"]
                params.master_program = rec["master_program"]
                params.platid = rec["platid"]
                params.devid = rec["devid"]
                # Validate workgroup size against device max
                gpu_dev = ocl_status.gpu_devices[0] if ocl_status.gpu_devices else None
                params.globalworkgrpsz = validate_globalworkgrpsz(
                    params.globalworkgrpsz, gpu_dev
                )
                # Blocksize for GPU master pattern (auto-scaled by VRAM)
                params.blocksize = rec.get("blocksize", 16)
                # nthreads for GPU master must be 4N+3
                if params.master_program == "EMEBSDmasterOpenCL":
                    params.nthreads = rec.get("nthreads_master_opencl",
                                              auto_detect_nthreads_4n3())
            else:
                params.mc_program = "EMMC"
                params.master_program = "EMEBSDmaster"

            params.sht_program = "EMEBSDmasterSHT"  # always CPU

            # Verify binaries exist
            for attr in ("mc_program", "master_program", "sht_program"):
                prog = getattr(params, attr)
                if prog and ocl_status.binaries.get(prog) == "":
                    logger.warning(f"{prog} binary not found")

            logger.info(
                f"Auto-selected programs: MC={params.mc_program}, "
                f"Master={params.master_program}, SHT={params.sht_program}, "
                f"mode={mode}"
            )

        except Exception as e:
            logger.error(f"Auto-select failed, using CPU defaults: {e}")
            if not params.mc_program:
                params.mc_program = "EMMC"
            if not params.master_program:
                params.master_program = "EMEBSDmaster"
            if not params.sht_program:
                params.sht_program = "EMEBSDmasterSHT"

        return params

    def start_simulation(
        self,
        xtal_path: str,
        params: Optional[SimulationParameters] = None
    ) -> str:
        """
        Start a new simulation.

        Args:
            xtal_path: Path to .xtal file (Windows path)
            params: Simulation parameters (optional, uses defaults if None)

        Returns:
            job_id: Unique identifier for this simulation job

        Raises:
            FileNotFoundError: If .xtal file doesn't exist
            RuntimeError: If simulation cannot be started
        """
        # Validate input
        xtal_file = Path(xtal_path)
        if not xtal_file.exists():
            raise FileNotFoundError(f".xtal file not found: {xtal_path}")

        if self.config is None:
            raise RuntimeError(
                "No configuration loaded. Please check:\n"
                "1. emsphinx_config.ini exists (or .template for auto-copy)\n"
                f"2. Expected path: {self.config_path}\n"
                "3. Open Simulation Settings to configure paths."
            )

        # Use default parameters if none provided
        if params is None:
            params = self.get_default_parameters()

        # Create job
        job_id = f"sim_{self.job_counter:04d}"
        self.job_counter += 1

        job = SimulationJob(
            job_id=job_id,
            xtal_path=str(xtal_file.absolute()),
            status=SimulationStatus.PENDING,
            progress_message="Initializing simulation..."
        )

        self.jobs[job_id] = job

        logger.info(f"Created simulation job {job_id} for {xtal_file.name}")

        # Start worker thread to actually run EMsoft
        worker = threading.Thread(
            target=self._run_simulation_thread,
            args=(job_id, xtal_file, params),
            daemon=True,
            name=f"sim-{job_id}",
        )
        job.worker_thread = worker
        job.status = SimulationStatus.RUNNING
        worker.start()

        return job_id

    # ------------------------------------------------------------------
    # Threaded simulation runner (no PyQt5 dependencies)
    # ------------------------------------------------------------------

    def _run_simulation_thread(
        self,
        job_id: str,
        xtal_file: Path,
        params: SimulationParameters,
    ):
        """Run the EMsoft simulation in a background thread.

        This replicates the logic of simulation_worker.SimulationWorker.run()
        but without PyQt5 signals — updates are written to the SimulationJob
        object which is polled by the FastAPI route.
        """
        job = self.jobs[job_id]
        job.log_lines = []

        # Heartbeat state: the last real log line and when it arrived. The
        # heartbeat thread (started below) reads this to fill silent gaps with
        # a ticking liveness signal instead of letting the status line freeze.
        _hb_state = {
            "last_real_ts": time.monotonic(),
            "last_real_msg": "Initializing simulation...",
        }

        def _progress(msg: str):
            job.progress_message = msg
            job.log_lines.append(msg)
            _hb_state["last_real_ts"] = time.monotonic()
            _hb_state["last_real_msg"] = msg
            # Parse EMsoft output for stage-based progress (never go backwards)
            ml = msg.lower()
            new_pct = job.progress_pct
            if 'emmcopencl' in ml and 'start' in ml:
                new_pct = max(new_pct, 5.0)
            elif 'lokal gefunden' in ml or 'ueberspringe' in ml:
                new_pct = max(new_pct, 28.0)  # MC skipped (local .h5 found)
            elif 'number of bse' in ml and job.progress_pct < 25.0:
                new_pct = max(new_pct, 25.0)  # MC phase reading data
            elif 'emebsdmastersht' in ml and 'start' in ml:
                new_pct = max(new_pct, 35.0)  # SHT phase starting
            elif 'emebsdmaster' in ml and 'start' in ml and 'sht' not in ml:
                new_pct = max(new_pct, 35.0)  # Master pattern starting
            elif 'independent beam direction' in ml:
                new_pct = max(new_pct, 40.0)  # Beam computation starting
            elif 'completed beam direction' in ml:
                m = re.search(r'completed beam direction\s+(\d+)\s+of\s+(\d+)', msg)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    new_pct = max(new_pct, 40.0 + (done / max(total, 1)) * 40.0)  # 40-80%
            elif 'spherical harmonic' in ml or 'writing sht' in ml:
                new_pct = max(new_pct, 82.0)
            elif ('erfolgreich' in ml or 'success' in ml) and job.progress_pct >= 35.0:
                new_pct = max(new_pct, 88.0)
            elif 'upload' in ml or 'stelle' in ml and 'queue' in ml:
                new_pct = max(new_pct, 92.0)
            job.progress_pct = new_pct
            logger.info(f"[{job_id}] {msg}")

        try:
            _progress("Preparing simulation environment...")

            # --- Convert Windows path to WSL path ---
            wsl_xtal_path = self._to_wsl_path(str(xtal_file))
            _progress(f"WSL path: {wsl_xtal_path}")

            # --- Validate and fix parameters ---
            param_warnings = params.validate_and_fix()
            for w in param_warnings:
                _progress(f"Parameter fix: {w}")

            # --- Auto-select GPU/CPU programs based on compute_mode ---
            params = self.auto_select_programs(params)
            _progress(f"Programs: MC={params.mc_program}, Master={params.master_program}, mode={params.compute_mode}")

            # --- Update config with custom parameters ---
            self._apply_params_to_config(params)

            # --- Inject material system ---
            material = self._extract_material(str(xtal_file))
            if not self.config.has_section("MaterialInfo"):
                self.config.add_section("MaterialInfo")
            self.config.set("MaterialInfo", "material_system", material)
            _progress(f"Material system: {material}")

            # --- Set local temp_sim output dir ---
            db_root = Path(__file__).parent.parent / "Database"
            temp_sim_dir = db_root / "temp_sim"
            if not self.config.has_section("TemporaryDirectories"):
                self.config.add_section("TemporaryDirectories")
            self.config.set("TemporaryDirectories", "Temp_Base_Folder_Windows", str(temp_sim_dir))

            # --- Inject server config from data/server_config.json ---
            self._sync_server_config_to_ini()

            # --- Store output type ---
            if not self.config.has_section("OutputOptions"):
                self.config.add_section("OutputOptions")
            self.config.set("OutputOptions", "output_type", params.output_type)

            # --- Import automation module ---
            _progress("Loading automation module...")
            automation_dir = Path(__file__).parent.parent / "crystal-structures-for-ebsd-main" / "_Phyton_Automization" / "windwos_to_WSL"
            if str(automation_dir) not in sys.path:
                sys.path.insert(0, str(automation_dir))

            import importlib
            try:
                import emsphinx_automation
                importlib.reload(emsphinx_automation)
            except ImportError as ie:
                raise RuntimeError(f"Failed to import automation script: {ie}")

            # --- Install log handler to capture progress ---
            class _JobLogHandler(logging.Handler):
                def emit(self_h, record):
                    _progress(record.getMessage())

            log_handler = _JobLogHandler()
            log_handler.setLevel(logging.INFO)
            auto_logger = logging.getLogger("emsphinx_automation")
            auto_logger.addHandler(log_handler)

            _progress(f"Starting EMsoft workflow ({params.output_type})...")

            # Create real upload queue + uploader thread for server mode sync
            sim_upload_queue = queue.Queue()
            uploader = threading.Thread(
                target=emsphinx_automation.uploader_worker,
                args=(sim_upload_queue,),
                daemon=True,
                name=f"uploader-{job_id}",
            )
            uploader.start()

            # Liveness heartbeat — EMEBSDmasterSHT computes the master pattern
            # in-memory on a Legendre grid and emits NO progress output during
            # that phase (often many minutes for complex cells), writing its
            # .sht only at the very end. Without this the status line freezes
            # at the last log line ("Attempting to set number of threads to N")
            # and the run looks dead. The thread refreshes progress_message
            # with elapsed time + live CPU cores whenever real output has gone
            # quiet. It touches ONLY progress_message, never log_lines, so the
            # persistent log stays clean.
            hb_stop = threading.Event()
            hb_start = time.monotonic()

            def _heartbeat():
                while not hb_stop.wait(HEARTBEAT_INTERVAL_SEC):
                    if (time.monotonic() - _hb_state["last_real_ts"]) < HEARTBEAT_INTERVAL_SEC:
                        continue  # real output is flowing — let it show
                    cores = sample_emsoft_cpu_cores()
                    if hb_stop.is_set():
                        break
                    job.progress_message = _format_heartbeat(
                        time.monotonic() - hb_start,
                        _hb_state["last_real_msg"],
                        cores,
                    )

            hb_thread = threading.Thread(
                target=_heartbeat, daemon=True, name=f"hb-{job_id}"
            )
            hb_thread.start()

            try:
                success = emsphinx_automation.process_single_xtal(
                    xtal_wsl=wsl_xtal_path,
                    config=self.config,
                    upload_queue=sim_upload_queue,
                    username=os.environ.get("USERNAME", "user"),
                )
            finally:
                hb_stop.set()
                hb_thread.join(timeout=2)
                auto_logger.removeHandler(log_handler)
                # Signal uploader to stop and wait for pending uploads
                sim_upload_queue.put(None)
                uploader.join(timeout=300)
                if uploader.is_alive():
                    _progress("Warning: upload thread still running after timeout")

            if job.status == SimulationStatus.CANCELLED:
                _progress("Simulation cancelled")
                return

            if success:
                _progress("Simulation completed successfully!")
                job.progress_pct = 95.0
                job.status = SimulationStatus.COMPLETED

                # Locate result files
                job.result_files = self._locate_results(xtal_file, params, db_root)
                if job.result_files:
                    _progress(f"Result files: {', '.join(f'{k}={Path(v).name}' for k, v in job.result_files.items())}")

                # Provenance sidecar for the EMsoft .sht (mirrors the GPU "Ours"
                # path) so BOTH engines are uniformly traceable: source xtal/cif +
                # params + engine. Best-effort — never fail the sim on a sidecar.
                sht_result = job.result_files.get("sht")
                if sht_result:
                    try:
                        from backend.api.services.sht_provenance import write_provenance_sidecar  # noqa: PLC0415
                        write_provenance_sidecar(
                            Path(sht_result),
                            xtal_path=xtal_file,
                            cif_dir=db_root / "CIF_Library",
                            params={"dmin": params.dmin, "npx": params.npx,
                                    "voltage_kV": params.ekev, "sig": params.sig,
                                    "omega": params.omega, "engine": "emsoft"},
                        )
                    except Exception as e:
                        logger.warning("EMsoft provenance sidecar skipped: %s", e)

                # Auto-cleanup: remove large temp files, keep NMLs for logging
                self._cleanup_temp_sim(xtal_file, db_root, _progress)
                job.progress_pct = 100.0
            else:
                _progress("Simulation process returned False")
                job.status = SimulationStatus.FAILED
                job.error_message = "Simulation process returned False"

        except Exception as e:
            logger.error(f"Simulation error for {job_id}: {e}", exc_info=True)
            error_str = str(e).lower()
            job.status = SimulationStatus.FAILED

            # Detect GPU out-of-memory errors and suggest CPU fallback
            oom_patterns = [
                "out of memory", "cl_mem_object_allocation_failure",
                "allocation failed", "memory allocation", "gpu memory",
                "cl_out_of_host_memory", "cl_out_of_resources",
                "insufficient memory", "cuda error",
            ]
            is_oom = any(pat in error_str for pat in oom_patterns)
            # Also check recent log lines for OOM indicators
            if not is_oom and job.log_lines:
                recent = " ".join(job.log_lines[-20:]).lower()
                is_oom = any(pat in recent for pat in oom_patterns)

            if is_oom and params.compute_mode != "cpu":
                job.error_message = (
                    f"GPU out of memory: {e}\n\n"
                    "The GPU ran out of VRAM during simulation. Try:\n"
                    "1. Set Compute Mode to 'CPU' and re-run\n"
                    "2. Reduce blocksize (e.g. 8 instead of 16)\n"
                    "3. Use a simpler crystal structure"
                )
                job.progress_message = "GPU out of memory — switch to CPU mode"
            else:
                job.error_message = str(e)
                job.progress_message = f"Error: {e}"

    def _to_wsl_path(self, path: str) -> str:
        """Convert a Windows path to WSL format."""
        if path.startswith('/'):
            return path
        try:
            automation_dir = Path(__file__).parent.parent / "crystal-structures-for-ebsd-main" / "_Phyton_Automization" / "windwos_to_WSL"
            if str(automation_dir) not in sys.path:
                sys.path.insert(0, str(automation_dir))
            from emsphinx_automation import normalize_to_wsl_path
            return normalize_to_wsl_path(path)
        except Exception:
            result = path.replace("\\", "/")
            result = re.sub(r'^([A-Za-z]):', lambda m: f'/mnt/{m.group(1).lower()}', result)
            return result

    def _extract_material(self, xtal_path: str) -> str:
        """Extract material system from .xtal file."""
        try:
            from simulation.file_sync_manager import FileSyncManager
            sync = FileSyncManager(None)
            return sync.extract_material_from_xtal(xtal_path)
        except Exception:
            return "Default"

    def _apply_params_to_config(self, params: SimulationParameters):
        """Write simulation parameters into the config object."""
        if self.config is None:
            return

        # Monte Carlo
        sec = "DefaultSimulationParametersEMMCOpenCL"
        if not self.config.has_section(sec):
            self.config.add_section(sec)
        self.config.set(sec, "EkeV", str(params.ekev))
        self.config.set(sec, "sig", str(params.sig))
        self.config.set(sec, "omega", str(params.omega))
        self.config.set(sec, "numsx", str(params.numsx))
        self.config.set(sec, "totnum_el", str(params.totnum_el))
        self.config.set(sec, "depthmax", str(params.depthmax))
        self.config.set(sec, "depthstep", str(params.depthstep))
        is_gpu = params.mc_program == "EMMCOpenCL" or params.compute_mode in ("auto", "gpu")
        if is_gpu:
            self.config.set(sec, "platid", str(params.platid))
            self.config.set(sec, "devid", str(params.devid))
            self.config.set(sec, "globalworkgrpsz", str(params.globalworkgrpsz))

        # SHT
        sec = "DefaultSimulationParametersEMEBSDmasterSHT"
        if not self.config.has_section(sec):
            self.config.add_section(sec)
        self.config.set(sec, "dmin", str(params.dmin))
        self.config.set(sec, "bandwidth", str(params.bandwidth))
        self.config.set(sec, "nthreads", str(params.nthreads))

        # Master (CPU)
        sec = "DefaultSimulationParametersEMEBSDmaster"
        if not self.config.has_section(sec):
            self.config.add_section(sec)
        self.config.set(sec, "dmin", str(params.dmin))
        self.config.set(sec, "npx", str(params.npx))
        self.config.set(sec, "nthreads", str(params.nthreads))
        self.config.set(sec, "combinesites", ".TRUE." if params.combinesites else ".FALSE.")
        self.config.set(sec, "useEnergyWeighting", ".TRUE." if params.useEnergyWeighting else ".FALSE.")
        self.config.set(sec, "doLegendre", ".TRUE." if params.doLegendre else ".FALSE.")
        self.config.set(sec, "Esel", str(params.Esel))
        self.config.set(sec, "uniform", ".TRUE." if params.uniform else ".FALSE.")

        # Master (GPU) — only written when GPU master is selected
        if params.master_program == "EMEBSDmasterOpenCL":
            sec = "DefaultSimulationParametersEMEBSDmasterOpenCL"
            if not self.config.has_section(sec):
                self.config.add_section(sec)
            self.config.set(sec, "dmin", str(params.dmin))
            self.config.set(sec, "npx", str(params.npx))
            self.config.set(sec, "nthreads", str(params.nthreads))
            self.config.set(sec, "platid", str(params.platid))
            self.config.set(sec, "devid", str(params.devid))
            self.config.set(sec, "globalworkgrpsz", str(params.globalworkgrpsz))
            self.config.set(sec, "blocksize", str(params.blocksize))
            self.config.set(sec, "Notify", "off")

        # Output options (needed by automation to pick GPU vs CPU master)
        if not self.config.has_section("OutputOptions"):
            self.config.add_section("OutputOptions")
        self.config.set("OutputOptions", "output_type", params.output_type)
        self.config.set("OutputOptions", "master_program",
                        params.master_program or "EMEBSDmaster")

    def _sync_server_config_to_ini(self):
        """Read data/server_config.json and inject paths into config [CentralDatabase]."""
        import json
        config_file = Path(__file__).parent.parent / "data" / "server_config.json"
        if not config_file.exists():
            return
        try:
            with open(config_file) as f:
                srv = json.load(f)
            if not srv.get("enabled"):
                return
            db_root = srv.get("database_root", "")
            if not db_root:
                return
            db_root = Path(db_root)
            if not self.config.has_section("CentralDatabase"):
                self.config.add_section("CentralDatabase")
            # Map discovered subfolders to config keys
            for subdir, key in [
                ("EBSD_SHT_Database", "sht_database_path_unc"),
                ("EBSD_H5_Cache", "h5_cache_path_unc"),
                ("EBSD_CIF_Library", "cif_library_path_unc"),
                ("EBSD_XTAL_Library", "xtal_library_path_unc"),
            ]:
                p = db_root / subdir
                if p.is_dir():
                    self.config.set("CentralDatabase", key, str(p))
        except Exception as e:
            logger.warning(f"Failed to sync server config: {e}")

    def _cleanup_temp_sim(self, xtal_file: Path, db_root: Path, progress_callback=None):
        """Remove large temp files after successful simulation, keep NMLs for logs."""
        sim_name = sanitize_sim_name(xtal_file.stem)
        temp_dir = db_root / "temp_sim" / sim_name
        if not temp_dir.is_dir():
            return

        removed_mb = 0.0
        keep_extensions = {".nml"}
        for f in list(temp_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() not in keep_extensions:
                try:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    f.unlink()
                    removed_mb += size_mb
                except Exception as e:
                    logger.warning("Cleanup failed for %s: %s", f, e)

        # Remove empty subdirectories
        for d in sorted(temp_dir.rglob("*"), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()  # Only removes if empty
                except OSError:
                    pass

        if removed_mb > 0 and progress_callback:
            progress_callback(f"Cleaned up {removed_mb:.0f} MB from temp_sim/{sim_name}")
        logger.info("Cleanup: removed %.0f MB from %s (NMLs kept)", removed_mb, temp_dir)

    def _locate_results(self, xtal_file: Path, params: SimulationParameters, db_root: Path) -> Dict[str, str]:
        """Try to locate result files after simulation."""
        results = {}
        sim_name = sanitize_sim_name(xtal_file.stem)

        # Primary: temp_sim/<sim_name>/ — where automation script copies results
        temp_workdir = db_root / "temp_sim" / sim_name
        search_dirs = [temp_workdir]
        # Also check SHT/H5 database dirs and WSL-accessible paths
        search_dirs.extend([
            db_root / "EBSD_SHT_Database",
            db_root / "EBSD_H5_Cache",
        ])

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for f in search_dir.rglob(f"*{sim_name}*"):
                if f.suffix == ".sht" and "sht" not in results:
                    results["sht"] = str(f)
                elif f.suffix == ".h5" and "master" in f.stem.lower() and "master" not in results:
                    results["master"] = str(f)
                elif f.suffix == ".h5" and "h5" not in results:
                    results["h5"] = str(f)
        return results

    def get_job_status(self, job_id: str) -> Optional[SimulationJob]:
        """
        Get status of a simulation job.

        Args:
            job_id: Job identifier

        Returns:
            SimulationJob if found, None otherwise
        """
        return self.jobs.get(job_id)

    def cancel_simulation(self, job_id: str) -> bool:
        """
        Cancel a running simulation.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False if job not found or not running
        """
        job = self.jobs.get(job_id)
        if job is None:
            logger.warning(f"Cannot cancel: job {job_id} not found")
            return False

        if job.status != SimulationStatus.RUNNING:
            logger.warning(f"Cannot cancel: job {job_id} is {job.status.value}")
            return False

        # Signal worker thread to stop
        if (hasattr(self, '_current_worker') and self._current_worker is not None
                and job.worker_thread is self._current_worker):
            self._current_worker.cancel()
            logger.info(f"Sent cancel signal to worker for job {job_id}")

        # Kill any WSL subprocess associated with this simulation
        self._kill_wsl_simulation_processes()

        job.status = SimulationStatus.CANCELLED
        job.progress_message = "Cancelled by user"

        logger.info(f"Cancelled job {job_id}")
        return True

    def _kill_wsl_simulation_processes(self) -> None:
        """Kill EMsoft simulation processes running in WSL.

        Sends SIGTERM to known EMsoft binaries (EMMCOpenCL, EMMC,
        EMEBSDmaster, EMEBSDmasterSHT, EMEBSDmasterOpenCL) via
        WSL's kill command.
        """
        emsoft_binaries = [
            "EMMCOpenCL", "EMMC", "EMEBSDmaster",
            "EMEBSDmasterSHT", "EMEBSDmasterOpenCL",
        ]
        for binary in emsoft_binaries:
            try:
                subprocess.run(
                    ["wsl", "pkill", "-f", binary],
                    capture_output=True, timeout=5,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass  # WSL not available or process not found — that's OK

    def get_all_jobs(self) -> List[SimulationJob]:
        """
        Get list of all jobs.

        Returns:
            List of SimulationJob objects
        """
        return list(self.jobs.values())

    def get_running_jobs(self) -> List[SimulationJob]:
        """
        Get list of currently running jobs.

        Returns:
            List of SimulationJob objects with status RUNNING
        """
        return [job for job in self.jobs.values() if job.status == SimulationStatus.RUNNING]

    def get_completed_jobs(self) -> List[SimulationJob]:
        """
        Get list of completed jobs.

        Returns:
            List of SimulationJob objects with status COMPLETED
        """
        return [job for job in self.jobs.values() if job.status == SimulationStatus.COMPLETED]

    def scan_missing_materials(
        self, output_type: str, params: SimulationParameters
    ) -> List[Dict[str, object]]:
        """Scan XTAL Library and find materials that still need simulation.

        Searches ALL material subfolders in Database/EBSD_H5_Cache/*/
        and Database/EBSD_SHT_Database/*/ so files in Default/, Fe/, Al/
        etc. are all found regardless of folder naming.

        Returns:
            List of dicts: [{"xtal_path": str, "stem": str, "missing": ["h5","sht","master"]}]
        """
        project_root = Path(__file__).parent.parent
        db_root = project_root / "Database"
        xtal_dir = db_root / "XTAL_Library"

        # Fallback to crystal-structures folder if XTAL_Library empty
        if not xtal_dir.exists() or not list(xtal_dir.glob("*.xtal")):
            xtal_dir = (project_root / "crystal-structures-for-ebsd-main"
                        / "_xtal_files")

        if not xtal_dir.exists():
            logger.warning("No XTAL directory found for batch scan")
            return []

        h5_cache = db_root / "EBSD_H5_Cache"
        sht_cache = db_root / "EBSD_SHT_Database"

        needs_sht = output_type in ("sht_only", "both")
        needs_master = output_type in ("master_only", "both")

        results = []
        for xtal_path in sorted(xtal_dir.glob("*.xtal")):
            stem = xtal_path.stem
            # EMsoft sanitizes filenames: α→alpha, spaces→_, etc.
            safe_stem = sanitize_sim_name(stem)
            missing = []

            # Check MC H5 — search ALL subfolders with sanitized name.
            # ekev_to_kv_label (round) MUST match both engines' on-disk naming
            # (GPU runner + EMsoft automation) — see path_utils.ekev_to_kv_label.
            h5_pattern = f"*/{safe_stem}_E{ekev_to_kv_label(params.ekev)}kV*.h5"
            h5_files = list(h5_cache.glob(h5_pattern)) if h5_cache.exists() else []
            h5_files = [f for f in h5_files if "master" not in f.name]
            if not h5_files:
                missing.append("h5")

            # Check SHT — search ALL subfolders for files containing stem in parentheses
            # SHT filenames: "Al15Fe5 (Al13Fe4) [oC16] {20kV}.sht"
            # Must match "(stem)" to avoid false positives (e.g. "Al" matching "Al13Fe4")
            if needs_sht:
                sht_files = list(sht_cache.glob("*/*.sht")) if sht_cache.exists() else []
                sht_files = [f for f in sht_files
                             if f"({stem})" in f.name or f.name == f"{stem}.sht"]
                if not sht_files:
                    missing.append("sht")

            # Check Master — search ALL subfolders with sanitized name
            if needs_master:
                master_pattern = f"*/{safe_stem}_master*.h5"
                master_files = list(h5_cache.glob(master_pattern)) if h5_cache.exists() else []
                if not master_files:
                    master_files = [f for f in h5_cache.glob("*/*master*.h5")
                                    if safe_stem in f.name] if h5_cache.exists() else []
                # A disc-masked (corrupt) master counts as missing so it is
                # regenerated rather than silently kept (pre-2026-06-22 corner bug).
                if not _has_usable_master(master_files):
                    missing.append("master")

            if missing:
                results.append({
                    "xtal_path": str(xtal_path),
                    "stem": stem,
                    "missing": missing,
                })

        return results

    def check_network_availability(self) -> bool:
        """
        Check if network drive is available.

        Returns:
            True if network drive is accessible, False otherwise
        """
        if self.config is None:
            return False

        try:
            # Import the automation script's network check function
            import sys
            automation_dir = self.config_path.parent
            if str(automation_dir) not in sys.path:
                sys.path.insert(0, str(automation_dir))

            from emsphinx_automation import check_network_drive
            return check_network_drive(self.config)

        except Exception as e:
            logger.error(f"Network check failed: {e}")
            return False

    def estimate_computation_time(self, params: SimulationParameters) -> float:
        """
        Estimate computation time in hours.

        Args:
            params: Simulation parameters

        Returns:
            Estimated time in hours (rough estimate)
        """
        # Very rough estimation based on:
        # - Resolution (numsx)
        # - Total electrons
        # Assume: 500M electrons @ 501px takes ~30 minutes on average GPU

        base_time_hours = 0.5  # 30 minutes baseline

        # Scale by resolution (quadratic)
        resolution_factor = (params.numsx / 501.0) ** 2

        # Scale by electron count (linear)
        electron_factor = params.totnum_el / 500000000.0

        estimated_hours = base_time_hours * resolution_factor * electron_factor

        return estimated_hours


class BatchQueueManager:
    """Manages sequential execution of multiple simulation jobs.

    Processes one SimulationWorker at a time. When a worker finishes
    (success or failure), starts the next one. Failed jobs do not stop
    the queue.

    Usage:
        manager = BatchQueueManager(controller, server_manager)
        manager.start_batch(xtal_paths, params)
    """

    def __init__(
        self,
        controller: SimulationController,
        server_manager=None,
    ):
        self._controller = controller
        self._server_manager = server_manager
        self._pending: List[str] = []
        self._params: Optional[SimulationParameters] = None
        self._current_worker = None
        self._current_job_id: Optional[str] = None
        self._is_running = False
        self._cancel_requested = False
        self._success_count = 0
        self._fail_count = 0

        # Callbacks (set by GUI)
        self.on_job_started = None    # (job_id, xtal_name) -> None
        self.on_job_finished = None   # (job_id, success, message) -> None
        self.on_batch_finished = None  # (success_count, fail_count) -> None
        self.on_progress = None       # (message) -> None
        self.on_stage = None          # (stage_name, percentage) -> None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def start_batch(self, xtal_paths: List[str], params: SimulationParameters):
        """Queue multiple xtal files for sequential processing."""
        if self._is_running:
            raise RuntimeError("Batch already running")

        self._pending = list(xtal_paths)
        self._params = params
        self._is_running = True
        self._cancel_requested = False
        self._success_count = 0
        self._fail_count = 0

        self._start_next()

    def cancel_remaining(self):
        """Cancel all pending jobs. Current job finishes normally."""
        self._cancel_requested = True
        cancelled = len(self._pending)
        self._pending.clear()
        logger.info(f"Batch cancelled: {cancelled} pending jobs removed")

    def _start_next(self):
        """Pop next xtal from queue and start a SimulationWorker."""
        if not self._pending or self._cancel_requested:
            self._is_running = False
            if self.on_batch_finished:
                self.on_batch_finished(self._success_count, self._fail_count)
            return

        xtal_path = self._pending.pop(0)
        xtal_name = Path(xtal_path).name

        try:
            params = self._controller.auto_select_programs(
                SimulationParameters(**vars(self._params))
            )
            job_id = self._controller.start_simulation(xtal_path, params)
            self._current_job_id = job_id

            from simulation.simulation_worker import SimulationWorker

            self._current_worker = SimulationWorker(
                job_id=job_id,
                xtal_path=xtal_path,
                config=self._controller.config,
                params=params,
                server_manager=self._server_manager,
            )

            # Wire signals
            if self.on_progress:
                self._current_worker.progress_update.connect(self.on_progress)
            if self.on_stage:
                self._current_worker.stage_changed.connect(self.on_stage)
            self._current_worker.simulation_finished.connect(
                self._on_worker_finished
            )

            # Update job status
            job = self._controller.get_job_status(job_id)
            if job:
                job.worker_thread = self._current_worker
                job.status = SimulationStatus.RUNNING

            if self.on_job_started:
                self.on_job_started(job_id, xtal_name)

            self._current_worker.start()

        except Exception as e:
            logger.error(f"Failed to start batch job for {xtal_name}: {e}")
            self._fail_count += 1
            if self.on_job_finished:
                self.on_job_finished("", False, f"{xtal_name}: {e}")
            # Continue with next
            self._start_next()

    def _on_worker_finished(self, success: bool, message: str, result_files: dict):
        """Handle current worker completion, start next job."""
        job_id = self._current_job_id or ""

        # Update job status
        job = self._controller.get_job_status(job_id)
        if job:
            job.status = (SimulationStatus.COMPLETED if success
                          else SimulationStatus.FAILED)
            job.progress_message = "Completed" if success else message
            job.result_files = result_files if success else None

        if success:
            self._success_count += 1
        else:
            self._fail_count += 1

        if self.on_job_finished:
            self.on_job_finished(job_id, success, message)

        self._current_worker = None
        self._current_job_id = None

        # Start next job
        self._start_next()


# ---------------------------------------------------------------------------
# Startup reaper — kill EMsoft processes orphaned by a previous backend session
# ---------------------------------------------------------------------------

# Longest / most-specific names first so the label picks the precise binary
# (e.g. "EMEBSDmasterSHT" rather than its substring "EMEBSDmaster").
ORPHAN_EMSOFT_BINARIES = (
    "EMEBSDmasterSHT",
    "EMEBSDmasterOpenCL",
    "EMEBSDmaster",
    "EMMCOpenCL",
    "EMMC",
)


def _default_reaper_runner(cmd: str, timeout: int = 15):
    """Platform-aware bash runner for the reaper; returns None on failure.

    Mirrors SimulationController._run_linux_cmd (wsl on Windows, native bash
    on Linux) but never raises — the reaper must not abort backend startup.
    """
    try:
        return SimulationController._run_linux_cmd(cmd, timeout=timeout)
    except Exception:
        return None


def reap_orphaned_emsoft_processes(runner=None, logger_=None) -> List[dict]:
    """Kill EMsoft processes orphaned by a previous backend session.

    Run ONCE at backend startup. EMsoft binaries execute inside the WSL2 VM
    via ``wsl bash -lc "... EMEBSDmaster... "``. When the backend (uvicorn)
    is restarted or killed, only the Windows-side ``wsl.exe`` relay dies —
    the compute process keeps running inside the WSL VM indefinitely, burning
    CPU (observed: a 10-hour orphan after a backend restart that pinned ~8
    cores). The only pre-existing cleanup, ``_kill_wsl_simulation_processes``,
    fires solely on an explicit user Cancel — never on restart — so orphans
    accumulate across restarts.

    At startup the backend has launched nothing yet, so any EMsoft process
    found is by definition an orphan and safe to kill. The function
    enumerates first so it can log exactly what it reaps (no silent kills),
    then sends SIGKILL.

    Assumes a single backend instance (this project's deployment model);
    a second concurrent backend would reap the first's running jobs. Never
    raises — startup must not be blocked or aborted by cleanup.

    Args:
        runner: callable(cmd: str, timeout: int) -> CompletedProcess|None,
            injectable for testing. Defaults to the platform-aware runner.
        logger_: logger to use (defaults to module logger).

    Returns:
        List of reaped descriptors: [{"pid", "name", "elapsed_sec"}].
    """
    runner = runner or _default_reaper_runner
    log = logger_ or logger
    reaped: List[dict] = []
    try:
        proc = runner("ps -eo pid=,etimes=,args=")
        if proc is None or getattr(proc, "returncode", 1) != 0 or not getattr(proc, "stdout", ""):
            return reaped

        for line in proc.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, etimes, args = parts
            if not pid.isdigit():
                continue
            matched = [b for b in ORPHAN_EMSOFT_BINARIES if b in args]
            if not matched:
                continue
            try:
                secs = int(etimes)
            except ValueError:
                secs = -1
            reaped.append({
                "pid": pid,
                "name": max(matched, key=len),
                "elapsed_sec": secs,
            })

        if not reaped:
            log.info("Startup reaper: no orphaned EMsoft processes found.")
            return reaped

        log.warning(
            "Startup reaper: killing %d orphaned EMsoft process(es) from a "
            "previous backend session: %s",
            len(reaped),
            ", ".join(
                f"{r['name']}(pid={r['pid']}, age={r['elapsed_sec']}s)"
                for r in reaped
            ),
        )
        pids = " ".join(r["pid"] for r in reaped)
        runner(f"kill -9 {pids}")
    except Exception:
        log.exception("Startup reaper failed (non-fatal)")
    return reaped


# ---------------------------------------------------------------------------
# Simulation heartbeat — liveness signal during silent EMsoft compute phases
# ---------------------------------------------------------------------------

# How often the heartbeat refreshes the status line (seconds).
HEARTBEAT_INTERVAL_SEC = 8.0


def _parse_proc_stat_ticks(stat_line: str):
    """utime+stime (in clock ticks) from a /proc/<pid>/stat line, or None.

    Robust against comm fields that contain spaces or parens: everything up
    to and including the LAST ')' is the (pid, comm) prefix; the remaining
    whitespace-split fields are state, ppid, ... with utime at offset 11 and
    stime at offset 12 (0-based).
    """
    rp = stat_line.rfind(")")
    if rp == -1:
        return None
    post = stat_line[rp + 1:].split()
    if len(post) < 13:
        return None
    try:
        return int(post[11]) + int(post[12])
    except ValueError:
        return None


def sample_emsoft_cpu_cores(runner=None, sleep_fn=None, monotonic_fn=None):
    """Return the number of CPU cores currently busy across EMsoft processes
    in WSL, or None if it can't be determined / no EMsoft process is running.

    Samples cumulative CPU ticks (utime+stime from /proc/<pid>/stat) ~1 s
    apart and converts the delta to busy cores via CLK_TCK. This is the
    heartbeat's true liveness signal: a value near 0 while a job is "running"
    means the compute has actually stalled, not merely gone quiet.

    Process selection matches comm starting with ``EMEBSD``/``EMMC`` — a
    prefix that survives ps's 15-char comm truncation (so EMEBSDmasterOpenCL
    is caught) and never matches the ps/cat/bash helpers themselves.

    IMPORTANT: all WSL commands here are trivial (``ps -eo pid=,comm=``,
    ``cat /proc/<pid>/stat``, ``getconf CLK_TCK``) with NO awk/quotes/``$``.
    An earlier awk-based one-liner failed silently because single quotes are
    stripped when the command is passed through
    subprocess(['wsl','bash','-lc', cmd]) on Windows, leaving bash to expand
    the awk ``$1``/``$2`` fields to empty and corrupting the script. Parsing
    is done in Python instead — the same robust style the startup reaper uses.

    ``runner``/``sleep_fn``/``monotonic_fn`` are injectable for testing;
    ``runner`` defaults to the platform-aware runner (wsl on Windows, native
    bash on Linux).
    """
    runner = runner or _default_reaper_runner
    sleep_fn = sleep_fn or time.sleep
    monotonic_fn = monotonic_fn or time.monotonic
    try:
        proc = runner("ps -eo pid=,comm=")
        if proc is None or getattr(proc, "returncode", 1) != 0 or not getattr(proc, "stdout", ""):
            return None
        pids = []
        for line in proc.stdout.splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                comm = parts[1].strip()
                if comm.startswith("EMEBSD") or comm.startswith("EMMC"):
                    pids.append(parts[0])
        if not pids:
            return None

        paths = " ".join(f"/proc/{p}/stat" for p in pids)

        def _sum_ticks():
            pr = runner(f"cat {paths}")
            if pr is None or not getattr(pr, "stdout", ""):
                return None
            total = 0
            got = False
            for line in pr.stdout.splitlines():
                t = _parse_proc_stat_ticks(line)
                if t is not None:
                    total += t
                    got = True
            return total if got else None

        t1 = _sum_ticks()
        ts1 = monotonic_fn()
        if t1 is None:
            return None
        sleep_fn(1.0)
        t2 = _sum_ticks()
        ts2 = monotonic_fn()
        if t2 is None:
            return None

        # Divide by the ACTUAL elapsed wall time, not the requested 1.0 s:
        # each _sum_ticks() spawns a WSL subprocess (~0.2-0.5 s), so the real
        # window is well over 1 s. Assuming 1 s overestimates busy cores
        # (e.g. reports 34 on a 24-thread box). /proc utime+stime already
        # aggregates all threads of the process.
        elapsed = ts2 - ts1
        if elapsed <= 0:
            return None

        hz = 100.0
        hp = runner("getconf CLK_TCK")
        if hp is not None and getattr(hp, "stdout", "").strip().isdigit():
            hz = float(hp.stdout.strip()) or 100.0

        return max(0.0, (t2 - t1) / hz / elapsed)
    except Exception:
        return None


def _format_heartbeat(elapsed_sec, last_real: str, cores) -> str:
    """Build the heartbeat status line: elapsed time + last real log + CPU.

    cores: float busy-cores, or None if unknown. < 0.5 cores while a job is
    supposed to be running is surfaced as a soft stall warning.
    """
    elapsed_sec = int(max(0, elapsed_sec))
    mm, ss = divmod(elapsed_sec, 60)
    hh, mm = divmod(mm, 60)
    if hh:
        et = f"{hh}h {mm:02d}m {ss:02d}s"
    elif mm:
        et = f"{mm}m {ss:02d}s"
    else:
        et = f"{ss}s"
    base = (last_real or "").strip().lstrip("| ").strip() or "computing"
    if cores is None:
        cpu = ""
    elif cores < 0.5:
        cpu = " · ⚠ CPU idle (possible stall)"
    else:
        cpu = f" · {cores:.0f} cores active"
    return f"⏳ {base} — {et} elapsed{cpu}"


# ---------------------------------------------------------------------------
# Adaptive dmin recommendation — keep master-pattern sims tractable per phase
# ---------------------------------------------------------------------------

# The EMsoft dynamical computation cost scales steeply with the number of
# reflections within dmin, which scales as (a/dmin)^3. Empirically the
# "reflection range" EMsoft prints is ~ a_max/dmin. Phases that already
# index well in this project sit at range ~7-15 (Al 8.1, Ni 7.0, Al6Fe
# 14.9, all at dmin=0.05). A 12.3 Å quasicrystal approximant at dmin=0.05
# lands at range ~25 -> the master pattern takes DAYS. Raising dmin so the
# range targets ~15 (the proven-good Al6Fe level) makes a giant cell as
# cheap as an already-working phase, while small cells keep the fine
# default (we never lower dmin below the floor).
DMIN_FLOOR_NM = 0.05          # never recommend finer than this (small cells keep full res)
DMIN_TARGET_RANGE = 15.0      # reflection range we aim for on large cells (Al6Fe ≈ 14.9)
DMIN_RANGE_OK = 12.0          # range_at_floor <= this -> small cell, no bump needed
DMIN_RANGE_MODERATE = 18.0    # <= this -> moderate; above -> large (needs the bump)


def recommend_dmin(a_max_nm, target_range: float = DMIN_TARGET_RANGE,
                   floor: float = DMIN_FLOOR_NM):
    """Recommend a dmin (nm) for a crystal of largest lattice parameter
    ``a_max_nm`` so the EMsoft reflection range (~a_max/dmin) stays tractable.

    Small cells keep the fine ``floor`` dmin (they're already cheap); large
    cells get a larger dmin so their range lands near ``target_range``. dmin
    is rounded UP to the nearest 0.01 nm (coarser = safer/faster, never finer
    than the target). Returns a dict describing the recommendation.

    Returns:
        {
          a_max_nm, floor_dmin, recommended_dmin,
          range_at_floor, range_at_recommended,
          level: "ok" | "moderate" | "large",
        }
        or None if a_max_nm is not a positive number.
    """
    try:
        a = float(a_max_nm)
    except (TypeError, ValueError):
        return None
    if not (a > 0) or not math.isfinite(a):
        return None

    raw = a / float(target_range)
    rec = max(floor, math.ceil(raw * 100.0) / 100.0)  # round up to 0.01 nm, floored
    range_at_floor = a / floor
    range_at_rec = a / rec

    if range_at_floor <= DMIN_RANGE_OK:
        level = "ok"
    elif range_at_floor <= DMIN_RANGE_MODERATE:
        level = "moderate"
    else:
        level = "large"

    return {
        "a_max_nm": round(a, 4),
        "floor_dmin": round(float(floor), 3),
        "recommended_dmin": round(rec, 3),
        "range_at_floor": round(range_at_floor, 1),
        "range_at_recommended": round(range_at_rec, 1),
        "level": level,
    }


def read_xtal_max_lattice_nm(xtal_path, h5_open=None):
    """Read the largest lattice parameter a/b/c (nm) from an EMsoft .xtal file.

    Returns float (nm) or None if it can't be read. ``h5_open`` is injectable
    for testing (defaults to h5py.File).
    """
    try:
        if h5_open is None:
            import h5py
            h5_open = h5py.File
        with h5_open(str(xtal_path), "r") as f:
            lp = f["CrystalData"]["LatticeParameters"][()]
        import numpy as np
        arr = np.asarray(lp, dtype=float).ravel()
        if arr.size < 3:
            return None
        return float(max(arr[0], arr[1], arr[2]))
    except Exception:
        return None


def read_xtal_min_occupancy(xtal_path, h5_open=None):
    """Smallest site occupancy in an EMsoft .xtal (AtomData index 3), or None.

    occ < 1 means partially-occupied / mixed (disordered) sites. EMsoft's
    dynamical master-pattern computation assumes an ordered periodic crystal;
    heavily disordered structures (e.g. statistical quasicrystal approximants
    from the ICSD, occupancies down to ~0.1) can HANG the master-pattern run
    or yield physically meaningless patterns. ``h5_open`` injectable for tests.
    """
    try:
        if h5_open is None:
            import h5py
            h5_open = h5py.File
        with h5_open(str(xtal_path), "r") as f:
            ad = f["CrystalData"]["AtomData"][()]
        import numpy as np
        arr = np.asarray(ad, dtype=float)
        if arr.size == 0:
            return None
        occ = arr[3] if arr.shape[0] == 5 else arr[:, 3]
        if np.asarray(occ).size == 0:
            return None
        return float(np.min(occ))
    except Exception:
        return None
