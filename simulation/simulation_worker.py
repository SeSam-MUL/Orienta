"""
Simulation Worker - Background Thread for Running EMsoft Simulations

Wraps the emsphinx_automation script in a QThread for GUI integration.
Emits signals for progress updates and completion.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional, Dict
import configparser
import getpass

from PyQt5.QtCore import QThread, pyqtSignal

from simulation.simulation_controller import (
    SimulationStatus, SimulationParameters, OutputType, sanitize_sim_name
)
from path_utils import ekev_to_kv_label

logger = logging.getLogger(__name__)


class _NoOpQueue:
    """Queue that silently discards items.

    Replaces the real queue.Queue that was passed to process_single_xtal().
    The automation script's upload_queue.put() calls become no-ops because
    all uploads are handled by FileSyncManager → ServerModeManager instead.
    """
    def put(self, item): pass
    def get(self): return None
    def task_done(self): pass
    def join(self): pass


class ProgressLogHandler(logging.Handler):
    """Forwards emsphinx_automation log records to the GUI progress_update signal.

    Installed on the emsphinx_automation logger before process_single_xtal() and
    removed in a finally block after. Thread-safe: pyqtSignal handles cross-thread
    delivery from QThread to the main thread automatically.
    """

    def __init__(self, emit_fn):
        super().__init__()
        self._emit_fn = emit_fn

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._emit_fn(msg)
        except Exception:
            pass


class SimulationWorker(QThread):
    """
    Background worker thread for running EMsoft simulations.

    Signals:
        progress_update(str): Emitted when progress message changes
        stage_changed(str, int): Emitted when simulation stage changes (stage_name, percentage)
        simulation_finished(bool, str, dict): Emitted when done (success, message, result_files)
    """

    progress_update = pyqtSignal(str)
    stage_changed = pyqtSignal(str, int)
    simulation_finished = pyqtSignal(bool, str, dict)

    def __init__(
        self,
        job_id: str,
        xtal_path: str,
        config: configparser.ConfigParser,
        params: SimulationParameters,
        server_manager=None,
        parent=None
    ):
        """
        Initialize simulation worker.

        Args:
            job_id: Unique job identifier
            xtal_path: Path to .xtal file (Windows path)
            config: Loaded ConfigParser instance
            params: Simulation parameters
            server_manager: ServerModeManager instance for network uploads (optional)
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.job_id = job_id
        self.xtal_path = Path(xtal_path)
        self.config = config
        self.params = params
        self.server_manager = server_manager
        self.upload_queue = _NoOpQueue()  # Automation uploads go via FileSyncManager
        self.username = getpass.getuser()

        # Flag to track cancellation
        self._is_cancelled = False

    def run(self):
        """
        Main worker thread execution.

        Runs the simulation using process_single_xtal from automation script.
        Supports output_type: sht_only, master_only, both.
        """
        try:
            self.progress_update.emit("Preparing simulation environment...")
            self.stage_changed.emit("Initialization", 0)

            # Convert Windows path to WSL path
            wsl_xtal_path = self._convert_to_wsl_path(str(self.xtal_path))

            # Update config with custom parameters (if different from defaults)
            self._update_config_with_params()

            # Bridge: inject QSettings server paths into config [CentralDatabase]
            # so automation script uses the same server location as the GUI
            self._sync_config_with_server_paths()

            # Inject material system for consistent subfolder naming
            material = self._get_material_for_xtal()
            self._material = material  # Store for skip-check and locate
            if not self.config.has_section("MaterialInfo"):
                self.config.add_section("MaterialInfo")
            self.config.set("MaterialInfo", "material_system", material)
            self.progress_update.emit(f"Material system: {material}")

            # Set output destination dynamically to project Database/temp_sim/
            # (no hardcoded paths — db_root is computed from project root)
            db_root = self._get_local_database_path()
            temp_sim_dir = db_root / "temp_sim"
            if not self.config.has_section("TemporaryDirectories"):
                self.config.add_section("TemporaryDirectories")
            self.config.set("TemporaryDirectories", "Temp_Base_Folder_Windows", str(temp_sim_dir))
            self.progress_update.emit(f"Local database: {db_root}")

            # Import automation script
            self.progress_update.emit("Loading automation module...")
            automation_module = self._import_automation_script()

            if automation_module is None:
                raise RuntimeError("Failed to import automation script")

            # Determine output type
            output_type = OutputType(self.params.output_type)
            stage_label = {
                OutputType.SHT_ONLY: "Monte Carlo + SHT",
                OutputType.MASTER_ONLY: "Monte Carlo + Master Pattern",
                OutputType.BOTH: "Monte Carlo + Master + SHT",
            }.get(output_type, "Monte Carlo + SHT")

            # --- Pre-check: skip if results already exist ---
            # Check 1: local Database/
            existing = self._check_local_database_for_results(output_type)
            if existing:
                self.progress_update.emit("Results already exist in local database:")
                for key, path in existing.items():
                    self.progress_update.emit(f"  {key}: {Path(path).name}")

                # Cross-sync: upload to server if not there yet
                if self.server_manager and self.server_manager.is_connected:
                    server_existing = self._check_server_for_results(output_type, material)
                    if not server_existing:
                        self.progress_update.emit("Uploading local results to server...")
                        self._queue_network_uploads(existing)

                self.progress_update.emit("Skipping simulation.")
                self.stage_changed.emit("Completed", 100)
                self.simulation_finished.emit(True, "Results already exist", existing)
                return

            # Check 2: server (authoritative paths from QSettings)
            server_existing = self._check_server_for_results(output_type, material)
            if server_existing:
                self.progress_update.emit("Results already exist on server:")
                for key, path in server_existing.items():
                    self.progress_update.emit(f"  {key}: {Path(path).name}")

                # Cross-sync: download from server to local
                self.progress_update.emit("Downloading server results to local database...")
                self._download_server_results(server_existing, material)

                self.progress_update.emit("Skipping simulation.")
                self.stage_changed.emit("Completed", 100)
                self.simulation_finished.emit(True, "Results downloaded from server", server_existing)
                return

            self.progress_update.emit(f"Starting EMsoft workflow ({output_type.value})...")
            self.stage_changed.emit(stage_label, 10)

            # Store output_type in config for automation script
            if not self.config.has_section("OutputOptions"):
                self.config.add_section("OutputOptions")
            self.config.set("OutputOptions", "output_type", self.params.output_type)

            # Emit diagnostic config info so user sees what's being used
            try:
                platid = self.config.get("DefaultSimulationParametersEMMCOpenCL", "platid", fallback=str(self.params.platid))
                devid = self.config.get("DefaultSimulationParametersEMMCOpenCL", "devid", fallback=str(self.params.devid))
                grpsz = self.config.get("DefaultSimulationParametersEMMCOpenCL", "globalworkgrpsz", fallback=str(self.params.globalworkgrpsz))
                emmc_exe = self.config.get("EMsoftPaths", "emmcopencl_executable_wsl", fallback="(not set)")
                sht_exe = self.config.get("EMsoftPaths", "emebsdmastersht_executable_wsl", fallback="(not set)")
                master_exe = self.config.get("EMsoftPaths", "emebsdmaster_executable_wsl", fallback="(not set)")
                self.progress_update.emit("--- Simulation Config ---")
                self.progress_update.emit(f"MC program: {self.params.mc_program or 'EMMCOpenCL'}")
                self.progress_update.emit(f"Output type: {self.params.output_type}")
                self.progress_update.emit(f"platid={platid}, devid={devid}, globalworkgrpsz={grpsz}")
                self.progress_update.emit(f"EMMCOpenCL:       {emmc_exe}")
                self.progress_update.emit(f"EMEBSDmasterSHT:  {sht_exe}")
                self.progress_update.emit(f"EMEBSDmaster:     {master_exe}")
                self.progress_update.emit(f"xtal:             {wsl_xtal_path}")
                self.progress_update.emit("-------------------------")
            except Exception:
                pass

            # Install log handler: forwards emsphinx_automation log messages to GUI panel
            log_handler = ProgressLogHandler(self.progress_update.emit)
            log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            log_handler.setLevel(logging.INFO)
            auto_logger = logging.getLogger("emsphinx_automation")
            auto_logger.addHandler(log_handler)

            try:
                success = automation_module.process_single_xtal(
                    xtal_wsl=wsl_xtal_path,
                    config=self.config,
                    upload_queue=self.upload_queue,
                    username=self.username
                )
            finally:
                auto_logger.removeHandler(log_handler)

            # Check if cancelled during execution
            if self._is_cancelled:
                self.progress_update.emit("Simulation cancelled")
                self.simulation_finished.emit(False, "Cancelled by user", {})
                return

            if success:
                self.progress_update.emit("Simulation completed successfully!")
                self.stage_changed.emit("Completed", 100)

                # Try to locate result files
                result_files = self._locate_result_files()

                # Queue uploads to network if server mode enabled
                if self.server_manager and self.server_manager.is_enabled:
                    self._queue_network_uploads(result_files)

                self.simulation_finished.emit(
                    True,
                    "Simulation completed successfully",
                    result_files
                )
            else:
                self.progress_update.emit("Simulation failed")
                self.simulation_finished.emit(False, "Simulation process returned False", {})

        except Exception as e:
            error_msg = f"Simulation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.progress_update.emit(f"Error: {str(e)}")
            self.simulation_finished.emit(False, error_msg, {})

    def cancel(self):
        """Request cancellation of the simulation."""
        self._is_cancelled = True
        logger.info(f"Cancellation requested for job {self.job_id}")

    def _convert_to_wsl_path(self, path: str) -> str:
        """
        Convert path to WSL-compatible format.

        Handles both native WSL paths (already start with /) and
        Windows paths that need conversion (C:\\...).

        Args:
            path: File path (Windows or WSL format)

        Returns:
            WSL path (e.g., /mnt/c/Users/... or /home/user/...)
        """
        # Already a WSL/Unix path - return as-is
        if path.startswith('/'):
            return path

        try:
            automation_dir = self._get_automation_dir()
            if str(automation_dir) not in sys.path:
                sys.path.insert(0, str(automation_dir))
            from emsphinx_automation import normalize_to_wsl_path

            return normalize_to_wsl_path(path)

        except Exception as e:
            logger.error(f"Path conversion failed: {e}")
            # Fallback: generic Windows -> WSL drive letter conversion
            import re
            result = path.replace("\\", "/")
            result = re.sub(r'^([A-Za-z]):', lambda m: f'/mnt/{m.group(1).lower()}', result)
            return result

    def _get_automation_dir(self) -> Path:
        """Get path to automation script directory."""
        script_dir = Path(__file__).parent.parent
        automation_dir = script_dir / "crystal-structures-for-ebsd-main" / "_Phyton_Automization" / "windwos_to_WSL"
        return automation_dir

    def _import_automation_script(self):
        """
        Import the emsphinx_automation module.

        Returns:
            Imported module or None if import fails
        """
        try:
            automation_dir = self._get_automation_dir()

            if str(automation_dir) not in sys.path:
                sys.path.insert(0, str(automation_dir))

            import emsphinx_automation
            return emsphinx_automation

        except ImportError as e:
            logger.error(f"Failed to import automation script: {e}")
            return None

    def _update_config_with_params(self):
        """Update config with custom simulation parameters."""
        try:
            # Update Monte Carlo parameters
            if not self.config.has_section("DefaultSimulationParametersEMMCOpenCL"):
                self.config.add_section("DefaultSimulationParametersEMMCOpenCL")

            section = "DefaultSimulationParametersEMMCOpenCL"
            self.config.set(section, "EkeV", str(self.params.ekev))
            self.config.set(section, "sig", str(self.params.sig))
            self.config.set(section, "omega", str(self.params.omega))
            self.config.set(section, "numsx", str(self.params.numsx))
            self.config.set(section, "totnum_el", str(self.params.totnum_el))
            self.config.set(section, "num_el", str(self.params.num_el))
            self.config.set(section, "multiplier", str(self.params.multiplier))
            # Advanced MC parameters
            self.config.set(section, "depthmax", str(self.params.depthmax))
            self.config.set(section, "depthstep", str(self.params.depthstep))
            # Only set GPU-specific params when actually using GPU mode
            is_gpu_mode = self.params.mc_program == "EMMCOpenCL"
            if is_gpu_mode:
                self.config.set(section, "platid", str(self.params.platid))
                self.config.set(section, "devid", str(self.params.devid))
                self.config.set(section, "globalworkgrpsz", str(self.params.globalworkgrpsz))

            # Update SHT parameters
            if not self.config.has_section("DefaultSimulationParametersEMEBSDmasterSHT"):
                self.config.add_section("DefaultSimulationParametersEMEBSDmasterSHT")

            section = "DefaultSimulationParametersEMEBSDmasterSHT"
            self.config.set(section, "dmin", str(self.params.dmin))
            self.config.set(section, "Butterfly", str(self.params.bandwidth))

            # Update Master pattern parameters
            if not self.config.has_section("DefaultSimulationParametersEMEBSDmaster"):
                self.config.add_section("DefaultSimulationParametersEMEBSDmaster")

            section = "DefaultSimulationParametersEMEBSDmaster"
            self.config.set(section, "dmin", str(self.params.dmin))
            self.config.set(section, "npx", str(self.params.npx))
            self.config.set(section, "nthreads", str(self.params.nthreads))
            # Advanced EMEBSDmaster parameters (Fortran boolean format)
            self.config.set(section, "combinesites",
                            ".TRUE." if self.params.combinesites else ".FALSE.")
            self.config.set(section, "useEnergyWeighting",
                            ".TRUE." if self.params.useEnergyWeighting else ".FALSE.")
            self.config.set(section, "doLegendre",
                            ".TRUE." if self.params.doLegendre else ".FALSE.")
            self.config.set(section, "Esel", str(self.params.Esel))
            self.config.set(section, "uniform",
                            ".TRUE." if self.params.uniform else ".FALSE.")

            # Update GPU Master pattern parameters (EMEBSDmasterOpenCL)
            is_gpu_master = self.params.master_program == "EMEBSDmasterOpenCL"
            if is_gpu_master:
                section_gpu = "DefaultSimulationParametersEMEBSDmasterOpenCL"
                if not self.config.has_section(section_gpu):
                    self.config.add_section(section_gpu)
                self.config.set(section_gpu, "dmin", str(self.params.dmin))
                self.config.set(section_gpu, "npx", str(self.params.npx))
                self.config.set(section_gpu, "nthreads", str(self.params.nthreads))
                self.config.set(section_gpu, "platid", str(self.params.platid))
                self.config.set(section_gpu, "devid", str(self.params.devid))
                self.config.set(section_gpu, "globalworkgrpsz",
                                str(self.params.globalworkgrpsz))
                self.config.set(section_gpu, "blocksize", str(self.params.blocksize))
                self.config.set(section_gpu, "Notify", "off")

            # Update output type
            if not self.config.has_section("OutputOptions"):
                self.config.add_section("OutputOptions")
            self.config.set("OutputOptions", "output_type", self.params.output_type)
            self.config.set("OutputOptions", "master_program",
                            self.params.master_program or "EMEBSDmaster")

            logger.info("Updated config with custom parameters")

        except Exception as e:
            logger.error(f"Failed to update config: {e}")

    def _sync_config_with_server_paths(self):
        """Inject QSettings server paths into config's [CentralDatabase] section.

        Bridges the gap between the GUI's authoritative server path
        (from QSettings via ServerModeManager) and the automation script's
        config-based path reading.  After this call, process_single_xtal()
        reads the same server paths as the rest of the GUI.
        """
        if not self.server_manager or not self.server_manager.is_enabled:
            # No server mode — remove CentralDatabase so automation
            # won't try to use stale config paths
            if self.config.has_section("CentralDatabase"):
                self.config.remove_section("CentralDatabase")
            return

        if not self.config.has_section("CentralDatabase"):
            self.config.add_section("CentralDatabase")

        path_map = {
            "sht_database_path_unc": self.server_manager.sht_database_path,
            "h5_cache_path_unc": self.server_manager.h5_cache_path,
            "cif_library_path_unc": self.server_manager.cif_network_path,
            "xtal_library_path_unc": self.server_manager.xtal_network_path,
        }
        for key, path in path_map.items():
            if path:
                self.config.set("CentralDatabase", key, str(path))

        logger.info("Synced CentralDatabase paths from QSettings to config")

    def _get_material_for_xtal(self) -> str:
        """Extract material system from .xtal file for consistent subfolder naming.

        Returns the same material string that FileSyncManager uses for uploads,
        ensuring automation skip-check and upload both use the same subfolder.
        """
        try:
            from simulation.file_sync_manager import FileSyncManager
            sync = FileSyncManager(self.server_manager)
            return sync.extract_material_from_xtal(str(self.xtal_path))
        except Exception as e:
            logger.warning(f"Material extraction failed, using 'Default': {e}")
            return "Default"

    def _get_local_database_path(self) -> Path:
        """Return the project's local Database/ folder (no hardcoded paths).

        Computed from this file's location: simulation_worker.py lives in
        simulation/, whose parent is the project root.
        """
        return Path(__file__).parent.parent / "Database"

    def _check_server_for_results(self, output_type: OutputType, material: str) -> Dict[str, str]:
        """Check if simulation results already exist on the server.

        Searches ALL material subfolders on the server (material/, Default/, etc.)
        so old files are found regardless of folder naming.

        Returns:
            Non-empty dict if ALL requested outputs exist on server,
            empty dict otherwise.
        """
        if not self.server_manager or not self.server_manager.is_connected:
            return {}

        found = {}
        sim_name = self.xtal_path.stem
        safe_name = sanitize_sim_name(sim_name)

        try:
            sht_base = self.server_manager.sht_database_path
            h5_base = self.server_manager.h5_cache_path

            if not sht_base or not h5_base:
                return {}

            # H5 check (Monte Carlo) — search ALL subfolders with sanitized name
            h5_pattern = f"*/{safe_name}_E{ekev_to_kv_label(self.params.ekev)}kV*.h5"
            h5_files = list(h5_base.glob(h5_pattern)) if h5_base.exists() else []
            h5_files = [f for f in h5_files if "master" not in f.name]
            if h5_files:
                found["h5"] = str(h5_files[0])

            # SHT check — search ALL subfolders for files containing original name
            if output_type in (OutputType.SHT_ONLY, OutputType.BOTH):
                sht_files = list(sht_base.glob("*/*.sht")) if sht_base.exists() else []
                sht_files = [f for f in sht_files if sim_name in f.name]
                if sht_files:
                    found["sht"] = str(sht_files[0])
                else:
                    return {}  # SHT requested but not found

            # Master check — search ALL subfolders with sanitized name
            if output_type in (OutputType.MASTER_ONLY, OutputType.BOTH):
                master_files = [f for f in h5_base.glob("*/*master*.h5")
                                if safe_name in f.name] if h5_base.exists() else []
                if master_files:
                    found["master"] = str(master_files[0])
                else:
                    return {}  # Master requested but not found

            if "h5" not in found:
                return {}

        except (OSError, PermissionError) as e:
            logger.debug(f"Server check failed (offline?): {e}")
            return {}

        return found

    def _check_local_database_for_results(self, output_type: OutputType) -> Dict[str, str]:
        """Check if simulation results already exist in the local Database.

        Searches ALL material subfolders (rglob) so files in Default/, Fe/,
        Al/, Si/ etc. are all found.  If a file is discovered in a wrong
        folder (e.g. Default/) and self._material is set, it is lazily
        migrated to the correct material folder.

        Returns:
            Non-empty dict with found file paths if ALL requested outputs exist,
            empty dict otherwise (meaning simulation should proceed).
        """
        found = {}
        sim_name = self.xtal_path.stem
        # EMsoft sanitizes filenames: α→alpha, spaces→_, etc.
        safe_name = sanitize_sim_name(sim_name)

        try:
            db_root = self._get_local_database_path()
            h5_base = db_root / "EBSD_H5_Cache"
            sht_base = db_root / "EBSD_SHT_Database"

            # H5 (Monte Carlo) — search ALL subfolders with sanitized name
            h5_pattern = f"*/{safe_name}_E{ekev_to_kv_label(self.params.ekev)}kV*.h5"
            h5_files = list(h5_base.glob(h5_pattern)) if h5_base.exists() else []
            h5_files = [f for f in h5_files if "master" not in f.name]
            if h5_files:
                found["h5"] = str(self._maybe_migrate(h5_files[0], h5_base))

            # SHT — search ALL subfolders for files containing original stem
            # (SHT filenames preserve original names in parentheses)
            if output_type in (OutputType.SHT_ONLY, OutputType.BOTH):
                sht_files = list(sht_base.glob("*/*.sht")) if sht_base.exists() else []
                sht_files = [f for f in sht_files if sim_name in f.name]
                if sht_files:
                    found["sht"] = str(self._maybe_migrate(sht_files[0], sht_base))
                else:
                    return {}  # SHT requested but not found

            # Master — search ALL subfolders with sanitized name
            if output_type in (OutputType.MASTER_ONLY, OutputType.BOTH):
                master_pattern = f"*/{safe_name}_master*.h5"
                master_files = list(h5_base.glob(master_pattern)) if h5_base.exists() else []
                if not master_files:
                    master_files = [f for f in h5_base.glob("*/*master*.h5")
                                    if safe_name in f.name]
                if master_files:
                    found["master"] = str(self._maybe_migrate(master_files[0], h5_base))
                else:
                    return {}  # Master requested but not found

            if "h5" not in found:
                return {}

        except Exception as e:
            logger.debug(f"Local database check failed: {e}")
            return {}

        return found

    def _maybe_migrate(self, file_path: Path, base_dir: Path) -> Path:
        """Move a file from Default/ (or wrong folder) to the correct material folder.

        Only migrates if self._material is set and the file is in a different folder.
        Returns the (possibly new) file path.
        """
        material = getattr(self, '_material', None)
        if not material or material in ("Default", "Mixed"):
            return file_path

        current_folder = file_path.parent.name
        if current_folder == material:
            return file_path  # Already in correct folder

        dest_dir = base_dir / material
        dest_path = dest_dir / file_path.name
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(dest_path))
            logger.info(f"Migrated: {current_folder}/{file_path.name} → {material}/{file_path.name}")
            return dest_path
        except Exception as e:
            logger.warning(f"Migration failed for {file_path.name}: {e}")
            return file_path  # Keep using original path

    def _download_server_results(self, server_files: Dict[str, str], material: str):
        """Download server results to local Database for offline access.

        Copies files from server paths to local Database/{subfolder}/{material}/.
        """
        db_root = self._get_local_database_path()

        _dest_map = {
            "h5": db_root / "EBSD_H5_Cache" / material,
            "sht": db_root / "EBSD_SHT_Database" / material,
            "master": db_root / "EBSD_H5_Cache" / material,
        }

        for key, server_path_str in server_files.items():
            dest_dir = _dest_map.get(key)
            if not dest_dir:
                continue
            server_path = Path(server_path_str)
            if not server_path.exists():
                continue
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / server_path.name
                if not dest_file.exists():
                    shutil.copy2(str(server_path), str(dest_file))
                    self.progress_update.emit(f"  Downloaded: {server_path.name} → {key}/{material}/")
                    logger.info(f"Downloaded from server: {server_path.name}")
            except Exception as e:
                logger.warning(f"Failed to download {server_path.name}: {e}")

    def _locate_result_files(self) -> Dict[str, str]:
        """Locate result files after simulation and move them into the Database structure.

        Files produced by the automation script land in Database/temp_sim/{sim_name}/.
        This method moves them to their final locations using material-based folders:
          - H5  → Database/EBSD_H5_Cache/{material}/
          - SHT → Database/EBSD_SHT_Database/{material}/
          - Master → Database/EBSD_H5_Cache/{material}/

        Returns:
            Dictionary with keys "h5", "sht", "master" pointing to final file paths
        """
        result_files = {}
        output_type = OutputType(self.params.output_type)
        sim_name = self.xtal_path.stem
        safe_name = sanitize_sim_name(sim_name)
        material = getattr(self, '_material', sim_name)

        try:
            db_root = self._get_local_database_path()
            temp_base = db_root / "temp_sim"

            h5_dest_dir = db_root / "EBSD_H5_Cache" / material
            sht_dest_dir = db_root / "EBSD_SHT_Database" / material

            # --- H5 (Monte Carlo, always produced) ---
            # Automation uses sanitized name for H5: e.g. alpha-_AlMnSi_E20kV...
            h5_pattern = f"{safe_name}_E{ekev_to_kv_label(self.params.ekev)}kV*.h5"
            # temp_sim uses sanitized name as subfolder
            temp_sim_dir = temp_base / safe_name
            h5_files = list(temp_sim_dir.glob(h5_pattern)) if temp_sim_dir.exists() else []
            if not h5_files:
                # Also check original name as temp subfolder (backwards compat)
                h5_files = list((temp_base / sim_name).glob(h5_pattern)) if (temp_base / sim_name).exists() else []
            if not h5_files:
                # Already moved to final location?  Search all subfolders.
                h5_files = list(db_root.glob(f"EBSD_H5_Cache/*/{h5_pattern}"))
                h5_files = [f for f in h5_files if "master" not in f.name]
            if h5_files:
                h5_src = h5_files[0]
                if h5_src.parent != h5_dest_dir:
                    h5_dest_dir.mkdir(parents=True, exist_ok=True)
                    h5_dest = h5_dest_dir / h5_src.name
                    shutil.move(str(h5_src), str(h5_dest))
                    result_files["h5"] = str(h5_dest)
                    self.progress_update.emit(f"H5  → Database/EBSD_H5_Cache/{material}/{h5_src.name}")
                else:
                    result_files["h5"] = str(h5_src)

            # --- SHT ---
            if output_type in (OutputType.SHT_ONLY, OutputType.BOTH):
                # Automation uses sanitized name for temp subfolder
                sht_temp = temp_sim_dir / "SHT_output" if temp_sim_dir.exists() else temp_base / sim_name / "SHT_output"
                sht_files = list(sht_temp.glob("*.sht")) if sht_temp.exists() else []
                if not sht_files:
                    # Search all subfolders for SHT containing original sim_name
                    sht_files = [f for f in db_root.glob("EBSD_SHT_Database/*/*.sht")
                                 if sim_name in f.name]
                if sht_files:
                    sht_src = sht_files[0]
                    if sht_src.parent != sht_dest_dir:
                        sht_dest_dir.mkdir(parents=True, exist_ok=True)
                        sht_dest = sht_dest_dir / sht_src.name
                        shutil.move(str(sht_src), str(sht_dest))
                        result_files["sht"] = str(sht_dest)
                        self.progress_update.emit(f"SHT → Database/EBSD_SHT_Database/{material}/{sht_src.name}")
                    else:
                        result_files["sht"] = str(sht_src)

            # --- Master pattern ---
            if output_type in (OutputType.MASTER_ONLY, OutputType.BOTH):
                master_temp = temp_sim_dir / "master_output" if temp_sim_dir.exists() else temp_base / sim_name / "master_output"
                master_files = list(master_temp.glob("*.h5")) if master_temp.exists() else []
                if not master_files:
                    # Search all subfolders for master files with sanitized name
                    master_files = [f for f in db_root.glob("EBSD_H5_Cache/*/*.h5")
                                    if "master" in f.name and safe_name in f.name]
                if master_files:
                    master_src = master_files[0]
                    if master_src.parent != h5_dest_dir:
                        h5_dest_dir.mkdir(parents=True, exist_ok=True)
                        master_dest = h5_dest_dir / master_src.name
                        shutil.move(str(master_src), str(master_dest))
                        result_files["master"] = str(master_dest)
                        self.progress_update.emit(f"Master → Database/EBSD_H5_Cache/{material}/{master_src.name}")
                    else:
                        result_files["master"] = str(master_src)

            # Cleanup temp_sim/{sim_name}/ if no files remain
            temp_sim_subdir = temp_base / sim_name
            if temp_sim_subdir.exists():
                remaining_files = [f for f in temp_sim_subdir.rglob("*") if f.is_file()]
                if not remaining_files:
                    shutil.rmtree(str(temp_sim_subdir), ignore_errors=True)
                    logger.info("Cleaned up temp_sim/%s", sim_name)

        except Exception as e:
            logger.error(f"Failed to locate/organize result files: {e}")

        return result_files

    def _queue_network_uploads(self, result_files: Dict[str, str]):
        """
        Queue result files for upload to network drives.

        Uses FileSyncManager to extract material from .xtal and organize uploads
        by material subfolder (Fe/, Al/, Si/, etc.)

        Args:
            result_files: Dictionary with "h5" and "sht" file paths
        """
        if not self.server_manager or not self.server_manager.is_connected:
            logger.info("Server mode disabled or offline - skipping uploads")
            return

        try:
            # Use FileSyncManager for material-aware uploads
            from simulation.file_sync_manager import FileSyncManager

            sync_manager = FileSyncManager(self.server_manager)

            # Queue uploads with automatic material extraction
            sync_manager.queue_simulation_results(
                result_files=result_files,
                xtal_path=str(self.xtal_path)
            )

            # Emit progress message
            material = sync_manager.extract_material_from_xtal(str(self.xtal_path))
            self.progress_update.emit(
                f"Queued uploads for material: {material} "
                f"({len(result_files)} files)"
            )

        except Exception as e:
            logger.error(f"Failed to queue uploads: {e}")
