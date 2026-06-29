"""
Server Mode Manager - Handles network connectivity and upload queue

Manages:
- Network connectivity checks (event-driven, no polling)
- Upload queue for background file transfers
- Server mode enable/disable state
"""

import queue
import threading
import logging
import time
from pathlib import Path
from typing import Optional
import configparser
import shutil

from PyQt5.QtCore import QObject, pyqtSignal, QSettings
from path_utils import resolve_path, discover_database_paths

logger = logging.getLogger(__name__)


class ServerModeManager(QObject):
    """
    Manages server mode functionality.

    Event-driven connectivity checks (NO automatic polling):
    - On GUI startup
    - Before simulation starts
    - When network paths are changed
    - When simulation tab is activated

    Signals:
        connectivity_changed(bool): Emitted when connectivity status changes
        upload_progress(str, int): Emitted during upload (filename, percentage)
        upload_status(str, str): Emitted for detailed status (filename, status_message)
    """

    # Signals
    connectivity_changed = pyqtSignal(bool)  # True = connected, False = offline
    offline_mode_changed = pyqtSignal(bool)  # True = offline, False = online
    upload_progress = pyqtSignal(str, int)   # filename, percentage
    upload_status = pyqtSignal(str, str)     # filename, status_message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.upload_queue = queue.Queue()
        self.uploader_thread: Optional[threading.Thread] = None
        self.is_enabled = True
        self.is_connected = False
        self.offline_mode = False
        self._first_check_done = False  # ensure signal fires on first check

        # Database root (single path, subfolders auto-discovered)
        self.database_root: Optional[Path] = None

        # Paths - Simulation results
        self.sht_database_path: Optional[Path] = None
        self.h5_cache_path: Optional[Path] = None

        # Paths - Dictionary library
        self.dictionary_library_path: Optional[Path] = None

        # Paths - Crystal file sync
        self.cif_network_path: Optional[Path] = None
        self.xtal_network_path: Optional[Path] = None

        # Settings
        self.settings = QSettings("KikuchipyGUI", "MainHub")

        # Load from settings
        self._load_from_settings()

    @property
    def needs_configuration(self) -> bool:
        """True when server mode is enabled but no database paths are set."""
        return (self.is_enabled
                and not self.database_root
                and not self.sht_database_path
                and not self.h5_cache_path)

    def _load_from_settings(self):
        """Load server mode configuration from QSettings.

        Priority: database_root (auto-discovery) > individual paths (legacy).
        """
        self.is_enabled = self.settings.value("simulation/server_mode_enabled", True, type=bool)
        self.offline_mode = self.settings.value("simulation/offline_mode", False, type=bool)

        # Try database_root first (new unified approach)
        db_root = self.settings.value("simulation/database_root", "")
        if db_root:
            self.database_root = resolve_path(db_root)
            discovered = discover_database_paths(self.database_root)
            if discovered["sht_database"]["exists"]:
                self.sht_database_path = discovered["sht_database"]["path"]
            if discovered["h5_cache"]["exists"]:
                self.h5_cache_path = discovered["h5_cache"]["path"]
            if discovered["dictionary_library"]["exists"]:
                self.dictionary_library_path = discovered["dictionary_library"]["path"]
            if discovered["cif_library"]["exists"]:
                self.cif_network_path = discovered["cif_library"]["path"]
            if discovered["xtal_library"]["exists"]:
                self.xtal_network_path = discovered["xtal_library"]["path"]
            logger.info(f"Database root: {self.database_root} (auto-discovered subfolders)")
        else:
            # Legacy: individual path settings
            sht_path = self.settings.value("simulation/sht_database_path", "")
            h5_path = self.settings.value("simulation/h5_cache_path", "")
            cif_path = self.settings.value("crystal/network_cif_path", "")
            xtal_path = self.settings.value("crystal/network_xtal_path", "")

            if sht_path:
                self.sht_database_path = resolve_path(sht_path)
            if h5_path:
                self.h5_cache_path = resolve_path(h5_path)
            if cif_path:
                self.cif_network_path = resolve_path(cif_path)
            if xtal_path:
                self.xtal_network_path = resolve_path(xtal_path)

        logger.info(f"Server mode loaded from settings: enabled={self.is_enabled}")

    def initialize(self, config: Optional[configparser.ConfigParser] = None):
        """
        Initialize server mode manager.

        Priority:
        1. QSettings (user-configured via GUI) - HIGHEST
        2. Config file (automation script defaults) - FALLBACK

        Args:
            config: Optional ConfigParser with network paths
        """
        try:
            # QSettings already loaded in __init__ via _load_from_settings()
            # Only use config as FALLBACK if QSettings are empty

            if not self.sht_database_path or not self.h5_cache_path:
                # No user settings - try config file as fallback
                if config and config.has_section("CentralDatabase"):
                    sht_str = config.get("CentralDatabase", "sht_database_path_unc", fallback="").strip()
                    h5_str = config.get("CentralDatabase", "h5_cache_path_unc", fallback="").strip()
                    if sht_str and h5_str:
                        self.sht_database_path = resolve_path(sht_str)
                        self.h5_cache_path = resolve_path(h5_str)
                        logger.info("Loaded network paths from config file (fallback)")
                    else:
                        logger.info("No network paths in config — configure via GUI: System Status > Server Settings")
                        # Don't disable server mode — user can still configure via dialog
                        return
                else:
                    logger.info("No network paths configured — configure via GUI: System Status > Server Settings")
                    return
            else:
                logger.info("Using network paths from QSettings (user-configured)")

            # Load crystal sync paths from config fallback if not in QSettings
            if not self.cif_network_path or not self.xtal_network_path:
                if config and config.has_section("CentralDatabase"):
                    cif_path = config.get("CentralDatabase", "cif_library_path_unc", fallback="")
                    xtal_path = config.get("CentralDatabase", "xtal_library_path_unc", fallback="")
                    if cif_path:
                        self.cif_network_path = resolve_path(cif_path)
                    if xtal_path:
                        self.xtal_network_path = resolve_path(xtal_path)
                    if cif_path or xtal_path:
                        logger.info("Loaded crystal sync paths from config file (fallback)")

            # Auto-create missing directories under database_root
            self._ensure_server_directories()

            # Start uploader thread
            if self.is_enabled:
                self._start_uploader_thread()

                # Initial connectivity check
                self.check_connectivity()

        except Exception as e:
            logger.error(f"Failed to initialize server mode: {e}")
            self.is_enabled = False

    def _ensure_server_directories(self):
        """Auto-create ALL standard subdirectories under database_root.

        Creates SHT, H5, CIF, and XTAL folders so that a fresh server
        location works out of the box (new user starting their own database).
        Uses the canonical names from DATABASE_SUBFOLDERS in path_utils.py.
        """
        if not self.database_root or not self.database_root.exists():
            return

        from path_utils import DATABASE_SUBFOLDERS

        # Map DATABASE_SUBFOLDERS key -> internal attribute name
        _attr_map = {
            "sht_database": "sht_database_path",
            "h5_cache": "h5_cache_path",
            "dictionary_library": "dictionary_library_path",
            "cif_library": "cif_network_path",
            "xtal_library": "xtal_network_path",
        }

        for key, dirname in DATABASE_SUBFOLDERS.items():
            attr = _attr_map[key]
            path = self.database_root / dirname
            current = getattr(self, attr, None)

            # Already exists and is set — nothing to do
            if current and current.exists():
                continue

            try:
                path.mkdir(parents=True, exist_ok=True)
                setattr(self, attr, path)
                logger.info(f"Auto-created server directory: {path}")
            except OSError as e:
                logger.warning(f"Could not create {path}: {e}")

    def check_connectivity(self) -> bool:
        """
        Check if network drives are accessible.

        This is event-driven - call manually when needed:
        - On GUI startup
        - Before simulation starts
        - When settings are saved
        - When simulation tab is activated

        Returns:
            True if connected, False otherwise
        """
        if not self.is_enabled or self.offline_mode:
            # Emit False on first call so the UI clears "Checking..."
            if not self._first_check_done or self.is_connected:
                self._first_check_done = True
                self.is_connected = False
                self.connectivity_changed.emit(False)
            return False

        try:
            # Guard: paths must be non-empty and absolute (not just "." or "")
            def _is_valid_path(p: Optional[Path]) -> bool:
                return bool(p and str(p).strip() and str(p) != '.' and p.is_absolute())

            connected = bool(
                _is_valid_path(self.sht_database_path) and
                _is_valid_path(self.h5_cache_path) and
                self.sht_database_path.is_dir() and
                self.h5_cache_path.is_dir()
            )

            # Emit on first check OR whenever status changes
            if connected != self.is_connected or not self._first_check_done:
                self._first_check_done = True
                self.is_connected = connected
                self.connectivity_changed.emit(connected)
                logger.info(f"Connectivity changed: {'ONLINE' if connected else 'OFFLINE'}")

            return connected

        except (OSError, PermissionError) as e:
            # Network error
            if self.is_connected or not self._first_check_done:
                self._first_check_done = True
                self.is_connected = False
                self.connectivity_changed.emit(False)
                logger.warning(f"Network connectivity lost: {e}")
            return False

    def _start_uploader_thread(self):
        """Start background uploader thread if not already running."""
        if self.uploader_thread is None or not self.uploader_thread.is_alive():
            self.uploader_thread = threading.Thread(
                target=self._uploader_worker,
                daemon=True
            )
            self.uploader_thread.start()
            logger.info("Upload worker thread started")

    def _uploader_worker(self):
        """
        Background worker that processes upload queue.

        Runs in daemon thread, processes uploads sequentially.
        Enhanced with:
        - Retry logic (3 attempts with exponential backoff)
        - Progress tracking
        - Detailed status messages
        """
        while True:
            try:
                task = self.upload_queue.get()

                if task is None:  # Shutdown signal
                    logger.info("Upload worker shutting down")
                    break

                source_path, dest_path = Path(task[0]), Path(task[1])

                if not source_path.exists():
                    logger.warning(f"Source file not found for upload: {source_path}")
                    self.upload_status.emit(source_path.name, "File not found")
                    continue

                # Get file size for logging
                file_size_gb = source_path.stat().st_size / (1024**3)
                file_size_mb = source_path.stat().st_size / (1024**2)

                # Retry logic (3 attempts)
                max_retries = 3
                success = False

                for attempt in range(max_retries):
                    try:
                        # Emit starting status
                        if attempt == 0:
                            logger.info(f"Uploading {source_path.name}... ({file_size_mb:.1f} MB)")
                            self.upload_status.emit(
                                source_path.name,
                                f"Starting upload ({file_size_mb:.1f} MB)"
                            )
                        else:
                            logger.info(f"Retry {attempt}/{max_retries} for {source_path.name}")
                            self.upload_status.emit(
                                source_path.name,
                                f"Retry {attempt}/{max_retries}"
                            )

                        # Create material subfolder
                        dest_path.parent.mkdir(parents=True, exist_ok=True)

                        # Copy with progress tracking
                        self._copy_with_progress(source_path, dest_path)

                        # Success
                        logger.info(f"Upload complete: {source_path.name}")
                        self.upload_progress.emit(source_path.name, 100)
                        self.upload_status.emit(source_path.name, "Complete")

                        success = True
                        break

                    except Exception as e:
                        logger.warning(f"Upload attempt {attempt+1} failed for {source_path.name}: {e}")

                        if attempt < max_retries - 1:
                            # Exponential backoff: 2^attempt seconds (2s, 4s, 8s)
                            backoff_time = 2 ** attempt
                            logger.info(f"Waiting {backoff_time}s before retry...")
                            self.upload_status.emit(
                                source_path.name,
                                f"Retrying in {backoff_time}s..."
                            )
                            time.sleep(backoff_time)
                        else:
                            # Final failure
                            logger.error(f"Upload failed after {max_retries} attempts: {source_path.name}")
                            self.upload_progress.emit(source_path.name, -1)  # -1 = failed
                            self.upload_status.emit(
                                source_path.name,
                                f"Failed: {str(e)}"
                            )

            except Exception as e:
                logger.error(f"Upload worker error: {e}")
            finally:
                self.upload_queue.task_done()

    def _copy_with_progress(self, source_path: Path, dest_path: Path, chunk_size: int = 1024*1024):
        """
        Copy file with progress tracking.

        Args:
            source_path: Source file
            dest_path: Destination file
            chunk_size: Chunk size in bytes (default: 1 MB)
        """
        file_size = source_path.stat().st_size
        copied = 0

        with open(source_path, 'rb') as src, open(dest_path, 'wb') as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break

                dst.write(chunk)
                copied += len(chunk)

                # Emit progress every 100 MB or at completion
                if copied % (100 * 1024 * 1024) == 0 or copied == file_size:
                    percentage = int((copied / file_size) * 100)
                    self.upload_progress.emit(source_path.name, percentage)

                    # Also emit status message
                    copied_mb = copied / (1024**2)
                    total_mb = file_size / (1024**2)
                    self.upload_status.emit(
                        source_path.name,
                        f"Uploading: {copied_mb:.1f}/{total_mb:.1f} MB ({percentage}%)"
                    )

    def queue_upload(self, source_path: str, dest_path: str):
        """
        Add file to upload queue.

        Only uploads if server mode is enabled AND connected.

        Args:
            source_path: Local file path (Windows path)
            dest_path: Network destination path
        """
        if not self.is_enabled:
            logger.info(f"Server mode disabled - skipping upload of {Path(source_path).name}")
            return

        if not self.is_connected:
            logger.warning(f"Network offline - skipping upload of {Path(source_path).name}")
            # TODO: Could implement retry queue here
            return

        self.upload_queue.put((source_path, dest_path))
        logger.info(f"Queued upload: {Path(source_path).name} → {dest_path}")

    def get_central_sht_path(self, filename: str, material_system: str = "Mixed") -> Optional[Path]:
        """
        Get central network path for a .sht file.

        Args:
            filename: SHT filename
            material_system: Material system subfolder (default: "Default")

        Returns:
            Full network path or None if server mode disabled
        """
        if not self.sht_database_path:
            return None

        return self.sht_database_path / material_system / filename

    def get_central_h5_path(self, filename: str, material_system: str = "Mixed") -> Optional[Path]:
        """
        Get central network path for a .h5 file.

        Args:
            filename: H5 filename
            material_system: Material system subfolder (default: "Default")

        Returns:
            Full network path or None if server mode disabled
        """
        if not self.h5_cache_path:
            return None

        return self.h5_cache_path / material_system / filename

    def get_central_dictionary_path(self, filename: str, material_system: str = "Mixed") -> Optional[Path]:
        """Get central network path for a dictionary .h5 file."""
        if not self.dictionary_library_path:
            return None
        return self.dictionary_library_path / material_system / filename

    def save_settings(self, enabled: bool, sht_path: str, h5_path: str):
        """
        Save server mode settings to QSettings and apply immediately.

        Args:
            enabled: Server mode enabled state
            sht_path: SHT database network path
            h5_path: H5 cache network path
        """
        self.settings.setValue("simulation/server_mode_enabled", enabled)
        self.settings.setValue("simulation/sht_database_path", sht_path)
        self.settings.setValue("simulation/h5_cache_path", h5_path)

        # Update internal state IMMEDIATELY
        old_enabled = self.is_enabled
        self.is_enabled = enabled
        self.sht_database_path = Path(sht_path) if sht_path else None
        self.h5_cache_path = Path(h5_path) if h5_path else None

        # Start/stop uploader thread based on enabled state
        if enabled and not old_enabled:
            # Enabling: Start uploader thread
            self._start_uploader_thread()
            logger.info("Server mode enabled - uploader thread started")
        elif not enabled and old_enabled:
            # Disabling: Stop uploader thread (graceful shutdown)
            self.shutdown()
            logger.info("Server mode disabled - uploader thread stopped")

        # Check connectivity immediately after settings change
        self.check_connectivity()

        logger.info(f"Server mode settings saved: enabled={enabled}")

    def save_crystal_paths(self, cif_path: str, xtal_path: str):
        """
        Save crystal sync network paths to QSettings.

        Args:
            cif_path: Network path for CIF library
            xtal_path: Network path for XTAL library
        """
        self.settings.setValue("crystal/network_cif_path", cif_path)
        self.settings.setValue("crystal/network_xtal_path", xtal_path)

        self.cif_network_path = Path(cif_path) if cif_path else None
        self.xtal_network_path = Path(xtal_path) if xtal_path else None

        logger.info(f"Crystal sync paths saved: cif={cif_path}, xtal={xtal_path}")

    def set_offline_mode(self, offline: bool):
        """
        Toggle offline mode. When offline, all network operations are skipped.

        Args:
            offline: True to enable offline mode
        """
        self.offline_mode = offline
        self.settings.setValue("simulation/offline_mode", offline)
        self.offline_mode_changed.emit(offline)
        logger.info(f"Offline mode: {'ON' if offline else 'OFF'}")

        if offline:
            # Update connectivity to reflect offline state
            if self.is_connected:
                self.is_connected = False
                self.connectivity_changed.emit(False)
        else:
            # Re-check connectivity when going back online
            self.check_connectivity()

    def get_central_cif_path(self, filename: str) -> Optional[Path]:
        """
        Get central network path for a CIF file.

        Args:
            filename: CIF filename

        Returns:
            Full network path or None if not configured
        """
        if not self.cif_network_path:
            return None
        return self.cif_network_path / filename

    def get_central_xtal_path(self, filename: str) -> Optional[Path]:
        """
        Get central network path for an XTAL file.

        Args:
            filename: XTAL filename

        Returns:
            Full network path or None if not configured
        """
        if not self.xtal_network_path:
            return None
        return self.xtal_network_path / filename

    def shutdown(self):
        """Graceful shutdown of uploader thread."""
        if self.uploader_thread and self.uploader_thread.is_alive():
            self.upload_queue.put(None)  # Shutdown signal
            logger.info("Waiting for upload worker to finish...")
            self.uploader_thread.join(timeout=10)

            if self.uploader_thread.is_alive():
                logger.warning("Upload worker did not finish in time")
