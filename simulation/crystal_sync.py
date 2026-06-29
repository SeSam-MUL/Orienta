"""
Crystal Sync Manager - Bidirectional CIF/XTAL file synchronization

Manages:
- QTimer-based periodic sync (5 min interval)
- Local → Server upload of new/modified files
- Server → Local download of new files
- Conflict detection via file hash comparison
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

logger = logging.getLogger(__name__)

# Default sync interval: 5 minutes
DEFAULT_SYNC_INTERVAL_MS = 5 * 60 * 1000


class SyncStatus(Enum):
    """Status of the sync manager."""
    IDLE = "idle"
    SYNCING = "syncing"
    ERROR = "error"
    DISABLED = "disabled"


class ConflictAction(Enum):
    """Resolution action for file conflicts."""
    KEEP_LOCAL = "keep_local"
    KEEP_SERVER = "keep_server"
    KEEP_BOTH = "keep_both"     # Rename one with timestamp suffix
    SKIP = "skip"


@dataclass
class SyncConflict:
    """Represents a file conflict between local and server."""
    filename: str
    file_type: str              # "cif" or "xtal"
    local_path: Path
    server_path: Path
    local_hash: str
    server_hash: str
    local_mtime: datetime
    server_mtime: datetime


@dataclass
class SyncResult:
    """Result of a sync operation."""
    uploaded: List[str] = field(default_factory=list)
    downloaded: List[str] = field(default_factory=list)
    conflicts: List[SyncConflict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


def file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class CrystalSyncManager(QObject):
    """
    Manages bidirectional sync of CIF and XTAL files between local and server.

    Signals:
        sync_started(): Emitted when sync begins
        sync_finished(SyncResult): Emitted when sync completes
        sync_error(str): Emitted on sync error
        conflict_detected(list): Emitted with list of SyncConflict objects needing resolution
        status_changed(str): Emitted when sync status changes
    """

    sync_started = pyqtSignal()
    sync_finished = pyqtSignal(object)          # SyncResult
    sync_error = pyqtSignal(str)
    conflict_detected = pyqtSignal(list)         # List[SyncConflict]
    status_changed = pyqtSignal(str)             # SyncStatus.value

    def __init__(self, server_manager, parent=None):
        """
        Initialize CrystalSyncManager.

        Args:
            server_manager: ServerModeManager instance
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.server_manager = server_manager
        self.status = SyncStatus.IDLE
        self.last_sync: Optional[datetime] = None
        self.sync_history: List[SyncResult] = []

        # Local paths (set via configure())
        self.local_cif_path: Optional[Path] = None
        self.local_xtal_path: Optional[Path] = None

        # Pending conflicts awaiting user resolution
        self._pending_conflicts: List[SyncConflict] = []

        # Timer for periodic sync
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self.sync)

    def configure(self, local_cif_path: Path, local_xtal_path: Path):
        """
        Configure local directory paths for sync.

        Args:
            local_cif_path: Path to local CIF files directory
            local_xtal_path: Path to local XTAL files directory
        """
        self.local_cif_path = local_cif_path
        self.local_xtal_path = local_xtal_path
        logger.info(f"Crystal sync configured: cif={local_cif_path}, xtal={local_xtal_path}")

    def start_periodic_sync(self, interval_ms: int = DEFAULT_SYNC_INTERVAL_MS):
        """
        Start periodic sync timer.

        Args:
            interval_ms: Sync interval in milliseconds (default: 5 min)
        """
        if not self._is_configured():
            logger.warning("Cannot start periodic sync: not configured")
            return

        self._sync_timer.start(interval_ms)
        logger.info(f"Periodic crystal sync started (interval: {interval_ms // 1000}s)")

        # Run initial sync immediately
        self.sync()

    def stop_periodic_sync(self):
        """Stop periodic sync timer."""
        self._sync_timer.stop()
        logger.info("Periodic crystal sync stopped")

    def is_running(self) -> bool:
        """Check if periodic sync is active."""
        return self._sync_timer.isActive()

    def sync(self):
        """
        Run a single sync cycle.

        Compares local and server directories, uploads/downloads as needed,
        and detects conflicts.
        """
        if not self._is_configured():
            logger.warning("Sync skipped: not configured")
            return

        if self.status == SyncStatus.SYNCING:
            logger.info("Sync already in progress, skipping")
            return

        self._set_status(SyncStatus.SYNCING)
        self.sync_started.emit()

        result = SyncResult()

        try:
            # Sync CIF files
            self._sync_directory(
                local_dir=self.local_cif_path,
                server_dir=self.server_manager.cif_network_path,
                file_type="cif",
                extensions={".cif"},
                result=result
            )

            # Sync XTAL files
            self._sync_directory(
                local_dir=self.local_xtal_path,
                server_dir=self.server_manager.xtal_network_path,
                file_type="xtal",
                extensions={".xtal"},
                result=result
            )

            self.last_sync = datetime.now()
            result.timestamp = self.last_sync
            self.sync_history.append(result)

            # Handle conflicts
            if result.conflicts:
                self._pending_conflicts = result.conflicts
                self.conflict_detected.emit(result.conflicts)
                logger.info(f"Sync found {len(result.conflicts)} conflicts")

            self._set_status(SyncStatus.IDLE)
            self.sync_finished.emit(result)

            logger.info(
                f"Sync complete: {len(result.uploaded)} uploaded, "
                f"{len(result.downloaded)} downloaded, "
                f"{len(result.conflicts)} conflicts"
            )

        except Exception as e:
            error_msg = f"Sync failed: {e}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)
            self._set_status(SyncStatus.ERROR)
            self.sync_error.emit(error_msg)

    def _sync_directory(
        self,
        local_dir: Optional[Path],
        server_dir: Optional[Path],
        file_type: str,
        extensions: set,
        result: SyncResult
    ):
        """
        Sync a single directory pair (local <-> server).

        Args:
            local_dir: Local directory path
            server_dir: Server directory path
            file_type: "cif" or "xtal"
            extensions: Set of file extensions to sync
            result: SyncResult to accumulate into
        """
        if not local_dir or not server_dir:
            logger.debug(f"Skipping {file_type} sync: paths not configured")
            return

        if not local_dir.is_dir():
            logger.warning(f"Local {file_type} directory not found: {local_dir}")
            return

        if not server_dir.is_dir():
            logger.warning(f"Server {file_type} directory not accessible: {server_dir}")
            result.errors.append(f"Server {file_type} path not accessible")
            return

        # Collect files from both sides
        local_files = self._list_files(local_dir, extensions)
        server_files = self._list_files(server_dir, extensions)

        local_names = set(local_files.keys())
        server_names = set(server_files.keys())

        # Files only on local → upload to server
        for name in local_names - server_names:
            try:
                self._copy_file(local_files[name], server_dir / name)
                result.uploaded.append(name)
                logger.info(f"Uploaded {file_type}: {name}")
            except Exception as e:
                result.errors.append(f"Upload failed for {name}: {e}")
                logger.error(f"Failed to upload {name}: {e}")

        # Files only on server → download to local
        for name in server_names - local_names:
            try:
                self._copy_file(server_files[name], local_dir / name)
                result.downloaded.append(name)
                logger.info(f"Downloaded {file_type}: {name}")
            except Exception as e:
                result.errors.append(f"Download failed for {name}: {e}")
                logger.error(f"Failed to download {name}: {e}")

        # Files on both sides → check for conflicts
        for name in local_names & server_names:
            local_path = local_files[name]
            server_path = server_files[name]

            try:
                local_h = file_hash(local_path)
                server_h = file_hash(server_path)

                if local_h != server_h:
                    # Content differs → conflict
                    conflict = SyncConflict(
                        filename=name,
                        file_type=file_type,
                        local_path=local_path,
                        server_path=server_path,
                        local_hash=local_h,
                        server_hash=server_h,
                        local_mtime=datetime.fromtimestamp(local_path.stat().st_mtime),
                        server_mtime=datetime.fromtimestamp(server_path.stat().st_mtime),
                    )
                    result.conflicts.append(conflict)
            except Exception as e:
                result.errors.append(f"Hash comparison failed for {name}: {e}")
                logger.error(f"Hash comparison failed for {name}: {e}")

    def resolve_conflict(self, conflict: SyncConflict, action: ConflictAction):
        """
        Resolve a single file conflict.

        Args:
            conflict: The conflict to resolve
            action: Resolution action to take
        """
        try:
            if action == ConflictAction.KEEP_LOCAL:
                self._copy_file(conflict.local_path, conflict.server_path)
                logger.info(f"Conflict resolved (keep local): {conflict.filename}")

            elif action == ConflictAction.KEEP_SERVER:
                self._copy_file(conflict.server_path, conflict.local_path)
                logger.info(f"Conflict resolved (keep server): {conflict.filename}")

            elif action == ConflictAction.KEEP_BOTH:
                # Rename server copy with timestamp suffix
                stem = conflict.server_path.stem
                suffix = conflict.server_path.suffix
                ts = conflict.server_mtime.strftime("%Y%m%d_%H%M%S")
                renamed = conflict.server_path.parent / f"{stem}_server_{ts}{suffix}"
                self._copy_file(conflict.server_path, renamed)
                # Then overwrite server with local
                self._copy_file(conflict.local_path, conflict.server_path)
                logger.info(f"Conflict resolved (keep both): {conflict.filename}")

            elif action == ConflictAction.SKIP:
                logger.info(f"Conflict skipped: {conflict.filename}")

            # Remove from pending
            if conflict in self._pending_conflicts:
                self._pending_conflicts.remove(conflict)

        except Exception as e:
            logger.error(f"Failed to resolve conflict for {conflict.filename}: {e}")

    def get_pending_conflicts(self) -> List[SyncConflict]:
        """Get list of unresolved conflicts."""
        return list(self._pending_conflicts)

    def _is_configured(self) -> bool:
        """Check if sync manager is properly configured."""
        return (
            self.local_cif_path is not None
            and self.local_xtal_path is not None
            and self.server_manager is not None
            and self.server_manager.is_enabled
            and (
                self.server_manager.cif_network_path is not None
                or self.server_manager.xtal_network_path is not None
            )
        )

    def _set_status(self, status: SyncStatus):
        """Update status and emit signal."""
        self.status = status
        self.status_changed.emit(status.value)

    @staticmethod
    def _list_files(directory: Path, extensions: set) -> Dict[str, Path]:
        """
        List files in a directory matching given extensions.

        Args:
            directory: Directory to scan
            extensions: Set of lowercase extensions (e.g., {".cif"})

        Returns:
            Dict mapping filename to full path
        """
        files = {}
        try:
            for f in directory.iterdir():
                if f.is_file() and f.suffix.lower() in extensions:
                    files[f.name] = f
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to list directory {directory}: {e}")
        return files

    @staticmethod
    def _copy_file(src: Path, dst: Path):
        """
        Copy a file from src to dst.

        Creates parent directories if needed.

        Args:
            src: Source file path
            dst: Destination file path
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, dst)
