"""
Database Browser Model - Browse simulation results across local and server

Scans local cache and server directories for SHT/H5/master files,
provides a unified view with location indicators, and supports
on-demand download with LRU cache management.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from PyQt5.QtCore import QObject, pyqtSignal

from simulation.lru_cache import LRUCacheManager
from path_utils import get_local_database_path, DATABASE_SUBFOLDERS

logger = logging.getLogger(__name__)


class FileLocation(Enum):
    """Where a file exists."""
    LOCAL_ONLY = "local_only"       # Only in local cache
    SERVER_ONLY = "server_only"     # Only on server (not cached)
    BOTH = "both"                   # Cached locally AND on server
    DOWNLOADING = "downloading"     # Currently being downloaded


@dataclass
class BrowserEntry:
    """A file entry in the database browser."""
    filename: str
    material: str                   # Material subfolder (Fe, Al, Si, etc.)
    file_type: str                  # "sht", "h5", "master"
    size_bytes: int
    location: FileLocation
    local_path: Optional[Path] = None
    server_path: Optional[Path] = None


class DatabaseBrowserModel(QObject):
    """
    Model for browsing simulation result files.

    Scans server and local cache directories, builds a unified file list
    with location indicators, and provides on-demand download.

    Signals:
        entries_updated(): Emitted when the file list changes
        download_started(str): Emitted when download begins (filename)
        download_finished(str, bool): Emitted when download ends (filename, success)
    """

    entries_updated = pyqtSignal()
    download_started = pyqtSignal(str)
    download_finished = pyqtSignal(str, bool)

    def __init__(self, server_manager, cache_manager: LRUCacheManager, parent=None):
        """
        Initialize database browser model.

        Args:
            server_manager: ServerModeManager instance
            cache_manager: LRUCacheManager instance
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.server_manager = server_manager
        self.cache_manager = cache_manager
        self._entries: List[BrowserEntry] = []
        self._offline_mode = False

    @property
    def offline_mode(self) -> bool:
        return self._offline_mode

    @offline_mode.setter
    def offline_mode(self, value: bool):
        self._offline_mode = value
        logger.info(f"Offline mode: {'ON' if value else 'OFF'}")

    def refresh(self):
        """Scan directories and rebuild file list."""
        self._entries.clear()

        # Scan server directories (if online)
        server_files: Dict[str, BrowserEntry] = {}
        if not self._offline_mode:
            server_files = self._scan_server()

        # Scan local Database/ folder (simulation outputs + downloads)
        local_files = self._scan_local_database()

        # Merge: start with server files
        merged: Dict[str, BrowserEntry] = {}

        for key, entry in server_files.items():
            if key in local_files:
                entry.location = FileLocation.BOTH
                entry.local_path = local_files[key].local_path
            merged[key] = entry

        # Add local-only files
        for key, entry in local_files.items():
            if key not in merged:
                entry.location = FileLocation.LOCAL_ONLY
                merged[key] = entry

        self._entries = sorted(merged.values(), key=lambda e: (e.material, e.filename))
        self.entries_updated.emit()

        logger.info(
            f"Browser refreshed: {len(self._entries)} files "
            f"({sum(1 for e in self._entries if e.location == FileLocation.SERVER_ONLY)} server-only, "
            f"{sum(1 for e in self._entries if e.location == FileLocation.LOCAL_ONLY)} local-only, "
            f"{sum(1 for e in self._entries if e.location == FileLocation.BOTH)} both)"
        )

    def get_entries(self, file_type_filter: Optional[str] = None,
                    material_filter: Optional[str] = None,
                    search_text: Optional[str] = None) -> List[BrowserEntry]:
        """
        Get filtered file entries.

        Args:
            file_type_filter: Filter by file type ("sht", "h5", "master", or None for all)
            material_filter: Filter by material ("Fe", "Al", etc., or None for all)
            search_text: Text to search in filename (case-insensitive)

        Returns:
            Filtered list of BrowserEntry objects
        """
        result = self._entries

        if file_type_filter:
            result = [e for e in result if e.file_type == file_type_filter]

        if material_filter:
            result = [e for e in result if e.material == material_filter]

        if search_text:
            search_lower = search_text.lower()
            result = [e for e in result if search_lower in e.filename.lower()]

        return result

    def get_materials(self) -> List[str]:
        """Get list of unique material categories."""
        return sorted(set(e.material for e in self._entries))

    def download_file(self, entry: BrowserEntry) -> bool:
        """
        Download a server file to local cache.

        Args:
            entry: BrowserEntry to download

        Returns:
            True if download succeeded
        """
        if entry.server_path is None:
            logger.warning(f"No server path for {entry.filename}")
            return False

        if not entry.server_path.exists():
            logger.warning(f"Server file not found: {entry.server_path}")
            return False

        try:
            self.download_started.emit(entry.filename)

            # Route to the correct Database/ subfolder based on file type
            _subfolder_map = {
                "sht":    DATABASE_SUBFOLDERS["sht_database"],
                "h5":     DATABASE_SUBFOLDERS["h5_cache"],
                "master": DATABASE_SUBFOLDERS["h5_cache"],
                "cif":    DATABASE_SUBFOLDERS["cif_library"],
                "xtal":   DATABASE_SUBFOLDERS["xtal_library"],
            }
            subfolder = _subfolder_map.get(entry.file_type, "Downloads")
            db_root = get_local_database_path()
            material_part = entry.material if entry.material and entry.material != "-" else ""
            if material_part:
                dest_path = db_root / subfolder / material_part / entry.filename
            else:
                dest_path = db_root / subfolder / entry.filename

            # Copy from server to Database/
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry.server_path, dest_path)

            # Update entry
            entry.location = FileLocation.BOTH
            entry.local_path = dest_path

            self.download_finished.emit(entry.filename, True)
            self.entries_updated.emit()

            logger.info(f"Downloaded: {entry.filename} to cache")
            return True

        except Exception as e:
            logger.error(f"Download failed for {entry.filename}: {e}")
            self.download_finished.emit(entry.filename, False)
            return False

    def upload_file(self, entry: BrowserEntry) -> bool:
        """
        Upload a local-only file to the server.

        Routes the file to the correct server subfolder based on file_type
        and material.  Creates destination directories as needed.

        Args:
            entry: BrowserEntry with location == LOCAL_ONLY

        Returns:
            True if upload succeeded
        """
        if entry.local_path is None or not entry.local_path.exists():
            logger.warning(f"Local file not found for upload: {entry.filename}")
            return False

        # Determine server destination based on file type
        _server_path_map = {
            "sht": self.server_manager.sht_database_path,
            "h5": self.server_manager.h5_cache_path,
            "master": self.server_manager.h5_cache_path,
            "cif": self.server_manager.cif_network_path,
            "xtal": self.server_manager.xtal_network_path,
        }
        server_base = _server_path_map.get(entry.file_type)
        if not server_base:
            logger.warning(f"No server path configured for type {entry.file_type}")
            return False

        try:
            # Build destination: material subfolder for SHT/H5/master, flat for CIF/XTAL
            material_part = entry.material if entry.material and entry.material != "-" else ""
            if material_part:
                dest_path = server_base / material_part / entry.filename
            else:
                dest_path = server_base / entry.filename

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry.local_path, dest_path)

            # Update entry
            entry.location = FileLocation.BOTH
            entry.server_path = dest_path

            self.entries_updated.emit()
            logger.info(f"Uploaded: {entry.filename} to server")
            return True

        except Exception as e:
            logger.error(f"Upload failed for {entry.filename}: {e}")
            return False

    def _scan_server(self) -> Dict[str, BrowserEntry]:
        """Scan server directories for SHT/H5/CIF/XTAL files."""
        files = {}

        # Scan SHT database
        if self.server_manager.sht_database_path:
            self._scan_server_dir(
                self.server_manager.sht_database_path,
                file_type="sht",
                extensions={".sht"},
                files=files
            )

        # Scan H5 cache
        if self.server_manager.h5_cache_path:
            self._scan_server_dir(
                self.server_manager.h5_cache_path,
                file_type="h5",
                extensions={".h5"},
                files=files
            )

        # Scan CIF library
        if self.server_manager.cif_network_path:
            self._scan_flat_dir(
                self.server_manager.cif_network_path,
                file_type="cif",
                extensions={".cif"},
                files=files
            )

        # Scan XTAL library
        if self.server_manager.xtal_network_path:
            self._scan_flat_dir(
                self.server_manager.xtal_network_path,
                file_type="xtal",
                extensions={".xtal"},
                files=files
            )

        return files

    def _scan_server_dir(self, base_dir: Path, file_type: str,
                         extensions: set, files: Dict[str, BrowserEntry]):
        """Scan a server base directory with material subfolders."""
        if not base_dir.is_dir():
            return

        try:
            for material_dir in base_dir.iterdir():
                if not material_dir.is_dir():
                    continue

                material = material_dir.name

                for f in material_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in extensions:
                        # Detect master files by naming convention
                        actual_type = file_type
                        if file_type == "h5" and "_master" in f.name.lower():
                            actual_type = "master"

                        key = f"{material}/{f.name}"
                        files[key] = BrowserEntry(
                            filename=f.name,
                            material=material,
                            file_type=actual_type,
                            size_bytes=f.stat().st_size,
                            location=FileLocation.SERVER_ONLY,
                            server_path=f,
                        )
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to scan server dir {base_dir}: {e}")

    def _scan_flat_dir(self, base_dir: Path, file_type: str,
                       extensions: set, files: Dict[str, BrowserEntry]):
        """Scan a flat directory (no material subfolders) for CIF/XTAL files.

        Also scans one level of subdirectories for material organization.
        """
        if not base_dir.is_dir():
            return

        try:
            # Scan top-level files
            for f in base_dir.iterdir():
                if f.is_file() and f.suffix.lower() in extensions:
                    # Key must match the material/filename format used by download_file()
                    # so that server entries and cached downloads merge correctly.
                    key = f"-/{f.name}"
                    files[key] = BrowserEntry(
                        filename=f.name,
                        material="-",
                        file_type=file_type,
                        size_bytes=f.stat().st_size,
                        location=FileLocation.SERVER_ONLY,
                        server_path=f,
                    )
                elif f.is_dir():
                    # Also scan material subfolders if they exist
                    material = f.name
                    for sub_f in f.iterdir():
                        if sub_f.is_file() and sub_f.suffix.lower() in extensions:
                            key = f"{material}/{sub_f.name}"
                            files[key] = BrowserEntry(
                                filename=sub_f.name,
                                material=material,
                                file_type=file_type,
                                size_bytes=sub_f.stat().st_size,
                                location=FileLocation.SERVER_ONLY,
                                server_path=sub_f,
                            )
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to scan dir {base_dir}: {e}")

    def _scan_local_database(self) -> Dict[str, BrowserEntry]:
        """Scan Database/ subfolders directly for all local files.

        Covers simulation outputs (SHT, H5) as well as downloaded CIF/XTAL files.
        Uses filesystem scan — no LRU index needed.

        Uses self.cache_manager.cache_dir as the root so tests can inject a
        temporary directory via the cache_manager fixture.  In production this
        equals get_local_database_path() (= Kikuchipy_GUI/Database/).
        """
        db_root = self.cache_manager.cache_dir
        files: Dict[str, BrowserEntry] = {}

        # SHT and H5 live in material subfolders
        self._scan_server_dir(
            db_root / DATABASE_SUBFOLDERS["sht_database"],
            file_type="sht", extensions={".sht"}, files=files
        )
        self._scan_server_dir(
            db_root / DATABASE_SUBFOLDERS["h5_cache"],
            file_type="h5", extensions={".h5"}, files=files
        )
        # CIF and XTAL are flat (or material subfolders)
        self._scan_flat_dir(
            db_root / DATABASE_SUBFOLDERS["cif_library"],
            file_type="cif", extensions={".cif"}, files=files
        )
        self._scan_flat_dir(
            db_root / DATABASE_SUBFOLDERS["xtal_library"],
            file_type="xtal", extensions={".xtal"}, files=files
        )

        # Fix paths: _scan_server_dir/_scan_flat_dir set server_path; move to local_path
        for entry in files.values():
            entry.location = FileLocation.LOCAL_ONLY
            entry.local_path = entry.server_path
            entry.server_path = None

        return files

    def _scan_local_cache(self) -> Dict[str, BrowserEntry]:
        """Legacy: scan LRU cache index. Kept for compatibility."""
        return self._scan_local_database()
