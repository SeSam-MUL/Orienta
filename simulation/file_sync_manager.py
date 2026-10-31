"""
File Sync Manager - Material extraction and upload coordination

Manages:
- Material extraction from .xtal files
- Upload coordination with ServerModeManager
- Upload history tracking
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from collections import Counter
from datetime import datetime
from dataclasses import dataclass

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


# Periodic table mapping (Z -> Symbol): the shared table (element_symbols.py,
# Z = 1 ... 118). This module's own copy stopped at Cf (98), so a heavier Z was
# reported as "Z<n>" in the material folder name.
from element_symbols import SYMBOL_OF as PERIODIC_TABLE  # noqa: E402

# Non-metallic elements (use for classification rules)
NON_METALS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


@dataclass
class UploadRecord:
    """Record of a file upload."""
    filename: str
    material: str
    status: str  # "queued", "uploading", "complete", "failed"
    timestamp: datetime
    file_size_mb: float
    error_message: Optional[str] = None


class FileSyncManager(QObject):
    """
    Manages file synchronization for simulation results.

    Responsibilities:
    - Extract material from .xtal metadata
    - Coordinate uploads with ServerModeManager
    - Track upload history

    Signals:
        upload_queued(str, str): Emitted when upload is queued (filename, material)
    """

    # Signals
    upload_queued = pyqtSignal(str, str)  # filename, material

    def __init__(self, server_manager, parent=None):
        """
        Initialize FileSyncManager.

        Args:
            server_manager: ServerModeManager instance
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.server_manager = server_manager
        self.upload_history = []  # List of UploadRecord objects

        if not HAS_H5PY:
            logger.warning("h5py not available - material extraction will be limited")

    def extract_material_from_xtal(self, xtal_path: str) -> str:
        """
        Extract primary material from .xtal HDF5 file.

        Uses the same logic as emsphinx_automation.py:
        - Read Atomtypes from CrystalData
        - Count element occurrences
        - Determine dominant metallic element

        Args:
            xtal_path: Path to .xtal file

        Returns:
            Material code: "Fe", "Al", "Si", "Ti", "Cu", etc., or "Mixed"
        """
        if not HAS_H5PY:
            logger.warning("h5py not available - using fallback material 'Mixed'")
            return "Mixed"

        try:
            xtal_file = Path(xtal_path)
            if not xtal_file.exists():
                logger.error(f"XTAL file not found: {xtal_path}")
                return "Mixed"

            with h5py.File(xtal_path, 'r') as f:
                if "CrystalData" not in f:
                    logger.warning(f"No CrystalData group in {xtal_file.name}")
                    return "Mixed"

                cd = f["CrystalData"]

                if "Atomtypes" not in cd:
                    logger.warning(f"No Atomtypes dataset in {xtal_file.name}")
                    return "Mixed"

                # Read atom types (array of atomic numbers)
                atomtypes = cd["Atomtypes"][()]
                element_counts = Counter(atomtypes.tolist())

                if not element_counts:
                    logger.warning(f"Empty Atomtypes in {xtal_file.name}")
                    return "Mixed"

                # Find dominant element
                dominant_z = element_counts.most_common(1)[0][0]
                dominant_symbol = PERIODIC_TABLE.get(dominant_z, f"Z{dominant_z}")

                # Apply classification rules
                material = self._classify_material(element_counts)

                logger.info(
                    f"Extracted material from {xtal_file.name}: {material} "
                    f"(dominant: {dominant_symbol}, composition: {dict(element_counts)})"
                )

                return material

        except Exception as e:
            logger.error(f"Failed to extract material from {xtal_path}: {e}")
            return "Mixed"

    def _classify_material(self, element_counts: Counter) -> str:
        """
        Classify material based on element composition.

        Rules:
        1. If dominant element is non-metal (O, C, N) -> use second element
        2. Common materials: Fe, Al, Si, Ti, Cu, Ni, etc.
        3. Complex alloys -> "Mixed"

        Args:
            element_counts: Counter of atomic numbers

        Returns:
            Material classification string
        """
        if not element_counts:
            return "Mixed"

        # Get top 2 elements
        top_elements = element_counts.most_common(2)
        dominant_z = top_elements[0][0]
        dominant_symbol = PERIODIC_TABLE.get(dominant_z, f"Z{dominant_z}")

        # If dominant is non-metal, try second element
        if dominant_symbol in NON_METALS and len(top_elements) > 1:
            second_z = top_elements[1][0]
            second_symbol = PERIODIC_TABLE.get(second_z, f"Z{second_z}")

            # If second is also non-metal, return Mixed
            if second_symbol in NON_METALS:
                return "Mixed"
            else:
                return second_symbol

        # Check if dominant is metallic
        if dominant_symbol not in NON_METALS:
            return dominant_symbol

        # All elements are non-metallic -> Mixed
        return "Mixed"

    def queue_simulation_results(
        self,
        result_files: Dict[str, str],
        xtal_path: str
    ):
        """
        Queue H5 and SHT files for upload with material-based organization.

        Args:
            result_files: Dictionary with "h5" and/or "sht" file paths
            xtal_path: Original .xtal file for material extraction
        """
        # Extract material from .xtal
        material = self.extract_material_from_xtal(xtal_path)

        logger.info(f"Queueing uploads for material: {material}")

        # Queue H5 upload (Monte Carlo data)
        if "h5" in result_files:
            h5_path = Path(result_files["h5"])
            if h5_path.exists():
                self._queue_single_file(h5_path, material, "h5")
            else:
                logger.warning(f"H5 file not found: {h5_path}")

        # Queue SHT upload
        if "sht" in result_files:
            sht_path = Path(result_files["sht"])
            if sht_path.exists():
                self._queue_single_file(sht_path, material, "sht")
            else:
                logger.warning(f"SHT file not found: {sht_path}")

        # Queue master pattern upload
        if "master" in result_files:
            master_path = Path(result_files["master"])
            if master_path.exists():
                self._queue_single_file(master_path, material, "master")
            else:
                logger.warning(f"Master file not found: {master_path}")

    def _queue_single_file(self, file_path: Path, material: str, file_type: str):
        """
        Queue a single file for upload.

        Args:
            file_path: Path to file
            material: Material classification
            file_type: "h5", "sht", or "master"
        """
        try:
            # Get central network path
            if file_type == "h5":
                central_path = self.server_manager.get_central_h5_path(
                    file_path.name,
                    material_system=material
                )
            elif file_type == "master":
                central_path = self.server_manager.get_central_h5_path(
                    file_path.name,
                    material_system=material
                )
            elif file_type == "dictionary":
                central_path = self.server_manager.get_central_dictionary_path(
                    file_path.name,
                    material_system=material
                )
            else:  # sht
                central_path = self.server_manager.get_central_sht_path(
                    file_path.name,
                    material_system=material
                )

            if not central_path:
                logger.warning(f"Could not get central path for {file_path.name}")
                return

            # Queue upload
            self.server_manager.queue_upload(str(file_path), str(central_path))

            # Record upload
            file_size_mb = file_path.stat().st_size / (1024**2)
            record = UploadRecord(
                filename=file_path.name,
                material=material,
                status="queued",
                timestamp=datetime.now(),
                file_size_mb=file_size_mb
            )
            self.upload_history.append(record)

            # Emit signal
            self.upload_queued.emit(file_path.name, material)

            logger.info(
                f"Queued {file_type.upper()} upload: {file_path.name} → "
                f"{material}/ ({file_size_mb:.1f} MB)"
            )

        except Exception as e:
            logger.error(f"Failed to queue upload for {file_path.name}: {e}")

            # Record failure
            record = UploadRecord(
                filename=file_path.name,
                material=material,
                status="failed",
                timestamp=datetime.now(),
                file_size_mb=0,
                error_message=str(e)
            )
            self.upload_history.append(record)

    def get_upload_history(self, limit: int = 50) -> list:
        """
        Get recent upload history.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of UploadRecord objects (most recent first)
        """
        return self.upload_history[-limit:][::-1]

    def clear_history(self):
        """Clear upload history."""
        self.upload_history.clear()
        logger.info("Upload history cleared")
