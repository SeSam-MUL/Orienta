"""
Batch Processing Module for EBSD Analysis.

Processes folder of EBSD files and exports all results to single Documentation.xlsx.
Each dataset occupies one column (B, C, D, ...).

Workflow:
1. Scan folder for supported files (*.h5, *.h5oina, *.ang, *.ctf)
2. For each file:
   - Load CrystalMap (via kikuchipy/orix)
   - Run full analysis pipeline (grain, deformation, texture, RX)
   - Export to Excel (one column per dataset)
3. Save consolidated Documentation.xlsx

Reference: matlab_testskripts/EBSDanalysis_frame.m
Architecture: the MTEX-equivalence design notes §3.4
"""

from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import logging

import numpy as np

try:
    from orix.io import load
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import analyze_grain_size
from analysis.bc_analysis import calculate_bc_histogram, fit_bc_gmm
from analysis.deformation_analysis import analyze_deformation
from analysis.texture_analysis import analyze_texture
from analysis.rx_analysis import analyze_recrystallization
from analysis.excel_exporter import ExcelExporter
from analysis.texture_components import get_texture_preset


logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processor for EBSD analysis.

    Processes folder of EBSD files and exports to single Excel file.

    Attributes:
        folder_path: Path to folder containing EBSD files
        output_path: Path to output Documentation.xlsx
        file_patterns: List of file extensions to process
        texture_preset: Name of texture preset (e.g., "FCC_Rolling")
        gb_threshold_deg: Grain boundary threshold (degrees)
        min_grain_pixels: Minimum pixels per grain
        min_intercept_um: Minimum grain intercept length (µm)
        tolerance_deg: Tolerance angle for texture components (degrees)
    """

    def __init__(
        self,
        folder_path: str,
        output_path: str,
        texture_preset: str = "FCC_Rolling",
        file_patterns: Optional[List[str]] = None,
        gb_threshold_deg: float = 4.0,
        min_grain_pixels: int = 3,
        min_intercept_um: float = 1.5,
        tolerance_deg: float = 10.0,
    ):
        """Initialize batch processor.

        Args:
            folder_path: Path to folder containing EBSD files
            output_path: Path to output Documentation.xlsx
            texture_preset: Name of texture preset (default: "FCC_Rolling")
            file_patterns: List of file extensions to process (default: [".h5", ".h5oina", ".ang", ".ctf"])
            gb_threshold_deg: Grain boundary threshold in degrees (default: 4.0)
            min_grain_pixels: Minimum pixels per grain (default: 3)
            min_intercept_um: Minimum grain intercept length in µm (default: 1.5)
            tolerance_deg: Tolerance angle for texture components in degrees (default: 10.0)
        """
        if not ORIX_AVAILABLE:
            raise ImportError("orix is required for batch processing. Install: pip install orix")

        self.folder_path = Path(folder_path)
        self.output_path = Path(output_path)
        self.texture_preset = texture_preset
        self.file_patterns = file_patterns or [".h5", ".h5oina", ".ang", ".ctf"]
        self.gb_threshold_deg = gb_threshold_deg
        self.min_grain_pixels = min_grain_pixels
        self.min_intercept_um = min_intercept_um
        self.tolerance_deg = tolerance_deg

        self.exporter: Optional[ExcelExporter] = None
        self.processed_count = 0
        self.failed_files: List[str] = []

    def find_files(self) -> List[Path]:
        """Find all EBSD files in folder matching file patterns.

        Returns:
            List of file paths

        Reference: EBSDanalysis_frame.m lines 30-32 (dir command)
        """
        files = []
        for pattern in self.file_patterns:
            files.extend(self.folder_path.glob(f"*{pattern}"))

        # Sort alphabetically for consistent ordering
        files.sort()

        logger.info(f"Found {len(files)} EBSD files in {self.folder_path}")
        return files

    def process_file(
        self,
        file_path: Path,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process single EBSD file through full analysis pipeline.

        Args:
            file_path: Path to EBSD file
            progress_callback: Optional callback(message, current, total)

        Returns:
            Dictionary with all analysis results, or None if processing failed

        Reference: EBSDanalysis_frame.m main loop (lines 95-400+)
        """
        dataset_name = file_path.stem  # Filename without extension

        try:
            # Load CrystalMap
            if progress_callback:
                progress_callback(f"Loading {dataset_name}...", self.processed_count, -1)

            crystal_map = load(str(file_path))

            # Extract step size from CrystalMap
            # CrystalMap has dx, dy properties
            if hasattr(crystal_map, 'dx') and crystal_map.dx is not None:
                step_size = float(crystal_map.dx)
            else:
                # Fallback: calculate from coordinates
                if crystal_map.x is not None and len(crystal_map.x) > 1:
                    step_size = float(np.median(np.diff(np.unique(crystal_map.x))))
                else:
                    step_size = 1.0  # Default if unable to determine

            # Wrap in EBSDDataset
            dataset = EBSDDataset(crystal_map, step_size, source=file_path.suffix[1:])

            # Get texture components
            texture_components = get_texture_preset(self.texture_preset)
            basic_info = {
                "step_micron": step_size,
                "dim_x": dataset.shape[1],
                "dim_y": dataset.shape[0],
                "n_grains": 0,  # Will be updated after grain reconstruction
                "rotation_deg_1": 0.0,  # User-defined rotations (not auto-detected)
                "rotation_deg_2": 0.0,
                "rotation_deg_3": 0.0,
                "gb_threshold_deg": self.gb_threshold_deg,
                "min_pixels": self.min_grain_pixels,
                "min_intercept_um": self.min_intercept_um,
            }

            # 1. BC Analysis
            if progress_callback:
                progress_callback(f"BC analysis: {dataset_name}", self.processed_count, -1)

            bc_hist = calculate_bc_histogram(dataset)
            bc_gmm = fit_bc_gmm(dataset)

            # 2. Grain Analysis
            if progress_callback:
                progress_callback(f"Grain reconstruction: {dataset_name}", self.processed_count, -1)

            from analysis.grain_analysis import reconstruct_grains
            grains = reconstruct_grains(
                dataset,
                angle_threshold_deg=self.gb_threshold_deg,
                min_pixels=self.min_grain_pixels,
            )

            basic_info["n_grains"] = grains.n_grains

            if grains.n_grains == 0:
                logger.warning(f"No grains found in {dataset_name} — skipping downstream analysis")
                return None

            grain_size_result = analyze_grain_size(dataset, grains)

            # Update basic_info with grain size metrics
            basic_info["mean_MaxDimX_um"] = float(grain_size_result.avg_maxDimX)
            basic_info["mean_MaxDimY_um"] = float(grain_size_result.avg_maxDimY)
            basic_info["mean_AR"] = float(grain_size_result.avg_AR)
            basic_info["median_AR"] = (
                float(np.median(grain_size_result.ar_per_grain))
                if len(grain_size_result.ar_per_grain) > 0 else 0.0
            )
            ecd = grain_size_result.ecd_per_grain
            basic_info["grains_above_5um"] = int(np.sum(ecd > 5.0))
            basic_info["grains_above_10um"] = int(np.sum(ecd > 10.0))
            basic_info["grains_above_15um"] = int(np.sum(ecd > 15.0))

            # 3. Deformation Analysis
            if progress_callback:
                progress_callback(f"Deformation analysis: {dataset_name}", self.processed_count, -1)

            deformation_result = analyze_deformation(
                dataset,
                grains,
                # Defaults match MATLAB: KAM order=2, threshold=7°, HAGB≥15°
            )

            # 4. Texture Analysis
            if progress_callback:
                progress_callback(f"Texture analysis: {dataset_name}", self.processed_count, -1)

            texture_result = analyze_texture(
                dataset,
                texture_components,
                tolerance_deg=self.tolerance_deg,
            )

            # 5. RX Analysis
            if progress_callback:
                progress_callback(f"RX classification: {dataset_name}", self.processed_count, -1)

            rx_result = analyze_recrystallization(
                dataset,
                grains,
                deformation_result.kam,
                gos_threshold_deg=1.25,
                gbc_min_fraction=0.7,
                gkam_threshold_deg=0.55,
                min_grain_radius_um=0.4,
            )

            # Consolidate results
            results = {
                "basic_info": basic_info,
                "bc_histogram": bc_hist,
                "bc_gmm": bc_gmm,
                "grain_size": grain_size_result,
                "deformation": deformation_result,
                "texture": texture_result,
                "rx": rx_result,
            }

            logger.info(f"Successfully processed {dataset_name}")
            return results

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.failed_files.append(str(file_path))
            return None

    def process_all(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> int:
        """Process all EBSD files in folder and export to Excel.

        Args:
            progress_callback: Optional callback(message, current, total)

        Returns:
            Number of successfully processed files

        Reference: EBSDanalysis_frame.m main loop
        """
        files = self.find_files()

        if not files:
            logger.warning("No EBSD files found in folder")
            return 0

        # Initialize Excel exporter
        self.exporter = ExcelExporter(str(self.output_path))

        total_files = len(files)
        self.processed_count = 0

        for i, file_path in enumerate(files, start=1):
            if progress_callback:
                progress_callback(
                    f"Processing file {i}/{total_files}: {file_path.name}",
                    i,
                    total_files,
                )

            results = self.process_file(file_path, progress_callback)

            if results is not None:
                # Export to Excel
                dataset_name = file_path.stem
                self.exporter.export_dataset(dataset_name, results)
                self.processed_count += 1

        # Save Excel file
        if progress_callback:
            progress_callback("Saving Excel file...", total_files, total_files)

        self.exporter.save()
        self.exporter.close()

        logger.info(
            f"Batch processing complete: {self.processed_count}/{total_files} files processed"
        )

        if self.failed_files:
            logger.warning(f"Failed files ({len(self.failed_files)}): {self.failed_files}")

        return self.processed_count


def batch_process_folder(
    folder_path: str,
    output_path: str,
    texture_preset: str = "FCC_Rolling",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    **kwargs,
) -> int:
    """Convenience function for batch processing.

    Args:
        folder_path: Path to folder containing EBSD files
        output_path: Path to output Documentation.xlsx
        texture_preset: Name of texture preset (default: "FCC_Rolling")
        progress_callback: Optional callback(message, current, total)
        **kwargs: Additional arguments passed to BatchProcessor

    Returns:
        Number of successfully processed files
    """
    processor = BatchProcessor(folder_path, output_path, texture_preset, **kwargs)
    return processor.process_all(progress_callback)
