"""
Batch Processing Module for EBSD Analysis.

Processes folder of EBSD files and exports all results to single Documentation.xlsx.
Each dataset occupies one column (B, C, D, ...).

Workflow:
1. Scan folder for supported files (*.h5, *.h5oina, *.ang, *.ctf)
2. For each file:
   - Load CrystalMap (analysis.xmap_loader: Aztec .h5oina, Orienta result .h5, .ang, .ctf)
   - Run full analysis pipeline (grain, deformation, texture, RX)
   - Export to Excel (one column per dataset)
3. Save consolidated Documentation.xlsx

Reference: matlab_testskripts/EBSDanalysis_frame.m
Architecture: the MTEX-equivalence design notes §3.4
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import logging
import tempfile

import numpy as np

try:
    import orix.io  # noqa: F401
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

from analysis.ang_hexgrid import AngGridRefused, HexNeighbourhoodUnsupported
from analysis.ebsd_dataset import EBSDDataset, NO_BAND_CONTRAST
from analysis.grain_analysis import (
    ECD_BASES, EDGE_RULES, MOST_MEASURED, analyze_grain_size, grain_size_basic_info,
)
from analysis.bc_analysis import calculate_bc_histogram, fit_bc_gmm
from analysis.deformation_analysis import DEFAULT_DEFORMATION_PARAMS, analyze_deformation, check_deformation_params
from analysis.texture_analysis import analyze_texture
from analysis.rx_analysis import DEFAULT_RX_CRITERIA, analyze_recrystallization, check_rx_criteria
from analysis.excel_exporter import (ExcelExporter, free_workbook_path, release_workbook_reservation,
                                     incomplete_workbook_note)
from analysis.texture_components import TEXTURE_PRESETS, get_texture_preset
from analysis.xmap_loader import load_xmap_file


logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """What a batch run did with each file of the folder.

    Attributes:
        processed: paths of the files that went into the workbook
        failed: ``{"file", "reason"}`` of each file that raised
        skipped: ``{"file", "reason"}`` of each file that was read but gave no result (no grains)
        warnings: ``{"file", "reason"}`` of each analysis step left out of a processed file's column; the
            reason starts with the step, e.g. ``"rx: needs native band contrast in the file"``
        n_files: number of files found in the folder
        workbook_path: the workbook that was written (``output_path``, or ``<name>_2<ext>`` ... when that
            file already existed); None when none was written
        save_error: why the workbook could not be written (also when it could not be started); None otherwise.
            The files whose results were lost with it are in ``failed``, not in ``processed``.
        edge_safeguard_files: names (not paths) of the processed files where the edge rule would have left out so
            many grains that it kept all of them (Sheet1 records it as ``edge_safeguard_applied``)
    """

    processed: List[str] = field(default_factory=list)
    failed: List[Dict[str, str]] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    n_files: int = 0
    warnings: List[Dict[str, str]] = field(default_factory=list)
    workbook_path: Optional[str] = None
    save_error: Optional[str] = None
    edge_safeguard_files: List[str] = field(default_factory=list)

    @property
    def n_processed(self) -> int:
        return len(self.processed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_files": self.n_files, "n_processed": self.n_processed, "processed": list(self.processed),
            "failed": [dict(item) for item in self.failed], "skipped": [dict(item) for item in self.skipped],
            "warnings": [dict(item) for item in self.warnings],
            "workbook_path": self.workbook_path, "save_error": self.save_error,
            "edge_safeguard_files": list(self.edge_safeguard_files),
        }


def _check_writable(path: Path) -> None:
    """Raise OSError when a workbook cannot be written at ``path`` (no such directory that can be made, a
    directory without write permission, a file that is open elsewhere)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, "ab"):                  # fails for a file that Excel holds open
            pass
    with tempfile.TemporaryFile(dir=str(path.parent)):
        pass


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
        rx_gos_threshold, rx_gbc_fraction, rx_gkam_threshold, rx_min_radius_um: the RX criteria of every file
            (see analysis.rx_analysis.DEFAULT_RX_CRITERIA); the workbook records them in Sheet1 rows 60-63
        rx_phase: the phase of the RX analysis of every file (Sheet1 row 70)
        kam_order, kam_threshold_deg, hagb_min_deg: the deformation parameters of every file (the KAM sheet and
            Sheet1 row 71 record them)
    """

    def __init__(
        self,
        folder_path: str,
        output_path: str,
        texture_preset: str = "FCC_Rolling",
        file_patterns: Optional[List[str]] = None,
        gb_threshold_deg: float = 4.0,
        min_grain_pixels: int = 3,
        min_intercept_um: Optional[float] = None,
        tolerance_deg: float = 10.0,
        grain_alpha: float = 3.1,
        grain_fill: bool = True,
        grain_edge_rule: str = "matlab_script",
        grain_ecd_basis: str = "cell_area",
        grain_stats_phase: Optional[str] = None,
        rx_gos_threshold: float = DEFAULT_RX_CRITERIA["rx_gos_threshold"],
        rx_gbc_fraction: float = DEFAULT_RX_CRITERIA["rx_gbc_fraction"],
        rx_gkam_threshold: float = DEFAULT_RX_CRITERIA["rx_gkam_threshold"],
        rx_min_radius_um: float = DEFAULT_RX_CRITERIA["rx_min_radius_um"],
        texture_phase="most_measured",
        rx_phase="most_measured",
        kam_order: int = DEFAULT_DEFORMATION_PARAMS["kam_order"],
        kam_threshold_deg: float = DEFAULT_DEFORMATION_PARAMS["kam_threshold_deg"],
        hagb_min_deg: float = DEFAULT_DEFORMATION_PARAMS["hagb_min_deg"],
    ):
        """Initialize batch processor.

        Args:
            folder_path: Path to folder containing EBSD files
            output_path: Path to output Documentation.xlsx
            texture_preset: Name of texture preset (default: "FCC_Rolling")
            file_patterns: List of file extensions to process (default: [".h5", ".h5oina", ".ang", ".ctf"])
            gb_threshold_deg: Grain boundary threshold in degrees (default: 4.0)
            min_grain_pixels: Minimum pixels per grain (default: 3)
            min_intercept_um: Minimum grain intercept length in µm, only recorded in Sheet 1 (the reconstruction does
                not use it). Default None = not given: the workbook cell stays empty, never a value nobody gave
            tolerance_deg: Tolerance angle for texture components in degrees (default: 10.0)
            grain_alpha: Hole-closing radius factor of the grain reconstruction (default: 3.1; 0 closes nothing)
            grain_fill: Build the display-only fill of the closed pixels in the reconstruction (default: True)
            grain_edge_rule: Edge-grain rule of the grain-size statistics: "matlab_script" (default),
                "mtex", "outer_row" or "keep_all" (see analysis.grain_analysis.find_edge_grains)
            grain_ecd_basis: ECD basis: "cell_area" (default) or "measured_pixels"
            grain_stats_phase: None (default) = the grain-size statistics of all phases, or "most_measured" =
                of the phase with the most measured pixels of each file (the script's matrix phase)
            rx_gos_threshold: RX criterion, maximum GOS of an RX grain in degrees (default 1.25, range 0 to 20)
            rx_gbc_fraction: RX criterion, minimum gBC as a fraction of the largest gBC (default 0.7, range 0 to 1)
            rx_gkam_threshold: RX criterion, maximum gKAM in degrees (default 0.55, range 0 to 10)
            rx_min_radius_um: RX criterion, minimum equivalent grain radius in um (default 0.4, range 0 to 100)
            texture_phase: the phase the texture is computed on: "most_measured" (default) = the phase with the most
                indexed pixels of each file, resolved per file, or a phase id (the same id in every file). The
                phase must be cubic; the workbook names it per column (Sheet 1, row 67)
            rx_phase: the phase the RX analysis is made on: "most_measured" (default) = the phase with the most
                indexed pixels of each file, resolved per file, or a phase id (the same id in every file); the
                workbook names it per column (Sheet 1, row 70)
            kam_order: KAM neighbour order, 1 to 5 (default 2)
            kam_threshold_deg: KAM threshold in degrees, more than 0 and at most 30 (default 7)
            hagb_min_deg: angle from which a boundary is HAGB, more than 3 and at most 90 degrees (default 15);
                SAGB runs from 3 degrees up to it

        Raises:
            ValueError: unknown grain_edge_rule, grain_ecd_basis or grain_stats_phase, or an RX criterion or a
                deformation parameter that is not a number inside its range, or a texture_phase or rx_phase that is
                neither "most_measured" nor an integer id
        """
        if not ORIX_AVAILABLE:
            raise ImportError("orix is required for batch processing. Install: pip install orix")

        if grain_edge_rule not in EDGE_RULES:
            raise ValueError(f"grain_edge_rule must be one of {EDGE_RULES}, got {grain_edge_rule!r}")
        if grain_ecd_basis not in ECD_BASES:
            raise ValueError(f"grain_ecd_basis must be one of {ECD_BASES}, got {grain_ecd_basis!r}")
        if grain_stats_phase not in (None, MOST_MEASURED):
            raise ValueError(f"grain_stats_phase must be None or {MOST_MEASURED!r}, got {grain_stats_phase!r}")
        if texture_preset not in TEXTURE_PRESETS:
            raise ValueError(f"texture_preset must be one of {sorted(TEXTURE_PRESETS)}, got {texture_preset!r}")
        if isinstance(tolerance_deg, bool) or not isinstance(tolerance_deg, (int, float)) or not 0 < tolerance_deg <= 30:
            raise ValueError(f"tolerance_deg must be more than 0 and at most 30 degrees, got {tolerance_deg!r}")
        is_default = isinstance(texture_phase, str) and texture_phase == "most_measured"
        is_id = isinstance(texture_phase, (int, np.integer)) and not isinstance(texture_phase, (bool, np.bool_))
        if not (is_default or is_id):
            raise ValueError(f"texture_phase must be 'most_measured' or a phase id (an integer), got {texture_phase!r}")
        rx_is_default = isinstance(rx_phase, str) and rx_phase == "most_measured"
        rx_is_id = isinstance(rx_phase, (int, np.integer)) and not isinstance(rx_phase, (bool, np.bool_))
        if not (rx_is_default or rx_is_id):
            raise ValueError(f"rx_phase must be 'most_measured' or a phase id (an integer), got {rx_phase!r}")
        check_rx_criteria(rx_gos_threshold=rx_gos_threshold, rx_gbc_fraction=rx_gbc_fraction,
                          rx_gkam_threshold=rx_gkam_threshold, rx_min_radius_um=rx_min_radius_um)
        check_deformation_params(kam_order=kam_order, kam_threshold_deg=kam_threshold_deg,
                                 hagb_min_deg=hagb_min_deg)

        self.folder_path = Path(folder_path)
        self.output_path = Path(output_path)
        self.texture_preset = texture_preset
        self.file_patterns = file_patterns or [".h5", ".h5oina", ".ang", ".ctf"]
        self.gb_threshold_deg = gb_threshold_deg
        self.min_grain_pixels = min_grain_pixels
        self.min_intercept_um = min_intercept_um
        self.tolerance_deg = tolerance_deg
        self.grain_alpha = grain_alpha
        self.grain_fill = grain_fill
        self.grain_edge_rule = grain_edge_rule
        self.grain_ecd_basis = grain_ecd_basis
        self.grain_stats_phase = grain_stats_phase
        self.rx_gos_threshold = float(rx_gos_threshold)
        self.rx_gbc_fraction = float(rx_gbc_fraction)
        self.rx_gkam_threshold = float(rx_gkam_threshold)
        self.rx_min_radius_um = float(rx_min_radius_um)
        self.texture_phase = texture_phase if isinstance(texture_phase, str) else int(texture_phase)
        self.rx_phase = rx_phase if isinstance(rx_phase, str) else int(rx_phase)
        self.kam_order = int(kam_order)
        self.kam_threshold_deg = float(kam_threshold_deg)
        self.hagb_min_deg = float(hagb_min_deg)

        self.exporter: Optional[ExcelExporter] = None
        self.processed_count = 0
        self.failed_files: List[str] = []
        self.n_files = 0
        self.processed_files: List[str] = []
        # why each file in failed_files failed, and the files that were read but gave no grains:
        # {"file", "reason"} each, reported with the failed ones
        self.failed_details: List[Dict[str, str]] = []
        self.skipped_files: List[Dict[str, str]] = []
        # steps left out of a processed file (no native band contrast, ...): {"file", "reason"} each
        self.warnings: List[Dict[str, str]] = []
        self.workbook_path: Optional[Path] = None
        self.save_error: Optional[str] = None
        # names of the processed files where the edge rule kept all grains (see BatchResult)
        self.edge_safeguard_files: List[str] = []
        #: file -> stable code, for the files refused by name (see analysis/ang_hexgrid.py)
        self.failed_codes: Dict[str, str] = {}

    def find_files(self) -> List[Path]:
        """Find all EBSD files in folder matching file patterns.

        Returns:
            List of file paths

        Reference: EBSDanalysis_frame.m lines 34-36 (dir command)
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

        Reference: EBSDanalysis_frame.m main loop (lines 99-404+)
        """
        dataset_name = file_path.stem  # Filename without extension

        try:
            # Load CrystalMap
            if progress_callback:
                progress_callback(f"Loading {dataset_name}...", self.processed_count, -1)

            # the same reader as the Analysis page's "Load CrystalMap"; the step comes with the map
            crystal_map, step_size, load_meta = load_xmap_file(str(file_path))
            if load_meta.get("hex_grid") is not None:
                # a HexGrid .ang is refused by name here (the grains and the KAM below use the four
                # neighbours of a square map), never read as a square map
                raise HexNeighbourhoodUnsupported(
                    "This .ang file has a hexagonal grid. The grains and the KAM of a folder "
                    "analysis use the four neighbours of a square map, so a hexagonal file "
                    "is not analysed here.")

            # Wrap in EBSDDataset
            dataset = EBSDDataset(crystal_map, step_size, source=file_path.suffix[1:])
            if not load_meta.get("step_from_file") and step_size == 1.0:
                # a result file written by an older Orienta carries no step: every length and area of the column
                # would be in pixel units under "um" headers, so say so (the Analysis page asks the EBSD viewer)
                self._warn(file_path, "step_size: step size not recorded in the file; 1.0 µm assumed")

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
                "grain_alpha": self.grain_alpha,
            }

            # 1. BC Analysis
            if progress_callback:
                progress_callback(f"BC analysis: {dataset_name}", self.processed_count, -1)

            # The BC histogram, the BC fit and the recrystallization block are calibrated on native band
            # contrast; without it dataset.bc is a surrogate (pattern quality, scores or a constant).
            native_bc = dataset.has_native_bc
            bc_hist = bc_gmm = None
            if native_bc:
                bc_hist = calculate_bc_histogram(dataset)
                bc_gmm = fit_bc_gmm(dataset)
                if bc_gmm is None:
                    self._warn(file_path, "bc_gmm: too few points for the fit")
            else:
                self._warn(file_path, f"bc_gmm: {NO_BAND_CONTRAST}")

            # 2. Grain Analysis
            if progress_callback:
                progress_callback(f"Grain reconstruction: {dataset_name}", self.processed_count, -1)

            from analysis.grain_analysis import reconstruct_grains
            grains = reconstruct_grains(
                dataset,
                angle_threshold_deg=self.gb_threshold_deg,
                min_pixels=self.min_grain_pixels,
                alpha=self.grain_alpha,
                fill=self.grain_fill,
            )

            basic_info["n_grains"] = grains.n_grains

            if grains.n_grains == 0:
                logger.warning(f"No grains found in {dataset_name} — skipping downstream analysis")
                self.skipped_files.append({"file": str(file_path), "reason": "no grains found"})
                return None

            # A step that fails drops out of this file's column with a warning, as it does in the single export
            # (routes.analysis._collect_export_results): the other steps of the file are still written. A file
            # whose texture phase is not cubic, for one, still gets every sheet but the texture ones.
            grain_size_result = None
            try:
                grain_size_result = analyze_grain_size(
                    dataset, grains, edge_rule=self.grain_edge_rule, ecd_basis=self.grain_ecd_basis,
                    stats_phase=self.grain_stats_phase)
                # Update basic_info with the grain size metrics (the same keys the route export fills)
                basic_info.update(grain_size_basic_info(grain_size_result))
            except Exception as e:
                self._warn(file_path, f"grain_size: {type(e).__name__}: {e}")

            # 3. Deformation Analysis
            if progress_callback:
                progress_callback(f"Deformation analysis: {dataset_name}", self.processed_count, -1)

            deformation_result = None
            try:
                deformation_result = analyze_deformation(
                    dataset,
                    grains,
                    kam_order=self.kam_order,
                    kam_threshold=self.kam_threshold_deg,
                    hagb_min=self.hagb_min_deg,       # the script's values by default: 2, 7°, HAGB from 15°
                )
            except Exception as e:
                self._warn(file_path, f"deformation: {type(e).__name__}: {e}")

            # 4. Texture Analysis
            if progress_callback:
                progress_callback(f"Texture analysis: {dataset_name}", self.processed_count, -1)

            texture_result = None
            try:
                texture_result = analyze_texture(
                    dataset,
                    texture_components,
                    tolerance_deg=self.tolerance_deg,
                    texture_phase=self.texture_phase,
                    preset_name=self.texture_preset,
                )
            except Exception as e:
                self._warn(file_path, f"texture: {type(e).__name__}: {e}")

            # 5. RX Analysis
            if progress_callback:
                progress_callback(f"RX classification: {dataset_name}", self.processed_count, -1)

            rx_result = None
            if not native_bc:
                self._warn(file_path, f"rx: {NO_BAND_CONTRAST}")
            elif deformation_result is None:
                self._warn(file_path, "rx: needs the KAM of the deformation step, which failed")
            else:
                try:
                    rx_result = analyze_recrystallization(
                        dataset,
                        grains,
                        deformation_result.kam,
                        gos_threshold_deg=self.rx_gos_threshold,
                        gbc_min_fraction=self.rx_gbc_fraction,
                        gkam_threshold_deg=self.rx_gkam_threshold,
                        min_grain_radius_um=self.rx_min_radius_um,
                        rx_phase=self.rx_phase,
                    )
                except Exception as e:
                    self._warn(file_path, f"rx: {type(e).__name__}: {e}")

            # Texture of the RX grains and of the substructure grains (sheet "SmallVSLargeGrain"): the split the
            # RX analysis made, the same components and tolerance as the texture step
            texture_by_rx = None
            split_left_out = False
            if rx_result is not None and texture_result is not None:
                from analysis.texture_analysis import split_phase_conflict
                conflict = split_phase_conflict(rx_result, texture_result)
                if conflict:
                    split_left_out = True
                    self._warn(file_path, f"SmallVSLargeGrain: {conflict}")
                else:
                    try:
                        from analysis.texture_analysis import calculate_texture_split_by_rx
                        texture_by_rx = calculate_texture_split_by_rx(
                            dataset, grains, rx_result.rx_grain_ids, texture_components, self.tolerance_deg,
                            texture_phase=self.texture_phase)
                    except Exception as e:
                        self._warn(file_path, f"texture_by_rx: {type(e).__name__}: {e}")

            # Consolidate results; a step that was left out has no key, so its cells stay unwritten
            results = {"basic_info": basic_info}
            for key, value in (("grain_size", grain_size_result), ("deformation", deformation_result),
                               ("texture", texture_result)):
                if value is not None:
                    results[key] = value
            if native_bc:
                results["bc_histogram"] = bc_hist
                results["bc_gmm"] = bc_gmm
                if rx_result is not None:
                    results["rx"] = rx_result
                if texture_by_rx is not None:
                    results["texture_by_rx"] = texture_by_rx
                if split_left_out:
                    results["texture_by_rx_left_out"] = True

            # Sheet "Citations": the grain method with this file's reconstruction and statistics settings
            try:
                results["citations"] = self._grain_citations(grain_size_result)
            except Exception as e:
                self._warn(file_path, f"citations: {type(e).__name__}: {e}")

            logger.info(f"Successfully processed {dataset_name}")
            return results

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.failed_files.append(str(file_path))
            self.failed_details.append({"file": str(file_path), "reason": f"{type(e).__name__}: {e}"})
            if isinstance(e, AngGridRefused):
                self.failed_codes[str(file_path)] = e.code
            return None

    def _grain_citations(self, grain_size_result=None) -> Dict[str, Any]:
        """What to cite for a file's grains: the answer of ``GET /api/citations/analysis`` for the batch's
        reconstruction and statistics settings (``grain_size_result`` adds the file's edge-grain counts)."""
        from backend.api.services.citations.bundle import citation_bundle
        from backend.api.services.citations.grains import grain_reconstruction_step
        recon = {"gb_threshold_deg": self.gb_threshold_deg, "min_pixels": self.min_grain_pixels,
                 "alpha": self.grain_alpha, "fill": self.grain_fill}
        step = grain_reconstruction_step(recon, edge_rule=self.grain_edge_rule, ecd_basis=self.grain_ecd_basis,
                                         stats_phase=self.grain_stats_phase, grain_size=grain_size_result)
        return citation_bundle([step])

    def _warn(self, file_path: Path, reason: str) -> None:
        """Note an analysis step that was left out of a processed file."""
        logger.warning("%s: %s", Path(file_path).name, reason)
        self.warnings.append({"file": str(file_path), "reason": reason})

    def process_all(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> int:
        """Process all EBSD files in folder and export to Excel.

        Args:
            progress_callback: Optional callback(message, current, total)

        Returns:
            Number of files in the workbook. When it is 0 (no files found, every file failed or was skipped,
            or the workbook could not be started or saved) no workbook is written.
            An existing file at ``output_path`` is never opened or overwritten: the run writes to the first
            free ``<name>_2<ext>``, ``<name>_3<ext>``, ... (see ``workbook_path`` after the run).

        Reference: EBSDanalysis_frame.m main loop
        """
        # a second run on the same processor starts from nothing
        self.processed_count = 0
        self.processed_files = []
        self.failed_files = []
        self.failed_details = []
        self.skipped_files = []
        self.warnings = []
        self.edge_safeguard_files = []
        self.workbook_path = None
        self.save_error = None

        files = self.find_files()
        self.n_files = len(files)

        if not files:
            logger.warning("No EBSD files found in folder")
            return 0

        # Never write into a workbook of an earlier run: cells this run does not write would keep the old
        # numbers under the new headers. A second run goes to <name>_2.xlsx, and so on.
        # (free_workbook_path reserves the name with an empty file, so a concurrent run takes the next one;
        # a run that writes no workbook releases it again)
        target = None
        try:
            target = free_workbook_path(self.output_path)
            if target != self.output_path:
                logger.warning("%s already exists: this run writes %s instead", self.output_path, target)
            _check_writable(target)
            self.exporter = ExcelExporter(str(target))
        except OSError as e:
            release_workbook_reservation(target)
            self.save_error = (f"The workbook {(target or self.output_path).name} cannot be written "
                               f"({type(e).__name__}: {e}); nothing was processed.")
            logger.error(self.save_error)
            return 0
        except Exception:
            release_workbook_reservation(target)
            raise

        total_files = len(files)
        stem_count = Counter(f.stem for f in files)

        try:
            for i, file_path in enumerate(files, start=1):
                if progress_callback:
                    progress_callback(
                        f"Processing file {i}/{total_files}: {file_path.name}",
                        i,
                        total_files,
                    )

                warnings_before = len(self.warnings)
                results = self.process_file(file_path, progress_callback)

                if results is None:
                    del self.warnings[warnings_before:]       # a file without a result has no column to warn about
                else:
                    # scan.ang and scan.ctf must not get the same column header
                    dataset_name = file_path.name if stem_count[file_path.stem] > 1 else file_path.stem
                    # Export to Excel; one file whose results cannot be written does not end the batch
                    try:
                        self.exporter.export_dataset(dataset_name, results)
                    except Exception as e:
                        logger.error(f"Failed to export {file_path}: {e}")
                        self.failed_files.append(str(file_path))
                        self.failed_details.append(
                            {"file": str(file_path), "reason": f"export: {type(e).__name__}: {e}"})
                        # drop the step warnings of a file that is not in the workbook, and mark its column
                        del self.warnings[warnings_before:]
                        self.exporter.mark_dataset_failed(dataset_name)
                        continue
                    self.processed_count += 1
                    self.processed_files.append(str(file_path))
                    if results["basic_info"].get("edge_safeguard_applied"):
                        self.edge_safeguard_files.append(file_path.name)

            # Save Excel file; a workbook without a single dataset would read like a result
            if self.processed_count > 0:
                if progress_callback:
                    progress_callback("Saving Excel file...", total_files, total_files)
                try:
                    self.exporter.save()
                    self.workbook_path = target
                except Exception as e:
                    # the results are in memory only: they are not in a workbook, so they are not "processed"
                    self.save_error = (f"The workbook {target.name} could not be written "
                                       f"({type(e).__name__}: {e}); nothing of this run was saved.")
                    self.save_error += incomplete_workbook_note(target)   # a half-written file must not pass as a workbook
                    logger.error(self.save_error)
                    for lost in self.processed_files:
                        self.failed_files.append(lost)
                        self.failed_details.append(
                            {"file": lost, "reason": "processed, but the workbook could not be written"})
                    self.processed_files = []
                    self.processed_count = 0
                    self.warnings = []
                    self.edge_safeguard_files = []
            else:
                logger.warning("No file gave a result: the workbook %s is not written", target)
        finally:
            self.exporter.close()
            release_workbook_reservation(target)           # the name goes back; an empty file goes with it

        logger.info(
            f"Batch processing complete: {self.processed_count}/{total_files} files processed"
        )

        if self.failed_files:
            logger.warning(f"Failed files ({len(self.failed_files)}): {self.failed_files}")
        if self.skipped_files:
            logger.warning(f"Skipped files without a result ({len(self.skipped_files)}): {self.skipped_files}")

        return self.processed_count

    def result(self) -> BatchResult:
        """What the last :meth:`process_all` did with each file."""
        return BatchResult(
            processed=list(self.processed_files),
            failed=[dict(item) for item in self.failed_details],
            skipped=[dict(item) for item in self.skipped_files],
            n_files=self.n_files,
            warnings=[dict(item) for item in self.warnings],
            workbook_path=str(self.workbook_path) if self.workbook_path is not None else None,
            save_error=self.save_error,
            edge_safeguard_files=list(self.edge_safeguard_files),
        )


def batch_process_folder(
    folder_path: str,
    output_path: str,
    texture_preset: str = "FCC_Rolling",
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    **kwargs,
) -> BatchResult:
    """Convenience function for batch processing.

    Args:
        folder_path: Path to folder containing EBSD files
        output_path: Path to output Documentation.xlsx
        texture_preset: Name of texture preset (default: "FCC_Rolling")
        progress_callback: Optional callback(message, current, total)
        **kwargs: Additional arguments passed to BatchProcessor

    Returns:
        BatchResult: the processed, failed (with reason) and skipped (with reason) files
    """
    processor = BatchProcessor(folder_path, output_path, texture_preset, **kwargs)
    processor.process_all(progress_callback)
    return processor.result()
