"""
Excel Export Module for EBSD Analysis Results.

Exports all analysis results to Documentation.xlsx with 14 sheets:
- Sheet 1: Basic Info (step, dimensions, #grains, rotation, thresholds)
- "BC hist": Band contrast histogram per dataset
- "BC fit (3G)": 3-Gaussian GMM fit parameters
- "BandContrast Group": BC_high/mid/low center + areaFrac
- "Grain Size hist": ECD, MaxDimX, MaxDimY distributions
- "KAM hist": KAM distribution per dataset
- "GOS hist, area weighted": GOS distribution per dataset
- "gKAM - area weighted": Grain-average KAM distribution per dataset
- "Grain Boundaries": HAGB/SAGB lengths, segment histograms, sphericity
- "SmallVSLargeGrain": Texture components for RX vs Substr vs All
- TextureIndex + Entropy (Sheet 1 cols X-Y)
- Texture components EBSD-based (Sheet 1 cols Z-AM, 14 components)
- RX metrics (Sheet 1 cols AN-AQ: RXedGrainFraction, highBC, lowGKAM, lowKAM)
- Surface plane fractions (Sheet 1 cols AT-AV: {111}/{100}/{110})

Reference: matlab_testskripts/EBSDanalysis_frame.m, RxxAnalysis.m, etc.
Architecture: the MTEX-equivalence design notes §3.3
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Font, Alignment, PatternFill
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


def excel_column(col_idx: int) -> str:
    """Convert 1-indexed column number to Excel column letter.

    Args:
        col_idx: Column index (1=A, 2=B, ..., 27=AA, etc.)

    Returns:
        Excel column letter (A, B, ..., Z, AA, AB, ...)

    Reference: EBSDanalysis_frame.m excelColumn() function
    """
    if not OPENPYXL_AVAILABLE:
        # Fallback implementation
        result = ""
        while col_idx > 0:
            col_idx -= 1
            result = chr(65 + (col_idx % 26)) + result
            col_idx //= 26
        return result
    else:
        return get_column_letter(col_idx)


class ExcelExporter:
    """Exports EBSD analysis results to Documentation.xlsx with 14 sheets.

    Attributes:
        file_path: Path to output Excel file
        wb: openpyxl Workbook instance
        dataset_count: Number of datasets exported (for column indexing)
    """

    def __init__(self, file_path: str):
        """Initialize Excel exporter.

        Args:
            file_path: Path to output Excel file (will be created/overwritten)
        """
        if not OPENPYXL_AVAILABLE:
            raise ImportError("openpyxl is required for Excel export. Install: pip install openpyxl")

        self.file_path = Path(file_path)
        self.dataset_count = 0

        # Create or load workbook. If a previous batch crashed mid-save the
        # file may exist with non-zero size but be a corrupt zip — load_workbook
        # would raise BadZipFile. Fall back to creating fresh so the user
        # isn't blocked from running another batch.
        loaded = False
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            try:
                self.wb = load_workbook(str(self.file_path))
                loaded = True
            except Exception as e:
                logger.warning(
                    "Could not load existing workbook %s (%s) — creating fresh",
                    self.file_path, e,
                )
        if not loaded:
            self.wb = Workbook()
            # Remove default sheet
            if "Sheet" in self.wb.sheetnames:
                self.wb.remove(self.wb["Sheet"])
            self._initialize_sheets()

    def _initialize_sheets(self):
        """Create all 14 sheets with headers."""

        # Sheet 1: Basic Info
        if "Sheet1" not in self.wb.sheetnames:
            ws1 = self.wb.create_sheet("Sheet1", 0)
        else:
            ws1 = self.wb["Sheet1"]

        # Row 1 = header row (dataset names in col 2+)
        ws1.cell(row=1, column=1, value="Metric")

        # Row labels for Sheet 1 (basic info + all metrics)
        # Labels start at row 2 to align with data rows
        row_labels = [
            "step_micron",         # row 2
            "dim_x",               # row 3
            "dim_y",               # row 4
            "n_grains",            # row 5
            "rotation_deg_1",      # row 6
            "rotation_deg_2",      # row 7
            "rotation_deg_3",      # row 8
            "gb_threshold_deg",    # row 9
            "min_pixels",          # row 10
            "min_intercept_um",    # row 11
            # Grain size metrics (rows 12-21)
            "mean_ECD_um",         # row 12
            "median_ECD_um",       # row 13
            "std_ECD_um",          # row 14
            "mean_MaxDimX_um",     # row 15
            "mean_MaxDimY_um",     # row 16
            "mean_AR",             # row 17
            "median_AR",           # row 18
            "grains_above_5um",    # row 19
            "grains_above_10um",   # row 20
            "grains_above_15um",   # row 21
            # Deformation metrics (rows 22-24)
            "mean_KAM_deg",        # row 22
            "HAGB_length_um",      # row 23
            "SAGB_length_um",      # row 24
            # Texture Index + Entropy (rows 25-26)
            "texture_index",       # row 25
            "entropy",             # row 26
        ]

        for i, label in enumerate(row_labels, start=2):
            ws1.cell(row=i, column=1, value=label)

        # Texture component labels (rows 27-40, 14 components)
        # Will be filled dynamically based on preset

        # RX metrics labels (rows 41-44)
        rx_labels = ["RXedGrainFraction", "highBC", "lowGKAM", "lowKAM"]
        for i, label in enumerate(rx_labels, start=41):
            ws1.cell(row=i, column=1, value=label)

        # Surface plane fractions (rows 45-47)
        plane_labels = ["plane_111_frac", "plane_100_frac", "plane_110_frac"]
        for i, label in enumerate(plane_labels, start=45):
            ws1.cell(row=i, column=1, value=label)

        # Sheet: BC hist
        if "BC hist" not in self.wb.sheetnames:
            ws_bc = self.wb.create_sheet("BC hist")
        else:
            ws_bc = self.wb["BC hist"]
        ws_bc.cell(row=1, column=1, value="BinCenters")

        # Pre-write bin centers (0-230, binWidth=10)
        bin_width = 10
        bin_centers = np.arange(5, 230, bin_width)  # 5, 15, 25, ..., 225
        for i, bc in enumerate(bin_centers, start=2):
            ws_bc.cell(row=i, column=1, value=float(bc))

        # Sheet: BC fit (3G)
        if "BC fit (3G)" not in self.wb.sheetnames:
            ws_fit = self.wb.create_sheet("BC fit (3G)")
        else:
            ws_fit = self.wb["BC fit (3G)"]

        fit_headers = ["datname", "Nbc",
                       "BC low w1", "mu1", "sigma1",
                       "BC med w2", "mu2", "sigma2",
                       "BC high w3", "mu3", "sigma3"]
        for col_idx, header in enumerate(fit_headers, start=1):
            ws_fit.cell(row=1, column=col_idx, value=header)

        # Sheet: BandContrast Group
        if "BandContrast Group" not in self.wb.sheetnames:
            ws_group = self.wb.create_sheet("BandContrast Group")
        else:
            ws_group = self.wb["BandContrast Group"]

        ws_group.cell(row=1, column=1, value="Metric")
        group_labels = ["BC_high_center", "BC_high_areaFrac",
                        "BC_mid_center", "BC_mid_areaFrac",
                        "BC_low_center", "BC_low_areaFrac"]
        for i, label in enumerate(group_labels, start=2):
            ws_group.cell(row=i, column=1, value=label)

        # Sheet: Grain Size hist
        if "Grain Size hist" not in self.wb.sheetnames:
            ws_gs = self.wb.create_sheet("Grain Size hist")
        else:
            ws_gs = self.wb["Grain Size hist"]

        # Sheet: KAM hist
        if "KAM hist" not in self.wb.sheetnames:
            ws_kam = self.wb.create_sheet("KAM hist")
        else:
            ws_kam = self.wb["KAM hist"]
        ws_kam.cell(row=1, column=1, value="Order")
        ws_kam.cell(row=2, column=1, value="Threshold")
        ws_kam.cell(row=4, column=1, value="BinCenters")

        # Sheet: GOS hist, area weighted
        if "GOS hist, area weighted" not in self.wb.sheetnames:
            ws_gos = self.wb.create_sheet("GOS hist, area weighted")
        else:
            ws_gos = self.wb["GOS hist, area weighted"]
        ws_gos.cell(row=1, column=1, value="GOS hist, area weighted")
        ws_gos.cell(row=2, column=1, value="BinCenters")

        # Sheet: gKAM - area weighted
        if "gKAM - area weighted" not in self.wb.sheetnames:
            ws_gkam = self.wb.create_sheet("gKAM - area weighted")
        else:
            ws_gkam = self.wb["gKAM - area weighted"]
        ws_gkam.cell(row=1, column=1, value="gKAM - area weighted")
        ws_gkam.cell(row=2, column=1, value="BinCenters")

        # Sheet: Grain Boundaries
        if "Grain Boundaries" not in self.wb.sheetnames:
            ws_gb = self.wb.create_sheet("Grain Boundaries")
        else:
            ws_gb = self.wb["Grain Boundaries"]

        gb_labels = ["GBperEBSD Al_area", "GBperfieldArea", "HAGBperEBSD Al_area",
                     "HAGBperarea", "SAGBperEBSD Al_area", "SAGBperarea",
                     "HAGBlength_micron", "SAGBlength_micron", "sphericity"]
        for i, label in enumerate(gb_labels, start=1):
            ws_gb.cell(row=i + 2, column=1, value=label)  # Row 3+

        # Segment histogram headers (row 11)
        ws_gb.cell(row=11, column=1, value="Misorientation [deg]")
        ws_gb.cell(row=11, column=2, value="BinCenters")

        # Sheet: SmallVSLargeGrain
        if "SmallVSLargeGrain" not in self.wb.sheetnames:
            ws_tx = self.wb.create_sheet("SmallVSLargeGrain")
        else:
            ws_tx = self.wb["SmallVSLargeGrain"]

        # Will be filled dynamically based on texture components
        ws_tx.cell(row=1, column=1, value="Component")
        ws_tx.cell(row=1, column=2, value="RX [%]")
        ws_tx.cell(row=1, column=3, value="Substr [%]")
        ws_tx.cell(row=1, column=4, value="All [%]")

    def export_dataset(self, dataset_name: str, results: Dict[str, Any]):
        """Export single dataset results to all sheets.

        Args:
            dataset_name: Name of dataset (e.g., "Al_sample1")
            results: Dictionary with all analysis results:
                - "basic_info": Dict[str, float] (step, dims, n_grains, rotation, thresholds)
                - "bc_histogram": BCHistogramResult
                - "bc_gmm": BCGMMResult
                - "grain_size": GrainSizeResult
                - "deformation": DeformationResult
                - "texture": TextureResult
                - "rx": RXResult
        """
        self.dataset_count += 1
        col_idx = self.dataset_count + 1  # B=2, C=3, etc.
        col_let = excel_column(col_idx)

        # Sheet 1: Basic Info
        ws1 = self.wb["Sheet1"]
        # Row 1 = header row with dataset names
        ws1.cell(row=1, column=col_idx, value=dataset_name)

        basic_info = results.get("basic_info", {})
        # Data rows start at 2, matching labels at rows 2-11
        row = 1
        for key in ["step_micron", "dim_x", "dim_y", "n_grains",
                    "rotation_deg_1", "rotation_deg_2", "rotation_deg_3",
                    "gb_threshold_deg", "min_pixels", "min_intercept_um"]:
            row += 1
            ws1.cell(row=row, column=col_idx, value=basic_info.get(key, 0))

        # Grain size metrics (rows 12-21, matching labels)
        if "grain_size" in results:
            gs = results["grain_size"]
            ws1.cell(row=12, column=col_idx, value=gs.avg_ECD)
            ws1.cell(row=13, column=col_idx, value=basic_info.get("median_ECD_um", 0))
            ws1.cell(row=14, column=col_idx, value=gs.std_ECD)
            ws1.cell(row=15, column=col_idx, value=gs.avg_maxDimX)
            ws1.cell(row=16, column=col_idx, value=gs.avg_maxDimY)
            ws1.cell(row=17, column=col_idx, value=gs.avg_AR)
            ws1.cell(row=18, column=col_idx, value=basic_info.get("median_AR", 0))
            ws1.cell(row=19, column=col_idx, value=basic_info.get("grains_above_5um", 0))
            ws1.cell(row=20, column=col_idx, value=basic_info.get("grains_above_10um", 0))
            ws1.cell(row=21, column=col_idx, value=basic_info.get("grains_above_15um", 0))

        # Deformation metrics (rows 22-24)
        if "deformation" in results:
            deform = results["deformation"]
            kam_mean = float(np.nanmean(deform.kam)) if deform.kam.size > 0 and np.any(np.isfinite(deform.kam)) else 0.0
            ws1.cell(row=22, column=col_idx, value=kam_mean)
            ws1.cell(row=23, column=col_idx, value=deform.hagb_length)
            ws1.cell(row=24, column=col_idx, value=deform.sagb_length)

        # Texture Index + Entropy (rows 25-26)
        if "texture" in results:
            tx = results["texture"]
            ws1.cell(row=25, column=col_idx, value=tx.texture_index if tx.texture_index else 0)
            ws1.cell(row=26, column=col_idx, value=tx.entropy if tx.entropy else 0)

            # Texture components (rows 27-40, 14 components)
            comp_row = 27
            for comp_name, vol_frac in tx.component_fractions.items():
                ws1.cell(row=comp_row, column=1, value=comp_name)
                ws1.cell(row=comp_row, column=col_idx, value=vol_frac)
                comp_row += 1

        # RX metrics (rows 41-44)
        if "rx" in results:
            rx = results["rx"]
            ws1.cell(row=41, column=col_idx, value=rx.rx_fraction)
            ws1.cell(row=42, column=col_idx, value=rx.high_bc_fraction)
            ws1.cell(row=43, column=col_idx, value=rx.low_gkam_fraction)
            ws1.cell(row=44, column=col_idx, value=rx.low_kam_fraction)

        # Surface plane fractions (rows 45-47)
        if "texture" in results:
            tx = results["texture"]
            ws1.cell(row=45, column=col_idx, value=tx.plane_111_fraction)
            ws1.cell(row=46, column=col_idx, value=tx.plane_100_fraction)
            ws1.cell(row=47, column=col_idx, value=tx.plane_110_fraction)

        # BC hist sheet
        if "bc_histogram" in results:
            bc_hist = results["bc_histogram"]
            ws_bc = self.wb["BC hist"]
            ws_bc.cell(row=1, column=col_idx, value=dataset_name)

            # Write probability-normalized counts
            for i, count in enumerate(bc_hist.counts, start=2):
                ws_bc.cell(row=i, column=col_idx, value=float(count))

        # BC fit (3G) sheet
        if "bc_gmm" in results and results["bc_gmm"] is not None:
            bc_gmm = results["bc_gmm"]
            ws_fit = self.wb["BC fit (3G)"]
            row = self.dataset_count + 1

            ws_fit.cell(row=row, column=1, value=dataset_name)
            ws_fit.cell(row=row, column=2, value=bc_gmm.n_samples)

            # Sort by means (low, mid, high)
            sorted_idx = np.argsort(bc_gmm.means)
            col = 3
            for idx in sorted_idx:
                ws_fit.cell(row=row, column=col, value=float(bc_gmm.weights[idx]))
                ws_fit.cell(row=row, column=col + 1, value=float(bc_gmm.means[idx]))
                ws_fit.cell(row=row, column=col + 2, value=float(bc_gmm.stds[idx]))
                col += 3

        # BandContrast Group sheet
        if "bc_gmm" in results and results["bc_gmm"] is not None:
            bc_gmm = results["bc_gmm"]
            ws_group = self.wb["BandContrast Group"]
            ws_group.cell(row=1, column=col_idx, value=dataset_name)

            sorted_idx = np.argsort(bc_gmm.means)
            ws_group.cell(row=2, column=col_idx, value=float(bc_gmm.means[sorted_idx[2]]))  # high center
            ws_group.cell(row=3, column=col_idx, value=float(bc_gmm.weights[sorted_idx[2]]))  # high frac
            ws_group.cell(row=4, column=col_idx, value=float(bc_gmm.means[sorted_idx[1]]))  # mid center
            ws_group.cell(row=5, column=col_idx, value=float(bc_gmm.weights[sorted_idx[1]]))  # mid frac
            ws_group.cell(row=6, column=col_idx, value=float(bc_gmm.means[sorted_idx[0]]))  # low center
            ws_group.cell(row=7, column=col_idx, value=float(bc_gmm.weights[sorted_idx[0]]))  # low frac

        # Grain Size hist sheet
        if "grain_size" in results:
            gs = results["grain_size"]
            ws_gs = self.wb["Grain Size hist"]
            ws_gs.cell(row=1, column=col_idx, value=dataset_name)

            # ECD histogram (computed from per-grain data)
            row_offset = 2
            ws_gs.cell(row=row_offset, column=1, value="BinCenters")
            ws_gs.cell(row=row_offset, column=2, value="ECD [µm]")
            if gs.ecd_per_grain is not None and len(gs.ecd_per_grain) > 0:
                ecd_counts, ecd_edges = np.histogram(gs.ecd_per_grain, bins=20)
                ecd_centers = (ecd_edges[:-1] + ecd_edges[1:]) / 2
                for i, (bin_center, count) in enumerate(zip(ecd_centers, ecd_counts), start=row_offset + 1):
                    ws_gs.cell(row=i, column=1, value=float(bin_center))
                    ws_gs.cell(row=i, column=col_idx, value=float(count))

        # KAM hist sheet
        if "deformation" in results:
            deform = results["deformation"]
            ws_kam = self.wb["KAM hist"]

            # Order and threshold (write once per dataset)
            ws_kam.cell(row=1, column=col_idx, value=deform.kam_order)
            ws_kam.cell(row=2, column=col_idx, value=deform.kam_threshold)
            ws_kam.cell(row=3, column=col_idx, value=dataset_name)

            # Histogram: bin centers are midpoints of kam_hist_edges
            bin_edges = deform.kam_hist_edges
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 if len(bin_edges) > 1 else bin_edges
            for i, (bin_center, count) in enumerate(zip(bin_centers, deform.kam_hist_counts), start=5):
                ws_kam.cell(row=4, column=1, value="BinCenters")  # Label only once
                ws_kam.cell(row=i, column=1, value=float(bin_center))
                ws_kam.cell(row=i, column=col_idx, value=float(count))

        # GOS hist sheet
        if "rx" in results:
            rx = results["rx"]
            ws_gos = self.wb["GOS hist, area weighted"]
            ws_gos.cell(row=1, column=col_idx, value=dataset_name)

            for i, (bin_center, count) in enumerate(zip(rx.gos_histogram[1], rx.gos_histogram[0]), start=3):
                ws_gos.cell(row=i, column=1, value=float(bin_center))
                ws_gos.cell(row=i, column=col_idx, value=float(count))

        # gKAM hist sheet
        if "rx" in results:
            rx = results["rx"]
            ws_gkam = self.wb["gKAM - area weighted"]
            ws_gkam.cell(row=1, column=col_idx, value=dataset_name)

            for i, (bin_center, count) in enumerate(zip(rx.gkam_histogram[1], rx.gkam_histogram[0]), start=3):
                ws_gkam.cell(row=i, column=1, value=float(bin_center))
                ws_gkam.cell(row=i, column=col_idx, value=float(count))

        # Grain Boundaries sheet
        if "deformation" in results:
            deform = results["deformation"]
            ws_gb = self.wb["Grain Boundaries"]
            ws_gb.cell(row=1, column=col_idx, value=dataset_name)

            # Basic GB metrics (rows 3-11)
            ws_gb.cell(row=3, column=col_idx, value=deform.gb_per_phase_area)
            ws_gb.cell(row=4, column=col_idx, value=deform.gb_per_field_area)
            ws_gb.cell(row=5, column=col_idx, value=deform.hagb_per_phase_area)
            ws_gb.cell(row=6, column=col_idx, value=deform.hagb_per_field_area)
            ws_gb.cell(row=7, column=col_idx, value=deform.sagb_per_phase_area)
            ws_gb.cell(row=8, column=col_idx, value=deform.sagb_per_field_area)
            ws_gb.cell(row=9, column=col_idx, value=deform.hagb_length)
            ws_gb.cell(row=10, column=col_idx, value=deform.sagb_length)
            # sphericity is an array per grain; write the mean
            sph = deform.sphericity
            sph_val = float(np.nanmean(sph)) if sph is not None and len(np.atleast_1d(sph)) > 0 else 0.0
            ws_gb.cell(row=11, column=col_idx, value=sph_val)

            # Segment histogram (rows 12+)
            if deform.gb_seg_hist_counts is not None:
                for i, (bin_center, count) in enumerate(zip(deform.gb_seg_hist_bins,
                                                             deform.gb_seg_hist_counts), start=12):
                    ws_gb.cell(row=i, column=2, value=float(bin_center))
                    ws_gb.cell(row=i, column=col_idx, value=float(count))

    def save(self):
        """Save workbook to file. Creates parent directory if needed."""
        # mkdir parent so a fresh user dir like Documents/EBSD_2026-04/
        # doesn't fail with FileNotFoundError on the first batch save.
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.wb.save(str(self.file_path))

    def close(self):
        """Close workbook."""
        self.wb.close()


def export_to_excel(file_path: str, dataset_name: str, results: Dict[str, Any]):
    """Export single dataset to Excel file.

    Convenience function for single-dataset export.

    Args:
        file_path: Path to output Excel file
        dataset_name: Name of dataset
        results: Dictionary with all analysis results
    """
    exporter = ExcelExporter(file_path)
    exporter.export_dataset(dataset_name, results)
    exporter.save()
    exporter.close()


def export_batch_to_excel(file_path: str, datasets: Dict[str, Dict[str, Any]]):
    """Export multiple datasets to single Excel file.

    Args:
        file_path: Path to output Excel file
        datasets: Dictionary of {dataset_name: results_dict}
    """
    exporter = ExcelExporter(file_path)

    for dataset_name, results in datasets.items():
        exporter.export_dataset(dataset_name, results)

    exporter.save()
    exporter.close()
