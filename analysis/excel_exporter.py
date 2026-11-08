"""
Excel Export Module for EBSD Analysis Results.

Exports analysis results to Documentation.xlsx with 11 sheets, one column per dataset unless noted:
- "Sheet1": basic info (step, dimensions, #grains, rotation, thresholds), grain-size metrics, KAM, boundary
  lengths, texture index + entropy, texture components, RX metrics, surface planes, grain settings and counts,
  the RX criteria used (rows 60-63), the plane families (rows 64-66), the phase, preset and tolerance of the
  texture (rows 67-69), the phase of the RX analysis (row 70) and the HAGB limit of the boundary classification
  (row 71)
- "BC hist": band contrast histogram (share of the pixels with a band contrast, fixed bins)
- "BC fit (3G)": 3-Gaussian GMM fit parameters (one row per dataset)
- "BandContrast Group": BC_high/mid/low center + areaFrac
- "Grain Size hist": ECD histogram, number of grains per 5 um bin (one block; see the differences below)
- "KAM hist": KAM distribution, probability per 0.25 deg bin
- "GOS hist, area weighted": GOS distribution, percent of the total grain area per 0.25 deg bin
- "gKAM - area weighted": grain-average KAM distribution, percent of the total grain area per 0.2 deg bin
- "Grain Boundaries": scalar boundary metrics, mean sphericity, segment length histogram (um per 2 deg bin)
- "Sphericity per grain": the sphericity of every grain (grain number in column A)
- "SmallVSLargeGrain": texture-component volume fractions of all grains, substructure grains and RX grains,
  one row per dataset (filled when ``results["texture"]`` and ``results["texture_by_rx"]`` are given)

Reference: matlab_testskripts/EBSDanalysis_frame.m, RxxAnalysis.m, etc.
Architecture: the MTEX-equivalence design notes §3.3

Differences from the script's workbook (the Documentation.xlsx the MATLAB scripts write). They are on purpose
unless a line says "not done"; docs/user-guide/Analysis.md lists them for users.
- "Grain Size hist": the script writes three blocks (maxDimX and maxDimY as probability, and an area-weighted
  ECD histogram, GrainSizeAnalysis_mtex6.m:160-190). Orienta writes ONE block: the ECD of the grains as counts
  (not normalised) on the script's bins 0:5:140. The maxDimX, maxDimY and area-weighted blocks are not done.
- "Grain Boundaries": the script puts a dataset in column n+2 (its bin edges sit in column B,
  DeformationAnalysis.m:100, :119); Orienta uses column n+1 like every other sheet and the bin centres in
  column A. Orienta labels the rows with "SAGB"/"perarea" where the script has "LAGB"/"perfieldArea"
  (DeformationAnalysis.m:46-47); the values are the same quantities. The per-grain sphericity is on its own
  sheet (the script writes it into the same column at row 10, :55-57) and row 11 holds its mean.
- "KAM hist": the script writes the bin EDGES under the label "BinCenters" (DeformationAnalysis.m:162);
  Orienta writes the bin midpoints, on the fixed grid 0-15 deg.
- "gKAM - area weighted": the script writes the GOS bin centres (0.125, 0.375, ...) although its bins are 0.2 deg
  wide (RxxAnalysis.m:129); Orienta writes the centres of the gKAM bins (0.1, 0.3, ...).
- Grain Boundaries segment bins: the script labels a bin by its lower edge (3, 5, ..., 63); Orienta writes the
  bin centre (4, 6, ..., 64).
- GOS and gKAM sheets: the values are percent of the total grain area (closed holes included), as MTEX's grain
  histogram returns (measured, independent of the step): each grain is weighted by its cell area and the sum
  is divided by the summed area of all grains (RXResult.total_grain_area); a grain outside the bins stays in
  the denominator.
- A workbook written by an older version of this exporter (it has no sheet "Sphericity per grain") is not
  written into: rows and bin centres moved, so old counts would stand under the wrong bins.
"""

import logging
import os
import threading
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


# The page tells the user what Sheet 1 records: the locale key `analysis:batch.workbookRecords` (and `...Export`) names
# the reconstruction parameters (rows 9-11 and 48), the edge rule, the ECD basis, the statistics phase, the three grain
# counts and the safeguard flag (GRAIN_STATS_KEYS) and the RX criteria (RX_CRITERIA_KEYS).
# tests/test_grain_stats_phase.py::test_sheet1_records_what_the_page_promises pins them: change a row, change the sentence.
# Sheet 1 rows 49-54, below grain_alpha (row 48): the grain-size settings and the grain counts they gave.
# Rows 55-59: the four columns of the script that rows 12-21 lack (spreads of MaxDim X/Y and aspect ratio,
# the area-weighted ECD) and the phase the statistics are made of.
GRAIN_STATS_KEYS = (
    "grain_edge_rule", "grain_ecd_basis", "n_grains_total", "n_edge_grains", "n_grains_used",
    "edge_safeguard_applied",
    "std_maxDimX", "std_maxDimY", "std_AR", "area_weighted_ECD", "grain_stats_phase",
)

# Sheet 1 rows 60-63, below the grain-size rows: the RX criteria the RX block (rows 41-44, the GOS and gKAM sheets
# and the SmallVSLargeGrain sheet) was made with, in degrees, as a fraction of the largest gBC, in degrees, in um.
# Written from the RXResult, so they are what was used; empty for a dataset whose RX analysis did not run.
RX_CRITERIA_KEYS = ("rx_gos_threshold_deg", "rx_gbc_min_fraction", "rx_gkam_threshold_deg", "rx_min_radius_um")


# Sheet 1 rows 45-47: the script's surface-plane numbers (one direction per pixel). Rows 64-66, below the RX criteria:
# the share of pixels that have any plane of the family in the surface (the physical reading of the column title).
PLANE_LABELS = ("pct of {111} planes in surface (one direction, as the script)",
                "pct of {100} planes in surface (one direction, as the script)",
                "pct of {110} planes in surface (one direction, as the script)")
PLANE_FAMILY_LABELS = ("pct of {111} planes in surface (any plane of the family)",
                       "pct of {100} planes in surface (any plane of the family)",
                       "pct of {110} planes in surface (any plane of the family)")
PLANE_FAMILY_FIELDS = ("plane_111_family_fraction", "plane_100_family_fraction", "plane_110_family_fraction")
# Sheet 1 rows 25 and 26: the script's norm(odf)^2 and entropy(odf) of the 5 degree kernel density of the texture phase
TEXTURE_INDEX_LABEL = "TextureIndex (norm(odf)^2, 5° kernel density)"
TEXTURE_ENTROPY_LABEL = "Entropy (-∫ f ln f, 5° kernel density)"
# Sheet 1 row 67, after the last row in use (64-66 are the plane families): the phase every texture number is of
TEXTURE_PHASE_ROW = 67
TEXTURE_PHASE_LABEL = "texture_phase"
# Rows 68 and 69: the preset and the tolerance angle (degrees) the component fractions were counted with
TEXTURE_PRESET_ROW = 68
TEXTURE_PRESET_LABEL = "texture_preset"
TEXTURE_TOLERANCE_ROW = 69
TEXTURE_TOLERANCE_LABEL = "texture_tolerance_deg"
# Row 70: the phase the RX block (rows 41-44, 60-63, the GOS and gKAM sheets, the RX columns of SmallVSLargeGrain)
# is of; empty for a dataset whose RX analysis did not run. Row 71: the angle from which a boundary counts as HAGB
# (degrees; SAGB runs from 3 degrees up to it); the KAM order and threshold are on the sheet "KAM hist" (B1:B2).
RX_PHASE_ROW = 70
RX_PHASE_LABEL = "rx_phase"
HAGB_LIMIT_ROW = 71
HAGB_LIMIT_LABEL = "hagb_min_deg"

# Row 11 keeps the value as entered, but the grain reconstruction has no such parameter
# (the MATLAB pipeline carried minLIL unused as well), so the label says so.
MIN_INTERCEPT_LABEL = "min_intercept_um (not applied)"

# "Grain Size hist" bins every dataset on the same edges (GrainSizeAnalysis_mtex6.m: binEdges = 0:5:140), so the
# bin centres in column A are the same for every dataset and the columns can be compared row by row.
GS_BIN_EDGES = np.arange(0, 145, 5, dtype=float)
GS_BIN_CENTERS = (GS_BIN_EDGES[:-1] + GS_BIN_EDGES[1:]) / 2.0
GS_FIRST_ROW = 3

# Every histogram sheet has one set of bin centres in column A, written once with the labels; a dataset only
# writes its counts, into the row of the bin they belong to (see _place_on_grid). A dataset whose bins are not
# on the grid is left out with a warning, so no dataset can move another dataset's centres.
#
# "KAM hist": the script bins on 'BinWidth' 0.25 from the data (DeformationAnalysis.m:150), i.e. edges that
# depend on the data. KAM is a mean of disorientations below the KAM threshold (7 deg by default), so it is
# bounded by that threshold; the fixed range 0-30 deg covers every threshold the page accepts (at most 30 deg)
# with the script's bin width. analyze_deformation bins on the same 0.25 deg grid from 0, so its counts land exactly.
# Deliberate difference: the script writes the bin EDGES (KAMhist_smoothed_Edges, DeformationAnalysis.m:162)
# under the label "BinCenters"; Orienta writes the true midpoints of the bins.
KAM_BIN_CENTERS = np.arange(0, 30, 0.25) + 0.125
KAM_FIRST_ROW = 5
# "GOS hist, area weighted" / "gKAM - area weighted": fixed edges 0:0.25:15 and 0:0.2:5 (RxxAnalysis.m:106-107,
# 119-120), the same as rx_analysis uses.
# Deliberate difference: the script's gKAM sheet gets the GOS centres (binedgesGOS(1:25) + 0.25/2,
# RxxAnalysis.m:129), 0.125, 0.375, ..., although its bins are 0.2 wide. Orienta writes the
# centres of the gKAM bins (0.1, 0.3, ...) instead.
GOS_BIN_CENTERS = np.arange(0, 15, 0.25) + 0.125
GKAM_BIN_CENTERS = np.arange(0, 5, 0.2) + 0.1
RX_HIST_FIRST_ROW = 3
# "Grain Boundaries" segment histogram: the script sums the segment lengths in bins [i, i+2) for i = 3:2:63
# (DeformationAnalysis.m:110-117); the centres are 4, 6, ..., 64. deformation_analysis starts at sagb_min=3
# with width 2 and stops at 63, so its default bins (30, centres 4..62) are a part of this grid.
# Deliberate difference: the script labels each segment bin by its lower edge (3, 5, ..., 63,
# DeformationAnalysis.m:119); Orienta writes the bin centre (4, 6, ..., 64).
GB_SEG_CENTERS = np.arange(4, 65, 2, dtype=float)

# "Grain Boundaries": rows 3-11 are the scalar metrics, the segment histogram follows below them.
GB_SEG_TITLE_ROW = 12
GB_SEG_HEADER_ROW = 13
GB_SEG_FIRST_ROW = 14

SPHERICITY_SHEET = "Sphericity per grain"


def _place_on_grid(ws, col: int, first_row: int, grid: np.ndarray, centres, counts, what: str) -> int:
    """Write counts into the row of their bin on a fixed grid of bin centres; return how many were left out.

    A bin is placed only when its centre is a grid centre and the bins of the dataset are as wide as the grid
    (otherwise a row would hold the count of another bin width). Everything else is left out and reported with
    a warning that gives the number of bins and their summed count; nothing is rebinned or guessed.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1)
    counts = np.asarray(counts, dtype=float).reshape(-1)
    # A result whose centres and counts do not belong together, or whose centres are not numbers, is a broken
    # result, not a bin grid mismatch: fail with the sheet's name instead of truncating or raising on int(nan).
    if len(centres) != len(counts):
        raise ValueError(f"{what}: {len(centres)} bin centres but {len(counts)} counts; "
                         f"the histogram result is inconsistent")
    if not np.all(np.isfinite(centres)):
        raise ValueError(f"{what}: {int(np.sum(~np.isfinite(centres)))} of {len(centres)} bin centres are "
                         f"NaN or infinite; the histogram result is inconsistent")
    n = len(centres)
    step = float(grid[1] - grid[0])
    same_width = n < 2 or bool(np.allclose(np.diff(centres), step, atol=1e-6))
    left_out, left_out_sum = 0, 0.0
    for centre, count in zip(centres, counts):
        k = int(round((centre - grid[0]) / step))
        if same_width and 0 <= k < len(grid) and abs(centre - grid[k]) < 1e-6:
            ws.cell(row=first_row + k, column=col, value=float(count))
        else:
            left_out += 1
            left_out_sum += float(count)
    if left_out:
        logger.warning("%s: %d of %d bins are not on the sheet's fixed bin grid (%g to %g, step %g) and are "
                       "left out (their counts sum to %g)", what, left_out, n, grid[0], grid[-1], step,
                       left_out_sum)
    return left_out


def _area_percent(rx, counts, what: str) -> np.ndarray:
    """Area-weighted counts as the share of the total grain area in percent (what MTEX returns).

    The counts are grain cell areas (closed holes included); the denominator is ``RXResult.total_grain_area``,
    the area of all grains, so a grain outside the bins stays in it. A result without it falls back to the sum
    of the counts (the share of the binned area), with a warning.
    """
    counts = np.asarray(counts, dtype=float)
    total = getattr(rx, "total_grain_area", None)
    if total is None or not total > 0:
        total = float(np.sum(counts))
        if not total > 0:
            return counts
        logger.warning("%s: the result does not carry the total grain area; the values are percent of the "
                       "binned area", what)
    return counts * (100.0 / float(total))


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


# The names this process holds right now (see free_workbook_path). The backend is one process, so an empty file
# that is not in this set was left by a run that died.
_LIVE_RESERVATIONS: set = set()
_RESERVATION_LOCK = threading.Lock()


def _reservation_key(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def free_workbook_path(path) -> Path:
    """``path`` if nothing exists there, else the first free ``<name>_2<ext>``, ``<name>_3<ext>``, ... --
    and the name is *reserved*: an empty file is created there, atomically (``open(..., "xb")``), so two runs
    that ask at the same moment never get the same name. The run writes its workbook over that file
    (``ExcelExporter`` treats an empty file as new); a run that ends calls
    :func:`release_workbook_reservation`, which removes the empty file if nothing was written.

    The empty file lives until the caller writes the workbook or releases the name; the folder is made if it
    does not exist.

    An existing EMPTY file that this process does not hold is the leftover of a run that died (a killed
    backend): it was never a workbook, so it is reused (and the log says so). A workbook with content, written
    by an earlier run, is never opened or overwritten, so two runs never share cells.

    Raises:
        OSError: the directory cannot be made or written (nothing is reserved then)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    with _RESERVATION_LOCK:
        while True:
            candidate = path if n == 1 else path.with_name(f"{path.stem}_{n}{path.suffix}")
            key = _reservation_key(candidate)
            if key in _LIVE_RESERVATIONS:
                n += 1
                continue
            try:
                with open(candidate, "xb"):
                    pass
            except FileExistsError:
                if _is_empty_file(candidate):
                    logger.warning("%s is an empty leftover of a run that did not finish (it never held a "
                                   "workbook); it is reused", candidate)
                else:
                    n += 1
                    continue
            _LIVE_RESERVATIONS.add(key)
            return candidate


def _is_empty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size == 0
    except OSError:
        return False


def release_workbook_reservation(path) -> None:
    """The run is over with the name :func:`free_workbook_path` gave it: delete the file if it is still empty.

    A file with content (a workbook, even a half-written one) is never touched; see
    :func:`set_aside_incomplete_workbook` for that. ``None`` is accepted (nothing was reserved).
    """
    if path is None:
        return
    path = Path(path)
    with _RESERVATION_LOCK:
        _LIVE_RESERVATIONS.discard(_reservation_key(path))
        try:
            if _is_empty_file(path):
                path.unlink()
        except OSError:
            logger.warning("Could not remove the empty reservation %s", path, exc_info=True)


def _set_aside(path):
    """``(kept, stuck)``: the new path of a file with content that was renamed, and whether such a file could not
    be renamed (a lock on Windows) and so still has the workbook name. ``(None, False)`` if there is nothing to set aside."""
    if path is None:
        return None, False
    path = Path(path)
    try:
        if not (path.is_file() and path.stat().st_size > 0):
            return None, False
    except OSError:
        logger.warning("Could not look at the incomplete file %s", path, exc_info=True)
        return None, False
    try:
        n = 1
        while True:
            suffix = ".incomplete" if n == 1 else f".incomplete_{n}"
            candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
            if not candidate.exists():
                path.rename(candidate)
                return candidate, False
            n += 1
    except OSError:
        logger.warning("Could not set aside the incomplete file %s", path, exc_info=True)
        return None, True


def set_aside_incomplete_workbook(path) -> Optional[Path]:
    """Rename a file with content that a failed export left behind to ``<name>.incomplete<ext>`` (the first
    free ``<name>.incomplete_2<ext>``, ... if that exists) and return the new path.

    None if there is no such file (missing, empty or ``path`` is None): nothing is renamed. The file is kept,
    never deleted, and never again taken for a workbook of the name it had. None also if the rename failed;
    use :func:`incomplete_workbook_note` for the sentence that tells the user which of the two it was.
    """
    return _set_aside(path)[0]


def incomplete_workbook_note(path) -> str:
    """Set the incomplete file at ``path`` aside (as :func:`set_aside_incomplete_workbook`) and return the sentence
    for the failure message: where the file was kept, or that it could not be renamed and still has the workbook
    name (then a reader must not take it for a finished workbook). An empty string if there was no such file."""
    kept, stuck = _set_aside(path)
    if kept is not None:
        return f" The incomplete file was kept as {kept.name}."
    if stuck:
        return (f" The incomplete file could not be renamed (it may be open in another program) and still has the "
                f"workbook name {Path(path).name}: it is not a finished workbook.")
    return ""


class ExcelExporter:
    """Exports EBSD analysis results to Documentation.xlsx with 11 sheets.

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
        if loaded and SPHERICITY_SHEET not in self.wb.sheetnames:
            # Only this layout's _initialize_sheets creates the sheet, so every workbook it has written carries it.
            # An older workbook has its bin centres and counts in other rows and columns: writing into it would
            # put old counts beside new bin centres without a trace.
            sheets = list(self.wb.sheetnames)
            self.wb.close()
            raise ValueError(
                f"{self.file_path} is a workbook in an older layout (it has no sheet {SPHERICITY_SHEET!r}; "
                f"sheets: {sheets}). Its histogram rows and columns differ from this version, so new results "
                f"would stand beside old counts under the wrong bin centres. Choose a new file name.")
        if loaded and "Sheet1" in self.wb.sheetnames and self.wb["Sheet1"]["B1"].value not in (None, ""):
            # dataset_count starts at 0, as the MATLAB frame restarts n for every run: this export begins in
            # column B again. Older cells are not cleared, so what lies beyond the new data is still the old run's.
            logger.warning("Workbook %s already holds a dataset in Sheet1!B1 (%r): the first dataset column is "
                           "overwritten, and cells of the earlier run that this export does not write stay as "
                           "they are", self.file_path, self.wb["Sheet1"]["B1"].value)
        # Also for a workbook that was loaded: it may come from an older version, so every label is written
        # again and a sheet added since is created. Values are only written per dataset by export_dataset.
        self._initialize_sheets()

    def _initialize_sheets(self):
        """Create the sheets that are missing and write all labels, headers and fixed bin centres.

        Safe to call on a workbook that already has them: every cell is written with its current value.
        """

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
            "n_grains (all phases)",  # row 5
            "rotation_deg_1",      # row 6
            "rotation_deg_2",      # row 7
            "rotation_deg_3",      # row 8
            "gb_threshold_deg",    # row 9
            "min_pixels",          # row 10
            MIN_INTERCEPT_LABEL,   # row 11
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
            TEXTURE_INDEX_LABEL,   # row 25
            TEXTURE_ENTROPY_LABEL,  # row 26
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
        for i, label in enumerate(PLANE_LABELS, start=45):
            ws1.cell(row=i, column=1, value=label)

        # Grain reconstruction hole-closing factor (row 48)
        ws1.cell(row=48, column=1, value="grain_alpha")

        # Grain-size statistics: edge rule, ECD basis, the grain counts they gave, the script's spreads
        # and weighted ECD, and the phase (rows 49-59)
        for i, label in enumerate(GRAIN_STATS_KEYS, start=49):
            ws1.cell(row=i, column=1, value=label)

        # The RX criteria of the RX block (rows 60-63)
        for i, label in enumerate(RX_CRITERIA_KEYS, start=60):
            ws1.cell(row=i, column=1, value=label)

        # The surface-plane family fractions (rows 64-66)
        for i, label in enumerate(PLANE_FAMILY_LABELS, start=64):
            ws1.cell(row=i, column=1, value=label)

        # The phase of the texture numbers (row 67)
        ws1.cell(row=TEXTURE_PHASE_ROW, column=1, value=TEXTURE_PHASE_LABEL)
        ws1.cell(row=TEXTURE_PRESET_ROW, column=1, value=TEXTURE_PRESET_LABEL)
        ws1.cell(row=TEXTURE_TOLERANCE_ROW, column=1, value=TEXTURE_TOLERANCE_LABEL)

        # The phase of the RX block (row 70) and the HAGB limit (row 71)
        ws1.cell(row=RX_PHASE_ROW, column=1, value=RX_PHASE_LABEL)
        ws1.cell(row=HAGB_LIMIT_ROW, column=1, value=HAGB_LIMIT_LABEL)

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
        ws_gs.cell(row=2, column=1, value="BinCenters (ECD [µm]), counts")
        for i, centre in enumerate(GS_BIN_CENTERS, start=GS_FIRST_ROW):
            ws_gs.cell(row=i, column=1, value=float(centre))

        # Sheet: KAM hist
        if "KAM hist" not in self.wb.sheetnames:
            ws_kam = self.wb.create_sheet("KAM hist")
        else:
            ws_kam = self.wb["KAM hist"]
        ws_kam.cell(row=1, column=1, value="Order")
        ws_kam.cell(row=2, column=1, value="Threshold")
        ws_kam.cell(row=4, column=1, value="BinCenters")
        for i, centre in enumerate(KAM_BIN_CENTERS, start=KAM_FIRST_ROW):
            ws_kam.cell(row=i, column=1, value=float(centre))

        # Sheet: GOS hist, area weighted
        if "GOS hist, area weighted" not in self.wb.sheetnames:
            ws_gos = self.wb.create_sheet("GOS hist, area weighted")
        else:
            ws_gos = self.wb["GOS hist, area weighted"]
        ws_gos.cell(row=1, column=1, value="GOS hist, area weighted")
        ws_gos.cell(row=2, column=1, value="BinCenters [deg]; values: % of the total grain area")
        for i, centre in enumerate(GOS_BIN_CENTERS, start=RX_HIST_FIRST_ROW):
            ws_gos.cell(row=i, column=1, value=float(centre))

        # Sheet: gKAM - area weighted
        if "gKAM - area weighted" not in self.wb.sheetnames:
            ws_gkam = self.wb.create_sheet("gKAM - area weighted")
        else:
            ws_gkam = self.wb["gKAM - area weighted"]
        ws_gkam.cell(row=1, column=1, value="gKAM - area weighted")
        ws_gkam.cell(row=2, column=1, value="BinCenters [deg]; values: % of the total grain area")
        for i, centre in enumerate(GKAM_BIN_CENTERS, start=RX_HIST_FIRST_ROW):
            ws_gkam.cell(row=i, column=1, value=float(centre))

        # Sheet: Grain Boundaries
        if "Grain Boundaries" not in self.wb.sheetnames:
            ws_gb = self.wb.create_sheet("Grain Boundaries")
        else:
            ws_gb = self.wb["Grain Boundaries"]

        gb_labels = ["GBperEBSD Al_area", "GBperfieldArea", "HAGBperEBSD Al_area",
                     "HAGBperarea", "SAGBperEBSD Al_area", "SAGBperarea",
                     "HAGBlength_micron", "SAGBlength_micron", "sphericity (mean)"]
        for i, label in enumerate(gb_labels, start=1):
            ws_gb.cell(row=i + 2, column=1, value=label)  # Rows 3-11

        # Segment histogram: title row, then the header row with BinCenters in column A (like the other
        # histogram sheets); the dataset names go over the counts in export_dataset
        ws_gb.cell(row=GB_SEG_TITLE_ROW, column=1, value="Misorientation [deg]")
        ws_gb.cell(row=GB_SEG_HEADER_ROW, column=1, value="BinCenters")
        for i, centre in enumerate(GB_SEG_CENTERS, start=GB_SEG_FIRST_ROW):
            ws_gb.cell(row=i, column=1, value=float(centre))

        # Sheet: Sphericity per grain (DeformationAnalysis.m:55-57 writes the vector, one column per dataset)
        if SPHERICITY_SHEET not in self.wb.sheetnames:
            ws_sph = self.wb.create_sheet(SPHERICITY_SHEET)
        else:
            ws_sph = self.wb[SPHERICITY_SHEET]
        ws_sph.cell(row=1, column=1, value="grain")

        # Sheet: SmallVSLargeGrain
        if "SmallVSLargeGrain" not in self.wb.sheetnames:
            ws_tx = self.wb.create_sheet("SmallVSLargeGrain")
        else:
            ws_tx = self.wb["SmallVSLargeGrain"]

        # One row per dataset (row n + 2 as in TextureAnalysis_mtex6.m:710-720); the three blocks of component
        # columns (all grains, substructure grains, RX grains) get their titles and names in export_dataset.
        ws_tx.cell(row=1, column=1, value="Volume fraction [%] of the pixels of each group")
        ws_tx.cell(row=2, column=1, value="Dataset")

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
            ws1.cell(row=row, column=col_idx, value=basic_info.get(key, ""))     # a missing key is an empty cell

        # Grain size metrics (rows 12-21, matching labels). A statistics phase (or a map) without a grain has no
        # mean, spread or median: those cells stay empty, never a 0.0 that reads like a measurement (rows 51-53,
        # the grain counts, say why).
        if "grain_size" in results:
            gs = results["grain_size"]
            no_grain = gs.n_grains_used == 0
            ws1.cell(row=12, column=col_idx, value=None if no_grain else gs.avg_ECD)
            ws1.cell(row=13, column=col_idx, value=None if no_grain else basic_info.get("median_ECD_um", ""))
            ws1.cell(row=14, column=col_idx, value=None if no_grain else gs.std_ECD)
            ws1.cell(row=15, column=col_idx, value=None if no_grain else gs.avg_maxDimX)
            ws1.cell(row=16, column=col_idx, value=None if no_grain else gs.avg_maxDimY)
            ws1.cell(row=17, column=col_idx, value=None if no_grain else gs.avg_AR)
            ws1.cell(row=18, column=col_idx, value=None if no_grain else basic_info.get("median_AR", ""))
            ws1.cell(row=19, column=col_idx, value=basic_info.get("grains_above_5um", ""))
            ws1.cell(row=20, column=col_idx, value=basic_info.get("grains_above_10um", ""))
            ws1.cell(row=21, column=col_idx, value=basic_info.get("grains_above_15um", ""))

        # Deformation metrics (rows 22-24)
        if "deformation" in results:
            deform = results["deformation"]
            # no finite KAM (no pixel has a neighbour in its grain): an empty cell, not a mean of 0
            kam_mean = float(np.nanmean(deform.kam)) if deform.kam.size > 0 and np.any(np.isfinite(deform.kam)) else None
            ws1.cell(row=22, column=col_idx, value=kam_mean)
            ws1.cell(row=23, column=col_idx, value=deform.hagb_length)
            ws1.cell(row=24, column=col_idx, value=deform.sagb_length)
            ws1.cell(row=HAGB_LIMIT_ROW, column=col_idx, value=getattr(deform, "hagb_min", None))

        # Texture Index + Entropy (rows 25-26)
        if "texture" in results:
            tx = results["texture"]
            # an entropy of 0.0 is a value; a missing one (None) is an empty cell, not a 0
            ws1.cell(row=25, column=col_idx, value=None if tx.texture_index is None else float(tx.texture_index))
            ws1.cell(row=26, column=col_idx, value=None if tx.entropy is None else float(tx.entropy))

            # Texture components (rows 27-40, 14 components)
            comp_row = 27
            renamed = [(r, ws1.cell(row=r, column=1).value, name)
                       for r, name in enumerate(tx.component_fractions, start=27)
                       if ws1.cell(row=r, column=1).value not in (None, "", name)]
            if renamed:
                logger.warning(
                    "Sheet1: dataset %r names %d texture components differently from the names already in "
                    "column A (first: row %d %r -> %r); the names are replaced, so the earlier datasets' "
                    "values in those rows belong to the old names (use one texture preset per workbook)",
                    dataset_name, len(renamed), *renamed[0])
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
            for i, value in enumerate((rx.gos_threshold, rx.gbc_min_fraction, rx.gkam_threshold,
                                       rx.min_grain_radius_um), start=60):
                ws1.cell(row=i, column=col_idx, value=float(value))
            ws1.cell(row=RX_PHASE_ROW, column=col_idx, value=getattr(rx, "phase_name", None))

        # Surface plane fractions (rows 45-47)
        if "texture" in results:
            tx = results["texture"]
            ws1.cell(row=45, column=col_idx, value=tx.plane_111_fraction)
            ws1.cell(row=46, column=col_idx, value=tx.plane_100_fraction)
            ws1.cell(row=47, column=col_idx, value=tx.plane_110_fraction)
            for i, field in enumerate(PLANE_FAMILY_FIELDS, start=64):
                value = getattr(tx, field, None)
                ws1.cell(row=i, column=col_idx, value=None if value is None else float(value))
            # the phase every texture number of this column is of
            ws1.cell(row=TEXTURE_PHASE_ROW, column=col_idx, value=getattr(tx, "phase_name", None))
            ws1.cell(row=TEXTURE_PRESET_ROW, column=col_idx, value=getattr(tx, "preset_name", None))
            tolerance = getattr(tx, "tolerance_angle", None)
            ws1.cell(row=TEXTURE_TOLERANCE_ROW, column=col_idx, value=None if tolerance is None else float(tolerance))

        # Grain reconstruction hole-closing factor (row 48)
        ws1.cell(row=48, column=col_idx, value=basic_info.get("grain_alpha", ""))

        # Grain-size statistics settings, counts, spreads and phase (rows 49-59). Rows 55-59 come from
        # the GrainSizeResult when there is one (the same source as rows 12-17), else from basic_info.
        gs_values = {}
        if "grain_size" in results:
            gs = results["grain_size"]
            none_if_no_grain = (lambda v: None) if gs.n_grains_used == 0 else (lambda v: v)
            gs_values = {
                "std_maxDimX": none_if_no_grain(gs.std_maxDimX), "std_maxDimY": none_if_no_grain(gs.std_maxDimY),
                "std_AR": none_if_no_grain(gs.std_AR), "area_weighted_ECD": none_if_no_grain(gs.area_weighted_ECD),
                "grain_stats_phase": gs.stats_phase_name if gs.stats_phase is not None else "all phases",
            }
        for i, key in enumerate(GRAIN_STATS_KEYS, start=49):
            ws1.cell(row=i, column=col_idx, value=gs_values.get(key, basic_info.get(key, "")))

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

            # ECD histogram on the fixed bins of the sheet (centres in column A are written once, with the labels)
            if gs.ecd_per_grain is not None and len(gs.ecd_per_grain) > 0:
                ecd = np.asarray(gs.ecd_per_grain, dtype=float)
                n_above = int(np.sum(ecd > GS_BIN_EDGES[-1]))
                if n_above:
                    logger.warning("Grain Size hist: %d of %d grains have an ECD above %g um and are left out "
                                   "of the histogram (dataset %r)", n_above, ecd.size, GS_BIN_EDGES[-1],
                                   dataset_name)
                ecd_counts, _ = np.histogram(ecd, bins=GS_BIN_EDGES)
                for i, count in enumerate(ecd_counts, start=GS_FIRST_ROW):
                    ws_gs.cell(row=i, column=col_idx, value=float(count))

        # KAM hist sheet
        if "deformation" in results:
            deform = results["deformation"]
            ws_kam = self.wb["KAM hist"]

            # Order and threshold (write once per dataset)
            ws_kam.cell(row=1, column=col_idx, value=deform.kam_order)
            ws_kam.cell(row=2, column=col_idx, value=deform.kam_threshold)
            ws_kam.cell(row=3, column=col_idx, value=dataset_name)

            # Histogram: the bin centres are midpoints of kam_hist_edges; the counts go to their row on the fixed grid
            bin_edges = np.asarray(deform.kam_hist_edges, dtype=float)
            if len(bin_edges) > 1:
                _place_on_grid(ws_kam, col_idx, KAM_FIRST_ROW, KAM_BIN_CENTERS,
                               (bin_edges[:-1] + bin_edges[1:]) / 2, deform.kam_hist_counts,
                               f"KAM hist (dataset {dataset_name!r})")

        # GOS hist sheet
        if "rx" in results:
            rx = results["rx"]
            ws_gos = self.wb["GOS hist, area weighted"]
            ws_gos.cell(row=1, column=col_idx, value=dataset_name)

            _place_on_grid(ws_gos, col_idx, RX_HIST_FIRST_ROW, GOS_BIN_CENTERS, rx.gos_histogram[1],
                           _area_percent(rx, rx.gos_histogram[0], f"GOS hist (dataset {dataset_name!r})"),
                           f"GOS hist (dataset {dataset_name!r})")

        # gKAM hist sheet
        if "rx" in results:
            rx = results["rx"]
            ws_gkam = self.wb["gKAM - area weighted"]
            ws_gkam.cell(row=1, column=col_idx, value=dataset_name)

            _place_on_grid(ws_gkam, col_idx, RX_HIST_FIRST_ROW, GKAM_BIN_CENTERS, rx.gkam_histogram[1],
                           _area_percent(rx, rx.gkam_histogram[0], f"gKAM hist (dataset {dataset_name!r})"),
                           f"gKAM hist (dataset {dataset_name!r})")

        # Grain Boundaries sheet
        if "deformation" in results:
            deform = results["deformation"]
            ws_gb = self.wb["Grain Boundaries"]
            ws_gb.cell(row=1, column=col_idx, value=dataset_name)

            # Basic GB metrics (rows 3-10)
            ws_gb.cell(row=3, column=col_idx, value=deform.gb_per_phase_area)
            ws_gb.cell(row=4, column=col_idx, value=deform.gb_per_field_area)
            ws_gb.cell(row=5, column=col_idx, value=deform.hagb_per_phase_area)
            ws_gb.cell(row=6, column=col_idx, value=deform.hagb_per_field_area)
            ws_gb.cell(row=7, column=col_idx, value=deform.sagb_per_phase_area)
            ws_gb.cell(row=8, column=col_idx, value=deform.sagb_per_field_area)
            ws_gb.cell(row=9, column=col_idx, value=deform.hagb_length)
            ws_gb.cell(row=10, column=col_idx, value=deform.sagb_length)

            # Sphericity per grain on its own sheet, one column per dataset, one row per grain; a grain
            # without a sphericity (NaN) stays an empty cell so the rows keep the grain order. The mean on
            # this sheet is the mean of exactly the values written there.
            sph = deform.sphericity
            sph = np.asarray(sph if sph is not None else [], dtype=float).reshape(-1)
            finite = np.isfinite(sph)
            ws_sph = self.wb[SPHERICITY_SHEET]
            ws_sph.cell(row=1, column=col_idx, value=dataset_name)
            for i, (value, ok) in enumerate(zip(sph, finite), start=2):
                ws_sph.cell(row=i, column=1, value=i - 1)                 # grain number
                if ok:
                    ws_sph.cell(row=i, column=col_idx, value=float(value))
            if finite.any():
                ws_gb.cell(row=11, column=col_idx, value=float(np.mean(sph[finite])))

            # Segment histogram below the scalar metrics: bin centres in column A, counts in the dataset column
            if deform.gb_seg_hist_counts is not None:
                ws_gb.cell(row=GB_SEG_HEADER_ROW, column=col_idx, value=dataset_name)
                _place_on_grid(ws_gb, col_idx, GB_SEG_FIRST_ROW, GB_SEG_CENTERS, deform.gb_seg_hist_bins,
                               deform.gb_seg_hist_counts, f"Grain Boundaries segment histogram (dataset {dataset_name!r})")

        # SmallVSLargeGrain: texture components of all / substructure / RX grains, one row for the dataset
        self._export_texture_split(dataset_name, results)

    def _export_texture_split(self, dataset_name: str, results: Dict[str, Any]) -> None:
        """Sheet "SmallVSLargeGrain": one row per dataset with the component fractions of all, substructure and RX grains.

        Layout as TextureAnalysis_mtex6.m:707-720: row 1 the block titles, row 2 the component names three times,
        the dataset in row n + 2 with its name in column A. All grains come from ``results["texture"]``, the other
        two blocks from ``results["texture_by_rx"]`` (``calculate_texture_split_by_rx``). A block whose input is
        missing stays empty. A dataset whose components differ from the names already in row 2 is left out, and so
        is a dataset with ``results["texture_by_rx_left_out"]`` (its RX phase is not its texture phase, see
        ``analysis.texture_analysis.split_phase_conflict``): none of its blocks is written.
        """
        if results.get("texture_by_rx_left_out"):
            return
        tx, split = results.get("texture"), results.get("texture_by_rx")
        if tx is None and split is None:
            return
        names = list(tx.component_fractions) if tx is not None else list(split["rx"])
        k = len(names)
        ws = self.wb["SmallVSLargeGrain"]
        expected = names * 3
        if any(ws.cell(row=2, column=2 + j).value not in (None, "", n) for j, n in enumerate(expected)):
            logger.warning("SmallVSLargeGrain: dataset %r has other texture components than the names already in "
                           "row 2 (use one texture preset per workbook); its row is left out", dataset_name)
            return
        row = self.dataset_count + 2
        ws.cell(row=row, column=1, value=dataset_name)
        blocks = []
        if tx is not None:
            blocks.append((0, "all Grains", tx.component_fractions))
        if split is not None:
            blocks.append((1, "Substructure Grains", split["substructure"]))
            blocks.append((2, "RX Grains", split["rx"]))
        for b, title, values in blocks:
            first = 2 + b * k
            ws.cell(row=1, column=first, value=title)
            for j, name in enumerate(names):
                ws.cell(row=2, column=first + j, value=name)
                v = values.get(name)
                if v is not None and np.isfinite(v):
                    ws.cell(row=row, column=first + j, value=float(v))

    def mark_dataset_failed(self, dataset_name: str, note: str = "export failed, column incomplete") -> None:
        """Mark the column of the dataset exported last as incomplete, on every sheet that carries its name.

        Call it after :meth:`export_dataset` raised: the header cells that were written (Sheet1 always, the
        other sheets as far as the export got) change from ``name`` to ``name (note)``.
        """
        col = self.dataset_count + 1
        marked = f"{dataset_name} ({note})"
        self.wb["Sheet1"].cell(row=1, column=col, value=marked)
        for ws in self.wb.worksheets:
            for row in (1, 3, GB_SEG_HEADER_ROW):
                if ws.cell(row=row, column=col).value == dataset_name:
                    ws.cell(row=row, column=col, value=marked)
        if "SmallVSLargeGrain" in self.wb.sheetnames:  # one row per dataset, name in column A
            ws_tx = self.wb["SmallVSLargeGrain"]
            if ws_tx.cell(row=self.dataset_count + 2, column=1).value == dataset_name:
                ws_tx.cell(row=self.dataset_count + 2, column=1, value=marked)
        if "BC fit (3G)" in self.wb.sheetnames:       # this sheet has one row per dataset
            ws_fit = self.wb["BC fit (3G)"]
            if ws_fit.cell(row=self.dataset_count + 1, column=1).value == dataset_name:
                ws_fit.cell(row=self.dataset_count + 1, column=1, value=marked)

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
