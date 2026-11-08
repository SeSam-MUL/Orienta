"""Tests for Excel export module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

try:
    from openpyxl import load_workbook
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import orix  # noqa: F401
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

from analysis.excel_exporter import (
    ExcelExporter,
    export_to_excel,
    export_batch_to_excel,
    excel_column,
)
from analysis.bc_analysis import BCHistogramResult, BCGMMResult
from analysis.grain_analysis import GrainSizeResult
from analysis.deformation_analysis import DeformationResult
from analysis.texture_analysis import TextureResult
from analysis.rx_analysis import RXResult


@pytest.fixture
def temp_excel_file():
    """Create temporary Excel file path."""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def mock_results():
    """Create mock analysis results for testing."""
    return {
        "basic_info": {
            "step_micron": 0.5,
            "dim_x": 200,
            "dim_y": 150,
            "n_grains": 1500,
            "rotation_deg_1": 0.0,
            "rotation_deg_2": 70.0,
            "rotation_deg_3": 0.0,
            "gb_threshold_deg": 4.0,
            "min_pixels": 3,
            "min_intercept_um": 1.5,
            "mean_MaxDimX_um": 5.2,
            "mean_MaxDimY_um": 4.8,
            "mean_AR": 1.15,
            "median_AR": 1.10,
            "grains_above_5um": 850,
            "grains_above_10um": 120,
            "grains_above_15um": 25,
        },
        "bc_histogram": BCHistogramResult(
            bin_edges=np.arange(0, 240, 10),
            bin_centers=np.arange(5, 230, 10),
            counts=np.random.rand(23),
            bin_width=10,
            bc_values=np.random.randint(0, 230, size=1000).astype(float),
        ),
        "bc_gmm": BCGMMResult(
            weights=np.array([0.25, 0.45, 0.30]),
            means=np.array([80.0, 140.0, 190.0]),
            stds=np.array([15.0, 20.0, 18.0]),
            bc_low_center=80.0,
            bc_low_frac=0.25,
            bc_mid_center=140.0,
            bc_mid_frac=0.45,
            bc_high_center=190.0,
            bc_high_frac=0.30,
            n_samples=30000,
            converged=True,
        ),
        "grain_size": GrainSizeResult(
            avg_maxDimX=5.2,
            std_maxDimX=2.1,
            avg_maxDimY=4.8,
            std_maxDimY=1.9,
            avg_ECD=3.5,
            std_ECD=2.1,
            avg_AR=1.15,
            std_AR=0.3,
            area_weighted_ECD=4.2,
            ecd_per_grain=np.random.rand(100) * 10 + 0.5,
            maxDimX_per_grain=np.random.rand(100) * 12 + 0.5,
            maxDimY_per_grain=np.random.rand(100) * 10 + 0.5,
            ar_per_grain=np.random.rand(100) * 2 + 1.0,
        ),
        "deformation": DeformationResult(
            kam=np.random.rand(200, 150) * 2,
            kam_hist_counts=np.random.rand(12),
            kam_hist_edges=np.arange(0, 3.25, 0.25),
            kam_order=2,
            kam_threshold=7.0,
            total_gb_length=2100.7,
            hagb_length=1250.5,
            sagb_length=850.2,
            gb_per_phase_area=0.025,
            gb_per_field_area=0.030,
            hagb_per_phase_area=0.015,
            hagb_per_field_area=0.018,
            sagb_per_phase_area=0.010,
            sagb_per_field_area=0.012,
            gb_seg_hist_counts=np.random.rand(30),
            gb_seg_hist_bins=np.arange(4, 63, 2, dtype=float),      # the 30 bins analyze_deformation gives
            sphericity=np.array([0.85]),
        ),
        "texture": TextureResult(
            component_fractions={
                "Brass": 12.5,
                "S": 18.2,
                "Copper": 8.5,
                "Goss": 3.2,
                "Cube": 5.1,
            },
            plane_111_fraction=45.8,
            plane_100_fraction=25.3,
            plane_110_fraction=18.9,
            texture_index=2.35,
            entropy=0.65,
            tolerance_angle=10.0,
        ),
        "rx": RXResult(
            rx_grain_ids=np.array([5, 12, 23, 45, 67]),
            rx_fraction=35.6,
            g_kam=np.random.rand(100) * 2,
            g_bc=np.random.rand(100) * 200,
            high_bc_fraction=42.3,
            low_gkam_fraction=38.5,
            low_kam_fraction=40.2,
            gos_histogram=(np.random.rand(60), np.arange(0, 15, 0.25) + 0.125),
            gkam_histogram=(np.random.rand(25), np.arange(0, 5, 0.2) + 0.1),
            gos_threshold=1.25,
            gbc_min_fraction=0.7,
            gkam_threshold=0.55,
            min_grain_radius_um=0.4,
            total_grain_area=1000.0,
        ),
        "texture_by_rx": {
            "rx": {"Brass": 10.0, "S": 20.0, "Copper": 5.0, "Goss": None, "Cube": 2.5},
            "substructure": {"Brass": 14.0, "S": 17.0, "Copper": 9.0, "Goss": 4.0, "Cube": 6.0},
            "n_rx_pixels": 120, "n_substructure_pixels": 880,
        },
    }


def test_excel_column():
    """Test Excel column letter conversion."""
    assert excel_column(1) == "A"
    assert excel_column(2) == "B"
    assert excel_column(26) == "Z"
    assert excel_column(27) == "AA"
    assert excel_column(28) == "AB"
    assert excel_column(52) == "AZ"
    assert excel_column(53) == "BA"


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_excel_exporter_initialization(temp_excel_file):
    """Test ExcelExporter initialization creates all sheets."""
    exporter = ExcelExporter(temp_excel_file)
    exporter.save()
    exporter.close()

    # Verify file exists
    assert Path(temp_excel_file).exists()

    # Load and check sheets
    wb = load_workbook(temp_excel_file)
    expected_sheets = [
        "Sheet1",
        "BC hist",
        "BC fit (3G)",
        "BandContrast Group",
        "Grain Size hist",
        "KAM hist",
        "GOS hist, area weighted",
        "gKAM - area weighted",
        "Grain Boundaries",
        "Sphericity per grain",
        "SmallVSLargeGrain",
    ]

    for sheet_name in expected_sheets:
        assert sheet_name in wb.sheetnames, f"Missing sheet: {sheet_name}"
    assert len(wb.sheetnames) == 11

    wb.close()


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_export_single_dataset(temp_excel_file, mock_results):
    """Test exporting single dataset."""
    export_to_excel(temp_excel_file, "Test_Dataset_1", mock_results)

    # Verify export
    wb = load_workbook(temp_excel_file)

    # Check Sheet1 has dataset name in column B
    ws1 = wb["Sheet1"]
    assert ws1.cell(row=1, column=2).value == "Test_Dataset_1"

    # Check basic info values
    assert ws1.cell(row=2, column=2).value == 0.5  # step_micron
    assert ws1.cell(row=3, column=2).value == 200  # dim_x
    assert ws1.cell(row=4, column=2).value == 150  # dim_y

    # Check BC hist sheet
    ws_bc = wb["BC hist"]
    assert ws_bc.cell(row=1, column=2).value == "Test_Dataset_1"

    # Check BC fit sheet
    ws_fit = wb["BC fit (3G)"]
    assert ws_fit.cell(row=2, column=1).value == "Test_Dataset_1"

    wb.close()


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_export_multiple_datasets(temp_excel_file, mock_results):
    """Test batch export of multiple datasets."""
    datasets = {
        "Dataset_A": mock_results,
        "Dataset_B": mock_results,
        "Dataset_C": mock_results,
    }

    export_batch_to_excel(temp_excel_file, datasets)

    # Verify all datasets present
    wb = load_workbook(temp_excel_file)
    ws1 = wb["Sheet1"]

    assert ws1.cell(row=1, column=2).value == "Dataset_A"
    assert ws1.cell(row=1, column=3).value == "Dataset_B"
    assert ws1.cell(row=1, column=4).value == "Dataset_C"

    wb.close()


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_bc_gmm_sorting(temp_excel_file, mock_results):
    """Test that BC GMM components are sorted by mean (low, mid, high)."""
    export_to_excel(temp_excel_file, "Test_Sort", mock_results)

    wb = load_workbook(temp_excel_file)
    ws_fit = wb["BC fit (3G)"]

    # Extract means (columns D, G, J)
    mu1 = ws_fit.cell(row=2, column=4).value
    mu2 = ws_fit.cell(row=2, column=7).value
    mu3 = ws_fit.cell(row=2, column=10).value

    # Should be sorted: mu1 < mu2 < mu3
    assert mu1 < mu2 < mu3

    wb.close()


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_texture_components_export(temp_excel_file, mock_results):
    """Test texture component export to Sheet1."""
    export_to_excel(temp_excel_file, "Test_Texture", mock_results)

    wb = load_workbook(temp_excel_file)
    ws1 = wb["Sheet1"]

    # Check that texture components are written (rows 27+)
    row = 27
    for comp_name in ["Brass", "S", "Copper", "Goss", "Cube"]:
        # Row label should be component name
        assert ws1.cell(row=row, column=1).value == comp_name
        # Value should be float
        val = ws1.cell(row=row, column=2).value
        assert isinstance(val, (int, float))
        assert 0 <= val <= 100
        row += 1

    wb.close()


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_rx_metrics_export(temp_excel_file, mock_results):
    """Test RX metrics export to Sheet1."""
    export_to_excel(temp_excel_file, "Test_RX", mock_results)

    wb = load_workbook(temp_excel_file)
    ws1 = wb["Sheet1"]

    # Check RX metrics (rows 41-44)
    assert ws1.cell(row=41, column=1).value == "RXedGrainFraction"
    assert ws1.cell(row=41, column=2).value == pytest.approx(35.6)

    assert ws1.cell(row=42, column=1).value == "highBC"
    assert ws1.cell(row=43, column=1).value == "lowGKAM"
    assert ws1.cell(row=44, column=1).value == "lowKAM"

    wb.close()


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_surface_plane_export(temp_excel_file, mock_results):
    """Test surface plane fraction export to Sheet1."""
    from analysis.excel_exporter import PLANE_LABELS
    export_to_excel(temp_excel_file, "Test_Planes", mock_results)

    wb = load_workbook(temp_excel_file)
    ws1 = wb["Sheet1"]

    # Check surface plane fractions (rows 45-47)
    assert ws1.cell(row=45, column=1).value == PLANE_LABELS[0]
    assert ws1.cell(row=45, column=2).value == pytest.approx(45.8)

    assert ws1.cell(row=46, column=1).value == PLANE_LABELS[1]
    assert ws1.cell(row=46, column=2).value == pytest.approx(25.3)

    assert ws1.cell(row=47, column=1).value == PLANE_LABELS[2]
    assert ws1.cell(row=47, column=2).value == pytest.approx(18.9)

    wb.close()


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_histogram_sheets_export(temp_excel_file, mock_results):
    """Test histogram data export to dedicated sheets."""
    export_to_excel(temp_excel_file, "Test_Histograms", mock_results)

    wb = load_workbook(temp_excel_file)

    # KAM hist sheet
    ws_kam = wb["KAM hist"]
    assert ws_kam.cell(row=3, column=2).value == "Test_Histograms"
    # Should have bin centers and counts
    assert ws_kam.cell(row=5, column=1).value is not None

    # GOS hist sheet
    ws_gos = wb["GOS hist, area weighted"]
    assert ws_gos.cell(row=1, column=2).value == "Test_Histograms"

    # gKAM hist sheet
    ws_gkam = wb["gKAM - area weighted"]
    assert ws_gkam.cell(row=1, column=2).value == "Test_Histograms"

    wb.close()


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_grain_boundaries_export(temp_excel_file, mock_results):
    """Test grain boundary metrics export."""
    export_to_excel(temp_excel_file, "Test_GB", mock_results)

    wb = load_workbook(temp_excel_file)
    ws_gb = wb["Grain Boundaries"]

    # Check dataset name
    assert ws_gb.cell(row=1, column=2).value == "Test_GB"

    # Check HAGB length
    hagb_val = ws_gb.cell(row=9, column=2).value
    assert hagb_val == pytest.approx(1250.5)

    # Check SAGB length
    sagb_val = ws_gb.cell(row=10, column=2).value
    assert sagb_val == pytest.approx(850.2)

    # Check sphericity
    sph_val = ws_gb.cell(row=11, column=2).value
    assert sph_val == pytest.approx(0.85)

    wb.close()


def test_import_without_openpyxl():
    """Test that module can be imported even without openpyxl."""
    # The import itself is what is tested: the module loads without openpyxl.
    from analysis.excel_exporter import excel_column
    assert excel_column(1) == "A"


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_exporter_requires_openpyxl():
    """Test that ExcelExporter raises error without openpyxl."""
    # Needs openpyxl to be installed; the error path cannot be reached from here without
    # removing the import, so this test only marks the requirement.
    pass


# ---------------------------------------------------------------------------------------------------------
# Cells written twice. One cell must not receive two different values within one export session,
# and a dataset's values must only go into the dataset's own column.
# ---------------------------------------------------------------------------------------------------------

def _two_datasets(first):
    """Two complete result sets that differ everywhere a collision could hide behind equal numbers."""
    import copy
    first = copy.deepcopy(first)
    first["deformation"].sphericity = np.array([0.85, 0.5, np.nan, 0.7])
    second = copy.deepcopy(first)
    rng = np.random.default_rng(11)
    second["basic_info"]["step_micron"] = 0.25
    second["bc_histogram"].counts = rng.random(23)
    second["bc_gmm"].means = np.array([70.0, 130.0, 200.0])
    second["grain_size"].ecd_per_grain = rng.random(80) * 25 + 3.0           # another range, other bin edges
    # KAM edges from another maximum (deformation_analysis builds them from each dataset's own maximum)
    second["deformation"].kam_hist_edges = np.arange(0, 5.25, 0.25)
    second["deformation"].kam_hist_counts = rng.random(20)
    # GB segment bins from another sagb_min (7 deg instead of 3): centres 8, 10, ..., 62
    second["deformation"].gb_seg_hist_bins = np.arange(8, 63, 2, dtype=float)
    second["deformation"].gb_seg_hist_counts = rng.random(28) + 5.0          # none equal to a bin centre
    second["deformation"].hagb_length = 99.5
    second["deformation"].sphericity = np.array([0.4, 0.6])                  # another number of grains
    second["texture"].component_fractions = {k: v + 1.0 for k, v in first["texture"].component_fractions.items()}
    second["texture_by_rx"] = {
        "rx": {k: 3.0 for k in first["texture_by_rx"]["rx"]},
        "substructure": {k: 7.0 + i for i, k in enumerate(first["texture_by_rx"]["substructure"])},
        "n_rx_pixels": 50, "n_substructure_pixels": 400}
    second["rx"].total_grain_area = 2000.0
    second["rx"].gos_histogram = (rng.random(40) + 7.0, np.arange(0, 10, 0.25) + 0.125)     # a shorter range
    second["rx"].gkam_histogram = (rng.random(25) + 7.0, np.arange(0, 5, 0.2) + 0.1)
    return first, second


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


# Sheets whose datasets run down the rows (one row per dataset) instead of across the columns.
_ROW_PER_DATASET = {"BC fit (3G)", "SmallVSLargeGrain"}
# Header cells that sit in a column of their own and are written before any dataset exists.
_HEADER_CELLS = {("BC fit (3G)", 1), ("SmallVSLargeGrain", 1)}


def _record_two_exports(monkeypatch, tmp_path, first, second):
    from openpyxl.worksheet.worksheet import Worksheet
    log, stage = [], ["init"]
    original = Worksheet.cell

    def spy(self, row, column=None, value=None):
        if value is not None:
            log.append((stage[0], self.title, row, column, value))
        return original(self, row, column, value)

    monkeypatch.setattr(Worksheet, "cell", spy)
    exporter = ExcelExporter(str(tmp_path / "guard.xlsx"))
    stage[0] = 1
    exporter.export_dataset("First", first)
    stage[0] = 2
    exporter.export_dataset("Second", second)
    monkeypatch.setattr(Worksheet, "cell", original)
    exporter.save()
    exporter.close()
    return log


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_no_cell_is_written_twice_with_different_values_and_every_value_goes_to_its_own_column(
        monkeypatch, tmp_path, mock_results):
    first, second = _two_datasets(mock_results)
    log = _record_two_exports(monkeypatch, tmp_path, first, second)
    assert len(log) > 500, "the spy saw too little of the export to be a guard"

    seen, collisions = {}, []
    for stage, title, row, column, value in log:
        key = (title, row, column)
        if key in seen and not _same(seen[key][1], value):
            collisions.append(f"{title}!r{row}c{column}: {seen[key][1]!r} (stage {seen[key][0]}) "
                              f"then {value!r} (stage {stage})")
        seen.setdefault(key, (stage, value))
    assert not collisions, "cells written twice with different values:\n" + "\n".join(sorted(set(collisions)))

    misplaced = []
    for stage, title, row, column, value in log:
        if stage == "init":
            if column >= 2 and (title, row) not in _HEADER_CELLS:
                misplaced.append(f"init: {title}!r{row}c{column} = {value!r}")
        elif title not in _ROW_PER_DATASET and column not in (1, stage + 1):
            misplaced.append(f"dataset {stage} (column {stage + 1}): {title}!r{row}c{column} = {value!r}")
    assert not misplaced, "values outside the dataset's column or the label column:\n" + "\n".join(sorted(set(misplaced)))

    # the bin centres of the histogram sheets are written once for the whole export (with the labels), not by
    # every dataset again
    from collections import Counter
    first_rows = {"KAM hist": 5, "GOS hist, area weighted": 3, "gKAM - area weighted": 3,
                  "Grain Boundaries": 14, "Grain Size hist": 3}
    times = Counter((title, row) for stage, title, row, column, value in log
                    if column == 1 and title in first_rows and row >= first_rows[title])
    repeated = {f"{t}!A{r}": n for (t, r), n in times.items() if n != 1}
    assert not repeated, f"centre cells written more than once in two exports: {repeated}"
    assert {t for t, _ in times} == set(first_rows), "a histogram sheet has no centres at all"


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_grain_boundaries_sheet_keeps_labels_bin_centres_and_counts_apart(temp_excel_file, mock_results):
    first, second = _two_datasets(mock_results)
    export_batch_to_excel(temp_excel_file, {"First": first, "Second": second})
    ws = load_workbook(temp_excel_file)["Grain Boundaries"]

    assert [ws.cell(row=r, column=1).value for r in range(3, 12)] == [
        "GBperEBSD Al_area", "GBperfieldArea", "HAGBperEBSD Al_area", "HAGBperarea", "SAGBperEBSD Al_area",
        "SAGBperarea", "HAGBlength_micron", "SAGBlength_micron", "sphericity (mean)"]
    assert (ws.cell(row=1, column=2).value, ws.cell(row=1, column=3).value) == ("First", "Second")
    assert ws.cell(row=9, column=2).value == pytest.approx(1250.5)
    assert ws.cell(row=9, column=3).value == pytest.approx(99.5)
    # the mean is the mean of the values on "Sphericity per grain", NaN grains left out
    assert ws.cell(row=11, column=2).value == pytest.approx(np.mean([0.85, 0.5, 0.7]))
    assert ws.cell(row=11, column=3).value == pytest.approx(0.5)

    # segment histogram below: header row with BinCenters in column A and the dataset names over the counts
    assert ws.cell(row=12, column=1).value == "Misorientation [deg]"
    assert ws.cell(row=13, column=1).value == "BinCenters"
    assert (ws.cell(row=13, column=2).value, ws.cell(row=13, column=3).value) == ("First", "Second")
    for k in range(31):
        assert ws.cell(row=14 + k, column=1).value == pytest.approx(4.0 + 2.0 * k)      # the same for both datasets
    for k, count in enumerate(first["deformation"].gb_seg_hist_counts):
        assert ws.cell(row=14 + k, column=2).value == pytest.approx(float(count))
    for k, count in enumerate(second["deformation"].gb_seg_hist_counts):
        assert ws.cell(row=14 + 2 + k, column=3).value == pytest.approx(float(count))   # centre 8 is the third bin
    assert ws.cell(row=14, column=3).value is None and ws.cell(row=15, column=3).value is None
    assert ws.cell(row=13, column=4).value is None


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_sphericity_per_grain_sheet_holds_the_vector_the_mean_is_made_of(temp_excel_file, mock_results):
    first, second = _two_datasets(mock_results)
    export_batch_to_excel(temp_excel_file, {"First": first, "Second": second})
    wb = load_workbook(temp_excel_file)
    ws = wb["Sphericity per grain"]

    assert (ws.cell(row=1, column=2).value, ws.cell(row=1, column=3).value) == ("First", "Second")
    # one column per dataset, one row per grain; a grain without a sphericity (NaN) is an empty cell
    assert [ws.cell(row=r, column=2).value for r in range(2, 7)] == [0.85, 0.5, None, 0.7, None]
    assert [ws.cell(row=r, column=3).value for r in range(2, 5)] == [0.4, 0.6, None]
    assert ws.cell(row=1, column=1).value == "grain"
    assert [ws.cell(row=r, column=1).value for r in range(2, 7)] == [1, 2, 3, 4, None]      # grain number
    for col in (2, 3):
        written = [ws.cell(row=r, column=col).value for r in range(2, 50)]
        written = [v for v in written if v is not None]
        mean_on_gb = wb["Grain Boundaries"].cell(row=11, column=col).value
        assert mean_on_gb == pytest.approx(float(np.mean(written)), rel=1e-12)


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_a_dataset_without_grains_has_no_sphericity_mean_and_no_column_values(
        temp_excel_file, mock_results, monkeypatch):
    from openpyxl.worksheet.worksheet import Worksheet
    mock_results["deformation"].sphericity = np.array([], dtype=float)
    writes, original = [], Worksheet.cell

    def spy(self, row, column=None, value=None):
        if value is not None:
            writes.append((self.title, row, column))
        return original(self, row, column, value)

    monkeypatch.setattr(Worksheet, "cell", spy)
    export_to_excel(temp_excel_file, "Empty", mock_results)
    monkeypatch.setattr(Worksheet, "cell", original)
    assert ("Grain Boundaries", 11, 2) not in writes                  # not even an empty string or a NaN
    wb = load_workbook(temp_excel_file)
    assert wb["Grain Boundaries"].cell(row=11, column=2).value is None
    assert wb["Sphericity per grain"].cell(row=1, column=2).value == "Empty"
    assert wb["Sphericity per grain"].cell(row=2, column=2).value is None


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_grain_size_hist_has_the_same_bin_centres_for_every_dataset(temp_excel_file, mock_results):
    # the reference script bins every dataset on 0:5:140 so the columns can be compared row by row
    first, second = _two_datasets(mock_results)
    export_batch_to_excel(temp_excel_file, {"First": first, "Second": second})
    ws = load_workbook(temp_excel_file)["Grain Size hist"]
    centres = [ws.cell(row=r, column=1).value for r in range(3, 31)]
    assert centres == [2.5 + 5.0 * k for k in range(28)]
    for col, res in ((2, first), (3, second)):
        counts = [ws.cell(row=r, column=col).value for r in range(3, 31)]
        expected, _ = np.histogram(res["grain_size"].ecd_per_grain, bins=np.arange(0, 145, 5))
        assert counts == [float(c) for c in expected]
    assert ws.cell(row=2, column=2).value is None      # no label inside a dataset's column


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_stale_labels_of_a_workbook_in_the_current_layout_are_refreshed_by_the_next_export(temp_excel_file, mock_results):
    from analysis.excel_exporter import GRAIN_STATS_KEYS, MIN_INTERCEPT_LABEL
    export_to_excel(temp_excel_file, "Old", mock_results)
    wb = load_workbook(temp_excel_file)
    ws1 = wb["Sheet1"]
    ws1.cell(row=5, column=1, value="n_grains")                    # labels as an older version wrote them
    ws1.cell(row=11, column=1, value="min_intercept_um")
    for i in range(49, 60):
        ws1.cell(row=i, column=1, value=f"stale_{i}")
    wb["Grain Boundaries"].cell(row=11, column=1, value="sphericity")
    wb.save(temp_excel_file)
    wb.close()

    exporter = ExcelExporter(temp_excel_file)                       # reuse the file
    exporter.export_dataset("New", mock_results)
    exporter.save()
    exporter.close()

    wb = load_workbook(temp_excel_file)
    ws1 = wb["Sheet1"]
    assert ws1.cell(row=5, column=1).value == "n_grains (all phases)"
    assert ws1.cell(row=11, column=1).value == MIN_INTERCEPT_LABEL
    assert [ws1.cell(row=i, column=1).value for i in range(49, 49 + len(GRAIN_STATS_KEYS))] == list(GRAIN_STATS_KEYS)
    assert wb["Grain Boundaries"].cell(row=11, column=1).value == "sphericity (mean)"
    assert wb["Sphericity per grain"].cell(row=1, column=2).value == "New"


# ---------------------------------------------------------------------------------------------------------
# Every histogram sheet has fixed bin centres; counts go into the row of their bin.
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_histogram_sheets_have_one_set_of_centres_and_counts_in_the_row_of_their_bin(temp_excel_file, mock_results):
    first, second = _two_datasets(mock_results)
    export_batch_to_excel(temp_excel_file, {"First": first, "Second": second})
    wb = load_workbook(temp_excel_file)

    ws = wb["KAM hist"]
    assert [ws.cell(row=5 + k, column=1).value for k in (0, 1, 59)] == [0.125, 0.375, 14.875]
    for col, res in ((2, first), (3, second)):
        counts = res["deformation"].kam_hist_counts
        assert [ws.cell(row=5 + k, column=col).value for k in range(len(counts))] == pytest.approx([float(c) for c in counts])
        assert ws.cell(row=5 + len(counts), column=col).value is None

    ws = wb["Grain Boundaries"]
    assert [ws.cell(row=14 + k, column=1).value for k in (0, 1, 30)] == [4.0, 6.0, 64.0]

    ws = wb["GOS hist, area weighted"]
    assert [ws.cell(row=3 + k, column=1).value for k in (0, 59)] == [0.125, 14.875]
    second_gos = second["rx"].gos_histogram[0] * 100.0 / 2000.0      # percent of the total grain area
    first_gos = first["rx"].gos_histogram[0] * 100.0 / 1000.0
    assert [ws.cell(row=3 + k, column=3).value for k in range(40)] == pytest.approx([float(c) for c in second_gos])
    assert ws.cell(row=3 + 40, column=3).value is None                    # the shorter range stays short
    assert [ws.cell(row=3 + k, column=2).value for k in range(60)] == pytest.approx([float(c) for c in first_gos])

    ws = wb["gKAM - area weighted"]
    assert [ws.cell(row=3 + k, column=1).value for k in (0, 24)] == pytest.approx([0.1, 4.9])
    assert [ws.cell(row=3 + k, column=3).value for k in range(25)] == pytest.approx(
        [float(c) * 100.0 / 2000.0 for c in second["rx"].gkam_histogram[0]])
    assert ws.cell(row=2, column=1).value == "BinCenters [deg]; values: % of the total grain area"


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_bins_that_are_not_on_the_fixed_grid_are_left_out_with_a_warning(temp_excel_file, mock_results, caplog):
    import logging
    # sagb_min of 2 deg gives centres 3, 5, ... (off the grid 4, 6, ...); a KAM bin width of 0.1 does not fit either
    mock_results["deformation"].gb_seg_hist_bins = np.arange(3, 62, 2, dtype=float)
    mock_results["deformation"].gb_seg_hist_counts = np.full(30, 2.0)
    mock_results["deformation"].kam_hist_edges = np.arange(0, 1.05, 0.1)
    mock_results["deformation"].kam_hist_counts = np.full(10, 0.1)
    with caplog.at_level(logging.WARNING, logger="analysis.excel_exporter"):
        export_to_excel(temp_excel_file, "Off", mock_results)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "Grain Boundaries segment histogram" in messages and "30 of 30 bins" in messages
    assert "KAM hist" in messages and "10 of 10 bins" in messages
    wb = load_workbook(temp_excel_file)
    assert all(wb["Grain Boundaries"].cell(row=r, column=2).value in (None, "Off") for r in range(14, 45))
    assert all(wb["KAM hist"].cell(row=r, column=2).value is None for r in range(5, 65))


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_ecd_above_the_last_bin_is_reported_with_its_count(temp_excel_file, mock_results, caplog):
    import logging
    mock_results["grain_size"].ecd_per_grain = np.array([3.0, 8.0, 141.0, 150.0, 300.0])
    with caplog.at_level(logging.WARNING, logger="analysis.excel_exporter"):
        export_to_excel(temp_excel_file, "Big", mock_results)
    assert any("3 of 5 grains have an ECD above 140" in r.getMessage() for r in caplog.records)
    ws = load_workbook(temp_excel_file)["Grain Size hist"]
    assert sum(ws.cell(row=r, column=2).value for r in range(3, 31)) == 2.0
    assert ws.cell(row=2, column=1).value == "BinCenters (ECD [µm]), counts"


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_a_workbook_in_the_old_layout_is_refused_and_left_untouched(temp_excel_file, mock_results):
    from openpyxl import Workbook
    # as an earlier layout wrote it: no sheet "Sphericity per grain", bin centres of Grain Boundaries in column B
    old = Workbook()
    ws1 = old.active
    ws1.title = "Sheet1"
    ws1["B1"], ws1["C1"] = "OldFirst", "OldSecond"
    gb = old.create_sheet("Grain Boundaries")
    gb["B11"], gb["B12"], gb["C12"] = "BinCenters", 4.0, 99.0
    old.save(temp_excel_file)
    old.close()
    before = Path(temp_excel_file).read_bytes()

    with pytest.raises(ValueError) as err:
        ExcelExporter(temp_excel_file)
    assert temp_excel_file in str(err.value) and "older layout" in str(err.value) and "new file name" in str(err.value)
    with pytest.raises(ValueError, match="new file name"):
        export_to_excel(temp_excel_file, "New", mock_results)
    assert Path(temp_excel_file).read_bytes() == before                  # nothing was written into it


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_a_workbook_in_the_current_layout_warns_that_the_first_dataset_column_is_overwritten_and_keeps_old_cells(
        temp_excel_file, mock_results, caplog):
    import logging
    first, second = _two_datasets(mock_results)
    export_batch_to_excel(temp_excel_file, {"OldFirst": first, "OldSecond": second})
    wb = load_workbook(temp_excel_file)
    wb["Grain Size hist"]["B2"] = "ECD [µm]"                      # cells a longer earlier run left behind
    wb["Grain Boundaries"]["A60"] = "old tail"
    old_second_counts = [wb["Grain Boundaries"].cell(row=r, column=3).value for r in range(14, 45)]
    wb.save(temp_excel_file)
    wb.close()

    with caplog.at_level(logging.WARNING, logger="analysis.excel_exporter"):
        exporter = ExcelExporter(temp_excel_file)
    assert any("Sheet1!B1" in r.getMessage() and "OldFirst" in r.getMessage() for r in caplog.records)
    exporter.export_dataset("New", mock_results)
    exporter.save()
    exporter.close()

    wb = load_workbook(temp_excel_file)
    assert wb["Sheet1"]["B1"].value == "New"                        # overwritten: the first dataset column
    assert wb["Sheet1"]["C1"].value == "OldSecond"                  # stays: what this export does not write
    assert [wb["Grain Boundaries"].cell(row=r, column=3).value for r in range(14, 45)] == old_second_counts
    assert wb["Grain Size hist"]["B2"].value == "ECD [µm]"
    assert wb["Grain Boundaries"]["A60"].value == "old tail"


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_a_fresh_workbook_does_not_warn_about_overwriting(temp_excel_file, mock_results, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="analysis.excel_exporter"):
        export_batch_to_excel(temp_excel_file, {"A": mock_results, "B": mock_results})
    assert not [r for r in caplog.records if "overwritten" in r.getMessage()]


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_another_texture_preset_in_the_same_workbook_is_reported(temp_excel_file, mock_results, caplog, tmp_path):
    import copy
    import logging
    other = copy.deepcopy(mock_results)
    other["texture"].component_fractions = {"Goss": 1.0, "Rot.Goss": 2.0, "Cube": 3.0, "Brass": 4.0, "S": 5.0}
    with caplog.at_level(logging.WARNING, logger="analysis.excel_exporter"):
        export_batch_to_excel(temp_excel_file, {"A": mock_results, "B": other})
    texture = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Sheet1:")
               and "texture components" in r.getMessage()]
    assert len(texture) == 1 and "dataset 'B'" in texture[0] and "row 27 'Brass' -> 'Goss'" in texture[0]
    split = [r.getMessage() for r in caplog.records if r.getMessage().startswith("SmallVSLargeGrain:")]
    assert len(split) == 1 and "dataset 'B'" in split[0] and "left out" in split[0]
    # the same preset twice is silent
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="analysis.excel_exporter"):
        export_batch_to_excel(str(tmp_path / "same.xlsx"), {"A": mock_results, "B": mock_results})
    assert not [r for r in caplog.records if "texture components" in r.getMessage()]


# ---------------------------------------------------------------------------------------------------------
# Counts survive the fixed grids end to end; a broken histogram result fails loudly.
# ---------------------------------------------------------------------------------------------------------

def _four_grain_map():
    """A 40 x 40 map of four grains (0, 12, 35 and 50 deg about z) with a little orientation scatter inside.

    The grains meet in boundaries of several misorientation angles, and the scatter gives every grain a GOS.
    """
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    from analysis.ebsd_dataset import EBSDDataset

    rng = np.random.default_rng(3)
    n = 40
    yy, xx = np.mgrid[0:n, 0:n]
    angle = np.where(xx < 20, np.where(yy < 20, 0.0, 12.0), np.where(yy < 20, 35.0, 50.0))
    angle = angle + rng.normal(0, 0.6, angle.shape) + (xx % 7) * 0.2
    rotations = Rotation.from_axes_angles([0, 0, 1], np.deg2rad(angle.ravel()))
    xmap = CrystalMap(rotations=rotations, phase_id=np.zeros(n * n, dtype=int),
                      x=np.tile(np.arange(n, dtype=float), n) * 0.5,
                      y=np.repeat(np.arange(n, dtype=float), n) * 0.5,
                      phase_list=PhaseList(names=["Al"], point_groups=["m-3m"]),
                      prop={"bc": 100 + rng.random(n * n) * 100})
    return EBSDDataset(xmap, step_size=0.5)


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_real_analysis_histograms_arrive_on_the_sheets_with_nothing_left_out(temp_excel_file, caplog):
    import logging
    from analysis.deformation_analysis import analyze_deformation
    from analysis.grain_analysis import reconstruct_grains
    from analysis.rx_analysis import analyze_recrystallization

    dataset = _four_grain_map()
    grains = reconstruct_grains(dataset, angle_threshold_deg=5.0, min_pixels=3)
    assert grains.n_grains == 4
    deform = analyze_deformation(dataset, grains)
    rx = analyze_recrystallization(dataset, grains, deform.kam)
    # the map really has what the sheets are to carry: boundaries in several bins, a GOS, a KAM
    assert len(deform.gb_seg_hist_bins) == 30 and np.count_nonzero(deform.gb_seg_hist_counts) >= 4
    assert rx.gos_histogram[0].sum() > 0 and rx.gkam_histogram[0].sum() > 0 and deform.kam_hist_counts.sum() > 0

    with caplog.at_level(logging.WARNING, logger="analysis.excel_exporter"):
        export_to_excel(temp_excel_file, "Real", {"deformation": deform, "rx": rx})
    assert not [r for r in caplog.records if "fixed bin grid" in r.getMessage()]

    wb = load_workbook(temp_excel_file)

    def column_sum(sheet, first_row, last_row):
        return sum(wb[sheet].cell(row=r, column=2).value or 0.0 for r in range(first_row, last_row + 1))

    assert column_sum("Grain Boundaries", 14, 44) == pytest.approx(float(deform.gb_seg_hist_counts.sum()))
    assert column_sum("KAM hist", 5, 64) == pytest.approx(float(deform.kam_hist_counts.sum()))
    assert rx.total_grain_area == 400.0              # 1600 cells of 0.25 um^2
    assert column_sum("GOS hist, area weighted", 3, 62) == pytest.approx(
        100.0 * float(rx.gos_histogram[0].sum()) / rx.total_grain_area)
    assert column_sum("gKAM - area weighted", 3, 27) == pytest.approx(
        100.0 * float(rx.gkam_histogram[0].sum()) / rx.total_grain_area)
    # and each count is in the row of its own bin: the sheet's centre is the result's centre
    ws = wb["Grain Boundaries"]
    for centre, count in zip(deform.gb_seg_hist_bins, deform.gb_seg_hist_counts):
        row = 14 + int(round((centre - 4.0) / 2.0))
        assert ws.cell(row=row, column=1).value == pytest.approx(float(centre))
        assert (ws.cell(row=row, column=2).value or 0.0) == pytest.approx(float(count))


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_a_nan_bin_centre_fails_loudly_with_the_sheet_named(temp_excel_file, mock_results):
    bins = np.array(mock_results["deformation"].gb_seg_hist_bins, dtype=float)
    bins[3] = np.nan
    mock_results["deformation"].gb_seg_hist_bins = bins
    with pytest.raises(ValueError, match=r"Grain Boundaries segment histogram.*1 of 30 bin centres are NaN"):
        export_to_excel(temp_excel_file, "Broken", mock_results)


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_centres_and_counts_of_different_length_fail_loudly_instead_of_being_cut(temp_excel_file, mock_results):
    mock_results["rx"].gos_histogram = (np.random.rand(59), mock_results["rx"].gos_histogram[1])
    with pytest.raises(ValueError, match=r"GOS hist.*60 bin centres but 59 counts"):
        export_to_excel(temp_excel_file, "Broken", mock_results)


# ---------------------------------------------------------------------------------------------------------
# SmallVSLargeGrain is filled; GOS/gKAM carry percent of the grain area; real texture and BC sheets.
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_small_vs_large_grain_has_one_row_per_dataset_with_three_blocks(temp_excel_file, mock_results):
    first, second = _two_datasets(mock_results)
    export_batch_to_excel(temp_excel_file, {"First": first, "Second": second})
    ws = load_workbook(temp_excel_file)["SmallVSLargeGrain"]
    names = ["Brass", "S", "Copper", "Goss", "Cube"]
    # row 1: block titles; row 2: the component names three times (TextureAnalysis_mtex6.m:710-720)
    assert (ws["B1"].value, ws["G1"].value, ws["L1"].value) == ("all Grains", "Substructure Grains", "RX Grains")
    assert [ws.cell(row=2, column=c).value for c in range(2, 17)] == names * 3
    # the dataset n sits in row n + 2, its name in column A
    assert (ws["A3"].value, ws["A4"].value) == ("First", "Second")
    assert [ws.cell(row=3, column=c).value for c in range(2, 7)] == pytest.approx(
        [first["texture"].component_fractions[n] for n in names])
    assert [ws.cell(row=3, column=c).value for c in range(7, 12)] == pytest.approx(
        [first["texture_by_rx"]["substructure"][n] for n in names])
    rx_row = [ws.cell(row=3, column=c).value for c in range(12, 17)]
    assert rx_row[:3] == pytest.approx([10.0, 20.0, 5.0]) and rx_row[4] == pytest.approx(2.5)
    assert rx_row[3] is None                                           # a group without pixels: empty, not 0
    assert [ws.cell(row=4, column=c).value for c in range(12, 17)] == [3.0] * 5


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_small_vs_large_grain_without_the_split_writes_only_the_all_grains_block(temp_excel_file, mock_results):
    del mock_results["texture_by_rx"]
    export_to_excel(temp_excel_file, "Only", mock_results)
    ws = load_workbook(temp_excel_file)["SmallVSLargeGrain"]
    assert ws["B1"].value == "all Grains" and ws["G1"].value is None and ws["L1"].value is None
    assert ws["A3"].value == "Only" and ws["B3"].value is not None and ws["G3"].value is None


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_gos_and_gkam_values_are_percent_of_the_total_grain_area(temp_excel_file, mock_results):
    mock_results["rx"].gos_histogram = (np.array([30.0, 10.0] + [0.0] * 58), mock_results["rx"].gos_histogram[1])
    mock_results["rx"].total_grain_area = 80.0                        # an area of 40 of the 80 is in the GOS range
    export_to_excel(temp_excel_file, "Pct", mock_results)
    ws = load_workbook(temp_excel_file)["GOS hist, area weighted"]
    assert (ws["B3"].value, ws["B4"].value) == (pytest.approx(37.5), pytest.approx(12.5))   # 30/80, 10/80 in %


@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_a_result_without_the_total_grain_area_falls_back_to_the_binned_area_with_a_warning(
        temp_excel_file, mock_results, caplog):
    import logging
    mock_results["rx"].total_grain_area = None
    mock_results["rx"].gos_histogram = (np.array([30.0, 10.0] + [0.0] * 58), mock_results["rx"].gos_histogram[1])
    with caplog.at_level(logging.WARNING, logger="analysis.excel_exporter"):
        export_to_excel(temp_excel_file, "NoTotal", mock_results)
    ws = load_workbook(temp_excel_file)["GOS hist, area weighted"]
    assert (ws["B3"].value, ws["B4"].value) == (pytest.approx(75.0), pytest.approx(25.0))
    assert any("does not carry the total grain area" in r.getMessage() for r in caplog.records)


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not available")
def test_texture_split_equals_the_whole_when_one_group_is_everything_and_the_pixel_mask_selects():
    from analysis.texture_analysis import (calculate_texture_component_fractions,
                                           calculate_texture_split_by_rx)
    from analysis.grain_analysis import reconstruct_grains
    from analysis.texture_components import get_texture_preset

    dataset = _four_grain_map()
    grains = reconstruct_grains(dataset, angle_threshold_deg=5.0, min_pixels=3)
    # three components keep the run short; the code path is the same for all 14
    comps = {k: v for k, v in get_texture_preset("FCC_Rolling").items() if k in ("Brass", "Goss", "Cube")}
    assert len(comps) == 3
    whole = calculate_texture_component_fractions(dataset, comps, 10.0)
    everyone = calculate_texture_split_by_rx(dataset, grains, np.arange(1, grains.n_grains + 1), comps, 10.0)
    assert everyone["rx"] == pytest.approx(whole) and everyone["n_rx_pixels"] == 1600
    assert everyone["n_substructure_pixels"] == 0
    assert set(everyone["substructure"].values()) == {None}          # an empty group is None, not 0 %

    # two grains as RX: each group's fractions are those of its own pixels
    ids = np.asarray(grains.grain_ids).reshape(-1)
    rx_ids = [1, 2]
    split = calculate_texture_split_by_rx(dataset, grains, rx_ids, comps, 10.0)
    rx_mask = np.isin(ids, rx_ids)
    assert split["n_rx_pixels"] == int(rx_mask.sum()) and split["n_substructure_pixels"] == int((~rx_mask).sum())
    assert split["rx"] == pytest.approx(calculate_texture_component_fractions(dataset, comps, 10.0, pixel_mask=rx_mask))
    # independent anchor: the pixel-weighted mean of the two groups is the whole
    for name in comps:
        mixed = (split["rx"][name] * split["n_rx_pixels"] + split["substructure"][name] * split["n_substructure_pixels"]) / 1600.0
        assert mixed == pytest.approx(whole[name], abs=1e-9)
    with pytest.raises(ValueError, match="pixel_mask"):
        calculate_texture_component_fractions(dataset, comps, 10.0, pixel_mask=np.ones(5, dtype=bool))


@pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not available")
@pytest.mark.skipif(not OPENPYXL_AVAILABLE, reason="openpyxl not installed")
def test_every_result_kind_from_the_real_analysis_arrives_on_its_sheet(temp_excel_file, caplog):
    import logging
    from analysis.bc_analysis import calculate_bc_histogram, fit_bc_gmm
    from analysis.deformation_analysis import analyze_deformation
    from analysis.grain_analysis import analyze_grain_size, reconstruct_grains
    from analysis.rx_analysis import analyze_recrystallization
    from analysis.texture_analysis import analyze_texture, calculate_texture_split_by_rx
    from analysis.texture_components import get_texture_preset

    dataset = _four_grain_map()
    grains = reconstruct_grains(dataset, angle_threshold_deg=5.0, min_pixels=3)
    # three components keep the run short; the code path is the same for all 14
    comps = {k: v for k, v in get_texture_preset("FCC_Rolling").items() if k in ("Brass", "Goss", "Cube")}
    assert len(comps) == 3
    deform = analyze_deformation(dataset, grains)
    rx = analyze_recrystallization(dataset, grains, deform.kam)
    gs = analyze_grain_size(dataset, grains)
    bc_hist, bc_gmm = calculate_bc_histogram(dataset), fit_bc_gmm(dataset)
    tx = analyze_texture(dataset, comps)
    by_rx = calculate_texture_split_by_rx(dataset, grains, np.array([1, 2]), comps, tx.tolerance_angle)
    results = {"grain_size": gs, "bc_histogram": bc_hist, "bc_gmm": bc_gmm, "deformation": deform, "rx": rx,
               "texture": tx, "texture_by_rx": by_rx}
    assert rx.total_grain_area == pytest.approx(float(np.sum(grains.area))) == 400.0

    with caplog.at_level(logging.WARNING, logger="analysis.excel_exporter"):
        export_to_excel(temp_excel_file, "Real", results)
    assert not [r for r in caplog.records if r.name == "analysis.excel_exporter"]
    wb = load_workbook(temp_excel_file)

    def column(sheet, first_row, last_row, col=2):
        return [wb[sheet].cell(row=r, column=col).value or 0.0 for r in range(first_row, last_row + 1)]

    # GOS / gKAM: the share of the total grain area in percent, so the occupied bins add up to 100
    assert sum(column("GOS hist, area weighted", 3, 62)) == pytest.approx(100.0)
    assert sum(column("gKAM - area weighted", 3, 27)) == pytest.approx(100.0)
    # Grain Size hist: one count per grain; BC hist: the counts of the result
    assert sum(column("Grain Size hist", 3, 30)) == pytest.approx(float(gs.ecd_per_grain.size))
    assert column("BC hist", 2, 24) == pytest.approx([float(c) for c in bc_hist.counts])
    # BC fit (3G): sorted by mean, one row for the dataset
    ws = wb["BC fit (3G)"]
    assert ws.cell(row=2, column=1).value == "Real" and ws.cell(row=2, column=2).value == bc_gmm.n_samples
    assert [ws.cell(row=2, column=c).value for c in (4, 7, 10)] == pytest.approx(sorted(float(m) for m in bc_gmm.means))
    # texture on Sheet1 and on SmallVSLargeGrain
    ws1 = wb["Sheet1"]
    for i, (name, frac) in enumerate(tx.component_fractions.items(), start=27):
        assert ws1.cell(row=i, column=1).value == name
        assert ws1.cell(row=i, column=2).value == pytest.approx(frac)
    ws = wb["SmallVSLargeGrain"]
    k = len(comps)
    for j, name in enumerate(comps):
        assert ws.cell(row=2, column=2 + j).value == name == ws.cell(row=2, column=2 + k + j).value
        assert ws.cell(row=3, column=2 + j).value == pytest.approx(tx.component_fractions[name])
        for block, key in ((1, "substructure"), (2, "rx")):
            expected = by_rx[key][name]
            assert ws.cell(row=3, column=2 + block * k + j).value == (pytest.approx(expected) if expected is not None else None)
    assert ws.cell(row=3, column=1).value == "Real"
