"""The workbook's "Citations" sheet: written by the single export and the batch once grains exist, the eleven
other sheets unchanged by it, added to a workbook that was written before it existed."""
import numpy as np
import openpyxl
import pytest

from analysis.ebsd_dataset import EBSDDataset
from analysis.excel_exporter import ExcelExporter, export_to_excel
from analysis.grain_analysis import reconstruct_grains
from tests.batch_load_stub import patch_batch_load
from tests.synthetic_grain_map import make_map, to_xmap

ELEVEN = ["Sheet1", "BC hist", "BC fit (3G)", "BandContrast Group", "Grain Size hist", "KAM hist",
          "GOS hist, area weighted", "gKAM - area weighted", "Grain Boundaries", "Sphericity per grain",
          "SmallVSLargeGrain"]

# A citation answer as GET /api/citations/analysis gives it (its keys), with texts typed here.
CITATIONS = {
    "bibtex": "@article{one,\n  title = {First}\n}\n\n@misc{two,\n  title = {Second}\n}\n",
    "methods": "Grains were reconstructed with a test sentence.",
    "plain": "Author, A. (2011). First.\nAuthor, B. (2010). Second.\n",
    "steps": [{"key": "analysis.grain_reconstruction", "params": {}}],
    "undeclared": [],
    "warnings": [],
    "app_version": {"version": "test"},
}


def _xmap(shape=(40, 50), n_seeds=14):
    m = make_map(shape=shape, n_seeds=n_seeds)
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.random.default_rng(0).uniform(60, 200, xmap.size)
    return xmap


def _sheets(path):
    wb = openpyxl.load_workbook(path)
    return {ws.title: {(c.row, c.column): c.value for row in ws.iter_rows() for c in row if c.value not in (None, "")}
            for ws in wb.worksheets}


def _basic():
    return {"basic_info": {"step_micron": 0.5, "dim_x": 50, "dim_y": 40, "n_grains": 7}}


def test_without_citations_the_workbook_keeps_its_eleven_sheets(tmp_path):
    path = str(tmp_path / "w.xlsx")
    export_to_excel(path, "scan", _basic())
    assert openpyxl.load_workbook(path).sheetnames == ELEVEN


def test_the_sheet_holds_methods_references_and_bibtex(tmp_path):
    path = str(tmp_path / "w.xlsx")
    export_to_excel(path, "scan", {**_basic(), "citations": CITATIONS})
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ELEVEN + ["Citations"]
    ws = wb["Citations"]
    assert [ws.cell(row=1, column=c).value for c in (1, 2, 4, 6)] == [
        "Dataset", "Methods text (with the settings of this dataset)", "References", "BibTeX"]
    assert ws["A2"].value == "scan"
    assert ws["B2"].value == "Grains were reconstructed with a test sentence."
    assert [ws["D2"].value, ws["D3"].value, ws["D4"].value] == ["Author, A. (2011). First.", "Author, B. (2010). Second.", None]
    assert ws["F2"].value == "@article{one,\n  title = {First}\n}"
    assert ws["F3"].value == "@misc{two,\n  title = {Second}\n}"
    assert ws["F4"].value is None


def test_the_eleven_sheets_are_the_same_with_and_without_the_citations(tmp_path):
    a, b = str(tmp_path / "a.xlsx"), str(tmp_path / "b.xlsx")
    export_to_excel(a, "scan", _basic())
    export_to_excel(b, "scan", {**_basic(), "citations": CITATIONS})
    sa, sb = _sheets(a), _sheets(b)
    assert list(sa) == ELEVEN
    for name in ELEVEN:
        assert sa[name] == sb[name], name


def test_one_row_per_dataset_and_the_failed_mark(tmp_path):
    ex = ExcelExporter(str(tmp_path / "w.xlsx"))
    ex.export_dataset("first", {**_basic(), "citations": CITATIONS})
    ex.export_dataset("second", {**_basic(), "citations": {**CITATIONS, "methods": "Other settings."}})
    ws = ex.wb["Citations"]
    assert [ws["A2"].value, ws["B2"].value] == ["first", "Grains were reconstructed with a test sentence."]
    assert [ws["A3"].value, ws["B3"].value] == ["second", "Other settings."]
    ex.mark_dataset_failed("second")
    assert ws["A3"].value == "second (export failed, column incomplete)"


def test_a_dataset_whose_later_sheet_fails_still_has_its_citation_row(tmp_path):
    ex = ExcelExporter(str(tmp_path / "w.xlsx"))
    broken = object()                     # no attribute of a deformation result: the Sheet1 KAM row raises
    with pytest.raises(AttributeError):
        ex.export_dataset("half", {**_basic(), "citations": CITATIONS, "deformation": broken})
    ex.mark_dataset_failed("half")
    ws = ex.wb["Citations"]
    assert ws["A2"].value == "half (export failed, column incomplete)"
    assert ws["B2"].value == "Grains were reconstructed with a test sentence."


def test_a_workbook_without_the_sheet_gets_it_at_the_next_export(tmp_path):
    path = str(tmp_path / "w.xlsx")
    export_to_excel(path, "old", _basic())                       # written as before: eleven sheets
    assert "Citations" not in openpyxl.load_workbook(path).sheetnames
    ex = ExcelExporter(path)                                     # loaded, not refused
    ex.export_dataset("new", {**_basic(), "citations": CITATIONS})
    ex.save()
    ex.close()
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ELEVEN + ["Citations"]
    assert wb["Citations"]["A2"].value == "new"


def test_a_shorter_reference_list_does_not_leave_old_rows(tmp_path):
    ex = ExcelExporter(str(tmp_path / "w.xlsx"))
    ex.export_dataset("a", {**_basic(), "citations": CITATIONS})
    ex.export_dataset("b", {**_basic(), "citations": {**CITATIONS, "plain": "Only one.\n",
                                                       "bibtex": "@misc{only,\n  title = {One}\n}\n"}})
    ws = ex.wb["Citations"]
    assert [ws["D2"].value, ws["D3"].value] == ["Only one.", None]
    assert [ws["F2"].value, ws["F3"].value] == ["@misc{only,\n  title = {One}\n}", None]


def test_a_bibliography_that_could_not_be_loaded_is_said_on_the_sheet(tmp_path):
    ex = ExcelExporter(str(tmp_path / "w.xlsx"))
    ex.export_dataset("a", {**_basic(), "citations": {**CITATIONS, "plain": "", "bibtex": "",
                                                       "warnings": ["The bibliography could not be loaded."]}})
    assert ex.wb["Citations"]["D2"].value == "The bibliography could not be loaded."


# --------------------------------------------------------------------------------------------------
# the page's export and the batch fill it from the real reconstruction settings
# --------------------------------------------------------------------------------------------------
def test_the_page_export_writes_the_settings_the_grains_were_made_with(tmp_path):
    import backend.api.routes.analysis as ar
    ds = EBSDDataset(_xmap(), step_size=0.5)
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=6.5, min_pixels=4, alpha=1.5, fill=False)
    ds.grain_recon_params = {"gb_threshold_deg": 6.5, "min_pixels": 4, "alpha": 1.5, "min_intercept_um": None,
                             "fill": False}
    saved = ar._dataset
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    ar._grain_stats_settings.update(edge_rule="keep_all", ecd_basis="measured_pixels")
    try:
        warnings = []
        results = ar._collect_export_results(warnings)
    finally:
        ar._dataset = saved
        ar._reset_grain_stats_settings()
    assert not [w for w in warnings if w.startswith("citations")]
    path = str(tmp_path / "page.xlsx")
    export_to_excel(path, "scan", results)
    ws = openpyxl.load_workbook(path)["Citations"]
    text = ws["B2"].value
    assert "below 6.5 deg" in text and "fewer than 4 measurements" in text and "(here 1.5, in pixel steps)" in text
    assert "no grain was left out of them" in text and "number of measured pixels" in text
    refs = [ws.cell(row=r, column=4).value for r in range(2, 7)]
    assert any("10.1016/j.ultramic.2011.08.002" in r for r in refs)
    assert any("10.4028/www.scientific.net/SSP.160.63" in r for r in refs)
    assert ws["F5"].value.startswith("@article{bachmann2011,")


def test_without_grains_the_page_export_has_no_citations(tmp_path):
    import backend.api.routes.analysis as ar
    saved = ar._dataset
    ar._dataset = EBSDDataset(_xmap(), step_size=0.5)
    try:
        results = ar._collect_export_results([])
    finally:
        ar._dataset = saved
    assert "citations" not in results


def test_the_batch_writes_the_sheet_with_its_own_settings(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    xmap = _xmap()
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "a.h5").write_bytes(b"x")
    (folder / "b.h5").write_bytes(b"x")
    proc = bp.BatchProcessor(str(folder), str(tmp_path / "batch.xlsx"), gb_threshold_deg=6.0, min_grain_pixels=4,
                             grain_alpha=2.0, grain_fill=False, grain_edge_rule="keep_all")
    assert proc.process_all() == 2
    assert not [w for w in proc.warnings if w["reason"].startswith("citations")]
    wb = openpyxl.load_workbook(str(proc.workbook_path))
    assert wb.sheetnames == ELEVEN + ["Citations"]
    ws = wb["Citations"]
    assert [ws["A2"].value, ws["A3"].value] == ["a", "b"]
    for cell in ("B2", "B3"):
        text = ws[cell].value
        assert "below 6 deg" in text and "fewer than 4 measurements" in text and "(here 2, in pixel steps)" in text
        assert "shown with the orientation" not in text and "no grain was left out of them" in text
