"""Robustness of the analysis export and the batch: a batch file whose texture step fails, missing numbers as empty
cells, the BC histogram normalisation, the step-size warning, the same workbook from the single export and the batch,
and the phase list that says why it is empty."""
import logging
from types import SimpleNamespace

import numpy as np
import openpyxl
import pytest

from analysis.ebsd_dataset import EBSDDataset, GrainSet
from analysis.grain_analysis import analyze_grain_size, reconstruct_grains
from analysis.deformation_analysis import analyze_deformation
from tests.batch_load_stub import patch_batch_load
from tests.synthetic_grain_map import make_map, to_xmap


def _xmap(shape=(40, 50), n_seeds=14, point_groups=None, bc=True):
    m = make_map(shape=shape, n_seeds=n_seeds)
    if point_groups is not None:
        m["point_groups"] = np.array(point_groups)
    xmap = to_xmap(m)
    if bc:
        xmap.prop["bc"] = np.random.default_rng(0).uniform(60, 200, xmap.size)
    return xmap


def _dataset(**kw):
    ds = EBSDDataset(_xmap(**kw), step_size=0.5)
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)
    return ds


def _sheets(path):
    """``{sheet: {(row, col): value}}`` of a workbook, empty cells left out."""
    wb = openpyxl.load_workbook(path)
    return {ws.title: {(c.row, c.column): c.value for row in ws.iter_rows() for c in row if c.value not in (None, "")}
            for ws in wb.worksheets}


# ---------------------------------------------------------------------------------------------
# a texture failure drops the texture step of the file, not the file
# ---------------------------------------------------------------------------------------------
def test_a_hexagonal_file_still_lands_in_the_batch_workbook_with_the_texture_step_left_out(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    xmap = _xmap(point_groups=["6/mmm", "4/mmm"])
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    (tmp_path / "hex.h5").write_bytes(b"x")
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"))
    assert proc.process_all() == 1
    res = proc.result()
    assert res.processed and not res.failed
    steps = [w["reason"].split(":", 1)[0] for w in res.warnings]
    assert "texture" in steps and "not cubic" in " ".join(w["reason"] for w in res.warnings)
    cells = _sheets(str(proc.workbook_path))["Sheet1"]
    assert cells.get((12, 2)) is not None and cells.get((22, 2)) is not None       # grain size and KAM are there
    assert (25, 2) not in cells and (45, 2) not in cells and (67, 2) not in cells  # the texture rows are empty
    assert not any(row >= 27 and row <= 40 for row, _ in cells if _ == 2)          # no component fraction
    assert (41, 2) in cells                                                         # RX is made
    assert "texture_by_rx" not in steps                                             # no split without a texture
    assert not [cell for cell in _sheets(str(proc.workbook_path))["SmallVSLargeGrain"] if cell[0] >= 3]
    assert "SmallVSLargeGrain" in openpyxl.load_workbook(str(proc.workbook_path)).sheetnames


def test_the_single_export_and_the_batch_drop_the_same_step_for_a_non_cubic_map(tmp_path, monkeypatch):
    import backend.api.routes.analysis as ar
    ds = _dataset(point_groups=["6/mmm", "4/mmm"])
    saved = ar._dataset
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        warnings = []
        ar._collect_export_results(warnings)
    finally:
        ar._dataset = saved
    assert [w.split(":", 1)[0] for w in warnings] == ["texture"]


# ---------------------------------------------------------------------------------------------
# a missing number is an empty cell
# ---------------------------------------------------------------------------------------------
def test_a_phase_without_a_grain_has_empty_statistics_cells(tmp_path):
    from analysis.excel_exporter import ExcelExporter
    ds = _dataset()
    g = ds.grains
    no_phase_1 = GrainSet(**{**g.__dict__, "grain_phase": np.zeros(g.n_grains, dtype=np.int64)})
    gs = analyze_grain_size(ds, no_phase_1, stats_phase=1)
    assert gs.n_grains_used == 0
    ex = ExcelExporter(str(tmp_path / "w.xlsx"))
    from analysis.grain_analysis import grain_size_basic_info
    ex.export_dataset("s", {"basic_info": grain_size_basic_info(gs), "grain_size": gs})
    ws = ex.wb["Sheet1"]
    for row in (12, 13, 14, 15, 16, 17, 18, 55, 56, 57, 58):
        assert ws.cell(row=row, column=2).value in (None, ""), (row, ws.cell(row=row, column=1).value)
    assert ws.cell(row=51, column=2).value == 0 and ws.cell(row=53, column=2).value == 0     # the counts say why
    assert ws.cell(row=59, column=2).value == "Al7FeCu2"
    # a phase with grains keeps its numbers
    ex.export_dataset("t", {"basic_info": {}, "grain_size": analyze_grain_size(ds, g, stats_phase=0)})
    assert ws.cell(row=12, column=3).value > 0


def test_a_map_without_a_finite_kam_has_an_empty_mean_kam_cell(tmp_path):
    from analysis.excel_exporter import ExcelExporter
    ds = _dataset()
    deform = analyze_deformation(ds, ds.grains)
    deform.kam = np.full_like(deform.kam, np.nan)
    ex = ExcelExporter(str(tmp_path / "w.xlsx"))
    ex.export_dataset("s", {"basic_info": {}, "deformation": deform})
    assert ex.wb["Sheet1"].cell(row=22, column=2).value in (None, "")
    ex.export_dataset("t", {"basic_info": {}, "deformation": analyze_deformation(ds, ds.grains)})
    assert ex.wb["Sheet1"].cell(row=22, column=3).value > 0


def test_a_reconstruction_without_a_grain_has_empty_boundary_cells(tmp_path):
    from analysis.excel_exporter import ExcelExporter
    ds = _dataset()
    none = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=10 ** 6, alpha=3.1, fill=False)
    assert none.n_grains == 0
    deform = analyze_deformation(ds, none)
    assert deform.hagb_length is None and deform.gb_per_field_area is None
    ex = ExcelExporter(str(tmp_path / "w.xlsx"))
    ex.export_dataset("s", {"basic_info": {}, "deformation": deform})
    assert ex.wb["Sheet1"].cell(row=23, column=2).value in (None, "")
    assert ex.wb["Sheet1"].cell(row=24, column=2).value in (None, "")
    for row in range(3, 11):
        assert ex.wb["Grain Boundaries"].cell(row=row, column=2).value in (None, ""), row
    ex.export_dataset("t", {"basic_info": {}, "deformation": analyze_deformation(ds, ds.grains)})
    assert ex.wb["Grain Boundaries"].cell(row=9, column=3).value > 0


def test_a_texture_component_that_could_not_be_built_is_an_empty_cell_not_a_zero(tmp_path, monkeypatch):
    from analysis import texture_analysis as ta
    from analysis.excel_exporter import ExcelExporter
    from analysis.texture_components import TEXTURE_PRESETS
    real = ta.component_variants

    def refuse_brass(comp, phase):
        if comp is TEXTURE_PRESETS["FCC_Rolling"]["Brass"]:
            raise ValueError("no such component")
        return real(comp, phase)

    monkeypatch.setattr(ta, "component_variants", refuse_brass)
    ds = _dataset()
    tx = ta.analyze_texture(ds, TEXTURE_PRESETS["FCC_Rolling"])
    assert tx.component_fractions["Brass"] is None
    assert all(v is not None for k, v in tx.component_fractions.items() if k != "Brass")
    ex = ExcelExporter(str(tmp_path / "w.xlsx"))
    ex.export_dataset("s", {"basic_info": {}, "texture": tx})
    ws = ex.wb["Sheet1"]
    names = list(tx.component_fractions)
    for i, name in enumerate(names):
        assert ws.cell(row=27 + i, column=1).value == name
        assert (ws.cell(row=27 + i, column=2).value is None) == (name == "Brass")


def test_a_phase_without_a_pixel_has_no_plane_share():
    """The quality filter can unindex every pixel of a phase; the phase stays in the phase list."""
    m = make_map(shape=(30, 30), n_seeds=8)
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.where(np.asarray(m["phase_id"]).reshape(-1) == 1, 0.0, 100.0)
    ds = EBSDDataset(xmap, step_size=0.5)
    assert (np.asarray(xmap.phase_id) == 1).any()
    ds.apply_quality_filter(bc_min=1)
    assert not (np.asarray(ds.xmap.phase_id) == 1).any()
    from analysis import texture_analysis as ta
    assert ta.calculate_surface_plane_fraction(ds, [1, 1, 1], texture_phase=1) is None
    assert ta.calculate_surface_plane_family_fraction(ds, [1, 1, 1], texture_phase=1) is None
    assert ta.calculate_surface_plane_fraction(ds, [1, 1, 1], texture_phase=0) is not None


def test_the_texture_route_sends_null_for_a_missing_number(monkeypatch):
    import backend.api.routes.analysis as ar
    from analysis import texture_analysis as ta
    from analysis.texture_components import TEXTURE_PRESETS
    from fastapi.testclient import TestClient
    from backend.api.main import app
    real = ta.component_variants

    def refuse_brass(comp, phase):
        if comp is TEXTURE_PRESETS["FCC_Rolling"]["Brass"]:
            raise ValueError("no such component")
        return real(comp, phase)

    monkeypatch.setattr(ta, "component_variants", refuse_brass)
    saved = ar._dataset
    ar._dataset = _dataset()
    ar._reset_grain_stats_settings()
    try:
        r = TestClient(app).post("/api/analysis/texture", json={})
    finally:
        ar._dataset = saved
    assert r.status_code == 200, r.text
    comps = {c["name"]: c["fraction"] for c in r.json()["components"]}
    assert comps["Brass"] is None and comps["Cube"] is not None


# ---------------------------------------------------------------------------------------------
# a file without a recorded step size
# ---------------------------------------------------------------------------------------------
def test_a_file_without_a_recorded_step_gets_the_warning_and_a_file_with_one_does_not(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    from analysis.xmap_loader import step_of_xmap
    xmap = _xmap()
    meta = {"step_from_file": False}

    def load(path):
        return xmap, step, dict(meta)

    monkeypatch.setattr(bp, "load_xmap_file", load)
    step = 1.0
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"))
    proc.process_file(tmp_path / "old.h5")
    assert [w["reason"] for w in proc.warnings if w["reason"].startswith("step_size")] == [
        "step_size: step size not recorded in the file; 1.0 µm assumed"]
    # a step the file states is not a guess, whatever its value; a derived step other than 1.0 is not 'assumed'
    for step, meta in ((1.0, {"step_from_file": True}), (0.5, {"step_from_file": False})):
        proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"))
        proc.process_file(tmp_path / "new.h5")
        assert not any("step" in w["reason"] for w in proc.warnings)


def test_the_step_warning_reaches_the_batch_result_and_the_log(tmp_path, monkeypatch, caplog):
    from analysis import batch_processor as bp
    xmap = _xmap()
    monkeypatch.setattr(bp, "load_xmap_file", lambda path: (xmap, 1.0, {"step_from_file": False}))
    (tmp_path / "old.h5").write_bytes(b"x")
    with caplog.at_level(logging.WARNING, logger="analysis.batch_processor"):
        res = bp.batch_process_folder(str(tmp_path), str(tmp_path / "doc.xlsx"))
    assert any("step size not recorded in the file; 1.0 µm assumed" in w["reason"] for w in res.warnings)
    assert "step size not recorded in the file; 1.0 µm assumed" in caplog.text


# ---------------------------------------------------------------------------------------------
# the single export and the batch give one workbook for one map
# ---------------------------------------------------------------------------------------------
def _same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == pytest.approx(b, rel=1e-9, abs=1e-12)
    return a == b


def test_the_single_export_and_the_batch_write_the_same_workbook_for_the_same_map(tmp_path, monkeypatch):
    """Real texture (not stubbed), default settings, every sheet cell by cell."""
    import backend.api.routes.analysis as ar
    from analysis import batch_processor as bp
    from analysis.excel_exporter import export_to_excel
    from analysis.xmap_loader import step_of_xmap
    xmap = _xmap(shape=(50, 60), n_seeds=20)
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "scan.h5").write_bytes(b"x")
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    res = bp.batch_process_folder(str(folder), str(tmp_path / "batch.xlsx"))
    assert res.processed and not res.failed

    # the page: load, reconstruct with the batch's default parameters, export with the default settings
    ds = EBSDDataset(xmap, step_of_xmap(xmap), source="h5")
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=4.0, min_pixels=3, alpha=3.1, fill=True)
    ds.grain_recon_params = {"gb_threshold_deg": 4.0, "min_pixels": 3, "alpha": 3.1, "min_intercept_um": None,
                             "fill": True}
    saved = ar._dataset
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        warnings = []
        results = ar._collect_export_results(warnings)
    finally:
        ar._dataset = saved
    assert warnings == [] and {"texture", "texture_by_rx", "rx", "bc_gmm", "bc_histogram"} <= set(results)
    single = str(tmp_path / "single.xlsx")
    export_to_excel(single, "scan", results)

    a, b = _sheets(single), _sheets(res.workbook_path)
    assert list(a) == list(b) and len(a) == 11
    assert len(a["Sheet1"]) > 100
    for sheet in a:
        assert set(a[sheet]) == set(b[sheet]), (sheet, sorted(set(a[sheet]) ^ set(b[sheet]))[:10])
        for cell, value in a[sheet].items():
            assert _same(value, b[sheet][cell]), (sheet, cell, value, b[sheet][cell])


# ---------------------------------------------------------------------------------------------
# the phase list says why it is empty
# ---------------------------------------------------------------------------------------------
def test_an_unreadable_phase_list_is_logged_with_its_reason(caplog):
    import backend.api.routes.analysis as ar

    class Broken:
        @property
        def xmap(self):
            raise RuntimeError("phase list unreadable")

    with caplog.at_level(logging.WARNING, logger=ar.logger.name):
        assert ar._phase_options(Broken()) == []
    assert any(r.levelno == logging.WARNING and "phase list unreadable" in r.getMessage() for r in caplog.records)
    assert ar._phase_options(_dataset())[0]["name"] == "Al"


# ---------------------------------------------------------------------------------------------
# the batch summary does not call the step-size note a left-out step
# ---------------------------------------------------------------------------------------------
def test_the_step_size_note_is_not_a_left_out_step_in_the_batch_summary():
    from analysis.batch_processor import BatchResult
    from backend.api.routes.analysis import _batch_summary_text
    note = {"file": "old.h5", "reason": "step_size: step size not recorded in the file; 1.0 µm assumed"}
    rx = {"file": "old.h5", "reason": "rx: needs native band contrast in the file"}
    other = {"file": "new.h5", "reason": "texture: not cubic"}
    only_note = _batch_summary_text(BatchResult(processed=["old.h5"], n_files=1, warnings=[note]))
    assert "left out" not in only_note and "step_size" not in only_note
    mixed = _batch_summary_text(BatchResult(processed=["old.h5", "new.h5"], n_files=2, warnings=[note, rx, other]))
    assert "2 file(s) with analysis steps left out (old.h5: rx; new.h5: texture)" in mixed and "step_size" not in mixed
    # a file with only the note is not counted among the files that had a step left out
    counted = _batch_summary_text(BatchResult(processed=["old.h5", "new.h5"], n_files=2, warnings=[note, other]))
    assert "1 file(s) with analysis steps left out (new.h5: texture)" in counted
