"""A minimum intercept that was not given is recorded as not given.

The value is only recorded in Sheet 1 (row 11, "min_intercept_um (not applied)"); the reconstruction does not use it.
When the page does not send it (the field is empty or invalid), the workbook cell stays empty: it must not show a
default (1.5) that nobody entered.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from analysis.ebsd_dataset import EBSDDataset
from tests.batch_load_stub import patch_batch_load
from tests.synthetic_grain_map import make_map, to_xmap

LABEL = "min_intercept_um (not applied)"


def _row_11(path):
    import openpyxl
    ws = openpyxl.load_workbook(path)[openpyxl.load_workbook(path).sheetnames[0]]
    assert ws.cell(row=11, column=1).value == LABEL
    return ws.cell(row=11, column=2).value


@pytest.fixture
def one_file_batch(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    monkeypatch.setattr(bp, "analyze_texture", lambda *a, **k: None)
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)

    def run(**kw):
        from analysis.excel_exporter import ExcelExporter
        proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), **kw)
        ex = ExcelExporter(str(tmp_path / "batch.xlsx"))
        results = proc.process_file(tmp_path / "s.h5")
        results.pop("texture", None)                          # None here: the texture step is stubbed out
        ex.export_dataset("ds", results)
        ex.save()
        return proc, _row_11(str(tmp_path / "batch.xlsx"))
    return run


def test_a_batch_without_a_min_intercept_leaves_the_cell_empty(one_file_batch):
    proc, cell = one_file_batch()
    assert proc.min_intercept_um is None
    assert cell is None


@pytest.mark.parametrize("given", [2.25, 0.0])
def test_a_batch_with_a_min_intercept_records_it(one_file_batch, given):
    proc, cell = one_file_batch(min_intercept_um=given)
    assert cell == given


def test_the_batch_route_passes_nothing_when_the_page_sends_nothing(monkeypatch):
    """The route drops a field that is not sent, and the processor's default is "not given", so the pair gives an
    empty cell (the first test)."""
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    import analysis.batch_processor as bp
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder", lambda folder, out, progress_callback=None, **kw:
                        seen.append(kw) or bp.BatchResult(processed=["a.h5"], n_files=1))
    saved = ar._dataset
    ar._dataset = None
    try:
        c = TestClient(app)
        r = c.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o"})
        c.get(f"/api/analysis/task/{r.json()['task_id']}")
    finally:
        ar._dataset = saved
    assert seen == [{}]
    import inspect
    assert inspect.signature(bp.BatchProcessor.__init__).parameters["min_intercept_um"].default is None


@pytest.fixture
def client_map(monkeypatch):
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    import analysis.texture_analysis as ta
    monkeypatch.setattr(ta, "analyze_texture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped")))
    m = make_map(shape=(60, 80), n_seeds=40)
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = EBSDDataset(to_xmap(m), step_size=m["step"])
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


def _reconstruct(client, **extra):
    body = {"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 0.0, "fill": False, **extra}
    r = client.post("/api/analysis/grains/reconstruct", json=body)
    assert r.status_code == 200, r.text
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task


def _exported_cell(ar, tmp_path):
    from analysis.excel_exporter import ExcelExporter
    ex = ExcelExporter(str(tmp_path / "e.xlsx"))
    ex.export_dataset("ds", ar._collect_export_results())
    ex.save()
    return _row_11(str(tmp_path / "e.xlsx"))


def test_a_reconstruction_without_a_min_intercept_records_none(client_map, tmp_path):
    client, ar = client_map
    _reconstruct(client)
    assert ar._dataset.grain_recon_params["min_intercept_um"] is None
    assert client.get("/api/analysis/status").json()["recon_params"]["min_intercept"] is None
    assert _exported_cell(ar, tmp_path) is None


def test_a_reconstruction_with_a_min_intercept_records_it(client_map, tmp_path):
    client, ar = client_map
    _reconstruct(client, min_intercept_um=0.9)
    assert ar._dataset.grain_recon_params["min_intercept_um"] == 0.9
    assert client.get("/api/analysis/status").json()["recon_params"]["min_intercept"] == 0.9
    assert _exported_cell(ar, tmp_path) == 0.9
    _reconstruct(client, min_intercept_um=0)
    assert _exported_cell(ar, tmp_path) == 0.0                    # 0 is a value


def test_a_negative_min_intercept_is_refused(client_map):
    client, _ = client_map
    r = client.post("/api/analysis/grains/reconstruct", json={"min_intercept_um": -1})
    assert r.status_code == 422
