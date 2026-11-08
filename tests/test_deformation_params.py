"""The deformation parameters (KAM order, KAM threshold, HAGB limit) travel from the request to the KAM map, the RX
analysis' gKAM, the Excel export and the batch, are remembered, reset with the map and recorded in the workbook.

DeformationAnalysis.m classifies a boundary as SAGB from 3 degrees up to the HAGB limit (l.15-17); the expected
boundary lengths here are summed from the boundary angles and lengths of the grains, not by the code under test.
"""
import numpy as np
import openpyxl
import pytest
from fastapi.testclient import TestClient

from analysis.deformation_analysis import (DEFAULT_DEFORMATION_PARAMS, analyze_deformation, calculate_kam,
                                           check_deformation_params)
from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from analysis.rx_analysis import calculate_grain_average_kam
from tests.batch_load_stub import patch_batch_load
from tests.synthetic_grain_map import make_map, to_xmap

PERMISSIVE = {"rx_gos_threshold": 20.0, "rx_gbc_fraction": 0.0, "rx_gkam_threshold": 10.0, "rx_min_radius_um": 0.0}


def _dataset(shape=(60, 80), n_seeds=30):
    xmap = to_xmap(make_map(shape=shape, n_seeds=n_seeds))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(60, 200, xmap.size)
    ds = EBSDDataset(xmap, step_size=0.5)
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)
    return ds


def test_the_defaults_are_the_scripts():
    assert DEFAULT_DEFORMATION_PARAMS == {"kam_order": 2, "kam_threshold_deg": 7.0, "hagb_min_deg": 15.0}


@pytest.mark.parametrize("bad", [
    {"kam_order": 0}, {"kam_order": 6}, {"kam_order": 2.0}, {"kam_order": True}, {"kam_order": "2"},
    {"kam_threshold_deg": 0}, {"kam_threshold_deg": 30.5}, {"kam_threshold_deg": "x"}, {"kam_threshold_deg": float("nan")},
    {"hagb_min_deg": 3}, {"hagb_min_deg": 2.9}, {"hagb_min_deg": 91}, {"hagb_min_deg": None},
])
def test_check_deformation_params_names_what_is_wrong(bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        check_deformation_params(**bad)


def test_check_deformation_params_accepts_the_edges():
    check_deformation_params(kam_order=1, kam_threshold_deg=0.001, hagb_min_deg=3.001)
    check_deformation_params(kam_order=5, kam_threshold_deg=30, hagb_min_deg=90)
    with pytest.raises(ValueError, match="not a deformation parameter"):
        check_deformation_params(kam_orde=2)


def test_the_hagb_limit_ends_the_sagb_range():
    ds = _dataset()
    g = ds.grains
    angles, length = np.asarray(g.boundary_type, float), np.asarray(g.boundary_length, float)
    assert ((angles >= 15.0) & (angles < 45.0)).any()           # the fixture has boundaries between the two limits
    for hagb in (15.0, 45.0, 62.0):
        res = analyze_deformation(ds, g, hagb_min=hagb)
        assert res.hagb_min == hagb
        assert res.hagb_length == pytest.approx(length[angles >= hagb].sum())
        assert res.sagb_length == pytest.approx(length[(angles >= 3.0) & (angles < hagb)].sum())
        assert res.total_gb_length == pytest.approx(length[angles >= 3.0].sum())
        assert res.sagb_length + res.hagb_length == pytest.approx(res.total_gb_length)     # no gap, no overlap
    default = analyze_deformation(ds, g)
    assert default.hagb_min == 15.0 and default.sagb_length == pytest.approx(length[(angles >= 3) & (angles < 15)].sum())


def test_an_explicit_sagb_max_still_wins():
    ds = _dataset()
    g = ds.grains
    angles, length = np.asarray(g.boundary_type, float), np.asarray(g.boundary_length, float)
    res = analyze_deformation(ds, g, sagb_max=10.0, hagb_min=45.0)
    assert res.sagb_length == pytest.approx(length[(angles >= 3.0) & (angles < 10.0)].sum())


def test_the_kam_order_and_threshold_are_used_and_recorded():
    ds = _dataset()
    res = analyze_deformation(ds, ds.grains, kam_order=3, kam_threshold=2.0)
    assert (res.kam_order, res.kam_threshold) == (3, 2.0)
    np.testing.assert_array_equal(res.kam, calculate_kam(ds, order=3, threshold_deg=2.0, grains=ds.grains))
    default = analyze_deformation(ds, ds.grains)
    assert not np.array_equal(np.nan_to_num(res.kam), np.nan_to_num(default.kam))


# ---------------------------------------------------------------------------------------------
# the routes
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def client_map():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = _dataset()
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])
        ar._reset_grain_stats_settings()


def _status_params(client):
    return client.get("/api/analysis/status").json()["deformation_params"]


def test_the_deformation_tab_runs_with_the_scripts_values_until_it_is_told_otherwise(client_map):
    client, _ = client_map
    assert _status_params(client) == {"kam_order": 2, "kam_threshold_deg": 7.0, "hagb_min_deg": 15.0}
    body = client.post("/api/analysis/deformation").json()
    assert body["deformation_params"] == {"kam_order": 2, "kam_threshold_deg": 7.0, "hagb_min_deg": 15.0}


def test_the_request_is_answered_remembered_and_a_field_left_out_keeps_the_last_value(client_map):
    client, ar = client_map
    ds = ar._dataset
    angles, length = np.asarray(ds.grains.boundary_type, float), np.asarray(ds.grains.boundary_length, float)
    r = client.post("/api/analysis/deformation", json={"kam_order": 3, "kam_threshold_deg": 0.3, "hagb_min_deg": 45.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deformation_params"] == {"kam_order": 3, "kam_threshold_deg": 0.3, "hagb_min_deg": 45.0}
    assert _status_params(client) == body["deformation_params"]
    assert body["sagb_length_um"] == pytest.approx(length[(angles >= 3) & (angles < 45)].sum())
    assert body["hagb_length_um"] == pytest.approx(length[angles >= 45].sum())
    # the KAM of the answer is the KAM of that order and threshold, and not the script's
    kam = calculate_kam(ds, order=3, threshold_deg=0.3, grains=ds.grains)
    assert body["mean_kam"] == pytest.approx(float(np.nanmean(kam)))
    assert body["mean_kam"] != pytest.approx(float(np.nanmean(calculate_kam(ds, order=3, grains=ds.grains))))   # threshold 7
    assert body["mean_kam"] != pytest.approx(float(np.nanmean(calculate_kam(ds, threshold_deg=0.3, grains=ds.grains))))   # order 2
    again = client.post("/api/analysis/deformation", json={"hagb_min_deg": 30.0}).json()
    assert again["deformation_params"] == {"kam_order": 3, "kam_threshold_deg": 0.3, "hagb_min_deg": 30.0}
    assert client.post("/api/analysis/deformation").json()["deformation_params"] == again["deformation_params"]


@pytest.mark.parametrize("body", [
    {"kam_order": 0}, {"kam_order": 6}, {"kam_order": 2.5}, {"kam_order": "2"}, {"kam_order": True},
    {"kam_threshold_deg": 0}, {"kam_threshold_deg": 30.01}, {"kam_threshold_deg": "7"},
    {"hagb_min_deg": 3}, {"hagb_min_deg": 91},
])
def test_a_value_out_of_range_is_a_422_and_changes_nothing(client_map, body):
    client, _ = client_map
    assert client.post("/api/analysis/deformation", json=body).status_code == 422
    assert _status_params(client) == DEFAULT_DEFORMATION_PARAMS


def test_the_parameters_are_reset_with_the_map(client_map):
    client, ar = client_map
    client.post("/api/analysis/deformation", json={"kam_order": 4, "kam_threshold_deg": 3.0, "hagb_min_deg": 20.0})
    ar._reset_grain_stats_settings()            # what /load and a file switch do
    assert _status_params(client) == DEFAULT_DEFORMATION_PARAMS


def test_the_kam_map_follows_the_remembered_threshold(client_map):
    client, _ = client_map
    before = client.get("/api/analysis/map/kam").json()["value_range"]["max"]
    assert before > 0.2
    client.post("/api/analysis/deformation", json={"kam_threshold_deg": 0.2})
    after = client.get("/api/analysis/map/kam").json()["value_range"]["max"]
    assert after <= 0.2 < before                # a mean of disorientations below 0.2 degrees


def test_the_rx_analysis_uses_the_kam_of_the_deformation_tab(client_map):
    client, ar = client_map
    ds = ar._dataset
    first = client.post("/api/analysis/rx", json=PERMISSIVE).json()
    client.post("/api/analysis/deformation", json={"kam_order": 1, "kam_threshold_deg": 0.2})
    second = client.post("/api/analysis/rx").json()          # same criteria, other KAM: not the cached answer
    kam = calculate_kam(ds, order=1, threshold_deg=0.2, grains=ds.grains)
    want = 100.0 * (np.isfinite(kam) & (kam < 0.55)).sum() / np.isfinite(kam).sum()
    assert second["low_kam_fraction"] == pytest.approx(want)
    assert second["low_kam_fraction"] != pytest.approx(first["low_kam_fraction"])


@pytest.mark.parametrize("change", [{"kam_threshold_deg": 0.2}, {"kam_order": 1}])
def test_one_changed_kam_parameter_alone_is_not_served_from_the_cached_rx_answer(client_map, change):
    client, ar = client_map
    ds = ar._dataset
    first = client.post("/api/analysis/rx", json=PERMISSIVE).json()
    client.post("/api/analysis/deformation", json=change)
    second = client.post("/api/analysis/rx").json()
    params = {**DEFAULT_DEFORMATION_PARAMS, **change}
    kam = calculate_kam(ds, order=params["kam_order"], threshold_deg=params["kam_threshold_deg"], grains=ds.grains)
    want = 100.0 * (np.isfinite(kam) & (kam < 0.55)).sum() / np.isfinite(kam).sum()
    assert second["low_kam_fraction"] == pytest.approx(want)
    assert second["low_kam_fraction"] != pytest.approx(first["low_kam_fraction"])


def test_the_export_and_the_workbook_use_and_record_the_parameters(client_map, tmp_path):
    client, ar = client_map
    ds = ar._dataset
    client.post("/api/analysis/deformation", json={"kam_order": 3, "kam_threshold_deg": 2.5, "hagb_min_deg": 45.0})
    client.post("/api/analysis/rx", json=PERMISSIVE)
    warnings = []
    results = ar._collect_export_results(warnings)
    deform = results["deformation"]
    assert (deform.kam_order, deform.kam_threshold, deform.hagb_min) == (3, 2.5, 45.0)
    # the RX analysis' gKAM is made of the same KAM
    kam = calculate_kam(ds, order=3, threshold_deg=2.5, grains=ds.grains)
    np.testing.assert_allclose(results["rx"].g_kam, calculate_grain_average_kam(kam, ds.grains.grain_ids))
    from analysis.excel_exporter import export_to_excel
    path = str(tmp_path / "w.xlsx")
    export_to_excel(path, "s", results)
    wb = openpyxl.load_workbook(path)
    assert (wb["KAM hist"]["B1"].value, wb["KAM hist"]["B2"].value) == (3, 2.5)
    assert wb["Sheet1"]["A71"].value == "hagb_min_deg" and wb["Sheet1"]["B71"].value == 45.0
    angles, length = np.asarray(ds.grains.boundary_type, float), np.asarray(ds.grains.boundary_length, float)
    assert wb["Sheet1"]["B23"].value == pytest.approx(length[angles >= 45].sum())
    assert wb["Sheet1"]["B24"].value == pytest.approx(length[(angles >= 3) & (angles < 45)].sum())


def test_the_kam_histogram_sheet_has_room_for_the_largest_threshold(tmp_path):
    """A KAM threshold of 30 degrees allows a KAM above 15 degrees: its bin is on the sheet's grid, not left out."""
    from analysis.excel_exporter import ExcelExporter, KAM_BIN_CENTERS, KAM_FIRST_ROW
    ds = _dataset()
    deform = analyze_deformation(ds, ds.grains)
    deform.kam_hist_edges = np.arange(0, 30.25, 0.25)
    deform.kam_hist_counts = np.zeros(len(deform.kam_hist_edges) - 1)
    deform.kam_hist_counts[-1] = 1.0                           # the bin 29.75-30
    ex = ExcelExporter(str(tmp_path / "w.xlsx"))
    ex.export_dataset("s", {"basic_info": {}, "deformation": deform})
    ws = ex.wb["KAM hist"]
    assert ws.cell(row=KAM_FIRST_ROW + len(KAM_BIN_CENTERS) - 1, column=2).value == 1.0
    assert KAM_BIN_CENTERS[-1] == pytest.approx(29.875)


# ---------------------------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------------------------
def _processor(tmp_path, monkeypatch, **kw):
    from analysis import batch_processor as bp
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(60, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    return bp, bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), **kw)


def test_the_batch_uses_the_parameters_and_the_workbook_records_them(tmp_path, monkeypatch):
    bp, proc = _processor(tmp_path, monkeypatch, kam_order=3, kam_threshold_deg=2.5, hagb_min_deg=45.0, **PERMISSIVE)
    res = proc.process_file(tmp_path / "s.h5")
    assert (res["deformation"].kam_order, res["deformation"].kam_threshold, res["deformation"].hagb_min) == (3, 2.5, 45.0)
    g = np.asarray(res["deformation"].kam)
    assert np.nanmax(g) <= 2.5
    (tmp_path / "s.h5").write_bytes(b"x")
    assert proc.process_all() == 1
    wb = openpyxl.load_workbook(str(proc.workbook_path))
    assert (wb["KAM hist"]["B1"].value, wb["KAM hist"]["B2"].value) == (3, 2.5)
    assert wb["Sheet1"]["B71"].value == 45.0
    default = _processor(tmp_path, monkeypatch)[1]
    assert (default.kam_order, default.kam_threshold_deg, default.hagb_min_deg) == (2, 7.0, 15.0)


def test_the_batch_refuses_values_out_of_range(tmp_path, monkeypatch):
    bp, _ = _processor(tmp_path, monkeypatch)
    for kw in ({"kam_order": 0}, {"kam_order": 2.0}, {"kam_threshold_deg": 31}, {"hagb_min_deg": 3.0}):
        with pytest.raises(ValueError, match=next(iter(kw))):
            bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), **kw)


def test_the_batch_route_passes_the_parameters_only_when_given(client_map, monkeypatch):
    client, _ = client_map
    import analysis.batch_processor as bp
    from analysis.batch_processor import BatchResult
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder",
                        lambda folder, out, progress_callback=None, **kw: seen.append(kw)
                        or BatchResult(processed=["a.h5"], n_files=1))
    for body in ({"folder_path": "f", "output_path": "o"},
                 {"folder_path": "f", "output_path": "o", "kam_order": 3, "kam_threshold_deg": 2.5,
                  "hagb_min_deg": 45.0}):
        r = client.post("/api/analysis/batch", json=body)
        assert r.status_code == 200
        client.get(f"/api/analysis/task/{r.json()['task_id']}")
    assert seen == [{}, {"kam_order": 3, "kam_threshold_deg": 2.5, "hagb_min_deg": 45.0}]
    for bad in ({"kam_order": 6}, {"kam_threshold_deg": 0}, {"hagb_min_deg": 3}, {"hagb_min_deg": 91}):
        assert client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o", **bad}).status_code == 422
