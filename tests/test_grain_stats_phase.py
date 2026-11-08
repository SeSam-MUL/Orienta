"""Grain-size statistics per phase, sample standard deviation, honest sheet 1.

The script ``GrainSizeAnalysis_mtex6.m`` computes everything on the grains of ONE phase when particles are
removed (band, safeguard, ECD, means), takes MATLAB ``std`` (divisor N - 1), and the Excel sheet 1 has to say
which reconstruction parameters made the grains.
"""
import logging
from pathlib import Path

import numpy as np
import pytest
from tests.batch_load_stub import patch_batch_load
from fastapi.testclient import TestClient

from analysis.ebsd_dataset import EBSDDataset, GrainSet
from analysis.grain_analysis import analyze_grain_size, find_edge_grains, reconstruct_grains
from tests.synthetic_grain_map import make_map, random_quaternions, to_xmap

TILE = 4


def _tile_map(rows, cols0, cols1, step=1.0):
    """rows x (cols0 + cols1) tiles of TILE x TILE pixels, one random orientation per tile; the left
    ``cols0`` tile columns are phase 0 (Al), the right ``cols1`` phase 1 (Al7FeCu2).
    Returns (dataset, grains, tile_phase, tile_gid) with the arrays shaped (rows, cols0 + cols1)."""
    cols = cols0 + cols1
    rng = np.random.default_rng(5)
    gq = random_quaternions(rng, rows * cols)
    q = np.repeat(np.repeat(gq.reshape(rows, cols, 4), TILE, axis=0), TILE, axis=1)
    tile_phase = np.zeros((rows, cols), dtype=np.int64)
    tile_phase[:, cols0:] = 1
    phase_id = np.repeat(np.repeat(tile_phase, TILE, axis=0), TILE, axis=1)
    m = {"quats": q, "phase_id": phase_id, "phase_names": np.array(["Al", "Al7FeCu2"]),
         "point_groups": np.array(["m-3m", "4/mmm"]), "step": step}
    ds = EBSDDataset(to_xmap(m), step_size=step)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, alpha=0.0, fill=False)
    assert g.n_grains == rows * cols
    tile_gid = g.grain_ids[::TILE, ::TILE][:rows, :cols]
    return ds, g, tile_phase, tile_gid


def _ring(tile_phase, phase):
    """Tiles of ``phase`` in the outer tile row/column of that phase's own extent (the 2 px band)."""
    ry, rx = np.nonzero(tile_phase == phase)
    on_edge = ((ry == ry.min()) | (ry == ry.max()) | (rx == rx.min()) | (rx == rx.max()))
    out = np.zeros(tile_phase.shape, dtype=bool)
    out[ry[on_edge], rx[on_edge]] = True
    return out


@pytest.fixture(scope="module")
def big():
    """12 x (12 + 8) tiles: phase 0 has 144 grains, phase 1 has 96, together 240."""
    return _tile_map(12, 12, 8)


@pytest.fixture(scope="module")
def small():
    """6 x (12 + 8) tiles: phase 0 has 72 grains, phase 1 has 48, together 120."""
    return _tile_map(6, 12, 8)


# ---------------------------------------------------------------------------------------------
# the phase filter
# ---------------------------------------------------------------------------------------------
def test_stats_phase_takes_the_band_and_the_counts_from_that_phase_only(big):
    ds, g, tile_phase, tile_gid = big
    expect = {
        None: (240, int(_outer(tile_phase).sum()), None, None),
        0: (144, int(_ring(tile_phase, 0).sum()), 0, "Al"),
        1: (96, int(_ring(tile_phase, 1).sum()), 1, "Al7FeCu2"),
    }
    assert expect[0][1] == 44 and expect[1][1] == 36 and expect[None][1] == 60
    for sp, (total, n_edge, resolved, name) in expect.items():
        r = analyze_grain_size(ds, g, stats_phase=sp)
        assert (r.n_grains_total, r.n_edge_grains, r.n_grains_used) == (total, n_edge, total - n_edge), sp
        assert r.edge_safeguard_applied is False
        assert (r.stats_phase, r.stats_phase_name) == (resolved, name)
        assert len(r.ecd_per_grain) == len(r.maxDimX_per_grain) == len(r.ar_per_grain) == total - n_edge


def _outer(tile_phase):
    out = np.zeros(tile_phase.shape, dtype=bool)
    out[[0, -1], :] = True
    out[:, [0, -1]] = True
    return out


def test_stats_phase_statistics_are_of_the_grains_of_that_phase(big):
    ds, g, tile_phase, tile_gid = big
    for p in (0, 1):
        keep = (tile_phase == p) & ~_ring(tile_phase, p)
        gids = tile_gid[keep]
        r = analyze_grain_size(ds, g, stats_phase=p)
        np.testing.assert_allclose(np.sort(r.ecd_per_grain), np.sort(g.ecd[gids - 1]))
        assert r.avg_ECD == pytest.approx(g.ecd[gids - 1].mean())
        np.testing.assert_allclose(np.sort(r.ar_per_grain), np.sort(g.aspect_ratio[gids - 1]))


def test_find_edge_grains_phase_limits_the_extent_to_the_phase(big):
    """The tile column where phase 1 starts is on the left edge of phase 1's extent only."""
    ds, g, tile_phase, tile_gid = big
    col = 12                                                   # first tile column of phase 1
    inner = tile_gid[1:-1, col]
    assert not find_edge_grains(ds, g, "matlab_script")[inner - 1].any()
    assert find_edge_grains(ds, g, "matlab_script", phase=1)[inner - 1].all()
    # a phase filter never flags a grain of another phase
    flagged = find_edge_grains(ds, g, "matlab_script", phase=1)
    assert not flagged[tile_gid[tile_phase == 0] - 1].any()


@pytest.mark.parametrize("rule", ["mtex", "outer_row"])
def test_find_edge_grains_phase_never_flags_other_phases(big, rule):
    ds, g, tile_phase, tile_gid = big
    flagged = find_edge_grains(ds, g, rule, phase=0)
    assert not flagged[tile_gid[tile_phase == 1] - 1].any()


def test_safeguard_counts_the_grains_of_the_phase(small):
    """Phase 0: 72 grains, 32 on the ring, 40 would remain (< 50): all stay. All phases: 120 grains,
    48 on the ring, 72 remain: the ring goes."""
    ds, g, tile_phase, tile_gid = small
    one = analyze_grain_size(ds, g, stats_phase=0)
    assert (one.n_grains_total, one.n_edge_grains, one.n_grains_used) == (72, 32, 72)
    assert one.edge_safeguard_applied is True
    both = analyze_grain_size(ds, g)
    assert (both.n_grains_total, both.n_edge_grains, both.n_grains_used) == (120, 48, 72)
    assert both.edge_safeguard_applied is False


def test_most_measured_picks_the_phase_with_the_most_measured_pixels(big):
    ds, g, *_ = big
    r = analyze_grain_size(ds, g, stats_phase="most_measured")
    assert (r.stats_phase, r.stats_phase_name, r.n_grains_total) == (0, "Al", 144)
    # the other way round: phase 1 is the larger one
    ds2, g2, *_ = _tile_map(12, 8, 12)
    r2 = analyze_grain_size(ds2, g2, stats_phase="most_measured")
    assert (r2.stats_phase, r2.stats_phase_name, r2.n_grains_total) == (1, "Al7FeCu2", 144)
    assert analyze_grain_size(ds2, g2, stats_phase=1).avg_ECD == r2.avg_ECD


def test_most_measured_counts_pixels_not_grains():
    """Phase 0: three big tiles (3 x 36 pixels); phase 1: five small ones (5 x 4 pixels)."""
    lab = np.zeros((6, 38), dtype=np.int64)
    for k in range(3):
        lab[:, k * 6:(k + 1) * 6] = k + 1
    cols = 18
    for k in range(5):
        lab[:2, cols + k * 4: cols + (k + 1) * 4] = 4 + k
    rng = np.random.default_rng(1)
    gq = random_quaternions(rng, 10)
    q = gq[lab]
    phase = np.where(lab == 0, -1, np.where(lab <= 3, 0, 1)).astype(np.int64)
    m = {"quats": q, "phase_id": phase, "phase_names": np.array(["Al", "Al7FeCu2"]),
         "point_groups": np.array(["m-3m", "4/mmm"]), "step": 1.0}
    ds = EBSDDataset(to_xmap(m), step_size=1.0)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, alpha=0.0, fill=False)
    assert g.n_grains == 8
    r = analyze_grain_size(ds, g, edge_rule="keep_all", stats_phase="most_measured")
    assert (r.stats_phase, r.n_grains_total) == (0, 3)


def test_all_phases_is_unchanged_by_the_new_argument(big):
    ds, g, *_ = big
    a = analyze_grain_size(ds, g)
    b = analyze_grain_size(ds, g, stats_phase=None)
    np.testing.assert_array_equal(a.ecd_per_grain, b.ecd_per_grain)
    assert (a.stats_phase, a.stats_phase_name) == (None, None)


def test_unknown_phase_raises_and_names_the_known_ids(big):
    ds, g, *_ = big
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        analyze_grain_size(ds, g, stats_phase=7)
    with pytest.raises(ValueError):
        analyze_grain_size(ds, g, stats_phase="bogus")
    with pytest.raises(ValueError):
        analyze_grain_size(ds, g, stats_phase=True)
    with pytest.raises(ValueError):
        analyze_grain_size(ds, g, stats_phase=-1)


def test_stats_phase_needs_the_phase_of_the_grains(big):
    ds, g, *_ = big
    bare = GrainSet(grain_ids=g.grain_ids, n_grains=g.n_grains, grain_size=g.grain_size, area=g.area,
                    ecd=g.ecd, aspect_ratio=g.aspect_ratio)
    with pytest.raises(ValueError, match="grain_phase"):
        analyze_grain_size(ds, bare, edge_rule="keep_all", stats_phase=0)
    analyze_grain_size(ds, bare, edge_rule="keep_all")            # all phases needs nothing of the kind


def test_a_phase_without_grains_gives_an_empty_result_that_names_it(big):
    ds, g, *_ = big
    no_phase_1 = GrainSet(**{**g.__dict__, "grain_phase": np.zeros(g.n_grains, dtype=np.int64)})
    r = analyze_grain_size(ds, no_phase_1, stats_phase=1)
    assert (r.n_grains_total, r.n_grains_used, r.stats_phase, r.stats_phase_name) == (0, 0, 1, "Al7FeCu2")
    assert r.avg_ECD == 0.0 and len(r.ecd_per_grain) == 0


# ---------------------------------------------------------------------------------------------
# sample standard deviation
# ---------------------------------------------------------------------------------------------
def test_std_is_the_sample_standard_deviation(big):
    ds, g, *_ = big
    # give the grains different sizes so that the spread is not zero
    g2 = GrainSet(**{**g.__dict__, "ecd": g.ecd * (1 + 0.01 * np.arange(g.n_grains)),
                     "aspect_ratio": g.aspect_ratio * (1 + 0.02 * np.arange(g.n_grains))})
    r = analyze_grain_size(ds, g2, edge_rule="keep_all")
    assert r.std_ECD > 0 and r.std_AR > 0
    assert r.std_ECD == pytest.approx(np.std(r.ecd_per_grain, ddof=1))
    assert r.std_AR == pytest.approx(np.std(r.ar_per_grain, ddof=1))
    assert r.std_maxDimX == pytest.approx(np.std(r.maxDimX_per_grain, ddof=1))
    assert r.std_maxDimY == pytest.approx(np.std(r.maxDimY_per_grain, ddof=1))
    assert r.std_ECD > np.std(r.ecd_per_grain)                    # N - 1, not N


def test_std_of_a_single_grain_is_zero_not_nan():
    lab = np.ones((5, 5), dtype=np.int64)
    rng = np.random.default_rng(2)
    m = {"quats": random_quaternions(rng, 2)[lab], "phase_id": np.zeros((5, 5), dtype=np.int64),
         "phase_names": np.array(["Al", "Al7FeCu2"]), "point_groups": np.array(["m-3m", "4/mmm"]), "step": 1.0}
    ds = EBSDDataset(to_xmap(m), step_size=1.0)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, alpha=0.0, fill=False)
    r = analyze_grain_size(ds, g, edge_rule="keep_all")
    assert r.n_grains_used == 1
    for v in (r.std_ECD, r.std_maxDimX, r.std_maxDimY, r.std_AR):
        assert v == 0.0 and not np.isnan(v)


# ---------------------------------------------------------------------------------------------
# the safeguard warning
# ---------------------------------------------------------------------------------------------
def test_safeguard_warning_is_logged_once_per_settings_change(caplog):
    ds, g, *_ = _tile_map(6, 12, 8)                            # own grains: the warning is keyed to them
    with caplog.at_level(logging.WARNING, logger="analysis.grain_analysis"):
        for _ in range(3):
            analyze_grain_size(ds, g, stats_phase=0)
        n_first = sum("keeping all grains" in r.getMessage() for r in caplog.records)
        analyze_grain_size(ds, g, stats_phase=1)               # a settings change: it is said again
        n_second = sum("keeping all grains" in r.getMessage() for r in caplog.records)
        analyze_grain_size(ds, g, stats_phase=1)
        n_third = sum("keeping all grains" in r.getMessage() for r in caplog.records)
    assert (n_first, n_second, n_third) == (1, 2, 2)


# ---------------------------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def client_two_phase():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    ds, _, _, _ = _tile_map(12, 12, 8)
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


def _reconstruct(client, expect=200, **extra):
    body = {"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 0.0, "fill": False, **extra}
    r = client.post("/api/analysis/grains/reconstruct", json=body)
    assert r.status_code == expect, r.text
    if expect != 200:
        return r
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    return task["result"]


def test_reconstruct_takes_the_phase_and_says_it_in_the_result(client_two_phase):
    client, ar = client_two_phase
    res = _reconstruct(client, stats_phase=1)
    assert (res["stats_phase"], res["stats_phase_name"]) == (1, "Al7FeCu2")
    assert (res["n_grains_total"], res["n_edge_grains"], res["n_grains_used"]) == (96, 36, 60)
    assert ar._grain_stats_settings["stats_phase"] == 1
    expect = analyze_grain_size(ar._dataset, ar._dataset.grains, stats_phase=1)
    assert res["median_ecd"] == pytest.approx(float(np.median(expect.ecd_per_grain)))
    assert res["median_aspect_ratio"] == pytest.approx(float(np.median(expect.ar_per_grain)))
    assert res["mean_ecd"] == pytest.approx(expect.avg_ECD)


def test_reconstruct_default_is_all_phases(client_two_phase):
    client, ar = client_two_phase
    res = _reconstruct(client)
    assert res["stats_phase"] is None and res["stats_phase_name"] is None
    assert res["n_grains_total"] == 240
    assert ar._grain_stats_settings["stats_phase"] is None


def test_grain_size_returns_the_medians_and_the_script_columns(client_two_phase):
    client, ar = client_two_phase
    _reconstruct(client, edge_rule="keep_all", ecd_basis="measured_pixels")
    body = client.post("/api/analysis/grain-size").json()
    expect = analyze_grain_size(ar._dataset, ar._dataset.grains, edge_rule="keep_all", ecd_basis="measured_pixels")
    assert body["median_ecd"] == pytest.approx(float(np.median(expect.ecd_per_grain)))
    assert body["median_aspect_ratio"] == pytest.approx(float(np.median(expect.ar_per_grain)))
    assert body["std_maxDimX"] == pytest.approx(expect.std_maxDimX)
    assert body["std_maxDimY"] == pytest.approx(expect.std_maxDimY)
    assert body["std_aspect_ratio"] == pytest.approx(expect.std_AR)
    assert body["area_weighted_ecd"] == pytest.approx(expect.area_weighted_ECD)
    assert (body["stats_phase"], body["stats_phase_name"]) == (None, None)


def test_grain_size_phase_is_remembered_omitted_keeps_and_null_means_all(client_two_phase):
    client, ar = client_two_phase
    _reconstruct(client)
    r = client.post("/api/analysis/grain-size", json={"stats_phase": 0}).json()
    assert (r["stats_phase"], r["n_grains_total"]) == (0, 144)
    assert ar._grain_stats_settings["stats_phase"] == 0
    r = client.post("/api/analysis/grain-size", json={"edge_rule": "keep_all"}).json()     # field left out
    assert (r["stats_phase"], r["n_grains_total"]) == (0, 144)
    r = client.post("/api/analysis/grain-size", json={"stats_phase": "most_measured"}).json()
    assert (r["stats_phase"], r["stats_phase_name"]) == (0, "Al")
    assert ar._grain_stats_settings["stats_phase"] == "most_measured"                      # the choice is kept
    r = client.post("/api/analysis/grain-size", json={"stats_phase": None}).json()          # explicit null
    assert (r["stats_phase"], r["n_grains_total"]) == (None, 240)
    assert ar._grain_stats_settings["stats_phase"] is None


def test_unknown_phase_id_is_a_400_with_the_known_ids_and_changes_nothing(client_two_phase):
    client, ar = client_two_phase
    _reconstruct(client, stats_phase=1)
    before = dict(ar._grain_stats_settings)
    grains_before = ar._dataset.grains
    r = client.post("/api/analysis/grain-size", json={"stats_phase": 9})
    assert r.status_code == 400
    assert "0" in r.json()["detail"] and "1" in r.json()["detail"] and "9" in r.json()["detail"]
    r = _reconstruct(client, expect=400, stats_phase=9)
    assert "[0, 1]" in r.json()["detail"]
    assert ar._grain_stats_settings == before and ar._dataset.grains is grains_before


@pytest.mark.parametrize("value", ["bogus", 1.5, [1], True])
def test_malformed_phase_is_a_422(client_two_phase, value):
    client, ar = client_two_phase
    _reconstruct(client)
    assert client.post("/api/analysis/grain-size", json={"stats_phase": value}).status_code == 422
    body = {"misorientation_threshold": 5.0, "min_grain_size": 1, "stats_phase": value}
    assert client.post("/api/analysis/grains/reconstruct", json=body).status_code == 422


def test_status_and_load_offer_the_phases_with_ids(client_two_phase):
    client, ar = client_two_phase
    options = client.get("/api/analysis/status").json()["phase_options"]
    # "cubic": whether the texture components (cubic) can be computed on the phase; the Texture tab marks the others
    assert options == [{"id": 0, "name": "Al", "cubic": True}, {"id": 1, "name": "Al7FeCu2", "cubic": False}]


# ---------------------------------------------------------------------------------------------
# sheet 1: real parameters, new rows, no accidental zeros
# ---------------------------------------------------------------------------------------------
NEW_ROWS = ["std_maxDimX", "std_maxDimY", "std_AR", "area_weighted_ECD", "grain_stats_phase"]


def test_sheet1_labels_and_new_rows(tmp_path):
    from analysis.excel_exporter import ExcelExporter
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ws = ex.wb[ex.wb.sheetnames[0]]
    assert [ws.cell(row=r, column=1).value for r in range(54, 61)] == [
        "edge_safeguard_applied", *NEW_ROWS, "rx_gos_threshold_deg"]
    info = {"std_maxDimX": 1.5, "std_maxDimY": 2.5, "std_AR": 0.5, "area_weighted_ECD": 9.0,
            "grain_stats_phase": "Al"}
    ex.export_dataset("ds", {"basic_info": info})
    assert [ws.cell(row=r, column=2).value for r in range(55, 60)] == [1.5, 2.5, 0.5, 9.0, "Al"]
    ex.export_dataset("empty", {"basic_info": {}})
    assert [ws.cell(row=r, column=3).value for r in range(55, 60)] in ([None] * 5, [""] * 5)


def _sheet1_column(path, col=2):
    from openpyxl import load_workbook
    ws = load_workbook(path)["Sheet1"]
    return {ws.cell(row=r, column=1).value: ws.cell(row=r, column=col).value for r in range(2, 70)}


def test_route_export_states_the_real_reconstruction_parameters(client_two_phase, tmp_path, monkeypatch):
    client, ar = client_two_phase
    import analysis.texture_analysis as ta
    monkeypatch.setattr(ta, "analyze_texture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped")))
    _reconstruct(client, misorientation_threshold=7.0, min_grain_size=2, alpha=2.5, min_intercept_um=0.9,
                 stats_phase=1)
    info = ar._collect_export_results()["basic_info"]
    assert (info["gb_threshold_deg"], info["min_pixels"], info["min_intercept_um"]) == (7.0, 2, 0.9)
    assert info["grain_alpha"] == 2.5
    gs = analyze_grain_size(ar._dataset, ar._dataset.grains, stats_phase=1)
    assert info["grain_stats_phase"] == "Al7FeCu2"
    assert info["std_maxDimX"] == pytest.approx(gs.std_maxDimX) and info["std_maxDimY"] == pytest.approx(gs.std_maxDimY)
    assert info["std_AR"] == pytest.approx(gs.std_AR)
    assert info["area_weighted_ECD"] == pytest.approx(gs.area_weighted_ECD)
    # the keys the exporter reads but the route never filled (they were 0 by accident)
    assert info["median_ECD_um"] == pytest.approx(float(np.median(gs.ecd_per_grain)))
    assert info["median_AR"] == pytest.approx(float(np.median(gs.ar_per_grain)))
    for edge, key in ((5.0, "grains_above_5um"), (10.0, "grains_above_10um"), (15.0, "grains_above_15um")):
        assert info[key] == int(np.sum(gs.ecd_per_grain > edge))

    from analysis.excel_exporter import ExcelExporter
    ex = ExcelExporter(str(tmp_path / "route.xlsx"))
    ex.export_dataset("ds", ar._collect_export_results())
    ex.save()
    col = _sheet1_column(str(tmp_path / "route.xlsx"))
    assert (col["gb_threshold_deg"], col["min_pixels"], col["min_intercept_um (not applied)"], col["grain_alpha"]) == (7.0, 2, 0.9, 2.5)
    assert col["grain_stats_phase"] == "Al7FeCu2" and col["median_ECD_um"] == pytest.approx(info["median_ECD_um"])
    assert col["std_AR"] == pytest.approx(gs.std_AR)


def test_route_export_without_a_phase_says_all_phases(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    import analysis.texture_analysis as ta
    monkeypatch.setattr(ta, "analyze_texture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped")))
    _reconstruct(client)
    assert ar._collect_export_results()["basic_info"]["grain_stats_phase"] == "all phases"


def test_route_export_without_remembered_parameters_does_not_invent_them(client_two_phase, monkeypatch):
    """Grains that were not made by the reconstruct route: the sheet does not claim 5 / 5 / 1.5."""
    client, ar = client_two_phase
    import analysis.texture_analysis as ta
    monkeypatch.setattr(ta, "analyze_texture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped")))
    ar._dataset.grains = reconstruct_grains(ar._dataset, angle_threshold_deg=5.0, min_pixels=1, alpha=0.0, fill=False)
    info = ar._collect_export_results()["basic_info"]
    assert info["gb_threshold_deg"] == "" and info["min_pixels"] == "" and info["min_intercept_um"] == ""


def test_batch_records_phase_columns_and_fills_every_key_the_exporter_reads(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    monkeypatch.setattr(bp, "analyze_texture", lambda *a, **k: None)
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), grain_edge_rule="keep_all",
                             grain_stats_phase="most_measured")
    res = proc.process_file(tmp_path / "s.h5")
    gs, info = res["grain_size"], res["basic_info"]
    assert gs.stats_phase is not None
    assert info["grain_stats_phase"] == gs.stats_phase_name
    assert info["std_maxDimX"] == pytest.approx(gs.std_maxDimX) and info["std_AR"] == pytest.approx(gs.std_AR)
    assert info["area_weighted_ECD"] == pytest.approx(gs.area_weighted_ECD)
    assert info["median_ECD_um"] == pytest.approx(float(np.median(gs.ecd_per_grain)))      # was never filled
    assert info["gb_threshold_deg"] == proc.gb_threshold_deg and info["min_pixels"] == proc.min_grain_pixels
    default = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"))
    assert default.grain_stats_phase is None
    assert default.process_file(tmp_path / "s.h5")["basic_info"]["grain_stats_phase"] == "all phases"
    for bad in ("Al", 0, "bogus"):
        with pytest.raises(ValueError):
            bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), grain_stats_phase=bad)


def test_batch_route_passes_the_phase_only_when_given(client_two_phase, monkeypatch):
    client, _ = client_two_phase
    import analysis.batch_processor as bp
    from analysis.batch_processor import BatchResult
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder", lambda folder, out, progress_callback=None, **kw: seen.append(kw) or BatchResult(processed=["a.h5"], n_files=1))
    for body in ({"folder_path": "f", "output_path": "o"},
                 {"folder_path": "f", "output_path": "o", "grain_stats_phase": "most_measured"}):
        r = client.post("/api/analysis/batch", json=body)
        assert r.status_code == 200
        client.get(f"/api/analysis/task/{r.json()['task_id']}")
    assert seen == [{}, {"grain_stats_phase": "most_measured"}]
    r = client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o", "grain_stats_phase": "Al"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------------------------
# ties and the phase of a grain
# ---------------------------------------------------------------------------------------------
def test_most_measured_tie_goes_to_the_lower_phase_id():
    """Both phases have the same number of measured pixels: the lower id wins, whatever the order."""
    ds, g, tile_phase, _ = _tile_map(4, 6, 6)
    assert (tile_phase == 0).sum() == (tile_phase == 1).sum()
    r = analyze_grain_size(ds, g, edge_rule="keep_all", stats_phase="most_measured")
    assert (r.stats_phase, r.stats_phase_name) == (0, "Al")


def test_the_safeguard_warning_is_said_again_after_a_run_without_it(caplog):
    """phase 0 (warns), all phases (no safeguard), phase 0 again: warns a second time."""
    ds, g, *_ = _tile_map(6, 12, 8)

    def said():
        return sum("keeping all grains" in r.getMessage() for r in caplog.records)

    with caplog.at_level(logging.WARNING, logger="analysis.grain_analysis"):
        analyze_grain_size(ds, g, stats_phase=0)
        assert said() == 1
        assert analyze_grain_size(ds, g).edge_safeguard_applied is False
        analyze_grain_size(ds, g, stats_phase=0)
        assert said() == 2


def test_min_intercept_is_written_as_entered_and_labelled_not_applied(client_two_phase, tmp_path, monkeypatch):
    from analysis.excel_exporter import ExcelExporter
    client, ar = client_two_phase
    import analysis.texture_analysis as ta
    monkeypatch.setattr(ta, "analyze_texture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped")))
    _reconstruct(client, min_intercept_um=0.9)
    ex = ExcelExporter(str(tmp_path / "route.xlsx"))
    ex.export_dataset("ds", ar._collect_export_results())
    ex.save()
    col = _sheet1_column(str(tmp_path / "route.xlsx"))
    assert col["min_intercept_um (not applied)"] == 0.9
    assert "min_intercept_um" not in col


def test_batch_min_intercept_is_labelled_not_applied_too(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    from analysis.excel_exporter import ExcelExporter
    monkeypatch.setattr(bp, "analyze_texture", lambda *a, **k: None)
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), min_intercept_um=2.25)
    ex = ExcelExporter(str(tmp_path / "batch.xlsx"))
    results = proc.process_file(tmp_path / "s.h5")
    results.pop("texture", None)                    # None here: the texture step is stubbed out
    ex.export_dataset("ds", results)
    ex.save()
    assert _sheet1_column(str(tmp_path / "batch.xlsx"))["min_intercept_um (not applied)"] == 2.25


def test_no_reconstruction_parameters_survive_the_grains(client_two_phase, monkeypatch):
    """The quality filter drops the grains and with them the parameters they were made with; grains made
    some other way afterwards are exported with empty parameter cells, not the old values."""
    client, ar = client_two_phase
    import analysis.texture_analysis as ta
    monkeypatch.setattr(ta, "analyze_texture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped")))
    _reconstruct(client, misorientation_threshold=7.0, min_grain_size=2, alpha=2.5, min_intercept_um=0.9)
    assert ar._dataset.grain_recon_params["gb_threshold_deg"] == 7.0
    ar._dataset.xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, ar._dataset.xmap.size)
    ar._dataset.apply_quality_filter(bc_min=1)
    assert ar._dataset.grains is None and ar._dataset.grain_recon_params is None
    ar._dataset.grains = reconstruct_grains(ar._dataset, angle_threshold_deg=5.0, min_pixels=1, alpha=0.0, fill=False)
    info = ar._collect_export_results()["basic_info"]
    assert [info[k] for k in ("gb_threshold_deg", "min_pixels", "min_intercept_um", "grain_alpha")] == [""] * 4
    # and stale parameters next to grains that are gone are not read either
    ar._dataset.grain_recon_params = {"gb_threshold_deg": 7.0}
    ar._dataset.grains = None
    assert ar._collect_export_results()["basic_info"]["gb_threshold_deg"] == ""


def test_a_missing_basic_info_key_is_an_empty_cell_not_a_zero(tmp_path, big):
    """Sheet 1 rows 2-11 and the five rows 13, 18-21 that come from basic_info."""
    from analysis.excel_exporter import ExcelExporter
    ds, g, *_ = big
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ws = ex.wb[ex.wb.sheetnames[0]]
    ex.export_dataset("ds", {"basic_info": {}, "grain_size": analyze_grain_size(ds, g)})
    for row in list(range(2, 12)) + [13, 18, 19, 20, 21]:
        assert ws.cell(row=row, column=2).value in (None, ""), (row, ws.cell(row=row, column=1).value)
    # a key that is there, even 0, is written
    ex.export_dataset("zero", {"basic_info": {"dim_x": 0, "grains_above_15um": 0},
                               "grain_size": analyze_grain_size(ds, g)})
    assert ws.cell(row=3, column=3).value == 0 and ws.cell(row=21, column=3).value == 0


# ---------------------------------------------------------------------------------------------
# What a chosen phase does to the rules, numbers that can tell grain sets apart
# ---------------------------------------------------------------------------------------------
def _expected_edge(tile_phase, rule, phase):
    """Edge tiles of the 12 x (12 + 8) tile map from the geometry alone. ``matlab_script`` works on the
    chosen phase's own extent; ``mtex`` and ``outer_row`` on the whole map (other phases are solid)."""
    rows, cols = tile_phase.shape
    ty, tx = np.indices(tile_phase.shape)
    sel = np.ones(tile_phase.shape, dtype=bool) if phase is None else tile_phase == phase
    if rule == "matlab_script":
        lo_r, hi_r, lo_c, hi_c = ty[sel].min(), ty[sel].max(), tx[sel].min(), tx[sel].max()
    else:
        lo_r, hi_r, lo_c, hi_c = 0, rows - 1, 0, cols - 1
    return sel & ((ty == lo_r) | (ty == hi_r) | (tx == lo_c) | (tx == hi_c))


@pytest.mark.parametrize("rule", ["matlab_script", "mtex", "outer_row"])
@pytest.mark.parametrize("phase", [None, 0, 1])
def test_each_rule_takes_the_geometry_it_is_defined_on(big, rule, phase):
    ds, g, tile_phase, tile_gid = big
    flagged = find_edge_grains(ds, g, rule, phase=phase)
    want = _expected_edge(tile_phase, rule, phase)
    np.testing.assert_array_equal(flagged[tile_gid - 1], want)
    assert flagged.sum() == want.sum()                              # nothing flagged outside the tiles


def test_a_chosen_phase_treats_the_other_phase_differently_per_rule(big):
    """The map is 12 x 20 tiles, phase 0 left (12 columns), phase 1 right (8). Counts by hand:
    phase 1 has 96 tiles; its own ring is 8 + 8 + 10 + 10 = 36 (matlab_script), the tiles touching the
    map border are 8 + 8 + 10 = 26 (mtex, outer_row): its 10 tiles next to phase 0 are not edge grains
    there. Phase 0 (144 tiles): own ring 44, map border 12 + 12 + 10 = 34."""
    ds, g, tile_phase, tile_gid = big
    count = {(rule, p): int(find_edge_grains(ds, g, rule, phase=p).sum())
             for rule in ("matlab_script", "mtex", "outer_row") for p in (0, 1)}
    assert count == {("matlab_script", 1): 36, ("mtex", 1): 26, ("outer_row", 1): 26,
                     ("matlab_script", 0): 44, ("mtex", 0): 34, ("outer_row", 0): 34}
    interface_1 = tile_gid[1:-1, 12] - 1                    # phase 1 next to phase 0
    interface_0 = tile_gid[1:-1, 11] - 1                    # phase 0 next to phase 1
    assert find_edge_grains(ds, g, "matlab_script", phase=1)[interface_1].all()
    assert find_edge_grains(ds, g, "matlab_script", phase=0)[interface_0].all()
    for rule in ("mtex", "outer_row"):
        assert not find_edge_grains(ds, g, rule, phase=1)[interface_1].any()
        assert not find_edge_grains(ds, g, rule, phase=0)[interface_0].any()
    # through the statistics: the counts are the ones above, rule by rule
    for rule in ("matlab_script", "mtex", "outer_row"):
        r = analyze_grain_size(ds, g, edge_rule=rule, stats_phase=1)
        assert (r.n_grains_total, r.n_edge_grains) == (96, count[(rule, 1)])


def test_phase_name_falls_back_to_the_id():
    from analysis.grain_analysis import phase_name
    ds, g, *_ = _tile_map(2, 2, 2)
    assert phase_name(ds, 0) == "Al"
    rng = np.random.default_rng(0)
    m = {"quats": random_quaternions(rng, 4).reshape(2, 2, 4).repeat(2, 0).repeat(2, 1),
         "phase_id": np.array([[0, 0, 1, 1]] * 4), "phase_names": np.array(["", "Al7FeCu2"]),
         "point_groups": np.array(["m-3m", "4/mmm"]), "step": 1.0}
    ds2 = EBSDDataset(to_xmap(m), step_size=1.0)
    assert phase_name(ds2, 0) == "phase 0" and phase_name(ds2, 1) == "Al7FeCu2"


# --- tiles of different size, expected numbers computed independently from the geometry -------------
STEP = 0.5


def _rect_spec(tile0, width0, tile1, width1, rows=48):
    """[(r0, r1, c0, c1, phase, hole)]: phase 0 on the left in tiles of tile0 pixels, phase 1 on the right
    in tiles of tile1; in every third tile row (i % 3 == 1) two tiles are one grain of double width; some
    tiles have a 2 x 2 hole in the middle (closed by the reconstruction, so cell area != measured pixels)."""
    rects = []
    for phase, tile, c_start, width in ((0, tile0, 0, width0), (1, tile1, width0, width1)):
        for i in range(rows // tile):
            j = 0
            while j < width // tile:
                merged = i % 3 == 1 and j == 2
                w = 2 * tile if merged else tile
                hole = (not merged) and (i + j) % 5 == 0
                rects.append((i * tile, (i + 1) * tile, c_start + j * tile, c_start + j * tile + w, phase, hole))
                j += 2 if merged else 1
    return rects


def _rect_dataset(rects, seed=11):
    ny, nx = max(r[1] for r in rects), max(r[3] for r in rects)
    rng = np.random.default_rng(seed)
    gq = random_quaternions(rng, len(rects))
    q = np.zeros((ny, nx, 4))
    phase = np.zeros((ny, nx), dtype=np.int64)
    for k, (r0, r1, c0, c1, ph, hole) in enumerate(rects):
        q[r0:r1, c0:c1] = gq[k]
        phase[r0:r1, c0:c1] = ph
        if hole:
            a, b = r0 + (r1 - r0 - 2) // 2, c0 + (c1 - c0 - 2) // 2
            phase[a:a + 2, b:b + 2] = -1
    m = {"quats": q, "phase_id": phase, "phase_names": np.array(["Al", "Al7FeCu2"]),
         "point_groups": np.array(["m-3m", "4/mmm"]), "step": STEP}
    return EBSDDataset(to_xmap(m), step_size=STEP), m


def _expected(rects, phase, basis, step=STEP):
    """The statistics of the ``matlab_script`` rule computed from the rectangles with plain numpy."""
    R = np.array([r[:4] for r in rects])
    P = np.array([r[4] for r in rects])
    sel = np.ones(len(rects), dtype=bool) if phase is None else P == phase
    edge = sel & ((R[:, 0] <= R[sel, 0].min() + 1) | (R[:, 1] - 1 >= R[sel, 1].max() - 2)
                  | (R[:, 2] <= R[sel, 2].min() + 1) | (R[:, 3] - 1 >= R[sel, 3].max() - 2))
    n_total, n_edge = int(sel.sum()), int(edge.sum())
    remaining = n_total - n_edge
    assert remaining >= 50 and remaining / n_total >= 0.15          # the safeguard does not apply to these maps
    valid = sel & ~edge
    h, w = (R[:, 1] - R[:, 0]) * step, (R[:, 3] - R[:, 2]) * step
    holes = np.array([4 if r[5] else 0 for r in rects])
    area = h * w if basis == "cell_area" else h * w - holes * step ** 2
    ecd = 2.0 * np.sqrt(area / np.pi)
    ar = np.maximum(h, w) / np.minimum(h, w)
    v = valid
    return {
        "n_total": n_total, "n_edge": n_edge, "n_used": int(v.sum()),
        "mean_ecd": ecd[v].mean(), "median_ecd": np.median(ecd[v]), "std_ecd": ecd[v].std(ddof=1),
        "mean_maxDimX": w[v].mean(), "std_maxDimX": w[v].std(ddof=1),
        "mean_maxDimY": h[v].mean(), "std_maxDimY": h[v].std(ddof=1),
        "mean_aspect_ratio": ar[v].mean(), "median_aspect_ratio": np.median(ar[v]),
        "std_aspect_ratio": ar[v].std(ddof=1),
        "area_weighted_ecd": float(np.sum(ecd[v] * area[v]) / np.sum(area[v])),
    }


COMBOS = [(p, b) for p in (None, 0, 1) for b in ("cell_area", "measured_pixels")]
RECTS_A = _rect_spec(4, 48, 6, 72)           # phase 1 is the larger one
RECTS_B = _rect_spec(6, 72, 4, 48)           # phase 0 is the larger one


def test_the_expected_numbers_can_tell_the_grain_sets_and_bases_apart():
    seen = {}
    for p, b in COMBOS:
        e = _expected(RECTS_A, p, b)
        seen[(p, b)] = (round(e["median_ecd"], 6), round(e["std_ecd"], 6), round(e["std_maxDimX"], 6),
                        round(e["std_aspect_ratio"], 6), round(e["area_weighted_ecd"], 6))
    assert len(set(seen.values())) == len(COMBOS)
    e0 = _expected(RECTS_A, 0, "cell_area")
    assert (e0["n_total"], e0["n_edge"], e0["n_used"]) == (140, 44, 96)
    e1 = _expected(RECTS_A, 1, "cell_area")
    assert (e1["n_total"], e1["n_edge"], e1["n_used"]) == (93, 35, 58)    # the bottom row (i = 7) has a merged tile: 11 tiles, not 12


@pytest.fixture(scope="module")
def rect_dataset_a():
    ds, _ = _rect_dataset(RECTS_A)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)
    assert g.n_grains == len(RECTS_A) and int(g.n_closed.sum()) == 4 * sum(r[5] for r in RECTS_A)
    return ds, g


@pytest.mark.parametrize("phase,basis", COMBOS)
def test_library_statistics_equal_the_independent_numbers(rect_dataset_a, phase, basis):
    ds, g = rect_dataset_a
    r = analyze_grain_size(ds, g, stats_phase=phase, ecd_basis=basis)
    e = _expected(RECTS_A, phase, basis)
    assert (r.n_grains_total, r.n_edge_grains, r.n_grains_used) == (e["n_total"], e["n_edge"], e["n_used"])
    assert r.edge_safeguard_applied is False
    for got, key in ((r.avg_ECD, "mean_ecd"), (np.median(r.ecd_per_grain), "median_ecd"), (r.std_ECD, "std_ecd"),
                     (r.avg_maxDimX, "mean_maxDimX"), (r.std_maxDimX, "std_maxDimX"),
                     (r.avg_maxDimY, "mean_maxDimY"), (r.std_maxDimY, "std_maxDimY"),
                     (r.avg_AR, "mean_aspect_ratio"), (np.median(r.ar_per_grain), "median_aspect_ratio"),
                     (r.std_AR, "std_aspect_ratio"), (r.area_weighted_ECD, "area_weighted_ecd")):
        assert got == pytest.approx(e[key], rel=1e-9), (key, phase, basis)


@pytest.fixture
def rect_client():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    ds, _ = _rect_dataset(RECTS_A)
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


SHEET_ROWS = {"mean_ecd": "mean_ECD_um", "median_ecd": "median_ECD_um", "std_ecd": "std_ECD_um",
              "mean_maxDimX": "mean_MaxDimX_um", "mean_maxDimY": "mean_MaxDimY_um",
              "mean_aspect_ratio": "mean_AR", "median_aspect_ratio": "median_AR",
              "std_maxDimX": "std_maxDimX", "std_maxDimY": "std_maxDimY", "std_aspect_ratio": "std_AR",
              "area_weighted_ecd": "area_weighted_ECD"}


def _sheet_by_label(path):
    from openpyxl import load_workbook
    ws = load_workbook(path)["Sheet1"]
    return ws, {ws.cell(row=r, column=1).value: r for r in range(2, 70) if ws.cell(row=r, column=1).value}


def test_route_answer_and_excel_cells_equal_the_independent_numbers(rect_client, tmp_path, monkeypatch):
    from analysis.excel_exporter import ExcelExporter
    client, ar = rect_client
    import analysis.texture_analysis as ta
    monkeypatch.setattr(ta, "analyze_texture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped")))
    _reconstruct(client, misorientation_threshold=5.0, min_grain_size=3, alpha=3.1, fill=False)
    ex = ExcelExporter(str(tmp_path / "phases.xlsx"))
    for phase, basis in COMBOS:
        body = client.post("/api/analysis/grain-size", json={"stats_phase": phase, "ecd_basis": basis}).json()
        e = _expected(RECTS_A, phase, basis)
        assert (body["n_grains_total"], body["n_edge_grains"], body["n_grains_used"]) == (
            e["n_total"], e["n_edge"], e["n_used"])
        for key in SHEET_ROWS:
            assert body[key] == pytest.approx(e[key], rel=1e-9), (key, phase, basis)
        ex.export_dataset(f"{phase}-{basis}", ar._collect_export_results())
    ex.save()
    ws, row_of = _sheet_by_label(str(tmp_path / "phases.xlsx"))
    for col, (phase, basis) in enumerate(COMBOS, start=2):
        e = _expected(RECTS_A, phase, basis)
        for key, label in SHEET_ROWS.items():
            assert ws.cell(row=row_of[label], column=col).value == pytest.approx(e[key], rel=1e-9), (label, phase, basis)
        assert ws.cell(row=row_of["n_grains_used"], column=col).value == e["n_used"]
        assert ws.cell(row=row_of["grain_stats_phase"], column=col).value == (
            "all phases" if phase is None else ["Al", "Al7FeCu2"][phase])


def test_batch_cells_equal_the_independent_numbers_for_two_matrix_phases(tmp_path, monkeypatch):
    """Two files, the larger phase is 1 in the first and 0 in the second: with most_measured each column
    carries its own phase name and the numbers of that phase."""
    from analysis import batch_processor as bp
    from analysis.excel_exporter import ExcelExporter
    monkeypatch.setattr(bp, "analyze_texture", lambda *a, **k: None)
    maps = {}
    for stem, rects in (("a", RECTS_A), ("b", RECTS_B)):
        _, m = _rect_dataset(rects)
        xmap = to_xmap(m)
        xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
        maps[stem] = xmap
    patch_batch_load(monkeypatch, bp, lambda path: maps[Path(path).stem])
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), grain_stats_phase="most_measured")
    ex = ExcelExporter(str(tmp_path / "batch.xlsx"))
    for stem in ("a", "b"):
        results = proc.process_file(tmp_path / f"{stem}.h5")
        results.pop("texture", None)                                   # None here: the texture step is stubbed out
        ex.export_dataset(stem, results)
    ex.save()
    ws, row_of = _sheet_by_label(str(tmp_path / "batch.xlsx"))
    for col, (rects, phase, name) in enumerate(((RECTS_A, 1, "Al7FeCu2"), (RECTS_B, 0, "Al")), start=2):
        e = _expected(rects, phase, "cell_area")
        assert ws.cell(row=row_of["grain_stats_phase"], column=col).value == name
        assert ws.cell(row=row_of["n_grains_used"], column=col).value == e["n_used"]
        for key, label in SHEET_ROWS.items():
            assert ws.cell(row=row_of[label], column=col).value == pytest.approx(e[key], rel=1e-9), (label, name)


def test_excel_rows_55_to_59_come_from_the_result_when_there_is_one(tmp_path, rect_dataset_a):
    from analysis.excel_exporter import ExcelExporter
    ds, g = rect_dataset_a
    gs = analyze_grain_size(ds, g, stats_phase=1)
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ws = ex.wb[ex.wb.sheetnames[0]]
    wrong = {"std_maxDimX": -1, "std_maxDimY": -1, "std_AR": -1, "area_weighted_ECD": -1, "grain_stats_phase": "x"}
    ex.export_dataset("ds", {"basic_info": wrong, "grain_size": gs})
    assert [ws.cell(row=r, column=2).value for r in range(55, 60)] == [
        gs.std_maxDimX, gs.std_maxDimY, gs.std_AR, gs.area_weighted_ECD, "Al7FeCu2"]
    assert ws.cell(row=5, column=1).value == "n_grains (all phases)"


# --- state written atomically -----------------------------------------------------------------------
def test_a_grain_size_call_does_not_store_settings_for_grains_that_were_replaced_meanwhile(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    _reconstruct(client, ecd_basis="cell_area")
    import analysis.grain_analysis as ga
    real = ga.analyze_grain_size

    def swapping(ds, grains, **kw):
        out = real(ds, grains, **kw)
        # a reconstruction finishes while this analysis runs: other grains, its own settings
        ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, alpha=0.0, fill=False)
        ar._grain_stats_settings.update(edge_rule="outer_row", ecd_basis="measured_pixels", stats_phase=1)
        return out

    monkeypatch.setattr(ga, "analyze_grain_size", swapping)
    r = client.post("/api/analysis/grain-size", json={"stats_phase": 0, "edge_rule": "mtex"})
    assert r.status_code == 200
    assert ar._grain_stats_settings == {"edge_rule": "outer_row", "ecd_basis": "measured_pixels", "stats_phase": 1}
    # without a swap the same call is remembered
    monkeypatch.setattr(ga, "analyze_grain_size", real)
    assert client.post("/api/analysis/grain-size", json={"stats_phase": 0, "edge_rule": "mtex"}).status_code == 200
    assert (ar._grain_stats_settings["edge_rule"], ar._grain_stats_settings["stats_phase"]) == ("mtex", 0)


def test_a_reconstruction_for_a_replaced_dataset_writes_nothing(client_two_phase, monkeypatch):
    client, ar = client_two_phase
    first = ar._dataset
    other, *_ = _tile_map(6, 12, 8)
    import analysis.grain_analysis as ga
    real = ga.reconstruct_grains
    before = dict(ar._grain_stats_settings)

    def replacing(ds, **kw):
        out = real(ds, **kw)
        ar._dataset = other                              # another file is loaded while this one is reconstructed
        return out

    monkeypatch.setattr(ga, "reconstruct_grains", replacing)
    body = {"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 0.0, "fill": False, "stats_phase": 1}
    r = client.post("/api/analysis/grains/reconstruct", json=body)
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "failed" and "replaced" in task["error"]
    assert first.grains is None and first.grain_recon_params is None
    assert other.grains is None and other.grain_recon_params is None          # the new file is untouched
    assert ar._grain_stats_settings == before


# --- batch: files without grains are reported ---------------------------------------------------------
def test_a_file_without_grains_is_listed_as_skipped_with_its_reason(tmp_path, monkeypatch, caplog):
    from analysis import batch_processor as bp
    monkeypatch.setattr(bp, "analyze_texture", lambda *a, **k: None)
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    (tmp_path / "empty_result.h5").write_bytes(b"")
    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), min_grain_pixels=10 ** 6)
    with caplog.at_level(logging.WARNING):
        assert proc.process_all() == 0
    assert proc.failed_files == []
    assert len(proc.skipped_files) == 1
    assert proc.skipped_files[0]["file"].endswith("empty_result.h5") and "no grains" in proc.skipped_files[0]["reason"]
    assert any("Skipped files" in r.getMessage() and "empty_result.h5" in r.getMessage() for r in caplog.records)


def test_loading_a_map_resets_the_phase_choice_through_post_load(client_two_phase, monkeypatch):
    import orix.io
    client, ar = client_two_phase
    m = make_map(shape=(20, 30), n_seeds=5)
    monkeypatch.setattr(orix.io, "load", lambda path: to_xmap(m))
    ar._grain_stats_settings.update(edge_rule="mtex", ecd_basis="measured_pixels", stats_phase=1)
    r = client.post("/api/analysis/load", json={"xmap_path": "scan.ang"})
    assert r.status_code == 200, r.text
    assert ar._grain_stats_settings == {"edge_rule": "matlab_script", "ecd_basis": "cell_area", "stats_phase": None}


def test_sheet1_records_what_the_page_promises(tmp_path):
    """The page says (locale key analysis:batch.workbookRecords) what Sheet 1 records: the reconstruction
    parameters, the edge rule, the ECD basis, the statistics phase, the three grain counts, the safeguard flag and the
    RX criteria. If a row goes or is renamed, the sentence has to change with it."""
    from analysis.excel_exporter import ExcelExporter, GRAIN_STATS_KEYS, MIN_INTERCEPT_LABEL, RX_CRITERIA_KEYS
    for key in ("grain_edge_rule", "grain_ecd_basis", "grain_stats_phase", "n_grains_total", "n_edge_grains",
                "n_grains_used", "edge_safeguard_applied"):
        assert key in GRAIN_STATS_KEYS, key
    assert RX_CRITERIA_KEYS == ("rx_gos_threshold_deg", "rx_gbc_min_fraction", "rx_gkam_threshold_deg", "rx_min_radius_um")
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ws = ex.wb[ex.wb.sheetnames[0]]
    labels = {ws.cell(row=r, column=1).value for r in range(1, 70)}
    for label in ("gb_threshold_deg", "min_pixels", MIN_INTERCEPT_LABEL, "grain_alpha", *GRAIN_STATS_KEYS, *RX_CRITERIA_KEYS):
        assert label in labels, label
    # fill is not recorded, and the sentence does not claim it
    assert not any("fill" in str(label) for label in labels if label)
