"""Edge-grain rules and ECD basis of the grain-size statistics.

Four rules decide which grains the statistics leave out: ``matlab_script`` (the default; the rule of
matlab_testskripts/GrainSizeAnalysis_mtex6.m), ``mtex`` (MTEX ``isBoundary``), ``outer_row`` (the earlier
Orienta rule) and ``keep_all``. The ECD basis is a separate choice: the cell area of the grain or the
number of its measured pixels times the step squared.
"""
from pathlib import Path

import numpy as np
import pytest
from tests.batch_load_stub import patch_batch_load
from fastapi.testclient import TestClient

from analysis.ebsd_dataset import EBSDDataset, GrainSet
from analysis.grain_analysis import (
    ECD_BASES,
    EDGE_RULES,
    GrainSizeResult,
    analyze_grain_size,
    edge_safeguard_triggered,
    find_edge_grains,
    reconstruct_grains,
)
from tests.synthetic_grain_map import make_map, random_quaternions, to_xmap

FIX = Path(__file__).parent / "fixtures" / "analysis_mtex_parity" / "synthetic_two_phase.npz"
PARAMS = {"p0": (5.0, 3, 3.1), "p1": (5.0, 3, 1.5), "p2": (15.0, 5, 3.1)}
SIX = ("edge_rule", "ecd_basis", "n_grains_total", "n_edge_grains", "n_grains_used", "edge_safeguard_applied")


# ---------------------------------------------------------------------------------------------
# small constructed maps
# ---------------------------------------------------------------------------------------------
def _dataset(labels, step=1.0):
    """A map whose pixels carry the grain label ``labels`` (0 = not indexed): one random orientation per
    label, one phase. Neighbouring labels differ by far more than the 5 degree threshold."""
    labels = np.asarray(labels)
    ny, nx = labels.shape
    n = int(labels.max())
    rng = np.random.default_rng(12)
    gq = random_quaternions(rng, n + 1)
    q = gq[labels]
    q[labels == 0] = np.array([1.0, 0.0, 0.0, 0.0])
    m = {"quats": q, "phase_id": np.where(labels > 0, 0, -1).astype(np.int64),
         "phase_names": np.array(["Al", "Al7FeCu2"]), "point_groups": np.array(["m-3m", "4/mmm"]), "step": step}
    return EBSDDataset(to_xmap(m), step_size=step)


def _tiles(rows, cols, size, holes=()):
    """rows x cols square tiles of ``size`` pixels, tile (i, j) has label i * cols + j + 1; tiles in
    ``holes`` are not indexed. Returns (labels, {(i, j): label})."""
    lab = np.zeros((rows * size, cols * size), dtype=np.int64)
    of_tile = {}
    for i in range(rows):
        for j in range(cols):
            of_tile[(i, j)] = i * cols + j + 1
            if (i, j) not in holes:
                lab[i * size:(i + 1) * size, j * size:(j + 1) * size] = of_tile[(i, j)]
    return lab, of_tile


def _reconstruct(ds, alpha=0.0, min_pixels=1):
    return reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=min_pixels, alpha=alpha, fill=False)


def _grain_of_label(g, labels):
    """label -> grain number of ours (read at the label's first pixel)."""
    out = {}
    for lab in np.unique(labels[labels > 0]):
        r, c = np.argwhere(labels == lab)[0]
        out[int(lab)] = int(g.grain_ids[r, c])
    return out


def _edge_labels(ds, g, labels, rule):
    mask = find_edge_grains(ds, g, rule)
    by_label = _grain_of_label(g, labels)
    return {lab for lab, gid in by_label.items() if gid > 0 and mask[gid - 1]}


# ---------------------------------------------------------------------------------------------
# the four rules
# ---------------------------------------------------------------------------------------------
def test_constants():
    assert EDGE_RULES == ("matlab_script", "mtex", "outer_row", "keep_all")
    assert ECD_BASES == ("cell_area", "measured_pixels")


@pytest.fixture(scope="module")
def framed():
    """A one pixel frame (one grain) around a 10 x 10 field of 4 x 4 pixel grains: 101 grains in 42 x 42."""
    lab = np.zeros((42, 42), dtype=np.int64)
    for i in range(10):
        for j in range(10):
            lab[1 + 4 * i:5 + 4 * i, 1 + 4 * j:5 + 4 * j] = 2 + i * 10 + j
    lab[lab == 0] = 1
    ds = _dataset(lab)
    g = _reconstruct(ds)
    assert g.n_grains == 101
    return lab, ds, g


def test_matlab_script_band_is_two_pixels_wide(framed):
    """Measured pixels in the first two rows/columns from the extent mark their grains: the frame (row 0)
    and the tiles that start in row 1 (their row 1 is within min + 1), nothing deeper."""
    lab, ds, g = framed
    got = _edge_labels(ds, g, lab, "matlab_script")
    ring = {2 + i * 10 + j for i in range(10) for j in range(10) if i in (0, 9) or j in (0, 9)}
    assert got == ring | {1}
    assert len(got) == 37


def test_outer_row_takes_only_grains_with_a_pixel_in_the_outer_row_or_column(framed):
    lab, ds, g = framed
    assert _edge_labels(ds, g, lab, "outer_row") == {1}


def test_mtex_takes_grains_with_an_edge_to_the_exterior(framed):
    lab, ds, g = framed
    assert _edge_labels(ds, g, lab, "mtex") == {1}


def test_keep_all_takes_none(framed):
    lab, ds, g = framed
    assert not find_edge_grains(ds, g, "keep_all").any()


def test_mtex_not_indexed_area_open_to_the_border_marks_its_neighbours_an_enclosed_one_does_not():
    """6 x 6 tiles of 5 pixels, nothing closed (alpha 0). A not-indexed strip from the top border (tiles
    (0,3) and (1,3)) is open to the border: the grains next to it are edge grains. The not-indexed tile
    (3,3) is enclosed: the grains around it are not."""
    holes = {(0, 3), (1, 3), (3, 3)}
    lab, t = _tiles(6, 6, 5, holes)
    ds = _dataset(lab)
    g = _reconstruct(ds, alpha=0.0)
    assert g.n_grains == 36 - 3
    border = {t[(i, j)] for i in range(6) for j in range(6) if (i in (0, 5) or j in (0, 5)) and (i, j) not in holes}
    next_to_open = {t[(1, 2)], t[(1, 4)], t[(2, 3)]}
    next_to_enclosed = {t[(2, 3)], t[(4, 3)], t[(3, 2)], t[(3, 4)]}
    got = _edge_labels(ds, g, lab, "mtex")
    assert got == border | next_to_open
    assert (next_to_enclosed - next_to_open - border).isdisjoint(got)
    # the earlier Orienta rule sees only the pixels in the outer row and column
    assert _edge_labels(ds, g, lab, "outer_row") == border


def test_mtex_open_regions_are_four_connected():
    """A not-indexed tile that meets the strip open to the border only at a corner is enclosed: its
    own neighbours are not edge grains (6 x 6 tiles of 5 pixels; open strip (0,2), corner tile (1,3))."""
    holes = {(0, 2), (1, 3)}
    lab, t = _tiles(6, 6, 5, holes)
    ds = _dataset(lab)
    g = _reconstruct(ds, alpha=0.0)
    border = {t[(i, j)] for i in range(6) for j in range(6) if (i in (0, 5) or j in (0, 5)) and (i, j) not in holes}
    assert _edge_labels(ds, g, lab, "mtex") == border | {t[(1, 2)]}      # (1,2) lies below the open strip
    assert t[(1, 4)] not in _edge_labels(ds, g, lab, "mtex") and t[(2, 3)] not in _edge_labels(ds, g, lab, "mtex")


def test_mtex_a_measurement_dropped_by_min_pixels_counts_as_not_indexed():
    """MTEX removes grains below minPixel and treats them as not indexed. A single dropped pixel at the end
    of a one pixel wide not-indexed corridor from the top border joins the open region, and the grain
    below it, which touches only that pixel, is an edge grain. (Found on a real map: 2 of 78 grains
    disagreed with MTEX when only originally not-indexed pixels counted.)"""
    lab, t = _tiles(6, 6, 5, {(0, 3)})
    speck = int(lab.max()) + 1
    lab[5:9, 17] = 0                      # corridor in tile (1,3), open to the hole above it
    lab[9, 17] = speck                    # one pixel with an orientation of its own, dropped at min_pixels 2
    ds = _dataset(lab)
    g = _reconstruct(ds, alpha=0.0, min_pixels=2)
    assert g.culled_mask.sum() == 1
    got = _edge_labels(ds, g, lab, "mtex")
    border = {t[(i, j)] for i in range(6) for j in range(6) if (i in (0, 5) or j in (0, 5)) and (i, j) != (0, 3)}
    # tile (1,3), which the corridor cuts in two (the label is read at its first pixel), borders the corridor;
    # tile (2,3) touches only the dropped pixel
    assert got == border | {t[(1, 3)], t[(2, 3)]}


def test_mtex_needs_the_cells_and_says_so():
    ny, nx = 6, 6
    ids = np.repeat(np.arange(1, 4), 12).reshape(ny, nx)
    ds = _dataset(ids)
    g = GrainSet(grain_ids=ids, n_grains=3, ecd=np.ones(3), area=np.ones(3), aspect_ratio=np.ones(3))
    with pytest.raises(ValueError, match="cells"):
        find_edge_grains(ds, g, "mtex")
    with pytest.raises(ValueError, match="cells"):
        analyze_grain_size(ds, g, edge_rule="mtex")
    # the other rules need no cells
    assert analyze_grain_size(ds, g, edge_rule="outer_row").n_grains_total == 3


@pytest.mark.parametrize("kw", [{"edge_rule": "nonsense"}, {"ecd_basis": "nonsense"}, {"edge_rule": None}])
def test_unknown_values_raise(framed, kw):
    _, ds, g = framed
    with pytest.raises(ValueError):
        analyze_grain_size(ds, g, **kw)
    with pytest.raises(ValueError):
        find_edge_grains(ds, g, kw.get("edge_rule", "nonsense"))


def test_unknown_values_raise_even_without_grains():
    ds = _dataset(np.ones((4, 4), dtype=np.int64))
    g = GrainSet(grain_ids=np.zeros((4, 4), dtype=np.int64), n_grains=0)
    with pytest.raises(ValueError):
        analyze_grain_size(ds, g, edge_rule="nonsense")


def test_remove_edge_grains_false_is_keep_all(framed):
    _, ds, g = framed
    r = analyze_grain_size(ds, g, remove_edge_grains=False, edge_rule="mtex")
    assert r.edge_rule == "keep_all"
    assert (r.n_grains_total, r.n_edge_grains, r.n_grains_used, r.edge_safeguard_applied) == (101, 0, 101, False)


def test_result_fields_and_defaults(framed):
    _, ds, g = framed
    r = analyze_grain_size(ds, g)
    assert isinstance(r, GrainSizeResult)
    assert (r.edge_rule, r.ecd_basis) == ("matlab_script", "cell_area")
    assert (r.n_grains_total, r.n_edge_grains, r.n_grains_used, r.edge_safeguard_applied) == (101, 37, 64, False)
    assert len(r.ecd_per_grain) == len(r.maxDimX_per_grain) == len(r.maxDimY_per_grain) == len(r.ar_per_grain) == 64
    full = analyze_grain_size(ds, g, edge_rule="outer_row")
    assert (full.n_edge_grains, full.n_grains_used) == (1, 100)


def test_zero_grains_still_name_the_settings():
    ds = _dataset(np.ones((4, 4), dtype=np.int64))
    g = GrainSet(grain_ids=np.zeros((4, 4), dtype=np.int64), n_grains=0)
    r = analyze_grain_size(ds, g, edge_rule="mtex", ecd_basis="measured_pixels")
    assert (r.edge_rule, r.ecd_basis) == ("mtex", "measured_pixels")
    assert (r.n_grains_total, r.n_edge_grains, r.n_grains_used, r.edge_safeguard_applied) == (0, 0, 0, False)


# ---------------------------------------------------------------------------------------------
# safeguards
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rule", ["matlab_script", "mtex"])
def test_safeguard_needs_fifty_remaining_grains(rule):
    assert edge_safeguard_triggered(rule, n_total=100, n_edge=51) is True        # 49 remain
    assert edge_safeguard_triggered(rule, n_total=100, n_edge=50) is False       # 50 remain


@pytest.mark.parametrize("rule", ["matlab_script", "mtex"])
def test_safeguard_needs_fifteen_percent_remaining(rule):
    assert edge_safeguard_triggered(rule, n_total=1000, n_edge=851) is True      # 14.9 % remain
    assert edge_safeguard_triggered(rule, n_total=1000, n_edge=850) is False     # 15.0 % remain


def test_safeguard_outer_row_is_the_tolerance_fraction():
    assert edge_safeguard_triggered("outer_row", n_total=20, n_edge=17) is False   # 85 % exactly
    assert edge_safeguard_triggered("outer_row", n_total=20, n_edge=18) is True
    assert edge_safeguard_triggered("outer_row", n_total=20, n_edge=18, edge_tolerance_fraction=0.95) is False


def test_safeguard_never_applies_without_edge_grains_or_for_keep_all():
    assert edge_safeguard_triggered("matlab_script", n_total=10, n_edge=0) is False
    assert edge_safeguard_triggered("keep_all", n_total=10, n_edge=10) is False


def test_matlab_safeguard_on_maps_49_against_50_remaining():
    """9 x 9 tiles of 2 pixels: 32 edge grains, 49 remain -> everything stays. 7 x 12: 34 edge, 50 remain."""
    lab, _ = _tiles(9, 9, 2)
    ds = _dataset(lab)
    g = _reconstruct(ds)
    r = analyze_grain_size(ds, g)
    assert (r.n_grains_total, r.n_edge_grains, r.n_grains_used, r.edge_safeguard_applied) == (81, 32, 81, True)
    lab, _ = _tiles(7, 12, 2)
    ds = _dataset(lab)
    g = _reconstruct(ds)
    r = analyze_grain_size(ds, g)
    assert (r.n_grains_total, r.n_edge_grains, r.n_grains_used, r.edge_safeguard_applied) == (84, 34, 50, False)


def test_outer_row_safeguard_on_a_map_keeps_all():
    """Two rows of tiles: every grain touches the outer row -> more than 85 % edge grains -> all stay."""
    lab, _ = _tiles(2, 6, 3)
    ds = _dataset(lab)
    g = _reconstruct(ds)
    r = analyze_grain_size(ds, g, edge_rule="outer_row")
    assert (r.n_grains_total, r.n_edge_grains, r.n_grains_used, r.edge_safeguard_applied) == (12, 12, 12, True)


# ---------------------------------------------------------------------------------------------
# the mtex rule against MTEX's own isBoundary
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fx():
    with np.load(FIX) as f:
        return {k: f[k] for k in f.files}


@pytest.mark.parametrize("p", list(PARAMS))
def test_mtex_rule_equals_mtex_is_boundary(fx, p):
    """Every grain of the synthetic map (holes, scratch, wide area, border strip, outliers), against the
    MTEX grain it is made of: the edge flag is exactly `grains.isBoundary`."""
    m = {k: fx[k] for k in ("quats", "phase_id", "phase_names", "point_groups")}
    m["step"] = float(fx["step"])
    ds = EBSDDataset(to_xmap(m), step_size=m["step"])
    thr, minpix, alpha = PARAMS[p]
    g = reconstruct_grains(ds, angle_threshold_deg=thr, min_pixels=minpix, alpha=alpha, fill=False)
    measured = (g.grain_ids > 0) & ~g.closed_mask
    lut = np.zeros(g.n_grains + 1, dtype=np.int64)
    lut[g.grain_ids[measured]] = fx[f"{p}_grain_ids"][measured]
    t = lut[1:] - 1
    ours = find_edge_grains(ds, g, "mtex")
    theirs = fx[f"{p}_is_boundary"][t]
    assert ours.dtype == bool and len(ours) == g.n_grains
    assert theirs.any() and not theirs.all()
    np.testing.assert_array_equal(ours, theirs)


# ---------------------------------------------------------------------------------------------
# ECD basis
# ---------------------------------------------------------------------------------------------
def test_measured_pixels_ecd_and_weights():
    """With a covered hole, the cell area of a grain is larger than its measured pixels; the statistics
    on the `measured_pixels` basis use grain_size * step**2 for the ECD and for the area weights."""
    m = make_map(shape=(40, 50), n_seeds=12)
    m["phase_id"][5:7, 20:24] = -1                       # a small hole, covered by the closing
    step = 0.7
    m["step"] = step
    ds = EBSDDataset(to_xmap(m), step_size=step)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)
    assert (g.n_closed > 0).any()
    cell = analyze_grain_size(ds, g, remove_edge_grains=False)
    meas = analyze_grain_size(ds, g, remove_edge_grains=False, ecd_basis="measured_pixels")
    area_px = g.grain_size * step ** 2
    ecd_px = 2.0 * np.sqrt(area_px / np.pi)
    np.testing.assert_allclose(meas.ecd_per_grain, ecd_px, rtol=1e-12)
    assert meas.avg_ECD == pytest.approx(ecd_px.mean())
    assert meas.std_ECD == pytest.approx(ecd_px.std(ddof=1))
    assert meas.area_weighted_ECD == pytest.approx(np.sum(ecd_px * area_px) / np.sum(area_px))
    np.testing.assert_allclose(cell.ecd_per_grain, g.ecd, rtol=1e-12)
    assert cell.area_weighted_ECD == pytest.approx(np.sum(g.ecd * g.area) / np.sum(g.area))
    assert not np.allclose(cell.ecd_per_grain, meas.ecd_per_grain)       # the covered pixels make a difference
    assert (meas.edge_rule, meas.ecd_basis) == ("keep_all", "measured_pixels")


def test_measured_pixels_needs_grain_sizes():
    ds = _dataset(np.repeat(np.arange(1, 4), 12).reshape(6, 6))
    g = GrainSet(grain_ids=np.repeat(np.arange(1, 4), 12).reshape(6, 6), n_grains=3,
                 ecd=np.ones(3), area=np.ones(3), aspect_ratio=np.ones(3))
    with pytest.raises(ValueError, match="grain_size"):
        analyze_grain_size(ds, g, edge_rule="keep_all", ecd_basis="measured_pixels")


# ---------------------------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def client_with_map():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    m = make_map(shape=(60, 80), n_seeds=40)
    m["phase_id"][5:7, 20:24] = -1                      # a small hole: closed at alpha 3.1, so cell area != pixels
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = EBSDDataset(to_xmap(m), step_size=m["step"])
    ar._grain_stats_settings.update(edge_rule="matlab_script", ecd_basis="cell_area")
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


def _reconstruct_via_route(client, **extra):
    body = {"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 0.0, "fill": False, **extra}
    r = client.post("/api/analysis/grains/reconstruct", json=body)
    assert r.status_code == 200, r.text
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    return task["result"]


def test_request_models_declare_both_values():
    from backend.api.routes.analysis import BatchProcessRequest, GrainReconstructionRequest, GrainSizeRequest
    f = GrainReconstructionRequest.model_fields
    assert f["edge_rule"].default == "matlab_script" and f["ecd_basis"].default == "cell_area"
    assert GrainSizeRequest.model_fields["edge_rule"].default is None
    assert BatchProcessRequest.model_fields["grain_edge_rule"].default is None
    assert BatchProcessRequest.model_fields["grain_ecd_basis"].default is None


def test_route_defaults_name_the_rule_in_the_task_result(client_with_map):
    client, ar = client_with_map
    res = _reconstruct_via_route(client)
    assert res["edge_rule"] == "matlab_script" and res["ecd_basis"] == "cell_area"
    expected = analyze_grain_size(ar._dataset, ar._dataset.grains)
    assert res["n_grains_total"] == ar._dataset.grains.n_grains == expected.n_grains_total
    assert (res["n_edge_grains"], res["n_grains_used"], res["edge_safeguard_applied"]) == (
        expected.n_edge_grains, expected.n_grains_used, expected.edge_safeguard_applied)
    assert set(SIX) <= set(res)


def test_route_round_trip_carries_the_six_fields_and_remembers_the_settings(client_with_map):
    client, ar = client_with_map
    res = _reconstruct_via_route(client, edge_rule="outer_row", ecd_basis="measured_pixels")
    assert (res["edge_rule"], res["ecd_basis"]) == ("outer_row", "measured_pixels")
    assert ar._grain_stats_settings == {"edge_rule": "outer_row", "ecd_basis": "measured_pixels", "stats_phase": None}
    expected = analyze_grain_size(ar._dataset, ar._dataset.grains, edge_rule="outer_row", ecd_basis="measured_pixels")
    assert res["n_edge_grains"] == expected.n_edge_grains
    assert res["mean_ecd"] == pytest.approx(expected.avg_ECD)

    # no body: the settings of the last run
    r = client.post("/api/analysis/grain-size")
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["edge_rule"], body["ecd_basis"]) == ("outer_row", "measured_pixels")
    assert body["mean_ecd"] == pytest.approx(expected.avg_ECD)
    assert set(SIX) <= set(body)

    # a body changes them and is remembered
    r = client.post("/api/analysis/grain-size", json={"edge_rule": "mtex", "ecd_basis": "cell_area", "stats_phase": None})
    assert r.status_code == 200, r.text
    mt = analyze_grain_size(ar._dataset, ar._dataset.grains, edge_rule="mtex", ecd_basis="cell_area")
    assert (r.json()["edge_rule"], r.json()["n_edge_grains"], r.json()["n_grains_used"]) == ("mtex", mt.n_edge_grains, mt.n_grains_used)
    assert ar._grain_stats_settings == {"edge_rule": "mtex", "ecd_basis": "cell_area", "stats_phase": None}
    # a body with only one field keeps the other
    r = client.post("/api/analysis/grain-size", json={"ecd_basis": "measured_pixels"})
    assert r.json()["edge_rule"] == "mtex" and r.json()["ecd_basis"] == "measured_pixels"


@pytest.mark.parametrize("url,body", [
    ("/api/analysis/grains/reconstruct", {"edge_rule": "nonsense"}),
    ("/api/analysis/grains/reconstruct", {"ecd_basis": "nonsense"}),
    ("/api/analysis/grain-size", {"edge_rule": "nonsense"}),
    ("/api/analysis/grain-size", {"ecd_basis": "nonsense"}),
    ("/api/analysis/batch", {"folder_path": "x", "output_path": "y", "grain_edge_rule": "nonsense"}),
    ("/api/analysis/batch", {"folder_path": "x", "output_path": "y", "grain_ecd_basis": "nonsense"}),
])
def test_unknown_values_are_refused_with_422(client_with_map, url, body):
    client, ar = client_with_map
    before = dict(ar._grain_stats_settings)
    assert client.post(url, json=body).status_code == 422
    assert ar._grain_stats_settings == before


def test_ecd_map_follows_the_last_settings(client_with_map):
    from backend.api.routes.analysis import _per_grain_image
    client, ar = client_with_map
    _reconstruct_via_route(client, ecd_basis="measured_pixels", alpha=3.1)
    g = ar._dataset.grains
    assert (g.n_closed > 0).any()
    seen = {}
    import backend.api.routes.analysis as mod
    real = mod.colormap_array_to_base64

    def spy(data, cmap):
        seen["data"] = np.array(data)
        return real(data, cmap)

    mod.colormap_array_to_base64 = spy
    try:
        r = client.get("/api/analysis/map/ecd")
        assert r.status_code == 200, r.text
        want = _per_grain_image(2.0 * np.sqrt(g.grain_size * ar._dataset.step_size ** 2 / np.pi), g.grain_ids)
        np.testing.assert_allclose(seen["data"], want)
        ar._grain_stats_settings["ecd_basis"] = "cell_area"
        client.get("/api/analysis/map/ecd")
        np.testing.assert_allclose(seen["data"], _per_grain_image(g.ecd, g.grain_ids))
        assert not np.allclose(_per_grain_image(g.ecd, g.grain_ids), want)
    finally:
        mod.colormap_array_to_base64 = real


def test_excel_export_follows_the_last_settings(client_with_map, monkeypatch):
    client, ar = client_with_map
    # The texture step (about 90 s of orix symmetry set-up whatever the map size) has nothing to do with the
    # grain-size settings; _collect_export_results logs a failing step and goes on.
    import analysis.texture_analysis as ta
    monkeypatch.setattr(ta, "analyze_texture", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skipped")))
    _reconstruct_via_route(client, edge_rule="outer_row", ecd_basis="measured_pixels")
    collected = ar._collect_export_results()
    info, gs = collected["basic_info"], collected["grain_size"]
    assert (gs.edge_rule, gs.ecd_basis) == ("outer_row", "measured_pixels")
    assert info["grain_edge_rule"] == "outer_row" and info["grain_ecd_basis"] == "measured_pixels"
    assert info["n_grains_total"] == gs.n_grains_total and info["n_edge_grains"] == gs.n_edge_grains
    assert info["n_grains_used"] == gs.n_grains_used and info["edge_safeguard_applied"] == gs.edge_safeguard_applied


@pytest.mark.parametrize("source", ["__from_indexing__", "scan.ang"])
def test_loading_a_map_resets_the_settings(monkeypatch, source):
    """Both branches of POST /load replace the dataset: the indexing result and a file from disk."""
    import backend.api.routes.analysis as ar
    import backend.api.routes.indexing as ix
    import orix.io
    from backend.api.main import app

    class _Result:
        def __init__(self, xmap):
            self.xmap = xmap
            self.metadata = {}

    m = make_map(shape=(20, 30), n_seeds=5)
    monkeypatch.setattr(ix, "get_last_indexing_result", lambda: _Result(to_xmap(m)))
    monkeypatch.setattr(orix.io, "load", lambda path: to_xmap(m))      # the route imports it inside the function
    saved = ar._dataset, dict(ar._grain_stats_settings)
    try:
        ar._grain_stats_settings.update(edge_rule="mtex", ecd_basis="measured_pixels")
        r = TestClient(app).post("/api/analysis/load", json={"xmap_path": source})
        assert r.status_code == 200, r.text
        assert ar._grain_stats_settings == {"edge_rule": "matlab_script", "ecd_basis": "cell_area", "stats_phase": None}
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


def test_switching_back_to_a_stashed_map_resets_the_settings():
    import backend.api.routes.analysis as ar
    saved = ar._dataset, dict(ar._grain_stats_settings), dict(ar._dataset_by_file)
    try:
        ar._grain_stats_settings.update(edge_rule="outer_row", ecd_basis="measured_pixels")
        ar.restore_dataset_for_file("no-such-file.h5")
        assert ar._grain_stats_settings == {"edge_rule": "matlab_script", "ecd_basis": "cell_area", "stats_phase": None}
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])
        ar._dataset_by_file.clear()
        ar._dataset_by_file.update(saved[2])


def test_batch_route_passes_the_two_values_only_when_given(client_with_map, monkeypatch):
    client, _ = client_with_map
    import analysis.batch_processor as bp
    from analysis.batch_processor import BatchResult
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder", lambda folder, out, progress_callback=None, **kw: seen.append(kw) or BatchResult(processed=["a.h5"], n_files=1))
    for body in ({"folder_path": "f", "output_path": "o"},
                 {"folder_path": "f", "output_path": "o", "grain_edge_rule": "mtex", "grain_ecd_basis": "measured_pixels"}):
        r = client.post("/api/analysis/batch", json=body)
        assert r.status_code == 200
        task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
        assert task["status"] == "completed", task
    assert seen[0] == {}
    assert seen[1] == {"grain_edge_rule": "mtex", "grain_ecd_basis": "measured_pixels"}


# ---------------------------------------------------------------------------------------------
# batch and Excel
# ---------------------------------------------------------------------------------------------
def test_batch_processor_records_rule_basis_and_counts(tmp_path, monkeypatch):
    from analysis import batch_processor as bp

    # The texture step (about 90 s of orix symmetry set-up whatever the map size) is not under test here.
    monkeypatch.setattr(bp, "analyze_texture", lambda *a, **k: None)
    xmap = to_xmap(make_map(shape=(40, 50), n_seeds=14))
    xmap.prop["bc"] = np.random.default_rng(0).uniform(20, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)

    proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"),
                             grain_edge_rule="outer_row", grain_ecd_basis="measured_pixels")
    assert (proc.grain_edge_rule, proc.grain_ecd_basis) == ("outer_row", "measured_pixels")
    results = proc.process_file(tmp_path / "sample.h5")
    gs = results["grain_size"]
    info = results["basic_info"]
    assert (gs.edge_rule, gs.ecd_basis) == ("outer_row", "measured_pixels")
    assert info["grain_edge_rule"] == "outer_row" and info["grain_ecd_basis"] == "measured_pixels"
    assert info["n_grains_total"] == gs.n_grains_total == info["n_grains"]
    assert info["n_edge_grains"] == gs.n_edge_grains and info["n_grains_used"] == gs.n_grains_used
    assert info["edge_safeguard_applied"] == gs.edge_safeguard_applied
    # defaults
    d = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"))
    assert (d.grain_edge_rule, d.grain_ecd_basis) == ("matlab_script", "cell_area")
    with pytest.raises(ValueError):
        bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), grain_edge_rule="nonsense")
    with pytest.raises(ValueError):
        bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), grain_ecd_basis="nonsense")


def test_excel_sheet1_rows_49_to_54(tmp_path):
    from analysis.excel_exporter import ExcelExporter
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ws = ex.wb[ex.wb.sheetnames[0]]
    labels = [ws.cell(row=r, column=1).value for r in range(48, 55)]
    assert labels == ["grain_alpha", "grain_edge_rule", "grain_ecd_basis", "n_grains_total", "n_edge_grains",
                      "n_grains_used", "edge_safeguard_applied"]
    info = {"grain_alpha": 3.1, "grain_edge_rule": "mtex", "grain_ecd_basis": "measured_pixels",
            "n_grains_total": 120, "n_edge_grains": 30, "n_grains_used": 90, "edge_safeguard_applied": False}
    ex.export_dataset("ds", {"basic_info": info})
    assert [ws.cell(row=r, column=2).value for r in range(48, 55)] == [3.1, "mtex", "measured_pixels", 120, 30, 90, False]
    ex.export_dataset("empty", {"basic_info": {}})
    assert [ws.cell(row=r, column=3).value for r in range(49, 55)] in ([None] * 6, [""] * 6)


# ---------------------------------------------------------------------------------------------
# the mtex rule against MTEX's own per-pixel answer on four real maps
# ---------------------------------------------------------------------------------------------
REAL_FIX = Path(__file__).parent / "fixtures" / "analysis_mtex_parity" / "real_is_boundary.npz"
#: The four scans by the alias their file has in local_names.json of the test-data folder.
REAL_MAPS = ("map_a", "map_b", "map_c", "map_d")


def _test_data_dir():
    """Test_data/ of this checkout; ORIENTA_TEST_DATA points to another copy."""
    import os
    from tests.data_deps import TEST_DATA_DIR
    return Path(os.environ.get("ORIENTA_TEST_DATA", TEST_DATA_DIR))


def _real_dataset(name):
    from tests.data_deps import LOCAL_NAMES, first_missing, local_name
    base = _test_data_dir()
    path = base / local_name(name, base)
    if first_missing(path) is not None:
        pytest.skip(f"needs the scan named '{name}' in {LOCAL_NAMES} of the test-data folder - measurement data "
                    "is not in the repository (set ORIENTA_TEST_DATA to a directory that has it)")
    if path.suffix == ".ang":
        from orix.io import load
        xmap = load(str(path))
        return EBSDDataset(xmap, step_size=float(xmap.dx), source=str(path))
    from tests.raw_h5oina import xmap_from_h5oina
    xmap, step = xmap_from_h5oina(str(path))
    return EBSDDataset(xmap, step_size=step, source=str(path))


@pytest.mark.parametrize("name", REAL_MAPS)
def test_mtex_rule_equals_mtex_is_boundary_per_pixel_on_real_maps(name):
    """Four real scans (not distributed with the repository: two Oxford .h5oina, two .ang; 5 deg, minPixel 3, alpha 3.1): the edge flag of the grain of every measured pixel equals
    MTEX's `isBoundary` of the grain MTEX puts the pixel in (numbers from real_is_boundary.npz, made by
    an MTEX script). Needs the four scans listed in local_names.json of the test-data folder; skipped without them."""
    ds = _real_dataset(name)
    with np.load(REAL_FIX) as f:
        shape = tuple(int(v) for v in f[f"{name}_shape"])
        n = shape[0] * shape[1]
        in_grain = np.unpackbits(f[f"{name}_in_grain"])[:n].astype(bool).reshape(shape)
        theirs = np.unpackbits(f[f"{name}_is_boundary"])[:n].astype(bool).reshape(shape)
    assert tuple(ds.shape) == shape
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)
    measured = (g.grain_ids > 0) & ~g.closed_mask
    assert in_grain[measured].all()                                   # MTEX keeps every measurement of ours
    edge = find_edge_grains(ds, g, "mtex")
    ours = np.zeros(shape, dtype=bool)
    ours[measured] = edge[g.grain_ids[measured] - 1]
    wrong = measured & (ours != theirs)
    assert not wrong.any(), f"{int(wrong.sum())} of {int(measured.sum())} measured pixels differ from isBoundary"


# ---------------------------------------------------------------------------------------------
# measurements dropped after the second min_pixels pass
# ---------------------------------------------------------------------------------------------
def test_segmentation_reports_the_measurements_dropped_after_the_second_pass():
    """On a small noisy map two measurements are still too small after the second pass.
    They are part of `culled_mask`, keep their cell (generators) and belong to no grain."""
    from tests.synthetic_grain_map import add_defects
    m = add_defects(make_map(seed=23, shape=(30, 40), n_seeds=14), seed=123, n_holes=25, n_outliers=60, wide=False)
    ds = EBSDDataset(to_xmap(m), step_size=0.5)
    g = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=1.5, fill=False)
    assert int(g.dropped_mask.sum()) == 2
    assert (g.culled_mask[g.dropped_mask]).all() and g.culled_mask.sum() > g.dropped_mask.sum()
    assert g.cells.generator[g.dropped_mask].all()
    assert (g.grain_ids[g.dropped_mask] == 0).all()
    assert not (g.dropped_mask & g.closed_mask).any()


def test_mtex_a_finally_dropped_measurement_is_not_an_open_cell():
    """The same corridor as above, but the pixel at its end is marked as dropped after the second pass: it
    keeps its cell and is not open, so the grain below it is not an edge grain."""
    lab, t = _tiles(6, 6, 5, {(0, 3)})
    speck = int(lab.max()) + 1
    lab[5:9, 17] = 0
    lab[9, 17] = speck
    ds = _dataset(lab)
    g = _reconstruct(ds, alpha=0.0, min_pixels=2)
    border = {t[(i, j)] for i in range(6) for j in range(6) if (i in (0, 5) or j in (0, 5)) and (i, j) != (0, 3)}
    assert _edge_labels(ds, g, lab, "mtex") == border | {t[(1, 3)], t[(2, 3)]}
    g.dropped_mask = np.zeros(lab.shape, dtype=bool)
    g.dropped_mask[9, 17] = True
    assert _edge_labels(ds, g, lab, "mtex") == border | {t[(1, 3)]}


# ---------------------------------------------------------------------------------------------
# pydantic literals and constants
# ---------------------------------------------------------------------------------------------
def test_pydantic_literals_are_the_rule_and_basis_constants():
    import typing
    from backend.api.routes.analysis import (BatchProcessRequest, EcdBasis, EdgeRule, GrainReconstructionRequest,
                                             GrainSizeRequest)
    assert typing.get_args(EdgeRule) == EDGE_RULES
    assert typing.get_args(EcdBasis) == ECD_BASES

    def literal_args(annotation):
        args = typing.get_args(annotation)
        if typing.get_origin(annotation) is typing.Literal:
            return args
        return next(a for a in (literal_args(x) for x in args if x is not type(None)) if a)

    for model, name, want in ((GrainReconstructionRequest, "edge_rule", EDGE_RULES),
                              (GrainReconstructionRequest, "ecd_basis", ECD_BASES),
                              (GrainSizeRequest, "edge_rule", EDGE_RULES),
                              (GrainSizeRequest, "ecd_basis", ECD_BASES),
                              (BatchProcessRequest, "grain_edge_rule", EDGE_RULES),
                              (BatchProcessRequest, "grain_ecd_basis", ECD_BASES)):
        assert literal_args(model.model_fields[name].annotation) == want, (model, name)


def test_ecd_map_on_measured_pixels_without_grain_size_is_a_400_with_the_reason(client_with_map):
    client, ar = client_with_map
    ny, nx = ar._dataset.shape
    ids = np.ones((ny, nx), dtype=np.int64)
    ar._dataset.grains = GrainSet(grain_ids=ids, n_grains=1, ecd=np.ones(1), area=np.ones(1), aspect_ratio=np.ones(1))
    ar._grain_stats_settings["ecd_basis"] = "measured_pixels"
    r = client.get("/api/analysis/map/ecd")
    assert r.status_code == 400
    assert "grain_size" in r.json()["detail"]
    ar._grain_stats_settings["ecd_basis"] = "cell_area"
    assert client.get("/api/analysis/map/ecd").status_code == 200


def test_reconstruct_task_says_when_its_grain_size_summary_failed(client_with_map, monkeypatch):
    """The summary is best effort. When it fails the result carries the reason and no edge counts, which
    would read like a map without grains."""
    client, ar = client_with_map
    import analysis.grain_analysis as ga

    def boom(*a, **k):
        raise ValueError("no cells here")

    monkeypatch.setattr(ga, "analyze_grain_size", boom)
    res = _reconstruct_via_route(client, edge_rule="mtex")
    assert res["n_grains"] > 0
    assert res["grain_stats_error"] == "no cells here"
    assert (res["edge_rule"], res["ecd_basis"]) == ("mtex", "cell_area")
    for key in ("n_grains_total", "n_edge_grains", "n_grains_used", "edge_safeguard_applied", "mean_ecd"):
        assert key not in res


# ---------------------------------------------------------------------------------------------
# the second pass
# ---------------------------------------------------------------------------------------------
def test_nothing_is_dropped_after_the_second_pass_when_there_is_no_second_pass_or_it_drops_nothing():
    """`dropped_mask` is all False with min_pixels 1 (no culling at all), when the first pass culls nothing
    (no second pass) and when the second pass drops nothing; the same map with the pixels of the seed-23
    search does drop (see the test above)."""
    from tests.synthetic_grain_map import add_defects
    lab, _ = _tiles(4, 4, 3)
    ds = _dataset(lab)
    g1 = _reconstruct(ds, min_pixels=1)                       # min_pixels 1: no pass at all
    assert g1.dropped_mask is not None and g1.dropped_mask.shape == lab.shape and not g1.dropped_mask.any()
    g2 = _reconstruct(ds, min_pixels=9)                       # every tile has 9 pixels: the first pass culls nothing
    assert not g2.culled_mask.any() and not g2.dropped_mask.any() and g2.n_grains == 16
    # a first pass that culls, a second that drops nothing: the speck is culled, the rest is big enough
    lab[0, 0] = lab.max() + 1                                 # one pixel of its own in tile (0,0)
    g3 = _reconstruct(_dataset(lab), min_pixels=3)
    assert g3.culled_mask.sum() == 1 and not g3.dropped_mask.any()
    # and the noisy map of the search drops something with the same call but another alpha/min_pixels
    m = add_defects(make_map(seed=23, shape=(30, 40), n_seeds=14), seed=123, n_holes=25, n_outliers=60, wide=False)
    noisy = EBSDDataset(to_xmap(m), step_size=0.5)
    assert reconstruct_grains(noisy, angle_threshold_deg=5.0, min_pixels=1, alpha=1.5, fill=False).dropped_mask.sum() == 0
    assert reconstruct_grains(noisy, angle_threshold_deg=5.0, min_pixels=3, alpha=1.5, fill=False).dropped_mask.sum() == 2
