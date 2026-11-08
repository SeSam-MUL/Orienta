"""One composition for a set of pixels comes from its SUMMED net counts.

On the spectrum source the per-pixel net intensities are unclipped and noisy,
and per-pixel at% clips a negative net at 0 -- so the MEAN of per-pixel at%
over a region is biased upward for a weak element (measured on a box of the Oxford reference scan:
Ti 0.64 at% as a mean of pixels, while the region's summed Ti net is
negative; an element the spectrum detects but AZtec did not export is in no
composition (Ti here), and Zn shows the same bias: 0.68 at% as a mean of pixels, 0.35 summed). The sum of the per-pixel nets is exactly linear, so the region is
quantified once, on ``sum(net cps x live time)`` over its pixels.
Spread statistics stay per pixel. The window source keeps the mean of
per-pixel at%. Every response says which it did: ``composition_method``.

The four consumers: ``/region-quantify``, the region inspector
(``/phase-map/region/{id}``, ``/phase-map/region-at``), the wand statistics
(``/phase-map/wand-stats``) and the export (regions, particles all pixels and
core, phases). The reference numbers come from the fit package plus
``eds_utils`` on the net maps and the file's live time, no service involved.

Real data: the Oxford reference scan (this checkout's ``Test_data``, or ``ORIENTA_TEST_DATA``);
the clip-bias case runs on a small synthetic cube written here.
"""
from __future__ import annotations

import base64

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import ndimage

from backend.api.services import eds_spectrum_service as svc
from backend.api.services import h5_session
from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE
from backend.eds_spectrum.cube import live_time
from eds_utils import (CountsByElement, counts_to_weight_pct,
                       weight_pct_to_atomic_pct)
from tests.test_eds_spectrum_cube import write_cube
from tests.test_eds_spectrum_fit import META
from tests.test_eds_spectrum_routes import (BOXES, _window_at_pct,
                                            direct_at_pct, direct_counts)
from tests.test_eds_spectrum_service import (  # noqa: F401  (fixtures)
    SYNTH_AREAS, WAIT, WINDOWS, _unit_spectra, blocked_fit, direct_net_maps,
    sampleb,
)

SUMMED, MEAN = "summed_net_counts", "mean_of_pixel_at_pct"
N_ROWS, N_COLS = 90, 120
#: A pixel of the Oxford reference scan outside both boxes: a one-pixel particle (no core).
LONE_PX = (5, 110)


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_session():
    """No open file, no crop, no job, no phase map before and after every test."""
    from backend.api.services import crop_window as crop_window_service
    from backend.api.services.phase_map_store import get_phase_map_store

    def _reset():
        h5_session.close_file()
        svc.reset()
        crop_window_service.clear_all()
        get_phase_map_store().clear()

    _reset()
    yield
    _reset()


@pytest.fixture
def ready(sampleb):
    """The Oxford reference scan open, its spectrum fit finished."""
    h5_session.open_file(str(sampleb))
    assert svc.get_result(str(sampleb), wait_s=WAIT) is not None
    return sampleb


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _box_idx(r0, r1, c0, c1, n_cols=N_COLS):
    rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    return (rows * n_cols + cols).ravel()


def _box_body(r0, r1, c0, c1):
    return {"row_start": r0, "row_end": r1, "col_start": c0, "col_end": c1}


def direct_summed(path, idx):
    """wt% and at% of the pixels ``idx`` from their SUMMED net counts.

    Net cps from the fit package (``direct_counts``: the line per element the
    quantification uses), times the live time read from the file, summed, then
    ``eds_utils`` on the spectrum curve. No service, no route.
    """
    counts, kv = direct_counts(path, idx)
    lt = live_time(str(path))[np.atleast_1d(idx)]
    summed = CountsByElement(
        {el: np.array([np.sum(np.asarray(v, dtype=np.float64) * lt)])
         for el, v in counts.items()}, source="spectrum",
        geometry=counts.geometry)
    wt = counts_to_weight_pct(summed, beam_kv=kv)
    return wt, weight_pct_to_atomic_pct(wt)


def _pack(mask):
    return base64.b64encode(
        np.packbits(np.asarray(mask, dtype=np.uint8).ravel()).tobytes()
    ).decode("ascii")


def _seed_regions(region_grid, n_regions):
    """A classification whose regions are ``region_grid``; never saved."""
    from backend.api.services.cif_phase_library import CifPhaseEntry
    from backend.api.services.phase_map_store import get_phase_map_store

    entry = CifPhaseEntry(
        key="al", cif_filename="Al.cif", formula="Al", space_group="Fm-3m",
        space_group_number=225, crystal_system="cubic",
        composition={"Al": 100.0}, elements=["Al"])
    shape = region_grid.shape
    store = get_phase_map_store()
    store.set_classification(
        phase_grid=np.zeros(shape, dtype=np.int32),
        score_grid=np.zeros(shape, dtype=np.float32),
        phase_entries=[entry], tolerance=15.0, min_score=0.3, file_path=None,
        region_grid=region_grid.astype(np.int32),
        region_phase=[0] * n_regions)
    return store.get_state()


def _two_box_grid():
    """Region 1 = both boxes of the Oxford reference scan plus one lone pixel; region 0 = the rest."""
    grid = np.zeros((N_ROWS, N_COLS), dtype=np.int32)
    for r0, r1, c0, c1 in BOXES:
        grid[r0:r1 + 1, c0:c1 + 1] = 1
    grid[LONE_PX] = 1
    return grid


def _build_tables(at_maps, composition_of, n_rows=N_ROWS, n_cols=N_COLS):
    from backend.api.services.eds_export import (ExportGeometry,
                                                 ExportOptions, build_tables)
    from backend.api.services.phase_map_store import get_phase_map_store
    return build_tables(
        get_phase_map_store().get_state(), at_maps,
        geometry=ExportGeometry(n_rows=n_rows, n_cols=n_cols,
                                step_x_um=0.5, step_y_um=0.5),
        options=ExportOptions(hash_source=False, read_acquisition_header=False,
                              library={}),
        composition_of=composition_of)


def _export_route(client, tmp_path):
    r = client.post("/api/eds/export", json={
        "dest_dir": str(tmp_path), "hash_source": False,
        "artefacts": {"phases": True, "regions": True, "particles": True,
                      "definitions": False, "xlsx": False, "map_png": False,
                      "labels": False, "pixels": False}})
    assert r.status_code == 200, r.text
    return r.json()


def _particle_row(table, region_id, r0, r1, c0, c1):
    rows = [p for p in table.rows
            if p["region_id"] == region_id
            and (p["bbox_row_min"], p["bbox_row_max"],
                 p["bbox_col_min"], p["bbox_col_max"]) == (r0, r1, c0, c1)]
    assert len(rows) == 1, (region_id, r0, r1, c0, c1, len(rows))
    return rows[0]


# ---------------------------------------------------------------------------
# /region-quantify
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r0,r1,c0,c1", BOXES)
def test_region_quantify_uses_summed_net_counts(ready, client, r0, r1, c0, c1):
    r = client.post("/api/eds/region-quantify", json=_box_body(r0, r1, c0, c1))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["composition_method"] == SUMMED
    assert body["quantification"]["source"] == "spectrum"

    idx = _box_idx(r0, r1, c0, c1)
    wt, at = direct_summed(ready, idx)
    per_px = direct_at_pct(ready, idx)
    data = body["data"]
    quantified = {el for el, v in data.items() if v["at_pct"] is not None}
    assert sorted(quantified) == sorted(str(el) for el in at)
    # C (exported, left out of the matrix-corrected composition) keeps counts only
    assert data["C"]["at_pct"] is None and data["C"]["counts"] is not None
    for el in at:
        got = data[str(el)]
        assert got["at_pct"]["mean"] == pytest.approx(
            float(at[el][0]), rel=1e-9, abs=1e-12), el
        assert got["wt_pct"]["mean"] == pytest.approx(
            float(wt[el][0]), rel=1e-9, abs=1e-12), el
        # the spread is still the pixels'
        px = np.asarray(per_px[el], dtype=np.float64)
        assert got["at_pct"]["std"] == pytest.approx(float(px.std()),
                                                     rel=1e-9, abs=1e-12), el
        assert got["at_pct"]["min"] == pytest.approx(float(px.min()),
                                                     rel=1e-9, abs=1e-12), el
        assert got["at_pct"]["max"] == pytest.approx(float(px.max()),
                                                     rel=1e-9, abs=1e-12), el
    # Not vacuous: on these boxes the mean of the pixels' at% is a different
    # number (Zn 0.68 against 0.35 summed on the first box; Ti would read
    # 0.64 against a negative summed net, but an element the spectrum detects
    # but AZtec did not export is in no composition).
    gap = max(abs(data[str(el)]["at_pct"]["mean"] - float(np.mean(per_px[el])))
              for el in at)
    assert gap > 0.3, gap


# ---------------------------------------------------------------------------
# the bias the summed composition removes
# ---------------------------------------------------------------------------
NXS, NYS = 16, 12


def _write_absent_half(path, dose=1e-3, seed=7):
    """An EDS-only cube whose right half carries NO Mn (Mn K area 0 there).

    The left half has Mn, so the sum fit keeps the Mn K family and every pixel
    gets an unclipped Mn net: noise around 0 on the right. Low dose (~7 400
    counts per pixel), so that noise is large against the per-pixel signal.
    """
    base, unit = _unit_spectra()
    rng = np.random.default_rng(seed)
    n = NXS * NYS
    spectra = np.zeros((n, META.n_channels))
    for i in range(n):
        right = (i % NXS) >= NXS // 2
        mean = 4 * dose * base
        for k, v in SYNTH_AREAS.items():
            if k == "Mn K" and right:
                continue
            mean += dose * v * rng.uniform(0.5, 1.5) * unit[k]
        spectra[i] = rng.poisson(np.clip(mean, 0, None))
    live = 3e-4 * rng.uniform(0.5, 1.5, n)
    write_cube(path, spectra, live, nx=NXS, ny=NYS)
    energies = META.energies()
    with h5py.File(path, "a") as f:
        w = f.create_group("1/EDS/Data/Window Integral")
        for lab, (lo, hi) in WINDOWS.items():
            band = (energies >= lo) & (energies <= hi)
            w.create_dataset(lab, data=(spectra[:, band].sum(axis=1) / live)
                             .astype(np.float32))
    return path


def test_summed_composition_has_no_clip_bias(tmp_path, client):
    path = _write_absent_half(tmp_path / "absent_half.h5oina")
    h5_session.open_file(str(path))
    assert svc.get_result(str(path), wait_s=WAIT) is not None

    r = client.post("/api/eds/region-quantify",
                    json=_box_body(0, NYS - 1, NXS // 2, NXS - 1))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["composition_method"] == SUMMED
    summed_mn = body["data"]["Mn"]["at_pct"]["mean"]

    idx = _box_idx(0, NYS - 1, NXS // 2, NXS - 1, n_cols=NXS)
    from backend.api.routes.eds import _build_at_pct_maps_for_loaded_file
    at_maps = _build_at_pct_maps_for_loaded_file()[0]
    mean_of_pixels = float(np.mean(np.asarray(at_maps["Mn"])[idx]))

    # 1 sigma of the region's summed Mn net, from the pixels themselves (the
    # truth is 0 on every one of them, so E[c^2] is the variance), put on the
    # at% scale with the other elements' summed counts as they are.
    _ff, maps, _y = direct_net_maps(path)
    lt = live_time(str(path))
    c = maps["Mn K"].astype(np.float64)[idx] * lt[idx]
    sigma_counts = float(np.sqrt(np.sum(c ** 2)))
    counts, kv = direct_counts(path, idx)
    one_sigma = CountsByElement(
        {el: np.array([sigma_counts if str(el) == "Mn"
                       else float(np.sum(np.asarray(v, dtype=np.float64)
                                         * lt[idx]))])
         for el, v in counts.items()}, source="spectrum")
    sigma_at = float(weight_pct_to_atomic_pct(
        counts_to_weight_pct(one_sigma, beam_kv=kv))["Mn"][0])
    assert sigma_at > 0

    assert 0.0 <= summed_mn <= 2 * sigma_at, (summed_mn, sigma_at)
    # the per-pixel clip at 0 pushes the mean of the pixels well above that
    assert mean_of_pixels > 2 * sigma_at, (mean_of_pixels, sigma_at)


# ---------------------------------------------------------------------------
# inspector, wand and /region-quantify: one helper
# ---------------------------------------------------------------------------
def test_region_inspector_and_wand_use_the_same_helper(ready, client):
    from backend.api.routes.eds import (_build_at_pct_maps_for_loaded_file,
                                        _composition_of)

    r0, r1, c0, c1 = BOXES[0]
    grid = np.zeros((N_ROWS, N_COLS), dtype=np.int32)
    grid[r0:r1 + 1, c0:c1 + 1] = 1
    _seed_regions(grid, 2)
    mask = grid == 1

    rq = client.post("/api/eds/region-quantify", json=_box_body(r0, r1, c0, c1))
    insp = client.get("/api/eds/phase-map/region/1")
    at_px = client.post("/api/eds/phase-map/region-at",
                        json={"row": r0, "col": c0})
    wand = client.post("/api/eds/phase-map/wand-stats",
                       json={"mask_b64": _pack(mask), "phase_index": 0})
    for resp in (rq, insp, at_px, wand):
        assert resp.status_code == 200, resp.text
    rq, insp, at_px, wand = rq.json(), insp.json(), at_px.json(), wand.json()
    for name, resp in (("region-quantify", rq), ("inspector", insp),
                       ("region-at", at_px), ("wand", wand)):
        assert resp["composition_method"] == SUMMED, name

    ref = {el: v["at_pct"]["mean"] for el, v in rq["data"].items()
           if v["at_pct"] is not None}
    at_maps = _build_at_pct_maps_for_loaded_file()[0]
    helper = _composition_of(at_maps)
    assert helper is not None
    assert {str(el): v for el, v in helper(mask).items()} == pytest.approx(
        ref, rel=1e-12, abs=1e-12)

    # the wand reports it to 2 dp, without traces below 0.05
    assert wand["mean_at_pct"] == {el: round(v, 2) for el, v in ref.items()
                                   if v >= 0.05}
    # the inspector: the same numbers, C and O left out (they never decided)
    expect = {el: v for el, v in wand["mean_at_pct"].items()
              if el not in _CHEM_IGNORE}
    for detail in (insp, at_px):
        assert {e["element"]: e["at_pct"] for e in detail["elements"]} == expect
        assert {e["element"]: e["enrichment"] for e in detail["elements"]} == {
            el: wand["enrichment"].get(el) for el in expect}
        # the spread is still the pixels'
        for e in detail["elements"]:
            px = np.asarray(at_maps[e["element"]], dtype=np.float64).ravel()
            assert e["spread"] == round(float(px[mask.ravel()].std()), 2)

    # the neighbour gap compares like with like: both regions summed
    other = {str(el): v for el, v in helper(grid == 0).items()}
    gap = max(abs(v - other[el]) for el, v in expect.items())
    assert [(n["region_id"], n["gap_at_pct"]) for n in insp["neighbours"]] == [
        (0, round(gap, 2))]

    # not the mean of the pixels: Zn reads 0.68 that way and 0.35 summed.
    # (Ti would read 0.64 against a negative summed net, but an element the
    # spectrum detects but AZtec did not export is in no composition.)
    zn_px = float(np.asarray(at_maps["Zn"]).ravel()[mask.ravel()].mean())
    assert zn_px - wand["mean_at_pct"]["Zn"] > 0.25, (zn_px, wand["mean_at_pct"])


# ---------------------------------------------------------------------------
# export: particles (all pixels and core), regions, phases
# ---------------------------------------------------------------------------
def test_export_particle_composition_summed(ready, client, tmp_path):
    from backend.api.routes.eds import (_build_at_pct_maps_for_loaded_file,
                                        _composition_of)

    grid = _two_box_grid()
    _seed_regions(grid, 2)
    at_maps = _build_at_pct_maps_for_loaded_file()[0]
    helper = _composition_of(at_maps)
    tables = _build_tables(at_maps, helper)
    assert tables.provenance["composition"]["composition_method"] == SUMMED
    measured = [str(el) for el in at_maps]
    cross = ndimage.generate_binary_structure(2, 1)

    def _check(row, mask2d, prefix=""):
        idx = np.flatnonzero(mask2d.ravel())
        _wt, at = direct_summed(ready, idx)
        for el in at:
            assert row[f"{prefix}mean_at_pct_{el}"] == pytest.approx(
                float(at[el][0]), rel=1e-9, abs=1e-12), (prefix, el)
        h = {str(k): v for k, v in helper(mask2d).items()}
        for el in measured:
            assert row[f"{prefix}mean_at_pct_{el}"] == pytest.approx(
                h[el], rel=1e-12, abs=1e-12), (prefix, el)

    for r0, r1, c0, c1 in BOXES:
        row = _particle_row(tables.particles, 1, r0, r1, c0, c1)
        box = np.zeros((N_ROWS, N_COLS), dtype=bool)
        box[r0:r1 + 1, c0:c1 + 1] = True
        _check(row, box)
        core = ndimage.binary_erosion(box, structure=cross)
        assert core.sum() == (r1 - r0 - 1) * (c1 - c0 - 1)
        _check(row, core, prefix="core_")
        # the spread is still the pixels'
        for el in measured:
            px = np.asarray(at_maps[el], dtype=np.float64).ravel()[box.ravel()]
            assert row[f"sd_within_at_pct_{el}"] == pytest.approx(
                float(px.std()), rel=1e-9, abs=1e-12), el

    lone = np.zeros((N_ROWS, N_COLS), dtype=bool)
    lone[LONE_PX] = True
    row = _particle_row(tables.particles, 1, LONE_PX[0], LONE_PX[0],
                        LONE_PX[1], LONE_PX[1])
    _check(row, lone)
    assert all(row[f"core_mean_at_pct_{el}"] is None for el in measured)

    # regions and phases report the same kind of number
    reg = {r["region_id"]: r for r in tables.regions.rows}
    _check(reg[1], grid == 1)
    _check(reg[0], grid == 0)
    phase0 = [p for p in tables.phases.rows if p["phase_index"] == 0][0]
    _check(phase0, np.ones((N_ROWS, N_COLS), dtype=bool))

    # the inspector's second-opinion ranking (MAD) stands on that composition
    from backend.api.services.eds_export import rank_candidates, to_scored_basis
    from backend.api.services.phase_map_store import get_phase_map_store
    entries = get_phase_map_store().get_state().phase_entries

    def _mad(comp):
        means = {el: round(v, 2) for el, v in comp.items()
                 if v >= 0.05 and el not in _CHEM_IGNORE}
        return rank_candidates(to_scored_basis(means), entries)[0][1]

    r0, r1, c0, c1 = BOXES[0]
    box = np.zeros((N_ROWS, N_COLS), dtype=bool)
    box[r0:r1 + 1, c0:c1 + 1] = True
    for row, mask2d in ((reg[1], grid == 1),
                        (_particle_row(tables.particles, 1, r0, r1, c0, c1), box)):
        m = mask2d.ravel()
        summed = {str(k): v for k, v in helper(m).items()}
        pixels = {el: float(np.asarray(at_maps[el]).ravel()[m].mean())
                  for el in measured}
        assert row["gap_mad_at_pct"] == pytest.approx(_mad(summed),
                                                      rel=1e-12, abs=1e-12)
        assert abs(_mad(summed) - _mad(pixels)) > 1e-3   # not vacuous

    # the route hands the helper to the export
    out = _export_route(client, tmp_path)
    assert out["provenance"]["composition"]["composition_method"] == SUMMED


# ---------------------------------------------------------------------------
# the window source keeps the mean of per-pixel at%
# ---------------------------------------------------------------------------
def test_window_source_unchanged(ready, client, tmp_path):
    from backend.api.routes.eds import (_build_at_pct_maps_for_loaded_file,
                                        _composition_of)
    from backend.api.services.chemistry_score import background_levels
    from backend.api.services.eds_particles import composition_for_mask
    from backend.api.services.eds_wand import selection_stats

    grid = _two_box_grid()
    _seed_regions(grid, 2)
    r0, r1, c0, c1 = BOXES[0]
    box = np.zeros((N_ROWS, N_COLS), dtype=bool)
    box[r0:r1 + 1, c0:c1 + 1] = True
    with svc.force_window_source():
        rq = client.post("/api/eds/region-quantify",
                         json=_box_body(r0, r1, c0, c1))
        assert rq.status_code == 200, rq.text
        rq = rq.json()
        assert rq["composition_method"] == MEAN
        window = _window_at_pct(_box_idx(r0, r1, c0, c1))
        for el in window:
            assert rq["data"][str(el)]["at_pct"]["mean"] == pytest.approx(
                float(np.mean(window[el])), rel=1e-9, abs=1e-12), el

        at_maps = _build_at_pct_maps_for_loaded_file()[0]
        assert _composition_of(at_maps) is None
        region1 = grid == 1
        ref = selection_stats(at_maps, region1,
                              background=background_levels(at_maps))

        wand = client.post("/api/eds/phase-map/wand-stats",
                           json={"mask_b64": _pack(region1), "phase_index": 0})
        assert wand.status_code == 200, wand.text
        wand = wand.json()
        assert wand["composition_method"] == MEAN
        assert wand["mean_at_pct"] == ref["mean_at_pct"]
        assert wand["enrichment"] == ref["enrichment"]

        insp = client.get("/api/eds/phase-map/region/1")
        assert insp.status_code == 200, insp.text
        insp = insp.json()
        assert insp["composition_method"] == MEAN
        assert {e["element"]: e["at_pct"] for e in insp["elements"]} == {
            el: v for el, v in ref["mean_at_pct"].items()
            if el not in _CHEM_IGNORE}

        tables = _build_tables(at_maps, None)
        assert tables.provenance["composition"]["composition_method"] == MEAN
        row = _particle_row(tables.particles, 1, r0, r1, c0, c1)
        old = composition_for_mask(dict(at_maps), box)
        core = composition_for_mask(dict(at_maps), ndimage.binary_erosion(
            box, structure=ndimage.generate_binary_structure(2, 1)))
        for el in at_maps:
            assert row[f"mean_at_pct_{el}"] == old[el][0], el
            assert row[f"core_mean_at_pct_{el}"] == core[el][0], el

        out = _export_route(client, tmp_path)
        assert out["provenance"]["composition"]["composition_method"] == MEAN


# ---------------------------------------------------------------------------
# crops: the live time is cut like the maps
# ---------------------------------------------------------------------------
def test_crop_region_composition(ready, client, monkeypatch):
    from backend.api.routes.eds import (_build_at_pct_maps_for_loaded_file,
                                        _composition_of)
    from backend.api.services.crop_window import CropWindow
    from backend.api.services.cropped_extractor import CroppedExtractor

    # crop rows 10..39, cols 20..59; the box rows 15..24, cols 30..44 of the
    # full grid is rows 5..14, cols 10..24 of the crop
    full_box, crop_box = (15, 24, 30, 44), (5, 14, 10, 24)
    full = client.post("/api/eds/region-quantify", json=_box_body(*full_box))
    assert full.status_code == 200, full.text
    full = full.json()
    _wt, direct = direct_summed(ready, _box_idx(*full_box))

    window = CropWindow(str(ready), 10, 20, 30, 40, (N_ROWS, N_COLS))
    monkeypatch.setattr("backend.api.routes.ebsd_viewer.get_active_crop_window",
                        lambda: window)
    ext = h5_session.get_active_extractor()
    assert isinstance(ext, CroppedExtractor)
    lt = live_time(str(ready)).reshape(N_ROWS, N_COLS)
    np.testing.assert_array_equal(ext.get_live_time(),
                                  lt[10:40, 20:60].ravel())

    crop = client.post("/api/eds/region-quantify", json=_box_body(*crop_box))
    assert crop.status_code == 200, crop.text
    crop = crop.json()
    assert crop["composition_method"] == SUMMED
    assert crop["n_pixels"] == full["n_pixels"] == 150
    assert sorted(crop["data"]) == sorted(full["data"])
    for el, v in full["data"].items():
        if v["at_pct"] is None:
            assert crop["data"][el]["at_pct"] is None, el
            continue
        for key in ("mean", "std", "min", "max"):
            assert crop["data"][el]["at_pct"][key] == pytest.approx(
                v["at_pct"][key], rel=1e-12, abs=1e-12), (el, key)
    for el in direct:
        assert crop["data"][str(el)]["at_pct"]["mean"] == pytest.approx(
            float(direct[el][0]), rel=1e-9, abs=1e-12), el

    # the inspector and the wand under the crop, on the crop's own grid
    grid = np.zeros((30, 40), dtype=np.int32)
    grid[5:15, 10:25] = 1
    _seed_regions(grid, 2)
    at_maps = _build_at_pct_maps_for_loaded_file()[0]
    helper = _composition_of(at_maps)
    got = {str(el): v for el, v in helper(grid == 1).items()}
    assert got == pytest.approx(
        {el: v["at_pct"]["mean"] for el, v in full["data"].items()
         if v["at_pct"] is not None},
        rel=1e-12, abs=1e-12)
    wand = client.post("/api/eds/phase-map/wand-stats",
                       json={"mask_b64": _pack(grid == 1), "phase_index": 0})
    assert wand.status_code == 200, wand.text
    assert wand.json()["mean_at_pct"] == {
        el: round(v["at_pct"]["mean"], 2) for el, v in full["data"].items()
        if v["at_pct"] is not None and v["at_pct"]["mean"] >= 0.05}


# ---------------------------------------------------------------------------
# the phase-map legend: one region, one composition
# ---------------------------------------------------------------------------
def _legend_expect(comp):
    """The legend's rule on a composition: no C/O, >= 0.5 at%, 2 dp."""
    return {str(el): round(v, 2) for el, v in comp.items()
            if str(el) not in _CHEM_IGNORE and v >= 0.5}


def test_legend_composition_is_the_inspectors(ready, client):
    from backend.api.routes.eds import (_build_at_pct_maps_for_loaded_file,
                                        _composition_of)

    grid = _two_box_grid()
    _seed_regions(grid, 2)
    at_maps = _build_at_pct_maps_for_loaded_file()[0]
    helper = _composition_of(at_maps)

    body = client.get("/api/eds/phase-map?include_image=false")
    assert body.status_code == 200, body.text
    regions = {r["region_id"]: r for r in body.json()["regions"]}
    assert sorted(regions) == [0, 1]
    for sid in (0, 1):
        legend = regions[sid]
        assert legend["composition_method"] == SUMMED, sid
        expect = _legend_expect(helper(grid == sid))
        assert legend["mean_at_pct"] == pytest.approx(expect, rel=1e-9), sid
        insp = client.get(f"/api/eds/phase-map/region/{sid}").json()
        assert insp["composition_method"] == SUMMED
        shown = {e["element"]: e["at_pct"] for e in insp["elements"]}
        assert {el: shown[el] for el in legend["mean_at_pct"]} == pytest.approx(
            legend["mean_at_pct"], rel=1e-9), sid

    # Not vacuous: the mean of the pixels and the summed composition differ
    # visibly (more than the legend's 0.01 at% rounding) on some element of
    # region 1, and the legend shows the summed value. (No element or
    # cut is pinned: which element differs depends on the quantification.)
    m = (grid == 1).ravel()
    summed = {str(el): v for el, v in helper(grid == 1).items()}
    differs = []
    for el, v in summed.items():
        if el in _CHEM_IGNORE or el not in at_maps or el not in regions[1]["mean_at_pct"]:
            continue
        px = float(np.asarray(at_maps[el], dtype=np.float64).ravel()[m].mean())
        if abs(px - v) > 0.05:
            differs.append(el)
            assert regions[1]["mean_at_pct"][el] == pytest.approx(round(v, 2)), (el, px, v)
    assert differs, summed


def test_legend_window_source_unchanged(ready, client):
    from backend.api.routes.eds import _build_at_pct_maps_for_loaded_file

    grid = _two_box_grid()
    _seed_regions(grid, 2)
    with svc.force_window_source():
        at_maps = _build_at_pct_maps_for_loaded_file()[0]
        body = client.get("/api/eds/phase-map?include_image=false")
    assert body.status_code == 200, body.text
    for legend in body.json()["regions"]:
        m = (grid == legend["region_id"]).ravel()
        pixels = {el: float(np.asarray(v, dtype=float).ravel()[m].mean())
                  for el, v in at_maps.items()}
        assert legend["composition_method"] == MEAN
        assert legend["mean_at_pct"] == _legend_expect(pixels)


def test_legend_degrades_while_the_fit_runs(blocked_fit, sampleb):
    """Pending: compositions left out, the marker says why, no method claimed."""
    import time

    from backend.api.routes import eds

    h5_session.open_file(str(sampleb))
    t0 = time.monotonic()
    while svc.status(str(sampleb))["stage"] != "fitting":
        assert time.monotonic() - t0 < WAIT, svc.status(str(sampleb))
        time.sleep(0.01)
    state = _seed_regions(_two_box_grid(), 2)
    notes = {}
    regions = eds._regions_payload(state, notes)
    assert [r["region_id"] for r in regions] == [0, 1]
    assert all(r["mean_at_pct"] == {} for r in regions)
    assert all(r["composition_method"] is None for r in regions)
    assert notes["eds_status"]["code"] == "eds_spectrum_fit_running"

    blocked_fit.set()
    assert svc.get_result(str(sampleb), wait_s=WAIT) is not None
    notes = {}
    regions = eds._regions_payload(state, notes)
    assert "eds_status" not in notes
    assert all(r["composition_method"] == SUMMED for r in regions)
    assert regions[1]["mean_at_pct"]


# ---------------------------------------------------------------------------
# the region-definition preview: smoothed selection, the region composition
# ---------------------------------------------------------------------------
SI_WINDOW = [{"name": "Si", "elements": [{"element": "Si", "min_at_pct": 20}]}]


def _claimed(defs):
    """The pixels the preview's definitions claim (its own smoothed rule)."""
    from backend.api.routes.eds import (_build_at_pct_maps_for_loaded_file,
                                        _resolve_scale)
    from backend.api.services.chemistry_score import background_levels
    from backend.api.services.eds_clustering import _smooth_maps
    from backend.api.services.phase_rules import (region_defs_from_list,
                                                  region_labels)
    at_maps, n_rows, n_cols, _fp = _build_at_pct_maps_for_loaded_file()
    scale, _report = _resolve_scale(None, None)
    smoothed = _smooth_maps(at_maps, n_rows, n_cols, scale)
    assign = region_labels(region_defs_from_list(defs), smoothed,
                           background=background_levels(at_maps))
    return at_maps, np.asarray(assign.labels).ravel()


def _preview_rule(comp):
    """The preview's display rule: 2 dp, strongest first, >= 0.5 at%."""
    rounded = {str(el): round(float(v), 2) for el, v in comp.items()}
    return {el: v for el, v in sorted(rounded.items(), key=lambda kv: -kv[1])
            if v >= 0.5}


def test_region_def_preview_reports_the_region_composition(ready, client):
    from backend.api.routes.eds import _composition_of

    r = client.post("/api/eds/phase-map/region-defs/preview",
                    json={"region_defs": SI_WINDOW})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["composition_method"] == SUMMED
    got = body["definitions"][0]

    at_maps, labels = _claimed(SI_WINDOW)
    mask = labels == 0
    assert got["n_pixels"] == int(mask.sum()) > 0
    idx = np.flatnonzero(mask)
    _wt, direct = direct_summed(ready, idx)
    expect = _preview_rule({el: float(v[0]) for el, v in direct.items()})
    assert got["mean_at_pct"] == pytest.approx(expect, rel=1e-9)
    assert list(got["mean_at_pct"]) == list(expect)
    helper = _composition_of(at_maps)
    assert got["mean_at_pct"] == pytest.approx(_preview_rule(helper(mask)),
                                               rel=1e-9)

    # not vacuous: the mean of the claimed pixels' at% is another composition
    pixels = _preview_rule({el: float(np.asarray(v, dtype=float).ravel()[mask].mean())
                            for el, v in at_maps.items()})
    assert pixels != got["mean_at_pct"]
    assert abs(pixels["Si"] - got["mean_at_pct"]["Si"]) > 0.5


def test_region_def_preview_window_source_unchanged(ready, client):
    with svc.force_window_source():
        r = client.post("/api/eds/phase-map/region-defs/preview",
                        json={"region_defs": SI_WINDOW})
        assert r.status_code == 200, r.text
        at_maps, labels = _claimed(SI_WINDOW)
    body = r.json()
    assert body["composition_method"] == MEAN
    mask = labels == 0
    got = body["definitions"][0]
    assert got["n_pixels"] == int(mask.sum()) > 0
    # the old rule exactly: each element's pixel mean rounded, then filtered
    old = {el: round(float(np.asarray(a, dtype=float)[mask].mean()), 2)
           for el, a in at_maps.items()}
    old = {str(el): v for el, v in sorted(old.items(), key=lambda kv: -kv[1])
           if v >= 0.5}
    assert got["mean_at_pct"] == old
    assert list(got["mean_at_pct"]) == list(old)
