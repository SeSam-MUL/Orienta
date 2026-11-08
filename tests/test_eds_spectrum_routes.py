"""Every EDS consumer knows which intensities it was handed.

A file with a spectrum cube serves NET line intensities through
``get_active_extractor()``. Here every consumer of those numbers is
checked for the three things that can go wrong once the maps are net:

* the numbers are priced with the net-intensity curve and SAY so
  (``quantification.source_data`` / ``fit``), on every route;
* the at% equal an independent computation -- the fit package plus
  ``eds_utils`` on the net maps, no service involved;
* consumers that mean RAW counts (preflight's counts-per-pixel threshold, the
  batch export's ``/EDS`` group) keep reading AZtec's windows and do not wait
  for the fit; the EDS chemistry prior, the particle rescue and the prior's
  pre-flight read the window at% (kept on the window path after a
  before/after measurement); and a prior on the file's own source decides
  ONCE whether it can run.

Real data: the Oxford reference scan (this checkout's ``Test_data``, or ``ORIENTA_TEST_DATA``).
Pending-fit cases hold the fit with the ``blocked_fit`` fixture of
``test_eds_spectrum_service``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import quote

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.services import eds_spectrum_service as svc
from backend.api.services import h5_session
from backend.eds_spectrum import fit as spectrum_fit
from backend.eds_spectrum.cube import read_meta
from backend.eds_spectrum.matrix import geometry_from_meta
from eds_utils import (build_counts_by_element, counts_to_weight_pct,
                       parse_element_name, weight_pct_to_atomic_pct)
from tests.test_eds_spectrum_service import (  # noqa: F401  (fixtures)
    WAIT, blocked_fit, direct_net_maps, sampleb, synth_files,
)

#: Pixels and rectangles of the Oxford reference scan (row, col) used below: the Si particle (the
#: window Si maximum sits at 84, 5), the Fe-richest pixel (20, 92) and one in
#: the middle of the map; a box in the middle and one around the Fe maximum.
PIXELS = [(84, 5), (20, 92), (45, 60)]
BOXES = [(40, 49, 50, 69), (16, 24, 88, 96)]     # r0, r1, c0, c1 (inclusive)


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_session():
    """No open file, no crop, no job before and after every test."""
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


@pytest.fixture
def pending(blocked_fit, sampleb):
    """The Oxford reference scan open, its spectrum fit held in the 'fitting' stage."""
    import time
    h5_session.open_file(str(sampleb))
    t0 = time.monotonic()
    while svc.status(str(sampleb))["stage"] != "fitting":
        assert time.monotonic() - t0 < WAIT, svc.status(str(sampleb))
        time.sleep(0.01)
    yield sampleb
    blocked_fit.set()
    svc.get_result(str(sampleb), wait_s=WAIT)


# ---------------------------------------------------------------------------
# the independent reference: fit package + eds_utils, no service
# ---------------------------------------------------------------------------
_DIRECT = {}


def direct_counts(path, index):
    """``CountsByElement`` of the net maps at ``index`` (int or index array),
    built from the fit package directly. Composition: AZtec's exported elements only -- an element the spectrum
    shows but AZtec did not export (Ti on the Oxford reference scan) is not in it."""
    key = str(path)
    if key not in _DIRECT:
        _ff, maps, _y = direct_net_maps(path)
        with h5py.File(path, "r") as f:
            exported = {str(parse_element_name(lab))
                        for lab in f["1/EDS/Data/Window Integral"].keys()}
        meta = read_meta(path)
        _DIRECT[key] = ({lab: m for lab, m in maps.items()
                         if str(parse_element_name(lab)) in exported},
                        float(meta.beam_kv), geometry_from_meta(meta))
    maps, kv, geometry = _DIRECT[key]
    idx = np.atleast_1d(index)
    pairs = [(lab, np.asarray(m[idx], dtype=np.float64)) for lab, m in maps.items()]
    # the file's geometry, as the service reads it from the header: the
    # matrix-corrected quantification (backend/eds_spectrum/matrix.py)
    return build_counts_by_element(pairs, beam_kv=kv, source="spectrum",
                                   geometry=geometry), kv


def direct_at_pct(path, index):
    counts, kv = direct_counts(path, index)
    return weight_pct_to_atomic_pct(counts_to_weight_pct(counts, beam_kv=kv))


def _window_at_pct(index):
    """The window path straight from the raw extractor's maps."""
    raw = h5_session.get_extractor()
    pairs = [(lab, np.asarray(raw.get_element_map(lab)[np.atleast_1d(index)],
                              dtype=np.float64))
             for lab in raw.get_available_elements()]
    kv = float(raw.get_eds_header()["beam_voltage_kV"])
    counts = build_counts_by_element(pairs, beam_kv=kv)
    return weight_pct_to_atomic_pct(counts_to_weight_pct(counts, beam_kv=kv))


def _is_spectrum(quant):
    return (quant is not None
            and str(quant.get("source_data", "")).startswith("spectrum fit")
            and quant.get("source") == "spectrum"
            and (quant.get("fit") or {}).get("model_version") == "eds-spectrum-fit/1")


# ---------------------------------------------------------------------------
# every route says where its numbers come from
# ---------------------------------------------------------------------------
FE, CU_L = quote("Fe Kα1"), quote("Cu Lα1,2")
ROUTES = [
    ("get", f"/api/eds/map/{FE}?mode=counts", None),
    ("get", f"/api/eds/map/{FE}?mode=wt_pct", None),
    ("get", f"/api/eds/map/{FE}?mode=at_pct", None),
    ("post", "/api/eds/quantify/pixel", {"row": 84, "col": 5}),
    ("post", "/api/eds/region-quantify",
     {"row_start": 40, "row_end": 49, "col_start": 50, "col_end": 69}),
    ("post", "/api/eds/suggest-phases", {"row": 84, "col": 5}),
    ("post", "/api/eds/chemistry-mask",
     {"filters": [{"element": "Si", "operator": ">", "min_val": 20.0}]}),
    ("post", "/api/eds/chemistry-mask",
     {"filters": [{"element": "Si", "operator": ">", "min_val": 1e4,
                   "unit": "counts"}]}),
    ("post", "/api/eds/probe", {"row": 84, "col": 5}),
    ("post", "/api/eds/linescan",
     {"start_row": 0, "start_col": 0, "end_row": 89, "end_col": 119,
      "layers": ["eds-Fe Kα1"], "display_mode": "at_pct"}),
    ("post", "/api/eds/linescan",
     {"start_row": 0, "start_col": 0, "end_row": 89, "end_col": 119,
      "layers": ["eds-Fe Kα1"], "display_mode": "counts"}),
]


ROUTE_IDS = ["map-counts", "map-wt", "map-at", "quantify-pixel",
             "region-quantify", "suggest-phases", "chemistry-mask-at",
             "chemistry-mask-counts", "probe", "linescan-at", "linescan-counts"]


@pytest.mark.parametrize("method,url,body", ROUTES, ids=ROUTE_IDS)
def test_every_route_names_the_spectrum_source(ready, client, method, url, body):
    r = getattr(client, method)(url, **({"json": body} if body else {}))
    assert r.status_code == 200, (url, r.text)
    quant = r.json().get("quantification")
    assert _is_spectrum(quant), (url, quant)
    fit = quant["fit"]
    # matrix-corrected spectrum source (the file's header gives the geometry):
    # Cu through K; _SPECTRUM_LINE_PREFERENCE (Cu L) applies only without it
    assert fit["lines"]["Cu"]["family"] == "K"
    assert np.isfinite(fit["chi2_red"])


def test_the_window_path_keeps_saying_window(ready, client):
    with svc.force_window_source():
        for method, url, body in ROUTES:
            r = getattr(client, method)(url, **({"json": body} if body else {}))
            assert r.status_code == 200, (url, r.text)
            quant = r.json()["quantification"]
            assert quant["source_data"] == ("vendor Window Integral "
                                            "(background included)"), url
            assert quant["source"] == "window" and "fit" not in quant, url


def test_the_shared_at_pct_builder_carries_the_fit(ready):
    from backend.api.routes.eds import (_build_at_pct_maps_for_loaded_file,
                                        _quant_provenance)
    at_maps, n_rows, n_cols, _fp = _build_at_pct_maps_for_loaded_file()
    assert (n_rows, n_cols) == (90, 120)
    assert _is_spectrum(_quant_provenance(at_maps))
    direct = direct_at_pct(ready, np.arange(n_rows * n_cols))
    assert sorted(map(str, at_maps)) == sorted(map(str, direct))
    for el in direct:
        np.testing.assert_allclose(at_maps[el], direct[el], rtol=1e-9, atol=0)


# ---------------------------------------------------------------------------
# the numbers: route == fit package + eds_utils
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row,col", PIXELS)
def test_quantify_pixel_equals_the_direct_computation(ready, client, row, col):
    r = client.post("/api/eds/quantify/pixel", json={"row": row, "col": col})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    idx = row * 120 + col
    direct = direct_at_pct(ready, idx)
    counts, _kv = direct_counts(ready, idx)
    assert sorted(_quantified(data)) == sorted(str(el) for el in direct)
    _carbon_left_out(data, r.json()["quantification"])
    for el in direct:
        assert data[str(el)]["at_pct"] == pytest.approx(
            float(direct[el][0]), rel=1e-9, abs=1e-12), el
        # the counts beside it are the net intensity of the family it used
        assert data[str(el)]["counts"] == pytest.approx(
            float(counts[el][0]), rel=1e-9, abs=1e-12), el
    assert r.json()["quantification"]["lines"]["Cu"]["line"] == "K"


@pytest.mark.parametrize("r0,r1,c0,c1", BOXES)
def test_region_quantify_equals_the_direct_computation(ready, client, r0, r1, c0, c1):
    r = client.post("/api/eds/region-quantify", json={
        "row_start": r0, "row_end": r1, "col_start": c0, "col_end": c1})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    idx = (rows * 120 + cols).ravel()
    direct = direct_at_pct(ready, idx)
    counts, _kv = direct_counts(ready, idx)
    # The composition ("mean") is the box's summed net counts quantified once
    # -- pinned in test_eds_region_composition.py. What stays
    # per pixel is pinned here: the spread of the pixels' at% and the mean of
    # their net cps.
    for el in direct:
        px = np.asarray(direct[el], dtype=np.float64)
        got = data[str(el)]["at_pct"]
        for key, ref in (("std", px.std()), ("min", px.min()),
                         ("max", px.max())):
            assert got[key] == pytest.approx(float(ref), rel=1e-9,
                                             abs=1e-12), (el, key)
        assert data[str(el)]["counts"]["mean"] == pytest.approx(
            float(np.mean(counts[el])), rel=1e-9, abs=1e-12), el
    assert r.json()["composition_method"] == "summed_net_counts"


@pytest.mark.parametrize("row,col", PIXELS)
def test_forcing_the_window_source_gives_the_window_numbers(ready, client, row, col):
    with svc.force_window_source():
        r = client.post("/api/eds/quantify/pixel", json={"row": row, "col": col})
    assert r.status_code == 200, r.text
    window = _window_at_pct(row * 120 + col)
    for el in window:
        assert r.json()["data"][str(el)]["at_pct"] == pytest.approx(
            float(window[el][0]), rel=1e-9, abs=1e-12), el


# ---------------------------------------------------------------------------
# /map counts: net counts are clipped for display, and it says so
# ---------------------------------------------------------------------------
def test_map_counts_clips_negative_net_counts_for_display(ready, client):
    net = h5_session.get_active_extractor().get_element_map("C Kα1,2")
    n_neg = int((net < 0).sum())
    assert n_neg > 0, "the test needs a map with negative net counts"
    r = client.get(f"/api/eds/map/{quote('C Kα1,2')}?mode=counts")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["min_val"] == 0.0
    assert body["max_val"] == pytest.approx(float(net.max()), rel=1e-6)
    quant = body["quantification"]
    assert quant["clipped_for_display"] is True
    assert quant["n_clipped"] == n_neg

    with svc.force_window_source():
        r = client.get(f"/api/eds/map/{quote('C Kα1,2')}?mode=counts")
    assert r.json()["quantification"]["clipped_for_display"] is False


# ---------------------------------------------------------------------------
# the EBSD Phase Map's EDS layers (/api/h5/eds/map?scope=dataset)
# ---------------------------------------------------------------------------
def _png_rgb(b64):
    import base64
    import io
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def _ramp_bottom(cmap="hot"):
    """The colour a map's minimum is drawn in (``colormap_array_to_base64``)."""
    import matplotlib
    return (np.asarray(matplotlib.colormaps[cmap](0.0)[:3]) * 255).astype(np.uint8)


@pytest.mark.parametrize("label", ["Fe Kα1", "C Kα1,2", "Zn Kα1"])
def test_phase_map_eds_layer_clips_net_maps_for_display(ready, client, label):
    """The Phase Map's ``eds:`` layer draws ``/api/h5/eds/map?scope=dataset``
    on a min-max ``hot`` ramp. Unclipped, zero sat 10-42 % up that ramp and
    the empty Al matrix came out as an opaque dark-red wash over the phase
    map. Clipped at 0 for display, every pixel without the line is drawn in
    the ramp's bottom colour -- the colour the window path draws its
    emptiest pixels in -- and the response says it was clipped."""
    net = h5_session.get_active_extractor().get_element_map_2d(label)
    absent = net <= 0
    assert (net < 0).any(), "the test needs a map with negative net counts"
    r = client.get(f"/api/h5/eds/map/{quote(label)}",
                   params={"cmap": "hot", "scope": "dataset"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["min_val"] == 0.0
    assert body["max_val"] == pytest.approx(float(net.max()), rel=1e-6)
    quant = body["quantification"]
    assert quant["source"] == "spectrum"
    assert quant["clipped_for_display"] is True
    assert quant["n_clipped"] == int((net < 0).sum())
    rgb = _png_rgb(body["image"])
    assert (rgb[absent] == _ramp_bottom()).all()
    # the matrix pixels of the window map are drawn in the same colour
    with svc.force_window_source():
        w = client.get(f"/api/h5/eds/map/{quote(label)}",
                       params={"cmap": "hot", "scope": "dataset"}).json()
    win = _png_rgb(w["image"])
    bottom = np.all(win == _ramp_bottom(), axis=-1)
    assert bottom.any()


@pytest.mark.parametrize("scope", ["dataset", "file"])
def test_phase_map_eds_layer_window_path_unchanged(ready, client, scope):
    """No cube in play -- ``force_window_source()`` or the H5 cockpit's
    ``file`` scope: the map is the raw window map, drawn as before, and the
    response carries no new key."""
    from backend.api.services.image_utils import colormap_array_to_base64
    label = "Fe Kα1"
    raw = h5_session.get_extractor().get_element_map_2d(label)
    if scope == "dataset":
        with svc.force_window_source():
            r = client.get(f"/api/h5/eds/map/{quote(label)}",
                           params={"cmap": "hot", "scope": scope})
    else:
        r = client.get(f"/api/h5/eds/map/{quote(label)}",
                       params={"cmap": "hot", "scope": scope})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"image", "element", "min_val", "max_val", "shape"}
    assert body["image"] == colormap_array_to_base64(raw, "hot")
    assert body["min_val"] == float(np.nanmin(raw))


def _quantified(data):
    """The elements a response actually quantified (an element the
    quantification left out keeps its counts and has no at%)."""
    def at(v):
        a = v.get("at_pct") if isinstance(v, dict) else None
        return a.get("mean") if isinstance(a, dict) else a
    return {el for el, v in data.items() if at(v) is not None}


def _carbon_left_out(data, quant):
    """The Oxford reference scan exports C K; the matrix-corrected composition leaves it out and
    says so (eds_utils.MATRIX_EXCLUDED_ELEMENTS)."""
    assert "C" in data and "C" not in _quantified(data)
    assert any(x["element"] == "C" for x in quant["excluded_windows"]), quant


# ---------------------------------------------------------------------------
# a bare symbol means the line the quantification uses (Cu -> Cu K here)
# ---------------------------------------------------------------------------
def test_a_bare_symbol_resolves_to_the_quantified_line(ready, client):
    res = svc.get_result(str(ready))
    for el, chosen in res.lines.items():
        assert res.family_for(el) == f"{el} {chosen['family']}", el
    assert res.family_for("Cu") == "Cu K"

    ext = h5_session.get_active_extractor()
    np.testing.assert_array_equal(ext.get_element_map("Cu"),
                                  ext.get_element_map("Cu Kα1"))
    cu_l = ext.get_element_map("Cu Lα1,2")
    cu_k = ext.get_element_map("Cu Kα1")
    assert float(cu_l.max()) != pytest.approx(float(cu_k.max()), rel=1e-3)

    # the phase-test dialog asks /map with a bare symbol
    r = client.get("/api/eds/map/Cu?mode=counts")
    assert r.status_code == 200, r.text
    assert r.json()["max_val"] == pytest.approx(float(cu_k.max()), rel=1e-6)


def test_linescan_counts_samples_each_layers_own_family(ready, client):
    ext = h5_session.get_active_extractor()
    body = {"start_row": 20, "start_col": 0, "end_row": 20, "end_col": 119,
            "n_samples": 120, "layers": ["eds-Cu Kα1", "eds-Cu Lα1,2"],
            "display_mode": "counts"}
    r = client.post("/api/eds/linescan", json=body)
    assert r.status_code == 200, r.text
    series = r.json()["series"]
    for label in ("Cu Kα1", "Cu Lα1,2"):
        row = ext.get_element_map(label).reshape(90, 120)[20]
        np.testing.assert_allclose(series[f"eds-{label}"], row, rtol=1e-6)


# ---------------------------------------------------------------------------
# an exported element the matrix correction leaves out (C on the Oxford reference scan): its
# counts map, a reason instead of an at% map, and the /elements list says so
# ---------------------------------------------------------------------------
def test_carbon_is_counts_only_and_named_not_quantified(ready, client):
    body = client.get("/api/eds/elements").json()
    assert body["not_quantified"] == ["C Kα1,2"]
    assert "C Kα1,2" in body["elements"]
    c = quote("C Kα1,2")
    assert client.get(f"/api/eds/map/{c}?mode=counts").status_code == 200
    r = client.get(f"/api/eds/map/{c}?mode=at_pct")
    assert r.status_code == 400 and "carbon is left out" in r.text


# ---------------------------------------------------------------------------
# a detected-not-exported element: a map, never a composition.
# The Oxford reference scan: Ti K at 5.2 sigma, not exported by AZtec.
# ---------------------------------------------------------------------------
def test_a_detected_element_is_a_map_and_not_a_composition(ready, client):
    r = client.get("/api/eds/elements")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "Ti Kα1" in body["detected_only"]
    assert not any(lab.split()[0] == "Ti" for lab in body["elements"])
    net = h5_session.get_active_extractor().get_element_map("Ti Kα1")

    # the notice's "add as map" on a page that shows at%: drawn as net
    # counts (display-clipped), and the response says so
    for mode in ("at_pct", "wt_pct", "counts"):
        m = client.get(f"/api/eds/map/{quote('Ti Kα1')}?mode={mode}")
        assert m.status_code == 200, (mode, m.text)
        js = m.json()
        assert js["mode"] == "counts" and js.get("detected_only") is True
        assert js["requested_mode"] == mode
        assert js["max_val"] == pytest.approx(float(net.max()), rel=1e-6)
        assert js["min_val"] == 0.0 and float(net.min()) < 0.0

    # a counts linescan samples it; an at% linescan has nothing for it
    body = {"start_row": 20, "start_col": 0, "end_row": 20, "end_col": 119,
            "n_samples": 120, "layers": ["eds-Ti Kα1", "eds-Al Kα1"]}
    lc = client.post("/api/eds/linescan", json={**body, "display_mode": "counts"})
    assert lc.status_code == 200, lc.text
    np.testing.assert_allclose(lc.json()["series"]["eds-Ti Kα1"],
                               net.reshape(90, 120)[20], rtol=1e-6)
    la = client.post("/api/eds/linescan", json={**body, "display_mode": "at_pct"})
    assert la.status_code == 200, la.text
    assert la.json()["series"]["eds-Ti Kα1"] == [None] * 120
    assert any(v is not None for v in la.json()["series"]["eds-Al Kα1"])

    # nothing quantifies it: pixel, region, the shared map builder
    q = client.post("/api/eds/quantify/pixel", json={"row": 84, "col": 5,
                                                     "display_mode": "at_pct"})
    assert q.status_code == 200, q.text
    assert "Ti" not in json.dumps(q.json().get("data", q.json()))
    rq = client.post("/api/eds/region-quantify",
                     json={"row_start": 40, "row_end": 49, "col_start": 50,
                           "col_end": 69})
    assert rq.status_code == 200, rq.text
    assert "Ti" not in rq.json()["data"]
    from backend.api.routes.eds import _build_at_pct_maps_for_loaded_file
    assert "Ti" not in {str(el) for el in _build_at_pct_maps_for_loaded_file()[0]}

    # and the notice sizes it from the whole map's summed net counts
    fit = svc.status(str(ready))["fit"]
    (ti,) = [d for d in fit["detected_not_exported"] if d["element"] == "Ti"]
    assert ti["label"] == "Ti Kα1"
    assert 0.0 < ti["at_pct_map_sum"] < 1.0


def test_two_families_of_one_element_are_expected_on_the_spectrum_source(caplog):
    pairs = [("Cu Kα1", np.ones(2)), ("Cu Lα1,2", np.ones(2)),
             ("Al Kα1", np.ones(2))]
    with caplog.at_level(logging.DEBUG, logger="eds_utils"):
        build_counts_by_element(pairs, source="spectrum")
    spec = [r for r in caplog.records if "windows for Cu" in r.getMessage()]
    assert spec and all(r.levelno == logging.DEBUG for r in spec)

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="eds_utils"):
        build_counts_by_element(pairs, source="window")
    win = [r for r in caplog.records if "windows for Cu" in r.getMessage()]
    assert win and all(r.levelno == logging.WARNING for r in win)


# ---------------------------------------------------------------------------
# preflight: counts per pixel are RAW window counts, fit or no fit
# ---------------------------------------------------------------------------
def _raw_window_median():
    raw = h5_session.get_extractor()
    tot = sum(np.asarray(raw.get_element_map(lab), dtype=np.float64)
              for lab in raw.get_available_elements())
    return float(np.median(tot))


def test_preflight_counts_per_pixel_are_window_counts(ready):
    from backend.api.services.eds_preflight import run_preflight
    out = run_preflight([])
    assert out["checks"]["eds_present"]["ok"] is True
    assert out["checks"]["signal"]["median_counts_per_px"] == pytest.approx(
        round(_raw_window_median(), 1), abs=1e-9)


def test_preflight_counts_do_not_wait_for_the_fit(pending, monkeypatch):
    """With a prior that reads the file's own source, the at% wait for the
    fit and the raw counts do not."""
    from backend.api.services import eds_indexing_prior as prior
    from backend.api.services.eds_preflight import run_preflight
    monkeypatch.setattr(prior, "PRIOR_EDS_SOURCE", None)
    out = run_preflight([])
    present = out["checks"]["eds_present"]
    assert present["ok"] is False
    assert "still being evaluated" in present["detail"]      # not a dict repr
    assert out["checks"]["signal"]["median_counts_per_px"] == pytest.approx(
        round(_raw_window_median(), 1), abs=1e-9)


@pytest.mark.parametrize("state", ["ready", "pending"])
def test_preflight_reads_the_priors_window_numbers(state, request):
    """The pre-flight predicts the prior, so it reads the prior's numbers:
    the window at% (measured window versus spectrum on the AZtec reference scan),
    with or without a finished fit. Pinned on a number the two sources give
    differently: the share of pixels where a phase's defining elements are
    all present (Al6Mn: Al and Mn >= 2 at%)."""
    path = request.getfixturevalue(state)
    from backend.api.services.eds_preflight import PRESENT_AT_PCT, run_preflight
    out = run_preflight(["Al6Mn.sht"], phase_formulas=["Al6Mn"])
    present = out["checks"]["eds_present"]
    assert present["ok"] is True
    raw = h5_session.get_extractor()
    want = sorted({str(parse_element_name(lab))
                   for lab in raw.get_available_elements()})
    assert [str(e) for e in present["elements"]] == want

    def coverage(at):
        ok = np.ones(90 * 120, dtype=bool)
        for el in ("Al", "Mn"):
            ok &= np.asarray(at[el], dtype=np.float64).ravel() >= PRESENT_AT_PCT
        return round(100.0 * float(ok.mean()), 2)

    every = np.arange(90 * 120)
    window = coverage({str(k): v for k, v in _window_at_pct(every).items()})
    (phase,) = out["phases"]
    assert phase["max_area_pct"] == window
    if state == "ready":
        spectrum = coverage({str(k): v for k, v in
                             direct_at_pct(path, every).items()})
        assert spectrum != window, (spectrum, window)


# ---------------------------------------------------------------------------
# the chemistry prior: kept on the window numbers; the refusal
# machinery below is for a prior that reads the file's own source
# ---------------------------------------------------------------------------
_PRIOR_PIXELS = [84 * 120 + 5, 20 * 120 + 92, 45 * 120 + 60]


def _assert_window_chemistry(got):
    assert len(got) == 90 * 120
    for i in _PRIOR_PIXELS:
        want = {str(k): float(v[0]) for k, v in _window_at_pct(i).items()
                if float(v[0]) > 0.0}
        assert set(got[i]) == set(want), (i, sorted(got[i]), sorted(want))
        for el, v in want.items():
            assert got[i][el] == pytest.approx(v, rel=1e-9), (i, el)


def test_the_prior_reads_the_window_numbers(ready):
    """Kept on the window path: the per-pixel chemistry the prior and the
    particle rescue weigh is the window at% of the raw maps, not the net at%
    the open file serves."""
    from backend.api.services import eds_indexing_prior as prior
    got = prior.measured_atpct_per_pixel()
    _assert_window_chemistry(got)
    # ...and that is a different answer from the file's own source here.
    spec = direct_at_pct(ready, _PRIOR_PIXELS[0])
    assert abs(got[_PRIOR_PIXELS[0]]["Si"] - float(spec["Si"][0])) > 0.5


def test_the_prior_does_not_wait_for_the_fit(pending):
    from backend.api.services import eds_indexing_prior as prior
    _assert_window_chemistry(prior.measured_atpct_per_pixel())


def test_an_indexing_run_during_the_fit_keeps_the_window_prior(pending):
    from backend.api.routes.indexing import _decide_eds_prior
    req = _request()
    run_req, refusal = _decide_eds_prior(req)
    assert refusal is None and run_req is req


@pytest.mark.parametrize("state", ["ready", "pending"])
def test_grain_phase_assignment_reads_the_window_numbers(state, request):
    """Grain phase assignment ranks with chemistry_fit, like the prior, so
    it reads the prior's numbers -- the window at% of the raw maps, with or
    without a finished fit."""
    path = request.getfixturevalue(state)
    from backend.api.routes.indexing import _grain_chemistry_maps
    at_maps, n_rows, n_cols, _fp = _grain_chemistry_maps()
    assert (n_rows, n_cols) == (90, 120)
    got = {str(k): np.asarray(v, dtype=np.float64) for k, v in at_maps.items()}
    every = np.arange(90 * 120)
    want = {str(k): np.asarray(v, dtype=np.float64)
            for k, v in _window_at_pct(every).items()}
    assert set(got) == set(want)
    for el in want:
        np.testing.assert_allclose(got[el], want[el], rtol=1e-9, atol=1e-12,
                                   err_msg=el)
    if state == "ready":
        # ...and that is not what the file's own source says here
        spec = direct_at_pct(path, _PRIOR_PIXELS[0])
        assert abs(got["Si"][_PRIOR_PIXELS[0]] - float(spec["Si"][0])) > 0.5


def test_pixel_chemistry_refuses_with_a_typed_reason_during_the_fit(
        pending, caplog, monkeypatch):
    """Pixel chemistry on the file's own source (PRIOR_EDS_SOURCE = None)."""
    from backend.api.services import eds_indexing_prior as prior
    from backend.api.services import eds_pixel_chemistry as epc
    monkeypatch.setattr(prior, "PRIOR_EDS_SOURCE", None)
    # an EBSD signal another test left in ebsd_viewer must not decide this
    monkeypatch.setattr(epc, "_get_active_signal", lambda: None)
    epc.reset_warning_dedupe()
    job = svc._job_for(str(pending), start=False)
    reasons = []
    with caplog.at_level(logging.WARNING):
        for step, i in enumerate((0, 1, 500, 9000)):
            # the fit moves on between pixels (the held fit would not), so
            # the message changes: still ONE log line
            job.progress = 0.30 + 0.01 * step
            reasons.append(epc.get_pixel_at_pct_with_reason(i))
    assert len({why.detail for _, why in reasons}) == 4
    assert all(at is None for at, _ in reasons)
    assert {why.code for _, why in reasons} == {"spectrum_fit_running"}
    assert all(why.expected is False for _, why in reasons)
    logged = [r for r in caplog.records if "spectrum_fit_running" in r.getMessage()]
    assert len(logged) == 1, [r.getMessage() for r in logged]


def test_pixel_chemistry_uses_the_net_intensities_when_ready(ready, monkeypatch):
    """Pixel chemistry on the file's own source (PRIOR_EDS_SOURCE = None)."""
    from backend.api.services import eds_indexing_prior as prior
    from backend.api.services import eds_pixel_chemistry as epc
    monkeypatch.setattr(prior, "PRIOR_EDS_SOURCE", None)
    monkeypatch.setattr(epc, "_get_active_signal", lambda: None)
    at, why = epc.get_pixel_at_pct_with_reason(84 * 120 + 5)
    assert why is None
    direct = direct_at_pct(ready, 84 * 120 + 5)
    for el in direct:
        assert at[str(el)] == pytest.approx(float(direct[el][0]), rel=1e-9)


@pytest.mark.parametrize("state", ["ready", "pending"])
def test_pixel_chemistry_reads_the_window_numbers(state, request, monkeypatch):
    """The phase test and the crystal hint rank with the prior's
    chemistry_fit, so they read the prior's numbers: the window at% of the
    raw maps, with or
    without a finished fit, and no spectrum-fit reason."""
    request.getfixturevalue(state)
    from backend.api.services import eds_pixel_chemistry as epc
    monkeypatch.setattr(epc, "_get_active_signal", lambda: None)
    for i in _PRIOR_PIXELS:
        at, why = epc.get_pixel_at_pct_with_reason(i)
        assert why is None
        want = {str(k): float(v[0]) for k, v in _window_at_pct(i).items()}
        assert set(at) == set(want)
        for el, v in want.items():
            assert at[el] == pytest.approx(v, rel=1e-9), (i, el)
    if state == "ready":
        spec = direct_at_pct(request.getfixturevalue("ready"), _PRIOR_PIXELS[0])
        assert abs(epc.get_pixel_at_pct(_PRIOR_PIXELS[0])["Si"]
                   - float(spec["Si"][0])) > 0.5


def _request(**kw):
    from backend.api.routes.indexing import IndexingStartRequest
    base = dict(method="hough", cif_paths=["a.cif", "b.cif"],
                eds_phase_strengths={"a.cif": 0.5, "b.cif": 0.0})
    base.update(kw)
    return IndexingStartRequest(**base)


def test_an_indexing_run_during_the_fit_runs_without_the_prior(pending, monkeypatch):
    """A prior on the file's own source: off for the whole run while the
    fit runs, with the typed reason."""
    from backend.api.routes.indexing import _decide_eds_prior
    from backend.api.services import eds_indexing_prior as prior
    monkeypatch.setattr(prior, "PRIOR_EDS_SOURCE", None)
    req = _request()
    run_req, refusal = _decide_eds_prior(req)
    assert refusal is not None and refusal.code == "spectrum_fit_running"
    assert run_req.eds_phase_strengths == {}
    assert req.eds_phase_strengths == {"a.cif": 0.5, "b.cif": 0.0}  # caller's untouched


def test_an_indexing_run_after_the_fit_keeps_the_prior(ready):
    from backend.api.routes.indexing import _decide_eds_prior
    req = _request()
    run_req, refusal = _decide_eds_prior(req)
    assert refusal is None and run_req is req


def test_a_run_without_the_prior_never_asks_for_the_fit(pending):
    from backend.api.routes.indexing import _decide_eds_prior
    req = _request(eds_phase_strengths={})
    run_req, refusal = _decide_eds_prior(req)
    assert refusal is None and run_req is req


def test_start_indexing_hands_the_run_the_decided_request(pending, monkeypatch, client):
    """The run closure sees the prior switched off, and the task says why
    (a prior on the file's own source)."""
    from backend.api.routes import indexing
    from backend.api.services import eds_indexing_prior as prior
    monkeypatch.setattr(prior, "PRIOR_EDS_SOURCE", None)
    captured = []
    monkeypatch.setattr(indexing, "_submit_run_guarded",
                        lambda fn, kind="indexing": captured.append(fn))
    from backend.api.routes import ebsd_viewer
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda: object())
    r = client.post("/api/indexing/start", json={
        "method": "hough", "cif_paths": ["a.cif", "b.cif"],
        "eds_phase_strengths": {"a.cif": 0.5}})
    assert r.status_code == 200, r.text
    task = indexing._indexing_tasks[r.json()["task_id"]]
    assert task["eds_prior_refused"]["code"] == "spectrum_fit_running"
    assert any("EDS chemistry prior OFF" in line for line in task["pending_log"])
    (fn,) = captured
    cells = dict(zip(fn.__code__.co_freevars, fn.__closure__))
    assert cells["req"].cell_contents.eds_phase_strengths == {}


def test_the_map_prior_does_not_swallow_a_lost_fit(pending, monkeypatch):
    """Decided at the start of the run; losing the chemistry half-way must
    not turn the prior off for the rest of it (a prior on the file's own
    source; the window numbers cannot be lost this way)."""
    from backend.api.services import eds_indexing_prior as prior
    monkeypatch.setattr(prior, "PRIOR_EDS_SOURCE", None)
    with pytest.raises(svc.SpectrumFitPending):
        prior.measured_atpct_per_pixel()


# ---------------------------------------------------------------------------
# batch export: the /EDS group holds AZtec's windows, under their labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ["ready", "pending"])
def test_batch_export_writes_the_raw_windows(state, request, tmp_path):
    request.getfixturevalue(state)
    from backend.api.routes.indexing import _write_eds_window_maps
    raw = h5_session.get_extractor()
    labels = raw.get_available_elements()
    with h5py.File(tmp_path / "x.h5", "w") as f:
        _write_eds_window_maps(f, h5_session.get_active_extractor())
    with h5py.File(tmp_path / "x.h5", "r") as f:
        grp = f["EDS"]
        assert list(grp.attrs["elements"]) == labels
        assert "Window Integral" in grp.attrs["source"]
        for lab in labels:
            np.testing.assert_array_equal(grp[lab.replace(" ", "_")][()],
                                          raw.get_element_map(lab))


def test_raw_window_consumers_follow_the_crop(ready, monkeypatch, tmp_path):
    """Under a crop, the raw windows they read are the crop's windows."""
    from backend.api.routes.indexing import _write_eds_window_maps
    from backend.api.services.crop_window import CropWindow
    from backend.api.services.eds_preflight import run_preflight

    window = CropWindow(str(ready), 10, 20, 30, 40, (90, 120))
    monkeypatch.setattr("backend.api.routes.ebsd_viewer.get_active_crop_window",
                        lambda: window)
    raw = h5_session.get_extractor()
    labels = raw.get_available_elements()

    def cut(lab):
        return np.asarray(raw.get_element_map(lab)).reshape(90, 120)[10:40, 20:60].ravel()

    with h5py.File(tmp_path / "c.h5", "w") as f:
        _write_eds_window_maps(f, h5_session.get_active_extractor())
    with h5py.File(tmp_path / "c.h5", "r") as f:
        for lab in labels:
            np.testing.assert_array_equal(f["EDS"][lab.replace(" ", "_")][()], cut(lab))

    tot = sum(cut(lab).astype(np.float64) for lab in labels)
    out = run_preflight([])
    assert out["checks"]["signal"]["median_counts_per_px"] == pytest.approx(
        round(float(np.median(tot)), 1), abs=1e-9)


# ---------------------------------------------------------------------------
# the 503 "still running" is not an error in the log
# ---------------------------------------------------------------------------
def test_fit_running_is_logged_at_info(pending, client, caplog):
    with caplog.at_level(logging.INFO, logger="backend.api.main"):
        r = client.post("/api/eds/quantify/pixel", json={"row": 1, "col": 1})
    assert r.status_code == 503
    recs = [x for x in caplog.records if "HTTP 503" in x.getMessage()]
    assert recs and all(x.levelno == logging.INFO for x in recs)


def test_other_server_errors_stay_errors(monkeypatch, synth_files, client, caplog):
    def boom(*_a, **_k):
        raise ValueError("synthetic fit failure")

    monkeypatch.setattr(svc, "RUN_IN_CHILD", False)     # the patch must reach it
    monkeypatch.setattr(svc, "fit_sum_spectrum", boom)
    h5_session.open_file(str(synth_files["A"]))
    with pytest.raises(svc.SpectrumFitFailed):
        svc.get_result(str(synth_files["A"]), wait_s=WAIT)
    with caplog.at_level(logging.INFO, logger="backend.api.main"):
        r = client.post("/api/eds/quantify/pixel", json={"row": 1, "col": 1})
    assert r.status_code == 500
    recs = [x for x in caplog.records if "HTTP 500" in x.getMessage()]
    assert recs and all(x.levelno == logging.ERROR for x in recs)


# ---------------------------------------------------------------------------
# the export says which source its at% came from
# ---------------------------------------------------------------------------
def _export(client, tmp_path, monkeypatch, quant_source="window"):
    from backend.api.routes import eds_export as export_routes
    from backend.api.services.cif_phase_library import CifPhaseEntry
    from backend.api.services.phase_map_store import PhaseMapState

    class _Store:
        def __init__(self, state):
            self._state = state

        def get_state(self):
            return self._state

    entry = CifPhaseEntry(key="Al", cif_filename="Al.cif", formula="Al",
                          space_group="Fm-3m", space_group_number=225,
                          crystal_system="cubic", composition={"Al": 100.0},
                          elements=["Al"])
    state = PhaseMapState(
        phase_grid=np.zeros((90, 120), dtype=np.int32),
        score_grid=np.full((90, 120), 0.75, dtype=np.float32),
        phase_entries=[entry], n_rows=90, n_cols=120, tolerance=15.0,
        min_score=0.3, region_grid=np.zeros((90, 120), dtype=np.int32),
        region_phase=[0], quant_source=quant_source)
    monkeypatch.setattr(export_routes, "get_phase_map_store", lambda: _Store(state))
    dest = tmp_path / "out"
    dest.mkdir(parents=True, exist_ok=True)
    r = client.post("/api/eds/export", json={"dest_dir": str(dest),
                                             "hash_source": False})
    assert r.status_code == 200, r.text
    return r.json()["provenance"]


def test_export_provenance_names_the_spectrum_fit(ready, client, tmp_path, monkeypatch):
    import json
    prov = _export(client, tmp_path / "s", monkeypatch)
    assert "eds-spectrum-fit/1" in json.dumps(prov)
    q = prov["quantification"]
    assert q["source"] == "spectrum"
    assert q["source_data"].startswith("spectrum fit")
    assert q["model_version"] == "eds-spectrum-fit/1"
    assert np.isfinite(q["chi2_red"])
    # matrix-corrected spectrum source (the file's header gives the geometry):
    # Cu through K; _SPECTRUM_LINE_PREFERENCE (Cu L) applies only without it
    assert q["lines"]["Cu"]["family"] == "K"
    assert q["lines"]["Fe"]["label"] == "Fe Kα1"
    assert "Aztec window integral" not in prov["caveat"]
    assert "net line intensities" in prov["caveat"]
    # matrix-corrected: the export says so, and no longer claims "no ZAF"
    assert "phi(rho z) matrix correction" in prov["caveat"]
    assert "no full ZAF" not in prov["caveat"]
    assert q["matrix_correction"]["exit_angle_deg"] == pytest.approx(81.0, abs=0.05)
    # the declared window overlaps do not apply to separated lines
    overlaps = prov["peak_overlaps"]
    assert overlaps["applies_to_this_scan"] == []
    assert "WINDOW INTEGRALS" not in overlaps["why"]
    assert {(d["interfering"], d["affected"])
            for d in overlaps["separated_by_the_fit"]} >= {("Mn", "Fe")}
    assert "Not DETECTED" not in prov["not_built"]["peak_overlap_warnings"]


def test_export_records_a_window_made_map_on_a_cube_file(ready, client, tmp_path,
                                                         monkeypatch):
    """A phase map classified on window numbers (an old sidecar, or one made
    while the window path was forced), exported once the file's spectra are
    evaluated: its regions were drawn by one source and its compositions are
    reported in the other. The provenance has to say both."""
    prov = _export(client, tmp_path / "ws", monkeypatch, quant_source="window")
    assert prov["quantification"]["source"] == "spectrum"
    comp = prov["composition"]
    assert comp["phase_map_quant_source"] == "window"
    note = comp["phase_map_quant_source_note"]
    assert "classified on the window numbers" in note
    assert "quantified from the spectrum numbers" in note


@pytest.mark.parametrize("map_source", ["spectrum", "window"])
def test_export_records_the_maps_own_source_without_a_note_when_it_matches(
        ready, client, tmp_path, monkeypatch, map_source):
    if map_source == "spectrum":
        prov = _export(client, tmp_path / "ss", monkeypatch, quant_source="spectrum")
    else:
        with svc.force_window_source():
            prov = _export(client, tmp_path / "ww", monkeypatch, quant_source="window")
    assert prov["quantification"]["source"] == map_source
    assert prov["composition"]["phase_map_quant_source"] == map_source
    assert prov["composition"]["phase_map_quant_source_note"] is None


def test_export_provenance_names_the_window_source(ready, client, tmp_path, monkeypatch):
    with svc.force_window_source():
        prov = _export(client, tmp_path / "w", monkeypatch)
    q = prov["quantification"]
    assert q["source"] == "window"
    assert q["source_data"] == "vendor Window Integral (background included)"
    assert "model_version" not in q
    assert "Aztec window integral" in prov["caveat"]
    assert ("Mn", "Fe") in {(d["interfering"], d["affected"])
                            for d in prov["peak_overlaps"]["applies_to_this_scan"]}
