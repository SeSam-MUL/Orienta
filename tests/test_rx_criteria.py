"""The four RX criteria of the page (GOS threshold, gBC fraction, gKAM threshold, minimum grain radius)
reach the analysis, the Excel export and the batch.

Probe geometry (20 rows, 70 columns, step 1 um), four grains side by side, each with its own reason to be, or not
to be, recrystallized:
  grain 1: 24 columns, one orientation, band contrast 200  -> area 480, GOS 0, gKAM 0
  grain 2: 20 columns, one orientation, band contrast 120  -> area 400, GOS 0, gKAM 0, gBC 0.6 of the maximum
  grain 3:  6 columns, one orientation, band contrast 200  -> area 120 (radius 6.2 um), GOS 0, gKAM 0
  grain 4: 20 columns turning by 0.3 degrees per column, bc 200 -> area 400, GOS 1.5 degrees, gKAM about 0.25
The expectation of each case is computed here from these grain values with plain numpy comparisons, not with
``classify_rx_grains``.
"""
import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from tests.synthetic_grain_map import _axis_angle, _qmul, random_quaternions, to_xmap

NY = 20
WIDTHS = [24, 20, 6, 20]
NX = sum(WIDTHS)
AREA = np.array([480.0, 400.0, 120.0, 400.0])
BC = np.array([200.0, 120.0, 200.0, 200.0])
GOS_DEG = np.array([0.0, 0.0, 0.0, 1.5])
# The gKAM of the turning grain (grain 4) is not derived here: it is the value ``analyze_recrystallization``
# returns for this probe (checked by test_the_probe_is_what_the_expectations_assume, abs 0.002). The other
# three grains have one orientation each, so 0.0 is exact. The expectations are independent of the
# classification code, not of the KAM code.
GKAM_DEG = np.array([0.0, 0.0, 0.0, 0.2478])
RED = (215, 48, 39)
PERMISSIVE = {"rx_gos_threshold": 20.0, "rx_gbc_fraction": 0.0, "rx_gkam_threshold": 10.0, "rx_min_radius_um": 0.0}
DEFAULTS = {"rx_gos_threshold": 1.25, "rx_gbc_fraction": 0.7, "rx_gkam_threshold": 0.55, "rx_min_radius_um": 0.4}
FIELDS = tuple(DEFAULTS)


def _labels():
    labels = np.zeros((NY, NX), int)
    col = 0
    for k, w in enumerate(WIDTHS, 1):
        labels[:, col:col + w] = k
        col += w
    return labels


def _xmap():
    labels = _labels()
    rng = np.random.default_rng(11)
    base = random_quaternions(rng, 5)
    q = np.empty((NY, NX, 4))
    for k in range(1, 5):
        q[labels == k] = base[k]
    cols = np.broadcast_to(np.arange(NX), (NY, NX))
    turn = _axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(0.3 * (cols - sum(WIDTHS[:3]))))
    q4 = _qmul(turn, np.broadcast_to(base[4], (NY, NX, 4)))
    q[labels == 4] = q4[labels == 4]
    m = {"quats": q, "phase_id": np.zeros((NY, NX), np.int64), "phase_names": np.array(["Al"]),
         "point_groups": np.array(["m-3m"]), "step": 1.0}
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.array([BC[k - 1] for k in labels.ravel()])
    return xmap


def _probe():
    ds = EBSDDataset(_xmap(), step_size=1.0)
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=1, alpha=3.1, fill=False)
    return ds


def expected_rx(gos, gbc, gkam, rmin):
    """The 1-based ids of the grains that pass all four criteria, from the grain values above."""
    radius = np.sqrt(AREA / np.pi)
    ok = (GOS_DEG < gos) & (BC > gbc * BC.max()) & (GKAM_DEG < gkam) & (radius > rmin)
    return [int(k) for k in np.where(ok)[0] + 1]


def _args(body):
    return (body["rx_gos_threshold"], body["rx_gbc_fraction"], body["rx_gkam_threshold"], body["rx_min_radius_um"])


def _png(body):
    return np.array(Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB"))


def _red_grains(body):
    """The grains whose pixels are all drawn red (and none of the others has a red pixel)."""
    red = (_png(body) == RED).all(axis=-1)
    labels = _labels()
    whole = [k for k in range(1, 5) if red[labels == k].all()]
    assert red.sum() == sum(int((labels == k).sum()) for k in whole)
    return whole


@pytest.fixture
def client_probe():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    saved = ar._dataset, dict(ar._grain_stats_settings), dict(ar._rx_criteria)
    ar._dataset = _probe()
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])
        ar._rx_criteria.clear()
        ar._rx_criteria.update(saved[2])


def test_the_probe_is_what_the_expectations_assume():
    ds = _probe()
    assert ds.grains.n_grains == 4
    assert list(ds.grains.area) == list(AREA)
    assert np.degrees(ds.grains.gos) == pytest.approx(GOS_DEG, abs=0.01)
    from analysis.deformation_analysis import calculate_kam
    from analysis.rx_analysis import analyze_recrystallization
    res = analyze_recrystallization(ds, ds.grains, calculate_kam(ds, grains=ds.grains), 20, 0, 10, 0)
    assert res.g_kam == pytest.approx(GKAM_DEG, abs=0.002) and res.g_bc == pytest.approx(BC)
    # every criterion alone removes grains from the permissive set: a criterion the backend ignored would show
    assert expected_rx(*_args(DEFAULTS)) == [1, 3]
    assert all(expected_rx(*_args(c)) != [1, 2, 3, 4] for n, c in CASES.items() if n != "permissive")


CASES = {
    "permissive": PERMISSIVE,
    "gos": {**PERMISSIVE, "rx_gos_threshold": 0.75},
    "gbc": {**PERMISSIVE, "rx_gbc_fraction": 0.8},
    "gkam": {**PERMISSIVE, "rx_gkam_threshold": 0.12},
    "radius": {**PERMISSIVE, "rx_min_radius_um": 8.0},
    "defaults": DEFAULTS,
}


def test_every_criterion_changes_the_classification(client_probe):
    client, _ = client_probe
    assert [expected_rx(*_args(CASES[k])) for k in ("permissive", "gos", "gbc", "gkam", "radius")] == [
        [1, 2, 3, 4], [1, 2, 3], [1, 3, 4], [1, 2, 3], [1, 2, 4]]
    for name, body in CASES.items():
        want = expected_rx(*_args(body))
        r = client.post("/api/analysis/rx", json=body)
        assert r.status_code == 200, (name, r.text)
        out = r.json()
        assert out["rx_grain_count"] == len(want), name
        assert out["rx_fraction"] == pytest.approx(AREA[[k - 1 for k in want]].sum() / AREA.sum() * 100), name
        assert out["total_grains"] == 4 and out["rx_needs_bc"] is False and out["rx_drawn"] is True
        # the map the page then fetches draws the same grains red, from the stored criteria
        assert _red_grains(client.get("/api/analysis/map/rx").json()) == want, name


def test_the_response_carries_the_criteria_that_were_used(client_probe):
    client, _ = client_probe
    body = {"rx_gos_threshold": 2.0, "rx_gbc_fraction": 0.5, "rx_gkam_threshold": 0.3, "rx_min_radius_um": 1.0}
    out = client.post("/api/analysis/rx", json=body).json()
    assert {k: out[k] for k in FIELDS} == body and out["success"] is True
    got = client.get("/api/analysis/map/rx").json()
    assert {k: got[k] for k in FIELDS} == body             # the map says which criteria drew it


def test_a_field_left_out_keeps_the_stored_value(client_probe):
    client, _ = client_probe
    client.post("/api/analysis/rx", json=PERMISSIVE)
    out = client.post("/api/analysis/rx", json={"rx_gos_threshold": 0.75}).json()
    assert {k: out[k] for k in FIELDS} == {**PERMISSIVE, "rx_gos_threshold": 0.75}
    assert out["rx_grain_count"] == 3
    out = client.post("/api/analysis/rx", json={}).json()
    assert {k: out[k] for k in FIELDS} == {**PERMISSIVE, "rx_gos_threshold": 0.75}


def test_the_map_alone_uses_the_defaults_until_the_page_classifies(client_probe):
    client, _ = client_probe
    got = client.get("/api/analysis/map/rx").json()
    assert {k: got[k] for k in FIELDS} == DEFAULTS and _red_grains(got) == [1, 3]
    assert got["rx_grain_count"] == 2


def test_a_get_of_the_map_never_changes_the_stored_criteria(client_probe):
    client, ar = client_probe
    client.get("/api/analysis/map/rx", params={"rx_gos_threshold": 20, "rx_gbc_fraction": 0, "cmap": "gray"})
    assert ar._rx_criteria == DEFAULTS


def test_status_returns_the_criteria(client_probe):
    client, _ = client_probe
    st = client.get("/api/analysis/status").json()
    assert {k: st[k] for k in FIELDS} == DEFAULTS
    client.post("/api/analysis/rx", json={"rx_gkam_threshold": 0.3})
    st = client.get("/api/analysis/status").json()
    assert {k: st[k] for k in FIELDS} == {**DEFAULTS, "rx_gkam_threshold": 0.3}


def test_the_criteria_go_back_to_the_defaults_when_a_map_is_loaded(client_probe, monkeypatch):
    client, ar = client_probe
    monkeypatch.setattr("analysis.xmap_loader.load_xmap_file",
                        lambda path: (_xmap(), 1.0, {"step_from_file": True}))
    client.post("/api/analysis/rx", json=PERMISSIVE)
    assert ar._rx_criteria == PERMISSIVE
    r = client.post("/api/analysis/load", json={"xmap_path": "other.h5"})
    assert r.status_code == 200, r.text
    assert ar._rx_criteria == DEFAULTS
    st = client.get("/api/analysis/status").json()
    assert {k: st[k] for k in FIELDS} == DEFAULTS


def test_the_criteria_go_back_to_the_defaults_when_the_map_of_a_file_is_restored(client_probe, monkeypatch):
    client, ar = client_probe
    monkeypatch.setattr(ar, "_dataset_by_file", {"other.h5": _probe()})      # undone with the monkeypatch
    client.post("/api/analysis/rx", json=PERMISSIVE)
    assert ar.restore_dataset_for_file("other.h5") is True
    assert ar._rx_criteria == DEFAULTS


@pytest.mark.parametrize("bad", [
    {"rx_gos_threshold": -0.1}, {"rx_gos_threshold": 20.1},
    {"rx_gbc_fraction": -0.01}, {"rx_gbc_fraction": 1.01},
    {"rx_gkam_threshold": -1}, {"rx_gkam_threshold": 10.5},
    {"rx_min_radius_um": -0.5}, {"rx_min_radius_um": 100.5},
    {"rx_gos_threshold": "abc"}, {"rx_gbc_fraction": "NaN"},
])
def test_a_criterion_out_of_its_range_is_a_422_and_stores_nothing(client_probe, bad):
    client, ar = client_probe
    r = client.post("/api/analysis/rx", json={**PERMISSIVE, **bad})
    assert r.status_code == 422, r.text
    assert ar._rx_criteria == DEFAULTS


def test_the_ends_of_the_ranges_are_accepted(client_probe):
    client, ar = client_probe
    lo = {"rx_gos_threshold": 0, "rx_gbc_fraction": 0, "rx_gkam_threshold": 0, "rx_min_radius_um": 0}
    hi = {"rx_gos_threshold": 20, "rx_gbc_fraction": 1, "rx_gkam_threshold": 10, "rx_min_radius_um": 100}
    for body in (lo, hi):
        assert client.post("/api/analysis/rx", json=body).status_code == 200
        assert ar._rx_criteria == {k: float(v) for k, v in body.items()}


def test_the_post_needs_a_map_and_grains(client_probe):
    client, ar = client_probe
    ar._dataset.grains = None
    r = client.post("/api/analysis/rx", json=PERMISSIVE)
    assert r.status_code == 400 and "Reconstruct grains first" in r.json()["detail"]
    assert ar._rx_criteria == DEFAULTS
    ar._dataset = None
    assert client.post("/api/analysis/rx", json=PERMISSIVE).status_code == 400


def test_without_native_band_contrast_the_criteria_are_stored_but_nothing_is_classified(client_probe):
    client, ar = client_probe
    del ar._dataset.xmap.prop["bc"]
    out = client.post("/api/analysis/rx", json=PERMISSIVE).json()
    assert out["rx_needs_bc"] is True and out["rx_drawn"] is False
    assert "rx_fraction" not in out and "rx_grain_count" not in out
    assert ar._rx_criteria == PERMISSIVE


def test_the_export_uses_the_stored_criteria_and_writes_the_four_values(client_probe, tmp_path):
    from openpyxl import load_workbook
    from analysis.excel_exporter import RX_CRITERIA_KEYS, export_to_excel
    client, ar = client_probe
    for body in (None, {"rx_gos_threshold": 2.0, "rx_gbc_fraction": 0.5, "rx_gkam_threshold": 0.3,
                        "rx_min_radius_um": 8.0}):
        used = DEFAULTS
        if body is not None:
            client.post("/api/analysis/rx", json=body)
            used = body
        want = expected_rx(*_args(used))
        assert want == ([1, 3] if body is None else [1, 2, 4])
        want_fraction = AREA[[k - 1 for k in want]].sum() / AREA.sum() * 100
        results = ar._collect_export_results()
        rx = results["rx"]
        assert (rx.gos_threshold, rx.gbc_min_fraction, rx.gkam_threshold, rx.min_grain_radius_um) == _args(used)
        assert rx.rx_fraction == pytest.approx(want_fraction)
        out = tmp_path / f"doc_{body is None}.xlsx"
        export_to_excel(str(out), "probe", results)
        ws = load_workbook(str(out))["Sheet1"]
        assert [ws.cell(row=60 + i, column=1).value for i in range(4)] == list(RX_CRITERIA_KEYS)
        assert [ws.cell(row=60 + i, column=2).value for i in range(4)] == pytest.approx(list(_args(used)))
        assert ws.cell(row=41, column=2).value == pytest.approx(want_fraction)


def test_the_export_leaves_the_criteria_empty_when_no_rx_analysis_ran(client_probe, tmp_path):
    from openpyxl import load_workbook
    from analysis.excel_exporter import export_to_excel
    _, ar = client_probe
    del ar._dataset.xmap.prop["bc"]
    results = ar._collect_export_results()
    assert "rx" not in results
    export_to_excel(str(tmp_path / "d.xlsx"), "probe", results)
    ws = load_workbook(str(tmp_path / "d.xlsx"))["Sheet1"]
    assert [ws.cell(row=60 + i, column=2).value for i in range(4)] == [None] * 4      # nothing was used


# ---------------------------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------------------------
def _batch_result(**kw):
    from analysis.batch_processor import BatchResult
    return BatchResult(**{"processed": [], "failed": [], "skipped": [], "n_files": 0, **kw})


def test_the_batch_route_passes_the_criteria_only_when_given_and_rejects_nonsense(client_probe, monkeypatch):
    client, _ = client_probe
    import analysis.batch_processor as bp
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder",
                        lambda folder, out, progress_callback=None, **kw: seen.append(kw)
                        or _batch_result(processed=["a.h5"], n_files=1))
    base = {"folder_path": "f", "output_path": "o"}
    for body in (base, {**base, **PERMISSIVE}, {**base, "rx_gkam_threshold": 0.3}):
        r = client.post("/api/analysis/batch", json=body)
        assert r.status_code == 200, r.text
        task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
        assert task["status"] == "completed", task
    assert seen[0] == {} and seen[1] == PERMISSIVE and seen[2] == {"rx_gkam_threshold": 0.3}
    for bad in ({"rx_gos_threshold": 21}, {"rx_gbc_fraction": 1.5}, {"rx_gkam_threshold": -1},
                {"rx_min_radius_um": 101}):
        assert client.post("/api/analysis/batch", json={**base, **bad}).status_code == 422, bad


def test_the_batch_processor_refuses_a_criterion_out_of_its_range(tmp_path):
    from analysis.batch_processor import BatchProcessor
    for bad in ({"rx_gos_threshold": 21}, {"rx_gbc_fraction": -0.1}, {"rx_gkam_threshold": 11},
                {"rx_min_radius_um": -1}):
        with pytest.raises(ValueError, match=next(iter(bad))):
            BatchProcessor(str(tmp_path), str(tmp_path / "o.xlsx"), **bad)
    p = BatchProcessor(str(tmp_path), str(tmp_path / "o.xlsx"))
    assert (p.rx_gos_threshold, p.rx_gbc_fraction, p.rx_gkam_threshold, p.rx_min_radius_um) == _args(DEFAULTS)


@pytest.mark.parametrize("name", ["permissive", "gos", "gbc", "gkam", "radius", "defaults"])
def test_the_batch_classifies_with_its_criteria_and_writes_them(tmp_path, monkeypatch, name):
    from openpyxl import load_workbook
    from analysis import batch_processor as bp
    from tests.batch_load_stub import patch_batch_load
    from tests.test_analysis_page_sync import _fake_texture
    (tmp_path / "a.h5").write_bytes(b"")
    monkeypatch.setattr(bp, "analyze_texture", _fake_texture)
    patch_batch_load(monkeypatch, bp, lambda path: _xmap())
    body = CASES[name]
    out = tmp_path / "doc.xlsx"
    res = bp.batch_process_folder(str(tmp_path), str(out), gb_threshold_deg=5.0, min_grain_pixels=1,
                                  rx_gos_threshold=body["rx_gos_threshold"], rx_gbc_fraction=body["rx_gbc_fraction"],
                                  rx_gkam_threshold=body["rx_gkam_threshold"], rx_min_radius_um=body["rx_min_radius_um"])
    assert res.n_processed == 1, res.failed
    ws = load_workbook(str(out))["Sheet1"]
    want = expected_rx(*_args(body))
    assert ws.cell(row=41, column=2).value == pytest.approx(AREA[[k - 1 for k in want]].sum() / AREA.sum() * 100)
    assert [ws.cell(row=60 + i, column=2).value for i in range(4)] == pytest.approx(list(_args(body)))


# ---------------------------------------------------------------------------------------------
# a POST that cannot store, and the cache of the RX result
# ---------------------------------------------------------------------------------------------
def test_a_post_that_could_not_store_because_the_map_changed_is_a_409(client_probe, monkeypatch):
    client, ar = client_probe
    real = ar._rx_panel

    def replace_meanwhile(dataset, grains, criteria, strict=False, **more):
        out = real(dataset, grains, criteria, strict, **more)
        ar._dataset = _probe()                     # another map is loaded while this one was classifying
        return out

    monkeypatch.setattr(ar, "_rx_panel", replace_meanwhile)
    r = client.post("/api/analysis/rx", json=PERMISSIVE)
    assert r.status_code == 409
    assert "Press Classify again" in r.json()["detail"] and "not stored" in r.json()["detail"]
    assert ar._rx_criteria == DEFAULTS


def test_a_post_whose_grains_were_replaced_meanwhile_is_a_409_too(client_probe, monkeypatch):
    client, ar = client_probe
    real = ar._rx_panel

    def reconstruct_meanwhile(dataset, grains, criteria, strict=False, **more):
        out = real(dataset, grains, criteria, strict, **more)
        dataset.grains = reconstruct_grains(dataset, angle_threshold_deg=5.0, min_pixels=1, alpha=3.1, fill=False)
        return out

    monkeypatch.setattr(ar, "_rx_panel", reconstruct_meanwhile)
    assert client.post("/api/analysis/rx", json=PERMISSIVE).status_code == 409
    assert ar._rx_criteria == DEFAULTS


@pytest.fixture
def rx_calls(monkeypatch):
    """Counts the calls of the RX analysis the routes make."""
    import analysis.rx_analysis as rxa
    real = rxa.analyze_recrystallization
    calls = []

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(rxa, "analyze_recrystallization", counting)
    return calls


def test_classify_and_the_map_fetch_that_follows_compute_the_rx_analysis_once(client_probe, rx_calls):
    client, _ = client_probe
    body = CASES["gos"]
    first = client.post("/api/analysis/rx", json=body).json()
    got = client.get("/api/analysis/map/rx").json()
    got_again = client.get("/api/analysis/map/rx", params={"cmap": "gray"}).json()
    assert len(rx_calls) == 1
    assert _red_grains(got) == _red_grains(got_again) == expected_rx(*_args(body))
    for key in ("rx_fraction", "rx_grain_count", "total_grains", "rx_drawn", *FIELDS):
        assert got[key] == first[key], key


def test_the_cached_fields_are_a_copy(client_probe, rx_calls):
    client, ar = client_probe
    client.get("/api/analysis/map/rx")
    ids, fields = ar._rx_panel(ar._dataset, ar._dataset.grains, dict(ar._rx_criteria))
    fields["rx_fraction"] = -1.0
    again = client.get("/api/analysis/map/rx").json()
    assert again["rx_fraction"] != -1.0 and len(rx_calls) == 1


def test_other_criteria_invalidate_the_cache(client_probe, rx_calls):
    client, _ = client_probe
    client.post("/api/analysis/rx", json=CASES["gos"])
    client.post("/api/analysis/rx", json=CASES["gbc"])
    assert len(rx_calls) == 2
    assert _red_grains(client.get("/api/analysis/map/rx").json()) == expected_rx(*_args(CASES["gbc"]))
    assert len(rx_calls) == 2


def test_new_grains_invalidate_the_cache(client_probe, rx_calls):
    client, ar = client_probe
    client.get("/api/analysis/map/rx")
    client.get("/api/analysis/map/rx")
    assert len(rx_calls) == 1
    ar._dataset.grains = reconstruct_grains(ar._dataset, angle_threshold_deg=5.0, min_pixels=1, alpha=3.1, fill=False)
    client.get("/api/analysis/map/rx")
    assert len(rx_calls) == 2


def test_another_map_invalidates_the_cache(client_probe, rx_calls):
    client, ar = client_probe
    client.get("/api/analysis/map/rx")
    ar._dataset = _probe()                          # the same pixels, another dataset object
    ar._reset_grain_stats_settings()
    client.get("/api/analysis/map/rx")
    assert len(rx_calls) == 2
    # also without the reset: the dataset object alone is part of the key
    ar._dataset = _probe()
    client.get("/api/analysis/map/rx")
    assert len(rx_calls) == 3


def test_the_quality_filter_clears_the_cache_and_the_next_result_is_for_the_remaining_pixels(client_probe, rx_calls):
    client, ar = client_probe
    before = client.get("/api/analysis/map/rx").json()
    assert before["rx_fraction"] == pytest.approx(600 / 1400 * 100)          # grains 1 and 3 of 1400 pixels
    assert ar._rx_cache
    r = client.post("/api/analysis/quality-filter", json={"bc_min": 150.0, "bands_min": 0})
    assert r.status_code == 200, r.text                                      # grain 2 (bc 120) is unindexed now
    assert not ar._rx_cache
    ar._dataset.grains = reconstruct_grains(ar._dataset, angle_threshold_deg=5.0, min_pixels=1, alpha=3.1, fill=False)
    after = client.get("/api/analysis/map/rx").json()
    assert len(rx_calls) == 2
    assert after["rx_fraction"] == pytest.approx(600 / 1000 * 100)           # the same grains of 1000 indexed pixels


def test_a_failed_analysis_is_not_cached(client_probe, monkeypatch):
    client, ar = client_probe
    import analysis.rx_analysis as rxa
    real = rxa.analyze_recrystallization
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("once")
        return real(*args, **kwargs)

    monkeypatch.setattr(rxa, "analyze_recrystallization", flaky)
    first = client.get("/api/analysis/map/rx").json()
    assert first["rx_drawn"] is False and "rx_fraction" not in first
    second = client.get("/api/analysis/map/rx").json()
    assert second["rx_drawn"] is True and len(calls) == 2


# ---------------------------------------------------------------------------------------------
# the cache never outlives its map or its grains; the map is built from one snapshot
# ---------------------------------------------------------------------------------------------
def test_a_reconstruction_clears_the_cache(client_probe):
    client, ar = client_probe
    client.get("/api/analysis/map/rx")
    assert ar._rx_cache
    r = client.post("/api/analysis/grains/reconstruct", json={
        "misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 3.1, "fill": False})
    assert r.status_code == 200, r.text
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task
    assert not ar._rx_cache


def test_a_result_for_a_map_that_was_replaced_meanwhile_is_not_cached(client_probe, monkeypatch):
    client, ar = client_probe
    import analysis.rx_analysis as rxa
    real = rxa.analyze_recrystallization
    first = ar._dataset

    def replace_during_compute(*args, **kwargs):
        out = real(*args, **kwargs)
        ar._dataset = _probe()                  # another map is loaded while the first one is being computed
        return out

    monkeypatch.setattr(rxa, "analyze_recrystallization", replace_during_compute)
    body = client.get("/api/analysis/map/rx").json()
    assert body["rx_drawn"] is True            # the answer for the map that was asked about
    assert "entry" not in ar._rx_cache or ar._rx_cache["entry"]["dataset"] is not first
    assert not ar._rx_cache


def test_the_map_is_built_from_the_grains_it_started_with(client_probe, monkeypatch):
    """The grains are replaced while the map is being built: the RX ids and the drawing use one set of grains."""
    client, ar = client_probe
    import analysis.rx_analysis as rxa
    old_grains = ar._dataset.grains
    real_panel = ar._rx_panel
    drawn_with = []

    def swap_after_panel(dataset, grains, criteria, strict=False, **more):
        out = real_panel(dataset, grains, criteria, strict, **more)
        dataset.grains = reconstruct_grains(dataset, angle_threshold_deg=5.0, min_pixels=1, alpha=3.1, fill=False)
        return out

    real_classes = rxa.rx_map_classes

    def record(dataset, grains, rx_ids, **more):
        drawn_with.append(grains)
        return real_classes(dataset, grains, rx_ids, **more)

    monkeypatch.setattr(ar, "_rx_panel", swap_after_panel)
    monkeypatch.setattr(rxa, "rx_map_classes", record)
    body = client.get("/api/analysis/map/rx").json()
    assert ar._dataset.grains is not old_grains            # the swap happened
    assert drawn_with == [old_grains] and drawn_with[0] is old_grains
    assert _red_grains(body) == [1, 3]
