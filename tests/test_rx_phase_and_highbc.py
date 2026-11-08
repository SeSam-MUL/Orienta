"""The RX analysis is made on ONE phase, and "high BC" counts the pixels of that phase against the largest band
contrast of all pixels (RxxAnalysis.m: ``RXedGrains(ph)``, ``sum(RXedGrains(ph).numPixel) /
sum(grains_smoothed(ph).numPixel)``, ``highBC = sum(ebsd(ph).bc > 0.7 * max(ebsd.bc)) / length(ebsd(ph))``).

The expected numbers are computed from the pixel arrays of the map here, not by the code under test.
"""
import base64
import io

import numpy as np
import openpyxl
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from analysis.ebsd_dataset import EBSDDataset
from analysis.grain_analysis import reconstruct_grains
from analysis.deformation_analysis import calculate_kam
from analysis.rx_analysis import (analyze_recrystallization, classify_rx_grains, calculate_grain_average_kam,
                                  grain_phase_ids, resolve_rx_phase, rx_map_classes)
from tests.batch_load_stub import patch_batch_load
from tests.synthetic_grain_map import add_defects, make_map, to_xmap

# every grain with a finite gKAM passes: the phase is then the only thing that decides
PERMISSIVE = {"rx_gos_threshold": 20.0, "rx_gbc_fraction": 0.0, "rx_gkam_threshold": 10.0, "rx_min_radius_um": 0.0}
KW = dict(gos_threshold_deg=20.0, gbc_min_fraction=0.0, gkam_threshold_deg=10.0, min_grain_radius_um=0.0)
RED, GREY = (215, 48, 39), (128, 128, 128)


def _dataset(bc=150.0, shape=(60, 80), n_seeds=30, defects=False):
    """A two-phase map; ``bc`` is a number or a function of the flat phase ids returning the band contrast per pixel;
    ``defects`` unindexes holes, a scratch and a square (their band contrast is whatever ``bc`` says)."""
    m = make_map(shape=shape, n_seeds=n_seeds)
    if defects:
        m = add_defects(m)
    xmap = to_xmap(m)
    phase = np.asarray(m["phase_id"]).reshape(-1)
    xmap.prop["bc"] = bc(phase) if callable(bc) else np.full(xmap.size, float(bc))
    ds = EBSDDataset(xmap, step_size=0.5)
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=5.0, min_pixels=3, alpha=3.1, fill=False)
    return ds


def _phase_of_grains(ds):
    """Phase of every grain from the pixels it covers (not from ``grains.grain_phase``)."""
    gid = ds.grains.grain_ids.reshape(-1)
    ph = np.asarray(ds.xmap.phase_id).reshape(-1)
    out = np.full(ds.grains.n_grains, -1)
    for g in range(1, ds.grains.n_grains + 1):
        out[g - 1] = ph[(gid == g) & (ph >= 0)][0]
    return out


def _expected_fraction(ds, rx_ids, phase):
    """RX pixels over the measured pixels in grains of the phase, from the pixel arrays."""
    gid = ds.grains.grain_ids.reshape(-1)
    ph = np.asarray(ds.xmap.phase_id).reshape(-1)
    in_grain = (gid > 0) & (ph == phase)
    return float((in_grain & np.isin(gid, rx_ids)).sum() / in_grain.sum() * 100.0)


@pytest.fixture(scope="module")
def two_phase():
    ds = _dataset()
    kam = calculate_kam(ds, grains=ds.grains)
    return ds, kam


def test_the_map_has_grains_of_both_phases_that_pass_the_criteria(two_phase):
    ds, kam = two_phase
    gk = calculate_grain_average_kam(kam, ds.grains.grain_ids)
    from analysis.bc_analysis import grain_average_bc
    gbc, _ = grain_average_bc(ds, ds.grains)
    every = classify_rx_grains(ds.grains, ds, gk, gbc, 20.0, 0.0, 10.0, 0.0)
    assert set(_phase_of_grains(ds)[every - 1]) == {0, 1}          # without a phase both phases are "RX"


@pytest.mark.parametrize("phase", [0, 1])
def test_only_the_grains_of_the_rx_phase_are_rx_and_the_fraction_is_of_its_pixels(two_phase, phase):
    ds, kam = two_phase
    res = analyze_recrystallization(ds, ds.grains, kam, rx_phase=phase, **KW)
    ids = np.asarray(res.rx_grain_ids)
    assert len(ids) > 0 and set(_phase_of_grains(ds)[ids - 1]) == {phase}
    assert res.phase_id == phase and res.phase_name == ["Al", "Al7FeCu2"][phase]
    assert res.rx_fraction == pytest.approx(_expected_fraction(ds, ids, phase))
    # the denominator is the phase's pixels: the other phase's grains do not dilute it (and do not add to it)
    other = analyze_recrystallization(ds, ds.grains, kam, rx_phase=1 - phase, **KW)
    assert not set(ids) & set(np.asarray(other.rx_grain_ids))
    assert res.rx_fraction == pytest.approx(100.0 * np.isin(ds.grains.grain_ids, ids).sum()
                                            / ((ds.grains.grain_ids > 0)
                                               & (np.asarray(ds.xmap.phase_id).reshape(ds.shape) == phase)).sum())


def test_most_measured_is_the_default_and_picks_the_phase_with_the_most_indexed_pixels(two_phase):
    ds, kam = two_phase
    ph = np.asarray(ds.xmap.phase_id)
    most = int(np.argmax([(ph == 0).sum(), (ph == 1).sum()]))
    assert resolve_rx_phase(ds) == resolve_rx_phase(ds, "most_measured") == most
    default = analyze_recrystallization(ds, ds.grains, kam, **KW)
    assert default.phase_id == most
    assert list(default.rx_grain_ids) == list(analyze_recrystallization(ds, ds.grains, kam, rx_phase=most, **KW).rx_grain_ids)


def test_an_unknown_or_malformed_rx_phase_is_a_value_error_that_lists_the_ids(two_phase):
    ds, kam = two_phase
    with pytest.raises(ValueError, match=r"rx phase 7 .*\[0, 1\]"):
        analyze_recrystallization(ds, ds.grains, kam, rx_phase=7, **KW)
    for bad in ("Al", 1.5, True, None, [1]):
        with pytest.raises(ValueError):
            resolve_rx_phase(ds, bad)


def test_the_rx_map_draws_the_grains_of_the_other_phases_neutral(two_phase):
    ds, kam = two_phase
    res = analyze_recrystallization(ds, ds.grains, kam, rx_phase=1, **KW)
    classes = rx_map_classes(ds, ds.grains, res.rx_grain_ids, phase_id=1)
    gp = np.concatenate([[-1], grain_phase_ids(ds, ds.grains)])[ds.grains.grain_ids]
    assert (classes[gp == 0] == -1).all() and (classes[gp == -1] == -1).all()       # phase 0 and no grain: neutral
    assert (classes[np.isin(ds.grains.grain_ids, res.rx_grain_ids)] == 3).all()
    # without a phase every grain keeps its GOS band
    assert (rx_map_classes(ds, ds.grains, None)[gp == 0] >= 0).all()


# ---------------------------------------------------------------------------------------------
# highBC: RxxAnalysis.m:47
# ---------------------------------------------------------------------------------------------
def _bright_holes(bright_pixels, bright_hole):
    """Band contrast: not-indexed pixels ``bright_hole`` (real Oxford files have band contrast there), every second
    pixel of phase 0 ``bright_pixels``, all other indexed pixels 100."""
    def bc(phase):
        out = np.where(phase >= 0, 100.0, bright_hole)
        out[np.flatnonzero(phase == 0)[::2]] = bright_pixels
        return out
    return bc


def test_high_bc_counts_the_pixels_of_the_phase_not_the_bright_not_indexed_ones():
    # threshold 0.7 x 255 = 178.5 of ALL pixels: no indexed pixel (at most 150) is above it, so highBC is 0; the code
    # that counted the not-indexed pixels in the numerator gave a positive number
    ds = _dataset(bc=_bright_holes(150.0, 255.0), defects=True)
    ph = np.asarray(ds.xmap.phase_id)
    bc_all = np.asarray(ds.bc)
    assert (ph < 0).sum() > 200 and 0.7 * bc_all.max() == pytest.approx(178.5)
    old = (bc_all > 0.7 * bc_all.max()).sum() / (ph >= 0).sum() * 100.0       # what the code counted before
    assert old > 1.0                                                          # the fixture can tell the two apart
    res = analyze_recrystallization(ds, ds.grains, calculate_kam(ds, grains=ds.grains), rx_phase=0, **KW)
    assert res.high_bc_fraction == pytest.approx(0.0)


def test_high_bc_is_the_phase_pixels_above_the_threshold_of_all_pixels():
    # largest value 270 (a not-indexed pixel): threshold 189, the 200 pixels (every second pixel of phase 0) count
    ds = _dataset(bc=_bright_holes(200.0, 270.0), defects=True)
    ph = np.asarray(ds.xmap.phase_id)
    bc_all = np.asarray(ds.bc)
    res = analyze_recrystallization(ds, ds.grains, calculate_kam(ds, grains=ds.grains), rx_phase=0, **KW)
    want = ((ph == 0) & (bc_all > 0.7 * 270.0)).sum() / (ph == 0).sum() * 100.0
    assert want == pytest.approx(50.0, abs=0.1) and res.high_bc_fraction == pytest.approx(want)


def test_high_bc_is_of_the_chosen_phase():
    ds = _dataset(bc=lambda phase: np.where(phase == 1, 240.0, 100.0))    # phase 1 is bright, phase 0 dark
    kam = calculate_kam(ds, grains=ds.grains)
    assert analyze_recrystallization(ds, ds.grains, kam, rx_phase=1, **KW).high_bc_fraction == pytest.approx(100.0)
    assert analyze_recrystallization(ds, ds.grains, kam, rx_phase=0, **KW).high_bc_fraction == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------
# the routes
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def client_two_phase():
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


def _rgb(body):
    return np.array(Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB"))


def test_rx_takes_the_phase_answers_it_and_remembers_it(client_two_phase):
    client, ar = client_two_phase
    ds = ar._dataset
    assert client.get("/api/analysis/status").json()["rx_phase"] == "most_measured"
    r = client.post("/api/analysis/rx", json={**PERMISSIVE, "phase": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rx_phase"] == {"id": 1, "name": "Al7FeCu2"}
    kam = calculate_kam(ds, grains=ds.grains)
    want = analyze_recrystallization(ds, ds.grains, kam, rx_phase=1, **KW)
    assert body["rx_fraction"] == pytest.approx(want.rx_fraction) and body["rx_grain_count"] == len(want.rx_grain_ids)
    assert body["high_bc_fraction"] == pytest.approx(want.high_bc_fraction)
    assert client.get("/api/analysis/status").json()["rx_phase"] == 1
    # a field left out keeps it; the criteria keep theirs too
    again = client.post("/api/analysis/rx", json={"rx_gos_threshold": 19.0}).json()
    assert again["rx_phase"]["id"] == 1 and client.get("/api/analysis/status").json()["rx_phase"] == 1
    # "most_measured" is a choice, not a reset
    r = client.post("/api/analysis/rx", json={"phase": "most_measured"})
    assert r.status_code == 200 and client.get("/api/analysis/status").json()["rx_phase"] == "most_measured"
    ph = np.asarray(ds.xmap.phase_id)
    assert r.json()["rx_phase"]["id"] == int(np.argmax([(ph == 0).sum(), (ph == 1).sum()]))


def test_an_unknown_phase_id_is_a_400_that_lists_the_ids_and_changes_nothing(client_two_phase):
    client, _ = client_two_phase
    client.post("/api/analysis/rx", json={"phase": 1})
    r = client.post("/api/analysis/rx", json={"phase": 9})
    assert r.status_code == 400 and "[0, 1]" in r.json()["detail"]
    assert client.get("/api/analysis/status").json()["rx_phase"] == 1


@pytest.mark.parametrize("value", ["Al", 1.5, True, [1], {"id": 1}])
def test_a_phase_of_another_type_is_a_422(client_two_phase, value):
    client, _ = client_two_phase
    assert client.post("/api/analysis/rx", json={"phase": value}).status_code == 422
    assert client.get("/api/analysis/status").json()["rx_phase"] == "most_measured"


def test_the_rx_phase_is_reset_with_the_other_settings(client_two_phase):
    client, ar = client_two_phase
    client.post("/api/analysis/rx", json={"phase": 1})
    ar._reset_grain_stats_settings()                       # what /load and a file switch do
    assert client.get("/api/analysis/status").json()["rx_phase"] == "most_measured"


def test_the_rx_map_and_its_numbers_follow_the_remembered_phase(client_two_phase):
    client, ar = client_two_phase
    ds = ar._dataset
    client.post("/api/analysis/rx", json={**PERMISSIVE, "phase": 1})
    body = client.get("/api/analysis/map/rx").json()
    assert body["rx_phase"] == {"id": 1, "name": "Al7FeCu2"}
    rgb = _rgb(body)
    gp = np.concatenate([[-1], grain_phase_ids(ds, ds.grains)])[ds.grains.grain_ids]
    assert (rgb[gp == 0] == GREY).all()                      # the grains of the other phase: neutral
    red = (rgb == RED).all(axis=-1)
    assert red.any() and (gp[red] == 1).all()
    kam = calculate_kam(ds, grains=ds.grains)
    want = analyze_recrystallization(ds, ds.grains, kam, rx_phase=1, **KW)
    assert body["rx_fraction"] == pytest.approx(want.rx_fraction)
    # the cached result of one phase is not served for the other
    client.post("/api/analysis/rx", json={"phase": 0})
    other = client.get("/api/analysis/map/rx").json()
    want0 = analyze_recrystallization(ds, ds.grains, kam, rx_phase=0, **KW)
    assert other["rx_phase"]["id"] == 0 and other["rx_grain_count"] == len(want0.rx_grain_ids)
    assert other["rx_grain_count"] != body["rx_grain_count"]


def test_the_export_is_of_the_remembered_rx_phase(client_two_phase, tmp_path):
    client, ar = client_two_phase
    client.post("/api/analysis/rx", json={**PERMISSIVE, "phase": 0})
    warnings = []
    rx = ar._collect_export_results(warnings)["rx"]
    assert rx.phase_id == 0 and warnings == []
    client.post("/api/analysis/rx", json={**PERMISSIVE, "phase": 1})
    warnings = []
    results = ar._collect_export_results(warnings)
    rx = results["rx"]
    assert rx.phase_id == 1 and rx.phase_name == "Al7FeCu2"
    from analysis.excel_exporter import export_to_excel
    path = str(tmp_path / "w.xlsx")
    export_to_excel(path, "s", results)
    ws = openpyxl.load_workbook(path)["Sheet1"]
    assert ws["A70"].value == "rx_phase" and ws["B70"].value == "Al7FeCu2"
    assert ws["B41"].value == pytest.approx(rx.rx_fraction) and ws["B42"].value == pytest.approx(rx.high_bc_fraction)


SPLIT_WARNING = ("the RX phase (Al7FeCu2) is not the texture phase (Al); the split needs one phase, "
                 "so the sheet is left out")


def _split_rows(path):
    ws = openpyxl.load_workbook(path)["SmallVSLargeGrain"]
    return [row for row in range(3, ws.max_row + 1) if ws.cell(row=row, column=1).value]


def test_an_rx_phase_that_is_not_the_texture_phase_leaves_the_split_out_and_says_so(client_two_phase, tmp_path):
    """The split counts texture-phase pixels in the RX grains: for another RX phase it would show every grain of the
    texture phase as "substructure" and no RX grain at all."""
    client, ar = client_two_phase
    from analysis.excel_exporter import export_to_excel
    # same phase: the split is made and no warning
    client.post("/api/analysis/rx", json={**PERMISSIVE, "phase": 0})
    warnings = []
    results = ar._collect_export_results(warnings)
    assert "texture_by_rx" in results and results["texture_by_rx"]["n_rx_pixels"] > 0 and warnings == []
    export_to_excel(str(tmp_path / "same.xlsx"), "s", results)
    assert _split_rows(str(tmp_path / "same.xlsx")) == [3]
    # another phase: no split, one warning that names both phases, and no row in the sheet
    client.post("/api/analysis/rx", json={"phase": 1})
    warnings = []
    results = ar._collect_export_results(warnings)
    assert "texture_by_rx" not in results and results["texture_by_rx_left_out"] is True
    assert warnings == ["SmallVSLargeGrain: " + SPLIT_WARNING]
    export_to_excel(str(tmp_path / "other.xlsx"), "s", results)
    assert _split_rows(str(tmp_path / "other.xlsx")) == []
    assert openpyxl.load_workbook(str(tmp_path / "other.xlsx"))["Sheet1"]["B70"].value == "Al7FeCu2"   # the rest is written


def test_the_split_guard_reaches_the_download_header(client_two_phase):
    import json
    import urllib.parse
    client, _ = client_two_phase
    client.post("/api/analysis/rx", json={**PERMISSIVE, "phase": 1})
    r = client.get("/api/analysis/export/excel-download", params={"name": "s"})
    assert r.status_code == 200
    assert "SmallVSLargeGrain: " + SPLIT_WARNING in json.loads(urllib.parse.unquote(r.headers["X-Orienta-Warnings"]))


# ---------------------------------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------------------------------
def _batch(tmp_path, monkeypatch, **kw):
    from analysis import batch_processor as bp
    m = make_map(shape=(40, 50), n_seeds=14)
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.random.default_rng(0).uniform(60, 200, xmap.size)
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    return bp, bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), **PERMISSIVE, **kw)


def test_the_batch_resolves_most_measured_per_file_and_takes_a_fixed_id(tmp_path, monkeypatch):
    bp, default = _batch(tmp_path, monkeypatch)
    res = default.process_file(tmp_path / "s.h5")
    xmap_ph = np.asarray(to_xmap(make_map(shape=(40, 50), n_seeds=14)).phase_id)
    most = int(np.argmax([(xmap_ph == 0).sum(), (xmap_ph == 1).sum()]))
    assert default.rx_phase == "most_measured" and res["rx"].phase_id == most
    _, fixed = _batch(tmp_path, monkeypatch, rx_phase=1 - most)
    assert fixed.process_file(tmp_path / "s.h5")["rx"].phase_id == 1 - most


def test_the_batch_refuses_a_malformed_rx_phase_and_leaves_a_missing_phase_out_with_a_warning(tmp_path, monkeypatch):
    bp, _ = _batch(tmp_path, monkeypatch)
    for bad in ("Al", 1.5, True, None):
        with pytest.raises(ValueError, match="rx_phase"):
            bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), rx_phase=bad)
    _, missing = _batch(tmp_path, monkeypatch, rx_phase=9)
    res = missing.process_file(tmp_path / "s.h5")
    assert res is not None and "rx" not in res and "texture_by_rx" not in res
    assert any(w["reason"].startswith("rx: ") and "[0, 1]" in w["reason"] for w in missing.warnings)


def test_the_batch_workbook_names_the_rx_phase(tmp_path, monkeypatch):
    bp, proc = _batch(tmp_path, monkeypatch, rx_phase=1)
    (tmp_path / "s.h5").write_bytes(b"x")
    assert proc.process_all() == 1
    ws = openpyxl.load_workbook(str(proc.workbook_path))["Sheet1"]
    assert ws["A70"].value == "rx_phase" and ws["B70"].value == "Al7FeCu2"


def test_the_batch_route_passes_the_rx_phase_only_when_given(client_two_phase, monkeypatch):
    client, _ = client_two_phase
    import analysis.batch_processor as bp
    from analysis.batch_processor import BatchResult
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder",
                        lambda folder, out, progress_callback=None, **kw: seen.append(kw)
                        or BatchResult(processed=["a.h5"], n_files=1))
    for body in ({"folder_path": "f", "output_path": "o"},
                 {"folder_path": "f", "output_path": "o", "rx_phase": "most_measured"},
                 {"folder_path": "f", "output_path": "o", "rx_phase": 2}):
        r = client.post("/api/analysis/batch", json=body)
        assert r.status_code == 200
        client.get(f"/api/analysis/task/{r.json()['task_id']}")
    assert seen == [{}, {"rx_phase": "most_measured"}, {"rx_phase": 2}]
    for bad in ("Al", 1.5, True):
        assert client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o",
                                                        "rx_phase": bad}).status_code == 422


def test_the_batch_leaves_the_split_out_for_an_rx_phase_that_is_not_the_texture_phase(tmp_path, monkeypatch):
    bp, other = _batch(tmp_path, monkeypatch, rx_phase=1)
    res = other.process_file(tmp_path / "s.h5")
    assert res["rx"].phase_id == 1 and res["texture"].phase_id == 0
    assert "texture_by_rx" not in res and res["texture_by_rx_left_out"] is True
    assert [w["reason"] for w in other.warnings] == ["SmallVSLargeGrain: " + SPLIT_WARNING]
    (tmp_path / "s.h5").write_bytes(b"x")
    assert other.process_all() == 1
    assert [w["reason"] for w in other.result().warnings] == ["SmallVSLargeGrain: " + SPLIT_WARNING]
    assert _split_rows(str(other.workbook_path)) == []
    # the same phase: the split is made
    _, same = _batch(tmp_path, monkeypatch, rx_phase=0)
    res = same.process_file(tmp_path / "s.h5")
    assert "texture_by_rx" in res and "texture_by_rx_left_out" not in res and same.warnings == []
