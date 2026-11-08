"""The Texture tab's preset and tolerance angle reach the analysis.

``POST /texture`` takes ``preset`` (one of the presets the page offers) and ``tolerance_deg`` (more than 0, at most 30),
remembers them with the phase (reset on a load, returned by ``/status``), and the Excel export, the split into RX and
substructure grains, the batch and the "Texture Components" map all use them. Sheet 1 records them (rows 68, 69).
"""
import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("orix")

from analysis.ebsd_dataset import EBSDDataset                                    # noqa: E402
from analysis.texture_components import TEXTURE_PRESETS                          # noqa: E402

CUBE = [1.0, 0.0, 0.0, 0.0]
OFF3 = [float(np.cos(np.radians(1.5))), 0.0, 0.0, float(np.sin(np.radians(1.5)))]     # Cube turned by 3 degrees about ND
# 28 degrees from every FCC_Rolling component (found by search): in no component at any tolerance used here
FAR = [-0.07189701520447721, -0.6852135615895414, -0.5049332994823804, -0.5199572648711621]
BLOCKS = [CUBE, OFF3, OFF3, FAR, FAR, FAR]          # 6 blocks of 4 x 4 pixels: 3 within 10 degrees of Cube, 1 within 2


def _xmap():
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation
    q = np.zeros((8, 12, 4))
    for b, bq in enumerate(BLOCKS):
        r0, c0 = (b // 3) * 4, (b % 3) * 4
        q[r0:r0 + 4, c0:c0 + 4] = bq
    return CrystalMap(rotations=Rotation(q.reshape(-1, 4)), phase_id=np.zeros(96, int),
                      x=np.tile(np.arange(12, dtype=float), 8), y=np.repeat(np.arange(8, dtype=float), 12),
                      phase_list=PhaseList(names=["Al"], point_groups=["m-3m"]),
                      prop={"bc": np.random.default_rng(2).uniform(20, 200, 96)})


@pytest.fixture
def api():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    from analysis.grain_analysis import reconstruct_grains
    saved = ar._dataset, dict(ar._texture_settings), dict(ar._grain_stats_settings)
    ds = EBSDDataset(_xmap(), step_size=1.0)
    ds.grains = reconstruct_grains(ds, angle_threshold_deg=2.0, min_pixels=1, alpha=0.0, fill=False)
    ar._dataset = ds
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._texture_settings.clear()
        ar._texture_settings.update(saved[1])
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[2])


def _fraction(body, name):
    return {c["name"]: c["fraction"] for c in body["components"]}[name]


def _sheet1(results, tmp_path):
    from analysis.excel_exporter import ExcelExporter
    ex = ExcelExporter(str(tmp_path / "d.xlsx"))
    ex.export_dataset("ds", results)
    return ex.wb[ex.wb.sheetnames[0]]


def _row_of(ws, label):
    return next(r for r in range(1, 80) if ws.cell(row=r, column=1).value == label)


# ------------------------------------------------------------------------------------------------------------------
# the route
# ------------------------------------------------------------------------------------------------------------------
def test_a_tolerance_changes_the_fractions_the_route_returns(api):
    client, ar = api
    wide = client.post("/api/analysis/texture", json={"tolerance_deg": 10}).json()
    narrow = client.post("/api/analysis/texture", json={"tolerance_deg": 2}).json()
    assert _fraction(wide, "Cube") == pytest.approx(3 / 6 * 100)
    assert _fraction(narrow, "Cube") == pytest.approx(1 / 6 * 100)
    assert (wide["texture_tolerance_deg"], narrow["texture_tolerance_deg"]) == (10.0, 2.0)
    assert narrow["texture_preset"] == "FCC_Rolling"


def test_a_preset_changes_the_components_the_route_returns(api):
    client, _ = api
    fcc = client.post("/api/analysis/texture", json={"preset": "FCC_Rolling"}).json()
    bcc = client.post("/api/analysis/texture", json={"preset": "BCC_Rolling"}).json()
    assert [c["name"] for c in fcc["components"]] == list(TEXTURE_PRESETS["FCC_Rolling"])
    assert [c["name"] for c in bcc["components"]] == list(TEXTURE_PRESETS["BCC_Rolling"])
    assert bcc["texture_preset"] == "BCC_Rolling"


def test_the_choices_are_remembered_returned_by_status_and_forgotten_on_a_file_switch(api):
    client, ar = api
    status = client.get("/api/analysis/status").json()
    assert (status["texture_preset"], status["texture_tolerance_deg"]) == ("FCC_Rolling", 10.0)
    client.post("/api/analysis/texture", json={"preset": "BCC_Rolling", "tolerance_deg": 4.5})
    status = client.get("/api/analysis/status").json()
    assert (status["texture_preset"], status["texture_tolerance_deg"]) == ("BCC_Rolling", 4.5)
    again = client.post("/api/analysis/texture").json()                      # fields left out: the last choice
    assert (again["texture_preset"], again["texture_tolerance_deg"]) == ("BCC_Rolling", 4.5)
    one = client.post("/api/analysis/texture", json={"tolerance_deg": 7}).json()      # one field changes, the other stays
    assert (one["texture_preset"], one["texture_tolerance_deg"]) == ("BCC_Rolling", 7.0)
    ar.restore_dataset_for_file("nothing-stashed-for-this-file")
    assert ar._texture_settings == ar.DEFAULT_TEXTURE_SETTINGS
    assert ar.DEFAULT_TEXTURE_SETTINGS == {"phase": "most_measured", "preset": "FCC_Rolling", "tolerance_deg": 10.0}


@pytest.mark.parametrize("body", [{"preset": "HCP_Rolling"}, {"preset": "fcc_rolling"}, {"preset": 3}, {"preset": ""},
                                  {"tolerance_deg": 0}, {"tolerance_deg": -1}, {"tolerance_deg": 30.5},
                                  {"tolerance_deg": "wide"}, {"tolerance_deg": True}])
def test_an_unknown_preset_or_a_tolerance_out_of_range_is_refused_and_changes_nothing(api, body):
    client, ar = api
    client.post("/api/analysis/texture", json={"preset": "BCC_Rolling", "tolerance_deg": 6})
    r = client.post("/api/analysis/texture", json=body)
    assert r.status_code == 422
    assert ar._texture_settings["preset"] == "BCC_Rolling" and ar._texture_settings["tolerance_deg"] == 6.0


def test_the_refusal_lists_the_allowed_values(api):
    client, _ = api
    text = str(client.post("/api/analysis/texture", json={"preset": "HCP_Rolling"}).json())
    assert "FCC_Rolling" in text and "BCC_Rolling" in text
    text = str(client.post("/api/analysis/texture", json={"tolerance_deg": 31}).json())
    assert "30" in text
    assert client.post("/api/analysis/texture", json={"tolerance_deg": 30}).status_code == 200      # the limit itself


def test_the_presets_of_the_request_are_the_presets_that_exist():
    from typing import get_args
    from backend.api.routes.analysis import TexturePreset
    assert set(get_args(TexturePreset)) == set(TEXTURE_PRESETS)


# ------------------------------------------------------------------------------------------------------------------
# the workbook, the split, the map
# ------------------------------------------------------------------------------------------------------------------
def test_the_workbook_cells_follow_the_tolerance_and_the_preset(api, tmp_path):
    client, ar = api
    cells = {}
    for tolerance in (10, 2):
        client.post("/api/analysis/texture", json={"preset": "FCC_Rolling", "tolerance_deg": tolerance})
        warnings = []
        results = ar._collect_export_results(warnings)
        assert "texture" in results, warnings
        ws = _sheet1(results, tmp_path)
        cube = ws.cell(row=_row_of(ws, "Cube"), column=2).value
        cells[tolerance] = (cube, ws.cell(row=68, column=2).value, ws.cell(row=69, column=2).value)
        assert (ws.cell(row=68, column=1).value, ws.cell(row=69, column=1).value) == ("texture_preset", "texture_tolerance_deg")
    assert cells[10][0] == pytest.approx(50.0) and cells[2][0] == pytest.approx(100 / 6)
    assert cells[10][1:] == ("FCC_Rolling", 10.0) and cells[2][1:] == ("FCC_Rolling", 2.0)
    client.post("/api/analysis/texture", json={"preset": "BCC_Rolling"})
    ws = _sheet1(ar._collect_export_results(), tmp_path)
    assert ws.cell(row=68, column=2).value == "BCC_Rolling"
    assert ws.cell(row=27, column=1).value == next(iter(TEXTURE_PRESETS["BCC_Rolling"]))


def test_the_export_split_uses_the_remembered_tolerance(api):
    client, ar = api
    cube_pixels = {}
    for tolerance in (10, 2):
        client.post("/api/analysis/texture", json={"tolerance_deg": tolerance})
        split = ar._collect_export_results()["texture_by_rx"]
        cube_pixels[tolerance] = round((split["rx"]["Cube"] or 0) * split["n_rx_pixels"] / 100
                                       + (split["substructure"]["Cube"] or 0) * split["n_substructure_pixels"] / 100)
    assert cube_pixels == {10: 48, 2: 16}


def _png(b64):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def test_the_component_map_uses_the_remembered_tolerance_and_preset(api):
    client, _ = api
    rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))               # noqa: E731
    client.post("/api/analysis/texture", json={"tolerance_deg": 10})
    wide = client.get("/api/analysis/map/texture_component").json()
    cube = rgb({e["name"]: e["color"] for e in wide["legend"]}["Cube"])
    assert tuple(_png(wide["image"])[1, 5]) == cube                              # 3 degrees off Cube, tolerance 10
    client.post("/api/analysis/texture", json={"tolerance_deg": 2})
    narrow = client.get("/api/analysis/map/texture_component").json()
    img = _png(narrow["image"])
    assert tuple(img[1, 1]) == cube and tuple(img[1, 5]) == rgb(narrow["no_component_color"])
    client.post("/api/analysis/texture", json={"preset": "BCC_Rolling"})
    bcc = client.get("/api/analysis/map/texture_component").json()
    assert [e["name"] for e in bcc["legend"]] == list(TEXTURE_PRESETS["BCC_Rolling"])
    # the answer names what the colours were counted with
    assert (wide["texture_preset"], wide["texture_tolerance_deg"]) == ("FCC_Rolling", 10.0)
    assert (narrow["texture_preset"], narrow["texture_tolerance_deg"]) == ("FCC_Rolling", 2.0)
    assert (bcc["texture_preset"], bcc["texture_tolerance_deg"]) == ("BCC_Rolling", 2.0)


# ------------------------------------------------------------------------------------------------------------------
# the batch
# ------------------------------------------------------------------------------------------------------------------
def test_the_batch_writes_the_given_preset_and_tolerance(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    from analysis.excel_exporter import ExcelExporter
    from tests.batch_load_stub import patch_batch_load
    xmap = _xmap()
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    out = {}
    for name, kw in (("default", {}), ("given", {"texture_preset": "BCC_Rolling", "tolerance_deg": 2.0})):
        proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), min_grain_pixels=1, grain_alpha=0.0, **kw)
        results = proc.process_file(tmp_path / "s.h5")
        ex = ExcelExporter(str(tmp_path / f"{name}.xlsx"))
        ex.export_dataset("ds", results)
        ws = ex.wb[ex.wb.sheetnames[0]]
        out[name] = (ws.cell(row=68, column=2).value, ws.cell(row=69, column=2).value, results["texture"].tolerance_angle)
        split = results["texture_by_rx"]
        out[name] += (split["n_rx_pixels"] + split["n_substructure_pixels"],)
    assert out["default"] == ("FCC_Rolling", 10.0, 10.0, 96)
    assert out["given"] == ("BCC_Rolling", 2.0, 2.0, 96)


def test_the_batch_split_counts_with_the_given_tolerance(tmp_path, monkeypatch):
    from analysis import batch_processor as bp
    from tests.batch_load_stub import patch_batch_load
    xmap = _xmap()
    patch_batch_load(monkeypatch, bp, lambda path: xmap)
    cube = {}
    for tolerance in (10.0, 2.0):
        proc = bp.BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), min_grain_pixels=1, grain_alpha=0.0,
                                 tolerance_deg=tolerance)
        split = proc.process_file(tmp_path / "s.h5")["texture_by_rx"]
        cube[tolerance] = round((split["rx"]["Cube"] or 0) * split["n_rx_pixels"] / 100
                                + (split["substructure"]["Cube"] or 0) * split["n_substructure_pixels"] / 100)
    assert cube == {10.0: 48, 2.0: 16}


@pytest.mark.parametrize("kw", [{"texture_preset": "HCP_Rolling"}, {"tolerance_deg": 0}, {"tolerance_deg": 31},
                                {"tolerance_deg": True}, {"tolerance_deg": "10"}])
def test_the_batch_refuses_an_unknown_preset_or_a_tolerance_out_of_range(tmp_path, kw):
    from analysis.batch_processor import BatchProcessor
    with pytest.raises(ValueError):
        BatchProcessor(str(tmp_path), str(tmp_path / "doc.xlsx"), **kw)


def test_the_batch_route_passes_preset_and_tolerance_only_when_given(api, monkeypatch):
    client, _ = api
    import analysis.batch_processor as bp
    seen = []
    monkeypatch.setattr(bp, "batch_process_folder", lambda folder, out, progress_callback=None, **kw:
                        seen.append(kw) or bp.BatchResult(processed=["a.h5"], n_files=1))
    for body in ({}, {"texture_preset": "BCC_Rolling", "texture_tolerance_deg": 4}):
        r = client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o", **body})
        assert r.status_code == 200
        client.get(f"/api/analysis/task/{r.json()['task_id']}")
    assert seen == [{}, {"texture_preset": "BCC_Rolling", "tolerance_deg": 4.0}]
    for bad in ({"texture_preset": "HCP_Rolling"}, {"texture_tolerance_deg": 0}, {"texture_tolerance_deg": 31}):
        assert client.post("/api/analysis/batch", json={"folder_path": "f", "output_path": "o", **bad}).status_code == 422
