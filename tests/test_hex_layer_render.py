"""GET /api/phasemap/layer on a hexagonal result: drawn on the display raster, nothing else changes."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.routes import phase_map  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from tests import hex_raster_oracle as oracle  # noqa: E402
from tests.test_phase_map_layer_endpoint import _make_fake_result  # noqa: E402

N_ROWS, N_COLS, STEP = 6, 5, 0.5


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


def _arm(monkeypatch, result=None, dataset=None):
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: dataset)


def _result(stamp, n_rows=N_ROWS, n_cols=N_COLS):
    fake = _make_fake_result(n_rows=n_rows, n_cols=n_cols)
    fake.metadata = {"scan_geometry": stamp} if stamp is not None else {}
    return fake


def _png(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def _expected(index_rgba, g_dict):
    """The index-space layer redrawn on the raster by the ORACLE's cell index (a plain argmin)."""
    sp = oracle.spec(g_dict, 4)
    owner = oracle.brute_cell_index(g_dict, sp)
    flat = index_rgba.reshape(-1, 4)
    return np.where((owner >= 0)[..., None], flat[np.maximum(owner, 0)], 0).astype(np.uint8), sp


@pytest.mark.parametrize("kind", ["phase", "bc"])
def test_a_hexagonal_result_layer_is_drawn_on_the_raster(client, monkeypatch, kind):
    g_dict = oracle.geometry_dict(N_ROWS, N_COLS, STEP, 0.5)
    fake = _result(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict())
    _arm(monkeypatch, fake)
    index_rgba = phase_map._compute_layer_rgba(kind=kind)           # what the layer was: (6, 5, 4)
    r = client.get("/api/phasemap/layer", params={"kind": kind})
    assert r.status_code == 200, r.text
    img = _png(r.json()["image"])
    want, sp = _expected(index_rgba, g_dict)
    assert img.shape == (sp["height"], sp["width"], 4)
    np.testing.assert_array_equal(img, want)
    assert int((img[..., 3] == 0).sum()) > 0                         # the corners no hexagon covers


def test_the_response_says_what_it_drew(client, monkeypatch):
    _arm(monkeypatch, _result(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()))
    body = client.get("/api/phasemap/layer", params={"kind": "phase"}).json()
    sp = oracle.spec(oracle.geometry_dict(N_ROWS, N_COLS, STEP, 0.5), 4)
    assert body["shape"] == [sp["height"], sp["width"]]
    assert body["display"]["kind"] == "hex_raster" and body["display"]["factor"] == 4
    assert body["display"]["shape"] == body["shape"] and body["display"]["step_um"] == sp["step_um"]


@pytest.mark.parametrize("stamp", [None, ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict()],
                         ids=["unknown", "square"])
def test_a_square_or_unknown_result_is_drawn_as_before(client, monkeypatch, stamp):
    _arm(monkeypatch, _result(stamp))
    index_rgba = phase_map._compute_layer_rgba(kind="phase")
    body = client.get("/api/phasemap/layer", params={"kind": "phase"}).json()
    assert body["shape"] == [N_ROWS, N_COLS] and "display" not in body
    np.testing.assert_array_equal(_png(body["image"]), index_rgba)


def test_a_result_on_a_crop_that_starts_on_an_odd_row_is_drawn_with_the_reversed_offset(client, monkeypatch):
    crop = ScanGeometry.hex(6, 5, 1.0).cropped(1, 0, 3, 5)           # row 0 of the crop is a shifted row
    assert crop.odd_row_shift == -0.5
    _arm(monkeypatch, _result(crop.to_dict(), n_rows=3, n_cols=5))
    index_rgba = phase_map._compute_layer_rgba(kind="phase")
    body = client.get("/api/phasemap/layer", params={"kind": "phase"}).json()
    want, sp = _expected(index_rgba, oracle.geometry_dict(3, 5, 1.0, -0.5))
    assert body["display"]["x0"] == sp["x0"] == -1.0
    np.testing.assert_array_equal(_png(body["image"]), want)


def test_the_analysis_dataset_is_drawn_on_the_raster_when_there_is_no_result(client, monkeypatch):
    fake = _result(None)
    dataset = SimpleNamespace(xmap=fake.xmap, shape=(N_ROWS, N_COLS),
                              scan_geometry=ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict(),
                              per_phase_data={})
    _arm(monkeypatch, None, dataset)
    index_rgba = phase_map._compute_layer_rgba(kind="phase")
    body = client.get("/api/phasemap/layer", params={"kind": "phase"}).json()
    want, _ = _expected(index_rgba, oracle.geometry_dict(N_ROWS, N_COLS, STEP, 0.5))
    np.testing.assert_array_equal(_png(body["image"]), want)


def test_display_index_asks_for_the_index_grid(client, monkeypatch):
    _arm(monkeypatch, _result(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()))
    index_rgba = phase_map._compute_layer_rgba(kind="phase")
    body = client.get("/api/phasemap/layer", params={"kind": "phase", "display": "index"}).json()
    assert body["shape"] == [N_ROWS, N_COLS] and "display" not in body
    np.testing.assert_array_equal(_png(body["image"]), index_rgba)


def test_display_raster_is_the_default_and_any_other_value_is_refused(client, monkeypatch):
    _arm(monkeypatch, _result(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()))
    default = client.get("/api/phasemap/layer", params={"kind": "phase"})
    named = client.get("/api/phasemap/layer", params={"kind": "phase", "display": "raster"})
    assert default.status_code == 200 and named.status_code == 200
    assert named.json() == default.json()
    for bad in ("INDEX", "Raster", "grid", ""):
        r = client.get("/api/phasemap/layer", params={"kind": "phase", "display": bad})
        assert r.status_code == 422, (bad, r.status_code)


def test_a_layer_whose_shape_is_neither_the_index_grid_nor_the_raster_is_refused(client, monkeypatch):
    _arm(monkeypatch, _result(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()))
    monkeypatch.setattr(phase_map, "_compute_layer_rgba", lambda **kw: np.zeros((7, 7, 4), dtype=np.uint8))
    r = client.get("/api/phasemap/layer", params={"kind": "phase"})
    assert r.status_code == 500
    assert "neither" in r.json()["detail"] and "(7, 7)" in r.json()["detail"]


def test_a_kind_drawn_on_the_raster_already_passes_through(client, monkeypatch):
    g = ScanGeometry.hex(N_ROWS, N_COLS, STEP)
    sp = oracle.spec(oracle.geometry_dict(N_ROWS, N_COLS, STEP, 0.5), 4)
    _arm(monkeypatch, _result(g.to_dict()))
    mark = np.zeros((sp["height"], sp["width"], 4), dtype=np.uint8)
    mark[3, 4] = (1, 2, 3, 255)
    monkeypatch.setattr(phase_map, "_compute_layer_rgba", lambda **kw: mark)
    body = client.get("/api/phasemap/layer", params={"kind": "grain-boundaries"}).json()
    np.testing.assert_array_equal(_png(body["image"]), mark)
    # ... but an ordinary kind of that shape is NOT passed through: it must be the index grid
    r = client.get("/api/phasemap/layer", params={"kind": "phase"})
    assert r.status_code == 500


def _dataset(stamp, fake):
    return SimpleNamespace(xmap=fake.xmap, shape=(N_ROWS, N_COLS), scan_geometry=stamp, per_phase_data={})


def test_a_result_with_a_crystal_map_is_drawn_with_its_own_stamp_whatever_dataset_is_loaded(monkeypatch):
    # the route draws the result (it has an xmap); a hexagonal dataset next to a square result must not decide
    square = ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict()
    fake = _result(square)
    _arm(monkeypatch, fake, _dataset(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict(), fake))
    assert phase_map._hex_layer_context("phase") is None


@pytest.mark.parametrize("kind", ["forward_ncc", "local_anomaly", "pc_sensitivity", "pattern_residual",
                                  "refined_ncc", "phase_margin", "assignment_source"])
def test_a_kind_that_reads_the_result_alone_takes_the_geometry_of_the_result(monkeypatch, kind):
    # A result without a crystal map next to a hexagonal analysis dataset: these layers are the result's, so the
    # dataset's geometry must not decide how they are drawn. A crystal-map kind of the same pair is the dataset's.
    result = SimpleNamespace(xmap=None, metadata={"scan_geometry": ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict()})
    dataset = _dataset(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict(), _result(None))
    _arm(monkeypatch, result, dataset)
    assert phase_map._hex_layer_context(kind) is None
    assert phase_map._hex_layer_context("phase") is not None


def test_the_route_draws_a_result_only_kind_with_the_stamp_of_the_result(client, monkeypatch):
    # The route, not only the helper: a phase-margin layer of a square result, with a hexagonal analysis
    # dataset loaded next to a result that holds no crystal map, stays on the index grid.
    n = N_ROWS * N_COLS
    result = SimpleNamespace(xmap=None, original_shape=(N_ROWS, N_COLS), selection_mask=None,
                             metadata={"scan_geometry": ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict(),
                                       "phase_check": {"margin_map": np.linspace(-0.2, 0.2, n)}})
    _arm(monkeypatch, result, _dataset(ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict(), _result(None)))
    r = client.get("/api/phasemap/layer", params={"kind": "phase_margin"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shape"] == [N_ROWS, N_COLS] and "display" not in body


def _arm_modules(monkeypatch, result, dataset):
    """get_available_maps imports the two getters from their modules at call time."""
    monkeypatch.setattr("backend.api.routes.indexing.get_last_indexing_result", lambda: result)
    monkeypatch.setattr("backend.api.routes.analysis.get_analysis_dataset", lambda: dataset)


def test_available_maps_names_the_geometry_of_the_dataset_the_routes_draw(client, monkeypatch):
    hexa = ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()
    fake = _result(None)
    _arm_modules(monkeypatch, None, _dataset(hexa, fake))                       # no result: the dataset is drawn
    assert client.get("/api/phasemap/available-maps").json()["dataset_scan_geometry"] == hexa
    _arm_modules(monkeypatch, _result(hexa), _dataset(hexa, fake))              # a result with a crystal map wins
    assert client.get("/api/phasemap/available-maps").json()["dataset_scan_geometry"] is None
    square = ScanGeometry.square(N_ROWS, N_COLS, STEP).to_dict()
    _arm_modules(monkeypatch, None, _dataset(square, fake))                     # a square dataset: nothing to say
    assert client.get("/api/phasemap/available-maps").json()["dataset_scan_geometry"] is None
    _arm_modules(monkeypatch, None, _dataset(None, fake))                       # an unknown one is not hexagonal
    assert client.get("/api/phasemap/available-maps").json()["dataset_scan_geometry"] is None


def test_available_maps_names_no_geometry_for_a_dataset_without_a_crystal_map(client, monkeypatch):
    # the map routes answer 400 for such a dataset, so there is no raster to name
    hexa = ScanGeometry.hex(N_ROWS, N_COLS, STEP).to_dict()
    bare = SimpleNamespace(xmap=None, shape=(N_ROWS, N_COLS), scan_geometry=hexa, per_phase_data={})
    _arm_modules(monkeypatch, None, bare)
    r = client.get("/api/phasemap/available-maps")
    assert r.status_code == 200
    assert r.json()["dataset_scan_geometry"] is None
