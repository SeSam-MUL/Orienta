"""The overview and the virtual BSE image of a hexagonal scan on the display raster."""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import scan_geometry  # noqa: E402
from tests import hex_raster_oracle as oracle  # noqa: E402
from tests.apex_helpers import wipe_backend  # noqa: E402
from tests.test_scan_geometry_lifecycle import _hex_scan, _load, _square_scan  # noqa: E402
from tests.test_apex_measurement_identity import _m1, _m2, files  # noqa: E402,F401  (files: a fixture)


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    yield
    wipe_backend()


def _png(b64):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def _owner(g):
    d = g.to_dict()
    return oracle.brute_cell_index(d, oracle.spec(d, 4)), oracle.spec(d, 4)


def _grey(png):
    return png if png.ndim == 2 else png[..., 0]


def test_the_overview_on_the_raster_shows_every_cell_where_it_lies(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    g = scan_geometry.geometry_for("scan_hex")
    assert g.kind == "hex"
    index_grey = _grey(_png(client.get("/api/ebsd/overview", params={"mode": "mean"}).json()["image"]))
    body = client.get("/api/ebsd/overview", params={"mode": "mean", "display": "raster"}).json()
    owner, sp = _owner(g)
    img = _png(body["image"])
    assert img.shape == (sp["height"], sp["width"], 4)
    assert body["display"]["shape"] == [sp["height"], sp["width"]] and body["display"]["factor"] == 4
    assert body["shape"] == list(g.grid_shape)                     # the index grid, as consumers expect
    inside = owner >= 0
    np.testing.assert_array_equal(img[..., 3] == 255, inside)
    np.testing.assert_array_equal(img[..., 0][inside], index_grey.reshape(-1)[owner[inside]])


def test_without_the_parameter_the_overview_is_what_it_was(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    body = client.get("/api/ebsd/overview", params={"mode": "mean"}).json()
    assert "display" not in body and body["shape"] == [6, 5]
    assert _grey(_png(body["image"])).shape == (6, 5)


def test_asking_for_the_raster_does_not_change_the_cached_overview(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    first = client.get("/api/ebsd/overview", params={"mode": "mean"}).json()
    client.get("/api/ebsd/overview", params={"mode": "mean", "display": "raster"})
    again = client.get("/api/ebsd/overview", params={"mode": "mean"}).json()
    assert again == first and "display" not in again


def test_a_square_scan_ignores_the_parameter(client, tmp_path):
    _load(client, _square_scan(tmp_path))
    plain = client.get("/api/ebsd/overview", params={"mode": "mean"}).json()
    asked = client.get("/api/ebsd/overview", params={"mode": "mean", "display": "raster"}).json()
    assert asked == plain and "display" not in asked


def test_the_virtual_bse_image_on_the_raster_shows_every_cell_where_it_lies(client, tmp_path):
    _load(client, _hex_scan(tmp_path))
    g = scan_geometry.geometry_for("scan_hex")
    index_rgb = _png(client.get("/api/ebsd/virtual-bse").json()["image"])            # (6, 5, 3)
    body = client.get("/api/ebsd/virtual-bse", params={"display": "raster"}).json()
    owner, sp = _owner(g)
    img = _png(body["image"])
    assert img.shape == (sp["height"], sp["width"], 4) and body["display"]["factor"] == 4
    inside = owner >= 0
    np.testing.assert_array_equal(img[..., 3] == 255, inside)
    np.testing.assert_array_equal(img[..., :3][inside], index_rgb.reshape(-1, 3)[owner[inside]])
    assert "display" not in client.get("/api/ebsd/virtual-bse").json()


def test_the_virtual_bse_image_of_a_square_scan_ignores_the_parameter(client, tmp_path):
    _load(client, _square_scan(tmp_path))
    assert "display" not in client.get("/api/ebsd/virtual-bse", params={"display": "raster"}).json()


def test_a_load_during_the_overview_leaves_no_picture_of_the_old_measurement_behind(client, files, monkeypatch):  # noqa: F811
    """The overview of measurement 1 (3 x 5) is still being computed when measurement 2 (5 x 3, the same
    dataset name) is loaded. The running call answers for measurement 1, on its own raster; the picture
    must not be cached for measurement 2, whose next overview is drawn from its own data and geometry."""
    from backend.api.routes import ebsd_viewer
    _m1(client, files)
    g1 = scan_geometry.geometry_for("noguid")
    assert list(g1.grid_shape) == [3, 5]
    real_reduce = ebsd_viewer._reduce_nav
    loaded = []

    def reduce_while_another_measurement_is_loaded(data, mode):
        if not loaded:
            loaded.append(_m2(client, files))
        return real_reduce(data, mode)

    monkeypatch.setattr(ebsd_viewer, "_reduce_nav", reduce_while_another_measurement_is_loaded)
    params = {"mode": "mean", "display": "raster"}
    running = client.get("/api/ebsd/overview", params=params)
    assert loaded and running.status_code == 200, running.text
    body = running.json()
    owner1, sp1 = _owner(g1)
    assert body["shape"] == [3, 5] and body["display"]["shape"] == [sp1["height"], sp1["width"]]
    assert _png(body["image"]).shape == (sp1["height"], sp1["width"], 4)

    monkeypatch.setattr(ebsd_viewer, "_reduce_nav", real_reduce)
    g2 = scan_geometry.geometry_for("noguid")
    assert list(g2.grid_shape) == [5, 3]
    nxt = client.get("/api/ebsd/overview", params=params)
    assert nxt.status_code == 200, nxt.text
    body = nxt.json()
    owner2, sp2 = _owner(g2)
    assert body["shape"] == [5, 3] and body["display"]["shape"] == [sp2["height"], sp2["width"]]
    img = _png(body["image"])
    assert img.shape == (sp2["height"], sp2["width"], 4)
    np.testing.assert_array_equal(img[..., 3] == 255, owner2 >= 0)


@pytest.mark.parametrize("route", ["/api/ebsd/overview", "/api/ebsd/virtual-bse"])
def test_the_raster_is_that_of_the_measurement_that_was_loaded(client, files, route):  # noqa: F811
    """One pattern file, two measurements of its metadata file (3 x 5 and 5 x 3 hexagonal), both under the
    dataset name "noguid": after loading the second, the raster is the second's, never the first's."""
    params = {"display": "raster", **({"mode": "mean"} if route.endswith("overview") else {})}
    for load, grid in ((_m1, [3, 5]), (_m2, [5, 3])):
        name = load(client, files)["dataset_name"]
        g = scan_geometry.geometry_for(name)
        assert list(g.grid_shape) == grid
        body = client.get(route, params=params).json()
        owner, sp = _owner(g)
        img = _png(body["image"])
        assert img.shape == (sp["height"], sp["width"], 4)
        assert body["shape"] == grid and body["display"]["shape"] == [sp["height"], sp["width"]]
        np.testing.assert_array_equal(img[..., 3] == 255, owner >= 0)
