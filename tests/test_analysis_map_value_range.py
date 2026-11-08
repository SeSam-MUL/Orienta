"""The scalar analysis maps say what their colours mean.

Every scalar map (band contrast, KAM, GOS, ECD) is drawn with a colormap scaled to the data's own minimum and maximum,
so a few percent of change in some grains looks like no change. The route therefore answers ``value_range``
(``min``, ``max``, ``unit``: the numbers the image was scaled with) and ``colormap_stops`` (five hex colours of the
colormap at 0, .25, .5, .75, 1), and the viewer draws a colour bar from them. Maps with fixed colours do not.
"""
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from analysis.ebsd_dataset import EBSDDataset
from tests.synthetic_grain_map import make_map, to_xmap


@pytest.fixture
def client_map():
    import backend.api.routes.analysis as ar
    from backend.api.main import app
    m = make_map(shape=(60, 80), n_seeds=40)
    m["phase_id"][5:7, 20:24] = -1                      # a small hole, closed at alpha 3.1: cell area != pixels
    xmap = to_xmap(m)
    xmap.prop["bc"] = np.random.default_rng(1).uniform(20, 200, xmap.size)
    saved = ar._dataset, dict(ar._grain_stats_settings)
    ar._dataset = EBSDDataset(xmap, step_size=m["step"])
    ar._reset_grain_stats_settings()
    try:
        yield TestClient(app), ar
    finally:
        ar._dataset = saved[0]
        ar._grain_stats_settings.clear()
        ar._grain_stats_settings.update(saved[1])


def _reconstruct(client, **extra):
    body = {"misorientation_threshold": 5.0, "min_grain_size": 1, "alpha": 3.1, "fill": False, **extra}
    r = client.post("/api/analysis/grains/reconstruct", json=body)
    assert r.status_code == 200, r.text
    task = client.get(f"/api/analysis/task/{r.json()['task_id']}").json()
    assert task["status"] == "completed", task


UNITS = {"bc": "", "kam": "°", "gos": "°", "ecd": "µm"}


@pytest.fixture
def spy(monkeypatch):
    """The array the route scaled the image with."""
    import backend.api.routes.analysis as mod
    seen = {}
    real = mod.colormap_array_to_base64

    def spy_(data, cmap):
        seen["data"] = np.array(data, dtype=float)
        return real(data, cmap)

    monkeypatch.setattr(mod, "colormap_array_to_base64", spy_)
    return seen


@pytest.mark.parametrize("kind", list(UNITS))
def test_the_range_is_the_min_and_max_the_image_was_scaled_with(client_map, spy, kind):
    client, _ = client_map
    _reconstruct(client)
    r = client.get(f"/api/analysis/map/{kind}", params={"cmap": "viridis"})
    assert r.status_code == 200, r.text
    body = r.json()
    data = spy["data"]
    assert body["value_range"] == {"min": float(np.nanmin(data)), "max": float(np.nanmax(data)), "unit": UNITS[kind]}
    assert body["value_range"]["max"] > body["value_range"]["min"]
    assert (body["min_val"], body["max_val"]) == (body["value_range"]["min"], body["value_range"]["max"])


@pytest.mark.parametrize("cmap", ["viridis", "gray", "RdYlBu_r"])
def test_the_five_stops_are_the_colormap_at_the_quarter_points(client_map, cmap):
    import matplotlib
    client, _ = client_map
    _reconstruct(client)
    stops = client.get("/api/analysis/map/kam", params={"cmap": cmap}).json()["colormap_stops"]
    assert len(stops) == 5 and all(re.fullmatch(r"#[0-9a-f]{6}", s) for s in stops)
    colormap = matplotlib.colormaps[cmap]
    for x, hex_colour in zip((0.0, 0.25, 0.5, 0.75, 1.0), stops):
        want = "#" + "".join(f"{int(round(c * 255)):02x}" for c in colormap(x)[:3])
        assert hex_colour == want
    assert len(set(stops)) > 1


def test_the_stops_follow_the_colormap_that_was_asked_for(client_map):
    client, _ = client_map
    _reconstruct(client)
    a = client.get("/api/analysis/map/gos", params={"cmap": "gray"}).json()["colormap_stops"]
    b = client.get("/api/analysis/map/gos", params={"cmap": "viridis"}).json()["colormap_stops"]
    assert a[0] == "#000000" and a[-1] == "#ffffff" and a != b


@pytest.mark.parametrize("kind", ["ipf", "ipf_x", "ipf_y", "rx", "texture_component", "grain_boundaries"])
def test_maps_with_fixed_colours_have_no_range(client_map, kind):
    client, _ = client_map
    _reconstruct(client)
    r = client.get(f"/api/analysis/map/{kind}")
    if r.status_code == 400:                              # rx needs band contrast of its own; this map has it
        pytest.skip(r.json()["detail"])
    assert r.status_code == 200, r.text
    assert "value_range" not in r.json() and "colormap_stops" not in r.json()


def test_switching_the_ecd_basis_changes_the_range_and_names_the_basis(client_map):
    """A map with a closed hole: its grain has more cell area than measured pixels. The basis the map is drawn on
    is the remembered one, so the range moves when /grain-size is called with the other basis."""
    client, ar = client_map
    _reconstruct(client)
    grains = ar._dataset.grains
    assert (np.asarray(grains.n_closed) > 0).any()
    size = np.asarray(grains.grain_size, dtype=float).copy()
    largest = int(np.argmax(grains.ecd))
    size[largest] = max(1.0, size[largest] * 0.3)           # the largest grain: most of its cell area is closed
    grains.grain_size = size

    cell = client.get("/api/analysis/map/ecd").json()
    assert cell["ecd_basis"] == "cell_area"
    assert client.post("/api/analysis/grain-size", json={"ecd_basis": "measured_pixels"}).status_code == 200
    pixels = client.get("/api/analysis/map/ecd").json()
    assert pixels["ecd_basis"] == "measured_pixels"
    assert pixels["value_range"]["max"] < cell["value_range"]["max"]
    assert cell["value_range"]["unit"] == "µm"
    assert "ecd_basis" not in client.get("/api/analysis/map/kam").json()
