"""Seeding a composition window from one pixel, and showing where it lands.

Both exist because first-time readers walked the flow and stopped:

    "I click 'From selected region' on my lumped region. But that region IS
     the mistake - it's Si and AlFeMnSi averaged together. The seeded window
     brackets the lump and nothing separates. At that point I would conclude
     the feature doesn't work."

    "'Try it' gives me '202 px'. I cannot see WHERE those pixels are."

A threshold for a silicon particle has to be read off a silicon PIXEL. These
pin the properties that make that work: one-sided windows, only genuinely
concentrated elements, the smoothed basis the classifier will use, and an
overlay whose opaque pixels are exactly the pixels claimed.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api.main import app
from backend.api.routes import eds as eds_routes

N_ROWS = N_COLS = 16


@pytest.fixture
def client(monkeypatch):
    """An Al matrix with one 4x4 silicon particle in the top-left corner."""
    n = N_ROWS * N_COLS
    al = np.full(n, 92.0)
    si = np.full(n, 6.0)
    fe = np.full(n, 2.0)
    g = np.arange(n).reshape(N_ROWS, N_COLS)
    particle = ((g // N_COLS < 4) & (g % N_COLS < 4)).ravel()
    al[particle], si[particle], fe[particle] = 50.0, 49.0, 1.0

    monkeypatch.setattr(
        eds_routes, "_build_at_pct_maps_for_loaded_file",
        lambda: ({"Al": al, "Si": si, "Fe": fe}, N_ROWS, N_COLS, "fake.h5oina"),
    )
    return TestClient(app), particle


def _seed(c, row, col, **kw):
    return c.post("/api/eds/phase-map/region-defs/seed-from-pixel",
                  json={"row": row, "col": col, "scale": 0, **kw})


# --- seeding -----------------------------------------------------------------

def test_it_writes_a_window_from_the_pixel_you_pointed_at(client):
    c, _ = client
    r = _seed(c, 1, 1)
    assert r.status_code == 200
    d = r.json()["definition"]
    assert [e["element"] for e in d["elements"]] == ["Si"]


def test_the_window_is_one_sided(client):
    """An upper bound on the element that makes the feature distinctive
    would exclude its purest pixels - the core of the particle."""
    c, _ = client
    e = _seed(c, 1, 1).json()["definition"]["elements"][0]
    assert e["min_at_pct"] > 0
    assert e["max_at_pct"] is None


def test_only_concentrated_elements_become_clauses(client):
    """One clause per measured element writes six that say "and the matrix
    is still the matrix" - true, useless, and brittle."""
    c, _ = client
    els = [e["element"] for e in _seed(c, 1, 1).json()["definition"]["elements"]]
    assert "Al" not in els and "Fe" not in els


def test_a_matrix_pixel_yields_no_clauses_rather_than_a_wrong_one(client):
    c, _ = client
    body = _seed(c, 12, 12).json()
    assert body["definition"]["elements"] == []


def test_it_reports_everything_measured_there_not_only_what_it_used(client):
    """So the UI can say what it left out, instead of silently handing back
    one clause out of three."""
    c, _ = client
    comp = _seed(c, 1, 1).json()["composition"]
    assert set(comp) >= {"Al", "Si", "Fe"}


def test_a_looser_tolerance_reaches_further_down_the_dilution_gradient(client):
    c, _ = client
    tight = _seed(c, 1, 1, tolerance=0.2).json()["definition"]["elements"][0]
    loose = _seed(c, 1, 1, tolerance=0.6).json()["definition"]["elements"][0]
    assert loose["min_at_pct"] < tight["min_at_pct"]


def test_the_seeded_window_actually_claims_the_feature_it_came_from(client):
    """The property that matters: seed from a pixel, and the window finds
    the thing that pixel belongs to."""
    c, particle = client
    d = _seed(c, 1, 1).json()["definition"]
    r = c.post("/api/eds/phase-map/region-defs/preview",
               json={"region_defs": [d], "scale": 0})
    assert r.json()["definitions"][0]["n_pixels"] == int(particle.sum())


def test_a_pixel_outside_the_map_is_refused_loudly(client):
    c, _ = client
    assert _seed(c, 999, 0).status_code == 400


def test_the_name_says_what_makes_the_window_distinctive(client):
    c, _ = client
    assert "Si" in _seed(c, 1, 1).json()["definition"]["name"]


# --- the overlay -------------------------------------------------------------

def _overlay(body):
    return np.array(Image.open(io.BytesIO(base64.b64decode(body["overlay"]))))


from PIL import Image  # noqa: E402  (imported here to keep the helper adjacent)


def test_the_preview_shows_where_not_only_how_many(client):
    c, _ = client
    r = c.post("/api/eds/phase-map/region-defs/preview",
               json={"region_defs": [{"name": "Si", "elements": [
                   {"element": "Si", "min_at_pct": 20}]}], "scale": 0})
    assert r.json()["overlay"]


def test_the_opaque_pixels_are_exactly_the_claimed_pixels(client):
    """A picture that disagrees with the count is worse than no picture."""
    c, particle = client
    body = c.post("/api/eds/phase-map/region-defs/preview",
                  json={"region_defs": [{"name": "Si", "elements": [
                      {"element": "Si", "min_at_pct": 20}]}],
                        "scale": 0}).json()
    a = _overlay(body)
    assert a.shape == (N_ROWS, N_COLS, 4)
    claimed = (a[..., 3] > 0).ravel()
    assert np.array_equal(claimed, particle)
    assert int(claimed.sum()) == body["definitions"][0]["n_pixels"]


def test_unclaimed_pixels_are_transparent_so_the_map_shows_through(client):
    c, _ = client
    body = c.post("/api/eds/phase-map/region-defs/preview",
                  json={"region_defs": [{"name": "Si", "elements": [
                      {"element": "Si", "min_at_pct": 20}]}],
                        "scale": 0}).json()
    a = _overlay(body)
    assert (a[..., 3] == 0).any()


def test_two_definitions_get_two_colours(client):
    c, _ = client
    body = c.post("/api/eds/phase-map/region-defs/preview",
                  json={"region_defs": [
                      {"name": "a", "elements": [
                          {"element": "Si", "min_at_pct": 20}]},
                      {"name": "b", "elements": [
                          {"element": "Fe", "min_at_pct": 1.5}]}],
                        "scale": 0}).json()
    a = _overlay(body)
    seen = {tuple(px) for px in a[a[..., 3] > 0][:, :3]}
    assert len(seen) == 2


def test_no_definitions_means_no_overlay_rather_than_a_blank_one(client):
    c, _ = client
    body = c.post("/api/eds/phase-map/region-defs/preview",
                  json={"region_defs": [], "scale": 0}).json()
    a = _overlay(body)
    assert (a[..., 3] == 0).all(), "nothing claimed, nothing painted"
