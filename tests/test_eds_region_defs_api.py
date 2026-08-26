"""The region-definition preview endpoint.

The editor cannot be honest without it: while you are typing thresholds, a
window that claims three pixels and one that claims half the map look exactly
the same. These pin that the numbers come from the real map, on the same
smoothed composition the classification will use, so the preview and the run
cannot disagree.
"""
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

N_ROWS = N_COLS = 10


@pytest.fixture
def client(monkeypatch):
    """A 10x10 map: an Al matrix with a 3x3 silicon particle in the corner."""
    n = N_ROWS * N_COLS
    al = np.full(n, 92.0)
    si = np.full(n, 6.0)
    fe = np.full(n, 2.0)
    g = np.arange(n).reshape(N_ROWS, N_COLS)
    particle = ((g // N_COLS < 3) & (g % N_COLS < 3)).ravel()
    al[particle], si[particle], fe[particle] = 55.0, 44.0, 1.0

    monkeypatch.setattr(
        eds_routes, "_build_at_pct_maps_for_loaded_file",
        lambda: ({"Al": al, "Si": si, "Fe": fe}, N_ROWS, N_COLS, "fake.h5oina"),
    )
    return TestClient(app), int(particle.sum())


def _post(c, defs, **kw):
    return c.post("/api/eds/phase-map/region-defs/preview",
                  json={"region_defs": defs, **kw})


def test_it_counts_what_a_window_would_claim(client):
    c, n_particle = client
    r = _post(c, [{"name": "Si", "elements": [
        {"element": "Si", "min_at_pct": 20}]}], scale=0)
    assert r.status_code == 200
    d = r.json()["definitions"][0]
    assert d["n_pixels"] == n_particle
    assert d["reason"] is None


def test_it_reports_the_composition_of_what_it_claimed(client):
    c, _ = client
    r = _post(c, [{"name": "Si", "elements": [
        {"element": "Si", "min_at_pct": 20}]}], scale=0)
    mean = r.json()["definitions"][0]["mean_at_pct"]
    assert mean["Si"] == pytest.approx(44.0, abs=0.5)


def test_it_says_what_is_still_unaccounted_for(client):
    c, n_particle = client
    r = _post(c, [{"name": "Si", "elements": [
        {"element": "Si", "min_at_pct": 20}]}], scale=0)
    body = r.json()
    assert body["unclaimed_pixels"] == N_ROWS * N_COLS - n_particle
    assert body["total_pixels"] == N_ROWS * N_COLS


def test_an_empty_window_claims_nothing_and_says_why(client):
    c, _ = client
    r = _post(c, [{"name": "wip"}], scale=0)
    d = r.json()["definitions"][0]
    assert d["n_pixels"] == 0
    assert d["reason"]


def test_a_window_on_an_unmeasured_element_says_so(client):
    c, _ = client
    r = _post(c, [{"name": "Mg", "elements": [
        {"element": "Mg", "min_at_pct": 5}]}], scale=0)
    d = r.json()["definitions"][0]
    assert d["n_pixels"] == 0
    assert "not measured" in d["reason"]


def test_the_first_window_wins_and_the_second_is_told_what_it_lost(client):
    c, n_particle = client
    r = _post(c, [
        {"name": "first", "elements": [{"element": "Si", "min_at_pct": 20}]},
        {"name": "second", "elements": [{"element": "Si", "min_at_pct": 10}]},
    ], scale=0)
    a, b = r.json()["definitions"]
    assert a["n_pixels"] == n_particle
    assert b["n_pixels"] == 0
    assert b["overlap_pixels"] == n_particle


def test_no_definitions_is_an_empty_answer_not_an_error(client):
    c, _ = client
    r = _post(c, [], scale=0)
    assert r.status_code == 200
    assert r.json()["definitions"] == []


def test_the_preview_reports_the_smoothing_it_used(client):
    """The preview and the run have to be looking at the same numbers; the
    only way to check that from outside is for the preview to say which."""
    c, _ = client
    assert _post(c, [], scale=0).json()["scale"] == 0
    assert _post(c, []).json()["scale"] > 0, "default is the module default"
