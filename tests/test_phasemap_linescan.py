"""Tests for POST /api/phasemap/linescan — scalar profile along a line.

Returns per-layer arrays of length ``n_samples`` sampled along the line
``(start_row, start_col) → (end_row, end_col)``. Used by the React Profile
Plot. Disallowed layers (phase, ipf, unknown) return all-null arrays.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


def _make_fake_result(
    *,
    n_rows: int = 4,
    n_cols: int = 3,
    ci_grid: np.ndarray | None = None,
    bc_grid: np.ndarray | None = None,
):
    """Build a stub indexing result with optional scalar grids."""
    result = MagicMock()
    result.original_shape = (n_rows, n_cols)
    result.selection_mask = None
    result.metadata = {}

    phase_grid = np.zeros((n_rows, n_cols), dtype=np.int32)
    xmap = MagicMock()
    xmap.phase_id = phase_grid.ravel()
    xmap.shape = (n_rows, n_cols)
    fe = MagicMock(); fe.name = "Fe"
    xmap.phases = {0: fe}

    prop: dict = {}
    if bc_grid is not None:
        prop["bc"] = bc_grid.astype(np.float32).ravel()
    if ci_grid is not None:
        xmap.scores = ci_grid.astype(np.float32).ravel()
    xmap.prop = prop

    result.xmap = xmap
    return result


def test_linescan_returns_400_when_no_result(monkeypatch):
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: None,
    )
    r = client.post(
        "/api/phasemap/linescan",
        json={
            "start_row": 0, "start_col": 0,
            "end_row": 3, "end_col": 2,
            "n_samples": 16,
            "layers": ["ci"],
        },
    )
    assert r.status_code == 400


def test_linescan_samples_ci_along_line(monkeypatch):
    """A horizontal line through row 0 returns ci values that match the grid."""
    ci = np.arange(12, dtype=np.float32).reshape(4, 3) * 0.1
    fake = _make_fake_result(ci_grid=ci)
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    r = client.post(
        "/api/phasemap/linescan",
        json={
            "start_row": 0, "start_col": 0,
            "end_row": 0, "end_col": 2,
            "n_samples": 3,
            "layers": ["ci"],
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["samples"] == 3
    # Three samples at (0,0), (0,1), (0,2) → ci = 0.0, 0.1, 0.2.
    series = data["series"]["ci"]
    assert series[0] == pytest.approx(0.0, abs=1e-3)
    assert series[1] == pytest.approx(0.1, abs=1e-3)
    assert series[2] == pytest.approx(0.2, abs=1e-3)


def test_linescan_clamps_n_samples(monkeypatch):
    """n_samples=1 → clamped to 2; n_samples=10000 → clamped to 512."""
    ci = np.zeros((4, 3), dtype=np.float32)
    fake = _make_fake_result(ci_grid=ci)
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    r_low = client.post(
        "/api/phasemap/linescan",
        json={
            "start_row": 0, "start_col": 0,
            "end_row": 3, "end_col": 2,
            "n_samples": 1,
            "layers": ["ci"],
        },
    )
    r_high = client.post(
        "/api/phasemap/linescan",
        json={
            "start_row": 0, "start_col": 0,
            "end_row": 3, "end_col": 2,
            "n_samples": 10000,
            "layers": ["ci"],
        },
    )
    assert r_low.status_code == 200
    assert r_high.status_code == 200
    assert r_low.json()["samples"] == 2
    assert r_high.json()["samples"] == 512
    assert len(r_low.json()["series"]["ci"]) == 2
    assert len(r_high.json()["series"]["ci"]) == 512


def test_linescan_unknown_layer_returns_null_array(monkeypatch):
    """Unknown layer ids (e.g. 'foobar') return [None]*n."""
    ci = np.zeros((4, 3), dtype=np.float32)
    fake = _make_fake_result(ci_grid=ci)
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    r = client.post(
        "/api/phasemap/linescan",
        json={
            "start_row": 0, "start_col": 0,
            "end_row": 3, "end_col": 2,
            "n_samples": 8,
            "layers": ["foobar", "ci"],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["series"]["foobar"] == [None] * 8
    # ci must still come back populated.
    assert all(v is not None for v in data["series"]["ci"])


def test_linescan_phase_and_ipf_return_null_arrays(monkeypatch):
    """Layer ids 'phase' and 'ipf-x/y/z' are disallowed → [None]*n."""
    ci = np.zeros((4, 3), dtype=np.float32)
    fake = _make_fake_result(ci_grid=ci)
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    r = client.post(
        "/api/phasemap/linescan",
        json={
            "start_row": 0, "start_col": 0,
            "end_row": 3, "end_col": 2,
            "n_samples": 5,
            "layers": ["phase", "ipf-x", "ipf-y", "ipf-z"],
        },
    )
    assert r.status_code == 200
    series = r.json()["series"]
    for lid in ("phase", "ipf-x", "ipf-y", "ipf-z"):
        assert series[lid] == [None] * 5, f"{lid} must be null-array"


def test_linescan_missing_scalar_returns_null_array(monkeypatch):
    """When the underlying array doesn't exist, the layer returns [None]*n (no crash)."""
    # No scores attribute on xmap, no 'bc' in prop → ci/bc layers null.
    fake = _make_fake_result(ci_grid=None, bc_grid=None)
    del fake.xmap.scores
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    r = client.post(
        "/api/phasemap/linescan",
        json={
            "start_row": 0, "start_col": 0,
            "end_row": 3, "end_col": 2,
            "n_samples": 6,
            "layers": ["ci", "bc", "kam", "gos"],
        },
    )
    assert r.status_code == 200
    series = r.json()["series"]
    for lid in ("ci", "bc", "kam", "gos"):
        assert series[lid] == [None] * 6, f"{lid} must be null-array"


def test_linescan_empty_layers_returns_empty_series(monkeypatch):
    """When layers=[], series is empty + samples is still clamped."""
    fake = _make_fake_result()
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    r = client.post(
        "/api/phasemap/linescan",
        json={
            "start_row": 0, "start_col": 0,
            "end_row": 3, "end_col": 2,
            "n_samples": 32,
            "layers": [],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["samples"] == 32
    assert data["series"] == {}
