"""Tests for POST /api/phasemap/region-stats — rectangular ROI summary.

Returns the phase histogram + scalar mean/std for the rectangle. Used by
the React Region-Stats tooltip after a rectangle drag.

Hermetic tests build a tiny 4x3 xmap and assert on phase counts + scalar
statistics over a known sub-rectangle.
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
    phase_grid: np.ndarray | None = None,
    ci_grid: np.ndarray | None = None,
    bc_grid: np.ndarray | None = None,
):
    """Build a stub indexing result whose xmap exposes the requested grids."""
    result = MagicMock()
    result.original_shape = (n_rows, n_cols)
    result.selection_mask = None
    result.metadata = {}

    if phase_grid is None:
        phase_grid = np.zeros((n_rows, n_cols), dtype=np.int32)
    assert phase_grid.shape == (n_rows, n_cols)

    xmap = MagicMock()
    xmap.phase_id = phase_grid.ravel()
    xmap.shape = (n_rows, n_cols)
    fe = MagicMock(); fe.name = "Fe"
    al = MagicMock(); al.name = "Al"
    xmap.phases = {0: fe, 1: al}

    prop: dict = {}
    if bc_grid is not None:
        prop["bc"] = bc_grid.astype(np.float32).ravel()
    if ci_grid is not None:
        xmap.scores = ci_grid.astype(np.float32).ravel()
    xmap.prop = prop

    result.xmap = xmap
    return result


def test_region_stats_returns_400_when_no_result(monkeypatch):
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: None,
    )
    r = client.post(
        "/api/phasemap/region-stats",
        json={"row_start": 0, "row_end": 1, "col_start": 0, "col_end": 1},
    )
    assert r.status_code == 400


def test_region_stats_phase_histogram_correct(monkeypatch):
    """Counts + fractions match the ground-truth grid."""
    # 4x3 grid: top half Fe (phase 0), bottom half Al (phase 1).
    pg = np.zeros((4, 3), dtype=np.int32)
    pg[2:, :] = 1
    fake = _make_fake_result(phase_grid=pg)
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    # Rectangle = full grid → 6 Fe + 6 Al.
    r = client.post(
        "/api/phasemap/region-stats",
        json={"row_start": 0, "row_end": 3, "col_start": 0, "col_end": 2},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["n_pixels"] == 12
    phases_by_id = {p["id"]: p for p in data["phases"]}
    assert phases_by_id[0]["count"] == 6
    assert phases_by_id[1]["count"] == 6
    assert phases_by_id[0]["fraction"] == pytest.approx(0.5)
    assert phases_by_id[1]["fraction"] == pytest.approx(0.5)
    assert phases_by_id[0]["name"] == "Fe"
    assert phases_by_id[1]["name"] == "Al"


def test_region_stats_scalar_mean_std(monkeypatch):
    """ci mean/std must match numpy ground truth on a sub-rectangle."""
    pg = np.zeros((4, 3), dtype=np.int32)
    ci = np.arange(12, dtype=np.float32).reshape(4, 3) * 0.1   # 0.0..1.1
    fake = _make_fake_result(phase_grid=pg, ci_grid=ci)
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    # Sub-rectangle rows 1..2, cols 1..2 → ci values {.4 .5 .7 .8}.
    r = client.post(
        "/api/phasemap/region-stats",
        json={"row_start": 1, "row_end": 2, "col_start": 1, "col_end": 2},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    expected = ci[1:3, 1:3].ravel()
    assert data["scalars"]["ci"]["mean"] == pytest.approx(float(expected.mean()), abs=1e-6)
    assert data["scalars"]["ci"]["std"] == pytest.approx(float(expected.std()), abs=1e-6)


def test_region_stats_scalar_null_when_all_nan(monkeypatch):
    """A scalar that's all-NaN over the rectangle returns null."""
    pg = np.zeros((4, 3), dtype=np.int32)
    ci = np.full((4, 3), np.nan, dtype=np.float32)
    fake = _make_fake_result(phase_grid=pg, ci_grid=ci)
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    r = client.post(
        "/api/phasemap/region-stats",
        json={"row_start": 0, "row_end": 3, "col_start": 0, "col_end": 2},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["scalars"]["ci"] is None
    # kam / gos / uncertainty also null (no source).
    assert data["scalars"]["kam"] is None
    assert data["scalars"]["gos"] is None
    assert data["scalars"]["uncertainty"] is None


def test_region_stats_reversed_drag_normalised(monkeypatch):
    """row_end < row_start (drag from bottom-right corner) is normalised."""
    pg = np.zeros((4, 3), dtype=np.int32)
    pg[2:, :] = 1
    fake = _make_fake_result(phase_grid=pg)
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    # Reversed corners — must equal the same answer as the canonical rect.
    r_rev = client.post(
        "/api/phasemap/region-stats",
        json={"row_start": 3, "row_end": 0, "col_start": 2, "col_end": 0},
    )
    r_fwd = client.post(
        "/api/phasemap/region-stats",
        json={"row_start": 0, "row_end": 3, "col_start": 0, "col_end": 2},
    )
    assert r_rev.status_code == 200
    assert r_fwd.status_code == 200
    rev = r_rev.json(); fwd = r_fwd.json()
    assert rev["n_pixels"] == fwd["n_pixels"] == 12
    assert rev["row_start"] == 0 and rev["row_end"] == 3
    assert rev["col_start"] == 0 and rev["col_end"] == 2
    # Phase histogram identical.
    counts_rev = {p["id"]: p["count"] for p in rev["phases"]}
    counts_fwd = {p["id"]: p["count"] for p in fwd["phases"]}
    assert counts_rev == counts_fwd


def test_region_stats_unindexed_appears_as_negative_one(monkeypatch):
    """Pixels with phase_id=-1 show up as id=-1 / name='unindexed'."""
    pg = np.zeros((4, 3), dtype=np.int32)
    pg[0, :] = -1   # whole top row unindexed
    fake = _make_fake_result(phase_grid=pg)
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_last_indexing_result",
        lambda: fake,
    )
    monkeypatch.setattr(
        "backend.api.routes.phase_map.get_analysis_dataset",
        lambda: None,
    )
    r = client.post(
        "/api/phasemap/region-stats",
        json={"row_start": 0, "row_end": 3, "col_start": 0, "col_end": 2},
    )
    assert r.status_code == 200
    data = r.json()
    by_id = {p["id"]: p for p in data["phases"]}
    assert -1 in by_id
    assert by_id[-1]["count"] == 3
    assert by_id[-1]["name"] == "unindexed"
