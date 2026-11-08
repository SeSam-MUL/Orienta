"""Region statistics with a mask and the line scan with points.

A selection that is not an index rectangle (a hexagonal scan, an ellipse, a lasso) reaches the backend as a
mask over its bounding rectangle, a line as a list of scan points. Everything is in index space (row, col).
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_phasemap_linescan import _make_fake_result as _line_fake
from tests.test_phasemap_region_stats import _make_fake_result as _region_fake


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


REGION = "/api/phasemap/region-stats"
LINE = "/api/phasemap/linescan"
# 4 rows x 3 columns, row-major. Phase ids by hand:
#   row 0: 0 0 1
#   row 1: 0 1 1
#   row 2: 1 1 0
#   row 3: 0 0 0
PG = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.int32)
# CI = 0.1 * (3 * row + col): 0.0 0.1 0.2 / 0.3 0.4 0.5 / 0.6 0.7 0.8 / 0.9 1.0 1.1
CI = np.arange(12, dtype=np.float32).reshape(4, 3) * 0.1


def _arm(monkeypatch, fake):
    monkeypatch.setattr("backend.api.routes.phase_map.get_last_indexing_result", lambda: fake)
    monkeypatch.setattr("backend.api.routes.phase_map.get_analysis_dataset", lambda: None)


def _rect(**extra):
    return {"row_start": 0, "row_end": 2, "col_start": 0, "col_end": 2, **extra}


DIAGONAL = [True, False, False, False, True, False, False, False, True]    # (0,0) (1,1) (2,2) of the 3 x 3 box


def test_a_mask_selects_the_pixels_the_statistics_are_taken_from(monkeypatch, client):
    _arm(monkeypatch, _region_fake(phase_grid=PG, ci_grid=CI))
    d = client.post(REGION, json=_rect(mask=DIAGONAL)).json()
    # Pixels (0,0) phase 0 ci 0.0, (1,1) phase 1 ci 0.4, (2,2) phase 0 ci 0.8.
    assert d["n_pixels"] == 3 and d["masked"] is True
    by_id = {p["id"]: p for p in d["phases"]}
    assert by_id[0]["count"] == 2 and by_id[1]["count"] == 1
    assert by_id[0]["fraction"] == pytest.approx(2 / 3)
    # mean = (0.0 + 0.4 + 0.8) / 3 = 0.4; std = sqrt((0.16 + 0 + 0.16) / 3) = sqrt(0.32 / 3)
    assert d["scalars"]["ci"]["mean"] == pytest.approx(0.4, abs=1e-6)
    assert d["scalars"]["ci"]["std"] == pytest.approx((0.32 / 3) ** 0.5, abs=1e-6)


def test_a_mask_of_all_true_changes_nothing_but_the_flag(monkeypatch, client):
    _arm(monkeypatch, _region_fake(phase_grid=PG, ci_grid=CI))
    plain = client.post(REGION, json=_rect()).json()
    full = client.post(REGION, json=_rect(mask=[True] * 9)).json()
    assert full.pop("masked") is True and "masked" not in plain
    assert full == plain


def test_a_mask_is_row_major_over_the_normalised_rectangle(monkeypatch, client):
    # A reversed drag (row_start > row_end) is normalised first; the mask follows the normalised box.
    # The box is 2 rows x 3 columns (rows 1-2, cols 0-2) and the mask is not equal to its transpose
    # reading, so only row-major order gives these pixels:
    #   mask [1,1,0, 0,0,1] row-major -> (1,0) (1,1) (2,2)
    #   (column-major would read (1,0) (2,0) (2,2) instead)
    # Phases: (1,0) = 0, (1,1) = 1, (2,2) = 0. CI: 0.3, 0.4, 0.8.
    _arm(monkeypatch, _region_fake(phase_grid=PG, ci_grid=CI))
    body = {"row_start": 2, "row_end": 1, "col_start": 2, "col_end": 0,
            "mask": [True, True, False, False, False, True]}
    d = client.post(REGION, json=body).json()
    assert d["n_pixels"] == 3
    assert {p["id"]: p["count"] for p in d["phases"]} == {0: 2, 1: 1}
    # mean = (0.3 + 0.4 + 0.8) / 3 = 0.5 (column-major: (0.3 + 0.6 + 0.8) / 3 = 0.5667)
    # std = sqrt((0.2^2 + 0.1^2 + 0.3^2) / 3) = sqrt(0.14 / 3)
    assert d["scalars"]["ci"]["mean"] == pytest.approx(0.5, abs=1e-6)
    assert d["scalars"]["ci"]["std"] == pytest.approx((0.14 / 3) ** 0.5, abs=1e-6)


@pytest.mark.parametrize("mask", [[True] * 8, [True] * 10, [False] * 9], ids=["short", "long", "empty"])
def test_a_mask_that_does_not_fit_or_selects_nothing_is_refused(monkeypatch, client, mask):
    _arm(monkeypatch, _region_fake(phase_grid=PG, ci_grid=CI))
    r = client.post(REGION, json=_rect(mask=mask))
    assert r.status_code == 422
    assert "mask" in r.json()["detail"]


def test_a_masked_rectangle_must_lie_inside_the_scan(monkeypatch, client):
    _arm(monkeypatch, _region_fake(phase_grid=PG, ci_grid=CI))       # 4 rows
    # Rows 2-5 are clipped to 2-3. The mask is sized for the CLIPPED rectangle (2 x 3), so only the
    # "must lie inside the scan" check can refuse it.
    body = {"row_start": 2, "row_end": 5, "col_start": 0, "col_end": 2, "mask": [True] * 6}
    r = client.post(REGION, json=body)
    assert r.status_code == 422 and "inside the scan" in r.json()["detail"]
    body.pop("mask")
    assert client.post(REGION, json=body).status_code == 200          # unmasked, it is clipped as before


def test_points_are_sampled_where_they_are(monkeypatch, client):
    _arm(monkeypatch, _line_fake(ci_grid=CI))
    d = client.post(LINE, json={"points": [[0, 0], [1, 1], [2, 2], [3, 2]], "layers": ["ci", "phase"]}).json()
    assert d["samples"] == 4
    # ci = 0.1 * (3 * row + col): 0.0, 0.4, 0.8, 1.1
    assert d["series"]["ci"] == pytest.approx([0.0, 0.4, 0.8, 1.1], abs=1e-6)
    assert d["series"]["phase"] == [None] * 4                         # not a scalar layer: as before


def test_a_point_is_row_then_column(monkeypatch, client):
    _arm(monkeypatch, _line_fake(ci_grid=CI))
    d = client.post(LINE, json={"points": [[3, 0], [0, 2]], "layers": ["ci"]}).json()
    assert d["series"]["ci"] == pytest.approx([0.9, 0.2], abs=1e-6)  # (3,0) = 0.9, (0,2) = 0.2


def test_points_are_not_interpolated(monkeypatch, client):
    _arm(monkeypatch, _line_fake(ci_grid=CI))
    d = client.post(LINE, json={"points": [[0, 0], [0, 0]], "layers": ["ci"]}).json()
    assert d["series"]["ci"] == pytest.approx([0.0, 0.0])


def test_points_ignore_n_samples(monkeypatch, client):
    _arm(monkeypatch, _line_fake(ci_grid=CI))
    d = client.post(LINE, json={"points": [[0, 0], [1, 0], [2, 0]], "n_samples": 50, "layers": ["ci"]}).json()
    assert d["samples"] == 3 and len(d["series"]["ci"]) == 3


@pytest.mark.parametrize(
    "points",
    [[[0, 0]], [[0, 0], [4, 0]], [[0, 0], [0, 3]], [[0, 0], [-1, 0]], [[0, 0], [0, -1]],
     [[0, 0, 1], [1, 1, 1]], [[0, 0], [1]], [[0, 0]] * 4097, []],
    ids=["one", "row", "col", "negative-row", "negative-col", "shape", "ragged", "too-many", "none"])
def test_points_that_do_not_fit_are_refused(monkeypatch, client, points):
    _arm(monkeypatch, _line_fake(ci_grid=CI))
    assert client.post(LINE, json={"points": points, "layers": ["ci"]}).status_code == 422


def test_a_point_beyond_64_bits_is_refused_not_a_server_error(monkeypatch, client):
    _arm(monkeypatch, _line_fake(ci_grid=CI))
    r = client.post(LINE, json={"points": [[0, 0], [10**30, 1]], "layers": ["ci"]})
    assert r.status_code == 422 and "outside the scan 4x3" in r.json()["detail"]


def test_the_name_of_the_refusal_says_what_is_wrong(monkeypatch, client):
    _arm(monkeypatch, _line_fake(ci_grid=CI))
    r = client.post(LINE, json={"points": [[0, 0], [4, 1]], "layers": ["ci"]})
    assert r.status_code == 422 and "outside the scan 4x3" in r.json()["detail"]


def test_a_line_needs_points_or_endpoints(monkeypatch, client):
    _arm(monkeypatch, _line_fake(ci_grid=CI))
    assert client.post(LINE, json={"layers": ["ci"]}).status_code == 422
    assert client.post(LINE, json={"start_row": 0, "start_col": 0, "layers": ["ci"]}).status_code == 422
    two = client.post(LINE, json={"start_row": 0, "start_col": 0, "end_row": 3, "end_col": 2,
                                  "n_samples": 2, "layers": ["ci"]})
    assert two.status_code == 200 and two.json()["samples"] == 2
    # ends of the line: (0,0) = 0.0 and (3,2) = 1.1
    assert two.json()["series"]["ci"] == pytest.approx([0.0, 1.1], abs=1e-6)
