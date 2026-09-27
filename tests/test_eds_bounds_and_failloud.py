"""
Regression tests: EDS fail-loud + bounds-check fixes in backend/api/routes/eds.py

Tests:
1. quantify-pixel with out-of-bounds row -> 400
2. suggest-phases with out-of-bounds pixel -> 400 (not 500, HTTPException passthrough)
3. quantify-pixel with valid pixel -> 200
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path so backend.* and eds_utils import cleanly.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Fake extractor
# ---------------------------------------------------------------------------

GRID_ROWS = 5
GRID_COLS = 4
N_PIXELS = GRID_ROWS * GRID_COLS  # 20

ELEMENTS = ["Al Ka1", "Fe Ka1"]


class _FakeExtractor:
    """Minimal extractor that simulates a 5x4 scan with 2 elements."""

    def get_available_elements(self):
        return list(ELEMENTS)

    def get_grid_dimensions(self):
        return (GRID_ROWS, GRID_COLS)

    def get_element_map(self, name):
        # Return a flat array of length N_PIXELS with deterministic values.
        rng = np.arange(N_PIXELS, dtype=np.float32) + 1.0
        return rng

    def get_element_map_2d(self, name):
        return self.get_element_map(name).reshape(GRID_ROWS, GRID_COLS)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    """TestClient with a fake open file session patched into the eds router."""
    # Patch h5_session helpers as seen by the eds route module.
    import backend.api.routes.eds as eds_mod

    fake_ext = _FakeExtractor()
    monkeypatch.setattr(eds_mod, "is_open", lambda: True)
    monkeypatch.setattr(eds_mod, "get_extractor", lambda: fake_ext)

    # Also patch the phase-suggestion CIF loader so it returns an empty
    # library (forces fallback to DEFAULT_PHASE_LIBRARY from eds_utils).
    # BOTH names: /suggest-phases moved to ..._with_skips on 2026-09-13
    # (6e1d5d73), so patching only the old one left this reading the real
    # Database/CIF_Library — the bounds checks below still passed, but the
    # empty-library fallback this fixture exists to force was never taken.
    monkeypatch.setattr(eds_mod, "load_cif_phase_library", lambda *a, **kw: {})
    monkeypatch.setattr(eds_mod, "load_cif_phase_library_with_skips",
                        lambda *a, **kw: ({}, []))

    from fastapi.testclient import TestClient
    from backend.api.main import app

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQuantifyPixelBoundsCheck:
    """POST /api/eds/quantify/pixel — row/col validation."""

    def test_out_of_bounds_row_returns_400(self, client):
        """row=99 is outside a 5x4 scan — must return 400, not 200 or 500."""
        resp = client.post(
            "/api/eds/quantify/pixel",
            json={"row": 99, "col": 0, "display_mode": "counts"},
        )
        assert resp.status_code == 400, (
            f"Expected 400 for out-of-bounds pixel, got {resp.status_code}: {resp.text}"
        )
        detail = resp.json().get("detail", "")
        assert "outside" in detail.lower() or "bounds" in detail.lower(), (
            f"Expected 'outside'/'bounds' in error detail, got: {detail!r}"
        )

    def test_out_of_bounds_col_returns_400(self, client):
        """col=99 is outside a 5x4 scan — must return 400."""
        resp = client.post(
            "/api/eds/quantify/pixel",
            json={"row": 0, "col": 99, "display_mode": "counts"},
        )
        assert resp.status_code == 400, (
            f"Expected 400 for out-of-bounds col, got {resp.status_code}: {resp.text}"
        )

    def test_negative_row_returns_400(self, client):
        """Negative row is also out of bounds."""
        resp = client.post(
            "/api/eds/quantify/pixel",
            json={"row": -1, "col": 0, "display_mode": "counts"},
        )
        assert resp.status_code == 400, (
            f"Expected 400 for negative row, got {resp.status_code}: {resp.text}"
        )

    def test_valid_pixel_returns_200(self, client):
        """A pixel inside the grid must return 200 with element data."""
        resp = client.post(
            "/api/eds/quantify/pixel",
            json={"row": 2, "col": 1, "display_mode": "counts"},
        )
        assert resp.status_code == 200, (
            f"Expected 200 for valid pixel, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert body["row"] == 2
        assert body["col"] == 1
        assert "data" in body, "Response must have 'data' field"
        # data maps element -> {counts, wt_pct, at_pct}
        assert isinstance(body["data"], dict)


class TestSuggestPhasesBoundsCheck:
    """POST /api/eds/suggest-phases — bounds check + HTTPException passthrough."""

    def test_out_of_bounds_pixel_returns_400_not_500(self, client):
        """row=99 must yield 400, NOT 500 (the HTTPException must propagate through
        the outer try/except in suggest_phases)."""
        resp = client.post(
            "/api/eds/suggest-phases",
            json={"row": 99, "col": 0},
        )
        assert resp.status_code == 400, (
            f"Expected 400 for out-of-bounds pixel (not 500), "
            f"got {resp.status_code}: {resp.text}"
        )
        detail = resp.json().get("detail", "")
        assert "outside" in detail.lower() or "bounds" in detail.lower(), (
            f"Expected 'outside'/'bounds' in error detail, got: {detail!r}"
        )

    def test_valid_pixel_does_not_raise(self, client):
        """A valid pixel must return 200 (or at least not 400/500)."""
        resp = client.post(
            "/api/eds/suggest-phases",
            json={"row": 0, "col": 0},
        )
        # eds_utils is available (project-internal), so should return 200.
        assert resp.status_code == 200, (
            f"Expected 200 for valid suggest-phases pixel, "
            f"got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert "suggestions" in body
        assert "atomic_pct" in body
