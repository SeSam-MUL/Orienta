"""Tests for POST /api/eds/probe — multi-layer hover-tooltip backend.

The probe endpoint is the backend half of Task 12 (HoverProbeOverlay): one
roundtrip returns EDS element values, BC value, and phase at a pixel.

Hermetic tests run by mocking the get_extractor() + _read_native_band_contrast +
phase store. The full E2E with a real H5OINA file is gated by file presence
and will be exercised manually during Phase 1 acceptance.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


@pytest.fixture
def mock_session(monkeypatch):
    """Replace h5_session + extractor with controllable stubs."""
    # Mock the open-file guard so probe doesn't 400 out.
    monkeypatch.setattr('backend.api.routes.eds.is_open', lambda: True)

    # Mock the extractor with a 4×3 scan (12 pixels, rows=4, cols=3).
    ext = MagicMock()
    ext.get_grid_dimensions.return_value = (4, 3)
    ext.get_available_elements.return_value = ['Al Kα1', 'Fe Kα1']
    # Element map is a 1-D counts array of length rows*cols.
    al_map = np.zeros(12, dtype=np.float32); al_map[5] = 1000.0   # pixel (1, 2)
    fe_map = np.zeros(12, dtype=np.float32); fe_map[5] = 200.0
    def fake_get_element_map(el):
        return {'Al Kα1': al_map, 'Fe Kα1': fe_map}[el]
    ext.get_element_map.side_effect = fake_get_element_map
    monkeypatch.setattr('backend.api.routes.eds.get_extractor', lambda: ext)


def test_probe_requires_open_file():
    """Returns 400 when no file is open."""
    with patch('backend.api.routes.eds.is_open', return_value=False):
        r = client.post('/api/eds/probe', json={'row': 1, 'col': 2})
        assert r.status_code == 400


def test_probe_returns_element_quantification(mock_session):
    """Probe at pixel (1, 2) returns counts + wt% + at% for both Al and Fe."""
    r = client.post('/api/eds/probe', json={'row': 1, 'col': 2, 'display_mode': 'at_pct'})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['row'] == 1
    assert data['col'] == 2
    assert 'elements' in data
    assert 'Al' in data['elements']
    assert 'Fe' in data['elements']
    assert data['elements']['Al']['counts'] == pytest.approx(1000.0)
    assert data['elements']['Fe']['counts'] == pytest.approx(200.0)
    # At% must be a finite number normalised to ~100%.
    assert 0 <= data['elements']['Al']['at_pct'] <= 100
    assert 0 <= data['elements']['Fe']['at_pct'] <= 100


def test_probe_bounds_checked(mock_session):
    """Negative or out-of-grid pixel returns 422."""
    r = client.post('/api/eds/probe', json={'row': -1, 'col': 0})
    assert r.status_code in (400, 422)
    r = client.post('/api/eds/probe', json={'row': 100, 'col': 0})
    assert r.status_code in (400, 422)


def test_probe_bc_null_when_unavailable(mock_session, monkeypatch):
    """When _read_native_band_contrast returns None, probe sets bc=null."""
    monkeypatch.setattr('backend.api.routes.eds._read_native_bc_safe', lambda: None)
    r = client.post('/api/eds/probe', json={'row': 1, 'col': 2})
    assert r.status_code == 200
    assert r.json()['bc'] is None


def test_probe_bc_value_when_available(mock_session, monkeypatch):
    """When _read_native_band_contrast returns an array, probe extracts the cell value."""
    bc_grid = np.zeros((4, 3), dtype=np.float32); bc_grid[1, 2] = 178.0
    monkeypatch.setattr('backend.api.routes.eds._read_native_bc_safe', lambda: bc_grid)
    r = client.post('/api/eds/probe', json={'row': 1, 'col': 2})
    assert r.status_code == 200
    assert r.json()['bc'] == pytest.approx(178.0)


def test_probe_bc_source_native_when_available(mock_session, monkeypatch):
    """When native BC is present, probe reports bc_source == 'native'."""
    bc_grid = np.zeros((4, 3), dtype=np.float32); bc_grid[1, 2] = 178.0
    monkeypatch.setattr('backend.api.routes.eds._read_native_bc_safe', lambda: bc_grid)
    r = client.post('/api/eds/probe', json={'row': 1, 'col': 2})
    assert r.status_code == 200
    body = r.json()
    assert body['bc'] is not None
    assert body['bc_source'] == 'native'


def test_probe_bc_source_null_when_unavailable(mock_session, monkeypatch):
    """When native BC is absent, probe reports bc_source is None (no computed path in the probe)."""
    monkeypatch.setattr('backend.api.routes.eds._read_native_bc_safe', lambda: None)
    r = client.post('/api/eds/probe', json={'row': 1, 'col': 2})
    assert r.status_code == 200
    body = r.json()
    assert body['bc'] is None
    assert body['bc_source'] is None


def test_probe_phase_null_when_no_phase_map(mock_session, monkeypatch):
    """When no phase map is loaded, probe returns phase=null (not crash)."""
    monkeypatch.setattr('backend.api.routes.eds._probe_phase_at_safe', lambda r, c: None)
    r = client.post('/api/eds/probe', json={'row': 1, 'col': 2})
    assert r.status_code == 200
    assert r.json()['phase'] is None


def test_probe_phase_when_present(mock_session, monkeypatch):
    """When a phase map exists, probe returns { id, name } at the pixel."""
    monkeypatch.setattr('backend.api.routes.eds._probe_phase_at_safe',
                        lambda r, c: {'id': 1, 'name': 'Al-rich (β)'})
    r = client.post('/api/eds/probe', json={'row': 1, 'col': 2})
    assert r.status_code == 200
    assert r.json()['phase'] == {'id': 1, 'name': 'Al-rich (β)'}


def _probe_with_windows(monkeypatch, windows):
    """Probe a 4x3 scan whose windows the test chooses (counts at pixel 5)."""
    monkeypatch.setattr('backend.api.routes.eds.is_open', lambda: True)
    ext = MagicMock()
    ext.get_grid_dimensions.return_value = (4, 3)
    ext.get_available_elements.return_value = list(windows)

    def fake_map(name):
        arr = np.zeros(12, dtype=np.float32)
        arr[5] = float(windows[name])
        return arr

    ext.get_element_map.side_effect = fake_map
    monkeypatch.setattr('backend.api.routes.eds.get_extractor', lambda: ext)
    monkeypatch.setattr('backend.api.routes.eds._probe_phase_at_safe',
                        lambda r, c: None)
    return client.post('/api/eds/probe',
                       json={'row': 1, 'col': 2, 'display_mode': 'at_pct'})


def test_probe_survives_one_unusable_window(monkeypatch):
    """One window the physics cannot price must not cost the whole tooltip.

    The probe is multi-layer: BC and phase do not go through EDS physics at
    all, and they used to disappear with the 400 this raised over one stray
    label. Since 2026-09-12 the skip is per window (eds_utils
    ._resolve_k_factors), so Al is quantified and only the stray one reports
    no composition.
    """
    r = _probe_with_windows(monkeypatch, {'Al Kα1': 1000.0, 'Xx Kα1': 50.0})
    assert r.status_code == 200, r.text
    els = r.json()['elements']
    assert els['Al']['at_pct'] == pytest.approx(100.0)
    assert els['Xx']['counts'] == pytest.approx(50.0)
    assert els['Xx']['wt_pct'] is None and els['Xx']['at_pct'] is None


def test_probe_degrades_to_counts_when_nothing_is_quantifiable(monkeypatch):
    """Even with NO usable window the other layers must still answer.

    Counts are a real measurement and are returned; wt%/at% are omitted
    rather than fabricated as 0.0.
    """
    r = _probe_with_windows(monkeypatch, {'Xx Kα1': 50.0})
    assert r.status_code == 200, r.text
    els = r.json()['elements']
    assert els['Xx Kα1']['counts'] == pytest.approx(50.0)
    assert els['Xx Kα1']['wt_pct'] is None
    assert 'bc' in r.json() and 'phase' in r.json()
