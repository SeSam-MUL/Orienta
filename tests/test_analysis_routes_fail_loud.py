"""Fail-loud tests for BC-calibrated analysis routes.

The 3-Gaussian Band-Contrast GMM and the BC quality-filter are calibrated on
REAL Oxford Band Contrast. When the loaded dataset carries only computed
Pattern Quality (Hough `pq`) or NCC match confidence (`scores`) — but no native
`bc` — these endpoints must fail loudly with HTTP 400, not silently run on
fallback data.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    from diffpy.structure import Lattice, Structure
    ORIX_AVAILABLE = True
except ImportError:
    ORIX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ORIX_AVAILABLE, reason="orix not available")


def _make_pq_dataset(ny: int = 5, nx: int = 5):
    """Build an EBSDDataset whose only quality metric is Hough pattern quality
    (`pq`) — i.e. NO native Band Contrast."""
    from analysis.ebsd_dataset import EBSDDataset

    n = ny * nx
    phase = Phase(
        name="Al",
        space_group=225,
        structure=Structure(lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90)),
    )
    xmap = CrystalMap(
        rotations=Rotation.identity(n),
        phase_id=np.zeros(n, dtype=int),
        x=np.tile(np.arange(nx), ny).astype(float),
        y=np.repeat(np.arange(ny), nx).astype(float),
        phase_list=PhaseList([phase]),
        prop={"pq": np.linspace(0.0, 1.0, n).astype(float)},
    )
    return EBSDDataset(xmap, step_size=1.0, source="indexing_result")


@pytest.fixture
def client_with_pq_dataset():
    """TestClient with the analysis module-level `_dataset` set to a pq-only
    dataset (no native BC). Restores the previous dataset afterwards."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    import backend.api.routes.analysis as ar

    saved = ar._dataset
    ar._dataset = _make_pq_dataset()
    try:
        yield TestClient(app)
    finally:
        ar._dataset = saved


def test_bc_gmm_fail_loud_without_native_bc(client_with_pq_dataset):
    r = client_with_pq_dataset.post("/api/analysis/bc_gmm")
    assert r.status_code == 400
    assert "native Band Contrast" in r.json()["detail"]


def test_quality_filter_fail_loud_without_native_bc(client_with_pq_dataset):
    r = client_with_pq_dataset.post("/api/analysis/quality-filter", json={})
    assert r.status_code == 400
    assert "native Band Contrast" in r.json()["detail"]


# ─── Native BC bridge for indexing results ───────────────────────────────────
#
# When loading `__from_indexing__`, the route must bridge the REAL native Band
# Contrast from the result's OWN source file into xmap.prop['bc'] so that
# has_native_bc / bc_gmm / quality-filter work for Oxford scans. It must NOT
# synthesise a surrogate when the source has no native BC (EDAX / synthetic /
# stripped) — those must still fail loud truthfully.

def _make_bcless_xmap(ny: int = 5, nx: int = 5):
    """A CrystalMap with only `pq` (no native `bc`), mirroring a fresh
    indexing result before the bridge runs."""
    n = ny * nx
    phase = Phase(
        name="Al",
        space_group=225,
        structure=Structure(lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90)),
    )
    return CrystalMap(
        rotations=Rotation.identity(n),
        phase_id=np.zeros(n, dtype=int),
        x=np.tile(np.arange(nx), ny).astype(float),
        y=np.repeat(np.arange(ny), nx).astype(float),
        phase_list=PhaseList([phase]),
        prop={"pq": np.linspace(0.0, 1.0, n).astype(float)},
    )


class _FakeResult:
    """Minimal stand-in for the indexing result: an xmap + a metadata dict
    carrying `source_file`, exactly like the real result."""
    def __init__(self, xmap, source_file):
        self.xmap = xmap
        self.metadata = {"source_file": source_file}


@pytest.fixture
def client_from_indexing(monkeypatch):
    """TestClient wired so /api/analysis/load '__from_indexing__' pulls a
    bc-less xmap tagged with a source_file. `read_native_band_contrast` is
    monkeypatched per-test to simulate an Oxford source (2D array) or a
    non-Oxford source (None)."""
    from fastapi.testclient import TestClient
    from backend.api.main import app
    import backend.api.routes.analysis as ar
    import backend.api.routes.indexing as ix

    ny, nx = 5, 5
    result = _FakeResult(_make_bcless_xmap(ny, nx), source_file="C:/fake/scan.h5oina")
    monkeypatch.setattr(ix, "get_last_indexing_result", lambda: result)

    saved = ar._dataset
    ar._dataset = None
    try:
        yield TestClient(app), ar, (ny, nx)
    finally:
        ar._dataset = saved


def test_from_indexing_bridges_native_bc(client_from_indexing, monkeypatch):
    """Oxford source with native BC → dataset.has_native_bc True and bc_gmm no
    longer 400s on the has_native_bc gate."""
    import backend.api.services.pattern_quality as pq
    client, ar, (ny, nx) = client_from_indexing

    native = np.linspace(0, 230, ny * nx, dtype=float).reshape(ny, nx)
    monkeypatch.setattr(pq, "read_native_band_contrast",
                        lambda src, n_rows=None, n_cols=None: native)

    r = client.post("/api/analysis/load", json={"xmap_path": "__from_indexing__"})
    assert r.status_code == 200, r.text
    assert ar._dataset is not None
    assert ar._dataset.has_native_bc is True

    gmm = client.post("/api/analysis/bc_gmm")
    # Must not be rejected by the native-BC gate anymore.
    if gmm.status_code == 400:
        assert "native Band Contrast" not in gmm.json()["detail"]


def test_from_indexing_no_native_bc_still_fails_loud(client_from_indexing, monkeypatch):
    """Non-Oxford source (read returns None) → no surrogate injected, bc_gmm
    still fails loud with the native-BC message."""
    import backend.api.services.pattern_quality as pq
    client, ar, _ = client_from_indexing

    monkeypatch.setattr(pq, "read_native_band_contrast",
                        lambda src, n_rows=None, n_cols=None: None)

    r = client.post("/api/analysis/load", json={"xmap_path": "__from_indexing__"})
    assert r.status_code == 200, r.text
    assert ar._dataset.has_native_bc is False

    gmm = client.post("/api/analysis/bc_gmm")
    assert gmm.status_code == 400
    assert "native Band Contrast" in gmm.json()["detail"]
