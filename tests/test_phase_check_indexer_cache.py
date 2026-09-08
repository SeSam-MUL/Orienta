"""The phase check must build each candidate phase's Hough indexer ONCE.

Measured on 2026-09-03 against a real run (3 phases, 31 grains checked):
`check_map` calls `hough_quats_fn` per grain per candidate phase, and
`_hough_quats_for_phase_batch` built a fresh indexer on every one of those 62
calls. Building it is the expensive half — a transient Windows commit charge of

    Al7FeCu2.cif  (P 1)    48.61 GiB      sd_1814127.cif (mmm)   5.59 GiB

against a commit limit of 83 GiB on that machine, while the backend process was
already holding 25 GB. The NVIDIA OpenCL driver could then not get host memory
for its context and every single call died with

    batch hough for phase failed (...Al7FeCu2.cif): Context failed: OUT_OF_HOST_MEMORY

after ~7 s. 62 of those = 6 min 25 s of doing nothing, and the frontend gave up
at its 300 s timeout, so the user just saw "Phase Verification does not work".

Reusing the indexer removes the spike AND is ~5x faster per call (measured:
peak 5.0 GiB instead of 48.6, 0.10 s instead of 0.51 s).

The second test is the safety half: a cache keyed too loosely would index a
grain against the WRONG detector geometry, which is silent and far worse than
slow.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ebsd_utils  # noqa: E402
from backend.api.routes.indexing import (  # noqa: E402
    _HOUGH_INDEXER_CACHE,
    _hough_quats_for_phase_batch,
)

# mmm phase on purpose: cheap enough to build for real in a test (the P1 one
# would ask for 48 GiB of commit).
CIF = Path(__file__).resolve().parents[1] / "Database" / "CIF_Library" / "sd_1814127.cif"

DET = {
    "pat_height": 156, "pat_width": 128,
    "sample_tilt": 70.0, "tilt": 0.0,
    "pc_x": 0.5, "pc_y": 0.5, "pc_z": 0.5, "binning": 1,
}


@pytest.fixture
def count_indexer_builds(monkeypatch):
    """Count real create_indexer calls (the expensive half).

    The cache is module-level and outlives a test, so each test starts from an
    empty one — otherwise a test would silently measure its predecessor's
    leftovers instead of its own behaviour.
    """
    _HOUGH_INDEXER_CACHE.clear()
    calls = []
    real = ebsd_utils.create_indexer

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(ebsd_utils, "create_indexer", counting)
    return calls


@pytest.fixture
def patterns():
    rng = np.random.default_rng(0)
    return (rng.random((8, DET["pat_height"], DET["pat_width"])) * 255).astype(np.float32)


@pytest.mark.skipif(not CIF.exists(), reason="phase library CIF not present")
def test_indexer_is_built_once_across_grains(count_indexer_builds, patterns):
    """Three grains, one phase, one detector -> one indexer build."""
    for _ in range(3):
        out = _hough_quats_for_phase_batch(patterns, str(CIF), DET)
        assert out.shape == (patterns.shape[0], 4)
    assert len(count_indexer_builds) == 1, (
        f"built the indexer {len(count_indexer_builds)}x for 3 grains of the same "
        "phase — each build is the multi-GiB allocation that exhausts the commit "
        "limit on a long-running backend"
    )


@pytest.mark.skipif(not CIF.exists(), reason="phase library CIF not present")
def test_changed_detector_geometry_rebuilds(count_indexer_builds, patterns):
    """A different projection centre must NOT reuse the cached indexer.

    Reusing it would index against a geometry the caller did not ask for, and
    nothing downstream could tell: the orientations come back looking valid.
    """
    _hough_quats_for_phase_batch(patterns, str(CIF), DET)
    moved = {**DET, "pc_x": 0.42}
    _hough_quats_for_phase_batch(patterns, str(CIF), moved)
    assert len(count_indexer_builds) == 2
