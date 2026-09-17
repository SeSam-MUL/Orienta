"""Fixtures for the citation-chain drift-guard tests.

These run the smallest genuine real job available rather than mocking the
pipeline under test — see tests/test_hough_camera_tilt.py, which runs the
same real-ROI Hough job and is the pattern followed here.
"""
import uuid

import numpy as np
import pytest

from backend.api.routes import indexing as indexing_routes
from backend.api.services.citations.provenance import PROVENANCE_SCHEMA


class _StubResult:
    """Enough of an IndexingResult for the citation endpoint."""

    def __init__(self, steps):
        self.metadata = {
            "provenance": {"schema": PROVENANCE_SCHEMA, "steps": steps}
        }


@pytest.fixture
def register_result():
    """Put a result in the registry and take it out again afterwards."""
    created = []

    def _register(steps):
        rid = f"test_{uuid.uuid4().hex[:8]}"
        indexing_routes._result_registry[rid] = _StubResult(steps)
        created.append(rid)
        return rid

    yield _register
    for rid in created:
        indexing_routes._result_registry.pop(rid, None)


@pytest.fixture
def stored_result_id(register_result):
    return register_result([
        {"key": "preprocessing.background", "params": {"background": "dynamic"}},
        {"key": "indexing.hough", "params": {"orienta_version": "0.3.0"}},
    ])


@pytest.fixture
def stored_result_id_no_steps(register_result):
    return register_result([])


@pytest.fixture
def stored_result_id_undeclared(register_result):
    return register_result([{"key": "some.addon.step", "params": {}}])


@pytest.fixture
def indexed_result_hough(h5oina_path, al_cif_path):
    """A real, small Hough indexing result.

    12x12 pixel ROI inside the single Al grain of the SampleB h5oina (same
    window as test_hough_camera_tilt.py) — small enough to stay fast, real
    enough that ``hough_index_patterns`` actually runs and its
    ``record_step("indexing.hough", ...)`` call actually fires.
    """
    pytest.importorskip("pyebsdindex")
    from orix.crystal_map import Phase, PhaseList

    from ebsd_utils import sanitize_cif
    from indexing_controller import (
        IndexingConfig,
        IndexingMethod,
        hough_index_patterns,
    )
    from safe_loader import load_ebsd_safe

    signal = load_ebsd_safe(str(h5oina_path), verbose=False)
    n_rows, n_cols = signal.data.shape[:2]
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[18:30, 2:14] = True

    phase = Phase.from_cif(sanitize_cif(str(al_cif_path)))
    phase.name = "Al"
    phase_list = PhaseList(phase)

    return hough_index_patterns(
        signal, phase_list, signal.detector,
        IndexingConfig(method=IndexingMethod.HOUGH), mask,
    )
