"""POST /api/analysis/pc-drift-to-seed: the Aztec PC map goes into the store in Bruker.

The light .h5 carries the PC as Oxford wrote it (``Pattern Center X``, ``Pattern
Center Y``, ``Detector Distance``). The calibration store - and with it the PC
Refinement page and every indexer - holds the BRUKER convention. The route stacked
the raw numbers and stored them as they were: for the Oxford reference scan (128 x 156 px patterns)
PCy 0.552 and PCz 0.694 stood where 0.327 and 0.846 belong, and the map was marked
a refinement that later refines could not replace.

The numbers are the Oxford reference scan's own: the raw triple of its h5oina header, and the Bruker
triple kikuchipy makes of it when it loads that file (read off the loaded signal,
not computed here). Aspect ratio of the 128 x 156 detector: 156 / 128.
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from backend.api.services.calibration_store import CalibrationStore

ROWS, COLS = 3, 4
SHAPE = (128, 156)                                  # pattern rows, cols
OXFORD = (0.5026, 0.5520, 0.6940)                   # header of the file: PCx, PCy, DD
BRUKER = (0.5026, 0.3273, 0.8458)                   # what kikuchipy loads from it


class _Det:
    def __init__(self):
        self.shape = SHAPE
        self.pc = np.array([0.5, 0.5, 0.5])
        self.sample_tilt, self.tilt, self.azimuthal = 70.0, 4.3, 0.0


@pytest.fixture
def route(monkeypatch):
    from backend.api.routes import analysis
    store = CalibrationStore()
    store.register("sampleB", SimpleNamespace(detector=_Det()))
    n = ROWS * COLS
    # a slightly varying map around the file's PC, so the structure can be checked
    dx = np.linspace(-1e-3, 1e-3, n)
    ds = SimpleNamespace(
        xmap=SimpleNamespace(prop={
            "pc_x": OXFORD[0] + dx, "pc_y": OXFORD[1] + 0 * dx, "dd": OXFORD[2] + 0 * dx}),
        size=n, shape=(ROWS, COLS))
    monkeypatch.setattr(analysis, "_dataset", ds)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    return analysis, store


def _seed(analysis):
    return asyncio.run(analysis.pc_drift_to_seed(analysis.PcSeedRequest(target_dataset="sampleB")))


def test_the_seed_is_stored_in_bruker(route):
    analysis, store = route
    res = _seed(analysis)
    np.testing.assert_allclose(res["mean_pc"], BRUKER, atol=1e-3)
    entry = store.get_entry("sampleB")
    np.testing.assert_allclose(entry.pc_single, BRUKER, atol=1e-3)
    assert entry.pc_map.shape == (ROWS, COLS, 3)
    # pixel by pixel, by hand: PCx as it is, PCy = 1 - y * 156/128, PCz = dd * 156/128
    aspect = SHAPE[1] / SHAPE[0]
    np.testing.assert_allclose(entry.pc_map[..., 1], 1 - OXFORD[1] * aspect, atol=1e-6)
    np.testing.assert_allclose(entry.pc_map[..., 2], OXFORD[2] * aspect, atol=1e-6)
    np.testing.assert_allclose(entry.pc_map[0, :, 0], OXFORD[0] + np.linspace(-1e-3, 1e-3, ROWS * COLS)[:COLS],
                               atol=1e-6)


def test_the_seed_is_the_files_map_not_a_refinement(route):
    analysis, store = route
    _seed(analysis)
    entry = store.get_entry("sampleB")
    assert entry.pc_source == "header"
    assert entry.pc_refinement is None


def test_a_later_refine_moves_the_seeded_map(route):
    """It used to be marked a genuine refinement, so a Global PC Refine left the
    seed in place and the page showed a PC indexing was not using."""
    analysis, store = route
    _seed(analysis)
    new_pc = np.array([0.5101, 0.3302, 0.8500])
    store.update_pc("sampleB", new_pc, source="refined")
    entry = store.get_entry("sampleB")
    np.testing.assert_allclose(entry.pc_map.reshape(-1, 3).mean(axis=0), new_pc, atol=1e-6)


def test_the_reported_mean_is_the_stored_one(route):
    """float32 mean of the map array vs the double mean the store keeps differ in the
    sixth digit (0.84589 vs 0.845838 for the Oxford reference scan); the response says the stored one."""
    analysis, store = route
    res = _seed(analysis)
    stored = store.get_entry("sampleB").pc_single
    np.testing.assert_array_equal(np.asarray(res["mean_pc"]), np.asarray(stored, dtype=float))
    assert np.asarray(res["mean_pc"]).dtype == np.float64
