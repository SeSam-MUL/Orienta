"""POST /api/analysis/pc-drift-to-seed also hands the map to an open PC Refinement page.

The page works on its own controller, which reads the calibration store only the
first time it is opened for a dataset. A seed written to the store alone left that
open page refining from its old PC, while the button said refinement would start
from the seed. The numbers are the Oxford reference scan's: the Bruker triple kikuchipy makes of the
header's Oxford one (see test_pc_seed_convention.py).
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from kikuchipy.detectors import EBSDDetector

from backend.api.services.calibration_store import CalibrationStore
from pc_controller import PCController, pc_is_per_pixel

ROWS, COLS = 3, 4
SHAPE = (128, 156)
OXFORD = (0.5026, 0.5520, 0.6940)
BRUKER = (0.5026, 0.3273, 0.8458)


class _Det:
    def __init__(self):
        self.shape = SHAPE
        self.pc = np.array([0.5, 0.5, 0.5])
        self.sample_tilt, self.tilt, self.azimuthal = 70.0, 4.3, 0.0


@pytest.fixture
def seeded_route(monkeypatch):
    from backend.api.routes import analysis, pcrefinement
    store = CalibrationStore()
    store.register("sampleB", SimpleNamespace(detector=_Det()))
    n = ROWS * COLS
    ds = SimpleNamespace(
        xmap=SimpleNamespace(prop={
            "pc_x": np.full(n, OXFORD[0]), "pc_y": np.full(n, OXFORD[1]), "dd": np.full(n, OXFORD[2])}),
        size=n, shape=(ROWS, COLS))
    monkeypatch.setattr(analysis, "_dataset", ds)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", store)
    sessions = {}
    monkeypatch.setattr(pcrefinement, "_sessions", sessions)
    return analysis, sessions


def _page_controller(sessions, tilt=3.0, sample_tilt=68.0):
    ctrl = PCController()
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.4, 0.4, 0.7), px_size=70.0,
                                      binning=4, tilt=tilt, sample_tilt=sample_tilt))
    sessions["sampleB"] = SimpleNamespace(ctrl=ctrl, sim_cache={0: "old simulation"})
    return ctrl


def _seed(analysis):
    return asyncio.run(analysis.pc_drift_to_seed(analysis.PcSeedRequest(target_dataset="sampleB")))


def test_an_open_page_gets_the_seed_map_with_its_own_geometry(seeded_route):
    analysis, sessions = seeded_route
    ctrl = _page_controller(sessions)
    res = _seed(analysis)
    assert res["pc_page_updated"] is True
    assert pc_is_per_pixel(ctrl.detector)
    np.testing.assert_allclose(np.asarray(ctrl.detector.pc).reshape(-1, 3).mean(axis=0), BRUKER, atol=1e-3)
    assert np.asarray(ctrl.detector.pc).shape == (ROWS, COLS, 3)
    # the page's own tilts, pixel size and binning are not replaced by the seed
    assert (float(ctrl.detector.tilt), float(ctrl.detector.sample_tilt)) == (3.0, 68.0)
    assert (float(ctrl.detector.px_size), int(ctrl.detector.binning)) == (70.0, 4)
    assert sessions["sampleB"].sim_cache == {}           # bands of the old PC are gone


def test_without_an_open_page_the_store_alone_is_updated(seeded_route):
    analysis, sessions = seeded_route
    res = _seed(analysis)
    assert res["pc_page_updated"] is False
    assert sessions == {}
