"""Deleting a dataset also drops its PC Refinement session.

The PC page keeps one session (calibration patterns, detector, phases, simulated
results) per dataset NAME. Names are recycled: a deepcopy or a crop of the same
source gets the same name again (``X_bg_dyn``). The session of a deleted dataset
stayed behind, so after deleting ``X_bg_dyn`` and processing again, the PC page of
the new dataset showed the deleted one's patterns and PC, and adding a pattern at the
same position answered "duplicate".
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from kikuchipy.detectors import EBSDDetector

SHAPE = (60, 60)


class _Axes:
    navigation_shape = (4, 4)


class _Signal:
    axes_manager = _Axes()

    def __init__(self):
        self.data = np.zeros((4, 4) + SHAPE, dtype=np.uint8)


@pytest.fixture
def env(monkeypatch):
    from backend.api.routes import ebsd_viewer
    from backend.api.routes import pcrefinement as R
    from backend.api.services.calibration_store import CalibrationStore
    monkeypatch.setattr(R, "_sessions", {})
    monkeypatch.setattr(R, "_parked_sessions", {})
    monkeypatch.setattr(R, "calibration_store", CalibrationStore())
    monkeypatch.setattr(ebsd_viewer, "calibration_store", CalibrationStore())
    monkeypatch.setattr(ebsd_viewer, "_raw_signals", {"X_bg_dyn": _Signal(), "Y": _Signal()})
    monkeypatch.setattr(ebsd_viewer, "_positions", {})
    monkeypatch.setattr(ebsd_viewer, "_signal_masks", {})
    monkeypatch.setattr(ebsd_viewer, "_dirty_datasets", set())
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "X_bg_dyn")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_signal", None)
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", None, raising=False)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal",
                        lambda *a, **k: ebsd_viewer._raw_signals.get(ebsd_viewer._active_dataset))
    return ebsd_viewer, R


def _calibrated(R, row=1, col=2):
    ctrl = R._get_controller()
    ctrl.attach_detector(EBSDDetector(shape=SHAPE, pc=(0.31, 0.32, 0.33), sample_tilt=70.0))
    ctrl.add_pattern((row, col), np.zeros(SHAPE, np.uint8))
    R._sim_cache[0] = {"ci": 0.8}
    return ctrl


def test_a_recycled_name_starts_with_a_fresh_pc_page(env):
    ebsd_viewer, R = env
    old = _calibrated(R)
    assert len(old.patterns) == 1

    out = asyncio.run(ebsd_viewer.delete_dataset("X_bg_dyn"))
    assert out["deleted"] == "X_bg_dyn"

    # the name comes back (process again): a new signal under the same name
    ebsd_viewer._raw_signals["X_bg_dyn"] = _Signal()
    ebsd_viewer._active_dataset = "X_bg_dyn"
    fresh = R._get_controller()
    assert fresh is not old
    assert fresh.patterns == [] and fresh.detector is None
    assert len(R._sim_cache) == 0
    r = asyncio.run(R.add_pattern(R.AddPatternRequest(row=1, col=2)))
    assert r["n_patterns"] == 1 and "duplicate" not in r     # not "duplicate" of the deleted one


def test_deleting_another_dataset_leaves_the_session_of_the_active_one(env):
    ebsd_viewer, R = env
    ctrl = _calibrated(R)                                    # session of X_bg_dyn
    asyncio.run(ebsd_viewer.delete_dataset("Y"))
    assert R._sessions["X_bg_dyn"].ctrl is ctrl
    assert len(ctrl.patterns) == 1


def test_a_dataset_without_a_session_is_deleted_all_the_same(env):
    ebsd_viewer, R = env
    out = asyncio.run(ebsd_viewer.delete_dataset("Y"))       # no PC page was ever opened for Y
    assert out["deleted"] == "Y" and "Y" not in ebsd_viewer._raw_signals


def test_drop_session_forgets_the_session_and_the_one_parked_for_this_file(env, monkeypatch):
    ebsd_viewer, R = env
    monkeypatch.setattr(R, "_current_file_key", lambda: "fileA")
    mine = R._PCSession(ctrl=SimpleNamespace(), file_key="fileA")
    mine_parked = R._PCSession(ctrl=SimpleNamespace(), file_key="fileA")
    R._sessions["X"] = mine
    R._sessions["Z"] = R._PCSession(ctrl=SimpleNamespace(), file_key="fileA")
    R._parked_sessions[("fileA", "X")] = mine_parked
    assert R.drop_session("X") is True
    assert "X" not in R._sessions
    assert "Z" in R._sessions                                # another dataset is not touched
    assert ("fileA", "X") not in R._parked_sessions          # its parked twin goes too
    assert R.drop_session("X") is False                      # nothing left to drop


def test_drop_session_keeps_the_session_of_another_file_with_the_same_name(env, monkeypatch):
    ebsd_viewer, R = env
    monkeypatch.setattr(R, "_current_file_key", lambda: "fileA")
    theirs = R._PCSession(ctrl=SimpleNamespace(), file_key="fileB")
    R._sessions["X"] = theirs                                # file B's, not yet parked
    R._parked_sessions[("fileC", "X")] = R._PCSession(ctrl=SimpleNamespace(), file_key="fileC")
    assert R.drop_session("X") is False
    assert "X" not in R._sessions                            # not served to file A ...
    assert R._parked_sessions[("fileB", "X")] is theirs      # ... but kept for file B
    assert ("fileC", "X") in R._parked_sessions
