"""Applying a calibration whose file another program holds writes nothing and answers with a code.

The member list of the calibration (which file runs on which version) is written BEFORE the PC reaches the
dataset: a held calibration file used to fail after the PC was written, as a bare 500, and the dataset
stayed on a PC that no calibration remembered giving it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402

PC_A = (0.5026, 0.3272, 0.8458)
PC_B = (0.5, 0.4, 0.6)


def _signal(wd=15.0, shape=(128, 156), pc=PC_B):
    import kikuchipy as kp
    from kikuchipy.detectors import EBSDDetector
    det = EBSDDetector(shape=shape, pc=pc, sample_tilt=70.0, tilt=4.32)
    sig = kp.signals.EBSD(np.zeros((2, 2) + shape, np.uint8), detector=det)
    sig.metadata.set_item("Acquisition_instrument.SEM.working_distance", wd)
    return sig


@pytest.fixture
def world(tmp_path, monkeypatch):
    from backend.api.main import app
    from backend.api.routes import pcrefinement
    from backend.api.services.calibration_store import calibration_store as store
    monkeypatch.setattr(pcrefinement, "_optimization_active", False)
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    store.clear()
    use.forget_all()
    for name in ("scan_a", "scan_b"):
        path = tmp_path / f"{name}.h5oina"
        path.write_bytes(name.encode() * 40)
        store.register(name, _signal(), file_key=str(path))
    store.update_pc("scan_a", PC_A, source="refined", refinement={"n_patterns": 9, "method": "PSO"})
    yield TestClient(app), store
    store.clear()
    use.forget_all()
    cal.set_calibrations_dir_for_test(None)


def _hold_the_calibration_file(monkeypatch):
    monkeypatch.setattr(cal, "_RETRY_PAUSE_S", 0)

    def held(*args, **kwargs):
        raise PermissionError(13, "in use by another program")

    monkeypatch.setattr(os, "replace", held)


def test_a_held_calibration_file_writes_no_pc_and_answers_with_a_code(world, monkeypatch):
    client, store = world
    made = client.post("/api/pc-calibrations", json={"name": "REM", "dataset": "scan_a"}).json()
    _hold_the_calibration_file(monkeypatch)
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]})
    assert r.status_code == 200, r.text
    answer = r.json()["results"][0]
    assert answer["dataset"] == "scan_b" and answer["status"] == "notWritable"
    np.testing.assert_allclose(store.get_pc("scan_b"), PC_B, atol=1e-9)         # nothing was written
    assert store.get_entry("scan_b").pc_source != "calibration"
    assert use.current_ref("scan_b", store) is None


def test_an_error_of_the_member_list_that_is_not_a_held_file_is_a_code_too(world, monkeypatch):
    client, store = world
    made = client.post("/api/pc-calibrations", json={"name": "REM", "dataset": "scan_a"}).json()

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]})
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["status"] == "notWritable"
    np.testing.assert_allclose(store.get_pc("scan_b"), PC_B, atol=1e-9)


def test_the_file_that_can_be_written_is_applied_as_before(world):
    client, store = world
    made = client.post("/api/pc-calibrations", json={"name": "REM", "dataset": "scan_a"}).json()
    r = client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]})
    assert r.json()["results"][0]["status"] == "applied"
    np.testing.assert_allclose(store.get_pc("scan_b"), PC_A, atol=1e-9)
    assert use.current_ref("scan_b", store)["id"] == made["id"]


def test_the_member_is_written_before_the_pc(world, monkeypatch):
    """The order itself: when the PC is written the calibration already lists the file."""
    client, store = world
    made = client.post("/api/pc-calibrations", json={"name": "REM", "dataset": "scan_a"}).json()
    from backend.api.routes import pcrefinement
    seen = {}
    real = pcrefinement._writeback_refined_pc

    def spy(name, *a, **k):
        seen["members"] = [m["name"] for m in cal.get(made["id"])["members"]]
        return real(name, *a, **k)

    monkeypatch.setattr(pcrefinement, "_writeback_refined_pc", spy)
    client.post(f"/api/pc-calibrations/{made['id']}/apply", json={"datasets": ["scan_b"]})
    assert "scan_b.h5oina" in seen["members"]
