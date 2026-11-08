"""A file opened again under another spelling of its name (case only) gets its calibration back.

The stash of a file is found by its canonical path, which ignores case where the file system does. The raw
dataset is named after the path as it was given, so the stash can hold the file's entry as "HiGainNi" while the
fresh one is "higainni". That is one file: the stashed calibration goes to the fresh entry, and nothing is left
behind under the old spelling.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ORIENTA_ALLOWED_HOSTS", "testserver")

from backend.api.services.calibration_store import CalibrationStore  # noqa: E402

TYPED = (0.52, 0.27, 0.57)
HEADER = (0.5026, 0.3273, 0.8458)


def _sig(pc):
    det = SimpleNamespace(shape=(60, 60), pc=np.asarray(pc, dtype=float), sample_tilt=70.0, tilt=0.0, azimuthal=0.0)
    return SimpleNamespace(detector=det)


def test_a_stash_entry_that_differs_in_case_only_is_the_fresh_entry():
    old = CalibrationStore()
    old.register("HiGainNi", _sig(HEADER), file_key="/d/x.h5")
    old.register_derived("odd", "HiGainNi")
    old.update_pc("HiGainNi", TYPED, source="manual")
    stash = old.snapshot(["HiGainNi", "odd"])
    fresh = CalibrationStore()
    fresh.register("higainni", _sig(HEADER), file_key="/d/x.h5")
    notice, discarded = fresh.restore(stash, skip="higainni")
    assert (notice, discarded) == ("user_pc_kept", None)
    assert fresh.get_pc("higainni") == pytest.approx(TYPED) and fresh.get_entry("higainni").pc_source == "manual"
    assert "HiGainNi" not in fresh
    assert fresh.get_entry("odd").parent_name == "higainni"


def test_other_names_are_not_taken_for_the_fresh_entry():
    old = CalibrationStore()
    old.register("HiGainNi", _sig(HEADER), file_key="/d/x.h5")
    old.update_pc("HiGainNi", TYPED, source="manual")
    fresh = CalibrationStore()
    fresh.register("higain", _sig(HEADER), file_key="/d/y.h5")
    fresh.restore(old.snapshot(["HiGainNi"]), skip="higain")
    assert fresh.get_pc("higain") == pytest.approx(HEADER)         # not touched
    assert "HiGainNi" in fresh                                     # inserted as it always was


@pytest.mark.skipif(sys.platform != "win32", reason="needs a file system that ignores case")
def test_a_file_with_a_copy_opened_under_another_spelling_has_no_twin_of_its_raw_dataset(tmp_path):
    """The stash holds the raw signal under the old spelling beside the copy: it must not come back as a
    dataset of its own, or choosing it shows a file without a PC while the PC lies on the fresh one."""
    from tests.apex_helpers import wipe_backend
    from backend.api.routes import ebsd_viewer as ev
    from backend.api.services import pc_calibration_apply as use
    from backend.api.services import pc_calibrations as cal
    base = Path(os.environ.get("ORIENTA_TEST_DATA") or Path(__file__).resolve().parents[1] / "Test_data")
    a, b = base / "HiGainNi.h5", base / "LoGainNi.h5"
    if not (a.exists() and b.exists()):
        pytest.skip("HiGainNi.h5 / LoGainNi.h5 are not in ORIENTA_TEST_DATA")
    from backend.api.main import app
    client = TestClient(app, client=("127.0.0.1", 5000))
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    wipe_backend()
    use.forget_all()
    try:
        assert client.post("/api/ebsd/load", json={"path": str(a)}).status_code == 200
        r = client.post("/api/ebsd/deepcopy", json={"name": "odd"})
        assert r.status_code == 200, r.text
        assert client.post("/api/ebsd/load", json={"path": str(b)}).status_code == 200
        respelled = str(a).replace("HiGainNi", "HIGAINNI")
        r = client.post("/api/ebsd/load", json={"path": respelled})
        assert r.status_code == 200, r.text
        assert "odd" in ev._raw_signals and "HIGAINNI" in ev._raw_signals      # the copy came back
        assert "HiGainNi" not in ev._raw_signals
    finally:
        wipe_backend()
        use.forget_all()
        cal.set_calibrations_dir_for_test(None)


@pytest.mark.skipif(sys.platform != "win32", reason="needs a file system that ignores case")
def test_loading_the_file_again_under_another_spelling_brings_its_pc_back(tmp_path):
    from tests.apex_helpers import wipe_backend
    from backend.api.services import pc_calibration_apply as use
    from backend.api.services import pc_calibrations as cal
    from backend.api.services.calibration_store import calibration_store
    base = Path(os.environ.get("ORIENTA_TEST_DATA") or Path(__file__).resolve().parents[1] / "Test_data")
    a, b = base / "HiGainNi.h5", base / "LoGainNi.h5"
    if not (a.exists() and b.exists()):
        pytest.skip("HiGainNi.h5 / LoGainNi.h5 are not in ORIENTA_TEST_DATA")
    from backend.api.main import app
    client = TestClient(app, client=("127.0.0.1", 5000))
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    wipe_backend()
    use.forget_all()
    try:
        assert client.post("/api/ebsd/load", json={"path": str(a)}).status_code == 200
        r = client.post("/api/pc/detector/update-pc", json={"pcx": TYPED[0], "pcy": TYPED[1], "pcz": TYPED[2]})
        assert r.status_code == 200, r.text
        assert client.post("/api/ebsd/load", json={"path": str(b)}).status_code == 200
        respelled = str(a).replace("HiGainNi", "HIGAINNI")
        assert respelled != str(a)
        r = client.post("/api/ebsd/load", json={"path": respelled})
        assert r.status_code == 200, r.text
        assert r.json()["dataset_name"] == "HIGAINNI"
        assert r.json()["pc"] == pytest.approx(TYPED) and r.json()["pc_notice"] == "user_pc_kept"
        assert calibration_store.get_pc("HIGAINNI") == pytest.approx(TYPED)
        assert "HiGainNi" not in calibration_store
    finally:
        wipe_backend()
        use.forget_all()
        cal.set_calibrations_dir_for_test(None)
