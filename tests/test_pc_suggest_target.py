"""``suggest`` names the file a calibration would be applied to: the file itself, or the file a copy was made from.

``copy-pc`` records on the entry of the TARGET FILE whose PC it took (``parent_name``), but that entry stays the entry
of its own file. Only a view of a file (a crop or a copy made in the viewer, ``is_view_of_parent``) hands the question
on to the file it views. Real files of the Test_data folder (``ORIENTA_TEST_DATA``), through the real routes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ORIENTA_ALLOWED_HOSTS", "testserver")

from tests.apex_helpers import wipe_backend  # noqa: E402
from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402

A_FILE, B_FILE = "HiGainNi.h5", "LoGainNi.h5"


def _data(name: str) -> Path:
    base = Path(os.environ.get("ORIENTA_TEST_DATA") or Path(__file__).resolve().parents[1] / "Test_data")
    path = base / name
    if not path.exists():
        pytest.skip(f"{name} is not in {base} (set ORIENTA_TEST_DATA to the folder that holds the test data)")
    return path


@pytest.fixture(autouse=True)
def clean(tmp_path):
    from backend.api.routes import pcrefinement
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")

    def wipe():
        wipe_backend()
        use.forget_all()
        pcrefinement._pc_before.clear()
        pcrefinement._sessions.clear()
        pcrefinement._parked_sessions.clear()
    wipe()
    yield
    wipe()
    cal.set_calibrations_dir_for_test(None)


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, client=("127.0.0.1", 5000))


def _post(client, url, body):
    r = client.post(url, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_a_file_that_took_the_pc_of_another_is_its_own_target(client):
    a, b = _data(A_FILE), _data(B_FILE)
    _post(client, "/api/ebsd/load", {"path": str(b)})
    ql = _post(client, "/api/batch-v2/quick-load", {"file_path": str(a)})
    out = _post(client, "/api/batch-v2/copy-pc", {
        "source_dataset": b.stem, "target_datasets": [ql["dataset_name"]],
        "source_file_path": str(b), "target_file_paths": [str(a)]})
    assert out["results"][0]["success"] is True, out
    from backend.api.services.calibration_store import calibration_store
    assert calibration_store.get_entry(ql["dataset_name"]).parent_name == b.stem, "the premise: copy-pc links the files"
    r = client.get("/api/pc-calibrations/suggest", params={"dataset": ql["dataset_name"], "all": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["target"] == ql["dataset_name"]


def test_a_crop_of_a_file_hands_the_question_on_to_its_file(client):
    a = _data(A_FILE)
    _post(client, "/api/ebsd/load", {"path": str(a)})
    _post(client, "/api/ebsd/crop", dict(shape="rect", row0=0, col0=0, rows=4, cols=4, name="odd"))
    r = client.get("/api/pc-calibrations/suggest", params={"dataset": "odd", "all": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["target"] == a.stem
