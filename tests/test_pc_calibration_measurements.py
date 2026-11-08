"""One pattern file, two measurements: the calibration reaches the measurement the file was last loaded as.

A pattern file whose header names no measurement (a GUID-less .up1) is paired by the user with a
measurement of an APEX metadata file, and the same file can be paired with another one later. Both are
loaded as the same dataset name; each has a stash of its own. Through the real routes on small synthetic
files, built as in ``test_apex_measurement_identity``.

By hand: the metadata file holds measurement 1 (PC of the file (0.5, 0.4, 0.6), which on a square 8 x 8
detector is (0.5, 0.6, 0.6) in Bruker's convention) and measurement 2 (PC (0.45, 0.35, 0.7), so
(0.45, 0.65, 0.7)); both want 15 patterns, so the one pattern file fits both.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402
from tests.apex_helpers import (  # noqa: E402
    GUID_A, GUID_B, make_apex_h5, make_up, measurement_spec, wipe_backend,
)

PC_1 = (0.5, 0.6, 0.6)
PC_2 = (0.45, 0.65, 0.7)
TYPED_OTHER = (0.52, 0.31, 0.58)       # typed on the other file, saved as the calibration
TYPED_Y = (0.48, 0.33, 0.62)          # typed on the other file afterwards, saved as the second calibration
NAME = "noguid"


@pytest.fixture(autouse=True)
def clean(tmp_path):
    from backend.api.routes import ebsd_viewer, pcrefinement
    from backend.api.services import variant_registry

    def wipe():
        wipe_backend()
        use.forget_all()
        ebsd_viewer._pc_load_notice.clear()
        variant_registry.clear_all()
        pcrefinement._pc_before.clear()
        pcrefinement._sessions.clear()
        pcrefinement._parked_sessions.clear()
    cal.set_calibrations_dir_for_test(tmp_path / "Calibrations")
    wipe()
    yield
    wipe()
    cal.set_calibrations_dir_for_test(None)


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture
def files(tmp_path):
    d = tmp_path / "f"
    d.mkdir()
    h5 = make_apex_h5(d / "s.h5", [
        measurement_spec(n_rows=3, n_cols=5, step_x=0.5),                                   # measurement 1
        measurement_spec(id=GUID_B, name="OIM Map 2", pc=(0.45, 0.35, 0.7), grid="HexGrid",
                         n_rows=5, n_cols=3, step_x=0.5),                                   # measurement 2
    ])
    return SimpleNamespace(
        h5=str(h5),
        up=str(make_up(d / f"{NAME}.up1", n=15, guid_bytes=None, version=3, nx=5, ny=3, hex_flag=1)),
        other=str(make_up(d / "other.up1", n=6, guid_bytes=None, version=3, nx=3, ny=2, hex_flag=0)),
    )


def _load(client, path, mid=None, h5=None):
    body = {"path": path}
    if mid:
        body["apex"] = {"h5_path": h5, "measurement_id": mid}
    r = client.post("/api/ebsd/load", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _type_pc(client, pc):
    r = client.post("/api/pc/detector/update-pc", json={"pcx": pc[0], "pcy": pc[1], "pcz": pc[2]})
    assert r.status_code == 200, r.text
    assert r.json()["pc_stored"] is True


def _store():
    from backend.api.services.calibration_store import calibration_store
    return calibration_store


@pytest.fixture
def world(client, files):
    """Measurement 1 of the file, then measurement 2, then the other file (the calibration's source, with a PC typed
    on it): the file is left behind as measurement 2, and measurement 1 sits in the stash of its own."""
    _load(client, files.up, GUID_A, files.h5)
    _load(client, files.up, GUID_B, files.h5)
    _load(client, files.other)
    _type_pc(client, TYPED_OTHER)
    r = client.post("/api/pc-calibrations", json={"name": "Up REM", "dataset": "other"})
    assert r.status_code == 200, r.text
    return client, r.json()["id"], files


def _rows(client, cal_id):
    r = client.get(f"/api/pc-calibrations/{cal_id}/candidates")
    assert r.status_code == 200, r.text
    return r.json()["candidates"]


def test_a_file_left_behind_is_one_row_the_measurement_it_was_last_loaded_as(world):
    client, cal_id, files = world
    rows = [r for r in _rows(client, cal_id) if r["dataset"].startswith(NAME)]
    assert len(rows) == 1, rows
    assert rows[0]["stashed"] is True and rows[0]["path"] == files.up
    assert rows[0]["pc"] == pytest.approx(PC_2, abs=1e-6)


def test_applying_to_the_file_left_behind_writes_the_measurement_it_was_last_loaded_as(world):
    client, cal_id, files = world
    r = client.post(f"/api/pc-calibrations/{cal_id}/apply", json={"datasets": [NAME], "accept_mismatch": [NAME]})
    assert r.status_code == 200, r.text
    assert [(a["dataset"], a["status"]) for a in r.json()["results"]] == [(NAME, "applied")]
    back = client.post("/api/ebsd/switch-file", json={"path": files.up})
    assert back.status_code == 200, back.text
    assert back.json()["pc"] == pytest.approx(TYPED_OTHER)                 # measurement 2 came back with it
    assert use.current_ref(NAME, _store())["version"] == 1
    # measurement 1 was not written
    _load(client, files.up, GUID_A, files.h5)
    assert _store().get_pc(NAME) == pytest.approx(PC_1, abs=1e-6)


def test_the_calibration_a_measurement_was_given_is_not_overwritten_by_its_namesake(world):
    """Measurement 2 (left behind) is given calibration Y, then measurement 1 is opened and given calibration X.
    What each was given is kept per measurement: opened again, measurement 2 still runs on Y."""
    client, x_id, files = world
    _type_pc(client, TYPED_Y)
    r = client.post("/api/pc-calibrations", json={"name": "Up Y", "dataset": "other"})
    assert r.status_code == 200, r.text
    y_id = r.json()["id"]
    r = client.post(f"/api/pc-calibrations/{y_id}/apply", json={"datasets": [NAME], "accept_mismatch": [NAME]})
    assert r.json()["results"][0]["status"] == "applied", r.text
    _load(client, files.up, GUID_A, files.h5)                                # measurement 1 is open now
    r = client.post(f"/api/pc-calibrations/{x_id}/apply", json={"datasets": [NAME], "accept_mismatch": [NAME]})
    assert r.json()["results"][0]["status"] == "applied", r.text
    assert use.current_ref(NAME, _store())["id"] == x_id
    _load(client, files.up, GUID_B, files.h5)                                # measurement 2 again
    assert _store().get_pc(NAME) == pytest.approx(TYPED_Y)
    ref = use.current_ref(NAME, _store())
    assert ref is not None and ref["id"] == y_id and ref["version"] == 1
