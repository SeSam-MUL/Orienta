"""The load API for EDAX APEX measurements: inspect, preview, pairing check, load, switch."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.problem import CODE_HEADER  # noqa: E402
from tests.apex_helpers import (  # noqa: E402
    GUID_A, GUID_A_BYTES, GUID_B, GUID_B_BYTES, make_apex_h5, make_up, measurement_spec, wipe_backend,
)


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    yield
    wipe_backend()


def _session(folder, **a_over):
    """session.h5 with A (hexagonal 3 x 5) and B (square 2 x 4)."""
    return make_apex_h5(folder / "session.h5", [
        measurement_spec(id=GUID_A, area="Area 1", n_rows=3, n_cols=5, **a_over),
        measurement_spec(id=GUID_B, area="Area 2", grid="SqrGrid", n_rows=2, n_cols=4, step_x=1.0),
    ])


def _inspect(client, path):
    r = client.get("/api/ebsd/inspect", params={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


# --------------------------------------------------------------------- inspect

def test_inspect_tells_a_plain_file_from_an_apex_file(client, tmp_path):
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_group("1/EBSD/Data")
    assert _inspect(client, other) == {"kind": "plain"}
    assert _inspect(client, tmp_path / "x.up1") == {"kind": "plain"}
    assert _inspect(client, _session(tmp_path))["kind"] == "apex"
    # a missing .h5 is no longer "plain" (the dialog would never appear): a coded refusal
    r = client.get("/api/ebsd/inspect", params={"path": str(tmp_path / "missing.h5")})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexFileNotFound"


def test_inspect_lists_the_measurements_with_the_pattern_file_found_for_each(client, tmp_path):
    h5 = _session(tmp_path)
    make_up(tmp_path / "called something else.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    out = _inspect(client, h5)
    a, b = out["measurements"]
    assert (a["area"], a["hex"], a["n_rows"], a["n_cols"]) == ("Area 1", True, 3, 5)
    assert a["pc"] == pytest.approx([0.5, 0.4, 0.6], abs=1e-6) and a["pc_convention"] == "tsl"
    assert a["consistent"] is True and a["has_sem"] and a["has_iq"]
    assert [(f["name"], f["matches_grid"]) for f in a["pattern_files"]] == [
        ("called something else.up1", True)]
    assert (b["hex"], b["pattern_files"]) == (False, [])         # no file names measurement B


def test_inspect_flags_a_pattern_file_with_the_wrong_number_of_patterns(client, tmp_path):
    h5 = _session(tmp_path)
    make_up(tmp_path / "p.up1", n=14, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    f = _inspect(client, h5)["measurements"][0]["pattern_files"][0]
    assert (f["matches_grid"], f["problem"]) == (False, "apexPatternCountMismatch")


# --------------------------------------------------------------------- preview

@pytest.mark.parametrize("kind", ["sem", "iq"])
def test_a_preview_is_a_png(client, tmp_path, kind):
    h5 = _session(tmp_path)
    r = client.get("/api/ebsd/apex/preview", params={"path": str(h5), "measurement": GUID_A, "kind": kind})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])    # the PNG signature


def test_a_preview_of_an_unknown_measurement_is_a_coded_refusal(client, tmp_path):
    h5 = _session(tmp_path)
    r = client.get("/api/ebsd/apex/preview", params={
        "path": str(h5), "measurement": "11111111-1111-1111-1111-111111111111"})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexMeasurementNotFound"


def test_a_failing_preview_is_still_a_coded_answer_but_leaves_a_traceback(client, tmp_path, monkeypatch, caplog):
    """The catch-all answers ``apexUnreadable`` for any failure (the file may have changed under us);
    a bug in the PNG code must not hide behind that answer, so the cause is logged with its traceback."""
    import logging
    import edax_apex

    def broken(_measurement, _kind):
        raise RuntimeError("the PNG writer fell over")

    monkeypatch.setattr(edax_apex, "preview_png", broken)
    h5 = _session(tmp_path)
    with caplog.at_level(logging.ERROR):
        r = client.get("/api/ebsd/apex/preview", params={"path": str(h5), "measurement": GUID_A})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexUnreadable"
    logged = [rec for rec in caplog.records if rec.levelno == logging.ERROR and rec.exc_info
              and "the PNG writer fell over" in str(rec.exc_info[1])]
    assert len(logged) == 1


# ---------------------------------------------------------------- check-pairing

def test_check_pairing_accepts_the_right_file_and_refuses_the_wrong_one(client, tmp_path):
    h5 = _session(tmp_path)
    up = make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    body = {"h5_path": str(h5), "measurement_id": GUID_A, "pattern_path": str(up)}
    ok = client.post("/api/ebsd/apex/check-pairing", json=body)
    assert ok.status_code == 200 and ok.json()["pairing"] == "guid"
    bad = client.post("/api/ebsd/apex/check-pairing", json={**body, "measurement_id": GUID_B})
    assert bad.status_code == 400 and bad.headers[CODE_HEADER.lower()] == "apexGuidMismatch"


# ------------------------------------------------------------------------ load

def test_an_apex_file_cannot_be_loaded_as_if_it_held_patterns(client, tmp_path):
    r = client.post("/api/ebsd/load", json={"path": str(_session(tmp_path))})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexNeedsMeasurement"


def test_a_pairing_found_in_the_folder_is_reported_and_remembered(client, tmp_path):
    _session(tmp_path)
    up = make_up(tmp_path / "scan.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    r = client.post("/api/ebsd/load", json={"path": str(up)})
    assert r.status_code == 200, r.text
    assert r.json()["apex"]["measurement_id"] == GUID_A and r.json()["apex"]["pairing"] == "guid"
    entry = client.get("/api/ebsd/loaded-files").json()["files"][0]
    assert entry["apex"]["measurement_id"] == GUID_A


def test_an_explicit_pairing_with_a_metadata_file_elsewhere_loads_and_survives_a_switch(client, tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "scans").mkdir()
    h5 = _session(tmp_path / "meta")
    hex_up = make_up(tmp_path / "scans" / "scan_hex.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    sq_up = make_up(tmp_path / "scans" / "scan_sq.up1", guid_bytes=None, version=3, hex_flag=0,
                    n=6, nx=3, ny=2)
    # on its own the hexagonal file is refused: the metadata is not in its folder
    refused = client.post("/api/ebsd/load", json={"path": str(hex_up)})
    assert refused.status_code == 400 and refused.headers[CODE_HEADER.lower()] == "edaxHexUpUnsupported"
    # ... the user names it
    r = client.post("/api/ebsd/load", json={
        "path": str(hex_up), "apex": {"h5_path": str(h5), "measurement_id": GUID_A}})
    assert r.status_code == 200, r.text
    assert r.json()["scan_geometry"]["kind"] == "hex" and r.json()["pc_origin"] == "apex"
    # another file, then back: the stored pairing is used, nobody asks again
    assert client.post("/api/ebsd/load", json={"path": str(sq_up)}).status_code == 200
    back = client.post("/api/ebsd/switch-file", json={"path": str(hex_up)})
    assert back.status_code == 200, back.text
    assert back.json()["scan_geometry"]["kind"] == "hex"


def test_a_load_with_the_wrong_measurement_is_refused_with_its_code(client, tmp_path):
    h5 = _session(tmp_path)
    up = make_up(tmp_path / "p.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    r = client.post("/api/ebsd/load", json={
        "path": str(up), "apex": {"h5_path": str(h5), "measurement_id": GUID_B}})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexGuidMismatch"
    assert client.get("/api/ebsd/loaded-files").json()["files"] == []      # nothing was registered


def test_two_measurements_of_one_file_load_one_after_the_other_into_the_list(client, tmp_path):
    h5 = _session(tmp_path)
    up_a = make_up(tmp_path / "a.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    up_b = make_up(tmp_path / "b.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=GUID_B_BYTES)
    for up, mid in ((up_a, GUID_A), (up_b, GUID_B)):
        r = client.post("/api/ebsd/load", json={
            "path": str(up), "apex": {"h5_path": str(h5), "measurement_id": mid}})
        assert r.status_code == 200, r.text
    files = client.get("/api/ebsd/loaded-files").json()["files"]
    assert [f["name"] for f in files] == ["a", "b"]
    assert [f["apex"]["measurement_id"] for f in files] == [GUID_A, GUID_B]
    assert [f["active"] for f in files] == [False, True]
