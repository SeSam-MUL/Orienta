"""APEX load API, pairing: a stored pairing follows what the file last loaded with,
a moved metadata file never blocks a switch, check-pairing reads only pattern files, and a
pattern file with a broken header is refused with its code also without metadata."""
from __future__ import annotations

import logging
import shutil
import struct
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edax_apex as apex  # noqa: E402
from backend.api.problem import CODE_HEADER  # noqa: E402
from tests.apex_helpers import (  # noqa: E402
    GUID_A, GUID_A_BYTES, GUID_B, make_apex_h5, make_up, measurement_spec, wipe_backend,
)

PRIVATE_FOLDER = "private_folder_name"


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    yield
    wipe_backend()


def _session(folder):
    return make_apex_h5(folder / "session.h5", [
        measurement_spec(id=GUID_A, area="Area 1", n_rows=3, n_cols=5),
        measurement_spec(id=GUID_B, area="Area 2", grid="SqrGrid", n_rows=2, n_cols=4, step_x=1.0)])


def _setup(tmp_path):
    """Metadata in ``meta``, two square pattern files without metadata beside them in ``scans``."""
    (tmp_path / "meta").mkdir()
    (tmp_path / "scans").mkdir()
    h5 = _session(tmp_path / "meta")
    sq = make_up(tmp_path / "scans" / "sq.up1", guid_bytes=None, version=3, hex_flag=0, n=8, nx=4, ny=2)
    other = make_up(tmp_path / "scans" / "other.up1", guid_bytes=None, version=3, hex_flag=0,
                    n=6, nx=3, ny=2)
    return h5, sq, other


def _pair(client, sq, h5):
    r = client.post("/api/ebsd/load", json={
        "path": str(sq), "apex": {"h5_path": str(h5), "measurement_id": GUID_B}})
    assert r.status_code == 200, r.text
    return r


def _entry(client, name):
    return next(f for f in client.get("/api/ebsd/loaded-files").json()["files"] if f["name"] == name)


# ---------------------------------------------------------------- r1: forgetting

def test_a_load_without_apex_forgets_the_stored_pairing(client, tmp_path):
    h5, sq, _ = _setup(tmp_path)
    _pair(client, sq, h5)
    assert _entry(client, "sq")["apex"]["measurement_id"] == GUID_B
    again = client.post("/api/ebsd/load", json={"path": str(sq)})          # path only, no metadata beside
    assert again.status_code == 200, again.text
    assert again.json()["apex"] is None
    assert "apex" not in _entry(client, "sq")


def test_a_load_that_pairs_by_guid_stores_that_pairing(client, tmp_path):
    h5 = _session(tmp_path)
    up = make_up(tmp_path / "scan.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    assert client.post("/api/ebsd/load", json={"path": str(up)}).status_code == 200
    stored = _entry(client, "scan")["apex"]
    assert (stored["measurement_id"], stored["pairing"]) == (GUID_A, "guid")
    assert Path(stored["h5_path"]).resolve() == h5.resolve()


# ------------------------------------------------- r1: a metadata file that is gone

def test_a_switch_whose_metadata_file_was_moved_loads_path_only_and_says_so(client, tmp_path, caplog):
    h5, sq, other = _setup(tmp_path)
    _pair(client, sq, h5)
    assert client.post("/api/ebsd/load", json={"path": str(other)}).status_code == 200
    shutil.move(str(h5), str(tmp_path / "moved.h5"))                    # e.g. an unplugged drive
    with caplog.at_level(logging.WARNING):
        back = client.post("/api/ebsd/switch-file", json={"path": str(sq)})
    assert back.status_code == 200, back.text
    body = back.json()
    assert body["apex_pairing_dropped"] is True and body["apex"] is None
    assert "apex" not in _entry(client, "sq")
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
              and "sq.up1" in r.getMessage() and "pairing" in r.getMessage().lower()]
    assert warned, [r.getMessage() for r in caplog.records]
    assert all(str(tmp_path) not in w for w in warned)


def test_a_switch_with_the_metadata_still_there_keeps_the_pairing(client, tmp_path):
    h5, sq, other = _setup(tmp_path)
    _pair(client, sq, h5)
    assert client.post("/api/ebsd/load", json={"path": str(other)}).status_code == 200
    back = client.post("/api/ebsd/switch-file", json={"path": str(sq)})
    assert back.status_code == 200, back.text
    assert back.json()["apex_pairing_dropped"] is False
    assert back.json()["apex"]["measurement_id"] == GUID_B
    assert _entry(client, "sq")["apex"]["measurement_id"] == GUID_B


# --------------------------------------------------- r2: check-pairing reads patterns

def test_check_pairing_refuses_a_file_that_is_not_a_pattern_file_without_echoing_its_bytes(
        client, tmp_path):
    h5 = _session(tmp_path)
    content = bytes(range(100, 116))                                   # 16 bytes at offset 42
    notes = tmp_path / "notes.txt"
    notes.write_bytes(struct.pack("<4I", 4, 8, 8, 58) + b"\x00" * 26 + content + b"\x00" * 64)
    echoed = str(uuid.UUID(bytes_le=content))                          # what a GUID read would print
    m = apex.read_measurement(h5, GUID_A)
    with pytest.raises(apex.ApexError) as ei:
        apex.check_pairing(m, notes)
    assert ei.value.code == "apexPatternFileUnreadable"
    assert "notes.txt" in str(ei.value) and echoed not in str(ei.value) and str(tmp_path) not in str(ei.value)
    r = client.post("/api/ebsd/apex/check-pairing", json={
        "h5_path": str(h5), "measurement_id": GUID_A, "pattern_path": str(notes)})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexPatternFileUnreadable"
    assert echoed not in r.json()["detail"] and str(tmp_path) not in r.json()["detail"]


def test_a_path_with_a_nul_is_a_coded_400_not_a_500(client, tmp_path):
    h5 = _session(tmp_path)
    nul = "scan\x00.up1"
    cases = [
        client.post("/api/ebsd/apex/check-pairing", json={
            "h5_path": str(h5), "measurement_id": GUID_A, "pattern_path": str(tmp_path / nul)}),
        client.post("/api/ebsd/apex/check-pairing", json={
            "h5_path": str(tmp_path / "s\x00.h5"), "measurement_id": GUID_A,
            "pattern_path": str(tmp_path / "p.up1")}),
        client.get("/api/ebsd/apex/preview", params={
            "path": str(tmp_path / "s\x00.h5"), "measurement": GUID_A}),
    ]
    for r in cases:
        assert r.status_code == 400, (r.status_code, r.text)
        assert r.headers[CODE_HEADER.lower()] == "apexBadPath"
    r = client.get("/api/ebsd/inspect", params={"path": str(tmp_path / "s\x00.h5")})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexBadPath"


# ------------------------------------------- r4: a broken header without metadata

@pytest.mark.parametrize("kind", ["truncated", "tiny"])
def test_a_broken_pattern_file_without_metadata_is_refused_with_its_code(client, tmp_path, kind):
    folder = tmp_path / PRIVATE_FOLDER
    folder.mkdir()
    up = folder / "bad.up1"
    if kind == "truncated":
        up.write_bytes(struct.pack("<4I", 3, 8, 8, 42) + b"\x00" * 14)    # a v3 header is 42 bytes
    else:
        up.write_bytes(b"\x00" * 10)
    r = client.post("/api/ebsd/load", json={"path": str(up)})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexPatternFileUnreadable"
    assert "bad.up1" in r.json()["detail"]
    assert str(tmp_path) not in r.json()["detail"] and PRIVATE_FOLDER not in r.json()["detail"]


# ------------------------------- a metadata file that changed under a stored pairing

def _paired_and_away(client, tmp_path):
    h5, sq, other = _setup(tmp_path)
    _pair(client, sq, h5)
    assert client.post("/api/ebsd/load", json={"path": str(other)}).status_code == 200
    return h5, sq


def _release_the_loaded_signals():
    """Windows will not rewrite a file whose memory map is still held by a loaded (or stashed) signal;
    forget the signals and keep the loaded-files list, which is what a later switch reads."""
    import gc
    from backend.api.routes import ebsd_viewer
    from backend.api.services import crop_window, scan_geometry
    from backend.api.services.calibration_store import calibration_store
    ebsd_viewer._raw_signals.clear()
    ebsd_viewer._registry_by_file.clear()
    ebsd_viewer._ebsd_signal = None
    crop_window.clear_all()
    scan_geometry.clear_all()
    calibration_store.clear()
    gc.collect()


def _switch_back_dropped(client, sq):
    back = client.post("/api/ebsd/switch-file", json={"path": str(sq)})
    assert back.status_code == 200, back.text
    assert back.json()["apex_pairing_dropped"] is True and back.json()["apex"] is None
    assert "apex" not in _entry(client, "sq")


def test_a_metadata_file_rewritten_without_the_measurement_drops_the_pairing(client, tmp_path):
    h5, sq = _paired_and_away(client, tmp_path)
    make_apex_h5(h5, [measurement_spec(id=GUID_A, area="Area 1", n_rows=3, n_cols=5)])   # B is gone
    _switch_back_dropped(client, sq)


def test_a_metadata_file_that_can_no_longer_be_read_drops_the_pairing(client, tmp_path):
    h5, sq = _paired_and_away(client, tmp_path)
    h5.write_bytes(b"this is not an HDF5 file" * 10)
    _switch_back_dropped(client, sq)


def test_a_pattern_file_that_now_names_another_measurement_is_refused_not_dropped(client, tmp_path):
    h5, sq = _paired_and_away(client, tmp_path)
    _release_the_loaded_signals()
    make_up(sq, guid_bytes=GUID_A_BYTES, hex_flag=0, n=8, nx=4, ny=2)     # the file changed: it is A's
    back = client.post("/api/ebsd/switch-file", json={"path": str(sq)})
    assert back.status_code == 400 and back.headers[CODE_HEADER.lower()] == "apexGuidMismatch"
    assert _entry(client, "sq")["apex"]["measurement_id"] == GUID_B       # nothing was forgotten


def test_a_pattern_file_with_another_number_of_patterns_is_refused_not_dropped(client, tmp_path):
    h5, sq = _paired_and_away(client, tmp_path)
    _release_the_loaded_signals()
    make_up(sq, guid_bytes=None, version=3, hex_flag=0, n=6, nx=3, ny=2)  # 6 patterns, B is 2 x 4 = 8
    back = client.post("/api/ebsd/switch-file", json={"path": str(sq)})
    assert back.status_code == 400 and back.headers[CODE_HEADER.lower()] == "apexPatternCountMismatch"
    assert _entry(client, "sq")["apex"]["measurement_id"] == GUID_B
