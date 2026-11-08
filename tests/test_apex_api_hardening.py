"""Carry-overs of the APEX load API: the EDS-only answer's PC origin, the measurement dict, an absolute
metadata path in the remembered pairing, and refusals that name files, never local folders."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edax_apex as apex  # noqa: E402
from backend.api.problem import CODE_HEADER  # noqa: E402
from tests.apex_helpers import (  # noqa: E402
    GUID_A, GUID_A_BYTES, make_apex_h5, make_up, measurement_spec, wipe_backend,
)

PRIVATE_FOLDER = "private_folder_name"          # a folder name that must never reach the user


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    yield
    wipe_backend()


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / PRIVATE_FOLDER
    f.mkdir()
    return f


def _assert_names_file_not_folder(text, name, tmp_path):
    assert name in text, text
    assert str(tmp_path) not in text and PRIVATE_FOLDER not in text, text
    assert os.path.dirname(str(tmp_path)) not in text, text


# ------------------------------------------------------ 1. the EDS-only answer

def test_the_eds_only_answer_carries_a_pc_origin_that_says_there_is_no_pattern_centre(tmp_path):
    """No patterns, so no PC at all. ``"missing"`` would say that kikuchipy's placeholder is in use
    (the vocabulary of ``pc_origin_of``); here nothing is in use, and ``/load`` of a file without a
    signal answers ``None`` (``_pc_state_for_response``)."""
    from backend.api.routes import ebsd_viewer
    answer = ebsd_viewer._load_eds_only_blocking(
        str(tmp_path / "eds_only.h5oina"), {"eds_only": True}, None, 0.0)
    assert "pc_origin" in answer
    assert answer["pc_origin"] is None and answer["pc_origin"] != "missing"
    assert answer["pc_defaulted"] is False and answer["scan_geometry"] is None


# ------------------------------------------------- 2. the measurement dict (API)

def test_a_measurement_dict_is_what_the_load_dialog_reads(folder):
    h5 = make_apex_h5(folder / "s.h5", [measurement_spec()])
    d = apex.list_measurements(h5)[0].to_dict()
    # written out by hand: the keys the load dialog and the pairing answer rely on
    assert set(d) == {
        "id", "name", "area", "group", "grid_type", "hex", "n_rows", "n_cols", "n_points",
        "step_x", "step_y", "sample_tilt", "elevation", "azimuth", "pc", "pc_convention",
        "beam_kv", "has_sem", "has_iq", "phases", "pattern_shape"}
    assert json.loads(json.dumps(d)) == d                           # plain data: it travels as JSON
    assert d["pc_convention"] == "tsl"
    assert d["pc"] == pytest.approx([0.5, 0.4, 0.6], abs=1e-6)       # the PC as written in the file
    assert all(type(v) is float for v in d["pc"])
    assert (d["id"], d["name"], d["area"], d["grid_type"], d["hex"]) == (
        GUID_A, "OIM Map 1", "Area 1", "HexGrid", True)
    assert (d["n_rows"], d["n_cols"], d["n_points"]) == (3, 5, 15)
    assert (d["sample_tilt"], d["elevation"], d["azimuth"], d["beam_kv"]) == (70.0, 5.0, 0.0, 20.0)
    assert d["step_x"] == 0.5 and d["step_y"] == pytest.approx(0.5 * 3 ** 0.5 / 2, abs=1e-6)
    assert (d["has_sem"], d["has_iq"]) == (True, True)
    assert d["phases"] == [
        {"index": 1, "name": "Phase A", "formula": "Phase A", "symmetry": 62,
         "laue_group": "6/mmm", "point_group": "6/mmm"},
        {"index": 2, "name": "Phase B", "formula": "Phase A", "symmetry": 43,
         "laue_group": "6/mmm", "point_group": "6/mmm"}]


# ----------------------------------------- 3. the remembered path is absolute

def test_a_remembered_metadata_path_is_absolute_so_a_switch_survives_a_new_working_directory(
        client, tmp_path, monkeypatch):
    h5 = make_apex_h5(tmp_path / "session.h5", [measurement_spec()])
    up = make_up(tmp_path / "scan.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    other_up = make_up(tmp_path / "other.up1", n=6, nx=3, ny=2, guid_bytes=None, version=3, hex_flag=0)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)
    r = client.post("/api/ebsd/load", json={
        "path": str(up), "apex": {"h5_path": "session.h5", "measurement_id": GUID_A}})   # relative
    assert r.status_code == 200, r.text
    stored = client.get("/api/ebsd/loaded-files").json()["files"][0]["apex"]["h5_path"]
    assert os.path.isabs(stored)
    assert Path(stored).resolve() == h5.resolve()
    assert client.post("/api/ebsd/load", json={"path": str(other_up)}).status_code == 200
    monkeypatch.chdir(tmp_path / "elsewhere")                         # the working directory moves
    back = client.post("/api/ebsd/switch-file", json={"path": str(up)})
    assert back.status_code == 200, back.text
    assert back.json()["scan_geometry"]["kind"] == "hex"


def test_the_batch_loader_remembers_the_pairing_it_found_in_the_folder(tmp_path):
    """``load_ebsd_file`` (batch indexing) is the second place a load registers a file."""
    from backend.api.routes import ebsd_viewer
    make_apex_h5(tmp_path / "session.h5", [measurement_spec()])
    up = make_up(tmp_path / "scan.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    assert ebsd_viewer.load_ebsd_file(str(up)) is True
    entry = ebsd_viewer._loaded_files[0]
    assert entry["apex"]["measurement_id"] == GUID_A and entry["apex"]["pairing"] == "guid"


# ------------------------------------------------ 4a. a pattern file's header

def _bad_header_file(path, kind):
    """UP files whose header ``read_up1_header`` refuses; each is 58+ bytes, with the GUID of A."""
    if kind == "version2":
        head = struct.pack("<4I", 2, 8, 8, 58)
    elif kind == "zero_size":
        head = struct.pack("<4I", 4, 0, 8, 58)
    else:
        raise AssertionError(kind)
    path.write_bytes(head + b"\x00" * 26 + GUID_A_BYTES + b"\x00" * 64)
    return path


def _truncated_v3(path):
    path.write_bytes(struct.pack("<4I", 3, 8, 8, 42) + b"\x00" * 14)     # 30 bytes: a v3 header is 42
    return path


@pytest.mark.parametrize("kind", ["version2", "zero_size", "truncated", "missing", "tiny"])
def test_a_pattern_file_with_an_unreadable_header_is_a_coded_refusal_that_names_only_the_file(
        folder, tmp_path, kind):
    h5 = make_apex_h5(folder / "s.h5", [measurement_spec()])
    m = apex.list_measurements(h5)[0]
    up = folder / "bad.up1"
    if kind in ("version2", "zero_size"):
        _bad_header_file(up, kind)
    elif kind == "truncated":
        _truncated_v3(up)
    elif kind == "tiny":
        up.write_bytes(b"\x00" * 10)
    with pytest.raises(apex.ApexError) as ei:
        apex.check_pairing(m, up)
    assert ei.value.code == "apexPatternFileUnreadable"
    _assert_names_file_not_folder(str(ei.value), "bad.up1", tmp_path)


@pytest.mark.parametrize("kind", ["version2", "zero_size", "truncated", "tiny", "suffix"])
def test_the_header_reader_itself_names_the_file_and_not_its_folder(folder, tmp_path, kind):
    """Also the path without APEX metadata (``resolve_up1_geometry``) reads headers with this."""
    from edax_up1 import read_up1_header
    up = folder / ("bad.txt" if kind == "suffix" else "bad.up1")
    if kind in ("version2", "zero_size"):
        _bad_header_file(up, kind)
    elif kind == "truncated":
        _truncated_v3(up)
    else:
        up.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError) as ei:
        read_up1_header(str(up))
    _assert_names_file_not_folder(str(ei.value), up.name, tmp_path)


@pytest.mark.parametrize("kind", ["version2", "zero_size"])
def test_the_routes_refuse_such_a_pattern_file_with_its_code_and_no_folder(client, folder, tmp_path, kind):
    h5 = make_apex_h5(folder / "s.h5", [measurement_spec()])
    up = _bad_header_file(folder / "bad.up1", kind)
    body = {"h5_path": str(h5), "measurement_id": GUID_A}
    check = client.post("/api/ebsd/apex/check-pairing", json={**body, "pattern_path": str(up)})
    load = client.post("/api/ebsd/load", json={"path": str(up), "apex": body})
    for r in (check, load):
        assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexPatternFileUnreadable"
        _assert_names_file_not_folder(r.json()["detail"], "bad.up1", tmp_path)


# --------------------------------------------- 4b. a square grid that cannot be

@pytest.mark.parametrize("step", [0.0, -1.0])
def test_a_square_measurement_with_a_bad_step_is_a_coded_refusal(folder, step):
    h5 = make_apex_h5(folder / "s.h5", [
        measurement_spec(grid="SqrGrid", n_rows=2, n_cols=4, step_x=step)])
    m = apex.list_measurements(h5)[0]
    with pytest.raises(apex.ApexError) as ei:             # not the bare ValueError of ScanGeometry
        apex.geometry_from_apex(m)
    assert ei.value.code == "apexInvalidGrid"
    assert "Area 1" in str(ei.value) and "OIM Map 1" in str(ei.value)


def test_the_load_route_answers_a_bad_square_step_with_that_code(client, folder):
    h5 = make_apex_h5(folder / "s.h5", [
        measurement_spec(grid="SqrGrid", n_rows=2, n_cols=4, step_x=0.0)])
    up = make_up(folder / "p.up1", n=8, nx=4, ny=2, hex_flag=0, guid_bytes=GUID_A_BYTES)
    r = client.post("/api/ebsd/load", json={
        "path": str(up), "apex": {"h5_path": str(h5), "measurement_id": GUID_A}})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexInvalidGrid"


# ---------------------------------------- 4c. h5py's own text in apexUnreadable

def test_an_unreadable_metadata_file_keeps_the_exception_type_and_drops_the_folder(folder, tmp_path):
    gone = folder / "missing.h5"
    with pytest.raises(apex.ApexError) as ei:
        apex.list_measurements(gone)
    assert ei.value.code == "apexUnreadable"
    msg = str(ei.value)
    _assert_names_file_not_folder(msg, "missing.h5", tmp_path)
    assert "FileNotFoundError" in msg                         # h5py's type for a file that is not there


def test_the_preview_route_does_not_leak_the_folder_of_an_unreadable_file(client, folder, tmp_path):
    r = client.get("/api/ebsd/apex/preview", params={
        "path": str(folder / "missing.h5"), "measurement": GUID_A})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexUnreadable"
    _assert_names_file_not_folder(r.json()["detail"], "missing.h5", tmp_path)


def test_without_paths_cuts_every_spelling_of_a_file_and_its_folder(tmp_path):
    f = tmp_path / PRIVATE_FOLDER / "x y.h5"
    for spelling in (str(f), f.as_posix(), str(f).replace("\\", "\\\\"), f.as_posix().replace("/", "\\")):
        out = apex.without_paths(f"cannot open {spelling!s} (errno 2)", f)
        assert out == "cannot open x y.h5 (errno 2)", (spelling, out)
    sibling = f"{tmp_path / PRIVATE_FOLDER}{os.sep}Calibration (1,2).bmp"
    assert apex.without_paths(f"no such file: '{sibling}'", f) == "no such file: 'Calibration (1,2).bmp'"


# ------------------------------------------------- 4d. NORDIF calibration set

_SETTING = [
    "[NORDIF]\t\t", "Software version\t3.1.2\t", "\t\t",
    "[Microscope]\t\t", "Manufacturer\tMakerCo\t", "Model\tModel-1\t", "Magnification\t200\t#",
    "Scan direction\tDirect\t", "Accelerating voltage\t20\tkV", "Working distance\t24.7\tmm",
    "Tilt angle\t70\t\xb0", "\t\t",
    "[Signal voltages]\t\t", "Minimum\t0.0\tV", "Maximum\t1.0\tV", "\t\t",
    "[Deflection voltages]\t\t", "Minimum\t-5.5\tV", "Maximum\t5.5\tV", "\t\t",
    "[Electron image]\t\t", "Frame rate\t0.25\tfps", "Resolution\t1000x1000\tpx", "Rotation\t0\t\xb0",
    "Flip x-axis\tFalse\t", "Flip y-axis\tFalse\t", "Calibration factor\t7273\t\xb5m/V",
    "Tilt axis\tx-axis\t", "\t\t",
    "[Aspect ratio]\t\t", "X-axis\t1.000\t", "Y-axis\t1.000\t", "\t\t",
    "[EBSD detector]\t\t", "Model\tCam-1\t", "Port position\t90\t", "Jumbo frames\tFalse\t", "\t\t",
    "[Detector angles]\t\t", "Euler 1\t0\t\xb0", "Euler 2\t0\t\xb0", "Euler 3\t0\t\xb0",
    "Azimuthal\t0\t\xb0", "Elevation\t0\t\xb0", "\t\t",
    "[Acquisition settings]\t\t", "Frame rate\t202\tfps", "Resolution\t60x60\tpx",
    "Exposure time\t3500\t\xb5s", "Gain\t0\t", "\t\t",
    "[Calibration settings]\t\t", "Frame rate\t10\tfps", "Resolution\t60x60\tpx",
    "Exposure time\t99950\t\xb5s", "Gain\t8\t", "\t\t",
    "[Specimen]\t\t", "Name\tSample\t", "Mounting\t1. ND||EB TD||TA\t", "\t\t",
    "[Phase 1]\t\t", "Name\t\t", "Pearson S.T.\t\t", "IT\t\t", "\t\t",
    "[Phase 2]\t\t", "Name\t\t", "Pearson S.T.\t\t", "IT\t\t", "\t\t",
    "[Region of interest]\t\t", "\t\t",
    "[Area]\t\t", "Top\t10.000 (25)\t\xb5m (px)", "Left\t10.000 (25)\t\xb5m (px)",
    "Width\t30.000 (75)\t\xb5m (px)", "Height\t20.000 (50)\t\xb5m (px)",
    "Step size\t1.500\t\xb5m", "Number of samples\t13x20\t#", "Scan time\t00:00:10\t", "\t\t",
    "[Points of interest]\t\t", "\t\t",
    "[Acquisition patterns]\t\t", "Acquisition (5,6)\t5,6\tpx", "\t\t",
    "[Calibration patterns]\t\t", "Calibration (425,447)\t425,447\tpx", "Calibration (294,532)\t294,532\tpx",
]


def _nordif_set(folder):
    """A NORDIF Setting.txt in the layout of the public kikuchipy-paper files (made-up values) that
    lists two calibration patterns, and no .bmp files beside it."""
    setting = folder / "Setting.txt"
    setting.write_bytes(("\r\n".join(_SETTING) + "\r\n").encode("latin-1"))
    return setting


def test_the_nordif_setting_used_here_is_one_kikuchipy_can_read_up_to_the_images(folder):
    """If this fails the fixture, not the refusal, is wrong: kikuchipy must get as far as the bitmaps."""
    from kikuchipy.io.plugins.nordif_calibration_patterns._api import file_reader
    with pytest.raises(FileNotFoundError):
        file_reader(str(_nordif_set(folder)))


def test_a_missing_calibration_bitmap_is_named_by_file_name_only(folder, tmp_path):
    import safe_loader
    setting = _nordif_set(folder)
    with pytest.raises(ValueError) as ei:
        safe_loader.load_ebsd_safe(str(setting), verbose=False)
    msg = str(ei.value)
    assert "Calibration (425,447).bmp" in msg                 # the first bitmap kikuchipy looks for
    assert "FileNotFoundError" in msg                         # the type of kikuchipy's error stays
    assert str(tmp_path) not in msg and PRIVATE_FOLDER not in msg, msg
    assert ei.value.code == "nordifCalibrationUnreadable"


def test_the_load_route_answers_a_missing_nordif_bitmap_with_its_code_and_no_folder(client, folder, tmp_path):
    setting = _nordif_set(folder)
    r = client.post("/api/ebsd/load", json={"path": str(setting)})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "nordifCalibrationUnreadable"
    assert "Calibration (425,447).bmp" in r.json()["detail"]
    assert str(tmp_path) not in r.json()["detail"] and PRIVATE_FOLDER not in r.json()["detail"]


def test_a_nordif_setting_that_does_not_exist_is_named_by_file_name_only(client, folder, tmp_path):
    import safe_loader
    gone = folder / "Setting.txt"
    with pytest.raises(ValueError) as ei:
        safe_loader.load_ebsd_safe(str(gone), verbose=False)
    assert ei.value.code == "nordifSettingMissing"
    _assert_names_file_not_folder(str(ei.value), "Setting.txt", tmp_path)
    r = client.post("/api/ebsd/load", json={"path": str(gone)})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "nordifSettingMissing"
