"""APEX load API, batch loader and preview: the batch loader keeps a remembered pairing, a switch says why it
dropped one, an IQ preview without IQ is refused by name, and the answer shapes are pinned."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.problem import CODE_HEADER  # noqa: E402
from tests.apex_helpers import (  # noqa: E402
    GUID_A, GUID_A_BYTES, GUID_B, make_apex_h5, make_up, measurement_spec, wipe_backend,
)
from tests.test_apex_load_pairing import _release_the_loaded_signals  # noqa: E402


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
    """Metadata in ``meta``; in ``scans`` (no metadata beside them) a hexagonal file of measurement
    A, a square file of B without a GUID, and another square file."""
    (tmp_path / "meta").mkdir()
    (tmp_path / "scans").mkdir()
    h5 = _session(tmp_path / "meta")
    hex_up = make_up(tmp_path / "scans" / "scan_hex.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    sq = make_up(tmp_path / "scans" / "sq.up1", guid_bytes=None, version=3, hex_flag=0, n=8, nx=4, ny=2)
    other = make_up(tmp_path / "scans" / "other.up1", guid_bytes=None, version=3, hex_flag=0,
                    n=6, nx=3, ny=2)
    return h5, hex_up, sq, other


def _pair(client, up, h5, mid):
    r = client.post("/api/ebsd/load", json={
        "path": str(up), "apex": {"h5_path": str(h5), "measurement_id": mid}})
    assert r.status_code == 200, r.text
    return r


def _entry(client, name):
    return next(f for f in client.get("/api/ebsd/loaded-files").json()["files"] if f["name"] == name)


# ------------------------------------------------------------ s1: the batch loader

def test_the_batch_loader_loads_a_paired_hex_file_with_its_stored_pairing(client, tmp_path):
    from backend.api.routes import ebsd_viewer
    from backend.api.services import scan_geometry
    from backend.api.services.calibration_store import calibration_store
    h5, hex_up, _, _ = _setup(tmp_path)
    _pair(client, hex_up, h5, GUID_A)
    assert ebsd_viewer.load_ebsd_file(str(hex_up)) is True            # path-only it would be refused
    g = scan_geometry.geometry_for("scan_hex")
    assert (g.kind, g.n_rows, g.n_cols) == ("hex", 3, 5)
    assert calibration_store.get_entry("scan_hex").pc_origin == "apex"
    stored = _entry(client, "scan_hex")["apex"]
    assert stored["measurement_id"] == GUID_A and Path(stored["h5_path"]).resolve() == h5.resolve()


def test_the_batch_loader_drops_a_stale_pairing(client, tmp_path):
    from backend.api.routes import ebsd_viewer
    h5, _, sq, _ = _setup(tmp_path)
    _pair(client, sq, h5, GUID_B)
    shutil.move(str(h5), str(tmp_path / "moved.h5"))
    assert ebsd_viewer.load_ebsd_file(str(sq)) is True                # loads without the pairing
    assert "apex" not in _entry(client, "sq")


def test_the_batch_loader_refuses_a_conflict_and_keeps_the_pairing(client, tmp_path):
    from backend.api.routes import ebsd_viewer
    h5, _, _, other = _setup(tmp_path)
    make_up(other, guid_bytes=None, version=3, hex_flag=0, n=8, nx=4, ny=2)      # B is 2 x 4 = 8
    _pair(client, other, h5, GUID_B)
    _release_the_loaded_signals()
    make_up(other, guid_bytes=GUID_A_BYTES, hex_flag=0, n=8, nx=4, ny=2)         # now it names A
    assert ebsd_viewer.load_ebsd_file(str(other)) is False
    assert _entry(client, "other")["apex"]["measurement_id"] == GUID_B


# --------------------------------------------------------- s2 + s3: why it was dropped

def _paired_and_away(client, tmp_path):
    h5, _, sq, other = _setup(tmp_path)
    _pair(client, sq, h5, GUID_B)
    assert client.post("/api/ebsd/load", json={"path": str(other)}).status_code == 200
    return h5, sq


def _switch(client, sq):
    back = client.post("/api/ebsd/switch-file", json={"path": str(sq)})
    assert back.status_code == 200, back.text
    return back.json()


def test_a_switch_that_keeps_the_pairing_has_no_reason(client, tmp_path):
    _, sq = _paired_and_away(client, tmp_path)
    body = _switch(client, sq)
    assert body["apex_pairing_dropped"] is False and body["apex_pairing_dropped_reason"] is None


@pytest.mark.parametrize("how, reason", [
    ("moved", "apexUnreadable"),
    ("garbage", "apexUnreadable"),
    ("without_measurement", "apexMeasurementNotFound"),
    ("bad_id", "apexBadId"),
    ("no_measurement_at_all", "apexNoMeasurement"),
])
def test_a_dropped_pairing_says_why(client, tmp_path, how, reason):
    h5, sq = _paired_and_away(client, tmp_path)
    if how == "moved":
        shutil.move(str(h5), str(tmp_path / "moved.h5"))
    elif how == "garbage":
        h5.write_bytes(b"this is not an HDF5 file" * 10)
    elif how == "without_measurement":
        make_apex_h5(h5, [measurement_spec(id=GUID_A, area="Area 1", n_rows=3, n_cols=5)])
    elif how == "bad_id":
        make_apex_h5(h5, [measurement_spec(id="not-a-guid")])
    else:
        with h5py.File(h5, "w") as f:
            f.create_group("TOP/Sample/Area 1/Something Else")
    body = _switch(client, sq)
    assert body["apex_pairing_dropped"] is True and body["apex_pairing_dropped_reason"] == reason
    assert body["apex"] is None and "apex" not in _entry(client, "sq")


# ------------------------------------------------------------------ t1: IQ preview

def _without_iq(h5_path):
    """Rewrite the per-point table of every measurement without its IQ column."""
    with h5py.File(h5_path, "r+") as f:
        for key in list(f["TOP/Sample"].keys()):
            for sub in f["TOP/Sample"][key]:
                ds = f[f"TOP/Sample/{key}/{sub}/EBSD/ANG/DATA/DATA"]
                old = ds[()]
                keep = [n for n in old.dtype.names if n != "IQ"]
                new = np.zeros(old.shape, dtype=[(n, old.dtype[n]) for n in keep])
                for n in keep:
                    new[n] = old[n]
                del f[f"TOP/Sample/{key}/{sub}/EBSD/ANG/DATA/DATA"]
                f.create_dataset(f"TOP/Sample/{key}/{sub}/EBSD/ANG/DATA/DATA", data=new)


def test_an_iq_preview_of_a_measurement_without_iq_is_refused_by_name(client, tmp_path):
    h5 = _session(tmp_path)
    _without_iq(h5)
    inspected = client.get("/api/ebsd/inspect", params={"path": str(h5)}).json()
    assert [m["has_iq"] for m in inspected["measurements"]] == [False, False]
    r = client.get("/api/ebsd/apex/preview", params={"path": str(h5), "measurement": GUID_A, "kind": "iq"})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexNoPreview"
    assert "IQ" in r.json()["detail"]
    assert client.get("/api/ebsd/apex/preview", params={
        "path": str(h5), "measurement": GUID_A, "kind": "sem"}).status_code == 200      # the rest still works


def test_any_other_failure_while_reading_a_preview_is_a_coded_refusal_without_the_folder(
        client, tmp_path, monkeypatch):
    import edax_apex
    h5 = _session(tmp_path)

    def boom(measurement, field):
        raise RuntimeError(f"cannot read {measurement.h5_path}")
    monkeypatch.setattr(edax_apex, "point_map", boom)
    r = client.get("/api/ebsd/apex/preview", params={"path": str(h5), "measurement": GUID_A, "kind": "iq"})
    assert r.status_code == 400 and r.headers[CODE_HEADER.lower()] == "apexUnreadable"
    assert "session.h5" in r.json()["detail"] and str(tmp_path) not in r.json()["detail"]
    assert "RuntimeError" in r.json()["detail"]


# ------------------------------------------------------------ t5: the answer shapes

def test_the_inspect_answer_has_the_keys_the_dialog_reads(client, tmp_path):
    h5 = _session(tmp_path)
    make_up(tmp_path / "scan.up1", n=15, nx=5, ny=3, guid_bytes=GUID_A_BYTES)
    out = client.get("/api/ebsd/inspect", params={"path": str(h5)}).json()
    assert set(out) == {"kind", "h5_path", "measurements", "folder", "folder_problem",
                        "unmatched_pattern_files"}
    assert out["kind"] == "apex" and out["h5_path"] == str(h5)
    expected = {"id", "name", "area", "group", "grid_type", "hex", "n_rows", "n_cols", "n_points",
                "step_x", "step_y", "sample_tilt", "elevation", "azimuth", "pc", "pc_convention",
                "beam_kv", "has_sem", "has_iq", "phases", "pattern_files", "consistent",
                "loadable", "problem", "measurement_problem", "problem_detail",
                "measured_at", "measured_at_kind", "measured_at_timezone", "pattern_shape"}
    for m in out["measurements"]:
        assert set(m) == expected
    files = out["measurements"][0]["pattern_files"]
    assert len(files) == 1 and set(files[0]) == {
        "path", "name", "matches_grid", "problem", "n_patterns", "pattern_shape", "pairing", "count",
        "names_measurement", "names_group"}
    assert files[0]["problem"] is None and files[0]["matches_grid"] is True
    assert client.get("/api/ebsd/inspect", params={"path": str(tmp_path / "x.up1")}).json() == {"kind": "plain"}


def test_the_load_answer_names_the_pairing_with_these_keys(client, tmp_path):
    h5, hex_up, _, _ = _setup(tmp_path)
    apex = _pair(client, hex_up, h5, GUID_A).json()["apex"]
    assert set(apex) == {"h5_path", "measurement_id", "measurement_name", "pairing"}
    assert (apex["measurement_id"], apex["measurement_name"], apex["pairing"]) == (
        GUID_A, "Area 1 / OIM Map 1", "guid")                 # the file's header names A
    assert Path(apex["h5_path"]).resolve() == h5.resolve()
