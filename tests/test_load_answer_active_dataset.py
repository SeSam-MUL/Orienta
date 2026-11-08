"""A load or switch answer describes the dataset the backend ends up with, not the file on disk.

Returning to a file restores its stashed datasets, and with them the dataset that was active when the
file was left (a crop, say). The viewer then works on that crop, so the answer of ``/load`` and
``/switch-file`` has to describe the crop: the viewer writes the answer into ``ebsdInfo`` and the data
store, and clicks on the overview map through the grid it finds there.

The scan is a hexagonal APEX measurement of 6 rows x 5 columns (made up). A crop of rows 1..3 starts
on an odd row, so it has the opposite row offset to the scan (hand calculation: the scan's odd rows
are shifted by +0.5 of a step, a window that starts on an odd row has its first row unshifted and its
second shifted the other way, -0.5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.apex_helpers import make_apex_h5, make_up, measurement_spec, wipe_backend  # noqa: E402

ROWS, COLS = 6, 5
CROP = dict(shape="rect", row0=1, col0=0, rows=3, cols=4, name="odd")


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    yield
    wipe_backend()


@pytest.fixture
def files(tmp_path):
    hex_dir = tmp_path / "hex"
    hex_dir.mkdir()
    make_apex_h5(hex_dir / "s.h5", [measurement_spec(n_rows=ROWS, n_cols=COLS, step_x=0.5)])
    hexup = make_up(hex_dir / "scan_hex.up1", n=ROWS * COLS, nx=COLS, ny=ROWS)
    other_dir = tmp_path / "sq"
    other_dir.mkdir()
    other = make_up(other_dir / "o.up1", guid_bytes=None, version=3, hex_flag=0, n=6, nx=3, ny=2)
    return hexup, other


def _load(client, path):
    r = client.post("/api/ebsd/load", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return r.json()


def _left_with_a_crop_active(client, hexup, other):
    """Hex scan loaded, cropped (the crop is active), then another file loaded."""
    _load(client, hexup)
    r = client.post("/api/ebsd/crop", json=CROP)
    assert r.status_code == 200, r.text
    assert r.json()["scan_geometry"]["odd_row_shift"] == -0.5          # the premise
    _load(client, other)


def _check_describes_the_crop(answer, info):
    # dataset
    assert answer["dataset_name"] == "odd"
    # grid: 3 rows x 4 columns; hyperspy order (columns, rows) for the navigation shape
    assert answer["grid_shape"] == [3, 4]
    assert answer["navigation_shape"] == [4, 3]
    assert answer["data_shape"] == [3, 4, 8, 8]
    assert answer["n_patterns"] == 12 and answer["pattern_count"] == 12
    # the detector does not change with a crop (8 x 8, made up)
    assert answer["signal_shape"] == [8, 8] and answer["pattern_shape"] == [8, 8]
    # geometry: the crop's own
    geometry = answer["scan_geometry"]
    assert geometry["kind"] == "hex"
    assert geometry["grid_shape"] == [3, 4]
    assert geometry["odd_row_shift"] == -0.5
    # and it is what the backend reports for the active dataset
    assert info["active_dataset"] == "odd"
    assert info["scan_geometry"]["grid_shape"] == geometry["grid_shape"]
    assert info["scan_geometry"]["odd_row_shift"] == geometry["odd_row_shift"]
    assert info["data_shape"][:2] == answer["grid_shape"]


def test_a_load_that_restores_a_crop_describes_the_crop(client, files):
    hexup, other = files
    _left_with_a_crop_active(client, hexup, other)
    answer = _load(client, hexup)
    _check_describes_the_crop(answer, client.get("/api/ebsd/info").json())


def test_a_switch_that_restores_a_crop_describes_the_crop(client, files):
    hexup, other = files
    _left_with_a_crop_active(client, hexup, other)
    r = client.post("/api/ebsd/switch-file", json={"path": str(hexup)})
    assert r.status_code == 200, r.text
    _check_describes_the_crop(r.json(), client.get("/api/ebsd/info").json())


def test_a_load_without_a_stash_still_describes_the_raw_scan(client, files):
    hexup, _ = files
    answer = _load(client, hexup)
    assert answer["dataset_name"] == "scan_hex"
    assert answer["grid_shape"] == [ROWS, COLS]
    assert answer["navigation_shape"] == [COLS, ROWS]
    assert answer["data_shape"][:2] == [ROWS, COLS]
    assert answer["n_patterns"] == ROWS * COLS
    assert answer["scan_geometry"]["odd_row_shift"] == 0.5


def test_the_restore_notice_is_still_reported_when_the_active_dataset_is_a_crop(client, files, monkeypatch):
    """The restore files its notice under the file's raw dataset; the answer is about the crop."""
    from backend.api.routes import ebsd_viewer
    hexup, other = files
    _left_with_a_crop_active(client, hexup, other)
    real_restore = ebsd_viewer.calibration_store.restore

    def restore(entries, skip=None):
        real_restore(entries, skip=skip)
        return "file_pc_wins", [0.1, 0.2, 0.3]

    monkeypatch.setattr(ebsd_viewer.calibration_store, "restore", restore)
    answer = _load(client, hexup)
    assert answer["dataset_name"] == "odd"
    assert answer["pc_notice"] == "file_pc_wins"
    assert answer["pc_discarded"] == [0.1, 0.2, 0.3]


def _restore_deciding(monkeypatch, notice, discarded=None):
    from backend.api.routes import ebsd_viewer
    real_restore = ebsd_viewer.calibration_store.restore

    def restore(entries, skip=None):
        real_restore(entries, skip=skip)
        return notice, discarded

    monkeypatch.setattr(ebsd_viewer.calibration_store, "restore", restore)


def test_a_kept_pc_is_not_reported_next_to_the_pc_of_a_crop(client, files, monkeypatch):
    """"The PC you set was kept" is a decision about the file's raw dataset. With a crop active the
    answer carries the crop's own PC, which that decision did not touch: the note would read as a
    statement about the PC shown beside it."""
    hexup, other = files
    _left_with_a_crop_active(client, hexup, other)
    _restore_deciding(monkeypatch, "user_pc_kept")
    answer = _load(client, hexup)
    assert answer["dataset_name"] == "odd"
    assert answer["pc_notice"] is None


def test_a_kept_pc_is_reported_when_the_raw_dataset_is_the_active_one(client, files, monkeypatch):
    from backend.api.services.calibration_store import calibration_store
    hexup, other = files
    _load(client, hexup)
    calibration_store.update_pc("scan_hex", [0.51, 0.33, 0.84], source="manual")   # gives the file a stash
    _load(client, other)
    _restore_deciding(monkeypatch, "user_pc_kept")
    answer = _load(client, hexup)
    assert answer["dataset_name"] == "scan_hex"
    assert answer["pc_notice"] == "user_pc_kept"


def test_the_pattern_centre_in_the_answer_is_the_active_datasets(client, files):
    """The crop has been given a pattern centre of its own before the file was left."""
    from backend.api.services.calibration_store import calibration_store
    hexup, other = files
    _load(client, hexup)
    assert client.post("/api/ebsd/crop", json=CROP).status_code == 200
    raw_pc = [float(v) for v in calibration_store.get_pc("scan_hex")]
    calibration_store.detach_from_parent("odd")
    assert calibration_store.update_pc("odd", (0.41, 0.52, 0.63), source="manual")
    assert [float(v) for v in calibration_store.get_pc("scan_hex")] == raw_pc   # the premise: they differ
    _load(client, other)
    answer = _load(client, hexup)
    assert answer["dataset_name"] == "odd"
    assert answer["pc"] == pytest.approx([0.41, 0.52, 0.63])
    assert answer["pc_store_source"] == "manual"
