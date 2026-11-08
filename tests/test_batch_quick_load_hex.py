"""POST /api/batch-v2/quick-load takes a hexagonal scan and says which grid and which metadata it found.

The batch runs hexagonal scans. A hexagonal pattern file is placed by the APEX metadata of its
measurement, so quick-load answers 200 with ``grid_kind``, the geometry the loader read and the
pairing (``apex``); a hexagonal file with no metadata is refused when it is added, with the code the
page translates, and leaves nothing behind. Real route, real UP files: a hexagonal one (3 x 5
points, with the APEX metadata that says so), the same without metadata, and a square one without any.
"""
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import batch_v2 as B
from backend.api.services.calibration_store import CalibrationStore
from tests.apex_helpers import GUID_A, make_apex_h5, make_up, measurement_spec

N_ROWS, N_COLS = 3, 5


@pytest.fixture
def store(monkeypatch):
    s = CalibrationStore()
    monkeypatch.setattr(B, "calibration_store", s)
    monkeypatch.setattr("backend.api.services.calibration_store.calibration_store", s)
    B._quick_loaded.clear()
    yield s
    B._quick_loaded.clear()


def _hex_scan(folder):
    make_apex_h5(folder / "session.h5", [measurement_spec(n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5)])
    return str(make_up(folder / "hex_scan.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=45))


def _hex_scan_without_metadata(folder):
    return str(make_up(folder / "lonely.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=45))


def _square_scan(folder, side=24, n_side=4):
    rng = np.random.default_rng(3)
    pats = rng.integers(0, 250, (n_side * n_side, side, side)).astype("u1")
    path = folder / "square_scan.up1"
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", 1, side, side, 16))
        f.write(pats.tobytes())
    return str(path)


def _post(path):
    return TestClient(app).post("/api/batch-v2/quick-load", json={"file_path": path})


def test_a_hexagonal_scan_with_metadata_is_loaded_and_says_so(store, tmp_path):
    resp = _post(_hex_scan(tmp_path))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["grid_kind"] == "hex"
    assert body["grid_shape"] == [N_ROWS, N_COLS] and body["pattern_count"] == N_ROWS * N_COLS
    assert (body["scan_geometry"]["kind"], body["scan_geometry"]["grid_shape"]) == ("hex", [N_ROWS, N_COLS])
    assert body["scan_geometry"]["step_x"] == pytest.approx(0.5)
    assert body["apex"]["measurement_id"] == GUID_A
    assert body["apex"]["measurement_name"] == "Area 1 / OIM Map 1"
    assert store.get_entry(body["dataset_name"]) is not None
    assert body["dataset_name"] in B._quick_loaded


def test_a_hexagonal_scan_without_metadata_is_refused_with_a_coded_400(store, tmp_path):
    resp = _post(_hex_scan_without_metadata(tmp_path))
    assert resp.status_code == 400, resp.text
    assert resp.headers["x-orienta-code"] == "edaxHexUpUnsupported"
    # the English sentence stays the whole of `detail`
    assert "hexagonal grid" in resp.json()["detail"]


def test_a_refused_hexagonal_scan_leaves_nothing_behind(store, tmp_path):
    from backend.api.services.batch_manager import store_name_for_file
    path = _hex_scan_without_metadata(tmp_path)
    name = store_name_for_file(Path(path))
    assert _post(path).status_code == 400
    assert store.get_entry(name) is None             # no calibration was registered ...
    assert B._quick_loaded == {}                     # ... and no signal is kept for it


def test_a_refusal_does_not_touch_the_calibration_of_another_file(store, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    ok = _post(_square_scan(other))
    assert ok.status_code == 200, ok.text
    hexdir = tmp_path / "hexdir"
    hexdir.mkdir()
    assert _post(_hex_scan_without_metadata(hexdir)).status_code == 400
    assert store.get_entry(ok.json()["dataset_name"]) is not None
    assert list(B._quick_loaded) == [ok.json()["dataset_name"]]


def test_a_square_scan_is_loaded_as_before(store, tmp_path):
    resp = _post(_square_scan(tmp_path))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["grid_shape"] == [4, 4] and body["pattern_shape"] == [24, 24]
    assert body["pattern_count"] == 16
    assert store.get_entry(body["dataset_name"]) is not None
    assert body["dataset_name"] in B._quick_loaded
    # the loader's copy of the geometry does not stay on the kept signal (it goes stale on a crop)
    kept = B._quick_loaded[body["dataset_name"]]["signal"]
    assert kept.metadata.get_item("Signal.scan_geometry", None) is None


def test_a_square_apex_scan_is_loaded_and_its_geometry_copy_is_dropped(store, tmp_path):
    make_apex_h5(tmp_path / "session.h5", [measurement_spec(
        grid="SqrGrid", n_rows=N_ROWS, n_cols=N_COLS, step_x=0.5)])
    path = str(make_up(tmp_path / "sq_scan.up1", n=N_ROWS * N_COLS, nx=N_COLS, ny=N_ROWS, hex_flag=0))
    resp = _post(path)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["grid_shape"] == [N_ROWS, N_COLS] and body["pattern_count"] == 15
    # the copy the loader attaches would describe the whole scan after a crop: it is not kept
    kept = B._quick_loaded[body["dataset_name"]]["signal"]
    assert kept.metadata.get_item("Signal.scan_geometry", None) is None
