"""A file left behind keeps one PC, not the per-pixel map its header brought; the map comes back when it is opened.

An Oxford file states a PC for every scan point. Left behind with nothing made on it, its stash holds no pattern and
no h5 handle, and no copy of that map either (it is read from the file again); the stash says it dropped one
(``map_dropped``), which is what the page reads as "has a PC per pixel". A calibration written into such a stash is one
number. When the file is opened, the number is given to the freshly read entry the way a typed PC is given to an open
file: the file's map moves rigidly so that its mean is that PC. Real files of the Test_data folder
(``ORIENTA_TEST_DATA``), through the real routes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ORIENTA_ALLOWED_HOSTS", "testserver")

from tests.apex_helpers import wipe_backend  # noqa: E402
from backend.api.services import pc_calibration_apply as use  # noqa: E402
from backend.api.services import pc_calibrations as cal  # noqa: E402
from tests.data_deps import local_name  # noqa: E402

#: an Oxford file whose header states a PC for every point of its 20 x 32 scan
MAP_FILE = local_name("oxford_pc_map_crop")   # an alias, see tests/data_deps.py
MAP_SHAPE = (20, 32, 3)
NI_FILE = "LoGainNi.h5"
TYPED_NI = (0.55, 0.30, 0.60)             # typed on the Ni file, saved as the calibration


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


def _store():
    from backend.api.services.calibration_store import calibration_store
    return calibration_store


def _post(client, url, body, code=200):
    r = client.post(url, json=body)
    assert r.status_code == code, r.text
    return r.json()


@pytest.fixture
def world(client):
    """The map file loaded, then the Ni file with a PC typed on it and saved as a calibration: the map file is left
    behind. Returns the calibration's id, the map file's path and its map as the file gave it."""
    from backend.api.routes import ebsd_viewer
    far, ni = _data(MAP_FILE), _data(NI_FILE)
    _post(client, "/api/ebsd/load", {"path": str(far)})
    entry = _store().get_entry(far.stem)
    assert entry.pc_map is not None and entry.pc_map.shape == MAP_SHAPE, "the premise: the header brings a map"
    header_map = entry.pc_map.copy()
    _post(client, "/api/ebsd/load", {"path": str(ni)})
    _post(client, "/api/pc/detector/update-pc", {"pcx": TYPED_NI[0], "pcy": TYPED_NI[1], "pcz": TYPED_NI[2]})
    cal_id = _post(client, "/api/pc-calibrations", {"name": "Ni REM", "dataset": ni.stem})["id"]
    stash = ebsd_viewer._registry_by_file[ebsd_viewer._canonical_path(str(far))]
    return cal_id, far, header_map, stash["calibration"][far.stem]


def test_a_file_left_behind_holds_no_copy_of_the_map_its_header_brought(world):
    _cal_id, _far, _header_map, stashed = world
    assert stashed.pc_map is None and stashed.map_dropped is True
    assert stashed.pc_source == "header"


def test_a_calibration_written_into_such_a_stash_is_one_number_and_the_file_still_reads_as_per_pixel(client, world):
    cal_id, far, _header_map, stashed = world
    out = _post(client, f"/api/pc-calibrations/{cal_id}/apply",
                {"datasets": [far.stem], "accept_mismatch": [far.stem]})["results"][0]
    assert out["status"] == "applied"
    assert stashed.pc_map is None and stashed.map_dropped is True
    assert stashed.pc_single == pytest.approx(TYPED_NI) and stashed.pc_source == "calibration"
    # the page asks the Ni file which files run on its version: the map file does, and says it has a map of its own
    followers = client.get(f"/api/pc-calibrations/for-dataset/LoGainNi").json()["followers"]
    assert followers == [{"dataset": far.stem, "per_pixel": True, "map_kind": "header"}]


def test_the_map_of_the_file_is_shifted_to_the_calibration_when_the_file_is_opened(client, world):
    cal_id, far, header_map, _stashed = world
    _post(client, f"/api/pc-calibrations/{cal_id}/apply", {"datasets": [far.stem], "accept_mismatch": [far.stem]})
    answer = _post(client, "/api/ebsd/switch-file", {"path": str(far)})
    assert answer["pc"] == pytest.approx(TYPED_NI) and answer["pc_source"] == "calibration"
    entry = _store().get_entry(far.stem)
    assert entry.pc_map is not None and entry.pc_map.shape == MAP_SHAPE
    # rigidly moved: the mean is the calibration's PC and every point keeps its distance from the mean
    assert entry.pc_map.reshape(-1, 3).mean(axis=0) == pytest.approx(TYPED_NI)
    old_dev = header_map - header_map.reshape(-1, 3).mean(axis=0)
    new_dev = entry.pc_map - entry.pc_map.reshape(-1, 3).mean(axis=0)
    assert np.allclose(new_dev, old_dev, atol=1e-9)
    assert entry.pc_source == "calibration"
    assert use.current_ref(far.stem, _store())["id"] == cal_id


def test_a_pc_typed_on_the_open_file_moves_its_map_when_the_file_comes_back(client):
    from backend.api.routes import ebsd_viewer
    far, ni = _data(MAP_FILE), _data(NI_FILE)
    typed = (0.51, 0.33, 0.83)
    _post(client, "/api/ebsd/load", {"path": str(far)})
    header_map = _store().get_entry(far.stem).pc_map.copy()
    _post(client, "/api/pc/detector/update-pc", {"pcx": typed[0], "pcy": typed[1], "pcz": typed[2]})
    _post(client, "/api/ebsd/load", {"path": str(ni)})
    stashed = ebsd_viewer._registry_by_file[ebsd_viewer._canonical_path(str(far))]["calibration"][far.stem]
    assert stashed.pc_map is None and stashed.map_dropped is True and stashed.pc_source == "manual"
    answer = _post(client, "/api/ebsd/switch-file", {"path": str(far)})
    assert answer["pc"] == pytest.approx(typed) and answer["pc_source"] == "manual"
    entry = _store().get_entry(far.stem)
    assert entry.pc_map.shape == MAP_SHAPE and entry.pc_map.reshape(-1, 3).mean(axis=0) == pytest.approx(typed)
    assert np.allclose(entry.pc_map - entry.pc_map.reshape(-1, 3).mean(axis=0),
                       header_map - header_map.reshape(-1, 3).mean(axis=0), atol=1e-9)


def test_a_map_the_user_made_stays_in_the_stash(client):
    from backend.api.routes import ebsd_viewer
    far, ni = _data(MAP_FILE), _data(NI_FILE)
    _post(client, "/api/ebsd/load", {"path": str(far)})
    made = np.tile(np.array([0.52, 0.31, 0.81]), MAP_SHAPE[:2] + (1,))
    made[0, 0] = (0.50, 0.30, 0.80)
    assert _store().update_pc_map(far.stem, made, refinement={"n_patterns": 9, "method": "PSO"})
    _post(client, "/api/ebsd/load", {"path": str(ni)})
    stashed = ebsd_viewer._registry_by_file[ebsd_viewer._canonical_path(str(far))]["calibration"][far.stem]
    assert stashed.map_dropped is False and np.array_equal(stashed.pc_map, made)
    _post(client, "/api/ebsd/switch-file", {"path": str(far)})
    assert np.array_equal(_store().get_entry(far.stem).pc_map, made)
