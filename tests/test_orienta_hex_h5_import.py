"""A hexagonal ``.h5`` that Orienta wrote comes back as a hexagonal result.

``/api/indexing/import-h5`` and ``/api/analysis/load`` read the stamp on ``/Indexing``
(``orienta_hex_h5.read_index_geometry``) and keep it, so the raster and the guards of a hexagonal
scan apply to an imported result. A file without a stamp is a square or legacy file and behaves as it
always did; a damaged stamp is refused and nothing is stored.

Files come from the writers (``tests/test_orienta_hex_h5_writers.py``): 3 rows x 4 columns, step_x
2.0 um, step_y = 2 * sqrt(3) / 2 = 1.7320508075688772 um (by hand), odd rows shifted by +-0.5 step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.problem import CODE_HEADER  # noqa: E402
from backend.api.routes import analysis as analysis_routes  # noqa: E402
from backend.api.routes import indexing as ix  # noqa: E402
from backend.api.services import orienta_hex_h5 as oh  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402
from tests import hex_raster_oracle as oracle  # noqa: E402
from tests.apex_helpers import wipe_backend  # noqa: E402
from tests.test_orienta_hex_h5_writers import _hex, _light, _rich  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SQRT3 = 1.7320508075688772                    # 2 * sqrt(3) / 2, by hand
WRITERS = {"light": _light, "rich": _rich}


@pytest.fixture
def client():
    from backend.api.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def fresh_backend():
    wipe_backend()
    ix._result_registry.clear()
    ix._active_result_id = None
    analysis_routes._dataset = None
    yield
    wipe_backend()
    ix._result_registry.clear()
    ix._active_result_id = None
    analysis_routes._dataset = None


def _import(client, path):
    return client.post("/api/indexing/import-h5", json={"path": str(path)})


def _imported(client, path):
    resp = _import(client, path)
    assert resp.status_code == 200, resp.text
    return ix._active_result_id, ix._result_registry[ix._active_result_id]


def _drop_stamp(path):
    """The file as an older (or a square) export wrote it: no stamp on /Indexing."""
    with h5py.File(path, "a") as f:
        for name in ("grid_type", "odd_row_shift", "shift_confirmed", "step_x", "step_y",
                     "odd_row_shift_assumed"):
            if name in f["Indexing"].attrs:
                del f["Indexing"].attrs[name]
        assert "grid_type" not in f["Indexing"].attrs


# ----------------------------------------------------------------------------- the stamp

@pytest.mark.parametrize("writer", ["light", "rich"])
@pytest.mark.parametrize("shift, confirmed", [(0.5, True), (-0.5, False)])
def test_import_hex_gets_stamp(client, tmp_path, writer, shift, confirmed):
    path = WRITERS[writer](tmp_path, _hex(shift, confirmed))
    _, result = _imported(client, path)
    stamp = result.metadata["scan_geometry"]
    # the grid the writer was given, by hand: 3 x 4 points, 2.0 um, 2 * sqrt(3) / 2 um between rows
    assert (stamp["kind"], stamp["n_rows"], stamp["n_cols"]) == ("hex", 3, 4)
    assert stamp["grid_shape"] == [3, 4]
    assert stamp["step_x"] == 2.0 and stamp["step_y"] == SQRT3 and stamp["steps_known"] is True
    assert stamp["odd_row_shift"] == shift and stamp["shift_confirmed"] is confirmed
    # a stamped file was not resampled; "unknown" (None) is for a file that says nothing
    assert result.metadata["hex_resampled"] is False


def test_import_square_stamp_is_kept_as_square(client, tmp_path):
    path = _light(tmp_path, ScanGeometry.square(3, 4, 2.0))
    with h5py.File(path, "a") as f:                  # a SqrGrid stamp, written by hand
        a = f["Indexing"].attrs
        a["grid_type"], a["odd_row_shift"], a["shift_confirmed"] = "SqrGrid", 0.0, True
        a["step_x"], a["step_y"] = 2.0, 2.0
    _, result = _imported(client, path)
    stamp = result.metadata["scan_geometry"]
    assert stamp["kind"] == "square" and stamp["grid_shape"] == [3, 4] and stamp["step_x"] == 2.0
    assert result.metadata["hex_resampled"] is False


def test_import_without_stamp_stays_unknown(client, tmp_path):
    path = _light(tmp_path, _hex())
    _drop_stamp(path)
    _, result = _imported(client, path)
    assert result.metadata["scan_geometry"] is None
    assert result.metadata["hex_resampled"] is None


# ------------------------------------------- the stamp in the answers the pages read to offer exports

@pytest.mark.parametrize("writer", ["light", "rich"])
@pytest.mark.parametrize("shift, confirmed", [(0.5, True), (-0.5, False)])
def test_import_and_last_result_answer_with_the_stamp(client, tmp_path, writer, shift, confirmed):
    """The pages offer .ctf and the square .ang from this field, because ``import-h5`` never sets the
    result the Indexing page holds. Expected values: the grid the writer was given (3 x 4, 2.0 um)."""
    path = WRITERS[writer](tmp_path, _hex(shift, confirmed))
    imported = _import(client, path)
    assert imported.status_code == 200, imported.text
    last = client.get("/api/indexing/result/last").json()
    for answer in (imported.json(), last):
        stamp = answer["scan_geometry"]
        assert (stamp["kind"], stamp["n_rows"], stamp["n_cols"]) == ("hex", 3, 4)
        assert stamp["step_x"] == 2.0
        assert stamp["odd_row_shift"] == shift and stamp["shift_confirmed"] is confirmed


def test_import_and_last_result_without_stamp_say_none(client, tmp_path):
    path = _light(tmp_path, _hex())
    _drop_stamp(path)
    imported = _import(client, path)
    assert imported.status_code == 200, imported.text
    assert imported.json()["scan_geometry"] is None
    last = client.get("/api/indexing/result/last").json()
    assert "scan_geometry" in last and last["scan_geometry"] is None


def test_last_result_says_the_stamp_of_the_active_result_not_the_loaded_dataset(client, tmp_path):
    """Two results, one hexagonal and one square: the answer follows the one that is active."""
    hex_id, _ = _imported(client, _light(tmp_path, _hex()))
    (tmp_path / "sq").mkdir()
    square = _light(tmp_path / "sq", ScanGeometry.square(3, 4, 2.0))
    with h5py.File(square, "a") as f:
        a = f["Indexing"].attrs
        a["grid_type"], a["odd_row_shift"], a["shift_confirmed"] = "SqrGrid", 0.0, True
        a["step_x"], a["step_y"] = 2.0, 2.0
    sq_id, _ = _imported(client, square)
    assert sq_id != hex_id
    assert client.get("/api/indexing/result/last").json()["scan_geometry"]["kind"] == "square"
    ix._active_result_id = hex_id
    assert client.get("/api/indexing/result/last").json()["scan_geometry"]["kind"] == "hex"


# --------------------------------------------------------------------------- the guards

def _resmooth(client, rid):
    return client.post("/api/refinement/resmooth", json={"result_id": rid, "smoothness_lambda": 1.0})


def test_guarded_route_refuses_imported_hex(client, tmp_path):
    rid, _ = _imported(client, _light(tmp_path, _hex()))
    resp = _resmooth(client, rid)
    assert resp.status_code == 400, resp.text
    assert resp.headers[CODE_HEADER.lower()] == "hex_neighbourhood_unsupported"


def test_imported_square_not_refused(client, tmp_path):
    """A file without a stamp, and a file with a square stamp, are not a hexagonal scan."""
    plain = _light(tmp_path, _hex())
    _drop_stamp(plain)
    rid, _ = _imported(client, plain)
    resp = _resmooth(client, rid)
    assert CODE_HEADER.lower() not in resp.headers, resp.text

    ix._result_registry.clear()
    (tmp_path / "sq").mkdir()
    square = _light(tmp_path / "sq", ScanGeometry.square(3, 4, 2.0))
    with h5py.File(square, "a") as f:
        a = f["Indexing"].attrs
        a["grid_type"], a["odd_row_shift"], a["shift_confirmed"] = "SqrGrid", 0.0, True
        a["step_x"], a["step_y"] = 2.0, 2.0
    rid, _ = _imported(client, square)
    resp = _resmooth(client, rid)
    assert CODE_HEADER.lower() not in resp.headers, resp.text


def test_imported_hex_draws_raster(client, tmp_path):
    """The layer route draws the imported result on the display raster of its hexagonal stamp."""
    _imported(client, _light(tmp_path, _hex(0.5, True)))
    resp = client.get("/api/phasemap/layer", params={"kind": "phase"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    want = oracle.spec(oracle.geometry_dict(3, 4, 2.0, 0.5), 4)       # the independent oracle
    assert body["display"]["kind"] == "hex_raster"
    assert body["shape"] == [want["height"], want["width"]] != [3, 4]


def test_imported_square_is_not_drawn_on_a_raster(client, tmp_path):
    plain = _light(tmp_path, _hex())
    _drop_stamp(plain)
    _imported(client, plain)
    body = client.get("/api/phasemap/layer", params={"kind": "phase"}).json()
    assert body["shape"] == [3, 4] and "display" not in body


# ------------------------------------------------------------------- damaged and missing

@pytest.mark.parametrize("damage, words", [
    ("odd_row_shift", "odd_row_shift"),            # a HexGrid stamp lacks a key
    ("grid_type", "no grid_type"),                 # stamp attributes without grid_type
    ("grid_shape", "no grid_shape"),               # a grid_type without a grid_shape
    ("shift", "+-0.5 steps, got 0.3"),             # a shift other than +-0.5
])
def test_a_damaged_stamp_is_refused_and_nothing_is_stored(client, tmp_path, damage, words):
    path = _light(tmp_path, _hex())
    with h5py.File(path, "a") as f:
        attrs = f["Indexing"].attrs
        if damage == "shift":
            attrs["odd_row_shift"] = 0.3
        else:
            del attrs[damage]
    resp = _import(client, path)
    assert resp.status_code == 400, resp.text
    assert words in resp.json()["detail"]
    assert ix._result_registry == {} and ix._active_result_id is None


def test_a_grid_shape_that_is_not_the_stored_data_is_refused(client, tmp_path):
    path = _light(tmp_path, _hex())
    with h5py.File(path, "a") as f:
        f["Indexing"].attrs["grid_shape"] = np.array([4, 3], np.int64)
    resp = _import(client, path)
    assert resp.status_code == 400, resp.text
    assert ix._result_registry == {}


def test_a_file_without_indexing_is_refused_as_before(client, tmp_path):
    path = tmp_path / "no_indexing.h5"
    with h5py.File(path, "w") as f:
        f.create_group("Other")
    resp = _import(client, path)
    assert resp.status_code == 400
    assert "has no /Indexing group" in resp.json()["detail"]
    assert ix._result_registry == {}


# --------------------------------------------------------------------------- load_xmap

def _load_xmap(client, path):
    return client.post("/api/analysis/load", json={"xmap_path": str(path)})


def test_load_xmap_keeps_stamp(client, tmp_path):
    path = _light(tmp_path, _hex(-0.5, False))
    resp = _load_xmap(client, path)
    assert resp.status_code == 200, resp.text
    assert resp.json()["shape"] == [3, 4]
    stamp = analysis_routes._dataset.scan_geometry
    assert (stamp["kind"], stamp["n_rows"], stamp["n_cols"]) == ("hex", 3, 4)
    assert stamp["step_x"] == 2.0 and stamp["step_y"] == SQRT3
    assert stamp["odd_row_shift"] == -0.5 and stamp["shift_confirmed"] is False
    assert analysis_routes._dataset.step_size == 2.0
    # the analysis basis of the guard sees it: the Analysis page's maps are refused
    refused = client.get("/api/analysis/map/kam")
    assert refused.status_code == 400
    assert refused.headers[CODE_HEADER.lower()] == "hex_analysis_page_pending"


def test_load_xmap_without_stamp_stays_unknown(client, tmp_path):
    path = _light(tmp_path, _hex())
    _drop_stamp(path)
    resp = _load_xmap(client, path)
    assert resp.status_code == 200, resp.text
    assert analysis_routes._dataset.scan_geometry is None


def test_load_xmap_refuses_a_damaged_stamp_and_keeps_the_dataset(client, tmp_path):
    path = _light(tmp_path, _hex())
    with h5py.File(path, "a") as f:
        del f["Indexing"].attrs["step_y"]
    analysis_routes._dataset = sentinel = object()
    resp = _load_xmap(client, path)
    assert resp.status_code == 400, resp.text
    assert "step_y" in resp.json()["detail"]
    assert analysis_routes._dataset is sentinel


def test_load_xmap_of_a_file_without_indexing_fails_as_before(client, tmp_path):
    path = tmp_path / "no_indexing.h5"
    with h5py.File(path, "w") as f:
        f.create_group("Other")
    resp = _load_xmap(client, path)
    assert resp.status_code == 400
    assert "Custom .h5 fallback failed" in resp.json()["detail"]
    assert "no /Indexing group" in resp.json()["detail"]
