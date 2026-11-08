"""A rich ``.h5`` is still saved in the situations where the refusal of a wrong source must not fire.

The rich export of a hexagonal result is refused when the file it would copy is not the scan the result
ran on. These are the honest cases that must keep working, through the real route and the real import:

* a result reloaded from its own rich ``.h5`` (the file is the open one afterwards) saves again, also
  when it is the copy of a crop run;
* a light ``.h5`` imported while the original source (a ``.up1`` pair, or the native parent of a crop)
  is open saves rich: a light file brings no patterns, so the open file stays the source;
* a result on a map that was resampled from a hexagonal file, saved with its OIM file open (the copy
  branch), writes the copy without a hexagonal stamp;
* a square result with a square HDF5 file open, and a native file of any step against its own stamp
  (the header stores floats that may be narrower than the stamp's).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import ebsd_viewer as ev, indexing as ix
from backend.api.services.scan_geometry import ScanGeometry
from tests import test_orienta_hex_h5_writers as W
from tests.test_orienta_hex_h5_writers import route  # noqa: F401  (fixture)

oh = W.oh
PARENT, CROP = W.PARENT, W.CROP_AT_ODD_ROW


@pytest.fixture(autouse=True)
def keep_registry():
    saved, active = dict(ix._result_registry), ix._active_result_id
    path, dataset = ev._ebsd_file_path, ev._active_dataset
    yield
    ix._result_registry.clear()
    ix._result_registry.update(saved)
    ix._active_result_id = active
    ev._ebsd_file_path, ev._active_dataset = path, dataset


def _import(path):
    resp = TestClient(app).post("/api/indexing/import-h5", json={"path": str(path)})
    assert resp.status_code == 200, resp.text
    return ix._result_registry[ix._active_result_id]


def _crop_stamp(route):
    route.result.metadata.update({"scan_geometry": CROP.to_dict(), "scan_row_offset": 1,
                                  "scan_col_offset": 1, "scan_shape": [5, 6]})


def test_a_rich_h5_reloaded_from_its_own_file_saves_again(route):
    first = W._post("h5", route.tmp / "first.h5")
    assert oh.is_native_hex_file(first)
    result = _import(first)
    assert Path(ev._ebsd_file_path).name == "first.h5"          # the rich file is the open one now
    assert result.metadata["scan_geometry"]["kind"] == "hex"
    second = W._post("h5", route.tmp / "second.h5")
    assert oh.is_native_hex_file(second)


def test_the_rich_copy_of_a_crop_run_reloaded_from_its_own_file_saves_again(route, monkeypatch):
    W._native_parent(route, monkeypatch)
    _crop_stamp(route)
    first = W._post("h5", route.tmp / "crop_first.h5")
    assert oh.read_header_geometry(first) == PARENT             # the copy keeps its parent's header
    _import(first)
    assert oh.is_native_hex_file(W._post("h5", route.tmp / "crop_second.h5"))


def test_a_light_h5_imported_with_the_up1_source_open_saves_rich(route):
    light = W._post("h5_light", route.tmp / "light.h5")
    up1 = ev._ebsd_file_path
    result = _import(light)
    assert ev._ebsd_file_path == up1                            # a light file brings no patterns
    assert result.metadata["scan_geometry"]["kind"] == "hex"
    W._post("h5", route.tmp / "rich_after_import.h5")


def test_the_light_h5_of_a_crop_imported_with_the_native_parent_open_saves_rich(route, monkeypatch):
    source = W._native_parent(route, monkeypatch)
    _crop_stamp(route)
    light = W._post("h5_light", route.tmp / "crop_light.h5")
    _import(light)
    assert Path(ev._ebsd_file_path) == Path(source)
    W._post("h5", route.tmp / "crop_rich_after_import.h5")


def test_a_result_on_a_resampled_map_with_its_oim_file_open_saves_the_copy(route, monkeypatch):
    source = W._source(route.tmp, "oim_scan.h5")
    monkeypatch.setattr(ev, "_ebsd_file_path", str(source), raising=False)
    route.result.metadata["hex_resampled"] = True
    out = W._post("h5", route.tmp / "resampled_copy.h5")
    with h5py.File(out, "r") as f:
        assert not (W.STAMP_KEYS & set(f["Indexing"].attrs))    # the copy states no hexagonal grid


def test_a_square_result_with_a_square_hdf5_file_open_saves_rich(route, monkeypatch):
    source = W._source(route.tmp, "square.h5oina")
    monkeypatch.setattr(ev, "_ebsd_file_path", str(source), raising=False)
    route.result.metadata["scan_geometry"] = ScanGeometry.square(3, 4, 2.0).to_dict()
    W._post("h5", route.tmp / "square_copy.h5")


@pytest.mark.parametrize("step", [0.1, 1 / 3, 0.0625, 12.7])
def test_a_native_file_of_any_step_passes_against_its_own_stamp(route, monkeypatch, step):
    geometry = ScanGeometry.hex(3, 4, step, odd_row_shift=0.5, shift_confirmed=True)
    W._native_parent(route, monkeypatch, geometry)
    route.result.metadata["scan_geometry"] = geometry.to_dict()
    W._post("h5", route.tmp / "any_step.h5")
