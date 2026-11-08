"""The four ``.h5`` writers carry the hexagonal blocks, and a light file holds true positions.

Writers: ``export_result_h5`` (batch, copy of the source), ``export_result_h5_light`` (batch,
fresh), and the route ``/api/indexing/export`` with format ``h5`` (fresh file for a source that is
not HDF5) and ``h5_light``. Hand scan used throughout: 3 rows x 4 columns, step_x 2.0 um, odd rows
shifted by +0.5 step, step_y = 2 * sqrt(3) / 2 = 1.7320508075688772 um. Positions (hand):

    row 0: x = 0, 2, 4, 6      y = 0
    row 1: x = 1, 3, 5, 7      y = 1.7320508075688772
    row 2: x = 0, 2, 4, 6      y = 3.4641016151377544

and with the shift -0.5 the odd row is x = -1, 1, 3, 5.

The checkpoint holds orientation channel 0 = row * 10 + column, so a file value says by itself
which scan point it belongs to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import orienta_hex_h5 as oh  # noqa: E402
from backend.api.services.checkpoint_writer import CheckpointWriter  # noqa: E402
from backend.api.services.result_exporter import (  # noqa: E402
    export_result_h5, export_result_h5_light,
)
from backend.api.services.scan_geometry import GeometryMismatchError, ScanGeometry  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SQRT3 = 1.7320508075688772                    # 2 * sqrt(3) / 2, by hand
X_PLUS = np.array([[0, 2, 4, 6], [1, 3, 5, 7], [0, 2, 4, 6]], np.float64)
X_MINUS = np.array([[0, 2, 4, 6], [-1, 1, 3, 5], [0, 2, 4, 6]], np.float64)
Y_HAND = np.repeat(np.array([[0.0], [SQRT3], [2 * SQRT3]]), 4, axis=1)
CAVEATS = [{"code": "houghTrigonalSetting", "phases": ["Cr2N"], "text": "a note"}]
STAMP_KEYS = {"grid_type", "odd_row_shift", "shift_confirmed", "step_x", "step_y",
              "odd_row_shift_assumed"}


def _hex(shift=0.5, confirmed=True):
    return ScanGeometry.hex(3, 4, 2.0, odd_row_shift=shift, shift_confirmed=confirmed)


def _source(tmp_path, name="Scan.h5oina"):
    """A source file without a Manufacturer: the vendor frame is the identity."""
    path = tmp_path / name
    with h5py.File(path, "w") as f:
        f.create_group("1/EBSD/Header")
        f.create_group("1/EBSD/Data")
    return path


def _checkpoint(tmp_path, source, geometry, *, space_group=162, point_group="-3m"):
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="b", method="spherical",
                     step_size_um=0.0 if geometry is not None else 2.0, geometry=geometry)
    rows, cols = np.mgrid[0:3, 0:4]
    ori = np.zeros((3, 4, 3), np.float32)
    ori[..., 0] = rows * 10 + cols
    ori[..., 1] = 0.5
    ori[..., 2] = 0.25
    cw.write_phase_result(
        "Cr2N", np.full((3, 4), 0.7, np.float32), ori,
        {"phase_file": "Cr2N.cif", "ci_mean": 0.7, "duration_sec": 1.0,
         "space_group": space_group, "point_group": point_group,
         "lattice_constants": [4.8, 4.8, 4.5, 90, 90, 120]})
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return cw.checkpoint_path


def _light(tmp_path, geometry, **kw):
    source = _source(tmp_path)
    cp = _checkpoint(tmp_path, source, geometry)
    return export_result_h5_light(str(source), cp, str(tmp_path / "out"), caveats=CAVEATS, **kw)


def _rich(tmp_path, geometry, source=None, **kw):
    source = source or _source(tmp_path)
    cp = _checkpoint(tmp_path, source, geometry)
    return export_result_h5(str(source), cp, str(tmp_path / "out"), caveats=CAVEATS, **kw)


def _attr(value):
    return value.decode() if isinstance(value, bytes) else value


def _assert_index_stamp(path, geometry, step_size_um=2.0):
    """The five attributes of the checkpoint, by hand, on /Indexing."""
    with h5py.File(path, "r") as f:
        a = f["Indexing"].attrs
        assert _attr(a["grid_type"]) == "HexGrid"
        assert float(a["odd_row_shift"]) == geometry.odd_row_shift
        assert bool(a["shift_confirmed"]) is geometry.shift_confirmed
        assert float(a["step_x"]) == 2.0
        assert float(a["step_y"]) == SQRT3
        assert [int(v) for v in a["grid_shape"]] == [3, 4]
        assert float(a["step_size_um"]) == 2.0
    assert oh.read_index_geometry(path) == geometry


# --------------------------------------------------------------------- light, batch

@pytest.mark.parametrize("shift, x_hand", [(0.5, X_PLUS), (-0.5, X_MINUS)])
def test_light_batch_hex_layout(tmp_path, shift, x_hand):
    geometry = _hex(shift)
    path = _light(tmp_path, geometry)
    _assert_index_stamp(path, geometry)
    with h5py.File(path, "r") as f:
        x, y = f["Indexing/X"], f["Indexing/Y"]
        assert x.dtype == np.float64 and y.dtype == np.float64        # positions verbatim
        np.testing.assert_array_equal(x[()], x_hand)
        np.testing.assert_array_equal(y[()], Y_HAND)
        assert "step_x" in x.attrs["description"] and "column * step_size_um" not in x.attrs["description"]
        assert _attr(x.attrs["unit"]) == "um" and _attr(y.attrs["unit"]) == "um"
        # the stamp's own sentence is absent on a confirmed shift
        assert "odd_row_shift_assumed" not in f["Indexing"].attrs
        # the batch light file has never carried a README; if it gets one, it needs the
        # hexagonal paragraph of X and Y too (see `_assert_hex_readme`)
        assert "README" not in f["Documentation"]


def _readme(path) -> str:
    with h5py.File(path, "r") as f:
        return _attr(f["Documentation/README"][()])


def _assert_hex_readme(path):
    """The README of a hexagonal file describes X and Y as positions (hand-written words), and the
    square sentence "X = column·step," is gone."""
    text = _readme(path)
    flat = " ".join(text.split())                       # the paragraph wraps lines
    assert "HEXAGONAL" in flat and "half a step_x" in flat and "step_y = step_x·√3/2" in flat
    assert "odd_row_shift" in flat
    assert "X = column·step," not in flat
    assert "Conventions:" in text                      # the rest of the README is still there


def _assert_square_readme(path):
    text = _readme(path)
    assert "X = column·step," in text and "HEXAGONAL" not in text


def test_light_batch_unconfirmed_shift_carries_its_sentence(tmp_path):
    path = _light(tmp_path, _hex(0.5, confirmed=False))
    with h5py.File(path, "r") as f:
        text = _attr(f["Indexing"].attrs["odd_row_shift_assumed"])
        assert bool(f["Indexing"].attrs["shift_confirmed"]) is False
    assert "assumption" in text and "to the right" in text
    (tmp_path / "m").mkdir()
    path = _light(tmp_path / "m", _hex(-0.5, confirmed=False))
    with h5py.File(path, "r") as f:
        assert "to the left" in _attr(f["Indexing"].attrs["odd_row_shift_assumed"])


# ---------------------------------------------------------------------- rich, batch

def test_rich_batch_stamp(tmp_path):
    geometry = _hex(0.5)
    path = _rich(tmp_path, geometry)
    _assert_index_stamp(path, geometry)
    with h5py.File(path, "r") as f:
        assert json.loads(_attr(f["Indexing/Citations"].attrs["caveats"])) == CAVEATS
    _assert_hex_readme(path)


def test_rich_batch_of_a_square_scan_keeps_its_readme(tmp_path):
    from backend.api.services.result_exporter import README_TEXT
    path = _rich(tmp_path, ScanGeometry.square(3, 4, 2.0))
    _assert_square_readme(path)
    assert _readme(path) == README_TEXT                         # the text of every earlier file


def test_rich_batch_refuses_a_hexagonal_source_under_a_square_checkpoint(tmp_path):
    """A square result over a copied file that holds a hexagonal header would keep the old stamp."""
    source = tmp_path / "hex_source.h5"
    oh.write_fresh_file(source, np.zeros((12, 5, 5), np.uint8), _hex(0.5))
    assert oh.is_native_hex_file(source)
    cp = _checkpoint(tmp_path, source, None)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="hexagonal"):
        export_result_h5(str(source), cp, str(out))
    assert not out.exists() or list(out.iterdir()) == []          # nothing, not even a .part file


def test_rich_batch_of_a_hexagonal_source_under_a_hexagonal_checkpoint_keeps_both(tmp_path):
    geometry = _hex(0.5)
    source = tmp_path / "hex_source.h5"
    oh.write_fresh_file(source, np.zeros((12, 5, 5), np.uint8), geometry)
    path = _rich(tmp_path, geometry, source=source)
    _assert_index_stamp(path, geometry)
    assert oh.is_native_hex_file(path)
    assert oh.read_header_geometry(path) == geometry


def test_a_hexagonal_checkpoint_with_another_step_is_refused(tmp_path):
    source = _source(tmp_path)
    cp = _checkpoint(tmp_path, source, _hex())
    for writer in (export_result_h5, export_result_h5_light):
        with pytest.raises(ValueError, match="two grids"):
            writer(str(source), cp, str(tmp_path / "out"), step_size=1.0)


# ----------------------------------------------------------------- orientations, trigonal

def test_rich_batch_orientations_are_bit_equal(tmp_path):
    """Source without Manufacturer. The rich writer stores the checkpoint array as it is: bit for
    bit, r * 10 + c at [r, c] (the point they belong to)."""
    path = _rich(tmp_path, _hex())
    rows, cols = np.mgrid[0:3, 0:4]
    with h5py.File(path, "r") as f:
        eul = f["Indexing/PerPhase/Cr2N/euler_angles"][()]
    assert eul.shape == (3, 4, 3)
    np.testing.assert_array_equal(eul[..., 0], (rows * 10 + cols).astype(np.float32))
    np.testing.assert_array_equal(eul[..., 1], np.float32(0.5))
    np.testing.assert_array_equal(eul[..., 2], np.float32(0.25))


def test_light_batch_orientations_hold_the_values_mod_two_pi(tmp_path):
    """Source without Manufacturer (vendor frame identity). The light writer sends the angles
    through a rotation first (the vendor frame step), which reduces the first angle into [0, 2 pi):
    the file holds (r * 10 + c) mod 2 pi at [r, c], by hand 0 1 2 3 / 3.7168 4.7168 5.7168 0.4336 /
    1.1504 2.1504 3.1504 4.1504 (rad), to float32 round-off. NOT bit-equal to the checkpoint."""
    path = _light(tmp_path, _hex())
    rows, cols = np.mgrid[0:3, 0:4]
    with h5py.File(path, "r") as f:
        per_phase = f["Indexing/PerPhase/Cr2N/euler_angles"][()]
        best = f["Indexing/Assignment/euler_angles"][()]
    hand = np.mod(rows * 10 + cols, 2 * np.pi)
    for arr in (per_phase, best):
        assert arr.shape == (3, 4, 3)
        np.testing.assert_allclose(arr[..., 0], hand, atol=1e-5)
        np.testing.assert_allclose(arr[..., 1], 0.5, atol=1e-6)
        np.testing.assert_allclose(arr[..., 2], 0.25, atol=1e-6)


def _reload(path):
    from backend.api.routes.analysis import _load_kikuchipy_rich_h5
    import crystal_symmetry as cs
    xmap = _load_kikuchipy_rich_h5(str(path))
    cs.frame_xmap(xmap)
    return xmap


@pytest.mark.parametrize("writer", ["light", "rich"])
def test_trigonal_setting_survives_hex_h5(tmp_path, writer):
    """Space group 162 reloads with the symmetry of its own setting, -31m, and the file states
    the Hough caveat of that setting."""
    from crystal_symmetry import frame_symmetry
    path = (_light if writer == "light" else _rich)(tmp_path, _hex())
    assert _reload(path).phases[0].point_group is frame_symmetry("-31m")
    with h5py.File(path, "r") as f:
        assert json.loads(_attr(f["Indexing/Citations"].attrs["caveats"]))[0]["code"] == "houghTrigonalSetting"


# ------------------------------------------------------------------------ the route

@pytest.fixture
def route(monkeypatch, tmp_path):
    """The route with a hexagonal 3 x 4 result of space group 162 and a signal of 12 patterns whose
    file is not HDF5 (so the rich export builds a fresh file from the signal)."""
    import kikuchipy as kp
    from diffpy.structure import Lattice, Structure
    from kikuchipy.detectors import EBSDDetector
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    from backend.api.routes import ebsd_viewer
    from backend.api.routes import indexing as indexing_mod
    from backend.api.services import scan_geometry as sg
    from indexing_controller import IndexingMethod, IndexingResult

    geometry = _hex(0.5)
    n = 12
    rows, cols = np.mgrid[0:3, 0:4]
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.deg2rad(np.array([[i * 9.0, 20.0, 30.0] for i in range(n)]))),
        phase_id=np.zeros(n, dtype=np.int32),
        x=cols.ravel().astype(np.float32), y=rows.ravel().astype(np.float32),
        phase_list=PhaseList(phases=[Phase(name="Cr2N", space_group=162,
                                           structure=Structure(lattice=Lattice(4.8, 4.8, 4.5, 90, 90, 120)))]))
    result = IndexingResult(
        xmap=xmap, selection_mask=np.ones(n, dtype=bool), original_shape=(3, 4),
        method=IndexingMethod.SPHERICAL, confidence_scores=np.full(n, 0.8, dtype=np.float32),
        metadata={"step_size_um": 0.25,          # not the scan's step: the stamp's step_x wins
                  "scan_geometry": geometry.to_dict(), "caveats": CAVEATS,
                  "recorded_lattices": {0: [4.8, 4.8, 4.5, 90, 90, 120], 1: [4.8, 4.8, 4.5, 90, 90, 120]}})
    monkeypatch.setitem(indexing_mod._result_registry, "hex_route", result)
    monkeypatch.setattr(indexing_mod, "_active_result_id", "hex_route")

    def signal():
        rng = np.random.default_rng(5)
        sig = kp.signals.EBSD(rng.integers(0, 255, (3, 4, 6, 5)).astype(np.uint8))
        sig.detector = EBSDDetector(shape=(6, 5), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)
        return sig

    up1 = tmp_path / "scan.up1"
    up1.write_bytes(b"not an hdf5 file")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", str(up1), raising=False)
    monkeypatch.setattr(ebsd_viewer, "_get_active_signal", lambda *a, **k: signal())
    monkeypatch.setattr(ebsd_viewer, "_active_dataset", "ds")
    sg.set_geometry("ds", geometry)
    try:
        yield SimpleNamespace(tmp=tmp_path, geometry=geometry, result=result, signal=signal)
    finally:
        sg.clear("ds")
        indexing_mod._result_registry.pop("hex_route", None)


def _post_raw(fmt, out):
    from fastapi.testclient import TestClient
    from backend.api.main import app
    return TestClient(app).post("/api/indexing/export", json={"format": fmt, "filename": str(out)})


def _post(fmt, out):
    resp = _post_raw(fmt, out)
    assert resp.status_code == 200, resp.text
    return out


def test_light_route_hex_layout(route):
    path = _post("h5_light", route.tmp / "light.h5")
    _assert_hex_readme(path)
    _assert_index_stamp(path, route.geometry)               # step_size_um is 2.0, not the 0.25
    with h5py.File(path, "r") as f:
        x, y = f["Indexing/X"], f["Indexing/Y"]
        assert x.dtype == np.float64 and y.dtype == np.float64
        np.testing.assert_array_equal(x[()], X_PLUS)
        np.testing.assert_array_equal(y[()], Y_HAND)
        assert "column * step_size_um" not in x.attrs["description"]
        assert float(f["Detector"].attrs["step_size_um"]) == 2.0
        assert json.loads(_attr(f["Indexing/Citations"].attrs["caveats"])) == CAVEATS


def test_trigonal_setting_survives_the_route(route):
    from crystal_symmetry import frame_symmetry
    for fmt in ("h5_light", "h5"):
        path = _post(fmt, route.tmp / f"{fmt}.h5")
        assert _reload(path).phases[0].point_group is frame_symmetry("-31m"), fmt


def test_fresh_rich_header_for_up1_source(route):
    """Route fmt h5 with a source that is not HDF5: the header of the fresh file is hexagonal."""
    path = _post("h5", route.tmp / "rich.h5")
    _assert_index_stamp(path, route.geometry)
    _assert_hex_readme(path)
    with h5py.File(path, "r") as f:
        hdr = f["Scan1/EBSD/Header"]
        assert bytes(hdr["Grid Type"][()]) == b"HexGrid"
        assert hdr["nRows"][0] == 3 and hdr["nColumns"][0] == 4
        assert hdr["Step X"].dtype == np.float64 and hdr["Step X"][0] == 2.0
        assert hdr["Step Y"].dtype == np.float64 and hdr["Step Y"][0] == SQRT3
        assert hdr["Odd Row Shift"].dtype == np.float64 and hdr["Odd Row Shift"][0] == 0.5
        assert hdr["Shift Confirmed"].dtype == np.int32 and hdr["Shift Confirmed"][0] == 1
        assert f["Scan1/EBSD/Data/Pattern"].shape[0] == 12      # 3 * 4 points, no padding
        assert bytes(f["Version"][()]) == b"Orienta rich export"
        assert bytes(f["Manufacturer"][()]) == b"EDAX"
    assert oh.is_native_hex_file(path)
    assert oh.read_header_geometry(path) == route.geometry


@pytest.mark.parametrize("rows, cols", [(4, 4), (4, 3), (2, 6)])
def test_a_signal_of_another_grid_is_refused(route, tmp_path, rows, cols):
    """The signal is 3 rows x 4 columns (12 patterns); 4 x 3 and 2 x 6 also hold 12 points."""
    from backend.api.routes.indexing import _write_fresh_edax_h5
    out = tmp_path / "mismatch.h5"
    other = ScanGeometry.hex(rows, cols, 2.0, shift_confirmed=True)
    with pytest.raises(GeometryMismatchError, match=f"{rows} x {cols}"):
        _write_fresh_edax_h5(str(out), route.signal(), scan_geometry=other)
    assert not out.exists()


# ------------------------------------- the stamp of the result, and the header of the dataset

@pytest.mark.parametrize("fmt", ["h5_light", "h5"])
@pytest.mark.parametrize("change, words", [
    ({"odd_row_shift": 0.7}, "cannot be used"),              # a hex scan shifts by +-0.5 steps
    ({"steps_known": False}, "not known"),                   # placeholder steps
    ({"n_rows": 4, "n_cols": 3}, "4 x 3 points, its grid is 3 x 4"),    # not original_shape (3, 4)
])
def test_an_unusable_hexagonal_stamp_is_refused_and_nothing_is_written(route, fmt, change, words):
    route.result.metadata["scan_geometry"] = {**_hex().to_dict(), **change}
    out = route.tmp / "refused.h5"
    resp = _post_raw(fmt, out)
    assert resp.status_code == 400, resp.text
    assert words in resp.json()["detail"]
    assert not out.exists()


def test_a_result_on_a_resampled_map_gets_no_hexagonal_block(route):
    """A hexagonal file resampled to a square grid: its stamp still says hex, but the map is
    square. Its dataset's geometry is square too. Hand values: X = column * 0.25 (the step of the
    result), float32."""
    from backend.api.services import scan_geometry as sg
    route.result.metadata["hex_resampled"] = True
    sg.set_geometry("ds", ScanGeometry.square(3, 4, 0.25))
    light = _post("h5_light", route.tmp / "resampled_light.h5")
    rich = _post("h5", route.tmp / "resampled_rich.h5")
    for path in (light, rich):
        with h5py.File(path, "r") as f:
            assert not (STAMP_KEYS & set(f["Indexing"].attrs)), path
        _assert_square_readme(path)
    with h5py.File(light, "r") as f:
        assert f["Indexing/X"].dtype == np.float32
        np.testing.assert_array_equal(f["Indexing/X"][()], np.tile(np.arange(4) * 0.25, (3, 1)))
    with h5py.File(rich, "r") as f:
        assert bytes(f["Scan1/EBSD/Header/Grid Type"][()]) == b"SqrGrid"


@pytest.mark.parametrize("dataset, stamp", [
    (ScanGeometry.square(3, 4, 2.0), "hex"),         # header would be square, /Indexing hexagonal
    (_hex(), "square"),                              # header hexagonal, /Indexing square
    (_hex(), None),                                  # header hexagonal, a result without a stamp
])
def test_a_header_and_a_stamp_that_disagree_are_refused(route, dataset, stamp):
    from backend.api.services import scan_geometry as sg
    sg.set_geometry("ds", dataset)
    route.result.metadata["scan_geometry"] = {
        "hex": _hex().to_dict(), "square": ScanGeometry.square(3, 4, 2.0).to_dict(), None: None}[stamp]
    out = route.tmp / "disagree.h5"
    resp = _post_raw("h5", out)
    assert resp.status_code == 400, resp.text
    assert "two different scans" in resp.json()["detail"]
    assert not out.exists()


def _native_source(route, monkeypatch, name="hex_source.h5"):
    """The route's source file is a native hexagonal Orienta file (an HDF5 source, so it is copied)."""
    from backend.api.routes import ebsd_viewer
    source = route.tmp / name
    oh.write_fresh_file(source, np.zeros((12, 6, 5), np.uint8), route.geometry)
    assert oh.is_native_hex_file(source)
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", str(source), raising=False)
    return source


@pytest.mark.parametrize("stamp", [None, "square", "resampled"])
def test_a_native_hex_source_under_a_result_without_a_hex_stamp_is_refused(route, monkeypatch, stamp):
    """The copy would keep the source's hexagonal header under a square result: the same refusal as
    the batch writer's, with nothing written."""
    _native_source(route, monkeypatch)
    route.result.metadata["scan_geometry"] = (
        None if stamp is None else ScanGeometry.square(3, 4, 2.0).to_dict() if stamp == "square"
        else _hex().to_dict())
    if stamp == "resampled":                       # a hex stamp on a map that was resampled: no hex block
        route.result.metadata["hex_resampled"] = True
    out = route.tmp / "refused.h5"
    resp = _post_raw("h5", out)
    assert resp.status_code == 400, resp.text
    assert "hexagonal Orienta file" in resp.json()["detail"]
    assert not out.exists()


def test_a_native_hex_source_under_a_hex_result_keeps_both(route, monkeypatch):
    _native_source(route, monkeypatch)
    path = _post("h5", route.tmp / "kept.h5")
    _assert_index_stamp(path, route.geometry)
    assert oh.is_native_hex_file(path)
    assert oh.read_header_geometry(path) == route.geometry


# ------------------- the scan the rich file is made from is the scan the result was computed on

OTHER_DATASET = "export_other_dataset"
PARENT = ScanGeometry.hex(5, 6, 2.0, odd_row_shift=0.5, shift_confirmed=True)
# a window of 3 x 4 points starting at row 1, column 1: row 1 is an odd row, so the window's own
# odd rows are the parent's even ones and the offset reverses (hand: +0.5 -> -0.5)
CROP_AT_ODD_ROW = PARENT.cropped(1, 1, 3, 4)


def _refused_as_other_dataset(resp, out):
    assert resp.status_code == 400, resp.text
    assert resp.headers.get("X-Orienta-Code") == OTHER_DATASET
    assert "dataset" in resp.json()["detail"] and "Nothing was written" in resp.json()["detail"]
    assert not out.exists()


@pytest.mark.parametrize("dataset", [
    # the same 3 x 4 grid and step, the odd rows on the other side (the crop of an odd row)
    ScanGeometry.hex(3, 4, 2.0, odd_row_shift=-0.5, shift_confirmed=True),
    # another scan of the same shape and shift: the step differs (hand: 2.4 against 2.0)
    ScanGeometry.hex(3, 4, 2.4, odd_row_shift=0.5, shift_confirmed=True),
    # 4 x 3 holds the same 12 points as 3 x 4
    ScanGeometry.hex(4, 3, 2.0, odd_row_shift=0.5, shift_confirmed=True),
])
def test_a_fresh_rich_file_is_refused_when_the_loaded_dataset_is_another_hexagonal_scan(
        route, dataset):
    from backend.api.services import scan_geometry as sg
    sg.set_geometry("ds", dataset)
    out = route.tmp / "other.h5"
    _refused_as_other_dataset(_post_raw("h5", out), out)


def test_a_run_on_a_crop_itself_still_saves_a_fresh_rich_file(route):
    """The stamp of a run on the crop is the geometry of the crop; the dataset holds the same."""
    from backend.api.services import scan_geometry as sg
    route.result.metadata["scan_geometry"] = CROP_AT_ODD_ROW.to_dict()
    sg.set_geometry("ds", CROP_AT_ODD_ROW)
    path = _post("h5", route.tmp / "crop_run.h5")
    assert oh.read_header_geometry(path) == CROP_AT_ODD_ROW
    assert oh.read_index_geometry(path) == CROP_AT_ODD_ROW
    assert CROP_AT_ODD_ROW.odd_row_shift == -0.5                         # by hand, see above


def _native_parent(route, monkeypatch, geometry=PARENT):
    from backend.api.routes import ebsd_viewer
    source = route.tmp / "parent.h5"
    oh.write_fresh_file(source, np.zeros((geometry.n_points, 6, 5), np.uint8), geometry)
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", str(source), raising=False)
    return source


def test_a_copied_native_source_under_a_stamp_of_another_scan_is_refused(route, monkeypatch):
    """The result ran on the 3 x 4 scan of the fixture; the open file is a 5 x 6 scan of the same
    step: its header, patterns and the result's /Indexing would be three different scans."""
    _native_parent(route, monkeypatch)
    out = route.tmp / "copy.h5"
    _refused_as_other_dataset(_post_raw("h5", out), out)


@pytest.mark.parametrize("header", [
    ScanGeometry.hex(3, 4, 2.0, odd_row_shift=-0.5, shift_confirmed=True),     # the other shift
    ScanGeometry.hex(3, 4, 2.4, odd_row_shift=0.5, shift_confirmed=True),      # another step
])
def test_a_copied_native_source_of_the_same_grid_but_another_scan_is_refused(
        route, monkeypatch, header):
    _native_parent(route, monkeypatch, header)
    out = route.tmp / "copy.h5"
    _refused_as_other_dataset(_post_raw("h5", out), out)


def test_a_copied_square_hdf5_source_under_a_hexagonal_stamp_is_refused(route, monkeypatch):
    """An .h5oina or an OIM file that was left open: the copy keeps its square header and the file
    would get a hexagonal /Indexing and a hexagonal README on top."""
    from backend.api.routes import ebsd_viewer
    source = _source(route.tmp, "left_open.h5oina")
    monkeypatch.setattr(ebsd_viewer, "_ebsd_file_path", str(source), raising=False)
    out = route.tmp / "copy.h5"
    _refused_as_other_dataset(_post_raw("h5", out), out)


def test_a_copied_native_source_the_result_ran_a_crop_of_saves(route, monkeypatch):
    """The result ran on the window at row 1, column 1 of the open 5 x 6 file. The copy keeps the
    parent's header, /Indexing states the window and where it sat (offsets 1, 1)."""
    _native_parent(route, monkeypatch)
    route.result.metadata.update({
        "scan_geometry": CROP_AT_ODD_ROW.to_dict(),
        "scan_row_offset": 1, "scan_col_offset": 1, "scan_shape": [5, 6]})
    path = _post("h5", route.tmp / "copy_of_parent.h5")
    assert oh.read_header_geometry(path) == PARENT
    assert oh.read_index_geometry(path) == CROP_AT_ODD_ROW
    with h5py.File(path, "r") as f:
        assert int(f["Indexing"].attrs["scan_row_offset"]) == 1
        assert int(f["Indexing"].attrs["scan_col_offset"]) == 1


@pytest.mark.parametrize("offsets", [(0, 1), (2, 1)])
def test_a_copied_native_source_with_other_row_offsets_than_the_stamp_needs_is_refused(
        route, monkeypatch, offsets):
    """Hand: the window at an even row (0 or 2) keeps the offset +0.5; the stamp says -0.5, which
    only a window at an odd row has."""
    _native_parent(route, monkeypatch)
    route.result.metadata.update({
        "scan_geometry": CROP_AT_ODD_ROW.to_dict(),
        "scan_row_offset": offsets[0], "scan_col_offset": offsets[1], "scan_shape": [5, 6]})
    out = route.tmp / "copy_of_parent.h5"
    _refused_as_other_dataset(_post_raw("h5", out), out)


def test_the_column_offset_does_not_change_the_geometry_of_a_window(route, monkeypatch):
    """Hand: a window at row 1, column 0 is a 3 x 4 grid with the offset reversed, as at column 1
    (the geometry knows no columns' parity): it is the same stamp."""
    _native_parent(route, monkeypatch)
    route.result.metadata.update({
        "scan_geometry": CROP_AT_ODD_ROW.to_dict(),
        "scan_row_offset": 1, "scan_col_offset": 0, "scan_shape": [5, 6]})
    _post("h5", route.tmp / "copy_of_parent.h5")


# -------------------------------------------------------------------- square: unchanged

def test_square_export_keys_unchanged(tmp_path, route):
    """A square scan gets none of the new keys, and the old ones keep their types."""
    from backend.api.routes.indexing import _write_fresh_edax_h5
    fresh = tmp_path / "square.h5"
    _write_fresh_edax_h5(str(fresh), route.signal(), pc_bruker=(0.5, 0.5, 0.5))
    with h5py.File(fresh, "r") as f:
        # written by hand from the layout of a fresh rich file before hexagonal scans
        assert sorted(f.keys()) == ["Manufacturer", "Scan1", "Version"]
        assert sorted(f["Scan1/EBSD"].keys()) == ["Data", "Header"]
        assert sorted(f["Scan1/EBSD/Header"].keys()) == [
            "Camera Azimuthal Angle", "Camera Elevation Angle", "Grid Type",
            "Pattern Bit Depth", "Pattern Center Calibration", "Pattern Height",
            "Pattern Width", "Sample Tilt", "Step X", "Step Y", "nColumns", "nRows"]
        hdr = f["Scan1/EBSD/Header"]
        assert bytes(hdr["Grid Type"][()]) == b"SqrGrid"
        assert hdr["Step X"].dtype == np.float32 and hdr["Step Y"].dtype == np.float32
        assert not (STAMP_KEYS & set(hdr.attrs))
    assert not oh.is_native_hex_file(fresh)

    square = ScanGeometry.square(3, 4, 2.0)
    outs = []
    for name, geometry in (("stamped", square), ("legacy", None)):          # a square stamp, none
        folder = tmp_path / name
        folder.mkdir()
        source = _source(folder, "Sq.h5oina")
        cp = _checkpoint(folder, source, geometry)
        outs.append(export_result_h5_light(str(source), cp, str(tmp_path / f"{name}_light")))
        outs.append(export_result_h5(str(source), cp, str(tmp_path / f"{name}_rich")))
    for path in outs:
        with h5py.File(path, "r") as f:
            assert not (STAMP_KEYS & set(f["Indexing"].attrs)), path
            assert float(f["Indexing"].attrs["step_size_um"]) == 2.0
    for path in outs[::2]:                                                  # the light files
        with h5py.File(path, "r") as f:
            assert f["Indexing/X"].dtype == np.float32                      # col * step, as before
            np.testing.assert_array_equal(f["Indexing/X"][()], np.tile(np.arange(4) * 2.0, (3, 1)))
            np.testing.assert_array_equal(
                f["Indexing/Y"][()], np.repeat(np.arange(3) * 2.0, 4).reshape(3, 4))
            assert "column" in f["Indexing/X"].attrs["description"]

    # the route, for a result and a dataset that are square
    from backend.api.services import scan_geometry as sg
    square_route = ScanGeometry.square(3, 4, 0.25)
    route.result.metadata["scan_geometry"] = square_route.to_dict()
    sg.set_geometry("ds", square_route)
    for fmt in ("h5_light", "h5"):
        path = _post(fmt, tmp_path / f"route_{fmt}.h5")
        _assert_square_readme(path)
        with h5py.File(path, "r") as f:
            assert not (STAMP_KEYS & set(f["Indexing"].attrs)), fmt
            if fmt == "h5_light":           # the rich route has never stated the step on /Indexing
                assert float(f["Indexing"].attrs["step_size_um"]) == 0.25
            else:
                assert "step_size_um" not in f["Indexing"].attrs
    with h5py.File(tmp_path / "route_h5.h5", "r") as f:
        assert bytes(f["Scan1/EBSD/Header/Grid Type"][()]) == b"SqrGrid"
        assert f["Scan1/EBSD/Header/Step X"].dtype == np.float32
    with h5py.File(tmp_path / "route_h5_light.h5", "r") as f:
        assert f["Indexing/X"].dtype == np.float32
        np.testing.assert_array_equal(f["Indexing/X"][()], np.tile(np.arange(4) * 0.25, (3, 1)))
