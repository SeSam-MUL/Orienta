"""Save crop of a hexagonal scan: ``crop_export.write_cropped_hex_h5``.

The writer takes the crop's signal and the geometry registered for the crop and writes an Orienta
``.h5`` that states the crop's own row offset. Hand scan used throughout: a 5 x 4 parent (rows x
columns), step_x 2.0 um, odd rows shifted by +0.5 step, step_y = 2 * sqrt(3) / 2 = 1.7320508 um,
patterns 6 x 5 pixels. Pattern [r, c] of the parent holds the value 10 * r + c in every pixel, so a
stored value says by itself which parent point it came from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import crop_export  # noqa: E402
from backend.api.services import orienta_hex_h5 as oh  # noqa: E402
from backend.api.services import scan_geometry as sg  # noqa: E402
from backend.api.services.crop_window import CropWindow  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

N_ROWS, N_COLS = 5, 4
SHAPE = (6, 5)
SQRT3 = 1.7320508075688772          # 2 * sqrt(3) / 2, by hand
PC = (0.51, 0.42, 0.63)
TILTS = {"sample_tilt": 70.0, "tilt": 4.0, "azimuthal": 0.0}


@pytest.fixture(autouse=True)
def clean_registry():
    sg.clear_all()
    yield
    sg.clear_all()


def _parent_geometry(confirmed=True):
    return sg.ScanGeometry.hex(N_ROWS, N_COLS, 2.0, odd_row_shift=0.5, shift_confirmed=confirmed)


def _signal(dtype=np.uint8, base=0):
    import kikuchipy as kp
    values = (10 * np.arange(N_ROWS)[:, None] + np.arange(N_COLS)[None, :]) + base
    data = np.broadcast_to(values[:, :, None, None], (N_ROWS, N_COLS) + SHAPE).astype(dtype)
    return kp.signals.EBSD(data.copy())


def _cut(parent_signal, parent_name, name, row0, col0, rows, cols, mask=None):
    """What POST /api/ebsd/crop does: slice the signal, register the cut geometry."""
    window = CropWindow("parent.h5", row0, col0, rows, cols, parent_signal_shape(parent_signal),
                        nav_mask=mask)
    cut = parent_signal.inav[col0:col0 + cols, row0:row0 + rows]
    geometry = sg.derive_cropped(name, parent_name, window)
    return cut, window, geometry


def parent_signal_shape(signal):
    nav = signal.axes_manager.navigation_shape        # hyperspy order: (x, y)
    return (int(nav[1]), int(nav[0]))


def _write(tmp_path, signal, window, geometry, name="crop.h5", **kwargs):
    out = tmp_path / name
    report = crop_export.write_cropped_hex_h5(
        signal, str(out), window, geometry, pc_bruker=PC, tilts=TILTS, **kwargs)
    return out, report


def _stored(path):
    with h5py.File(path, "r") as f:
        return f[oh.PATTERN_PATH][()]


def test_even_row0_keeps_plus_half(tmp_path):
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(), "parent", "c", 0, 1, 3, 3)
    out, report = _write(tmp_path, signal, window, geometry)

    stamped = oh.read_header_geometry(out)
    assert (stamped.n_rows, stamped.n_cols) == (3, 3)
    assert stamped.odd_row_shift == 0.5 and stamped.shift_confirmed is True    # row0 = 0: no reversal
    assert stamped.step_x == 2.0 and stamped.step_y == pytest.approx(SQRT3)
    # parent rows 0-2, columns 1-3: values 10 * r + c, in file order
    assert _stored(out)[:, 0, 0].tolist() == [1, 2, 3, 11, 12, 13, 21, 22, 23]
    assert report["hex"] is True and report["odd_row_shift"] == 0.5
    assert report["n_points"] == 9 and report["grid_shape"] == [3, 3]
    assert (report["datasets_cut"], report["datasets_copied"],
            report["electron_images_cut"], report["electron_images_full"]) == (0, 0, 0, 0)
    assert oh.is_native_hex_file(out)
    assert not list(tmp_path.glob("*.part"))


def test_a_window_without_a_source_file_does_not_write_the_word_None(tmp_path):
    """A crop cut from patterns held in memory may have no source file; the provenance then says
    nothing about one (an empty string), not the text "None"."""
    from dataclasses import replace
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(), "parent", "c", 0, 0, 3, 3)
    out, _ = _write(tmp_path, signal, replace(window, source_file=None), geometry)
    with h5py.File(out, "r") as f:
        stored = f["CropProvenance"].attrs["source_file"]
    assert stored != "None"
    assert stored == ""
    # and a window that has one still states it
    out2, _ = _write(tmp_path, signal, window, geometry, "with_source.h5")
    with h5py.File(out2, "r") as f:
        assert f["CropProvenance"].attrs["source_file"] == "parent.h5"


def test_odd_row0_reverses_the_offset_and_an_assumed_direction_stays_assumed(tmp_path):
    sg.set_geometry("parent", _parent_geometry(confirmed=False))
    signal, window, geometry = _cut(_signal(), "parent", "c", 1, 0, 3, 4)
    out, report = _write(tmp_path, signal, window, geometry)

    stamped = oh.read_header_geometry(out)
    assert stamped.odd_row_shift == -0.5               # parent row 1 is the crop's row 0
    assert stamped.shift_confirmed is False
    assert report["odd_row_shift"] == -0.5 and report["shift_confirmed"] is False


def test_nested_crop_follows_registry(tmp_path):
    """Rows 1-3 of the parent start on an odd row: -0.5. Row 1 of THAT crop starts on an odd row
    again, so the offset reverses back to +0.5 -- from the registry's geometry, not from a constant."""
    sg.set_geometry("parent", _parent_geometry())
    first, w1, g1 = _cut(_signal(), "parent", "c1", 1, 0, 3, 4)
    out1, _ = _write(tmp_path, first, w1, g1, "c1.h5")
    assert oh.read_header_geometry(out1).odd_row_shift == -0.5

    second, w2, g2 = _cut(first, "c1", "c2", 1, 1, 2, 3)
    assert g2 is sg.geometry_for("c2")
    out2, _ = _write(tmp_path, second, w2, sg.geometry_for("c2"), "c2.h5")
    stamped = oh.read_header_geometry(out2)
    assert (stamped.n_rows, stamped.n_cols) == (2, 3)
    assert stamped.odd_row_shift == 0.5
    # c2 row 0 is parent row 2, columns 1-3; row 1 is parent row 3
    assert _stored(out2)[:, 0, 0].tolist() == [21, 22, 23, 31, 32, 33]


def test_mask_recorded(tmp_path):
    sg.set_geometry("parent", _parent_geometry())
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, :] = True
    mask[0, 0] = True
    signal, window, geometry = _cut(_signal(), "parent", "c", 0, 0, 3, 3, mask=mask)
    out, report = _write(tmp_path, signal, window, geometry)

    with h5py.File(out, "r") as f:
        prov = f["CropProvenance"]
        assert np.array_equal(prov["nav_mask"][()], mask)
        assert (int(prov.attrs["row0"]), int(prov.attrs["rows"]), int(prov.attrs["cols"])) == (0, 3, 3)
        assert int(prov.attrs["n_selected"]) == 4
        assert list(prov.attrs["original_shape"]) == [5, 4]
        assert str(prov.attrs["source_file"]) == "parent.h5"
        assert f[oh.PATTERN_PATH].shape[0] == 9                    # the whole box is saved
    codes = [n["code"] for n in report["note_items"]]
    assert "cropMaskRecorded" in codes
    mask_note = next(n for n in report["note_items"] if n["code"] == "cropMaskRecorded")
    assert mask_note["params"] == {"points": 9}
    assert "all 9 pixels are saved" in mask_note["text"]


def test_a_crop_without_a_shape_has_no_mask_note_and_says_what_is_not_carried(tmp_path):
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(), "parent", "c", 0, 0, 2, 2)
    out, report = _write(tmp_path, signal, window, geometry)
    codes = [n["code"] for n in report["note_items"]]
    assert codes == ["cropEdsNotCarried"]
    assert report["notes"] == [n["text"] for n in report["note_items"]]
    assert "EDS" in report["notes"][0]
    assert report["pc_map_dropped"] is False and report["patterns_converted"] is False
    with h5py.File(out, "r") as f:
        assert "nav_mask" not in f["CropProvenance"]


def test_pc_map_and_a_missing_pc_are_stated(tmp_path):
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(), "parent", "c", 0, 0, 2, 2)
    out = tmp_path / "c.h5"
    report = crop_export.write_cropped_hex_h5(signal, str(out), window, geometry, pc_bruker=None,
                                              tilts=TILTS, pc_map_present=True)
    codes = [n["code"] for n in report["note_items"]]
    assert "cropPcMapDropped" in codes and "cropNoPatternCentre" in codes
    assert report["pc_map_dropped"] is True
    with h5py.File(out, "r") as f:
        assert "Pattern Center Calibration" not in f[oh.HEADER_PATH]


def test_valid_mask_refused(tmp_path):
    from dataclasses import replace
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(), "parent", "c", 0, 0, 2, 2)
    masked = replace(geometry, valid_mask=np.array([[True, False], [True, True]]))
    out = tmp_path / "c.h5"
    with pytest.raises(ValueError, match="valid mask"):
        crop_export.write_cropped_hex_h5(signal, str(out), window, masked, pc_bruker=PC, tilts=TILTS)
    assert not out.exists() and not list(tmp_path.glob("*.part"))


def test_a_square_geometry_or_the_wrong_grid_is_refused_and_leaves_nothing(tmp_path):
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(), "parent", "c", 0, 0, 3, 3)
    out = tmp_path / "c.h5"
    with pytest.raises(ValueError, match="square"):
        crop_export.write_cropped_hex_h5(signal, str(out), window, sg.ScanGeometry.square(3, 3, 1.0))
    wrong = sg.ScanGeometry.hex(2, 3, 2.0, shift_confirmed=True)
    with pytest.raises(ValueError, match="crop window"):
        crop_export.write_cropped_hex_h5(signal, str(out), window, wrong)
    # a geometry that fits the window but not the signal is stopped by the writer of the file
    other = sg.ScanGeometry.hex(3, 3, 2.0, shift_confirmed=True)
    short = _signal().inav[0:3, 0:2]
    with pytest.raises(ValueError):
        crop_export.write_cropped_hex_h5(short, str(out), window, other)
    assert not out.exists() and not list(tmp_path.glob("*.part"))


def test_a_failed_write_keeps_the_file_that_was_there(tmp_path):
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(), "parent", "c", 0, 0, 3, 3)
    out = tmp_path / "c.h5"
    out.write_bytes(b"earlier file")
    short = _signal().inav[0:3, 0:2]
    with pytest.raises(ValueError):
        crop_export.write_cropped_hex_h5(short, str(out), window, geometry)
    assert out.read_bytes() == b"earlier file"
    assert not list(tmp_path.glob("*.part"))


def test_uint16_not_squeezed(tmp_path):
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(np.uint16, base=40000), "parent", "c", 0, 0, 2, 2)
    out, report = _write(tmp_path, signal, window, geometry)
    stored = _stored(out)
    assert stored.dtype == np.uint16
    assert stored[:, 0, 0].tolist() == [40000, 40001, 40010, 40011]     # 40000 + 10 * r + c
    assert report["patterns_converted"] is False


def test_other_dtypes_are_converted_and_the_answer_says_so(tmp_path):
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(np.float32), "parent", "c", 0, 0, 2, 2)
    out, report = _write(tmp_path, signal, window, geometry)
    stored = _stored(out)
    assert stored.dtype == np.uint8
    # rescaled over the crop's range (0..11): the smallest value goes to 0, the largest to 255
    assert stored[0, 0, 0] == 0 and stored[3, 0, 0] == 255
    assert report["patterns_converted"] is True
    note = next(n for n in report["note_items"] if n["code"] == "cropPatternsConverted")
    assert note["params"] == {"dtype": "float32"}


def test_pc_and_tilts_are_stated(tmp_path):
    sg.set_geometry("parent", _parent_geometry())
    signal, window, geometry = _cut(_signal(), "parent", "c", 0, 0, 2, 2)
    out, _ = _write(tmp_path, signal, window, geometry)
    with h5py.File(out, "r") as f:
        hdr = f[oh.HEADER_PATH]
        assert hdr["Sample Tilt"][0] == 70.0 and hdr["Camera Elevation Angle"][0] == 4.0
        cal = hdr["Pattern Center Calibration"]
        # EDAX TSL convention, by hand: x* = PCx, y* = 1 - PCy, z* = PCz * rows / min(rows, cols)
        assert cal["x-star"][0] == pytest.approx(0.51)
        assert cal["y-star"][0] == pytest.approx(0.58)
        assert cal["z-star"][0] == pytest.approx(0.63 * 6 / 5)


def _oim_form_hex_file(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("EDAX"))
        f.create_dataset("Version", data=np.bytes_("OIM Analysis 8.6.0"))
        hdr = f.create_group("Scan1/EBSD/Header")
        hdr.create_dataset("Grid Type", data=np.bytes_("HexGrid"))
        hdr.create_dataset("nColumns", data=np.array([4], np.int32))
        hdr.create_dataset("nRows", data=np.array([4], np.int32))
        hdr.create_dataset("Step X", data=np.array([1.0], np.float32))
        hdr.create_dataset("Step Y", data=np.array([0.866], np.float32))
        data = f.create_group("Scan1/EBSD/Data")
        data.create_dataset("Pattern", data=np.zeros((14, 6, 5), np.uint8))
    return path


def test_oim_form_hex_still_refused(tmp_path):
    src = _oim_form_hex_file(tmp_path / "oim.h5")
    out = tmp_path / "crop.h5oina"
    window = CropWindow(str(src), 0, 0, 2, 2, (4, 4))
    with pytest.raises(ValueError, match="hex"):
        crop_export.write_cropped_h5oina(str(src), str(out), window)
    assert not out.exists()


# ------------------------------------------------------------------ the whole way, by the routes
#
# A 4 x 5 parent, step 1.0, shift +0.5 (confirmed), 6 x 5 pixel patterns: pixel [0, 0] holds the parent
# row and pixel [0, 1] the parent column of the point. Crop rows 1-3, cols 1-3 with POST /api/ebsd/crop,
# save with POST /api/ebsd/crop/export, reload with the loader.
#
# By hand, x = col + 0.5 on an odd row and y = row * 0.8660254 in the parent. The crop starts on parent
# row 1 (odd) at parent col 1, whose x is 1.5; measured from the crop's own corner:
#   crop row 0 = parent row 1: x = 1.5, 2.5, 3.5  -> 0, 1, 2
#   crop row 1 = parent row 2: x = 1, 2, 3        -> -0.5, 0.5, 1.5
#   crop row 2 = parent row 3: x = 1.5, 2.5, 3.5  -> 0, 1, 2
# so the crop's odd row sits half a step LEFT of its even rows: shift -0.5. y = 0, 0.8660254, 1.7320508.

ROUND_PC = (0.51, 0.42, 0.63)
ROUND_SHAPE = (6, 5)
HALF_SQRT3 = 0.8660254037844386        # sqrt(3) / 2, by hand


@pytest.fixture
def routes():
    from fastapi.testclient import TestClient
    from backend.api.main import app
    from tests.apex_helpers import wipe_backend
    wipe_backend()
    yield TestClient(app)
    wipe_backend()


def _round_trip_parent(path):
    n = 4 * 5
    pats = np.zeros((n, ) + ROUND_SHAPE, np.uint8)
    for k in range(n):
        pats[k, 0, 0] = k // 5          # the parent row
        pats[k, 0, 1] = k % 5           # the parent column
    oh.write_fresh_file(path, pats, sg.ScanGeometry.hex(4, 5, 1.0, shift_confirmed=True),
                        pc_bruker=ROUND_PC, tilts={"sample_tilt": 70.0, "tilt": 4.0})
    return path


def _load_by_loader(path):
    from safe_loader import load_ebsd_safe
    return load_ebsd_safe(str(path), verbose=False)


def _save_crop(client, row0, col0, rows, cols, out):
    r = client.post("/api/ebsd/crop", json={"shape": "rect", "row0": row0, "col0": col0,
                                            "rows": rows, "cols": cols})
    assert r.status_code == 200, r.text
    r = client.post("/api/ebsd/crop/export", json={"path": str(out)})
    assert r.status_code == 200, r.text
    return r.json()


def _parents(signal):
    """(parent row, parent col) of every point of a loaded crop, in file order."""
    data = np.asarray(signal.data)
    return [(int(data[r, c, 0, 0]), int(data[r, c, 0, 1]))
            for r in range(data.shape[0]) for c in range(data.shape[1])]


def test_round_trip_positions_and_identity(routes, tmp_path):
    parent = _round_trip_parent(tmp_path / "parent.h5")
    r = routes.post("/api/ebsd/load", json={"path": str(parent)})
    assert r.status_code == 200, r.text
    out = tmp_path / "crop.h5"
    answer = _save_crop(routes, 1, 1, 3, 3, out)
    assert answer["hex"] is True and answer["odd_row_shift"] == -0.5
    assert answer["n_points"] == 9 and answer["n_selected"] == 9

    reloaded = _load_by_loader(out)
    geometry = sg.geometry_of_signal(reloaded)
    assert (geometry.kind, geometry.n_rows, geometry.n_cols) == ("hex", 3, 3)
    assert geometry.odd_row_shift == -0.5 and geometry.shift_confirmed is True
    assert geometry.step_x == 1.0 and geometry.step_y == pytest.approx(HALF_SQRT3, abs=1e-7)
    x, y = geometry.positions()
    np.testing.assert_allclose(x, [[0, 1, 2], [-0.5, 0.5, 1.5], [0, 1, 2]], atol=1e-9)
    np.testing.assert_allclose(y, [[0] * 3, [0.8660254] * 3, [1.7320508] * 3], atol=1e-7)
    # identity: the points are the parent's (1,1) .. (3,3), in file order
    assert _parents(reloaded) == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    np.testing.assert_allclose(np.asarray(reloaded.detector.pc).ravel(), ROUND_PC, atol=1e-6)


def test_the_reloaded_crop_can_be_cut_again(routes, tmp_path):
    """The reloaded crop (shift -0.5) is cut from its local row 1, an odd row again: the offset
    reverses back to +0.5, and the second file gives right positions.

    Local rows 1-2, local cols 0-1 of the reloaded crop are the parent's rows 2-3, cols 1-2. By hand
    (parent x = col + 0.5 on an odd row; origin at parent (2,1), x = 1):
      row 0 = parent row 2: x = 1, 2     -> 0, 1
      row 1 = parent row 3: x = 1.5, 2.5 -> 0.5, 1.5      so the odd row is half a step RIGHT: +0.5
    """
    parent = _round_trip_parent(tmp_path / "parent.h5")
    assert routes.post("/api/ebsd/load", json={"path": str(parent)}).status_code == 200
    first = tmp_path / "crop1.h5"
    _save_crop(routes, 1, 1, 3, 3, first)

    r = routes.post("/api/ebsd/load", json={"path": str(first)})
    assert r.status_code == 200, r.text
    assert r.json()["scan_geometry"]["odd_row_shift"] == -0.5
    second = tmp_path / "crop2.h5"
    answer = _save_crop(routes, 1, 0, 2, 2, second)
    assert answer["odd_row_shift"] == 0.5

    reloaded = _load_by_loader(second)
    geometry = sg.geometry_of_signal(reloaded)
    assert (geometry.n_rows, geometry.n_cols, geometry.odd_row_shift) == (2, 2, 0.5)
    x, y = geometry.positions()
    np.testing.assert_allclose(x, [[0, 1], [0.5, 1.5]], atol=1e-9)
    np.testing.assert_allclose(y, [[0, 0], [0.8660254, 0.8660254]], atol=1e-7)
    assert _parents(reloaded) == [(2, 1), (2, 2), (3, 1), (3, 2)]
    np.testing.assert_allclose(np.asarray(reloaded.detector.pc).ravel(), ROUND_PC, atol=1e-6)
