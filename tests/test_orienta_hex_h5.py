"""The hexagonal blocks of an Orienta .h5: write, read back, and refuse a damaged stamp."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import orienta_hex_h5 as oh  # noqa: E402
from backend.api.services.scan_geometry import ScanGeometry  # noqa: E402

SQRT3 = math.sqrt(3.0)
HEADER = "Scan1/EBSD/Header"


def _hex_3x4(confirmed=True):
    # step_x 2.0, step_y = 2 * sqrt(3) / 2 by the ScanGeometry.hex default.
    return ScanGeometry.hex(3, 4, 2.0, shift_confirmed=confirmed)


def _write_file(path, geometry, *, data=True, patterns=True):
    """A native file the way the native writer lays it out: header, patterns, /Indexing with a map."""
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("EDAX"))
        f.create_dataset("Version", data=np.bytes_("Orienta rich export"))
        hdr = f.create_group(HEADER)
        oh.write_header_block(hdr, geometry)
        if patterns:
            f.create_dataset("Scan1/EBSD/Data/Pattern",
                             data=np.zeros((geometry.n_points, 2, 2), np.uint8))
        idx = f.create_group("Indexing")
        oh.write_index_block(idx, geometry)
        if data:
            x, y = geometry.positions()
            idx.create_dataset("X", data=x)
            idx.create_dataset("Y", data=y)
    return path


# ------------------------------------------------------------------------------ round trip

@pytest.mark.parametrize("shift", [0.5, -0.5])
@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.parametrize("step_x", [0.1, 2.0, 0.15])
def test_block_round_trip_is_exact(tmp_path, step_x, confirmed, shift):
    g = ScanGeometry.hex(3, 4, step_x, odd_row_shift=shift, shift_confirmed=confirmed)
    path = _write_file(tmp_path / "a.h5", g)
    for back in (oh.read_header_geometry(path), oh.read_index_geometry(path)):
        assert back == g                                   # every field, bit for bit
        assert back.step_x == step_x                       # the literal, not the written value
        assert back.step_y == step_x * SQRT3 / 2.0         # hand: rows sqrt(3)/2 steps apart
        assert back.shift_confirmed is confirmed           # False stays False
        assert back.odd_row_shift == shift
    with h5py.File(path, "r") as f:
        # An unconfirmed offset is stated on both blocks, a confirmed one on neither.
        assert (oh.ASSUMED_ATTR in f[HEADER].attrs) is (not confirmed)
        assert (oh.ASSUMED_ATTR in f["Indexing"].attrs) is (not confirmed)
        if not confirmed:
            assert "assumption" in f["Indexing"].attrs[oh.ASSUMED_ATTR]
            assert "assumption" in f[HEADER].attrs[oh.ASSUMED_ATTR]


def test_the_stored_values_are_the_documented_ones(tmp_path):
    # Hand values of the 3 x 4 scan with step_x 2.0 and a confirmed +0.5 shift.
    path = _write_file(tmp_path / "a.h5", _hex_3x4())
    with h5py.File(path, "r") as f:
        hdr = f[HEADER]
        assert hdr["Grid Type"][()] == b"HexGrid"
        assert hdr["nRows"][()].tolist() == [3] and hdr["nColumns"][()].tolist() == [4]
        assert hdr["Step X"].dtype == np.float64 and hdr["Step X"][()].tolist() == [2.0]
        assert hdr["Step Y"].dtype == np.float64
        assert hdr["Step Y"][0] == pytest.approx(1.7320508075688772)     # 2 * sqrt(3) / 2
        assert hdr["Odd Row Shift"].dtype == np.float64 and hdr["Odd Row Shift"][()].tolist() == [0.5]
        assert hdr["Shift Confirmed"].dtype == np.int32 and hdr["Shift Confirmed"][()].tolist() == [1]
        idx = f["Indexing"].attrs
        assert idx["grid_type"] == "HexGrid"
        assert idx["odd_row_shift"] == 0.5 and bool(idx["shift_confirmed"]) is True
        assert idx["step_x"] == 2.0 and idx["step_size_um"] == 2.0
        assert idx["grid_shape"].tolist() == [3, 4]


def test_positions_read_back_by_hand(tmp_path):
    x, y = oh.read_index_geometry(_write_file(tmp_path / "a.h5", _hex_3x4())).positions()
    assert x.tolist() == [[0, 2, 4, 6], [1, 3, 5, 7], [0, 2, 4, 6]]      # hand
    assert y[:, 0].tolist() == pytest.approx([0.0, 1.7320508, 3.4641016])


def test_odd_crop_shift_minus_half(tmp_path):
    # A 5 x 4 parent, +0.5, step_x 2. Window rows 1..3 (three rows from row 1): parent row 1
    # sits at x = 1,3,5,7, row 2 at 0,2,4,6, row 3 at 1,3,5,7. The window starts at its own
    # x = 0, so subtract 1: [0,2,4,6], [-1,1,3,5], [0,2,4,6] (hand, not from cropped()).
    parent = ScanGeometry.hex(5, 4, 2.0, shift_confirmed=True)
    crop = parent.cropped(1, 0, 3, 4)
    path = _write_file(tmp_path / "crop.h5", crop)
    for back in (oh.read_header_geometry(path), oh.read_index_geometry(path)):
        assert back.odd_row_shift == -0.5
        assert back.grid_shape == (3, 4)
        x, _ = back.positions()
        assert x.tolist() == [[0, 2, 4, 6], [-1, 1, 3, 5], [0, 2, 4, 6]]
    with h5py.File(path, "r") as f:
        assert f[HEADER]["Odd Row Shift"][()].tolist() == [-0.5]
        assert f["Indexing"].attrs["odd_row_shift"] == -0.5


# --------------------------------------------------------------------------- damaged stamps

HEADER_KEYS = ["Grid Type", "nRows", "nColumns", "Step X", "Step Y", "Odd Row Shift", "Shift Confirmed"]
INDEX_KEYS = ["grid_type", "grid_shape", "step_x", "step_y", "odd_row_shift", "shift_confirmed"]


@pytest.mark.parametrize("key", HEADER_KEYS)
def test_malformed_stamp_refused_header_missing_key(tmp_path, key):
    path = _write_file(tmp_path / "a.h5", _hex_3x4())
    with h5py.File(path, "a") as f:
        del f[HEADER][key]
    with pytest.raises(ValueError):
        oh.read_header_geometry(path)


@pytest.mark.parametrize("key", INDEX_KEYS)
def test_malformed_stamp_refused_index_missing_key(tmp_path, key):
    path = _write_file(tmp_path / "a.h5", _hex_3x4())
    with h5py.File(path, "a") as f:
        del f["Indexing"].attrs[key]
    with pytest.raises(ValueError):                 # also for grid_type: never read as square
        oh.read_index_geometry(path)


@pytest.mark.parametrize("shift", [0.0, 0.3, 1.0, -0.25])
def test_malformed_stamp_refused_bad_shift(tmp_path, shift):
    path = _write_file(tmp_path / "a.h5", _hex_3x4())
    with h5py.File(path, "a") as f:
        f[HEADER]["Odd Row Shift"][0] = shift
        f["Indexing"].attrs["odd_row_shift"] = shift
    with pytest.raises(ValueError):
        oh.read_header_geometry(path)
    with pytest.raises(ValueError):
        oh.read_index_geometry(path)


def test_malformed_stamp_refused_bad_ratio(tmp_path):
    path = _write_file(tmp_path / "a.h5", _hex_3x4())
    with h5py.File(path, "a") as f:
        f[HEADER]["Step Y"][0] = 2.0                # a square lattice's row spacing (hand: 1.732)
        f["Indexing"].attrs["step_y"] = 2.0
    with pytest.raises(ValueError):
        oh.read_header_geometry(path)
    with pytest.raises(ValueError):
        oh.read_index_geometry(path)


def test_malformed_stamp_refused_unknown_grid_type(tmp_path):
    path = _write_file(tmp_path / "a.h5", _hex_3x4())
    with h5py.File(path, "a") as f:
        del f[HEADER]["Grid Type"]
        f[HEADER].create_dataset("Grid Type", data=np.bytes_("TriGrid"))
        f["Indexing"].attrs["grid_type"] = "TriGrid"
    with pytest.raises(ValueError):
        oh.read_header_geometry(path)
    with pytest.raises(ValueError):
        oh.read_index_geometry(path)


@pytest.mark.parametrize("stored", [(4, 3), (3, 5), (12, 1)])
def test_a_stored_shape_that_disagrees_with_the_data_is_refused(tmp_path, stored):
    # The attribute says one shape, the stored X/Y map is 3 x 4: neither can be trusted.
    path = _write_file(tmp_path / "a.h5", _hex_3x4())
    with h5py.File(path, "a") as f:
        f["Indexing"].attrs["grid_shape"] = list(stored)
    with pytest.raises(ValueError, match="grid_shape"):
        oh.read_index_geometry(path)


def test_the_shape_comes_from_the_phase_maps_when_there_is_no_xy(tmp_path):
    path = _write_file(tmp_path / "a.h5", _hex_3x4(), data=False)
    with h5py.File(path, "a") as f:
        f.create_dataset("Indexing/PerPhase/Al/confidence_index", data=np.zeros((3, 4), np.float32))
    assert oh.read_index_geometry(path) == _hex_3x4()
    with h5py.File(path, "a") as f:
        f["Indexing"].attrs["grid_shape"] = [4, 3]
    with pytest.raises(ValueError, match="grid_shape"):
        oh.read_index_geometry(path)


def test_a_header_whose_point_count_disagrees_with_the_patterns_is_refused(tmp_path):
    path = _write_file(tmp_path / "a.h5", _hex_3x4(), patterns=False)
    assert oh.read_header_geometry(path) == _hex_3x4()      # nothing to compare with: it reads
    with h5py.File(path, "a") as f:
        f.create_dataset("Scan1/EBSD/Data/Pattern", data=np.zeros((11, 2, 2), np.uint8))
    with pytest.raises(ValueError, match="patterns"):
        oh.read_header_geometry(path)


# ------------------------------------------------------------------------------ square

def test_square_writes_nothing(tmp_path):
    sq = ScanGeometry.square(3, 4, 1.5)
    with h5py.File(tmp_path / "s.h5", "w") as f:
        hdr = f.create_group(HEADER)
        idx = f.create_group("Indexing")
        assert oh.write_header_block(hdr, sq) is False
        assert oh.write_index_block(idx, sq) is False
        assert list(hdr.keys()) == [] and list(hdr.attrs.keys()) == []
        assert list(idx.keys()) == [] and list(idx.attrs.keys()) == []
    assert oh.read_index_geometry(tmp_path / "s.h5") is None     # no stamp: a plain square file


def test_a_square_header_reads_as_none_and_is_not_native(tmp_path):
    path = tmp_path / "s.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("EDAX"))
        f.create_dataset("Version", data=np.bytes_("Orienta rich export"))
        hdr = f.create_group(HEADER)
        hdr.create_dataset("Grid Type", data=np.bytes_("SqrGrid"))
        f.create_group("Indexing")
    assert oh.read_header_geometry(path) is None
    assert oh.is_native_hex_file(path) is False


# ------------------------------------------------------------------------- is_native_hex_file

def test_is_native_hex_file(tmp_path):
    good = _write_file(tmp_path / "good.h5", _hex_3x4())
    assert oh.is_native_hex_file(good) is True

    def broken(name, mutate):
        p = _write_file(tmp_path / name, _hex_3x4())
        with h5py.File(p, "a") as f:
            mutate(f)
        return p

    def other_version(f):
        del f["Version"]
        f.create_dataset("Version", data=np.bytes_("EDAX OIM"))

    def other_maker(f):
        del f["Manufacturer"]
        f.create_dataset("Manufacturer", data=np.bytes_("Bruker"))

    assert oh.is_native_hex_file(broken("v.h5", other_version)) is False
    assert oh.is_native_hex_file(broken("m.h5", other_maker)) is False
    # (v.h5 above is a padded OIM copy: HexGrid header, but OIM's Version, so it is not ours.)

    # A square header that still holds an Odd Row Shift entry is not a hexagonal file.
    def square_with_shift(f):
        del f[HEADER]["Grid Type"]
        f[HEADER].create_dataset("Grid Type", data=np.bytes_("SqrGrid"))

    assert oh.is_native_hex_file(broken("q.h5", square_with_shift)) is False
    text = tmp_path / "t.h5"
    text.write_text("not hdf5")
    assert oh.is_native_hex_file(text) is False
    assert oh.is_native_hex_file(tmp_path / "missing.h5") is False


@pytest.mark.parametrize("key", ["Odd Row Shift", "Shift Confirmed", "Step X"])
def test_a_native_file_that_lost_a_header_key_is_native_and_raises(tmp_path, key):
    # Otherwise it would go to the padded OIM loader and be read as a wrong layout, silently.
    path = _write_file(tmp_path / "a.h5", _hex_3x4())
    with h5py.File(path, "a") as f:
        del f[HEADER][key]
    assert oh.is_native_hex_file(path) is True
    with pytest.raises(ValueError, match="lacks|no "):
        oh.read_header_geometry(path)


# ------------------------------------------------- what each reader returns or raises, per case

def test_index_without_a_stamp_but_with_legacy_attrs_is_none(tmp_path):
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as f:
        idx = f.create_group("Indexing")
        idx.attrs["grid_shape"] = [3, 4]
        idx.attrs["step_size_um"] = 1.5
        idx.attrs["method"] = "dictionary"
        idx.create_dataset("X", data=np.zeros((3, 4)))
    assert oh.read_index_geometry(path) is None


def test_file_without_indexing_raises_in_the_index_reader(tmp_path):
    path = tmp_path / "noidx.h5"
    with h5py.File(path, "w") as f:
        f.create_group(HEADER)
    with pytest.raises(ValueError, match="no /Indexing"):
        oh.read_index_geometry(path)


def test_file_without_a_header_raises_in_the_header_reader(tmp_path):
    path = tmp_path / "nohdr.h5"
    with h5py.File(path, "w") as f:
        f.create_group("Indexing")
    with pytest.raises(ValueError, match="Header"):
        oh.read_header_geometry(path)


def test_an_oim_style_hex_header_raises_in_the_header_reader(tmp_path):
    # OIM states the grid, shape and steps (float32) but neither an offset direction nor
    # a confirmation: Orienta will not guess one.
    path = tmp_path / "oim.h5"
    with h5py.File(path, "w") as f:
        hdr = f.create_group(HEADER)
        hdr.create_dataset("Grid Type", data=np.bytes_("HexGrid"))
        hdr.create_dataset("nRows", data=np.array([3], np.int32))
        hdr.create_dataset("nColumns", data=np.array([4], np.int32))
        hdr.create_dataset("Step X", data=np.array([0.15], np.float32))
        hdr.create_dataset("Step Y", data=np.array([0.1299038], np.float32))
    with pytest.raises(ValueError, match="odd_row_shift"):
        oh.read_header_geometry(path)


# ----------------------------------------------- which stored arrays give the independent shape

def _stamped_file(path, add_maps):
    """A stamped /Indexing of a 3 x 4 hex scan with no X/Y; ``add_maps(idx)`` stores the maps."""
    with h5py.File(path, "w") as f:
        idx = f.create_group("Indexing")
        oh.write_index_block(idx, _hex_3x4())
        add_maps(idx)
    return path


@pytest.mark.parametrize("name,dtype", [("selection_mask", bool), ("confidence_index", np.float32)])
def test_single_phase_rich_layout_gives_the_shape(tmp_path, name, dtype):
    # The route writes /Indexing/confidence_index, phase_id, selection_mask and euler_angles.
    def maps(idx):
        idx.create_dataset(name, data=np.zeros((3, 4), dtype))
        idx.create_dataset("euler_angles", data=np.zeros((3, 4, 3), np.float32))
    path = _stamped_file(tmp_path / "a.h5", maps)
    assert oh.read_index_geometry(path) == _hex_3x4()
    with h5py.File(path, "a") as f:
        f["Indexing"].attrs["grid_shape"] = [4, 3]          # the stamp disagrees with the map
    with pytest.raises(ValueError, match="grid_shape"):
        oh.read_index_geometry(path)


def test_a_flat_per_phase_confidence_index_is_never_a_shape_source(tmp_path):
    # 1-D, 12 entries: it cannot tell 3 x 4 from 4 x 3, so the stamp is not compared with it.
    def maps(idx):
        idx.create_dataset("PerPhase/Al/confidence_index", data=np.zeros(12, np.float32))
    path = _stamped_file(tmp_path / "a.h5", maps)
    with h5py.File(path, "a") as f:
        f["Indexing"].attrs["grid_shape"] = [4, 3]
    # Nothing to compare: the attribute is the only shape. (Pins that no 1-D array is read as one.)
    assert oh.read_index_geometry(path).grid_shape == (4, 3)


def test_a_flat_phase_map_does_not_hide_a_2d_source_behind_it(tmp_path):
    def maps(idx):
        idx.create_dataset("PerPhase/Al/confidence_index", data=np.zeros(12, np.float32))
        idx.create_dataset("PerPhase/Fe/confidence_index", data=np.zeros((3, 4), np.float32))
    path = _stamped_file(tmp_path / "a.h5", maps)
    assert oh.read_index_geometry(path) == _hex_3x4()
    with h5py.File(path, "a") as f:
        f["Indexing"].attrs["grid_shape"] = [4, 3]
    with pytest.raises(ValueError, match="grid_shape"):
        oh.read_index_geometry(path)
