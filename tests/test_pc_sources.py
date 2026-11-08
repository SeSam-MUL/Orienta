"""Reading the pattern centre of another file: .ang, .osc, EDAX and Oxford .h5, Orienta exports.

Expected values are hand conversions of numbers a file states; the real files (EDAX LoGainNi.h5, the
The Oxford reference scan, an EDAX .osc) are checked against the Bruker values the harness log recorded
for them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.services import pc_sources as src  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _test_data(*parts):
    base = Path(os.environ.get("ORIENTA_TEST_DATA") or ROOT / "Test_data")
    path = base.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"{path.name} is not in {base} (set ORIENTA_TEST_DATA to the folder that holds the test data)")
    return path


# ------------------------------------------------------------------ .ang

ANG = """# TEM_PIXperUM            1.000000
# x-star                  {x}
# y-star                  {y}
# z-star                  {z}
# WorkingDistance         14.000000
#
# Column names: phi1 PHI phi2 x y
  0.5 0.5 0.5 0.0 0.0
"""


def _ang(tmp_path, x="0.507262", y="0.737924", z="0.558489", extra=""):
    p = tmp_path / "scan.ang"
    p.write_text(extra + ANG.format(x=x, y=y, z=z), encoding="utf-8")
    return p


def test_an_ang_header_is_read_as_tsl_and_needs_the_pattern_shape(tmp_path):
    """x* 0.507262, y* 0.737924, z* 0.558489 are the numbers of the EDAX LoGainNi.h5 header; the Bruker PC the
    harness log records for it is (0.5073, 0.2621, 0.5585). By hand for 60 x 60: y = 1 - 0.737924 = 0.262076,
    z unchanged (min side / rows = 1)."""
    got = src.read_pc(_ang(tmp_path), shape=(60, 60))
    assert got.pc_bruker == pytest.approx((0.507262, 0.262076, 0.558489), abs=1e-9)
    assert (got.source_kind, got.convention, got.shape, got.shape_assumed) == ("ang", "tsl", (60, 60), True)
    assert got.notes
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(_ang(tmp_path))
    assert exc.value.code == "needsShape"


def test_zero_stars_and_an_orienta_state_line_mean_no_pc(tmp_path):
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(_ang(tmp_path, "0.000000", "0.000000", "0.000000"), shape=(60, 60))
    assert exc.value.code == "noPcInFile"
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(_ang(tmp_path, "0.0", "0.0", "0.0", extra="# ORIENTA_PC: pc_missing\n"), shape=(60, 60))
    assert exc.value.code == "noPcInFile" and exc.value.params == {"state": "pc_missing"}


def test_an_ang_without_stars_has_no_pc(tmp_path):
    p = tmp_path / "x.ang"
    p.write_text("# TEM_PIXperUM 1.0\n  0.5 0.5 0.5 0.0 0.0\n", encoding="utf-8")
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(p, shape=(60, 60))
    assert exc.value.code == "noPcInFile"


def test_only_the_header_is_read(tmp_path):
    """A star-like line among the data columns is not a header."""
    p = _ang(tmp_path)
    with open(p, "a", encoding="utf-8") as handle:
        handle.write("# x-star 9.0\n")
    assert src.read_pc(p, shape=(60, 60)).stated[0] == pytest.approx(0.507262)


# ------------------------------------------------------------------ HDF5

def _edax(tmp_path, with_shape=True, **stars):
    p = tmp_path / "edax.h5"
    with h5py.File(p, "w") as f:
        f["Manufacturer"] = np.array([b"EDAX"])
        header = f.create_group("Scan 1/EBSD/Header")
        cal = header.create_group("Pattern Center Calibration")
        for k, v in stars.items():
            cal[k] = np.array([v], dtype=np.float32)
        if with_shape:
            header["Pattern Width"] = np.array([156])
            header["Pattern Height"] = np.array([128])
    return p


def test_an_edax_h5_states_its_pattern_shape_and_is_converted_with_it(tmp_path):
    """TSL (0.50, 0.60, 0.70) on 128 x 156 patterns: y = 1 - 0.6 = 0.4, z * min(128, 156) / 128 = 0.7."""
    got = src.read_pc(_edax(tmp_path, **{"x-star": 0.5, "y-star": 0.6, "z-star": 0.7}))
    assert got.pc_bruker == pytest.approx((0.5, 0.4, 0.7), abs=1e-6)
    assert got.shape == (128, 156) and got.shape_assumed is False and got.source_kind == "edax_h5"


def test_an_edax_h5_with_a_partial_calibration_has_no_pc(tmp_path):
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(_edax(tmp_path, **{"x-star": 0.5, "y-star": 0.6}))
    assert exc.value.code == "noPcInFile"


def _oxford(tmp_path, x, y, dd, w=156, h=128):
    p = tmp_path / "ox.h5oina"
    with h5py.File(p, "w") as f:
        data = f.create_group("1/EBSD/Data")
        data["Pattern Center X"] = np.asarray(x, dtype=np.float32)
        data["Pattern Center Y"] = np.asarray(y, dtype=np.float32)
        data["Detector Distance"] = np.asarray(dd, dtype=np.float32)
        header = f.create_group("1/EBSD/Header")
        header["Pattern Width"] = np.array([w])
        header["Pattern Height"] = np.array([h])
    return p


def test_an_oxford_map_is_averaged_and_converted_by_hand(tmp_path):
    """Oxford (0.5026, 0.5520, 0.6940) on 128 x 156 (aspect 1.21875): y = 1 - 0.5520 * 1.21875 = 0.32725,
    z = 0.6940 * 1.21875 = 0.845813; the map's two pixels differ, the PC is their mean."""
    got = src.read_pc(_oxford(tmp_path, [0.5025, 0.5027], [0.5519, 0.5521], [0.6939, 0.6941]))
    assert got.pc_bruker == pytest.approx((0.5026, 0.32725, 0.845813), abs=1e-4)
    assert got.per_pixel is True and got.convention == "oxford" and got.notes


def test_a_constant_oxford_value_is_not_called_a_map(tmp_path):
    got = src.read_pc(_oxford(tmp_path, [0.5, 0.5], [0.55, 0.55], [0.69, 0.69]))
    assert got.per_pixel is False and got.notes == []


def test_an_orienta_export_states_the_pc_its_run_used_in_bruker_numbers(tmp_path):
    p = tmp_path / "export.h5"
    with h5py.File(p, "w") as f:
        det = f.create_group("Detector")
        det["pc"] = np.array([0.5026, 0.3273, 0.8458])
        det.attrs["shape"] = np.array([128, 156])
    got = src.read_pc(p)
    assert got.pc_bruker == pytest.approx((0.5026, 0.3273, 0.8458)) and got.source_kind == "orienta_export"
    assert got.shape == (128, 156) and got.shape_assumed is False


def test_an_orienta_export_of_a_file_without_a_pc_has_none(tmp_path):
    p = tmp_path / "export.h5"
    with h5py.File(p, "w") as f:
        det = f.create_group("Detector")
        det["pc"] = np.array([0.5, 0.5, 0.5])
        det.attrs["pc_missing"] = True
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(p)
    assert exc.value.code == "noPcInFile"


def test_a_non_finite_pc_in_an_orienta_export_has_none(tmp_path):
    """No conversion runs on an export's Bruker numbers, so nothing but this module can refuse a NaN."""
    p = tmp_path / "export.h5"
    with h5py.File(p, "w") as f:
        f.create_group("Detector")["pc"] = np.array([np.nan, 0.5, 0.5])
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(p)
    assert exc.value.code == "noPcInFile"


# ------------------------------------------------------------------ what cannot be read

def test_unreadable_and_foreign_files_are_refused_with_a_code(tmp_path):
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(tmp_path / "nope.ang")
    assert exc.value.code == "fileNotFound"
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(other)
    assert exc.value.code == "unsupportedFile"
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not hdf5")
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(bad)
    assert exc.value.code == "unreadableFile"


# ------------------------------------------------------------------ the shape rule of pc_convention

@pytest.mark.parametrize("shape", [(60.5, 60), (60, 60, 60), (0, 60), (-60, 60), (float("nan"), 60), "60", 60])
def test_a_caller_shape_that_is_not_whole_rows_and_columns_is_refused_not_cut(tmp_path, shape):
    """pc_convention refuses 60.5 or a stack shape; this module must not cut them to 60 or to the first two
    entries before it gets there."""
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(_ang(tmp_path), shape=shape)
    assert exc.value.code == "needsShape"


def test_a_whole_shape_given_as_floats_or_a_list_is_accepted(tmp_path):
    got = src.read_pc(_ang(tmp_path), shape=[60.0, 60.0])
    assert got.shape == (60, 60)


def test_a_non_whole_shape_stated_by_the_file_is_not_cut_but_replaced_by_the_callers(tmp_path):
    """Pattern Height 127.5 is no shape: the caller's (128, 156) is used and flagged as assumed."""
    p = _oxford(tmp_path, [0.5, 0.5], [0.55, 0.55], [0.69, 0.69], h=127.5)
    got = src.read_pc(p, shape=(128, 156))
    assert got.shape == (128, 156) and got.shape_assumed is True
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(p)
    assert exc.value.code == "needsShape"


def test_a_pc_the_conversion_overflows_on_is_refused_as_no_pc(tmp_path):
    """Oxford y = 1.7e308 * 1.21875 is not a float; the convention service says badPc, this module noPcInFile."""
    p = tmp_path / "huge.h5oina"
    with h5py.File(p, "w") as f:
        data = f.create_group("1/EBSD/Data")
        data["Pattern Center X"] = np.array([0.5])
        data["Pattern Center Y"] = np.array([1.7e308])
        data["Detector Distance"] = np.array([0.7])
        header = f.create_group("1/EBSD/Header")
        header["Pattern Width"] = np.array([156])
        header["Pattern Height"] = np.array([128])
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(p)
    assert exc.value.code == "noPcInFile"


# ------------------------------------------------------------------ what a file can get wrong

def _code(path, **kw):
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(path, **kw)
    return exc.value.code


def test_zero_stars_of_an_edax_h5_mean_no_pc(tmp_path):
    """All three stars 0 is the 'never calibrated' header, not the PC (0, 1, 0)."""
    assert _code(_edax(tmp_path, **{"x-star": 0.0, "y-star": 0.0, "z-star": 0.0})) == "noPcInFile"


def test_an_oxford_map_of_unindexed_pixels_only_has_no_pc(tmp_path):
    with pytest.raises(src.PcSourceError) as exc:
        src.read_pc(_oxford(tmp_path, [0, 0], [0, 0], [0, 0]))
    assert exc.value.code == "noPcInFile" and "no pixel of the map" in str(exc.value)


def test_an_infinite_oxford_pixel_is_left_out_not_averaged(tmp_path):
    got = src.read_pc(_oxford(tmp_path, [np.inf, 0.5, 0.5], [0.55] * 3, [0.69] * 3))
    assert got.pc_bruker[0] == pytest.approx(0.5)
    assert any("1 of 3" in n for n in got.notes)


def test_an_oxford_map_without_one_component_is_unreadable(tmp_path):
    p = tmp_path / "nokey.h5oina"
    with h5py.File(p, "w") as f:
        data = f.create_group("1/EBSD/Data")
        data["Pattern Center X"] = np.array([0.5])
        data["Detector Distance"] = np.array([0.7])
    assert _code(p, shape=(60, 60)) == "unreadableFile"


def test_the_oxford_mean_leaves_out_pixels_without_a_pc_and_says_how_many(tmp_path):
    """Five pixels; 0, 3 (NaN x) and 4 (y = 0) have no complete PC, so the mean is over pixels 1 and 2:
    Oxford (0.5, 0.55, 0.69) on 128 x 156: y = 1 - 0.55 * 1.21875 = 0.3296875, z = 0.69 * 1.21875 = 0.8409375."""
    p = _oxford(tmp_path, [0, 0.5, 0.5, np.nan, 0.5], [0, 0.55, 0.55, 0.55, 0], [0, 0.69, 0.69, 0.69, 0.69])
    got = src.read_pc(p)
    assert got.pc_bruker == pytest.approx((0.5, 0.3296875, 0.8409375), abs=1e-6)
    assert got.per_pixel is False
    assert any("3 of 5" in n for n in got.notes)


def test_a_complete_oxford_map_names_no_left_out_pixels(tmp_path):
    got = src.read_pc(_oxford(tmp_path, [0.5025, 0.5027], [0.5519, 0.5521], [0.6939, 0.6941]))
    assert not any("left out" in n for n in got.notes)


def test_a_zero_pc_from_an_osc_means_no_pc(tmp_path, monkeypatch):
    import types
    import edax_up1
    p = tmp_path / "s.osc"
    p.write_bytes(b"x")
    monkeypatch.setattr(edax_up1, "read_osc_metadata", lambda path: types.SimpleNamespace(pc=(0.0, 0.0, 0.0)))
    assert _code(p, shape=(60, 60)) == "noPcInFile"


@pytest.mark.parametrize("make", [
    lambda cal: cal.create_dataset("x-star", shape=(0,), dtype="f4"),     # an empty dataset
    lambda cal: cal.create_group("x-star"),                                # a group, not a dataset
    lambda cal: cal.create_dataset("x-star", data="abc"),                  # a string
])
def test_a_broken_edax_calibration_is_unreadable_not_a_raw_error(tmp_path, make):
    p = tmp_path / "broken.h5"
    with h5py.File(p, "w") as f:
        header = f.create_group("Scan 1/EBSD/Header")
        cal = header.create_group("Pattern Center Calibration")
        make(cal)
        cal["y-star"] = np.array([0.6])
        cal["z-star"] = np.array([0.7])
    assert _code(p, shape=(60, 60)) == "unreadableFile"


def test_a_broken_oxford_map_is_unreadable_not_a_raw_error(tmp_path):
    p = tmp_path / "broken.h5oina"
    with h5py.File(p, "w") as f:
        data = f.create_group("1/EBSD/Data")
        data["Pattern Center X"] = "abc"
        data["Pattern Center Y"] = np.array([0.5])
        data["Detector Distance"] = np.array([0.7])
    assert _code(p, shape=(60, 60)) == "unreadableFile"


def test_an_oxford_map_with_components_of_different_size_is_unreadable(tmp_path):
    assert _code(_oxford(tmp_path, [0.5, 0.5], [0.55], [0.69, 0.69])) == "unreadableFile"


def _deny_reading(monkeypatch, target):
    import builtins
    real = builtins.open

    def fake(file, *args, **kwargs):
        if str(file) == str(target):
            raise PermissionError(13, "denied", str(file))
        return real(file, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", fake)


def test_an_ang_that_cannot_be_opened_is_unreadable_not_a_permission_error(tmp_path, monkeypatch):
    p = _ang(tmp_path)
    _deny_reading(monkeypatch, p)
    assert _code(p, shape=(60, 60)) == "unreadableFile"


def test_an_osc_that_cannot_be_opened_is_unreadable_not_without_a_pc(tmp_path, monkeypatch):
    p = tmp_path / "s.osc"
    p.write_bytes(b"x" * 64)
    _deny_reading(monkeypatch, p)
    assert _code(p, shape=(60, 60)) == "unreadableFile"


def _edax_and_export(tmp_path, star_x=0.5, export=(0.3, 0.3, 0.3)):
    p = _edax(tmp_path, **{"x-star": star_x, "y-star": 0.6, "z-star": 0.7})
    with h5py.File(p, "a") as f:
        f.create_group("Detector")["pc"] = np.array(export)
    return p


def test_an_export_with_a_manufacturer_header_gives_the_header_pc_as_the_loader_does(tmp_path):
    """safe_loader lets the header win over /Detector/pc: TSL (0.5, 0.6, 0.7) on 128 x 156 is (0.5, 0.4, 0.7)."""
    got = src.read_pc(_edax_and_export(tmp_path))
    assert got.pc_bruker == pytest.approx((0.5, 0.4, 0.7), abs=1e-6)
    assert got.source_kind == "edax_h5"


def test_an_oxford_file_with_an_export_group_gives_the_oxford_header(tmp_path):
    p = _oxford(tmp_path, [0.5, 0.5], [0.55, 0.55], [0.69, 0.69])
    with h5py.File(p, "a") as f:
        f.create_group("Detector")["pc"] = np.array([0.3, 0.3, 0.3])
    got = src.read_pc(p)
    assert got.source_kind == "oxford_h5" and got.pc_bruker[0] == pytest.approx(0.5)


def test_a_header_without_a_usable_pc_does_not_fall_back_to_the_export(tmp_path):
    """The loader would not apply /Detector/pc there either, so the file proposes none."""
    p = _edax_and_export(tmp_path, star_x=np.nan)
    assert _code(p) == "noPcInFile"


@pytest.mark.parametrize("pc", [np.full((4, 3), 0.5), np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.5, 0.5])])
def test_an_export_pc_of_any_size_but_three_values_has_none(tmp_path, pc):
    p = tmp_path / "export.h5"
    with h5py.File(p, "w") as f:
        f.create_group("Detector")["pc"] = pc
    assert _code(p) == "noPcInFile"


@pytest.mark.parametrize("which", [0, 1, 2])
def test_a_map_that_varies_in_one_component_only_is_a_map_with_a_spread_noted(tmp_path, which):
    """Each component alone varies by 0.01 (two pixels): the population standard deviation is 0.005."""
    cols = [[0.5, 0.5], [0.55, 0.55], [0.69, 0.69]]
    cols[which] = [cols[which][0], cols[which][0] + 0.01]
    got = src.read_pc(_oxford(tmp_path, *cols))
    assert got.per_pixel is True
    parts = ["0", "0", "0"]
    parts[which] = "0.005"
    expected = f"Standard deviation over the map (Oxford numbers): x {parts[0]}, y {parts[1]}, z {parts[2]}."
    assert [n for n in got.notes if n.startswith("Standard deviation")] == [expected]


def test_a_nan_pixel_does_not_hide_a_varying_map(tmp_path):
    got = src.read_pc(_oxford(tmp_path, [0.5, 0.52, np.nan], [0.55] * 3, [0.69] * 3))
    assert got.per_pixel is True


def test_an_assumed_shape_is_named_with_the_components_that_depend_on_it(tmp_path):
    ang = src.read_pc(_ang(tmp_path), shape=(60, 70)).notes[0]
    assert "60 x 70" in ang and "rows x columns" in ang and "depend" in ang
    ox = src.read_pc(_oxford(tmp_path, [0.5, 0.5], [0.55, 0.55], [0.69, 0.69], h=127.5), shape=(128, 156))
    assert any("128 x 156" in n and "y and z" in n and "depend" in n for n in ox.notes)
    edax = src.read_pc(_edax(tmp_path, with_shape=False, **{"x-star": 0.5, "y-star": 0.6, "z-star": 0.7}),
                       shape=(128, 156))
    assert any("128 x 156" in n and "only z" in n for n in edax.notes)


@pytest.mark.parametrize("shape,z_bruker", [
    ((60, 80), 0.7),             # wider than tall: z* is in units of the shorter side = the height, unchanged
    ((60, 60), 0.7),             # square: unchanged
    ((80, 60), 0.7 * 60 / 80),   # taller than wide: z* is in units of the width 60, so z = 0.7 * 60 / 80 = 0.525
])
def test_a_tsl_note_names_only_z_and_only_for_a_taller_than_wide_pattern(tmp_path, monkeypatch, shape, z_bruker):
    """TSL (0.5, 0.6, 0.7): y = 1 - 0.6 = 0.4 on every shape; z changes only when rows > columns."""
    import types
    import edax_up1
    got = src.read_pc(_ang(tmp_path, x="0.5", y="0.6", z="0.7"), shape=shape)
    assert got.pc_bruker == pytest.approx((0.5, 0.4, z_bruker), abs=1e-6)
    note = got.notes[0]
    assert "only z" in note and "taller than wide" in note and "y and z" not in note
    osc = tmp_path / "s.osc"
    osc.write_bytes(b"x")
    monkeypatch.setattr(edax_up1, "read_osc_metadata", lambda path: types.SimpleNamespace(pc=(0.5, 0.6, 0.7)))
    got_osc = src.read_pc(osc, shape=shape)
    assert got_osc.pc_bruker == pytest.approx((0.5, 0.4, z_bruker), abs=1e-6)
    assert "only z" in got_osc.notes[0] and "y and z" not in got_osc.notes[0]


@pytest.mark.parametrize("shape,y_bruker,z_bruker", [
    ((60, 80), 1 - 0.55 * 80 / 60, 0.69 * 80 / 60),      # Oxford y and z are in units of the width: both move
    ((80, 60), 1 - 0.55 * 60 / 80, 0.69 * 60 / 80),
])
def test_an_oxford_note_names_y_and_z(tmp_path, shape, y_bruker, z_bruker):
    got = src.read_pc(_oxford(tmp_path, [0.5, 0.5], [0.55, 0.55], [0.69, 0.69], h=127.5), shape=shape)
    assert got.pc_bruker == pytest.approx((0.5, y_bruker, z_bruker), abs=1e-5)
    assert any("y and z" in n and "depend" in n for n in got.notes)


def test_the_osc_note_names_the_assumed_shape_too(tmp_path, monkeypatch):
    import types
    import edax_up1
    p = tmp_path / "s.osc"
    p.write_bytes(b"x")
    monkeypatch.setattr(edax_up1, "read_osc_metadata", lambda path: types.SimpleNamespace(pc=(0.5, 0.6, 0.7)))
    got = src.read_pc(p, shape=(79, 80))
    assert got.pc_bruker == pytest.approx((0.5, 0.4, 0.7 * 79 / 79))
    assert any("79 x 80" in n and "rows x columns" in n and "only z" in n for n in got.notes)


def test_a_stated_shape_adds_no_assumption_note(tmp_path):
    got = src.read_pc(_edax(tmp_path, **{"x-star": 0.5, "y-star": 0.6, "z-star": 0.7}))
    assert got.notes == []


def test_an_ang_header_line_without_a_line_break_is_not_read_whole(tmp_path):
    """20 MB without a newline is one 'line'; the header reader must stop after a bounded read."""
    import tracemalloc
    p = tmp_path / "long.ang"
    with open(p, "w", encoding="utf-8") as handle:
        handle.write("# " + "x" * (20 * 1024 * 1024))
    tracemalloc.start()
    try:
        code = _code(p, shape=(60, 60))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert code == "noPcInFile"
    assert peak < 2 * 1024 * 1024


def test_a_long_data_line_after_a_complete_header_does_not_spoil_the_pc(tmp_path):
    p = _ang(tmp_path)
    with open(p, "a", encoding="utf-8") as handle:
        handle.write("0.5 " * 5000 + "\n")
    assert src.read_pc(p, shape=(60, 60)).stated[0] == pytest.approx(0.507262)


def test_an_ang_header_line_over_the_limit_after_the_stars_has_no_pc(tmp_path):
    """A header that bursts the limit is no header, even with its three stars read."""
    p = tmp_path / "wide.ang"
    p.write_text(ANG.format(x="0.5", y="0.6", z="0.7").replace("#\n", "# " + "x" * 10000 + "\n", 1), encoding="utf-8")
    assert _code(p, shape=(60, 60)) == "noPcInFile"


def test_an_ang_header_that_runs_past_the_line_limit_has_no_pc(tmp_path):
    p = tmp_path / "late.ang"
    p.write_text("# filler\n" * 500 + ANG.format(x="0.5", y="0.6", z="0.7"), encoding="utf-8")
    assert _code(p, shape=(60, 60)) == "noPcInFile"


def test_a_pc_outside_what_a_detector_can_be_is_proposed_with_a_note(tmp_path):
    """Only a proposal, so nothing is refused. TSL (0.5, 0.6, 3.0) on 60 x 60 is Bruker (0.5, 0.4, 3.0):
    z is above 1.6, x and y are inside."""
    got = src.read_pc(_ang(tmp_path, "0.5", "0.6", "3.0"), shape=(60, 60))
    assert got.pc_bruker == pytest.approx((0.5, 0.4, 3.0))
    notes = [n for n in got.notes if "outside what a detector can be" in n]
    assert len(notes) == 1 and "check z." in notes[0] and "0.05-0.95" in notes[0] and "0.2-1.6" in notes[0]


def test_an_orienta_export_far_outside_the_detector_range_names_all_three_axes(tmp_path):
    p = tmp_path / "export.h5"
    with h5py.File(p, "w") as f:
        f.create_group("Detector")["pc"] = np.array([5.0, -3.0, 100.0])
    got = src.read_pc(p)
    assert got.pc_bruker == (5.0, -3.0, 100.0)
    assert any("check x, y, z." in n for n in got.notes)


def test_an_orienta_export_of_three_zeros_has_no_pc(tmp_path):
    p = tmp_path / "export.h5"
    with h5py.File(p, "w") as f:
        f.create_group("Detector")["pc"] = np.zeros(3)
    assert _code(p) == "noPcInFile"


def test_a_plausible_pc_gets_no_range_note(tmp_path):
    got = src.read_pc(_ang(tmp_path), shape=(60, 60))
    assert not any("outside" in n for n in got.notes)


# ------------------------------------------------------------------ every note has a stable code

def _items(got):
    return {i["code"]: i for i in got.note_items()}


def test_every_kind_of_note_carries_its_code_and_params_beside_the_english_text(tmp_path):
    """Hand values: a map of 5 pixels of which 3 are unindexed, on 128 x 156 given by the caller (the file states no
    size); a z of 3.0 is outside 0.2-1.6."""
    p = _oxford(tmp_path, [0, 0.5, 0.52, np.nan, 0.5], [0, 0.55, 0.55, 0.55, 0], [0, 0.69, 0.69, 0.69, 0.69], h=127.5)
    got = src.read_pc(p, shape=(128, 156))
    items = _items(got)
    assert items["pixelsLeftOut"]["params"] == {"left_out": 3, "total": 5}
    assert items["assumedShape"]["params"] == {"rows": 128, "cols": 156, "convention": "oxford"}
    assert items["stdOverMap"]["params"]["x"] == pytest.approx(0.01) and items["stdOverMap"]["params"]["y"] == 0
    assert "meanOfMap" in items
    assert items["pixelsLeftOut"]["text"].startswith("3 of 5 pixels")      # the English sentence stays
    far = src.read_pc(_ang(tmp_path, "0.5", "0.6", "3.0"), shape=(60, 60))
    assert _items(far)["outsideDetectorRange"]["params"]["axes"] == ["z"]
    assert "orientaExport" in _items(src.read_pc(_orienta_export(tmp_path)))
    assert all(i["code"] != "other" for i in _items(got).values()) and far.to_dict()["note_items"]


def _orienta_export(tmp_path):
    p = tmp_path / "export_ok.h5"
    with h5py.File(p, "w") as f:
        f.create_group("Detector")["pc"] = np.array([0.5, 0.4, 0.7])
    return p


# ------------------------------------------------------------------ real files

def test_the_real_oxford_scan_gives_the_values_kikuchipy_loads(tmp_path):
    """The Oxford reference scan is tracked in every checkout; kikuchipy hands the application the mean
    (0.5026, 0.3273, 0.8458) for it."""
    path = _test_data("EBSD_SampleB_extrusion_withPattern Sample_B Arbeitsbereich 1 Elementverteilungsdaten 1.h5oina")
    got = src.read_pc(path)
    assert got.pc_bruker == pytest.approx((0.5026, 0.3273, 0.8458), abs=6e-4)
    assert got.shape == (128, 156) and got.per_pixel is True


def test_the_real_edax_h5_gives_the_harness_truth():
    """EDAX LoGainNi.h5 (60 x 60): the harness log's Bruker truth (0.5073, 0.2621, 0.5585)."""
    got = src.read_pc(_test_data("LoGainNi.h5"))
    assert got.pc_bruker == pytest.approx((0.5073, 0.2621, 0.5585), abs=6e-5)
    assert got.shape == (60, 60) and got.source_kind == "edax_h5"


def test_the_real_osc_gives_the_harness_truth():
    """An EDAX .osc file (79 x 79 patterns): the harness log's Bruker truth (0.5499, 0.4974, 0.702)."""
    got = src.read_pc(_test_data("new test", "Scan5.osc"), shape=(79, 79))
    assert got.pc_bruker == pytest.approx((0.5499, 0.4974, 0.702), abs=6e-5)
    assert got.source_kind == "osc" and got.shape_assumed is True


def test_a_map_whose_deviations_overflow_has_a_mean_and_no_spread_note(tmp_path):
    """Hand calculation: x = 1e155 and 3e155 have mean 2e155 and deviations 1e155; 1e155 squared is 1e310 (not a float)."""
    p = tmp_path / "huge.h5oina"
    with h5py.File(p, "w") as f:
        data = f.create_group("1/EBSD/Data")
        data["Pattern Center X"] = np.array([1e155, 3e155])
        data["Pattern Center Y"] = np.array([0.5, 0.5])
        data["Detector Distance"] = np.array([0.7, 0.7])
        header = f.create_group("1/EBSD/Header")
        header["Pattern Width"] = np.array([156])
        header["Pattern Height"] = np.array([128])
    got = src.read_pc(p)
    assert got.pc_bruker[0] == pytest.approx(2e155, rel=1e-12)
    assert got.pc_bruker[1:] == pytest.approx((1 - 0.5 * 1.21875, 0.7 * 1.21875))
    assert got.per_pixel is True
    assert [it["code"] for it in got.note_items()] == ["meanOfMap", "outsideDetectorRange"]
