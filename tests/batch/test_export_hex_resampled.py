"""The batch writers for a hexagonal checkpoint: ``.ctf`` and the square ``.ang`` hold the scan
resampled to a square grid, the ``.ang`` stays a HexGrid file.

Hand values, from ``tests/test_hex_resample.py`` (3 rows x 4 columns, step 1, odd rows shifted by
+0.5 step, so point ``p`` of the index rectangle is point ``p`` of the scan): the 12 cells of the
square grid take the points

    cell:   0 1 2 3  4 5 6 7  8 9 10 11
    point:  0 1 2 3  4 4 5 6  8 9 10 11

(cells 4 and 5 are ties between two points of the shifted row and take the lower column; point 7,
the last of the shifted row, is nobody's nearest). So 12 source points, 12 points written, 1
unused.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from backend.api.services import ang_export, hex_resample
from backend.api.services.checkpoint_writer import CheckpointWriter
from backend.api.services.result_exporter import _write_ctf, export_all, export_ang_ctf
from backend.api.services.scan_geometry import ScanGeometry

from .test_export_ang_ctf_facts import _ctf_rows, _make, load_ang_rows

POINT_INDEX = [0, 1, 2, 3, 4, 4, 5, 6, 8, 9, 10, 11]
LATTICE = [4.0495] * 3 + [90.0] * 3


def _hex(step=1.0, confirmed=True, shift=+0.5):
    return ScanGeometry.hex(3, 4, step, odd_row_shift=shift, shift_confirmed=confirmed)


def _text(path):
    return Path(path).read_text(encoding="utf-8")


def _ctf_line(path, key):
    return next(ln for ln in _text(path).splitlines() if ln.startswith(key + "\t"))


def _ang_header(path):
    return [ln for ln in _text(path).splitlines() if ln.startswith("#")]


def _ang_value(path, key):
    pat = re.compile(r"^#\s*" + re.escape(key) + r"\s*:?\s+(.*)$")
    return [m.group(1).strip() for ln in _ang_header(path) if (m := pat.match(ln))]


NOTE_CONFIRMED = ("Resampled from a hexagonal scan to a square grid by nearest neighbour at step 1 "
                  "um (source points 12, points written 12, source points unused 1); nothing was "
                  "interpolated.")


# --------------------------------------------------------------------------- the .ctf

def test_ctf_euler_angles_are_the_light_h5_euler_angles_at_the_chosen_points(tmp_path):
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(), step=1.0)
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["h5_light", "ctf"])
    with h5py.File(res["h5_light"], "r") as f:
        e_h5 = np.degrees(np.asarray(f["Indexing/Assignment/euler_angles"])).reshape(-1, 3)
    e_ctf = _ctf_rows(res["ctf"])[:, 5:8]
    assert e_ctf.shape == (12, 3)
    expected = e_h5[POINT_INDEX]
    d = (e_ctf - expected + 180.0) % 360.0 - 180.0
    np.testing.assert_allclose(d, 0.0, atol=1e-2)
    # cells 4 and 5 take the same point, cell 3 and 4 do not
    np.testing.assert_allclose(e_ctf[4], e_ctf[5], atol=1e-4)
    assert not np.allclose(e_ctf[3], e_ctf[4], atol=1e-2)


def test_ctf_states_the_grid_of_the_resampling(tmp_path):
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(), step=1.0)
    ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    assert Path(ctf).name == "result_Scan.ctf"
    assert _ctf_line(ctf, "XCells") == "XCells\t4" and _ctf_line(ctf, "YCells") == "YCells\t3"
    assert float(_ctf_line(ctf, "XStep").split("\t")[1]) == 1.0
    assert float(_ctf_line(ctf, "YStep").split("\t")[1]) == 1.0
    rows = _ctf_rows(ctf)
    assert rows[:, 1].tolist() == [0.0, 1.0, 2.0, 3.0] * 3          # X: column * step
    assert rows[:, 2].tolist() == [0.0] * 4 + [1.0] * 4 + [2.0] * 4  # Y: row * step


def test_ctf_of_a_float32_widened_step_states_the_decimal_step(tmp_path):
    # A step that arrives as a float32 header value (0.15 -> 0.15000000596046448): the grid of the
    # file is the one of OIM's decimal step, and so is every number the file states.
    widened = 0.15000000596046448
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(step=widened), step=widened)
    _ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    assert _ctf_line(ctf, "XStep") == "XStep\t0.15" and _ctf_line(ctf, "YStep") == "YStep\t0.15"
    rows = _ctf_rows(ctf)
    # x = column * 0.15 = 0 0.15 0.30 0.45, y = row * 0.15 = 0 0.15 0.30
    assert [f"{v:.4f}" for v in rows[:4, 1]] == ["0.0000", "0.1500", "0.3000", "0.4500"]
    assert [f"{v:.4f}" for v in rows[::4, 2]] == ["0.0000", "0.1500", "0.3000"]


def test_ctf_band_contrast_follows_the_gathered_quality_not_the_source_order(tmp_path):
    # the source holds Band Contrast 10 + p at point p; cell c takes point POINT_INDEX[c]
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(), step=1.0,
                       band_contrast=np.arange(10, 22))
    _ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    rows = _ctf_rows(ctf)
    assert rows[:, 9].tolist() == [10.0 + p for p in POINT_INDEX]
    assert rows[:, 9].tolist() != [10.0 + c for c in range(12)]    # not the source order
    assert rows[:, 3].tolist() == [8.0] * 12                      # Bands: the neutral value
    assert "BC column" not in _ctf_line(ctf, "Prj")               # native Band Contrast: no note


def test_ctf_without_any_quality_writes_zeros_and_says_so(tmp_path):
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(), step=1.0)
    _ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    assert _ctf_rows(ctf)[:, 9].tolist() == [0.0] * 12
    assert "BC column = 0 (no native Band Contrast in source)" in _ctf_line(ctf, "Prj")


def test_a_computed_image_quality_is_gathered_once_and_the_source_is_not_read_again(
        tmp_path, monkeypatch):
    calls = []

    def fake_quality(*args, **kwargs):
        calls.append(args)
        return np.arange(100, 112, dtype=float)             # 100 + p at point p, in source order

    monkeypatch.setattr("backend.api.services.pattern_quality.image_quality_from_file",
                        fake_quality)
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(), step=1.0)
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["ctf", "ang_square"])
    assert len(calls) == 1, "one read of the source patterns, in export_ang_ctf"
    # Scaled to 255 at the largest value (111, point 11, which a cell takes): (100 + p) * 255 / 111,
    # whole numbers in the .ctf, five decimals in the .ang
    expected = [(100 + p) * 255 / 111 for p in POINT_INDEX]
    np.testing.assert_allclose(load_ang_rows(res["ang_square"])[:, 5], expected, atol=1e-4)
    assert _ctf_rows(res["ctf"])[:, 9].tolist() == [float((100 + p) * 255 // 111) for p in POINT_INDEX]
    assert "computed FFT image quality" in _ctf_line(res["ctf"], "Prj")


def test_ctf_prj_carries_the_note_and_the_resampled_origin(tmp_path):
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(), step=1.0)
    scan = {"scan_row_offset": 5, "scan_col_offset": 7, "scan_shape": [90, 120]}
    _ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"),
                               source_h5_path=str(source), scan_provenance=scan)
    prj = _ctf_line(ctf, "Prj")
    assert "Note: " + NOTE_CONFIRMED in prj
    assert "column 7" not in prj and "row 5" not in prj        # not true after resampling
    assert "assumption" not in prj                             # the direction is confirmed
    assert "\t" not in prj.split("\t", 1)[1]


def test_the_direction_is_called_an_assumption_when_no_file_states_it(tmp_path):
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(confirmed=False), step=1.0)
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["ctf", "ang_square"])
    assumption = ("The direction of the row offset is an assumption (no file states it): the odd "
                  "rows are placed half a step to the right of the even rows.")
    assert assumption in _ctf_line(res["ctf"], "Prj")
    assert any(assumption in v for v in _ang_value(res["ang_square"], "ORIENTA_NOTE"))


# --------------------------------------------------------------------------- the square .ang

def test_the_square_ang_holds_the_chosen_points_on_a_square_grid(tmp_path):
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(), step=1.0,
                       band_contrast=np.arange(10, 22))
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["ang", "ang_square"], name_suffix="__ab12")
    assert Path(res["ang"]).name == "result_Scan__ab12.ang"
    assert Path(res["ang_square"]).name == "result_Scan__ab12_square.ang"
    hexagonal, square = res["ang"], res["ang_square"]
    assert _ang_value(hexagonal, "GRID:") == ["HexGrid"]          # the hexagonal file stays hexagonal
    assert _ang_value(square, "GRID:") == ["SqrGrid"]
    assert _ang_value(square, "XSTEP:") == ["1.000000"] and _ang_value(square, "YSTEP:") == ["1.000000"]
    hex_rows, sq_rows = load_ang_rows(hexagonal), load_ang_rows(square)
    assert hex_rows.shape == sq_rows.shape == (12, 10)
    # the TSL-frame Euler angles of the square file are those of the hexagonal one at the chosen points
    np.testing.assert_allclose(sq_rows[:, :3], hex_rows[POINT_INDEX, :3], atol=2e-5)
    assert sq_rows[:, 3].tolist() == [0.0, 1.0, 2.0, 3.0] * 3
    assert sq_rows[:, 4].tolist() == [0.0] * 4 + [1.0] * 4 + [2.0] * 4
    assert sq_rows[:, 5].tolist() == [10.0 + p for p in POINT_INDEX]      # IQ follows the gather
    np.testing.assert_allclose(sq_rows[:, 6], hex_rows[POINT_INDEX, 6], atol=2e-5)   # and CI


def test_the_square_ang_carries_the_note_and_a_resampled_origin(tmp_path):
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(), step=1.0)
    scan = {"scan_row_offset": 5, "scan_col_offset": 7, "scan_shape": [90, 120]}
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["ang", "ang_square"], scan_provenance=scan)
    assert _ang_value(res["ang_square"], "ORIENTA_NOTE") == [NOTE_CONFIRMED]
    assert NOTE_CONFIRMED not in _text(res["ang"])
    assert any("row 5, column 7" in v for v in _ang_value(res["ang"], "ORIENTA_SCAN_ORIGIN"))
    square_origin = " ".join(_ang_value(res["ang_square"], "ORIENTA_SCAN_ORIGIN"))
    assert "column 7" not in square_origin and "um" in square_origin
    assert _ang_value(res["ang_square"], "ORIENTA_SCAN_GRID") == []      # not a hexagonal file


# --------------------------------------------------------------------------- phases

def _multi_checkpoint(tmp_path, phases, owner):
    """A 3 x 4 hexagonal checkpoint (step 1) whose point ``(r, c)`` is won by ``phases[owner[r][c]]``.
    ``phases`` is a list of (name, space_group, point_group, lattice)."""
    source = tmp_path / "Scan.h5oina"
    with h5py.File(source, "w") as f:
        f.create_dataset("Manufacturer", data=np.array([b"EDAX"]))
        f.create_group("1/EBSD/Header").create_dataset("Beam Voltage", data=np.float32(20.0))
        f.create_group("1/EBSD/Data")
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="multi", method="spherical", step_size_um=1.0,
                     geometry=_hex())
    owner = np.asarray(owner)
    rng = np.random.default_rng(3)
    for k, (name, sg, pg, lattice) in enumerate(phases):
        ci = np.where(owner == k, 0.9, 0.2).astype(np.float32)
        meta = {"phase_file": f"{name}.cif", "ci_mean": 0.5, "duration_sec": 1.0,
                "point_group": pg, "lattice_constants": lattice}
        if sg is not None:
            meta["space_group"] = sg
        cw.write_phase_result(
            name, ci, rng.uniform(0.1, 3.0, (3, 4, 3)).astype(np.float32), meta)
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw


AL = ("Al", 225, "m-3m", [4.0495] * 3 + [90.0] * 3)
TI = ("Ti", 194, "6/mmm", [2.95, 2.95, 4.68, 90.0, 90.0, 120.0])
TRIG = ("Trig", 162, "-3m", [4.0, 4.0, 5.0, 90.0, 90.0, 120.0])


def _phase_table(path):
    lines = _text(path).splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("Phases\t"))
    n = int(lines[start].split("\t")[1])
    return n, [ln.split("\t") for ln in lines[start + 1:start + 1 + n]]


def _symmetry_of(path, name):
    blocks, cur = [], None
    for ln in _ang_header(path):
        if re.match(r"^#\s*Phase\s+\d+\s*$", ln):
            cur = {}
            blocks.append(cur)
        m = re.match(r"^#\s*(MaterialName|Symmetry)\s+(.*)$", ln)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2).strip()
    return [b["Symmetry"] for b in blocks if b.get("MaterialName") == name]


def test_a_phase_only_at_the_unused_point_is_left_out_and_the_rows_name_the_others(tmp_path):
    # Phase B (Ti) wins only point 7 = row 1, column 3, which no cell takes. A wins points 0-6,
    # C wins row 2 (points 8-11). The cells therefore hold A in cells 0-7 and C in cells 8-11, and
    # orix drops B from the resampled map while A and C keep their ids 0 and 2.
    owner = [[0, 0, 0, 0], [0, 0, 0, 1], [2, 2, 2, 2]]
    source, cw = _multi_checkpoint(tmp_path, [AL, TI, TRIG], owner)
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["ctf", "ang_square"])
    assert res["ctf"] and res["ang_square"], res.get("_errors")
    n, table = _phase_table(res["ctf"])
    assert n == 2
    assert [(row[2], row[3], row[4]) for row in table] == [("Al", "11", "225"), ("Trig", "7", "162")]
    assert _ctf_rows(res["ctf"])[:, 0].tolist() == [1.0] * 8 + [2.0] * 4
    # the .ang numbers its two phases 1 and 2 as well, and names their symmetry as TSL does
    assert load_ang_rows(res["ang_square"])[:, 7].tolist() == [1.0] * 8 + [2.0] * 4
    assert _symmetry_of(res["ang_square"], "Al") == ["43"]
    assert _symmetry_of(res["ang_square"], "Trig") == ["32"]
    assert _symmetry_of(res["ang_square"], "Ti") == []


def test_the_square_ang_writes_tsl_symmetry_codes_by_the_numbers_of_its_own_phases(tmp_path):
    # A hexagonal phase wins only the unused point 7, Al points 0-6, and a phase recorded with the
    # point group 312 (which orix writes as "312" and TSL calls 32) the bottom row. The checkpoint
    # lists its phases alphabetically: AaTi, Al, T312, so the run numbers them 1, 2, 3. The square
    # file holds Al as phase 1 and the 312 phase as phase 2; a code table numbered with the three
    # phases of the run would give phase 1 the code of AaTi (62) and phase 2 the code of Al (43).
    aa_ti = ("AaTi",) + TI[1:]
    t312 = ("T312", None, "312", [4.0, 4.0, 5.0, 90.0, 90.0, 120.0])
    owner = [[1, 1, 1, 1], [1, 1, 1, 0], [2, 2, 2, 2]]
    source, cw = _multi_checkpoint(tmp_path, [aa_ti, AL, t312], owner)
    written = export_ang_ctf(
        cw.checkpoint_path, str(tmp_path / "o"), write_ang=False, write_ctf=False,
        write_ang_square=True, source_h5_path=str(source))
    assert tuple(written) == (None, None)                    # unpacks into two, as always
    path = written.ang_square
    assert path, "the square .ang was not written"
    assert _symmetry_of(path, "Al") == ["43"]
    assert _symmetry_of(path, "T312") == ["32"]
    assert _symmetry_of(path, "AaTi") == []
    assert load_ang_rows(path)[:, 7].tolist() == [1.0] * 8 + [2.0] * 4


def test_laue_class_and_space_group_of_a_trigonal_and_a_hexagonal_phase(tmp_path):
    # space group 162 (P-31m, point group -3m): Laue 7. Ti, 6/mmm, space group 194: Laue 9.
    owner = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]]
    source, cw = _multi_checkpoint(tmp_path, [AL, TI, TRIG], owner)
    _ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    n, table = _phase_table(ctf)
    assert n == 2
    assert [(row[2], row[3], row[4]) for row in table] == [("Ti", "9", "194"), ("Trig", "7", "162")]
    assert table[0][1] == "90.0;90.0;120.0" and table[0][0] == "2.950;2.950;4.680"
    assert _ctf_rows(ctf)[:, 0].tolist() == [1.0] * 8 + [2.0] * 4


def _frame_lines(path):
    return [ln for ln in _ang_header(path) if ln.startswith("# ORIENTA_CRYSTAL_FRAME: ")]


def test_the_hexagonal_batch_files_state_the_crystal_frame(tmp_path):
    # Ti (194) wins row 0 and 1, the P-31m phase (162) the bottom row: all three files hold both
    owner = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]]
    source, cw = _multi_checkpoint(tmp_path, [AL, TI, TRIG], owner)
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["ang", "ctf", "ang_square"])
    assert res["ang"] and res["ctf"] and res["ang_square"], res.get("_errors")
    for path in (res["ang"], res["ang_square"]):
        lines = _frame_lines(path)
        assert len(lines) == 2, path
        assert "X||a" in lines[0] and "Z||c*" in lines[0]
        assert "Phase Trig: space group 162, point group -31m" in lines[1] and "along Y" in lines[1]
    prj = _ctf_line(res["ctf"], "Prj")
    assert "X||a" in prj and "Phase Trig: point group -31m" in prj and "along Y" in prj
    # the Euler angles are not touched by the statement: the .ctf rows are those of the .h5
    assert _ctf_rows(res["ctf"]).shape[0] == 12


def test_a_cubic_only_hexagonal_checkpoint_states_no_crystal_frame(tmp_path):
    owner = [[0, 0, 0, 0]] * 3
    source, cw = _multi_checkpoint(tmp_path, [AL], owner)
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["ang", "ctf", "ang_square"])
    assert res["ang"] and res["ctf"] and res["ang_square"], res.get("_errors")
    assert _frame_lines(res["ang"]) == [] and _frame_lines(res["ang_square"]) == []
    assert "X||a" not in _ctf_line(res["ctf"], "Prj")


def test_the_resampled_files_state_the_frame_of_the_phases_they_hold(tmp_path):
    # the only Ti point is point 7, which no cell takes: the square files hold Al alone
    owner = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    source, cw = _multi_checkpoint(tmp_path, [AL, TI], owner)
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                     formats=["ang", "ctf", "ang_square"])
    assert len(_frame_lines(res["ang"])) == 1                  # the hexagonal file holds all 12 points
    assert _frame_lines(res["ang_square"]) == []
    assert "X||a" not in _ctf_line(res["ctf"], "Prj")


# --------------------------------------------------------------------------- _write_ctf

def _cell_map(n=12, phase_id=None):
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation
    phi1 = np.deg2rad(10.0 * np.arange(n))
    phases = PhaseList([Phase(name="Al", point_group="m-3m"), Phase(name="Ti", point_group="6/mmm"),
                        Phase(name="Fe", point_group="m-3m")])
    return CrystalMap(
        rotations=Rotation.from_euler(np.stack([phi1, np.full(n, 0.3), np.full(n, 0.2)], axis=1)),
        phase_id=np.zeros(n, dtype=int) if phase_id is None else phase_id,
        x=np.tile(np.arange(4.0), 3)[:n], y=np.repeat(np.arange(3.0), 4)[:n],
        phase_list=phases, scan_unit="um")


def _source_with_quality(tmp_path):
    path = tmp_path / "Src.h5oina"
    with h5py.File(path, "w") as f:
        data = f.create_group("1/EBSD/Data")
        data.create_dataset("Band Contrast", data=np.arange(100, 112, dtype=np.uint8))
        data.create_dataset("Bands", data=np.full(12, 99, dtype=np.uint8))
        f.create_group("1/EBSD/Header")
    return str(path)


def test_ctf_resampled_skips_source_quality(tmp_path):
    """The source holds Band Contrast 100.. and Bands 99 in the order of ITS points, and as many of
    them as the map has cells, so the size check would pass: a resampled map must not read them."""
    source = _source_with_quality(tmp_path)
    given = np.arange(5, 17)
    out = tmp_path / "a.ctf"
    _write_ctf(str(out), _cell_map(), ["Al", "Ti", "Fe"], 1.0, source_h5_path=source,
               resampled=True, resampled_quality=given, resampled_quality_native=True)
    rows = _ctf_rows(out)
    assert rows[:, 9].tolist() == given.astype(float).tolist()
    assert rows[:, 3].tolist() == [8.0] * 12
    assert "BC column" not in _ctf_line(out, "Prj")
    # not resampled: the source's own channels, as before
    plain = tmp_path / "b.ctf"
    _write_ctf(str(plain), _cell_map(), ["Al"], 1.0, source_h5_path=source)
    assert _ctf_rows(plain)[:, 9].tolist() == [float(v) for v in range(100, 112)]
    assert _ctf_rows(plain)[:, 3].tolist() == [99.0] * 12


def test_ctf_resampled_without_quality_is_honest_zeros(tmp_path):
    out = tmp_path / "a.ctf"
    _write_ctf(str(out), _cell_map(), ["Al", "Ti", "Fe"], 1.0,
               source_h5_path=_source_with_quality(tmp_path), resampled=True)
    assert _ctf_rows(out)[:, 9].tolist() == [0.0] * 12
    assert "BC column = 0 (no native Band Contrast in source)" in _ctf_line(out, "Prj")


def test_ctf_resampled_scales_a_computed_quality_to_255(tmp_path):
    out = tmp_path / "a.ctf"
    q = np.arange(100, 112, dtype=float)
    _write_ctf(str(out), _cell_map(), ["Al", "Ti", "Fe"], 1.0, resampled=True, resampled_quality=q)
    # q * 255 / 111, whole part (exact integer arithmetic; no value but the largest is a whole number)
    assert _ctf_rows(out)[:, 9].tolist() == [float(v * 255 // 111) for v in range(100, 112)]
    assert "computed FFT image quality" in _ctf_line(out, "Prj")


def test_ctf_resampled_refuses_a_quality_of_another_length(tmp_path):
    with pytest.raises(ValueError, match="12 cells"):
        _write_ctf(str(tmp_path / "a.ctf"), _cell_map(), ["Al", "Ti", "Fe"], 1.0,
                   resampled=True, resampled_quality=np.arange(7))


def test_ctf_resampled_table_and_rows_come_from_the_one_map(tmp_path):
    # Phase Ti (id 1) is in no cell: the map holds Al (id 0) and Fe (id 2) only. The names the
    # caller passes include Ti; the file must not.
    pid = np.array([0] * 6 + [2] * 5 + [-1])
    xmap = _cell_map(phase_id=pid)
    assert sorted(int(i) for i in xmap.phases.ids if i >= 0) == [0, 2]
    out = tmp_path / "a.ctf"
    _write_ctf(str(out), xmap, ["Al", "Ti", "Fe"], 1.0, resampled=True)
    n, table = _phase_table(out)
    assert n == 2 and [row[2] for row in table] == ["Al", "Fe"]
    assert _ctf_rows(out)[:, 0].tolist() == [1.0] * 6 + [2.0] * 5 + [0.0]    # 0: not indexed


# --------------------------------------------------------------------------- refusals

def test_a_square_checkpoint_refuses_the_square_ang(tmp_path):
    source, cw = _make(tmp_path, lattice=LATTICE)
    out = tmp_path / "o"
    with pytest.raises(ValueError, match="hexagonal"):
        export_ang_ctf(cw.checkpoint_path, str(out), write_ang_square=True,
                       source_h5_path=str(source))
    assert not out.exists()
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o2"), formats=["ang_square"])
    assert res["ang_square"] is None and "hexagonal" in res["_errors"]["ang_ctf"]


def test_a_square_checkpoint_writes_its_ctf_as_before(tmp_path):
    source, cw = _make(tmp_path, lattice=LATTICE, band_contrast=np.arange(10, 22))
    _ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    assert _ctf_rows(ctf)[:, 9].tolist() == [float(v) for v in range(10, 22)]     # the source order
    assert "Resampled" not in _text(ctf)
    assert _ctf_line(ctf, "XStep") == "XStep\t0.4"


def test_a_scan_that_cannot_be_resampled_is_refused_not_logged(tmp_path, monkeypatch):
    def refuse(geometry, out_step=None):
        raise ValueError("row 1 of the scan has no measured point")

    monkeypatch.setattr(hex_resample, "square_resampling", refuse)
    source, cw = _make(tmp_path, lattice=LATTICE, geometry=_hex(), step=1.0)
    out = tmp_path / "o"
    for kwargs in ({"write_ctf": True}, {"write_ctf": False, "write_ang_square": True}):
        with pytest.raises(ValueError, match="no measured point"):
            export_ang_ctf(cw.checkpoint_path, str(out), source_h5_path=str(source), **kwargs)
    assert not out.exists()                                    # refused before any write
    res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o2"),
                     formats=["ctf", "ang_square"])
    assert res["ctf"] is None and res["ang_square"] is None
    assert "no measured point" in res["_errors"]["ang_ctf"]
    # the hexagonal .ang needs no resampling and is written
    ang, _ = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o3"), write_ctf=False,
                            source_h5_path=str(source))
    assert _ang_value(ang, "GRID:") == ["HexGrid"]


# --------------------------------------------------------------------------- read back by other programs

def _ti_only_export(tmp_path, formats):
    """The 3 x 4 hexagonal checkpoint (step 1) whose every point is Ti (6/mmm, space group 194)."""
    owner = [[1, 1, 1, 1]] * 3
    source, cw = _multi_checkpoint(tmp_path, [AL, TI], owner)
    return export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"), formats=formats)


def _angle_difference(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def test_orix_reads_the_ctf_as_a_3_by_4_map_of_a_hexagonal_phase(tmp_path):
    """orix takes x and y from the rows (``XCells`` is not read): 12 points on a 3 x 4 grid, the phase
    from the Laue number 9 (6/mmm), the space group 194 and the angle gamma 120 of the phase line."""
    from orix.io import load
    res = _ti_only_export(tmp_path, ["h5_light", "ctf"])
    with h5py.File(res["h5_light"], "r") as f:
        e_h5 = np.degrees(np.asarray(f["Indexing/Assignment/euler_angles"])).reshape(-1, 3)
    xmap = load(res["ctf"])
    assert xmap.size == 12 and xmap.shape == (3, 4)
    assert xmap.x.tolist() == [0.0, 1.0, 2.0, 3.0] * 3                 # column * step
    assert xmap.y.tolist() == [0.0] * 4 + [1.0] * 4 + [2.0] * 4         # row * step
    phase = xmap.phases[int(xmap.phases.ids[0])]
    assert (phase.name, phase.point_group.name, int(phase.space_group.number)) == ("Ti", "6/mmm", 194)
    assert phase.structure.lattice.gamma == pytest.approx(120.0, abs=1e-6)
    # row by row the Euler angles are those of the points the cells take
    got = np.asarray(xmap.rotations.to_euler(degrees=True))
    np.testing.assert_allclose(_angle_difference(got, e_h5[POINT_INDEX]), 0.0, atol=1e-2)


def test_orix_reads_the_square_ang_as_a_3_by_4_map(tmp_path):
    """The same checks as for the ``.ctf``: the Euler angles are those of the points the cells take
    (the ``.h5`` holds them per point; the source is an EDAX one, so ``.ang`` and ``.h5`` share a frame)
    and the phase is the Ti one of the Laue class 6/mmm."""
    from orix.io import load
    res = _ti_only_export(tmp_path, ["h5_light", "ang_square"])
    with h5py.File(res["h5_light"], "r") as f:
        e_h5 = np.degrees(np.asarray(f["Indexing/Assignment/euler_angles"])).reshape(-1, 3)
    xmap = load(res["ang_square"])
    assert xmap.size == 12 and xmap.shape == (3, 4)
    assert xmap.x.tolist() == [0.0, 1.0, 2.0, 3.0] * 3
    assert xmap.y.tolist() == [0.0] * 4 + [1.0] * 4 + [2.0] * 4
    phase = xmap.phases[int(xmap.phases.ids[0])]
    # a .ang names the symmetry by its TSL code (62), which orix reads as the rotation group 622 of
    # the Laue class 6/mmm
    assert phase.name == "Ti" and phase.point_group.name == "622"
    assert phase.point_group.laue.name == "6/mmm"
    got = np.asarray(xmap.rotations.to_euler(degrees=True))
    np.testing.assert_allclose(_angle_difference(got, e_h5[POINT_INDEX]), 0.0, atol=1e-2)


def test_parse_ctf_of_the_ai_module_reads_the_grid_and_the_rows(tmp_path):
    from ebsd_ai.data.ctf_parser import parse_ctf
    res = _ti_only_export(tmp_path, ["h5_light", "ctf"])
    with h5py.File(res["h5_light"], "r") as f:
        e_h5 = np.degrees(np.asarray(f["Indexing/Assignment/euler_angles"])).reshape(-1, 3)
    scan = parse_ctf(res["ctf"])
    assert scan.scan_shape == (3, 4)                                   # YCells x XCells
    assert scan.step_sizes == (1.0, 1.0)
    assert scan.phase_names == ["Not indexed", "Ti"] and scan.phase_ids.tolist() == [1] * 12
    # the k-th row holds the Euler angles of the point cell k takes (the .h5 holds them per point)
    got = np.degrees(scan.euler_angles)
    np.testing.assert_allclose(_angle_difference(got, e_h5[POINT_INDEX]), 0.0, atol=1e-2)


def _bunge_matrix(phi1, big_phi, phi2):
    """Passive Bunge matrix g = Rz(phi2) Rx(Phi) Rz(phi1), degrees in, written out by hand."""
    def rz(a):
        c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def rx(a):
        c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
        return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    return rz(phi2) @ rx(big_phi) @ rz(phi1)


def test_the_misorientation_of_two_cells_read_by_orix_is_the_bunge_matrix_one(tmp_path):
    """Two cells with Euler angles (10, 20, 30) and (65, 25, 40) degrees (cells 0 and 5 of
    ``euler = (10 + 11 k, 20 + k, 30 + 2 k)``). With D = gB gA^T the rotation angle is
    arccos((trace D - 1) / 2) = 64.52448 degrees without symmetry. The value does not depend on the
    crystal frame: a turn of the frame conjugates D, which keeps its angle. With the 12 proper
    operations of 622 on both sides the smallest angle is 20.89045 degrees (MTEX gives the same)."""
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.io import load
    from orix.quaternion import Rotation
    from diffpy.structure import Lattice, Structure

    n = 12
    triples = np.array([[10.0 + 11.0 * k, 20.0 + k, 30.0 + 2.0 * k] for k in range(n)])
    ti = Phase(name="Ti", space_group=194,
               structure=Structure(lattice=Lattice(2.95, 2.95, 4.68, 90, 90, 120)))
    xmap = CrystalMap(
        rotations=Rotation.from_euler(np.deg2rad(triples)), phase_id=np.zeros(n, dtype=int),
        x=np.tile(np.arange(4.0), 3) * 2.0, y=np.repeat(np.arange(3.0), 4) * 2.0,
        phase_list=PhaseList(phases=[ti]), scan_unit="um")
    path = tmp_path / "pair.ctf"
    _write_ctf(str(path), xmap, ["Ti"], 2.0, resampled=True)

    d = _bunge_matrix(*triples[5]) @ _bunge_matrix(*triples[0]).T
    by_hand = np.degrees(np.arccos((np.trace(d) - 1.0) / 2.0))
    assert by_hand == pytest.approx(64.52448, abs=1e-4)

    read = load(str(path))
    a, b = read.rotations[0], read.rotations[5]
    assert float(np.degrees((~a * b).angle[0])) == pytest.approx(by_hand, abs=1e-4)
    assert float(np.degrees((b * ~a).angle[0])) == pytest.approx(by_hand, abs=1e-4)

    oa, ob = read.orientations[0], read.orientations[5]       # carry the 6/mmm of the phase line
    assert float(np.degrees(oa.angle_with(ob))[0]) == pytest.approx(20.89045, abs=1e-3)
