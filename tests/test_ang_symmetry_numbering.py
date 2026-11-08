"""The phase numbers of a ``.ang`` / ``.ctf`` are those of the map that is written.

orix drops a phase of a map that no cell takes and numbers the phases of the file that remain, from 1.
A table of TSL ``# Symmetry`` codes numbered by the full phase list therefore states the symmetry of
the wrong phase for every phase after the dropped one. Every expectation below is a hand table (the
TSL code of a Laue group), never the output of the code under test.

Hand table (TSL # Symmetry): m-3m 43, 4/mmm 42, 6/mmm 62, -3m 32, 312 (Laue -3m) 32,
2/m unique axis b 20, 2/m unique axis c 2.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pytest
from diffpy.structure import Lattice, Structure
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation

from backend.api.services import ang_export
from backend.api.services.checkpoint_writer import CheckpointWriter
from backend.api.services.result_exporter import export_ang_ctf
from backend.api.services.scan_geometry import ScanGeometry

ROWS, COLS = 3, 4

# name, space group (or None), point group, lattice
AL = ("Al", 225, "m-3m", [4.05, 4.05, 4.05, 90, 90, 90])
MG = ("Mg", 194, "6/mmm", [3.21, 3.21, 5.21, 90, 90, 120])
SN = ("Sn", 141, "4/mmm", [5.83, 5.83, 3.18, 90, 90, 90])
TR312 = ("Tr312", None, "312", [4.0, 4.0, 5.0, 90, 90, 120])
MONO_B = ("MonoB", None, "2/m", [5.0, 6.0, 7.0, 90, 100, 90])
MONO_C = ("MonoC", None, "2/m", [5.0, 6.0, 7.0, 90, 90, 100])


def _checkpoint(tmp_path, phases, winners, geometry=None):
    """A checkpoint of ``phases`` on a 3 x 4 grid; ``winners[name]`` are the flat cell numbers
    where that phase has the high confidence (a phase with no entry wins nowhere)."""
    source = tmp_path / "Scan.h5oina"
    with h5py.File(source, "w") as f:
        f.create_dataset("Manufacturer", data=np.array([b"EDAX"]))
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(ROWS, COLS), batch_id="sym", method="hough", step_size_um=1.0,
                     geometry=geometry)
    rng = np.random.default_rng(3)
    for name, sg, pg, lat in phases:
        ci = np.full(ROWS * COLS, 0.1, np.float32)
        ci[list(winners.get(name, []))] = 0.9
        attrs = {"phase_file": name + ".cif", "ci_mean": 0.5, "duration_sec": 1.0,
                 "point_group": pg, "lattice_constants": lat}
        if sg is not None:
            attrs["space_group"] = sg
        cw.write_phase_result(name, ci.reshape(ROWS, COLS),
                              rng.uniform(0, 3, (ROWS, COLS, 3)).astype(np.float32), attrs)
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw


def _symmetry_by_name(path):
    """``{MaterialName: '# Symmetry' value}`` of the phase blocks of a ``.ang`` header."""
    blocks, cur = [], None
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.startswith("#"):
            break
        if re.match(r"^#\s*Phase\s+\d+\s*$", ln):
            cur = {}
            blocks.append(cur)
        m = re.match(r"^#\s*(MaterialName|Symmetry)\s+(.*)$", ln)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2).strip()
    return {b["MaterialName"]: b["Symmetry"] for b in blocks}


def _ctf_table(path):
    """(phase names of the ``Phases`` table, phase numbers of the rows)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("Phases\t"))
    n = int(lines[i].split("\t")[1])
    names = [lines[i + 1 + k].split("\t")[2] for k in range(n)]
    rows = [ln.split("\t") for ln in lines[i + n + 2:] if ln.strip()]
    return names, [int(r[0]) for r in rows]


@pytest.mark.parametrize("shift", [0.5, -0.5])
def test_hex_batch_symmetry_follows_the_written_phases(tmp_path, shift):
    """Mg wins cells 0-5, Sn cells 6-11, Al wins none. A batch lists its phases by name, so Al is
    the first of three and orix writes Mg as phase 1 and Sn as phase 2.
    Hand: Mg 6/mmm = 62, Sn 4/mmm = 42."""
    geo = ScanGeometry.hex(ROWS, COLS, 1.0, odd_row_shift=shift, shift_confirmed=True)
    source, cw = _checkpoint(tmp_path, [MG, AL, SN],
                             {"Mg": range(0, 6), "Sn": range(6, 12)}, geometry=geo)
    ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    assert ang and ctf
    assert _symmetry_by_name(ang) == {"Mg": "62", "Sn": "42"}
    names, rows = _ctf_table(ctf)
    assert names == ["Mg", "Sn"] and sorted(set(rows)) == [1, 2]


def test_orix_reads_the_hex_batch_ang_back_with_the_right_groups(tmp_path):
    from orix.io import load
    geo = ScanGeometry.hex(ROWS, COLS, 1.0, odd_row_shift=0.5, shift_confirmed=True)
    source, cw = _checkpoint(tmp_path, [MG, AL, SN],
                             {"Mg": range(0, 6), "Sn": range(6, 12)}, geometry=geo)
    ang, _ = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    xmap = load(ang)
    laue = {p.name: p.point_group.laue.name for pid, p in xmap.phases if pid >= 0}
    assert laue == {"Mg": "6/mmm", "Sn": "4/mmm"}


def test_square_batch_symmetry_is_the_tsl_code_of_every_written_phase(tmp_path):
    """The square batch .ang has no rewrite of its own: orix writes the name of the rotation group
    (312 for the point group 312, 112 for 2/m), which is no TSL code. Al wins nothing."""
    winners = {"Mg": [0, 1], "Tr312": [2, 3], "MonoB": [4, 5], "MonoC": [6, 7], "Sn": [8, 9, 10, 11]}
    source, cw = _checkpoint(tmp_path, [AL, MG, TR312, MONO_B, MONO_C, SN], winners)
    ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    assert _symmetry_by_name(ang) == {
        "Mg": "62", "Tr312": "32", "MonoB": "20", "MonoC": "2", "Sn": "42"}
    names, rows = _ctf_table(ctf)
    assert names == ["Mg", "MonoB", "MonoC", "Sn", "Tr312"]          # the batch lists phases by name


def test_square_batch_with_a_phase_that_wins_nowhere_still_writes_the_ctf(tmp_path):
    """Three phases, the first (by name) wins nowhere: the .ctf lists the two phases that are in the map and
    numbers the rows 1 and 2 (it used to raise "Phase 'Al' not found" and come back None)."""
    source, cw = _checkpoint(tmp_path, [MG, AL, SN], {"Mg": range(0, 6), "Sn": range(6, 12)})
    ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    assert ang and ctf
    names, rows = _ctf_table(ctf)
    assert names == ["Mg", "Sn"]
    assert rows == [1] * 6 + [2] * 6                        # row-major: Mg first, then Sn
    assert _symmetry_by_name(ang) == {"Mg": "62", "Sn": "42"}
    # the Laue class column of the table: 6/mmm = 9, 4/mmm = 5 (Oxford numbering)
    lines = Path(ctf).read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("Phases\t"))
    assert [lines[i + 1 + k].split("\t")[3] for k in range(2)] == ["9", "5"]


def test_square_batch_without_an_empty_phase_keeps_its_ctf(tmp_path):
    """Every phase wins somewhere: the table and the numbers are what they were."""
    source, cw = _checkpoint(tmp_path, [MG, AL, SN],
                             {"Mg": range(0, 4), "Al": range(4, 8), "Sn": range(8, 12)})
    _ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    names, rows = _ctf_table(ctf)
    assert names == ["Al", "Mg", "Sn"]                         # by name; Mg is cells 0-3
    assert rows == [2] * 4 + [1] * 4 + [3] * 4


def _map_with_an_empty_middle_phase():
    pl = PhaseList([
        Phase("Al", space_group=225, structure=Structure(lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90))),
        Phase("Mg", space_group=194, structure=Structure(lattice=Lattice(3.21, 3.21, 5.21, 90, 90, 120))),
        Phase("Sn", space_group=141, structure=Structure(lattice=Lattice(5.83, 5.83, 3.18, 90, 90, 90))),
    ])
    pid = np.array([0] * 6 + [1] * 2 + [2] * 4)
    ys, xs = np.mgrid[0:3, 0:4]
    full = CrystalMap(rotations=Rotation.random(12), phase_id=pid, x=xs.ravel().astype(float),
                      y=ys.ravel().astype(float), phase_list=pl, prop={"ci": np.ones(12)})
    # a slice that holds no Mg cell keeps Mg in its phase list (a map built with the cells of
    # only Al and Sn would have dropped it already); the cleanup of the phase map leaves such maps
    emptied = full[full.phase_id != 1]
    assert emptied.phases.names == ["Al", "Mg", "Sn"] and emptied.phases_in_data.names == ["Al", "Sn"]
    return emptied


def test_the_route_codes_are_those_of_the_map_that_is_written():
    """``build_export_xmap`` hands back the codes with the map: Al is 1, Sn is 2 (the phase list
    still names Mg, which no cell takes). Hand: Al 43, Sn 42."""
    m = _map_with_an_empty_middle_phase()
    mask = np.ones(12, bool)
    mask[6:8] = False                                        # the two Mg cells are not in the map
    xm, codes = ang_export.build_export_xmap(m, (3, 4), mask, 1.0)
    assert [p.name for pid, p in xm.phases if pid >= 0] == ["Al", "Sn"]
    assert codes == {1: "43", 2: "42"}
