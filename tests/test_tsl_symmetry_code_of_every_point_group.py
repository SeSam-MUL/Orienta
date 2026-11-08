"""The TSL ``# Symmetry`` code of a ``.ang`` is known for every crystal class a phase can have.

A phase built from a space group alone (what the batch makes of a checkpoint) is a plain orix
``Phase``. For the space groups 6-9 (point group ``m``) plain orix gives a point group without a Laue
group, so the code was looked up as ``None`` and the whole ``.ang`` / ``.ctf`` export of a square
batch was refused. The group is now read through ``crystal_symmetry.phase_point_group``, which the
exports use everywhere else.

Hand table, one space group per point group (the TSL code is the one of the Laue class: -1 -> 1,
2/m -> 20 (unique axis b) or 2 (unique axis c), mmm -> 22, 4/m -> 4, 4/mmm -> 42, -3 -> 3, -3m -> 32,
6/m -> 6, 6/mmm -> 62, m-3 -> 23, m-3m -> 43). The point group of each number is that of the
International Tables.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
from diffpy.structure import Lattice, Structure
from orix.crystal_map import CrystalMap, Phase
from orix.quaternion import Rotation

from backend.api.services import ang_export
from backend.api.services.result_exporter import (_tsl_codes_of_map, export_all, export_ang_ctf)
from backend.api.services.checkpoint_writer import CheckpointWriter

TRICLINIC = (5.0, 6.0, 7.0, 80, 85, 95)
MONOCLINIC_B = (5.0, 6.0, 7.0, 90, 100, 90)
MONOCLINIC_C = (5.0, 6.0, 7.0, 90, 90, 100)
ORTHORHOMBIC = (5.0, 6.0, 7.0, 90, 90, 90)
TETRAGONAL = (4.0, 4.0, 6.0, 90, 90, 90)
HEXAGONAL = (4.0, 4.0, 6.0, 90, 90, 120)
CUBIC = (4.0, 4.0, 4.0, 90, 90, 90)

# space group, point group (International Tables), lattice, TSL code by hand
CLASSES = [
    (1, "1", TRICLINIC, "1"), (2, "-1", TRICLINIC, "1"),
    (3, "2", MONOCLINIC_B, "20"), (6, "m", MONOCLINIC_B, "20"), (7, "m", MONOCLINIC_B, "20"),
    (8, "m", MONOCLINIC_B, "20"), (9, "m", MONOCLINIC_B, "20"), (10, "2/m", MONOCLINIC_B, "20"),
    (16, "222", ORTHORHOMBIC, "22"), (25, "mm2", ORTHORHOMBIC, "22"), (47, "mmm", ORTHORHOMBIC, "22"),
    (75, "4", TETRAGONAL, "4"), (81, "-4", TETRAGONAL, "4"), (83, "4/m", TETRAGONAL, "4"),
    (89, "422", TETRAGONAL, "42"), (99, "4mm", TETRAGONAL, "42"), (111, "-42m", TETRAGONAL, "42"),
    (123, "4/mmm", TETRAGONAL, "42"),
    (143, "3", HEXAGONAL, "3"), (147, "-3", HEXAGONAL, "3"),
    (149, "312", HEXAGONAL, "32"), (150, "321", HEXAGONAL, "32"), (156, "3m1", HEXAGONAL, "32"),
    (157, "31m", HEXAGONAL, "32"), (162, "-31m", HEXAGONAL, "32"), (164, "-3m1", HEXAGONAL, "32"),
    (168, "6", HEXAGONAL, "6"), (174, "-6", HEXAGONAL, "6"), (175, "6/m", HEXAGONAL, "6"),
    (177, "622", HEXAGONAL, "62"), (183, "6mm", HEXAGONAL, "62"), (187, "-6m2", HEXAGONAL, "62"),
    (191, "6/mmm", HEXAGONAL, "62"),
    (195, "23", CUBIC, "23"), (200, "m-3", CUBIC, "23"),
    (207, "432", CUBIC, "43"), (215, "-43m", CUBIC, "43"), (221, "m-3m", CUBIC, "43"),
]


def _map_of(phase):
    return CrystalMap(rotations=Rotation.from_euler(np.zeros((4, 3))), phase_id=np.zeros(4, dtype=int),
                      x=np.arange(4, dtype=float), y=np.zeros(4), phase_list=phase_list_of(phase))


def phase_list_of(phase):
    from orix.crystal_map import PhaseList
    return PhaseList(phases=[phase])


@pytest.mark.parametrize("space_group, point_group, lattice, code", CLASSES,
                         ids=[f"sg{c[0]}-{c[1]}" for c in CLASSES])
def test_a_plain_phase_of_every_point_group_gets_its_tsl_code(space_group, point_group, lattice, code):
    phase = Phase(name="P", space_group=space_group,
                  structure=Structure(lattice=Lattice(*lattice)))
    assert _tsl_codes_of_map(_map_of(phase)) == {1: code}


@pytest.mark.parametrize("space_group", [6, 10])
def test_a_monoclinic_phase_with_the_unique_axis_c_is_code_2(space_group):
    """The cell decides between the two settings of 2/m: gamma != 90 is the c-unique cell (code 2)."""
    phase = Phase(name="P", space_group=space_group,
                  structure=Structure(lattice=Lattice(*MONOCLINIC_C)))
    assert _tsl_codes_of_map(_map_of(phase)) == {1: "2"}


def test_a_phase_without_a_point_group_is_refused_by_name():
    with pytest.raises(ang_export.SymmetryUnknown, match="Nothing"):
        _tsl_codes_of_map(_map_of(Phase(name="Nothing")))


# ------------------------------------------------------------------------------ the batch itself

def _checkpoint(tmp_path, name, space_group, point_group, lattice):
    source = tmp_path / "Scan.h5oina"
    with h5py.File(source, "w") as f:
        f.create_dataset("Manufacturer", data=np.array([b"EDAX"]))
    cw = CheckpointWriter(str(source))
    cw.init_metadata(grid_shape=(3, 4), batch_id="sym", method="hough", step_size_um=1.0)
    rng = np.random.default_rng(3)
    attrs = {"phase_file": name + ".cif", "ci_mean": 0.5, "duration_sec": 1.0, "point_group": point_group,
             "lattice_constants": list(lattice)}
    if space_group is not None:
        attrs["space_group"] = space_group
    cw.write_phase_result(name, np.full((3, 4), 0.9, np.float32),
                          rng.uniform(0, 3, (3, 4, 3)).astype(np.float32), attrs)
    cw.compute_auto_assignment(confidence_threshold=0.0)
    return source, cw


@pytest.mark.parametrize("space_group, point_group", [(6, "m"), (7, "m"), (8, "m"), (9, "m"), (25, "mm2")])
def test_a_square_batch_of_a_monoclinic_or_mm2_phase_writes_ang_and_ctf(
        tmp_path, space_group, point_group):
    """Space groups 6-9 (Pm, Pc, Cm, Cc) and 25 were refused: no .ang and no .ctf. Hand: m is 2/m
    in the Laue class, code 20 (cell b-unique); mm2 is mmm, code 22."""
    lattice = MONOCLINIC_B if point_group == "m" else ORTHORHOMBIC
    source, cw = _checkpoint(tmp_path, "Ph", space_group, point_group, lattice)
    ang, ctf = export_ang_ctf(cw.checkpoint_path, str(tmp_path / "o"), source_h5_path=str(source))
    assert ang and ctf
    from pathlib import Path
    symmetry = [ln for ln in Path(ang).read_text(encoding="utf-8").splitlines() if ln.startswith("# Symmetry")]
    assert [s.split()[-1] for s in symmetry] == ["20" if point_group == "m" else "22"]


def test_a_refused_phase_costs_the_ang_and_ctf_record_only(tmp_path):
    """A phase with no point group has no code: the .ang / .ctf of that file is refused with the
    reason in the record, and the light .h5 of the same file is still written."""
    source, cw = _checkpoint(tmp_path, "Ph", 225, "m-3m", CUBIC)
    import backend.api.services.result_exporter as exporter
    real = exporter._tsl_codes_of_map

    def refusing(xmap):
        raise ang_export.SymmetryUnknown("phase 'Ph': no code")

    exporter._tsl_codes_of_map = refusing
    try:
        res = export_all(str(source), cw.checkpoint_path, str(tmp_path / "o"),
                         formats=["ang", "ctf", "h5_light"])
    finally:
        exporter._tsl_codes_of_map = real
    assert res["ang"] is None and res["ctf"] is None
    assert "no code" in res["_errors"]["ang_ctf"]
    assert res["h5_light"] and "h5_light" not in res["_errors"]
