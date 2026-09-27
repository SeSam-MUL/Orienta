"""The library audit: does each .xtal still describe the crystal its CIF does?

A .xtal holds an asymmetric unit; what the simulation uses is its orbit under
the space group, and nothing in the file shows that orbit. Two numbers do — how
many atoms the cell has and what it weighs — so the audit expands the .xtal the
way the forward simulation does and holds it against the CIF.

Run against the real library on 2026-09-26 it found three of 35 files wrong:
Si (16 atoms instead of 8), sd_1816951 (Mg32 Cu8 instead of Mg8 Cu16, the
origin-choice bug in its composition-inverting form) and
Mg17Al12_mp-2151_conventional_standard (74 atoms instead of 58, a different
fault, and a duplicate of a file that is correct).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

pytest.importorskip("pymatgen")

from backend.forward_sim.crystal.xtal_audit import audit_library, audit_one  # noqa: E402

_SI_CIF = """data_si
_cell_length_a 5.43090
_cell_length_b 5.43090
_cell_length_c 5.43090
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number 227
_symmetry_space_group_name_H-M 'Fd-3m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_Wyckoff_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si Si 8a 0.125 0.125 0.125 1
"""


def _write_xtal(path: Path, *, space_group: int, lattice_a: float,
                sites: list[tuple[int, float, float, float]], setting: int = 1) -> Path:
    """Minimal EMsoft CrystalData. AtomData is (5, N): x, y, z, occupancy, DWF."""
    atomdata = np.zeros((5, len(sites)), dtype=np.float32)
    for i, (_Z, x, y, z) in enumerate(sites):
        atomdata[:, i] = (x, y, z, 1.0, 0.005)
    with h5py.File(path, "w") as f:
        cd = f.create_group("CrystalData")
        cd.create_dataset("AtomData", data=atomdata)
        cd.create_dataset("Atomtypes", data=np.array([s[0] for s in sites], dtype=np.int32))
        cd.create_dataset("Natomtypes", data=np.array([len(sites)], dtype=np.int32))
        cd.create_dataset("LatticeParameters",
                          data=np.array([lattice_a, lattice_a, lattice_a, 90.0, 90.0, 90.0],
                                        dtype=np.float32))
        cd.create_dataset("SpaceGroupNumber", data=np.array([space_group], dtype=np.int32))
        cd.create_dataset("SpaceGroupSetting", data=np.array([setting], dtype=np.int32))
        cd.create_dataset("CrystalSystem", data=np.array([1], dtype=np.int32))
    return path


def test_the_shipped_silicon_cell_is_reported_as_differing(tmp_path):
    """(0.375, 0.125, 0.375) is what Si.xtal holds: the 16-fold site."""
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF, encoding="utf-8")
    xtal = _write_xtal(tmp_path / "Si.xtal", space_group=227, lattice_a=0.54309,
                       sites=[(14, 0.375, 0.125, 0.375)])

    row = audit_one(xtal, cif)
    assert row.status == "differs"
    assert (row.xtal_atoms, row.cif_atoms) == (16, 8)
    assert row.xtal_rho == pytest.approx(4.658, abs=0.01)
    assert row.cif_rho == pytest.approx(2.329, abs=0.01)
    assert "origin choice 2" in row.note


def test_the_corrected_silicon_cell_passes(tmp_path):
    """The same CIF, with the asymmetric unit moved to origin choice 1."""
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF, encoding="utf-8")
    xtal = _write_xtal(tmp_path / "Si.xtal", space_group=227, lattice_a=0.54309,
                       sites=[(14, 0.75, 0.75, 0.75)])

    row = audit_one(xtal, cif)
    assert row.ok, f"{row.status}: {row.xtal_formula} vs {row.cif_formula}"
    assert (row.xtal_atoms, row.cif_atoms) == (8, 8)
    assert row.xtal_rho == pytest.approx(2.329, abs=0.01)


def test_a_single_origin_phase_passes(tmp_path):
    """Aluminium: one origin, nothing to get wrong, and it must stay that way."""
    cif = tmp_path / "Al.cif"
    cif.write_text(_SI_CIF.replace("data_si", "data_al").replace("5.43090", "4.04900")
                   .replace("227", "225").replace("Fd-3m", "Fm-3m")
                   .replace("Si Si 8a 0.125 0.125 0.125", "Al Al 4a 0 0 0"), encoding="utf-8")
    xtal = _write_xtal(tmp_path / "Al.xtal", space_group=225, lattice_a=0.40490,
                       sites=[(13, 0.0, 0.0, 0.0)])

    row = audit_one(xtal, cif)
    assert row.ok, f"{row.status}: {row.note}"
    assert row.xtal_atoms == 4
    assert row.xtal_rho == pytest.approx(2.700, abs=0.005)


def test_a_xtal_without_a_cif_is_reported_not_passed(tmp_path):
    """Silence is not a pass: an unmatched file has to be visible in the report."""
    xtal_dir = tmp_path / "XTAL_Library"
    cif_dir = tmp_path / "CIF_Library"
    xtal_dir.mkdir()
    cif_dir.mkdir()
    _write_xtal(xtal_dir / "Orphan.xtal", space_group=225, lattice_a=0.40490,
                sites=[(13, 0.0, 0.0, 0.0)])

    rows = audit_library(xtal_dir, cif_dir)
    assert [r.status for r in rows] == ["no cif"]
    assert not rows[0].ok


# --- one test per comparison ----------------------------------------------
#
# The "differs" test above trips all three checks at once, so disabling any one
# of them left the suite green (measured: atom-count, formula and density could
# each be deleted with 4 passed). These isolate two of them. The third,
# atom count, cannot be isolated: the formula string carries the counts, so
# anything that changes the count changes the formula too. It is kept for the
# message it produces, not for coverage it adds.

def test_only_the_density_check_can_see_a_wrong_lattice_constant(tmp_path):
    """Right atoms, right formula, cell 2 % too large.

    This is the comparison the whole branch exists for: a cell can have exactly
    the atoms its CIF says and still be the wrong crystal.
    """
    cif = tmp_path / "Si.cif"
    cif.write_text(_SI_CIF, encoding="utf-8")
    xtal = _write_xtal(tmp_path / "Si.xtal", space_group=227, lattice_a=0.54309 * 1.02,
                       sites=[(14, 0.75, 0.75, 0.75)])

    row = audit_one(xtal, cif)
    assert row.status == "differs"
    assert row.xtal_atoms == row.cif_atoms == 8          # the count agrees
    assert row.xtal_formula == row.cif_formula == "Si8"  # and so does the formula
    assert row.xtal_rho != pytest.approx(row.cif_rho, rel=0.01)


def test_only_the_formula_check_can_see_a_substituted_element(tmp_path):
    """Nickel where the CIF says cobalt: same count, densities 0.4 % apart.

    58.693 against 58.933 g/mol is inside the 1 % the density check allows, so
    nothing but the formula distinguishes them.
    """
    cif = tmp_path / "Co.cif"
    cif.write_text(_SI_CIF.replace("data_si", "data_co").replace("5.43090", "3.52380")
                   .replace("227", "225").replace("Fd-3m", "Fm-3m")
                   .replace("Si Si 8a 0.125 0.125 0.125", "Co Co 4a 0 0 0"), encoding="utf-8")
    xtal = _write_xtal(tmp_path / "Co.xtal", space_group=225, lattice_a=0.35238,
                       sites=[(28, 0.0, 0.0, 0.0)])   # 28 = Ni, not Co

    row = audit_one(xtal, cif)
    assert row.status == "differs"
    assert row.xtal_atoms == row.cif_atoms == 4
    assert row.xtal_rho == pytest.approx(row.cif_rho, rel=0.01)
    assert (row.xtal_formula, row.cif_formula) == ("Ni4", "Co4")
