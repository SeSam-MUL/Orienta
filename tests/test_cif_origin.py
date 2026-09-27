"""The origin choice has to be honoured before the CIF is expanded, not after.

`tests/test_origin_choice.py` covers the reading end: a `.xtal` that records its
origin choice expands correctly. This covers the writing end, which is where the
damage was actually done — `Si.xtal` holds a 16-atom cell because the converter
asked pymatgen to expand an origin-choice-2 coordinate and pymatgen answered
with choice-1 operators.

The load-bearing measurement is `test_pymatgen_ignores_the_origin_marker`: every
way of writing "origin choice 2" on the H-M symbol comes back at twice the
density. That is why stamping `SpaceGroupSetting = 2` into the .xtal cannot fix
anything — by then the wrong orbit already exists.

The synthetic cells here are the two library files that are wrong, written out
in full, so these tests still mean something in a clone where `Database/` is
absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("spglib")

from backend.forward_sim.crystal.cif_origin import (  # noqa: E402
    CifOriginError,
    analyse_cif,
    choice_from_symops,
    structure_from_cif,
)

_AVOGADRO = 6.02214076e23
_DATABASE = Path(__file__).resolve().parents[1] / "Database"

_HEADER = """data_{block}
_cell_length_a {a}
_cell_length_b {a}
_cell_length_c {a}
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_Int_Tables_number {number}
_symmetry_space_group_name_H-M '{symbol}'
loop_
_atom_site_label
_atom_site_type_symbol
{wyckoff_column}_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
"""


def _cif(tmp_path: Path, name: str, *, a: float, number: int, symbol: str,
         sites: list[tuple[str, str, str, str, str, str]], block: str = "test",
         occupancies: list[str] | None = None) -> Path:
    """A minimal CIF. An empty Wyckoff symbol drops the column rather than
    leaving a hole in the loop — a short row is a malformed CIF, not a file
    that declines to say which site it is."""
    with_wyckoff = any(s[2] for s in sites)
    text = _HEADER.format(
        block=block, a=a, number=number, symbol=symbol,
        wyckoff_column="_atom_site_Wyckoff_symbol\n" if with_wyckoff else "")
    for i, (label, element, wyckoff, x, y, z) in enumerate(sites):
        occ = occupancies[i] if occupancies else "1"
        cells = [label, element] + ([wyckoff or "."] if with_wyckoff else []) + [x, y, z, occ]
        text += " ".join(cells) + "\n"
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _density(structure) -> float:
    return structure.composition.weight / (_AVOGADRO * structure.volume * 1e-24)


# --- silicon: the cell that started it ------------------------------------

def _silicon_choice_2(tmp_path: Path, symbol: str = "Fd-3m", wyckoff: str = "8a") -> Path:
    """Wyckoff 8a at (1/8,1/8,1/8) — how every database publishes silicon."""
    return _cif(tmp_path, "si.cif", a=5.43090, number=227, symbol=symbol,
                sites=[("Si", "Si", wyckoff, "0.125", "0.125", "0.125")])


@pytest.mark.parametrize("symbol", ["Fd-3m", "Fd-3m O2", "Fd-3m :2", "Fd-3m S", "Fd-3m Z"])
def test_pymatgen_ignores_the_origin_marker(tmp_path, symbol):
    """The premise of this whole module, measured rather than assumed.

    If this ever starts failing, pymatgen has learned to read the marker and
    `structure_from_cif` can be simplified — but until then, no amount of
    labelling in the .xtal repairs what the expansion already got wrong.
    """
    from pymatgen.io.cif import CifParser

    path = _silicon_choice_2(tmp_path, symbol=symbol)
    structure = CifParser(str(path)).parse_structures()[0]
    assert len(structure) == 16, f"{symbol}: expected pymatgen's origin-blind 16 atoms"
    assert _density(structure) == pytest.approx(4.658, abs=0.01)


def test_silicon_comes_out_at_its_real_density(tmp_path):
    structure, report = structure_from_cif(_silicon_choice_2(tmp_path))
    assert len(structure) == 8
    assert _density(structure) == pytest.approx(2.329, abs=0.005)
    assert report.shifted is True
    assert report.cif_origin_choice == 2
    assert any("Wyckoff" in e for e in report.evidence), report.evidence


def test_a_choice_1_silicon_cell_is_left_alone(tmp_path):
    """The other block of the real Si.cif: 8a sits at the origin in choice 1."""
    path = _cif(tmp_path, "si1.cif", a=5.43090, number=227, symbol="Fd-3m O1",
                sites=[("Si", "Si", "8a", "0", "0", "0")])
    structure, report = structure_from_cif(path)
    assert len(structure) == 8
    assert report.shifted is False
    assert report.cif_origin_choice == 1


# --- MgCu2: the one where the atom count stayed right ----------------------

def test_mgcu2_multiplicities_swap_so_the_count_hides_the_error(tmp_path):
    """sd_1816951: Mg 8a and Cu 16d, read origin-blind, become Mg16 Cu8.

    8 + 16 == 16 + 8, so the cell still has 24 atoms and nothing about its size
    looks wrong — the composition has simply inverted from MgCu2 to Mg2Cu, and
    the master pattern is then built from the wrong scatterers.
    """
    from pymatgen.io.cif import CifParser

    path = _cif(tmp_path, "mgcu2.cif", a=7.0309, number=227, symbol="Fd-3m",
                sites=[("Mg", "Mg", "8a", "0.125", "0.125", "0.125"),
                       ("Cu", "Cu", "16d", "0.5", "0.5", "0.5")])

    blind = CifParser(str(path)).parse_structures()[0]
    assert len(blind) == 24, "the count is not the tell here"
    assert blind.composition.get_el_amt_dict() == {"Mg": 16.0, "Cu": 8.0}

    structure, report = structure_from_cif(path)
    assert report.shifted is True
    assert structure.composition.get_el_amt_dict() == {"Mg": 8.0, "Cu": 16.0}
    assert _density(structure) == pytest.approx(5.787, abs=0.01)


# --- how the choice is decided --------------------------------------------

@pytest.mark.parametrize("choice", ["1", "2"])
def test_explicit_operators_are_read_back_as_the_setting_they_are(choice):
    """Every two-origin group, both settings, straight out of spglib."""
    import spglib

    from backend.forward_sim.crystal.origin_choice import ORIGIN_SHIFT_TO_CHOICE_1

    def _xyz_strings(hall):
        data = spglib.get_symmetry_from_database(hall)
        out = []
        for R, t in zip(data["rotations"], data["translations"]):
            terms = []
            for row, shift in zip(R, t):
                parts = [f"{'+' if c > 0 else '-'}{axis}"
                         for c, axis in zip(row, "xyz") if c]
                if abs(shift) > 1e-9:
                    parts.append(f"{shift:+.6f}")
                terms.append("".join(parts) or "0")
            out.append(",".join(terms))
        return out

    halls = {}
    for hall in range(1, 531):
        t = spglib.get_spacegroup_type(hall)
        if t.get("choice") == choice:
            halls.setdefault(t["number"], hall)

    for number in sorted(ORIGIN_SHIFT_TO_CHOICE_1):
        hall = halls.get(number)
        assert hall, f"spglib has no choice-{choice} entry for space group {number}"
        assert choice_from_symops(_xyz_strings(hall), number) == int(choice), (
            f"space group {number}: choice-{choice} operators were not recognised"
        )


def test_a_block_that_contradicts_itself_is_refused(tmp_path):
    """Symbol says choice 1, the Wyckoff multiplicity says choice 2."""
    path = _silicon_choice_2(tmp_path, symbol="Fd-3m O1")
    with pytest.raises(CifOriginError, match="contradicts itself"):
        structure_from_cif(path)


def test_a_block_that_says_nothing_is_refused(tmp_path):
    """No operators, no Wyckoff, no formula, no marker — and two origins."""
    path = _cif(tmp_path, "mute.cif", a=5.43090, number=227, symbol="Fd-3m",
                sites=[("Si", "Si", "", "0.125", "0.125", "0.125")])
    with pytest.raises(CifOriginError, match="says nothing that decides"):
        structure_from_cif(path)


def test_a_single_origin_group_is_never_shifted(tmp_path):
    """206 of the 230 groups cannot have this problem and must not be touched."""
    path = _cif(tmp_path, "al.cif", a=4.0490, number=225, symbol="Fm-3m",
                sites=[("Al", "Al", "4a", "0", "0", "0")])
    structure, report = structure_from_cif(path)
    assert report.shifted is False
    assert len(structure) == 4
    assert _density(structure) == pytest.approx(2.700, abs=0.005)


def test_the_formula_sum_can_decide_when_there_are_no_wyckoff_symbols(tmp_path):
    path = _cif(tmp_path, "formula.cif", a=5.43090, number=227, symbol="Fd-3m",
                sites=[("Si", "Si", "", "0.125", "0.125", "0.125")])
    text = path.read_text(encoding="utf-8").replace(
        "_symmetry_Int_Tables_number", "_chemical_formula_sum 'Si8'\n_symmetry_Int_Tables_number")
    path.write_text(text, encoding="utf-8")
    structure, report = structure_from_cif(path)
    assert len(structure) == 8
    assert any("formula" in e for e in report.evidence), report.evidence


# --- against the real library ---------------------------------------------

_LIBRARY = _DATABASE / "CIF_Library"
_needs_library = pytest.mark.skipif(
    not _LIBRARY.is_dir(), reason="Database/CIF_Library is not present in this checkout"
)


@_needs_library
def test_every_library_cif_still_builds():
    """No refusals, and only the two known files move.

    This is the regression net: 34 of the 36 CIFs must come out of the new path
    exactly as they came out of the old one.
    """
    refused, shifted = [], []
    for path in sorted(_LIBRARY.rglob("*.cif")):
        if ".P1-backup-" in path.name:
            continue
        try:
            _, report = structure_from_cif(path)
        except CifOriginError as exc:
            refused.append(f"{path.name}: {exc}")
            continue
        if report.shifted:
            shifted.append(path.stem)
    assert refused == []
    assert sorted(shifted) == ["Si", "sd_1816951"], shifted


@_needs_library
@pytest.mark.parametrize("stem, atoms, rho", [
    ("Si", 8, 2.329),
    ("sd_1816951", 24, 5.787),
    ("Al3Fe2Si_mp-1190708_symmetrized", 96, 4.726),
])
def test_the_three_two_origin_library_files(stem, atoms, rho):
    """Al3Fe2Si carries explicit choice-1 operators and needs no correction."""
    path = next(_LIBRARY.rglob(f"{stem}.cif"), None)
    if path is None:
        pytest.skip(f"{stem}.cif is not in this checkout")
    structure, _ = structure_from_cif(path)
    assert len(structure) == atoms
    assert _density(structure) == pytest.approx(rho, abs=0.01)


@_needs_library
def test_the_multi_block_silicon_file_picks_a_usable_block():
    path = next(_LIBRARY.rglob("Si.cif"), None)
    if path is None:
        pytest.skip("Si.cif is not in this checkout")
    verdicts = analyse_cif(path)
    assert len(verdicts) >= 2
    assert all(v.usable for v in verdicts), [v.problem for v in verdicts if not v.usable]
    # both the standardised and the published block describe the same crystal
    assert {v.choice for v in verdicts} == {1, 2}


# --- what the review broke -------------------------------------------------
#
# Three findings, each of which turned a loud failure into a silent wrong
# answer. The old path (pymatgen's CifParser) refused all three inputs; the new
# one accepted them, so these are regressions this module introduced and the
# tests are the proof they are shut.

def _symop_xyz_strings(number: int, choice: str) -> list[str]:
    """spglib's operators for one setting, as the xyz strings a CIF carries."""
    import spglib

    hall = next(h for h in range(1, 531)
                if spglib.get_spacegroup_type(h)["number"] == number
                and spglib.get_spacegroup_type(h).get("choice") == choice)
    data = spglib.get_symmetry_from_database(hall)
    out = []
    for R, t in zip(data["rotations"], data["translations"]):
        terms = []
        for row, shift in zip(R, t):
            parts = [f"{'+' if c > 0 else '-'}{axis}" for c, axis in zip(row, "xyz") if c]
            if abs(shift) > 1e-9:
                parts.append(f"{shift:+.6f}")
            terms.append("".join(parts) or "0")
        out.append(",".join(terms))
    return out


def test_a_cif_that_lists_a_whole_cell_is_refused(tmp_path):
    """Expanding an already-expanded cell gave 64 silicon atoms at 18.6 g/cm3.

    `_build_shifted` expands every listed site, so a loop that is not an
    asymmetric unit gets expanded a second time. pymatgen refuses such a file
    ("occupancies sum to > 1"); this must too, because the number that comes
    out of it is the density the Monte Carlo runs on.

    The block carries explicit choice-2 operators, so the origin choice is
    settled and the builder is actually reached — with only Wyckoff symbols to
    go on, a doubled listing destroys its own evidence and gets refused one
    step earlier, which proves nothing about this guard.
    """
    # (0.125, 0.375, 0.375) is a genuine member of the orbit of
    # (0.125, 0.125, 0.125) under sg 227: the same atom, listed twice.
    path = _cif(tmp_path, "full.cif", a=5.43090, number=227, symbol="Fd-3m",
                sites=[("Si1", "Si", "", "0.125", "0.125", "0.125"),
                       ("Si2", "Si", "", "0.125", "0.375", "0.375")])
    ops = _symop_xyz_strings(227, "2")
    symop_loop = "loop_\n_symmetry_equiv_pos_as_xyz\n" + "\n".join(f"'{o}'" for o in ops)
    text = path.read_text(encoding="utf-8").replace(
        "loop_\n_atom_site_label", symop_loop + "\nloop_\n_atom_site_label", 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(CifOriginError, match="already been expanded"):
        structure_from_cif(path)


def test_a_mixed_site_is_not_mistaken_for_an_expanded_cell(tmp_path):
    """Two species at the SAME coordinate are one site, and must still build."""
    path = _cif(tmp_path, "mixed.cif", a=5.43090, number=227, symbol="Fd-3m",
                sites=[("Si1", "Si", "8a", "0.125", "0.125", "0.125"),
                       ("Ge1", "Ge", "8a", "0.125", "0.125", "0.125")],
                occupancies=["0.6", "0.4"])
    structure, report = structure_from_cif(path)
    assert report.shifted is True
    # The eightfold site, shared: 8 x 0.6 silicon and 8 x 0.4 germanium. The
    # sites come back as coincident pairs rather than one disordered site --
    # a different representation of the same cell, which _build_xtal_atomdata
    # handles either way.
    amounts = structure.composition.get_el_amt_dict()
    assert amounts["Si"] == pytest.approx(4.8)
    assert amounts["Ge"] == pytest.approx(3.2)


def test_a_coordinate_that_cannot_be_read_raises_instead_of_dropping_the_atom(tmp_path):
    """MgCu2 with copper written as `1/2` came back as Mg8 at 0.929 g/cm3.

    The atom was dropped and nothing said so — and `_sites` feeds the origin
    decision as well as the build, so a dropped atom can also remove the
    evidence that would have refused the file.
    """
    path = _cif(tmp_path, "fraction.cif", a=7.0309, number=227, symbol="Fd-3m",
                sites=[("Mg", "Mg", "8a", "0.125", "0.125", "0.125"),
                       ("Cu", "Cu", "16d", "1/2", "1/2", "1/2")])
    with pytest.raises(CifOriginError, match="cannot read"):
        structure_from_cif(path)


def test_a_site_that_declines_to_give_a_coordinate_is_still_skipped(tmp_path):
    """`?` and `.` are the CIF's way of saying "no value" — not a parse failure."""
    from backend.forward_sim.crystal.cif_origin import _sites

    from pymatgen.io.cif import CifFile

    path = _cif(tmp_path, "unknown.cif", a=5.43090, number=227, symbol="Fd-3m O1",
                sites=[("Si", "Si", "8a", "0", "0", "0"),
                       ("Xx", "Si", "8a", "?", "?", "?")])
    block = list(CifFile.from_file(str(path)).data.values())[0]
    assert len(_sites(block.data)) == 1


@pytest.mark.parametrize("symbol, expected", [
    ("Fd-3m :2", 2), ("Fd-3m:1", 1), ("Fd-3m O1", 1), ("Fd-3m O2", 2), ("Fd-3m o2", 2),
    ("Fd-3m", None), ("Fd-3m S", None), ("Fd-3m Z", None),
    ("P 1", None), ("C 1 2/c 1", None), ("A 1 2/a 1", None),
])
def test_only_unambiguous_origin_markers_are_read(symbol, expected):
    """Two separate bugs in one rule.

    A bare trailing digit matched `'C 1 2/c 1'` and `'P 1'`, both of which are
    in this library, and read them as "origin choice 1". And S/Z is no longer
    read at all: the International Tables do not define it, the two readings in
    circulation are opposites, and no CIF here uses it — so the rule could never
    be checked. Inverting it used to change silicon from 8 atoms to 16 without
    failing a single test.
    """
    from backend.forward_sim.crystal.cif_origin import choice_from_symbol

    assert choice_from_symbol(symbol) == expected
