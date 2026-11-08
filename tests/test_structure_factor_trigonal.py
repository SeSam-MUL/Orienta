"""Structure factors of trigonal and hexagonal space groups: one sum per reflection.

diffsims' ``calculate_structure_factor`` sums the atoms for the first vector of
each symmetry family and copies the value to the rest of the family. The family
comes from ``phase.point_group``, and orix gives the two orientations of a point
group (``-31m``/``-3m1``, ``312``/``321``, ``31m``/``3m1``) one and the same
symmetry. For a space group with the other orientation the "family" holds
reflections that differ physically. Cr2N (P-31m) came out wrong in 42 of 212
reflections.

Every expectation is a sum written out here: the positions of a general atom
from the cell ``phase_cell`` completes, or the Cr2N positions of the
International Tables (6k and 2d of P-31m), with f(g) of a lone atom at the origin
as the only shared ingredient. Nothing is taken from ``calculate_structure_factors``
or from diffsims' family sharing.
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("orix")
pytest.importorskip("diffsims")

from diffpy.structure import Atom, Lattice, Structure  # noqa: E402
from diffsims.crystallography import ReciprocalLatticeVector as R  # noqa: E402
from diffsims.utils.sim_utils import _get_kinematical_structure_factor as _K  # noqa: E402
from orix.crystal_map import Phase  # noqa: E402

import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402
import phase_cell as pc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Hand sums
# --------------------------------------------------------------------------


def _f_single(element, lattice, hkl):
    """f(g) of one element: a lone atom at the origin gives exactly the atomic
    scattering factor, whatever the cell."""
    st = Structure(atoms=[Atom(element, xyz=[0, 0, 0])], lattice=Lattice(*lattice))
    return np.abs(_K(st, hkl, st.lattice.rnorm(hkl), scattering_params="xtables"))


def _hand_sum(lattice, sites, hkl):
    """|F| = |sum_j occ_j f_j(g) exp(2 pi i h.x_j)| for ``sites`` = (element, occupancy, xyz)."""
    hkl = np.asarray(hkl, float)
    F = np.zeros(len(hkl), complex)
    for el, occ, pos in sites:
        F += occ * _f_single(el, lattice, hkl) * np.exp(2j * np.pi * (hkl @ np.asarray(pos, float)))
    return np.abs(F)


def _hkl_grid(lattice, d_min=1.1, span=4):
    lat = Lattice(*lattice)
    hk = np.array([h for h in itertools.product(range(-span, span + 1), repeat=3)
                   if h != (0, 0, 0)], float)
    return hk[1.0 / lat.rnorm(hk) >= d_min]


def _production_F(phase, hkl, *, plain=False):
    """|F| as the reflector code gets it (``plain``: diffsims' own method)."""
    r = R(pc.complete_unit_cell(phase), hkl=np.asarray(hkl, float))
    if plain:
        r.calculate_structure_factor()
    else:
        pc.calculate_structure_factors(r)
    return np.abs(np.asarray(r.structure_factor).ravel())


# --------------------------------------------------------------------------
# Cr2N, P-31m (162): the four reflections of the finding
# --------------------------------------------------------------------------

CR2N_LATTICE = (4.8, 4.8, 4.472, 90, 90, 120)
_X, _Z = 0.346, 0.244

#: Cr on 6k (x,0,z) of P-31m: (x,0,z) (0,x,z) (-x,-x,z) (-x,0,-z) (0,-x,-z) (x,x,-z);
#: N on 2d (1/3,2/3,z) with z = 1/2 (0.74) and z = 0 (0.12), and on 1a (0,0,0).
CR2N_SITES = (
    [("Cr", 1.0, p) for p in [(_X, 0, _Z), (0, _X, _Z), (-_X, -_X, _Z),
                              (-_X, 0, -_Z), (0, -_X, -_Z), (_X, _X, -_Z)]]
    + [("N", .74, (1 / 3, 2 / 3, .5)), ("N", .74, (2 / 3, 1 / 3, .5)),
       ("N", .12, (1 / 3, 2 / 3, 0)), ("N", .12, (2 / 3, 1 / 3, 0)),
       ("N", 1.0, (0, 0, 0))]
)

#: (hkl, |F|) summed by hand from the positions above with the International Tables
#: (xtables) scattering factors the code uses, no Debye-Waller factor. pymatgen reads
#: the same CIF to 0.053 / 8.768 / 16.104 / 14.817. (A hand sum with the Lobato
#: factors instead gives 0.057 / 8.795 / 16.136 / 14.847: 0.3 % apart, same pattern.)
CR2N_REFERENCE = [((2, 1, -1), 0.0533), ((1, 1, 2), 8.768), ((1, 1, 1), 16.104),
                  ((2, -1, 1), 14.817)]

CR2N_CIF = """data_Cr2N
_cell_length_a 4.80000
_cell_length_b 4.80000
_cell_length_c 4.47200
_cell_angle_alpha 90.0000
_cell_angle_beta 90.0000
_cell_angle_gamma 120.0000
_symmetry_space_group_name_H-M 'P -3 1 m'
_symmetry_Int_Tables_number 162
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Cr1 Cr 0.346000 0.000000000000 0.244000 1.0000
N2 N 0.333333333333 0.666666666667 0.500000000000 0.7400
N3 N 0.333333333333 0.666666666667 0.000000000000 0.1200
N4 N 0.000000000000 0.000000000000 0.000000000000 1.0000
"""


@pytest.fixture(scope="module")
def cr2n(tmp_path_factory):
    path = tmp_path_factory.mktemp("cr2n") / "Cr2N.cif"
    path.write_text(CR2N_CIF, encoding="utf-8")
    phase = eu.hough_phase_from_cif(str(path))
    eu._normalize_element_labels(phase)
    return phase


def test_the_hand_sums_of_the_finding_are_the_reference_numbers():
    """The reference numbers are what the sum written out above gives."""
    hkl = np.array([h for h, _ in CR2N_REFERENCE], float)
    got = _hand_sum(CR2N_LATTICE, CR2N_SITES, hkl)
    assert np.allclose(got, [f for _, f in CR2N_REFERENCE], atol=5e-4)


def test_cr2n_four_reflections_against_the_reference(cr2n):
    """Asked together with their twins, so that family sharing would bite: (2 1 -1)
    next to (1 2 -1) and (2 1 1) is the pair the old code mixed up."""
    wanted = [h for h, _ in CR2N_REFERENCE]
    companions = [(1, 2, -1), (2, 1, 1), (-1, 2, 1), (1, 1, -2), (2, -1, -1)]
    hkl = np.array(wanted + companions, float)
    got = _production_F(cr2n, hkl)
    for (h, want), f in zip(CR2N_REFERENCE, got):
        assert f == pytest.approx(want, abs=5e-4), h
    # and every companion equals its own hand sum
    assert np.allclose(got, _hand_sum(CR2N_LATTICE, CR2N_SITES, hkl), rtol=0, atol=1e-8)


def test_cr2n_every_reflection_equals_its_hand_sum(cr2n):
    hkl = _hkl_grid(CR2N_LATTICE, d_min=1.0, span=5)
    assert len(hkl) > 150
    got = _production_F(cr2n, hkl)
    want = _hand_sum(CR2N_LATTICE, CR2N_SITES, hkl)
    assert np.abs(got - want).max() <= 1e-8 * want.max()


_NOW_RIGHT = (
    "diffsims now computes these structure factors correctly for a plain orix Phase: "
    "the family sharing no longer mixes reflections of the seven space groups. That "
    "would make the per-reflection sum in phase_cell.calculate_structure_factors (and "
    "PER_REFLECTION_SPACE_GROUPS) unnecessary for plain phases. Check the orix and "
    "diffsims versions, decide whether to remove the sum, and delete these guards "
    "together with it. Do not turn this into a skip: it has to keep failing until then.")


def test_cr2n_diffsims_alone_is_wrong_here(cr2n):
    """Guard for the tests above, on a plain orix ``Phase`` (what a master pattern or
    an outside caller hands in; the ``FramePhase`` of the fixture would be computed
    right by diffsims): diffsims' own method misreports Cr2N, and
    ``calculate_structure_factors`` does not."""
    plain_phase = Phase(name="Cr2N plain", space_group=162, structure=cr2n.structure)
    assert type(plain_phase) is Phase
    hkl = _hkl_grid(CR2N_LATTICE, d_min=1.0, span=5)
    want = _hand_sum(CR2N_LATTICE, CR2N_SITES, hkl)
    plain = _production_F(plain_phase, hkl, plain=True)
    bad = int((np.abs(plain - want) > 1e-6 * want.max()).sum())
    assert bad > 20, _NOW_RIGHT
    ours = _production_F(plain_phase, hkl)
    assert np.abs(ours - want).max() <= 1e-8 * want.max()


def test_cr2n_reflector_list_rows_carry_their_own_value(cr2n):
    """The default Hough list: every row's |F| is the sum for that very row, and the
    twin of the extinct-looking row is not given its value."""
    ref = eu._filtered_reflectors(cr2n, 1.0, 0.1)
    hkl = np.rint(np.asarray(ref.hkl, float)).astype(int)
    got = np.abs(np.asarray(ref.structure_factor)).ravel()
    want = _hand_sum(CR2N_LATTICE, CR2N_SITES, hkl)
    assert len(hkl) > 100
    assert np.abs(got - want).max() <= 1e-8 * want.max()
    rows = {tuple(r) for r in hkl.tolist()}
    assert (2, 1, -1) not in rows        # |F| = 0.053, far below a tenth of 16.1
    assert (2, 1, 1) in rows             # its twin, |F| = 2.9


def test_cr2n_single_family_lookup_and_custom_spec_rows(cr2n):
    """``_extra_family`` (the family table's detail) and ``rlv_for_spec`` (a custom
    selection) give the value of the reflection they are asked about."""
    for rep, want in [((2, 1, -1), 0.0533), ((1, 1, 2), 8.768), ((2, -1, 1), 14.817)]:
        assert hr._extra_family(cr2n, rep)["f"] == pytest.approx(want, abs=5e-4), rep
    rlv = hr.rlv_for_spec(cr2n, {"mode": "custom", "families": [[1, 1, 2], [2, 1, 1]]})
    hkl = np.rint(np.asarray(rlv.hkl, float)).reshape(-1, 3)
    got = np.abs(np.asarray(rlv.structure_factor)).ravel()
    want = _hand_sum(CR2N_LATTICE, CR2N_SITES, hkl)
    assert len(hkl) >= 12
    assert np.abs(got - want).max() <= 1e-8 * want.max()


# --------------------------------------------------------------------------
# Every trigonal and hexagonal space group: a general-position structure
# --------------------------------------------------------------------------

GENERAL_LATTICE = (4.1, 4.1, 6.3, 90, 90, 120)
GENERAL_ATOMS = [("Al", (.11, .23, .31)), ("Fe", (.42, .07, .65))]

#: Orix maps these to a point group whose orientation is not the space group's,
#: measured with orix 0.14.1 and 0.15.0 and the structure above.
WRONG_WITH_DIFFSIMS_ALONE = (149, 151, 153, 157, 159, 162, 163)


def _general_cell(sg):
    st = Structure(atoms=[Atom(el, xyz=xyz) for el, xyz in GENERAL_ATOMS],
                   lattice=Lattice(*GENERAL_LATTICE))
    return pc.complete_unit_cell(Phase(name=f"sg{sg}", space_group=sg, structure=st))


def _hand_sum_of_cell(cell, hkl):
    sites = [(str(a.element), float(a.occupancy), f)
             for a, f in zip(cell.structure, pc.true_fractional(cell))]
    return _hand_sum(GENERAL_LATTICE, sites, hkl)


@pytest.mark.parametrize("sg", range(143, 195))
def test_every_reflection_is_its_own_sum(sg):
    """Each |F| equals the sum over the atoms of its own reflection, in every
    trigonal and hexagonal space group (143-194). In the seven groups of
    ``PER_REFLECTION_SPACE_GROUPS`` that is the per-reflection sum; in all others
    diffsims' own method is used, so a group outside the set that diffsims gets
    wrong fails here."""
    cell = _general_cell(sg)
    hkl = _hkl_grid(GENERAL_LATTICE)
    assert len(hkl) > 200
    r = R(cell, hkl=hkl)
    pc.calculate_structure_factors(r)
    got = np.abs(np.asarray(r.structure_factor).ravel())
    want = _hand_sum_of_cell(cell, hkl)
    assert np.abs(got - want).max() <= 1e-8 * want.max()


@pytest.mark.parametrize("sg", [g for g in range(143, 195) if g not in WRONG_WITH_DIFFSIMS_ALONE])
def test_groups_outside_the_set_keep_diffsims_values(sg):
    """The 45 other groups (150, 152, ..., 187-190, ...) are deliberately left to
    diffsims: it is right for a general-position structure (the test above), and
    the helper returns its values bit for bit. A per-reflection sum there would
    differ in the last digits and, for real structures such as pi-Al8FeMg3Si6 (189),
    in the fourth."""
    assert sg not in pc.PER_REFLECTION_SPACE_GROUPS
    cell = _general_cell(sg)
    hkl = _hkl_grid(GENERAL_LATTICE)
    r_helper, r_plain = R(cell, hkl=hkl), R(cell, hkl=hkl)
    pc.calculate_structure_factors(r_helper)
    r_plain.calculate_structure_factor()
    assert np.array_equal(np.asarray(r_helper.structure_factor),
                          np.asarray(r_plain.structure_factor))


@pytest.mark.parametrize("sg", WRONG_WITH_DIFFSIMS_ALONE)
def test_the_listed_groups_are_really_wrong_with_diffsims_alone(sg):
    """Guard: for these groups the family sharing does misreport, so the test above
    is not green by accident. Plain orix ``Phase``; fails (no skip) if a later
    diffsims/orix repairs it, with the steps to take."""
    cell = _general_cell(sg)
    hkl = _hkl_grid(GENERAL_LATTICE)
    r = R(cell, hkl=hkl)
    r.calculate_structure_factor()
    plain = np.abs(np.asarray(r.structure_factor).ravel())
    want = _hand_sum_of_cell(cell, hkl)
    bad = int((np.abs(plain - want) > 1e-6 * want.max()).sum())
    assert bad >= 40, f"space group {sg}: " + _NOW_RIGHT
    r2 = R(cell, hkl=hkl)
    pc.calculate_structure_factors(r2)
    ours = np.abs(np.asarray(r2.structure_factor).ravel())
    assert np.abs(ours - want).max() <= 1e-8 * want.max()


def test_the_set_is_exactly_the_seven_wrong_groups():
    """Not the pair partners (150, 152, ..., 187-190): they are right with diffsims
    and a per-reflection sum there moves pi-Al8FeMg3Si6 (see the guard below)."""
    assert pc.PER_REFLECTION_SPACE_GROUPS == frozenset(WRONG_WITH_DIFFSIMS_ALONE)
    assert not {150, 152, 154, 156, 158, 160, 161, 164, 187, 188, 189, 190, 194}         & pc.PER_REFLECTION_SPACE_GROUPS


def test_other_space_groups_keep_diffsims_own_method(monkeypatch):
    """Outside the set the helper IS diffsims' method (no change for cubic, hcp, ...)."""
    calls = []
    orig = R.calculate_structure_factor

    def spy(self, *a, **k):
        calls.append(1)
        return orig(self, *a, **k)

    monkeypatch.setattr(R, "calculate_structure_factor", spy)
    cell = _general_cell(194)
    r = R(cell, hkl=np.array([[1, 0, 0], [1, 1, 0]], float))
    pc.calculate_structure_factors(r)
    assert calls == [1]
    r2 = R(_general_cell(162), hkl=np.array([[1, 0, 0], [1, 1, 0]], float))
    pc.calculate_structure_factors(r2)
    assert calls == [1]          # P-31m: not diffsims' method


# --------------------------------------------------------------------------
# pi-Al8FeMg3Si6 (P-62m, 189): the default Hough list does not move
# --------------------------------------------------------------------------

#: Perlitz & Westgren 1942, ICSD 27140; coordinates as published (0.3333 0.6667).
PI_CIF = """data_pi
_cell_length_a 6.62
_cell_length_b 6.62
_cell_length_c 7.92
_cell_angle_alpha 90.
_cell_angle_beta 90.
_cell_angle_gamma 120.
_space_group_name_H-M_alt 'P -6 2 m'
_space_group_IT_number 189
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Fe1 Fe 0 0 0 1.
Al1 Al 0 0 0.5 1.
Al2 Al 0.403 0 0 1.
Mg1 Mg 0.444 0 0.5 1.
Al3 Al 0.3333 0.6667 0.231 1.
Si1 Si 0.75 0 0.222 1.
"""


def _default_list(phase):
    eu._FILTERED_CACHE.clear()
    del eu._FILTERED_CACHE_ORDER[:]
    hr._CACHE.clear()
    ref = eu._reflectors_for_phase(phase.deepcopy(), 1.0, 0.1, 70)
    return (np.rint(np.asarray(ref.hkl, float)).astype(int),
            np.abs(np.asarray(ref.structure_factor)).ravel())


def test_pi_default_hough_list_is_the_same_before_and_after_the_fix(tmp_path, monkeypatch):
    """The measured regression: with 189 in the set, the per-reflection sum differs
    from diffsims' by 2.9e-4 of Fmax and changes the 70 rows kept. The list built by
    the production path must equal, row for row and value for value, the one built
    with diffsims' own method (the code before the fix)."""
    cif = tmp_path / "pi.cif"
    cif.write_text(PI_CIF, encoding="utf-8")
    phase = eu.hough_phase_from_cif(str(cif))
    assert phase.space_group.number == 189
    try:
        hkl_after, f_after = _default_list(phase)
        monkeypatch.setattr(eu, "calculate_structure_factors",
                            lambda rlv, scattering_params="xtables":
                            rlv.calculate_structure_factor(scattering_params))
        hkl_before, f_before = _default_list(phase)
    finally:
        eu._FILTERED_CACHE.clear()
        del eu._FILTERED_CACHE_ORDER[:]
        hr._CACHE.clear()
    assert len(hkl_before) == 70
    assert np.array_equal(hkl_after, hkl_before)
    assert np.array_equal(f_after, f_before)


# --------------------------------------------------------------------------
# The orientation symmetry of the other trigonal groups stays as orix has it
# --------------------------------------------------------------------------

def test_the_orientation_symmetry_of_the_other_trigonal_groups_is_untouched():
    """Only the seven groups of PER_REFLECTION_SPACE_GROUPS get a group of their
    own from ``frame_symmetry`` (tests/test_crystal_symmetry_trigonal.py); the
    others, and the hexagonal ones, are orix' objects."""
    from crystal_symmetry import frame_symmetry
    from orix.quaternion.symmetry import get_point_group
    for sg in (150, 152, 154, 156, 158, 164, 166, 187, 189):
        assert frame_symmetry(space_group=sg) is get_point_group(sg), sg
    for sg in pc.PER_REFLECTION_SPACE_GROUPS:
        assert frame_symmetry(space_group=sg) is not get_point_group(sg), sg


# --------------------------------------------------------------------------
# No call site can go back to diffsims' method
# --------------------------------------------------------------------------

_SKIP_DIRS = {"tests", "tasks", "_archive", "docs", "frontend", "electron", "node_modules",
              ".git", ".claude", "Database", "Test_data", "examples", "diagnostic_runs",
              "matlab_testskripts", "images_for_presentation", "branding", "dxa", "logs",
              "runtime", "build", "dist", "__pycache__", "licenses",
              "crystal-structures-for-ebsd-main"}
_DIRECT = re.compile(r"\.calculate_structure_factor\(")


def _source_files():
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT).parts
        if any(part in _SKIP_DIRS or part.startswith(".") for part in rel[:-1]):
            continue
        yield p


def test_no_module_calls_the_family_sharing_method_directly():
    """Every reflector path goes through ``phase_cell.calculate_structure_factors``."""
    offenders = []
    for p in _source_files():
        if p.name == "phase_cell.py":
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            code = line.split("#", 1)[0]
            if _DIRECT.search(code):
                offenders.append(f"{p.relative_to(ROOT)}:{n}")
    assert not offenders, "call phase_cell.calculate_structure_factors instead: " + ", ".join(offenders)


def test_the_source_scan_sees_the_files_it_is_meant_to_police():
    names = {p.name for p in _source_files()}
    assert {"ebsd_utils.py", "hough_reflectors.py", "indexing_controller.py", "phase_cell.py"} <= names
