"""The complete unit cell of an orix ``Phase`` (``phase_cell``), every crystal system.

orix keeps atoms in a frame of its own and diffsims' ``sanitise_phase`` expands
them there, which is wrong wherever ``a`` is not parallel to ``a*`` or ``c`` not
to ``c*`` (hexagonal, trigonal, monoclinic, triclinic) and keeps duplicates
(orthorhombic Cmcm: 16 -> 24 atoms).

Every expectation here comes from outside the code under test: a hand
enumeration of the space-group operators written out in this file, the
closed-form structure factor of hcp, the extinction rule of the lattice, or
pymatgen reading the same CIF text. The only thing shared with the code under
test is the atomic scattering factor f(g), taken once per element and reflection
from a single atom at the origin, so that it cancels as a common factor.
"""
import itertools
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
import phase_cell as pc  # noqa: E402


# --------------------------------------------------------------------------
# The crystals. Operators are written out by hand (International Tables, as
# (rotation rows, translation)); a symbol like "-x" is spelled as a row.
# --------------------------------------------------------------------------

def _op(r, t=(0, 0, 0)):
    return np.array(r, float), np.array(t, float)


I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
NEG = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]


def _mono_c2m():
    # C2/m, unique axis b: +-(x,y,z), +-(-x,y,-z), plus C centring
    base = [_op(I3), _op(NEG),
            _op([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            _op([[1, 0, 0], [0, -1, 0], [0, 0, 1]])]
    return base + [(r, t + np.array([.5, .5, 0])) for r, t in base]


def _ortho_cmcm():
    # Cmcm (63): the eight operators of Pmcn's family listed in the Tables,
    # plus C centring
    h = [0, 0, .5]
    base = [_op(I3), _op([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], h),
            _op([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], h),
            _op([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), _op(NEG),
            _op([[1, 0, 0], [0, 1, 0], [0, 0, -1]], h),
            _op([[1, 0, 0], [0, -1, 0], [0, 0, 1]], h),
            _op([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])]
    return base + [(r, t + np.array([.5, .5, 0])) for r, t in base]


def _tri_p1bar():
    return [_op(I3), _op(NEG)]


def _hex_p63mmc_2c():
    # only the operators that move an atom on 2c (1/3,2/3,1/4); the other
    # twenty-two of the 24 operators of P6_3/mmc return the same two atoms
    return [_op(I3), _op([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            _op([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
            _op([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [0, 0, .5]),
            _op([[1, -1, 0], [1, 0, 0], [0, 0, 1]], [0, 0, .5]),
            _op([[0, 1, 0], [-1, 1, 0], [0, 0, 1]], [0, 0, .5])]


def _rhomb_r3m_6c():
    # R-3m, hexagonal axes, atom on 6c (0,0,z): +-(x,y,z) plus R centring
    base = [_op(I3), _op(NEG)]
    out = []
    for c in ([0, 0, 0], [2 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3]):
        out += [(r, t + np.array(c)) for r, t in base]
    return out


def _cubic_fcc():
    return [(np.eye(3), np.array(c)) for c in ([0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0])]


def _cubic_bcc():
    return [(np.eye(3), np.array(c)) for c in ([0, 0, 0], [.5, .5, .5])]


#: name -> (space-group number, lattice, [(element, site)], hand operators)
CRYSTALS = {
    "hcp-Mg": (194, (3.2094, 3.2094, 5.2108, 90, 90, 120), [("Mg", (1 / 3, 2 / 3, .25))],
               _hex_p63mmc_2c()),
    "alpha-Ti": (194, (2.95, 2.95, 4.68, 90, 90, 120), [("Ti", (1 / 3, 2 / 3, .25))],
                 _hex_p63mmc_2c()),
    "monoclinic-C2/m": (12, (9.0, 5.0, 4.5, 90, 105, 90),
                        [("Al", (.15, 0, .30)), ("Si", (0, .25, 0)), ("Mg", (.3, .2, .6))],
                        _mono_c2m()),
    "orthorhombic-Cmcm": (63, (4.0, 10.0, 7.0, 90, 90, 90),
                          [("Cu", (0, .4, .25)), ("Mg", (0, .13, .25)), ("Al", (0, .35, .05))],
                          _ortho_cmcm()),
    "trigonal-R-3m": (166, (4.31, 4.31, 11.27, 90, 90, 120), [("Sb", (0, 0, .2336))],
                      _rhomb_r3m_6c()),
    "triclinic-P-1": (2, (5.1, 6.2, 7.3, 80, 95, 100),
                      [("Al", (.1, .2, .3)), ("Si", (.6, .4, .9))], _tri_p1bar()),
    "cubic-Fm-3m": (225, (4.0495, 4.0495, 4.0495, 90, 90, 90), [("Al", (0, 0, 0))],
                    _cubic_fcc()),
    "cubic-Im-3m": (229, (2.8665, 2.8665, 2.8665, 90, 90, 90), [("Fe", (0, 0, 0))],
                    _cubic_bcc()),
}

#: Atoms in the cell: the multiplicities of the Wyckoff sites above (Tables).
#: C2/m: Al on 4i, Si on 4g, Mg on 8j; Cmcm: Cu 4c, Mg 4c, Al 8f.
EXPECTED_ATOMS = {"hcp-Mg": 2, "alpha-Ti": 2, "monoclinic-C2/m": 16, "orthorhombic-Cmcm": 16,
                  "trigonal-R-3m": 6, "triclinic-P-1": 4, "cubic-Fm-3m": 4, "cubic-Im-3m": 2}


def _cif_text(name):
    sg, lat, atoms, _ops = CRYSTALS[name]
    from diffpy.structure.spacegroups import GetSpaceGroup
    hm = GetSpaceGroup(sg).short_name
    lines = [f"data_{name.replace('/', '_')}",
             f"_symmetry_space_group_name_H-M   '{hm}'",
             f"_symmetry_Int_Tables_number      {sg}",
             f"_cell_length_a {lat[0]}", f"_cell_length_b {lat[1]}", f"_cell_length_c {lat[2]}",
             f"_cell_angle_alpha {lat[3]}", f"_cell_angle_beta {lat[4]}",
             f"_cell_angle_gamma {lat[5]}", "loop_", " _atom_site_label",
             " _atom_site_type_symbol", " _atom_site_fract_x", " _atom_site_fract_y",
             " _atom_site_fract_z", " _atom_site_occupancy"]
    for i, (el, (x, y, z)) in enumerate(atoms):
        lines.append(f"  {el}{i + 1} {el} {x:.8f} {y:.8f} {z:.8f} 1.0")
    return "\n".join(lines) + "\n"


def _from_cif(tmp_path, name):
    path = tmp_path / (name.replace("/", "_") + ".cif")
    path.write_text(_cif_text(name), encoding="utf-8")
    return eu.hough_phase_from_cif(str(path)), path


def _asymmetric_phase(name):
    """Only the asymmetric unit, as a user of orix would build it."""
    sg, lat, atoms, _ops = CRYSTALS[name]
    st = Structure(atoms=[Atom(el, xyz=site) for el, site in atoms], lattice=Lattice(*lat))
    return Phase(name=name, space_group=sg, structure=st)


def _hand_cell(name):
    """(element, fractional position mod 1) of the whole cell from the hand operators."""
    _sg, _lat, atoms, ops = CRYSTALS[name]
    cell = []
    for el, site in atoms:
        for r, t in ops:
            p = (r @ np.array(site, float) + t) % 1.0
            p = np.where(np.abs(p - 1.0) < 1e-9, 0.0, p)
            if not any(e == el and np.all(np.abs((q - p + .5) % 1 - .5) < 1e-6) for e, q in cell):
                cell.append((el, p))
    return cell


def _f_single(element, lattice, hkl):
    """f(g) of one element at the hkl of ``lattice``: a lone atom at the origin gives
    exactly the atomic scattering factor, whatever the cell."""
    st = Structure(atoms=[Atom(element, xyz=[0, 0, 0])], lattice=Lattice(*lattice))
    g = st.lattice.rnorm(hkl)
    return np.abs(_K(st, hkl, g, scattering_params="xtables"))


def _hand_F(name, hkl):
    """|F| from the hand-enumerated cell: sum f_j exp(2 pi i h.x_j)."""
    _sg, lat, _atoms, _ops = CRYSTALS[name]
    cell = _hand_cell(name)
    F = np.zeros(len(hkl), complex)
    for el, pos in cell:
        F += _f_single(el, lat, hkl) * np.exp(2j * np.pi * (hkl @ pos))
    return np.abs(F)


def _orienta_F(phase, hkl):
    r = R(pc.complete_unit_cell(phase), hkl=np.asarray(hkl, float))
    r.calculate_structure_factor()
    return np.abs(np.asarray(r.structure_factor).ravel())


def _hkl_grid(lattice, d_min=1.1, span=5):
    lat = Lattice(*lattice)
    hk = np.array([h for h in itertools.product(range(-span, span + 1), repeat=3)
                   if h != (0, 0, 0)], float)
    return hk[1.0 / lat.rnorm(hk) >= d_min]


def _atoms_mod1(phase):
    """Set of (element, rounded true-frame position) of a phase."""
    out = set()
    for a, f in zip(phase.structure, pc.true_fractional(phase)):
        f = np.round(f % 1.0, 3) % 1.0
        out.add((str(a.element),) + tuple(float(x) for x in f))
    return out


def _hand_set(name):
    return {(el,) + tuple(float(x) for x in np.round(p, 3) % 1.0) for el, p in _hand_cell(name)}


# --------------------------------------------------------------------------
# The cell itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CRYSTALS))
def test_the_cell_is_what_the_hand_operators_make(tmp_path, name):
    phase, _ = _from_cif(tmp_path, name)
    cell = pc.complete_unit_cell(phase)
    assert len(cell.structure) == EXPECTED_ATOMS[name]
    assert _atoms_mod1(cell) == _hand_set(name)


@pytest.mark.parametrize("name", list(CRYSTALS))
def test_asymmetric_unit_and_full_cell_give_the_same_cell(tmp_path, name):
    full, _ = _from_cif(tmp_path, name)
    asym = _asymmetric_phase(name)
    assert len(asym.structure) == len(CRYSTALS[name][2])
    a, b = pc.complete_unit_cell(asym), pc.complete_unit_cell(full)
    assert _atoms_mod1(a) == _atoms_mod1(b) == _hand_set(name)
    hkl = _hkl_grid(CRYSTALS[name][1])
    assert np.allclose(_orienta_F(asym, hkl), _orienta_F(full, hkl), atol=1e-9)


@pytest.mark.parametrize("name", list(CRYSTALS))
def test_completing_twice_is_completing_once(tmp_path, name):
    phase, _ = _from_cif(tmp_path, name)
    once = pc.complete_unit_cell(phase)
    twice = pc.complete_unit_cell(once)
    thrice = pc.complete_unit_cell(twice)
    assert len(once.structure) == len(twice.structure) == len(thrice.structure)
    assert _atoms_mod1(once) == _atoms_mod1(twice) == _atoms_mod1(thrice)
    assert np.allclose([a.xyz for a in once.structure], [a.xyz for a in twice.structure])


@pytest.mark.parametrize("name", list(CRYSTALS))
def test_structure_factors_equal_the_hand_sum(tmp_path, name):
    """|F| of the whole cell, summed by hand from the hand-enumerated atoms."""
    phase, _ = _from_cif(tmp_path, name)
    hkl = _hkl_grid(CRYSTALS[name][1])
    got, want = _orienta_F(phase, hkl), _hand_F(name, hkl)
    assert len(hkl) > 50
    assert np.abs(got - want).max() <= 1e-9 * want.max()


# --------------------------------------------------------------------------
# hcp: the closed form, and what it forbids
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hcp-Mg", "alpha-Ti"])
def test_hcp_structure_factor_is_the_closed_form(tmp_path, name):
    """|F| = 2 f(g) |cos(pi((h-k)/3 + l/2))| for two atoms on 2c of P6_3/mmc."""
    sg, lat, atoms, _ = CRYSTALS[name]
    element = atoms[0][0]
    phase, _ = _from_cif(tmp_path, name)
    hkl = _hkl_grid(lat, d_min=0.9, span=7)
    h, k, l = hkl.T
    want = 2 * _f_single(element, lat, hkl) * np.abs(np.cos(np.pi * ((h - k) / 3 + l / 2)))
    got = _orienta_F(phase, hkl)
    assert np.abs(got - want).max() <= 1e-9 * want.max()
    # the two families the old path got wrong, named: 000l with l odd and hh-2h-l
    # with l odd (3-index h h l) vanish, and so does (2 -1 1)
    for row in ([0, 0, 1], [0, 0, 3], [1, 1, 1], [2, 2, 1], [2, -1, 1], [1, 1, 3]):
        assert _orienta_F(phase, np.array([row]))[0] < 1e-9 * want.max(), row
    # and the strong ones are there: {0002} is the strongest of the low reflections
    assert _orienta_F(phase, np.array([[0, 0, 2]]))[0] > 0.5 * want.max()


@pytest.mark.parametrize("name", ["hcp-Mg", "alpha-Ti"])
def test_the_old_expansion_would_break_hcp(tmp_path, name):
    """Guard for the test above: diffsims' own sanitise_phase really does give
    the extinct (2 -1 1) a structure factor here. If a future diffsims fixes
    that, this test skips and says so, and the module can be retired."""
    phase, _ = _from_cif(tmp_path, name)
    r = R(phase.deepcopy(), hkl=np.array([[2.0, -1.0, 1.0]]))
    r.sanitise_phase()
    r.calculate_structure_factor()
    if np.abs(r.structure_factor).ravel()[0] < 1e-6:
        pytest.skip("this diffsims expands the atoms correctly: phase_cell is no longer needed")
    assert np.abs(r.structure_factor).ravel()[0] > 1.0


# --------------------------------------------------------------------------
# extinction rules, one per crystal system
# --------------------------------------------------------------------------

def _rows(lat, d_min=1.1):
    return _hkl_grid(lat, d_min=d_min)


def _check_rule(phase, lat, forbidden, allowed_min=3):
    hkl = _rows(lat)
    F = _orienta_F(phase, hkl)
    fm = F.max()
    odd = np.array([bool(forbidden(*map(int, row))) for row in hkl])
    assert odd.any() and (~odd).sum() >= allowed_min
    assert F[odd].max() < 1e-8 * fm, hkl[odd][np.argmax(F[odd])]
    assert (F[~odd] > 1e-3 * fm).sum() >= allowed_min


def test_c2m_monoclinic_C_centring(tmp_path):
    phase, _ = _from_cif(tmp_path, "monoclinic-C2/m")
    _check_rule(phase, CRYSTALS["monoclinic-C2/m"][1], lambda h, k, l: (h + k) % 2)


def test_cmcm_orthorhombic_C_centring_and_atom_count(tmp_path):
    phase, _ = _from_cif(tmp_path, "orthorhombic-Cmcm")
    assert len(pc.complete_unit_cell(phase).structure) == 16
    _check_rule(phase, CRYSTALS["orthorhombic-Cmcm"][1], lambda h, k, l: (h + k) % 2)


def test_trigonal_R_centring_in_hexagonal_axes(tmp_path):
    phase, _ = _from_cif(tmp_path, "trigonal-R-3m")
    _check_rule(phase, CRYSTALS["trigonal-R-3m"][1], lambda h, k, l: (-h + k + l) % 3)


CORUNDUM = """\
data_corundum
_symmetry_space_group_name_H-M   'R -3 c'
_symmetry_Int_Tables_number      167
_cell_length_a 4.7589
_cell_length_b 4.7589
_cell_length_c 12.9910
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 120
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Al1 Al 0.00000000 0.00000000 0.35216000 1.0
  O1  O  0.30640000 0.00000000 0.25000000 1.0
"""


def test_R_minus3c_corundum_rule_and_count(tmp_path):
    """R-3c (167) in hexagonal axes: reflections need -h+k+l = 3n, and h-h0l needs
    l = 2n. Twelve Al and eighteen O in the hexagonal cell."""
    path = tmp_path / "corundum.cif"
    path.write_text(CORUNDUM, encoding="utf-8")
    phase = eu.hough_phase_from_cif(str(path))
    cell = pc.complete_unit_cell(phase)
    assert sorted(str(a.element) for a in cell.structure) == ["Al"] * 12 + ["O"] * 18
    lat = (4.7589, 4.7589, 12.991, 90, 90, 120)
    hkl = _hkl_grid(lat, d_min=0.9, span=7)
    F = _orienta_F(phase, hkl)
    forb = np.array([((-h + k + l) % 3 != 0) or (k == -h and l % 2 != 0)
                     for h, k, l in hkl.astype(int)])
    assert forb.any() and (~forb).any()
    assert F[forb].max() < 1e-7 * F.max()
    assert (F[~forb] > 1e-4 * F.max()).sum() > 10


def test_cubic_controls(tmp_path):
    fcc, _ = _from_cif(tmp_path, "cubic-Fm-3m")
    _check_rule(fcc, CRYSTALS["cubic-Fm-3m"][1], lambda h, k, l: len({h % 2, k % 2, l % 2}) > 1)
    bcc, _ = _from_cif(tmp_path, "cubic-Im-3m")
    _check_rule(bcc, CRYSTALS["cubic-Im-3m"][1], lambda h, k, l: (h + k + l) % 2)
    # one orbit of four, and of two, as the Tables give them
    assert len(pc.complete_unit_cell(fcc).structure) == 4
    assert len(pc.complete_unit_cell(bcc).structure) == 2


def test_triclinic_keeps_every_atom_once(tmp_path):
    phase, _ = _from_cif(tmp_path, "triclinic-P-1")
    assert len(phase.structure) == 4
    cell = pc.complete_unit_cell(phase)
    assert len(cell.structure) == 4
    assert sorted(str(a.element) for a in cell.structure) == ["Al", "Al", "Si", "Si"]
    # Friedel pair: P-1 is centrosymmetric, so |F(hkl)| = |F(-h-k-l)|
    hkl = np.array([[1, 2, 3], [-1, -2, -3], [2, -1, 1], [-2, 1, -1]], float)
    F = _orienta_F(phase, hkl)
    assert np.allclose(F[0], F[1]) and np.allclose(F[2], F[3])


# --------------------------------------------------------------------------
# an independent reader: pymatgen on the same CIF text
# --------------------------------------------------------------------------

def _pymatgen_reference(path):
    """Structure in the CIF's own frame with pymatgen's site positions."""
    pytest.importorskip("pymatgen")
    import warnings

    from pymatgen.core import Structure as PS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = PS.from_file(str(path))
    st = Structure(lattice=Lattice(*s.lattice.abc, *s.lattice.angles))
    for site in s:
        sp = max(site.species.items(), key=lambda kv: kv[1])[0]
        st.append(Atom(sp.symbol, xyz=site.frac_coords,
                       occupancy=float(sum(site.species.values()))))
    return st


@pytest.mark.parametrize("name", list(CRYSTALS))
def test_pymatgen_agrees_on_atoms_and_structure_factors(tmp_path, name):
    phase, path = _from_cif(tmp_path, name)
    ref = _pymatgen_reference(path)
    cell = pc.complete_unit_cell(phase)
    assert len(cell.structure) == len(ref)
    hkl = _hkl_grid(CRYSTALS[name][1])
    Fref = np.abs(_K(ref, hkl, ref.lattice.rnorm(hkl), scattering_params="xtables"))
    got = _orienta_F(phase, hkl)
    assert np.abs(got - Fref).max() <= 1e-9 * Fref.max()


# --------------------------------------------------------------------------
# what the helper promises besides the numbers
# --------------------------------------------------------------------------

def test_the_input_phase_is_not_modified(tmp_path):
    phase, _ = _from_cif(tmp_path, "hcp-Mg")
    before = [(str(a.element), tuple(a.xyz)) for a in phase.structure]
    pc.complete_unit_cell(phase)
    pc.complete_unit_cell(_asymmetric_phase("hcp-Mg"))
    assert before == [(str(a.element), tuple(a.xyz)) for a in phase.structure]


def test_the_phase_stays_usable_for_orixs_own_expansion():
    """orix remembers the lattice it re-based from, for ``expand_asymmetric_unit``;
    writing the structure back through the setter would overwrite it."""
    asym = _asymmetric_phase("alpha-Ti")
    kept = np.array(asym._diffpy_lattice)
    out = pc.complete_unit_cell(asym)
    assert np.allclose(np.array(out._diffpy_lattice), kept)
    assert len(out.expand_asymmetric_unit().structure) == 2


def test_four_digit_coordinates_do_not_split_an_orbit(tmp_path):
    """0.3333 / 0.6667 written with four digits: the operators map 0.3333 to
    0.6666 and 0.6667 to 0.3334, 1e-4 off the other atom. Two atoms, not six."""
    sg, lat, _atoms, _ = CRYSTALS["hcp-Mg"]
    st = Structure(atoms=[Atom("Mg", xyz=[0.3333, 0.6667, 0.25])], lattice=Lattice(*lat))
    cell = pc.complete_unit_cell(Phase(name="Mg", space_group=sg, structure=st))
    assert len(cell.structure) == 2


def test_different_elements_on_one_site_both_stay_and_occupancy_is_kept():
    sg, lat, _atoms, _ = CRYSTALS["cubic-Fm-3m"]
    st = Structure(lattice=Lattice(*lat), atoms=[
        Atom("Fe", xyz=[0, 0, 0], occupancy=0.6), Atom("Mn", xyz=[0, 0, 0], occupancy=0.4)])
    cell = pc.complete_unit_cell(Phase(name="x", space_group=sg, structure=st))
    assert sorted((str(a.element), round(a.occupancy, 3)) for a in cell.structure) == \
        [("Fe", 0.6)] * 4 + [("Mn", 0.4)] * 4


def test_without_a_space_group_it_refuses_and_without_atoms_it_invents_none():
    sg, lat, _atoms, _ = CRYSTALS["cubic-Fm-3m"]
    with pytest.raises(ValueError, match="space group"):
        pc.complete_unit_cell(Phase(name="x", structure=Structure(
            atoms=[Atom("Al", xyz=[0, 0, 0])], lattice=Lattice(*lat))))
    empty = Phase(name="x", space_group=sg, structure=Structure(lattice=Lattice(*lat)))
    assert len(pc.complete_unit_cell(empty).structure) == 0


def test_sanitise_reflectors_replaces_the_phase_and_leaves_the_original_alone(tmp_path):
    phase, _ = _from_cif(tmp_path, "hcp-Mg")
    rlv = R(phase, hkl=np.array([[2.0, -1.0, 1.0], [0.0, 0.0, 2.0]]))
    n0 = len(phase.structure)
    pc.sanitise_reflectors(rlv)
    assert rlv.phase is not phase and len(phase.structure) == n0
    rlv.calculate_structure_factor()
    F = np.abs(rlv.structure_factor).ravel()
    assert F[0] < 1e-9 and F[1] > 1.0


# --------------------------------------------------------------------------
# the duplicate tolerance is a distance
# --------------------------------------------------------------------------

def test_three_digit_coordinates_do_not_split_an_orbit(tmp_path):
    """hcp Mg written as (0.333, 0.667, 0.25): diffpy's CIF reader keeps six
    near-copies of the two atoms, 1e-3 of the cell edge apart. The cell has two."""
    path = tmp_path / "mg3.cif"
    path.write_text(_cif_text("hcp-Mg").replace("0.33333333", "0.333")
                    .replace("0.66666667", "0.667"), encoding="utf-8")
    phase = eu.hough_phase_from_cif(str(path))
    assert len(phase.structure) > 2          # the premise: the reader did split it
    cell = pc.complete_unit_cell(phase)
    assert len(cell.structure) == 2
    # and the structure factors are those of hcp up to the rounding of the file
    hkl = _hkl_grid(CRYSTALS["hcp-Mg"][1], d_min=1.5)
    want = _hand_F("hcp-Mg", hkl)
    assert np.abs(_orienta_F(phase, hkl) - want).max() < 2e-2 * want.max()


def test_sites_half_an_angstrom_apart_stay_apart_and_rounding_is_merged():
    """Two aluminium atoms 0.49 A apart (the closest distinct pair in the crystal
    library) are two atoms; 0.015 A apart (rounding) they are one."""
    def cell_of(dx):
        st = Structure(lattice=Lattice(10, 10, 10, 90, 90, 90), atoms=[
            Atom("Al", xyz=[0, 0, 0]), Atom("Al", xyz=[dx, 0, 0])])
        return pc.complete_unit_cell(Phase(name="x", space_group=1, structure=st))
    assert len(cell_of(0.049).structure) == 2
    assert len(cell_of(0.0015).structure) == 1
    # across the cell boundary too: 0.9985 is 0.015 A from 0
    assert len(cell_of(0.9985).structure) == 1


def test_library_closest_pair_of_sites_is_not_merged():
    from tests.data_deps import CIF_LIBRARY, need
    path = need(CIF_LIBRARY / "α-(AlFeSi) (Fe23Al81Si15).cif")
    phase = eu.hough_phase_from_cif(str(path))
    cell = pc.complete_unit_cell(phase)
    assert len(cell.structure) == len(phase.structure) == 246
    f = pc.true_fractional(cell)
    base = np.asarray(cell.structure.lattice.stdbase, float)
    d = f[:, None, :] - f[None, :, :]
    d -= np.rint(d)
    dist = np.linalg.norm(d @ base, axis=2) + np.eye(len(f)) * 1e9
    assert 0.3 < dist.min() < 0.7           # the pair that a 0.03-A tolerance would merge


# --------------------------------------------------------------------------
# elements and occupancies
# --------------------------------------------------------------------------

def test_an_atomic_number_as_element_becomes_its_symbol_and_still_scatters():
    """diffpy accepts ``Atom(13)``; diffsims' ``sanitise_phase`` turned that into
    "Al". Left as 13 the scattering factor is not found and |F| is silently zero."""
    sg, lat, _atoms, _ = CRYSTALS["cubic-Fm-3m"]
    num = Phase(name="x", space_group=sg, structure=Structure(
        atoms=[Atom(13, xyz=[0, 0, 0])], lattice=Lattice(*lat)))
    sym = Phase(name="x", space_group=sg, structure=Structure(
        atoms=[Atom("Al", xyz=[0, 0, 0])], lattice=Lattice(*lat)))
    cell = pc.complete_unit_cell(num)
    assert [a.element for a in cell.structure] == ["Al"] * 4
    hkl = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]], float)
    F = _orienta_F(num, hkl)
    assert F.min() > 1.0
    assert np.allclose(F, _orienta_F(sym, hkl))
    # 13 and "Al" on one site are one atom, not two
    both = Phase(name="x", space_group=sg, structure=Structure(
        atoms=[Atom(13, xyz=[0, 0, 0]), Atom("Al", xyz=[0, 0, 0])], lattice=Lattice(*lat)))
    assert len(pc.complete_unit_cell(both).structure) == 4


def test_two_entries_of_one_element_on_one_site_keep_their_occupancies():
    """A split site listed twice (Fe 0.6 and Fe 0.4 at one position): both stay, and
    |F| is that of one fully occupied Fe site. Equal entries are duplicates."""
    sg, lat, _atoms, _ = CRYSTALS["cubic-Fm-3m"]

    def phase(*occs):
        return Phase(name="x", space_group=sg, structure=Structure(
            lattice=Lattice(*lat),
            atoms=[Atom("Fe", xyz=[0, 0, 0], occupancy=o) for o in occs]))

    split = pc.complete_unit_cell(phase(0.6, 0.4))
    assert sorted(round(a.occupancy, 3) for a in split.structure) == [0.4] * 4 + [0.6] * 4
    hkl = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0]], float)
    assert np.allclose(_orienta_F(phase(0.6, 0.4), hkl), _orienta_F(phase(1.0), hkl))
    assert len(pc.complete_unit_cell(phase(0.5, 0.5)).structure) == 4   # equal: a duplicate


def test_the_tolerance_is_a_distance_in_angstrom_not_a_fraction():
    """A 40 A cell, two Al at x = 0 and x = 0.0075: 0.3 A apart, two atoms. A tolerance of
    0.02 in fractional coordinates (0.8 A here) would merge them; 0.02 A merges what a
    CIF's rounding makes."""
    def n_atoms(dx):
        st = Structure(lattice=Lattice(40, 40, 40, 90, 90, 90), atoms=[
            Atom("Al", xyz=[0, 0, 0]), Atom("Al", xyz=[dx, 0, 0])])
        return len(pc.complete_unit_cell(Phase(name="x", space_group=1, structure=st)).structure)

    assert n_atoms(0.0075) == 2            # 0.3 A
    assert n_atoms(0.0003) == 1            # 0.012 A


def test_equal_rows_at_one_position_are_merged_as_the_docstring_says():
    """Same element, same occupancy, same coordinates: one atom (a split site written as
    two identical 0.5 rows is halved, as with diffsims' sanitise_phase)."""
    st = Structure(lattice=Lattice(4, 4, 4, 90, 90, 90), atoms=[
        Atom("Fe", xyz=[0, 0, 0], occupancy=0.5), Atom("Fe", xyz=[0, 0, 0], occupancy=0.5)])
    cell = pc.complete_unit_cell(Phase(name="x", space_group=1, structure=st))
    assert [a.occupancy for a in cell.structure] == [0.5]
    assert "merged" in pc.complete_unit_cell.__doc__
