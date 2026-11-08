"""Reflector families of the seven trigonal space groups (149, 151, 153, 157, 159, 162, 163).

orix derives the point group of a plain ``Phase`` from the space group number and
has no name for the setting with the two-fold axes (or mirror normals) along Y, so for
these groups it hands out the other setting. The reflector code folds reflections into
families with ``phase.point_group``; with the wrong setting a family holds reflections
that are not equivalent, so the multiplicity of a family, its structure factor (taken
from one row of it) and the rows of a selection a user makes are wrong. Cr2N (P-31m)
showed it: {10-11} has 12 members and {2-1-11} has 6, a plain phase said 6 and 12.

A phase from the CIF reader (``FramePhase``) carries the true group. The reflector code
now gives every phase of these seven groups the true group, whatever class it arrives
in, and leaves all other phases alone.

Every expectation is written out here from the symmetry operations of the space group
(diffpy) and a sum over the atoms of the whole cell; nothing is taken from the code under
test. The family of a reflection is its orbit under the rotation parts of the operations
(the point group, which is what orix and diffsims fold by); for a group with inversion
this is the Laue class including the Friedel partner.
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
from diffpy.structure.spacegroups import GetSpaceGroup  # noqa: E402
from diffsims.crystallography import ReciprocalLatticeVector as R  # noqa: E402
from diffsims.utils.sim_utils import _get_kinematical_structure_factor as _K  # noqa: E402
from orix.crystal_map import Phase, PhaseList  # noqa: E402

import crystal_symmetry as cs  # noqa: E402
import ebsd_utils as eu  # noqa: E402
import hough_reflectors as hr  # noqa: E402
from crystal_symmetry import FramePhase  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CR2N_CIF = ROOT / "Database" / "CIF_Library" / "Cr2N_Zenodo-20376534.cif"

SEVEN = (149, 151, 153, 157, 159, 162, 163)
HEX = (4.1, 4.1, 6.3, 90, 90, 120)
HEX_ATOMS = [("Al", 1.0, (.11, .23, .31)), ("Fe", 1.0, (.42, .07, .65))]

CR2N_LATTICE = (4.8, 4.8, 4.472, 90, 90, 120)
#: asymmetric unit of the CIF: Cr on 6k, N on 2d with two occupancies, N on 1a
CR2N_ATOMS = [("Cr", 1.0, (.346, 0, .244)), ("N", .74, (1 / 3, 2 / 3, .5)),
              ("N", .12, (1 / 3, 2 / 3, 0)), ("N", 1.0, (0, 0, 0))]


# --------------------------------------------------------------------------
# The hand calculation
# --------------------------------------------------------------------------

def _ops(sg):
    ops = list(GetSpaceGroup(sg).iter_symops())
    return [np.rint(o.R).astype(int) for o in ops], [np.asarray(o.t, float) for o in ops]


def orbit(sg, hkl):
    """The family of a reflection: {h R} over the rotation parts of the space group."""
    rots, _ = _ops(sg)
    v = np.array([int(x) for x in hkl])
    return frozenset(tuple(int(x) for x in v @ r) for r in rots)


def whole_cell(sg, sites, tol=1e-4):
    rots, ts = _ops(sg)
    cell = []
    for el, occ, xyz in sites:
        pts = []
        for r, t in zip(rots, ts):
            p = (r @ np.asarray(xyz, float) + t) % 1.0
            if not any(np.abs(((p - q + .5) % 1.0) - .5).max() < tol for q in pts):
                pts.append(p)
        cell += [(el, occ, p) for p in pts]
    return cell


def _f_lone(element, lattice, hkl):
    st = Structure(atoms=[Atom(element, xyz=[0, 0, 0])], lattice=Lattice(*lattice))
    return np.abs(_K(st, hkl, st.lattice.rnorm(hkl), scattering_params="xtables"))


def hand_F(lattice, cell, hkl):
    hkl = np.asarray(hkl, float).reshape(-1, 3)
    F = np.zeros(len(hkl), complex)
    for el, occ, pos in cell:
        F += occ * _f_lone(el, lattice, hkl) * np.exp(2j * np.pi * (hkl @ pos))
    return np.abs(F)


def grid(lattice, d_min, span=None):
    lat = Lattice(*lattice)
    span = span or int(max(lattice[:3]) / d_min) + 2
    g = np.array([h for h in itertools.product(range(-span, span + 1), repeat=3)
                  if h != (0, 0, 0)], float)
    return g[1.0 / lat.rnorm(g) >= d_min - 1e-9]


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------

def general_phase(sg, kind, lattice=HEX, atoms=HEX_ATOMS):
    st = Structure(atoms=[Atom(el, xyz=xyz) for el, _occ, xyz in atoms],
                   lattice=Lattice(*lattice))
    cls = {"plain": Phase, "frame": FramePhase}[kind]
    return cls(name=f"sg{sg}", space_group=sg, structure=st)


def cr2n_phase(kind):
    if not CR2N_CIF.is_file():
        pytest.skip("Database/CIF_Library/Cr2N_Zenodo-20376534.cif is not available")
    if kind == "frame":
        return hr.phase_from_cif(CR2N_CIF)
    phase = Phase.from_cif(eu.sanitize_cif(str(CR2N_CIF)))
    phase.name = CR2N_CIF.stem
    return phase


@pytest.fixture(autouse=True)
def fresh_caches():
    """A table must come from a computation of its own, not from the cache entry of an
    earlier test (plain and frame-aware phases of one crystal share their entries)."""
    def clear():
        with hr._CACHE_LOCK:
            hr._CACHE.clear()
        with eu._FILTERED_CACHE_LOCK:
            eu._FILTERED_CACHE.clear()
            eu._FILTERED_CACHE_ORDER.clear()
    clear()
    yield
    clear()


def table_of(t):
    return {"reps": [tuple(r) for r in t.reps], "d": list(t.d), "f": list(t.f),
            "mult": list(t.mult), "fmax": t.fmax, "rows": dict(t.row_to_family)}


# --------------------------------------------------------------------------
# 1. The families are the orbits of the space group
# --------------------------------------------------------------------------

def _check_table_against_hand(sg, phase, lattice, sites, min_d=1.0, check_f=True):
    cell = whole_cell(sg, sites)
    tab = hr.stage_table(phase) if min_d == hr.DEFAULT_RULE["min_d"] else hr.candidate_table(phase, min_d)
    assert len(tab.reps) > 5
    members = {}
    for row, i in tab.row_to_family.items():
        members.setdefault(i, set()).add(row)
    g = grid(lattice, min_d)
    F = hand_F(lattice, cell, g) if check_f else None
    fmax_hand = F.max() if check_f else None
    for i, rep in enumerate(tab.reps):
        want = orbit(sg, rep)
        assert members[i] == want, (rep, sorted(members[i] ^ want)[:4])
        assert tab.mult[i] == len(want), rep
        if check_f:
            assert tab.f[i] == pytest.approx(hand_F(lattice, cell, [rep])[0],
                                             abs=1e-3 * fmax_hand), rep
    # no reflection is lost: every strong row of the hand list is in a family
    if check_f and min_d == hr.DEFAULT_RULE["min_d"]:
        thr = hr.DEFAULT_RULE["f_threshold"] * F.max()
        for h, f in zip(g, F):
            key = tuple(int(x) for x in h)
            if f > thr * 1.01:
                assert key in tab.row_to_family, (key, f)
            elif f < thr * 0.99:
                assert key not in tab.row_to_family, (key, f)
    return tab


@pytest.mark.parametrize("kind", ["frame", "plain"])
@pytest.mark.parametrize("sg", SEVEN)
def test_stage_table_families_are_the_orbits_of_the_space_group(sg, kind):
    _check_table_against_hand(sg, general_phase(sg, kind), HEX, HEX_ATOMS)


@pytest.mark.parametrize("kind", ["frame", "plain"])
def test_cr2n_families_and_the_two_named_ones(kind):
    """Cr2N from the CIF: {10-11} has 12 members and |F| 2.62, {2-1-11} has 6 and |F|
    14.82; the plain phase used to report 6 and 12."""
    phase = cr2n_phase(kind)
    _check_table_against_hand(162, phase, CR2N_LATTICE, CR2N_ATOMS)
    _check_table_against_hand(162, phase, CR2N_LATTICE, CR2N_ATOMS, min_d=0.7, check_f=False)
    families = {f["label"]: f for f in hr.describe(phase, None)["families"]}
    assert families["{10-11}"]["mult"] == 12
    assert families["{10-11}"]["f"] == pytest.approx(2.6214, abs=5e-4)
    assert families["{2-1-11}"]["mult"] == 6
    assert families["{2-1-11}"]["f"] == pytest.approx(14.8168, abs=5e-4)
    # and the same by the hand orbits, so the numbers above are not just remembered
    assert len(orbit(162, (1, 0, -1))) == 12 and len(orbit(162, (2, -1, 1))) == 6
    cell = whole_cell(162, CR2N_ATOMS)
    assert hand_F(CR2N_LATTICE, cell, [(1, 0, -1)])[0] == pytest.approx(2.6214, abs=5e-4)
    assert hand_F(CR2N_LATTICE, cell, [(2, -1, 1)])[0] == pytest.approx(14.8168, abs=5e-4)
    for label, fam in families.items():
        assert fam["mult"] == len(orbit(162, fam["hkl"])), label


# --------------------------------------------------------------------------
# 2. A plain phase and a frame-aware phase of one crystal give one answer
# --------------------------------------------------------------------------

def _everything(phase):
    tab = table_of(hr.stage_table(phase))
    cand = table_of(hr.candidate_table(phase, 0.8))
    hkl, rows = hr._rule_rows(phase, hr.DEFAULT_RULE)
    ref = eu._reflectors_for_phase(phase)
    desc = hr.describe(phase, None)
    ext = hr.describe(phase, None, extended=True)
    fam = hr._extra_family(phase, (1, 0, -1))
    return {"stage": tab, "cand": cand, "rule_hkl": hkl.tolist(), "rule_rows": rows.tolist(),
            "ref_hkl": np.asarray(ref.hkl).tolist(),
            "ref_f": np.abs(np.asarray(ref.structure_factor)).ravel().tolist(),
            "describe": desc, "extended": ext,
            "family": (fam["rep"], fam["f"], fam["d"], fam["mult"], fam["rows"].tolist())}


def _fresh_both(make):
    """Everything for the plain phase, then everything for the frame-aware one, each
    from empty caches."""
    out = []
    for kind in ("plain", "frame"):
        with hr._CACHE_LOCK:
            hr._CACHE.clear()
        with eu._FILTERED_CACHE_LOCK:
            eu._FILTERED_CACHE.clear()
            eu._FILTERED_CACHE_ORDER.clear()
        out.append(_everything(make(kind)))
    return out


@pytest.mark.parametrize("sg", SEVEN)
def test_plain_phase_gives_the_table_of_the_frame_phase(sg):
    plain, frame = _fresh_both(lambda kind: general_phase(sg, kind))
    assert plain == frame


def test_cr2n_plain_phase_gives_the_table_of_the_frame_phase():
    plain, frame = _fresh_both(cr2n_phase)
    assert plain == frame


@pytest.mark.parametrize("sg", SEVEN)
def test_plain_and_frame_phase_share_cache_entries(sg):
    plain, frame = general_phase(sg, "plain"), general_phase(sg, "frame")
    assert hr._structure_hash(plain) == hr._structure_hash(frame)
    assert hr.stage_table(plain) is hr.stage_table(frame)


@pytest.mark.parametrize("sg", [164, 166, 225, 12])
def test_other_space_groups_keep_their_phase_object_and_cache_key(sg):
    """Nothing is converted outside the seven: the very same object comes back."""
    if sg == 225:
        phase = general_phase(sg, "plain", (4.05, 4.05, 4.05, 90, 90, 90), [("Al", 1.0, (0, 0, 0))])
    elif sg == 12:
        phase = general_phase(sg, "plain", (5.0, 3.0, 4.0, 90, 105, 90),
                              [("Al", 1.0, (.11, 0, .31)), ("Fe", 1.0, (.42, 0, .65))])
    else:
        phase = general_phase(sg, "plain")
    assert cs.trigonal_frame_phase(phase) is phase
    assert not cs.has_trigonal_y_setting(phase)
    pl = PhaseList(phases=[phase])
    assert cs.trigonal_frame_phase_list(pl) is pl
    assert eu._standard_setting_phase_list(pl) is pl


@pytest.mark.parametrize("sg", SEVEN)
def test_the_seven_become_frame_phases_and_keep_name_and_cell(sg):
    plain = general_phase(sg, "plain")
    frame = cs.trigonal_frame_phase(plain)
    assert isinstance(frame, FramePhase) and frame is not plain
    assert cs.trigonal_frame_phase(frame) is frame
    assert frame.name == plain.name and frame.space_group.number == sg
    assert np.allclose(frame.structure.lattice.abcABG(), plain.structure.lattice.abcABG())
    assert len(frame.structure) == len(plain.structure)
    assert cs.point_group_key(frame.point_group) == cs._TRIGONAL_Y_SETTING[sg]
    assert cs.trigonal_frame_phase(None) is None


# --------------------------------------------------------------------------
# 3. Controls outside the seven are written out and do not move
# --------------------------------------------------------------------------

def _order_by_spacing(tab):
    idx = sorted(range(len(tab.reps)), key=lambda i: (-round(tab.d[i], 4), -tab.f[i]))
    return [(tuple(tab.reps[i]), tab.mult[i]) for i in idx]


def test_control_fcc_aluminium_225():
    phase = general_phase(225, "plain", (4.05, 4.05, 4.05, 90, 90, 90), [("Al", 1.0, (0, 0, 0))])
    got = {rep: mult for rep, mult in _order_by_spacing(hr.stage_table(phase))}
    # {200} {400} {220} {111} {311} {222} of an fcc crystal with a = 4.05 A, d >= 1 A
    assert got == {(2, 0, 0): 6, (4, 0, 0): 6, (2, 2, 0): 12, (1, 1, 1): 8,
                   (3, 1, 1): 24, (2, 2, 2): 8}
    tab = hr.stage_table(phase)
    assert [tuple(r) for r in tab.reps] == [(2, 0, 0), (4, 0, 0), (2, 2, 0), (1, 1, 1),
                                            (3, 1, 1), (2, 2, 2)]
    assert tab.mult == [6, 6, 12, 8, 24, 8]


def test_control_164_first_families_and_orbits():
    phase = general_phase(164, "plain")
    tab = hr.stage_table(phase)
    assert [tuple(r) for r in tab.reps][:8] == [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
                                                (0, 0, 5), (0, 0, 6), (1, 0, 0), (1, 0, -1)]
    assert tab.mult[:8] == [2, 2, 2, 2, 2, 2, 6, 6]
    assert len(tab.reps) == 24
    _check_table_against_hand(164, phase, HEX, HEX_ATOMS)


def test_control_166_first_families_and_orbits():
    phase = general_phase(166, "plain")
    tab = hr.stage_table(phase)
    assert [tuple(r) for r in tab.reps][:8] == [(0, 0, 3), (0, 0, 6), (1, 0, -2), (1, 0, -5),
                                                (2, 0, -1), (1, 0, 1), (2, -1, 0), (2, -1, 3)]
    assert tab.mult[:8] == [2, 2, 6, 6, 6, 6, 6, 12]
    assert len(tab.reps) == 10
    _check_table_against_hand(166, phase, HEX, HEX_ATOMS, check_f=False)


MONOCLINIC = (5.0, 3.0, 4.0, 90, 105, 90)
MONO_ATOMS = [("Al", 1.0, (.11, 0, .31)), ("Fe", 1.0, (.42, 0, .65))]


def test_control_12_frame_aware_phase_first_families_and_orbits():
    """C2/m as the app builds it (b-unique, two-fold axis along Y). The strongest
    families by spacing, and every family equal to its orbit."""
    phase = general_phase(12, "frame", MONOCLINIC, MONO_ATOMS)
    assert cs.trigonal_frame_phase(phase) is phase
    first = _order_by_spacing(hr.stage_table(phase))[:6]
    assert first == [((0, 0, 1), 2), ((1, 1, 0), 4), ((2, 0, 0), 2), ((2, 0, -1), 2),
                     ((1, 1, -1), 4), ((1, 1, 1), 4)]
    _check_table_against_hand(12, phase, MONOCLINIC, MONO_ATOMS)


# --------------------------------------------------------------------------
# 4. A selection of one's own: the rows of the overlay
# --------------------------------------------------------------------------

def _old_family(phase_plain, hkl):
    """The family the plain phase used to give: orix' symmetrise by the point group it
    derives from the space group number (the other setting)."""
    r = R(phase_plain.deepcopy(), hkl=np.array([hkl], float))
    return {tuple(int(x) for x in row)
            for row in np.rint(np.asarray(r.to_miller().symmetrise(unique=True).hkl)).reshape(-1, 3)}


@pytest.mark.parametrize("kind", ["plain", "frame"])
@pytest.mark.parametrize("sg", (149, 157, 162))
def test_selection_rows_are_the_true_orbit(sg, kind):
    plain = general_phase(sg, "plain")
    phase = plain if kind == "plain" else general_phase(sg, "frame")
    # a reflection whose family differs between the two settings
    probe = next(h for h in [(2, 1, 0), (1, 0, 1), (2, 1, 1), (3, 1, 0), (2, 1, 2), (1, 0, 2),
                             (3, 2, 1), (4, 1, 1)]
                 if _old_family(plain, h) != set(orbit(sg, h)))
    want = set(orbit(sg, probe))
    assert _old_family(plain, probe) != want
    info = hr._extra_family(phase, probe)
    assert {tuple(int(x) for x in r) for r in info["rows"]} == want
    assert info["mult"] == len(want)
    spec = {"mode": "custom", "families": [list(probe)]}
    hkl, rows = hr.spec_rows(phase, spec)
    got = {tuple(int(x) for x in r) for r in rows}
    assert got and got <= want
    # the overlay draws the rlv of the selection; it carries the true group
    rlv = hr.rlv_for_spec(phase, spec)
    drawn = {tuple(int(x) for x in r) for r in np.rint(np.asarray(rlv.hkl)).reshape(-1, 3)}
    assert drawn == got
    assert cs.point_group_key(rlv.phase.point_group) == cs._TRIGONAL_Y_SETTING[sg]


def test_selection_that_the_default_list_does_not_hold_gets_the_whole_true_orbit():
    """A hand-typed family finer than the default cut: the whole orbit, in both classes."""
    probe = (4, 1, 1)
    for kind in ("plain", "frame"):
        phase = general_phase(162, kind)
        hkl, rows = hr.spec_rows(phase, {"mode": "custom", "families": [list(probe)]})
        got = {tuple(int(x) for x in r) for r in rows}
        info = hr.validate_family(phase, list(probe))
        assert info["mult"] == len(orbit(162, probe))
        assert got <= set(orbit(162, probe)) and len(got) >= 1


# --------------------------------------------------------------------------
# 5. The ways into PyEBSDIndex and the geometrical overlay see the true group
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sg", (149, 162))
def test_pole_families_handed_to_the_indexer_do_not_depend_on_the_phase_class(sg):
    kp = pytest.importorskip("kikuchipy")
    from kikuchipy.indexing._hough_indexing import _get_pyebsdindex_phaselist

    out = {}
    for kind in ("plain", "frame"):
        phase = general_phase(sg, kind)
        rows = eu._reflectors_for_phase(phase).hkl.tolist()
        fixed = eu._standard_setting_phase_list(PhaseList(phases=[phase]))
        bi = _get_pyebsdindex_phaselist(fixed, [rows])[0]
        out[kind] = np.asarray(bi.polefamilies).tolist()
    assert out["plain"] == out["frame"]
    # and these are the families of the space group: no two of the poles are in one orbit
    poles = [tuple(int(round(x)) for x in p) for p in out["plain"]]
    for a, b in itertools.combinations(poles, 2):
        assert b not in orbit(sg, a), (a, b)


@pytest.mark.parametrize("sg", (149, 157, 162))
def test_geometrical_overlay_rows_of_a_plain_phase_are_closed_under_the_true_group(
        monkeypatch, sg):
    """``_simulate_pattern_for_orientation`` symmetrises its reflectors itself."""
    import kikuchipy as kp
    import kikuchipy.simulations as ksim
    from orix.quaternion import Rotation

    import indexing_controller as ic

    seen = {}

    class Recorder:
        def __init__(self, ref, *a, **k):
            seen["ref"] = ref
            raise RuntimeError("stop here: the reflectors are all this test wants")

    monkeypatch.setattr(ksim, "KikuchiPatternSimulator", Recorder)
    detector = kp.detectors.EBSDDetector(shape=(60, 60), pc=(0.5, 0.5, 0.5), sample_tilt=70.0)
    rows = {}
    for kind in ("plain", "frame"):
        seen.clear()
        phase = general_phase(sg, kind)
        assert ic._simulate_pattern_for_orientation(
            detector, PhaseList(phases=[phase]), Rotation.identity(), 0) is None
        rows[kind] = [tuple(int(round(x)) for x in r) for r in np.asarray(seen["ref"].hkl)]
    # the same rows in the same order: the order is that of unique(), which folds by the group
    assert rows["plain"] == rows["frame"]
    for r in rows["plain"]:
        assert set(orbit(sg, r)) <= set(rows["plain"]), r
