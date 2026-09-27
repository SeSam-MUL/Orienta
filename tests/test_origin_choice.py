"""The origin choice of the 24 two-origin space groups, checked against spglib.

Background: a Mac tester's silicon simulation ran at 4.658 g/cm3 — exactly twice
the real 2.329. `Si.cif` puts silicon on Wyckoff 8a at (1/8,1/8,1/8), correct in
Fd-3m origin choice 2; diffpy hands out origin choice 1 operators, where those
same numbers are the 16-fold 16c/16d site. The orbit came out 16 instead of 8,
doubling the density the Monte Carlo uses and the atom sum that
`compute_Ug_table` builds the crystal potential from.

These tests pin the three things that have to stay true:
  1. diffpy's operators really are a valid setting for all 230 groups (it is not
     broken, it is specific) — and specifically origin choice 1 for all 24;
  2. the shift table in `origin_choice` is the one spglib implies, re-derived
     here rather than trusted;
  3. silicon comes out at its literature density, and Al/Ni do not move.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

spglib = pytest.importorskip("spglib")
from diffpy.structure.spacegroups import GetSpaceGroup  # noqa: E402

from backend.forward_sim.crystal.origin_choice import (  # noqa: E402
    ORIGIN_SHIFT_TO_CHOICE_1,
    has_two_origin_choices,
    to_choice_1,
)

_AVOGADRO = 6.02214076e23


def _opset(rotations, translations):
    return {
        (tuple(np.asarray(R, int).ravel()), tuple(np.round(np.asarray(t, float) % 1.0, 4)))
        for R, t in zip(rotations, translations)
    }


def _halls_by_number():
    out: dict[int, list[tuple[int, str]]] = {}
    for hall in range(1, 531):
        t = spglib.get_spacegroup_type(hall)
        out.setdefault(t["number"], []).append((hall, t.get("choice", "")))
    return out


@pytest.fixture(scope="module")
def halls():
    return _halls_by_number()


def test_diffpy_operators_are_a_valid_setting_for_all_230(halls):
    """diffpy is not wrong — every group matches one of spglib's settings.

    Worth stating, because the fix is about WHICH setting, not about replacing
    the operator source.
    """
    unmatched = []
    for n in range(1, 231):
        dsg = GetSpaceGroup(n)
        d_ops = _opset([o.R for o in dsg.symop_list], [o.t for o in dsg.symop_list])
        ok = False
        for hall, _choice in halls.get(n, []):
            s = spglib.get_symmetry_from_database(hall)
            if _opset(s["rotations"], s["translations"]) == d_ops:
                ok = True
                break
        if not ok:
            unmatched.append(n)
    assert unmatched == [], f"diffpy operators match no spglib setting for: {unmatched}"


def test_diffpy_gives_origin_choice_1_for_every_two_origin_group(halls):
    """The premise of the whole fix, measured rather than assumed."""
    wrong = []
    for n in sorted(ORIGIN_SHIFT_TO_CHOICE_1):
        dsg = GetSpaceGroup(n)
        d_ops = _opset([o.R for o in dsg.symop_list], [o.t for o in dsg.symop_list])
        matched = None
        for hall, choice in halls.get(n, []):
            s = spglib.get_symmetry_from_database(hall)
            if _opset(s["rotations"], s["translations"]) == d_ops:
                matched = choice
                break
        if matched != "1":
            wrong.append((n, matched))
    assert wrong == [], f"expected origin choice 1 from diffpy, got: {wrong}"


def test_the_shift_table_is_the_one_spglib_implies(halls):
    """Re-derive all 24 shifts and compare with the table, entry by entry.

    The table is written out by hand in the module so the import stays cheap and
    spglib-free; this is what keeps it honest.
    """
    grid = [f / 8 for f in range(8)]
    candidates = [np.array(v) for v in itertools.product(grid, repeat=3)]

    derived: dict[int, tuple] = {}
    for n, entries in halls.items():
        h1 = [h for h, c in entries if c == "1"]
        h2 = [h for h, c in entries if c == "2"]
        if not (h1 and h2):
            continue
        s1 = spglib.get_symmetry_from_database(h1[0])
        o1 = _opset(s1["rotations"], s1["translations"])
        s2 = spglib.get_symmetry_from_database(h2[0])
        for p in candidates:
            conj = set()
            for R, t in zip(s2["rotations"], s2["translations"]):
                R = np.asarray(R, int)
                tt = (np.asarray(t, float) + R @ p - p) % 1.0
                conj.add((tuple(R.ravel()), tuple(np.round(tt, 4))))
            if conj == o1:
                derived[n] = tuple(round(float(x), 4) for x in p)
                break

    assert set(derived) == set(ORIGIN_SHIFT_TO_CHOICE_1), (
        "the set of two-origin groups changed: "
        f"spglib says {sorted(set(derived) ^ set(ORIGIN_SHIFT_TO_CHOICE_1))}"
    )
    for n, shift in sorted(derived.items()):
        assert np.allclose(shift, ORIGIN_SHIFT_TO_CHOICE_1[n]), (
            f"sg {n}: table says {ORIGIN_SHIFT_TO_CHOICE_1[n]}, spglib implies {shift}"
        )


def _orbit(sg_number: int, xyz) -> int:
    sg = GetSpaceGroup(sg_number)
    base = np.asarray(xyz, float)
    return len({tuple(np.round((op.R @ base + op.t) % 1.0, 6)) for op in sg.symop_list})


def test_silicon_8a_expands_to_eight_atoms_only_in_the_right_setting():
    """The measurement the whole branch turns on."""
    cif_coordinate = (0.125, 0.125, 0.125)     # Si.cif, Wyckoff 8a, origin choice 2
    assert _orbit(227, cif_coordinate) == 16, (
        "expected the untreated coordinate to land on the 16-fold site — if this "
        "changed, diffpy's setting for 227 changed and the fix needs revisiting"
    )
    fixed = to_choice_1(cif_coordinate, 227, setting=2)
    assert _orbit(227, fixed) == 8


def test_silicon_density_comes_out_right_once_the_origin_is_honoured():
    """2.329 g/cm3, the literature value — it was 4.658."""
    a_nm = 0.54309
    fixed = to_choice_1((0.125, 0.125, 0.125), 227, setting=2)
    n_atoms = _orbit(227, fixed)
    rho = n_atoms * 28.086 / (_AVOGADRO * (a_nm * 1e-7) ** 3)
    assert n_atoms == 8
    assert rho == pytest.approx(2.329, abs=0.005)


@pytest.mark.parametrize(
    "sg, a_nm, A, rho_lit, name",
    [(225, 0.4049, 26.982, 2.699, "Al"), (225, 0.35238, 58.693, 8.908, "Ni")],
)
def test_single_origin_phases_are_untouched(sg, a_nm, A, rho_lit, name):
    """The fix must be a no-op for the 206 groups with one origin."""
    assert not has_two_origin_choices(sg)
    for setting in (1, 2):
        assert np.allclose(to_choice_1((0.0, 0.0, 0.0), sg, setting), (0.0, 0.0, 0.0))
    n_atoms = _orbit(sg, (0.0, 0.0, 0.0))
    rho = n_atoms * A / (_AVOGADRO * (a_nm * 1e-7) ** 3)
    assert rho == pytest.approx(rho_lit, rel=0.01), f"{name}: {rho:.3f} vs {rho_lit}"


def test_a_choice_1_coordinate_is_returned_unchanged():
    """Only setting 2 moves; a file that really is choice 1 must not shift."""
    xyz = (0.375, 0.125, 0.375)
    assert np.allclose(to_choice_1(xyz, 227, setting=1), xyz)


# ---------------------------------------------------------------------------
# The sign of the shift
# ---------------------------------------------------------------------------
#
# The tests above cannot see it. `test_the_shift_table_is_the_one_spglib_implies`
# derives p with the same conjugation convention the module uses, so a flipped
# sign cancels out; and every functional assertion uses silicon, whose 8a site is
# the one place where +p and -p give the SAME set. The first version of this
# module added the shift and passed all of them, while putting 72 atoms in a
# spinel cell that has 56.
#
# These two compare against spglib's choice-2 operators directly, over general
# points, which is the only thing that pins the direction.

def _orbit_set(sg_number: int, xyz, hall: int | None = None):
    """The orbit as a SET of positions, via diffpy (default) or an spglib hall."""
    base = np.asarray(xyz, float)
    if hall is None:
        sg = GetSpaceGroup(sg_number)
        return {tuple(np.round((op.R @ base + op.t) % 1.0, 6)) for op in sg.symop_list}
    s = spglib.get_symmetry_from_database(hall)
    return {
        tuple(np.round((np.asarray(R, int) @ base + np.asarray(t, float)) % 1.0, 6))
        for R, t in zip(s["rotations"], s["translations"])
    }


def _hall_for(number: int, choice: str) -> int:
    for hall in range(1, 531):
        t = spglib.get_spacegroup_type(hall)
        if t["number"] == number and t.get("choice") == choice:
            return hall
    raise AssertionError(f"no hall entry for sg {number} choice {choice}")


@pytest.mark.parametrize("sg", sorted(ORIGIN_SHIFT_TO_CHOICE_1))
def test_shifted_coordinate_reproduces_the_choice_2_orbit(sg):
    """For general points, not just special ones: the shifted coordinate expanded
    with choice-1 operators must give the choice-2 orbit, shifted the same way.

    This is what fixes the direction. With the sign inverted it fails for
    sg 70, 88, 141, 142, 203, 227 and 228.
    """
    hall2 = _hall_for(sg, "2")
    shift = np.asarray(ORIGIN_SHIFT_TO_CHOICE_1[sg], float)
    probes = [(0.125, 0.0, 0.0), (0.2624, 0.2624, 0.2624), (0.125, 0.375, 0.125),
              (0.3, 0.1, 0.45), (0.5, 0.25, 0.125)]
    for x in probes:
        want = {tuple(np.round((np.asarray(r) - shift) % 1.0, 6))
                for r in _orbit_set(sg, x, hall=hall2)}
        got = _orbit_set(sg, to_choice_1(x, sg, setting=2))
        assert got == want, (
            f"sg {sg}, point {x}: expanding the shifted coordinate with choice-1 "
            f"operators gave {len(got)} atoms, the choice-2 orbit has {len(want)}"
        )


def test_spinel_has_its_56_atoms():
    """A real structure whose sites are NOT silicon's coincidental one.

    MgAl2O4, sg 227 origin choice 2: Mg on 8a, Al on 16d, O on 32e (u=0.2624)
    — 56 atoms and 3.58 g/cm3. The inverted sign gave 8 + 32 + 32 = 72.
    """
    sites = [((0.125, 0.125, 0.125), 8, "Mg 8a"),
             ((0.5, 0.5, 0.5), 16, "Al 16d"),
             ((0.2624, 0.2624, 0.2624), 32, "O 32e")]
    total = 0
    for xyz, expected, label in sites:
        n = len(_orbit_set(227, to_choice_1(xyz, 227, setting=2)))
        assert n == expected, f"{label}: {n} atoms, expected {expected}"
        total += n
    assert total == 56

    a_nm = 0.8083
    mass = 8 * 24.305 + 16 * 26.982 + 32 * 15.999
    rho = mass / (_AVOGADRO * (a_nm * 1e-7) ** 3)
    assert rho == pytest.approx(3.58, abs=0.02)
