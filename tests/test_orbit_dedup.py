"""Two symmetry images of one atom have to collapse to one atom.

`_expand_symmetry_orbit` decided that with `tuple(np.round(r, 6))` — an exact
key on coordinates that come out of a **float32** `AtomData` array. A site on a
special position is then only on it to about seven digits, and images that
should coincide get different keys.

Measured on the real library: `Mg17Al12_mp-2151_conventional_standard.xtal`
stores (0.672099, 0.327900, 0.327900), which misses the special-position
relation in the sixth decimal. Its 8-fold site expanded to 24, the cell came out
with 74 atoms instead of 58, and the density with it — 2.647 instead of 2.095
g/cm3. The file was never wrong; the expander was.

The gap the fix sits in was measured over every .xtal in the library:

    widest spread inside a coincident cluster :  6e-8
    closest approach of two DISTINCT sites    :  1.0e-1

so a 1e-4 grid has three orders of magnitude of room on the one side and three
on the other. Across the library it changes the atom count of exactly one file
out of 35 — that one.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.forward_sim.crystal.structure_matrix import _ORBIT_GRID, _expand_symmetry_orbit
from backend.forward_sim.crystal.xtal_io import Atom, CrystalStructure

_AVOGADRO = 6.02214076e23


def _structure(space_group: int, a_nm: float, sites: list[tuple[int, tuple[float, float, float]]]):
    return CrystalStructure(
        lattice=(a_nm, a_nm, a_nm, 90.0, 90.0, 90.0),
        space_group=space_group,
        crystal_system=1,
        atoms=[Atom(Z=Z, xyz=np.asarray(xyz, dtype=np.float64), occ=1.0, B=0.005)
               for Z, xyz in sites],
    )


# The four representatives of Mg17Al12 exactly as the .xtal stores them, rounded
# to float32 the way the file does.
_MG17AL12 = [
    (12, (0.0, 0.0, 0.0)),
    (12, (0.672099, 0.327900, 0.327900)),
    (12, (0.143345, 0.856655, 0.540347)),
    (13, (0.909369, 0.090631, 0.275813)),
]


def test_mg17al12_has_its_58_atoms():
    """2 + 8 + 24 + 24. It was 2 + 24 + 24 + 24 = 74."""
    structure = _structure(217, 1.05326, _MG17AL12)
    orbit = _expand_symmetry_orbit(structure)
    assert len(orbit) == 58

    mass = sum({12: 24.305, 13: 26.982}[int(e[0])] * float(e[1]) for e in orbit)
    rho = mass / (_AVOGADRO * (1.05326 * 1e-7) ** 3)
    assert rho == pytest.approx(2.095, abs=0.005)


def test_the_site_that_broke_it_expands_eightfold_on_its_own():
    """Isolating the one representative, so a change elsewhere cannot mask it."""
    structure = _structure(217, 1.05326, [_MG17AL12[1]])
    assert len(_expand_symmetry_orbit(structure)) == 8


@pytest.mark.parametrize("wobble", [0.0, 1e-7, -1e-7, 5e-7])
def test_a_site_slightly_off_its_special_position_still_expands_once(wobble):
    """float32 storage is worth about 1e-7; the answer must not depend on it.

    This is the actual failure: the exact coordinate gives 8, and the same
    coordinate plus a rounding error used to give 24.
    """
    x, y, z = _MG17AL12[1][1]
    structure = _structure(217, 1.05326, [(12, (x + wobble, y - wobble, z + wobble))])
    assert len(_expand_symmetry_orbit(structure)) == 8


def test_distinct_sites_are_not_merged():
    """The other direction: a coarser key must not eat real atoms.

    Silicon's 16c site, whose images are 0.125 apart — 1250 grid cells, so the
    tolerance would have to grow by three orders of magnitude to reach them.
    """
    structure = _structure(227, 0.54309, [(14, (0.375, 0.125, 0.375))])
    assert len(_expand_symmetry_orbit(structure)) == 16


def test_an_atom_at_the_cell_edge_is_one_atom_not_two():
    """0.999999 and 0.0 are the same position; a grid without the wrap splits them."""
    at_origin = _structure(225, 0.4049, [(13, (0.0, 0.0, 0.0))])
    just_under = _structure(225, 0.4049, [(13, (1.0 - 1e-7, 1.0 - 1e-7, 1.0 - 1e-7))])
    assert len(_expand_symmetry_orbit(at_origin)) == 4
    assert len(_expand_symmetry_orbit(just_under)) == 4


def test_the_grid_sits_between_the_two_measured_distances():
    """If someone retunes the grid, these are the numbers it has to stay between.

    Coincident images differ by at most 6e-8 (so the cell must be wider than
    that), distinct sites by at least 1.0e-1 (so it must be far narrower).
    """
    cell = 1.0 / _ORBIT_GRID
    assert cell > 6e-8 * 10, "too fine: rounding noise would split one atom in two"
    assert cell < 1.0e-1 / 10, "too coarse: distinct sites could be merged"
