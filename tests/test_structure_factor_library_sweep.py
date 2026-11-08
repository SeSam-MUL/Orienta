"""Structure factors of every CIF in the crystal library, against pymatgen.

For each CIF: the unit cell Orienta builds (the production reader
``hough_phase_from_cif`` plus ``phase_cell``) against pymatgen's own reading of
the same file, which expands the CIF's symmetry operators in the CIF's frame
and knows nothing about orix' rotated frame.

* **Atoms:** the number of distinct positions per cell is pymatgen's number of
  sites.
* **Positions:** |F| of every low-order reflection, with every atom replaced by
  the same element, so only the positions count. pymatgen's coordinates are
  symmetry-exact where a CIF rounds (0.3333), hence a tolerance of 2e-3 of the
  strongest reflection; the old expansion is off by order one on the 17
  affected CIFs.
* **Chemistry:** where every site holds one element, |F| with the real elements
  and occupancies, with the same tolerance. Mixed sites ("0.884Al + 0.116Si") are
  taken as their dominant element by Orienta (``ebsd_utils._normalize_element_labels``),
  which is a convention and not under test here.

Si.cif and sd_1816951.cif (Fd-3m, origin choice 2) are read wrongly by pymatgen
(it expands with choice-1 operators: 16 and 40 atoms where the crystals have 8 and
24); they are checked against the known cell instead.

Skips visibly when the library is absent.
"""
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("orix")
pytest.importorskip("diffsims")
pytest.importorskip("pymatgen")

from tests.data_deps import CIF_LIBRARY, requires  # noqa: E402

pytestmark = requires(CIF_LIBRARY)

from diffpy.structure import Atom, Lattice, Structure  # noqa: E402
from diffsims.crystallography import ReciprocalLatticeVector as R  # noqa: E402
from diffsims.utils.sim_utils import _get_kinematical_structure_factor as _K  # noqa: E402

import ebsd_utils as eu  # noqa: E402
import phase_cell as pc  # noqa: E402

#: pymatgen ignores the origin-choice marker (see module docstring).
ORIGIN_CHOICE_2 = {"Si.cif": 8, "sd_1816951.cif": 24}

TOL = 2e-3
D_MIN = 1.2


def _cifs():
    if not CIF_LIBRARY.is_dir():
        return []
    return sorted(p for p in CIF_LIBRARY.glob("*.cif") if "backup" not in p.name)


CIFS = _cifs()
IDS = [p.name.encode("ascii", "replace").decode() for p in CIFS]


def _pymatgen(path):
    from pymatgen.core import Structure as PS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PS.from_file(str(path))


def _hkl(lattice):
    hk = np.array([h for h in itertools.product(range(-4, 5), repeat=3) if h != (0, 0, 0)],
                  float)
    return hk[1.0 / lattice.rnorm(hk) >= D_MIN]


def _orienta_cell(path):
    phase = eu.hough_phase_from_cif(str(path))
    eu._normalize_element_labels(phase)
    return pc.complete_unit_cell(phase)


def _F(phase, hkl):
    r = R(phase, hkl=hkl)
    pc.calculate_structure_factors(r)     # what the reflector code does (see phase_cell)
    return np.abs(np.asarray(r.structure_factor).ravel())


def _distinct_positions(cell):
    found = []
    for f in pc.true_fractional(cell):
        if not any(np.all(np.abs((f - g + .5) % 1 - .5) < 1e-3) for g in found):
            found.append(f)
    return len(found)


def _is_simple(s, cell):
    """Every site one element, and Orienta's labels are those elements."""
    if any(len(site.species) != 1 for site in s):
        return False
    pmg = sorted(sp.symbol for site in s for sp in site.species)
    return pmg == sorted(str(a.element) for a in cell.structure)


def _positions_only(cell):
    """The cell with one aluminium atom, full occupancy, at each distinct position
    (a mixed site is two atoms at one position in Orienta, one site in pymatgen)."""
    flat = cell.deepcopy()
    keep, seen = [], []
    for a, f in zip(flat.structure, pc.true_fractional(flat)):
        if any(np.all(np.abs((f - g + .5) % 1 - .5) < 1e-3) for g in seen):
            continue
        seen.append(f)
        a.element, a.occupancy = "Al", 1.0
        keep.append(a)
    del flat.structure[:]
    flat.structure.extend(keep)
    return flat


@pytest.mark.parametrize("path", CIFS, ids=IDS)
def test_atoms_and_positions_agree_with_pymatgen(path):
    cell = _orienta_cell(path)
    if path.name in ORIGIN_CHOICE_2:
        assert _distinct_positions(cell) == ORIGIN_CHOICE_2[path.name]
        return
    s = _pymatgen(path)
    assert _distinct_positions(cell) == len(s)

    # positions only: every atom the same element at full occupancy
    ref = Structure(lattice=Lattice(*s.lattice.abc, *s.lattice.angles))
    for site in s:
        ref.append(Atom("Al", xyz=site.frac_coords))
    hkl = _hkl(ref.lattice)
    Fref = np.abs(_K(ref, hkl, ref.lattice.rnorm(hkl), scattering_params="xtables"))
    got = _F(_positions_only(cell), hkl)
    assert len(hkl) > 40
    assert np.abs(got - Fref).max() <= TOL * Fref.max()


@pytest.mark.parametrize("path", CIFS, ids=IDS)
def test_structure_factors_with_elements_agree_where_the_sites_are_ordered(path):
    if path.name in ORIGIN_CHOICE_2:
        pytest.skip("pymatgen expands this CIF with the wrong origin choice")
    s = _pymatgen(path)
    cell = _orienta_cell(path)
    if not _is_simple(s, cell):
        pytest.skip("mixed sites: Orienta takes the dominant element (convention)")
    ref = Structure(lattice=Lattice(*s.lattice.abc, *s.lattice.angles))
    for site in s:
        sp = next(iter(site.species))
        ref.append(Atom(sp.symbol, xyz=site.frac_coords,
                        occupancy=float(site.species[sp])))
    hkl = _hkl(ref.lattice)
    Fref = np.abs(_K(ref, hkl, ref.lattice.rnorm(hkl), scattering_params="xtables"))
    assert np.abs(_F(cell, hkl) - Fref).max() <= TOL * Fref.max()


def test_the_sweep_really_covers_the_non_cubic_systems():
    """The ordered-site comparison skips mixed sites; make sure what it still compares
    includes the crystal systems the old expansion broke (otherwise a green sweep would
    prove nothing about them)."""
    covered = {}
    for path in CIFS:
        if path.name in ORIGIN_CHOICE_2:
            continue
        s = _pymatgen(path)
        if all(len(site.species) == 1 for site in s):
            lat = s.lattice
            a, b, c = lat.abc
            al, be, ga = lat.angles
            if abs(ga - 120) < 1 and abs(al - 90) < 1:
                key = "hexagonal/trigonal"
            elif abs(al - 90) < 1 and abs(ga - 90) < 1 and abs(be - 90) > 0.01:
                key = "monoclinic"
            elif abs(al - 90) > 0.01 or abs(be - 90) > 0.01 or abs(ga - 90) > 0.01:
                key = "triclinic/other oblique"
            else:
                key = "rectangular"
            covered.setdefault(key, []).append(path.name)
    assert covered.get("hexagonal/trigonal") and covered.get("monoclinic")
    assert covered.get("triclinic/other oblique") or covered.get("monoclinic")
    assert len(covered.get("rectangular", [])) >= 3


def test_silicon_and_mgcu2_have_the_cells_of_their_crystals():
    """The two origin-choice-2 CIFs, against what the crystals are, not pymatgen:
    diamond Si has 8 atoms and no {222}; the MgCu2 Laves phase 8 Mg + 16 Cu."""
    si = _orienta_cell(CIF_LIBRARY / "Si.cif")
    assert len(si.structure) == 8
    F = _F(si, np.array([[1, 1, 1], [2, 2, 0], [2, 2, 2], [4, 0, 0], [1, 0, 0], [2, 1, 0]],
                        float))
    assert F[0] > 1 and F[1] > 1 and F[3] > 1
    assert F[2] < 1e-6 * F.max() and F[4] < 1e-6 * F.max() and F[5] < 1e-6 * F.max()
    mgcu = _orienta_cell(CIF_LIBRARY / "sd_1816951.cif")
    els = sorted(str(a.element) for a in mgcu.structure)
    assert els == ["Cu"] * 16 + ["Mg"] * 8
