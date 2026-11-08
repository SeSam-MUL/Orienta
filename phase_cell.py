"""The complete unit cell of an orix ``Phase``, for structure factors in every crystal system.

Why this module exists
----------------------
orix keeps ``Phase.structure`` in a frame of its own (``e1 || a``, ``e3 || c*``)
and re-places the atoms with ``placeInLattice``, which keeps their CARTESIAN
position. The ``Atom.xyz`` an orix phase hands out are therefore fractional
coordinates of a rotated lattice. They equal the coordinates of the CIF only
when the orix frame and diffpy's standard frame coincide (``a || a*`` and
``c || c*``): cubic, tetragonal and orthorhombic cells. For hexagonal, trigonal,
monoclinic and triclinic cells they do not (hcp Mg: ``(1/3, 2/3, 1/4)`` becomes
``(0.577, 0.577, 0.25)``).

``diffsims`` knows this: ``calculate_structure_factor`` multiplies the atom
positions with ``inv(stdbase @ recbase)`` before it sums. But
``ReciprocalLatticeVector.sanitise_phase`` does not. It applies the space-group
operators to the ROTATED coordinates, and it does not remove duplicates modulo
one. hcp Mg (2 atoms) comes out with 24, and the extinct ``(2-1-11)`` reflection
gets |F| = 8.7 where the hand formula gives 0. The missing duplicate removal also
hits cells with right angles: the old expansion keeps atoms that land on the
same site, so some orthorhombic phases of the crystal library grow as well.

What it does
------------
:func:`complete_unit_cell` converts the atoms to true fractional coordinates
with the transform ``diffsims`` uses, expands them with the space group in that
frame, wraps them into ``[0, 1)``, removes duplicates (same element, same
position within 0.02 A, so a mixed site keeps both of its elements) and
converts back, so ``calculate_structure_factor`` stays correct. A phase that
already holds the whole cell, or only its asymmetric unit, comes out the same
way, and applying it twice changes nothing.

:func:`sanitise_reflectors` is the drop-in for ``rlv.sanitise_phase()``.
"""
from __future__ import annotations

import copy
import logging

import numpy as np

from crystal_symmetry import TRIGONAL_Y_SETTING_SPACE_GROUPS

logger = logging.getLogger(__name__)

#: Two atoms of one element closer than this (Angstrom, nearest periodic image)
#: are one atom. It is a distance and not a fractional coordinate because the
#: rounding of a CIF is: coordinates written with three digits (0.333 for 1/3) put
#: the images of an orbit 1e-3 of the cell edge apart, which no fractional
#: tolerance below that merges and one above it cannot tell from a real pair of
#: sites in a large cell. The closest pair of distinct sites in the crystal
#: library is 0.49 A apart; 0.02 A is the rounding of a 20 A cell to three digits.
POSITION_TOL = 0.02


def _frame_matrix(phase) -> np.ndarray:
    """``M`` with ``orix_xyz = true_fractional @ M``.

    The inverse of the transform ``diffsims`` applies to ``atom.xyz`` in
    ``calculate_structure_factor``; kept in this one function so the two can
    only disagree if ``diffsims`` changes, which the tests would see.
    """
    lat = phase.structure.lattice
    return np.asarray(lat.stdbase, dtype=float) @ np.asarray(lat.recbase, dtype=float)


def true_fractional(phase) -> np.ndarray:
    """Fractional coordinates, in the cell of the CIF, of every atom of ``phase``.

    ``(n_atoms, 3)``. These are the coordinates the space-group operators and
    the Miller indices of the CIF refer to.
    """
    xyz = np.array([a.xyz for a in phase.structure], dtype=float).reshape(-1, 3)
    return xyz @ np.linalg.inv(_frame_matrix(phase))


def _element_key(atom) -> str:
    """The element symbol; an atomic number (an ``int`` element) becomes its symbol,
    which is what the scattering factors are looked up by (diffsims' own
    ``sanitise_phase`` converted it; a number left in place scatters nothing)."""
    from diffsims.structure_factor.atomic_scattering_parameters import (
        _get_string_from_element_id,
    )

    element = getattr(atom, "element", "")
    if np.issubdtype(type(element), np.integer):
        return _get_string_from_element_id(int(element))
    return str(element)


def _present(positions, elements, occupancies, element, occupancy, frac, base, tol) -> bool:
    """Is there already the same atom within ``tol`` Angstrom of ``frac``?

    The same atom: same element, same occupancy. Two entries of one element at
    one site with different occupancies are two parts of a split site and both
    count (their occupancies add up in the structure factor); equal ones are a
    duplicate, which is what a CIF reader leaves behind when it cannot tell the
    images of an orbit apart.

    ``base``: rows are the lattice vectors (Angstrom) of the true frame.
    """
    if not positions:
        return False
    idx = [i for i, (e, o) in enumerate(zip(elements, occupancies))
           if e == element and abs(o - occupancy) < 1e-6]
    if not idx:
        return False
    diff = np.asarray(positions)[idx] - frac
    diff -= np.rint(diff)                     # nearest periodic image
    return bool(np.any(np.linalg.norm(diff @ base, axis=1) < tol))


def complete_unit_cell(phase, tol: float = POSITION_TOL):
    """A copy of ``phase`` whose structure holds the whole unit cell, each atom once.

    ``phase`` may hold the full cell (as read from a CIF) or only the
    asymmetric unit, in any crystal system. The result is expressed in orix'
    frame, like every ``Phase.structure``, so ``calculate_structure_factor``
    gives the structure factors of the CIF's cell. Occupancies, labels and
    thermal parameters of the atoms are carried unchanged (an atomic number as
    element becomes the symbol); atoms of different elements on one site are both
    kept, and so are two entries of one element with different occupancies.

    Entries of one element with the same occupancy at the same position (within
    ``POSITION_TOL``, an Angstrom distance) are one atom and are merged, whether
    they are symmetry images or two rows written for the same site: a split site
    listed as two 0.5 rows at identical coordinates comes out as one 0.5 atom, as it
    did with diffsims' ``sanitise_phase``. Write such a site with different
    occupancies, or different coordinates, to keep both.

    The input is not modified. Raises ``ValueError`` without a space group (the
    message ``diffsims`` gave), and returns an unchanged copy for a phase
    without atoms: no atom is ever invented.
    """
    from diffpy.structure.symmetryutilities import expandPosition

    if phase.space_group is None:
        raise ValueError(f"The phase {phase} must have a space group set")
    out = phase.deepcopy()
    structure = out.structure
    if len(structure) == 0:
        return out

    M = _frame_matrix(out)
    to_true = np.linalg.inv(M)
    base = np.asarray(out.structure.lattice.stdbase, dtype=float)

    positions: list = []          # true fractional, wrapped into [0, 1)
    elements: list = []
    occupancies: list = []
    atoms: list = []
    for atom in list(structure):
        element = _element_key(atom)
        occupancy = float(getattr(atom, "occupancy", 1.0))
        frac = np.asarray(atom.xyz, dtype=float) @ to_true
        frac = frac - np.floor(frac)
        if _present(positions, elements, occupancies, element, occupancy, frac, base, tol):
            continue                              # already in an earlier orbit
        orbit = expandPosition(out.space_group, frac)[0]
        # the atom itself first, so a phase that is already complete keeps its
        # own coordinates as the representative of each orbit
        for pos in [frac] + [np.asarray(p, dtype=float) for p in orbit]:
            pos = pos - np.floor(pos)
            if _present(positions, elements, occupancies, element, occupancy, pos, base, tol):
                continue
            new_atom = copy.deepcopy(atom)
            new_atom.element = element
            new_atom.xyz = pos @ M
            positions.append(pos)
            elements.append(element)
            occupancies.append(occupancy)
            atoms.append(new_atom)

    # Rewrite the atoms in place: assigning ``phase.structure`` would run orix'
    # frame setter again and overwrite the lattice it keeps for
    # ``expand_asymmetric_unit``.
    del structure[:]
    structure.extend(atoms)
    return out


def sanitise_reflectors(rlv) -> None:
    """Replace ``rlv.phase`` by its complete unit cell. The drop-in for ``rlv.sanitise_phase()``.

    ``ReciprocalLatticeVector.sanitise_phase`` expands the structure with the
    space group in orix' rotated frame (wrong for hexagonal, trigonal, monoclinic
    and triclinic cells) and keeps duplicates. This one does neither, and it does
    not touch the phase object the vectors were built from: ``rlv.phase`` is
    replaced by a completed copy.
    """
    rlv.phase = complete_unit_cell(rlv.phase)


#: Space groups whose structure factors diffsims must not share across a family.
#:
#: ``calculate_structure_factor`` computes the first vector of each symmetry
#: family and copies the value to the rest. The families come from
#: ``phase.point_group``, and orix gives the two orientations of one point group
#: the same symmetry (``-31m`` and ``-3m1`` are both D3d, ``312``/``321`` both
#: D3, ``31m``/``3m1`` both C3v). Where the space group has the other
#: orientation, its mirror planes and two-fold axes lie 30 degrees about ``c``
#: away from the group orix used, and a "family" holds reflections that differ
#: physically. Measured with a general-position structure in orix 0.14.1 and
#: 0.15.0 alike, 60 of 284 reflections are wrong in 149, 151, 153, 157 and 162
#: and 48 of 284 in 159 and 163 (Cr2N, P-31m: 42 of 212). Those seven groups are
#: the whole set.
#:
#: The other trigonal and hexagonal groups (150, 152, 154, 156, 158, 160, 161,
#: 187-190, ...) are deliberately NOT in it: diffsims is right there for a
#: general-position structure (all of 143-194 are checked in
#: ``tests/test_structure_factor_trigonal.py``). Adding them is not harmless
#: either: for pi-Al8FeMg3Si6 (189) the per-reflection sum differs from diffsims
#: by 2.9e-4 of the strongest |F| (the CIF's rounded 0.3333 makes members of one
#: family differ in the fourth digit; with exact coordinates the last-digit noise
#: still reorders tied rows), which changes the rows kept at the cut to 70 in the
#: default Hough list.
#:
#: The cost is one sum per reflection instead of one per family.
#:
#: The set is the one ``crystal_symmetry.frame_symmetry`` answers with its own
#: group for the orientation symmetry (the setting with the two-fold axes along
#: Y instead of orix' X), and it is taken from there.
#:
#: A ``FramePhase`` (and any phase that went through ``frame_phase``) carries
#: that group as its ``point_group``. orix' ``Miller.symmetrise`` and the
#: diffsims families are built from it, so for these seven groups they now hold
#: the true symmetry classes instead of the ones of the other setting. That is
#: intended, and it changes which reflections share a family, hence the
#: reflector lists of these groups. A plain orix ``Phase`` still gets orix' group
#: and the old families.
PER_REFLECTION_SPACE_GROUPS = TRIGONAL_Y_SETTING_SPACE_GROUPS


def needs_per_reflection_structure_factor(phase) -> bool:
    """True if ``phase`` has a space group of :data:`PER_REFLECTION_SPACE_GROUPS`."""
    sg = getattr(getattr(phase, "space_group", None), "number", None)
    return sg is not None and int(sg) % 1000 in PER_REFLECTION_SPACE_GROUPS


def calculate_structure_factors(rlv, scattering_params: str = "xtables") -> None:
    """The drop-in for ``rlv.calculate_structure_factor()``.

    Identical to diffsims' method except for the space groups in
    :data:`PER_REFLECTION_SPACE_GROUPS`: there every reflection gets its own sum
    over the atoms and nothing is copied between reflections.
    """
    if not needs_per_reflection_structure_factor(rlv.phase):
        rlv.calculate_structure_factor(scattering_params)
        return
    from diffsims.utils.sim_utils import _get_kinematical_structure_factor

    hkl = np.asarray(rlv.hkl, dtype=float)
    flat = hkl.reshape(-1, 3)
    lattice = rlv.phase.structure.lattice
    F = _get_kinematical_structure_factor(
        structure=rlv.phase.structure,
        g_indices=flat,
        g_hkls_array=lattice.rnorm(flat),
        scattering_params=scattering_params,
    )
    rlv._structure_factor = np.asarray(F, dtype="complex128").reshape(hkl.shape[:-1])
