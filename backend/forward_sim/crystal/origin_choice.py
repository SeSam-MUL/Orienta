"""Origin choice (ITA setting 1 vs 2) for the 24 space groups that have two.

WHY THIS EXISTS
---------------
Found 2026-09-25 from a Mac tester's report: our own Monte Carlo was running
silicon at 4.658 g/cm3, exactly twice its real 2.329. The cause was not the
density code but the symmetry expansion underneath it.

``Si.cif`` places silicon on Wyckoff **8a at (1/8, 1/8, 1/8)** — which is
correct in Fd-3m **origin choice 2**, the convention crystallographic databases
publish in. ``diffpy.structure.spacegroups.GetSpaceGroup(227)`` returns **origin
choice 1** operators. Under those, the very same three numbers lie on the
16-fold **16c/16d** site: same coordinates, different origin, a different
Wyckoff position. The orbit came out 16 instead of 8, so every quantity derived
from the expansion doubled — the density, and with it the electron mean free
path the MC uses, and the atom sum in ``compute_Ug_table``, i.e. the crystal
potential the whole dynamical master pattern is built from.

Measured over all 230 groups against spglib: diffpy's operator set matches a
valid spglib setting **everywhere** (it is not wrong), but for **all 24** groups
with two origin choices it is choice 1. So the mismatch is systematic for any
structure whose coordinates came from a database CIF, not a silicon quirk.

spglib cannot referee this for us: asked about the bad 16-atom silicon cell it
answers "Fd-3m (227)", because a 16c/16d arrangement genuinely is one. The only
thing that distinguishes right from wrong here is which origin the coordinates
were written in — so that has to be carried, not guessed.

THE SHIFTS
----------
``ORIGIN_SHIFT_TO_CHOICE_1[n]`` is the vector to ADD to a fractional coordinate
written in origin choice 2 to obtain the same atom in origin choice 1.

The table is not transcribed from the International Tables. It was derived by
searching the 1/8 grid for the vector ``p`` with ``ops_1 = T(p) ops_2 T(-p)``
for every group, using spglib's two settings, and
``tests/test_origin_choice.py`` re-derives it and fails if any entry drifts.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: The conjugation vector ``p`` with ``ops_1 = T(-p) ops_2 T(+p)``, per group.
#:
#: SUBTRACT it from an origin-choice-2 fractional coordinate to get the same
#: atom in origin choice 1. The sign is not cosmetic and it is not obvious: the
#: coordinate map that satisfies ``op1(phi(x)) == phi(op2(x))`` for this ``p`` is
#: ``phi(x) = x - p`` (substituting ``x + q`` forces ``R(q+p) = q+p`` for every
#: rotation, i.e. ``q = -p``). The first version of this module added it, which
#: is right for silicon by coincidence — at its 8a site ``+p`` and ``-p`` land on
#: the same 8-atom set — and wrong for 7 of the 24 groups: spinel MgAl2O4 in
#: sg 227 came out with 72 atoms instead of 56.
#:
#: Derived from spglib (see the module docstring) and pinned by a test.
ORIGIN_SHIFT_TO_CHOICE_1: dict[int, tuple[float, float, float]] = {
    48:  (0.25,  0.25,  0.25),
    50:  (0.25,  0.25,  0.0),
    59:  (0.25,  0.25,  0.0),
    68:  (0.0,   0.25,  0.25),
    70:  (0.375, 0.375, 0.375),
    85:  (0.25,  0.75,  0.0),
    86:  (0.25,  0.25,  0.25),
    88:  (0.0,   0.75,  0.375),
    125: (0.25,  0.25,  0.0),
    126: (0.25,  0.25,  0.25),
    129: (0.25,  0.75,  0.0),
    130: (0.25,  0.75,  0.0),
    133: (0.25,  0.75,  0.25),
    134: (0.25,  0.75,  0.25),
    137: (0.25,  0.75,  0.25),
    138: (0.25,  0.75,  0.25),
    141: (0.0,   0.25,  0.375),
    142: (0.0,   0.25,  0.375),
    201: (0.25,  0.25,  0.25),
    203: (0.375, 0.375, 0.375),
    222: (0.25,  0.25,  0.25),
    224: (0.25,  0.25,  0.25),
    227: (0.375, 0.375, 0.375),
    228: (0.125, 0.125, 0.125),
}


def has_two_origin_choices(space_group: int) -> bool:
    """True for the 24 ITA groups that are published in two origin choices."""
    return int(space_group) in ORIGIN_SHIFT_TO_CHOICE_1


def to_choice_1(xyz, space_group: int, setting: int):
    """Return ``xyz`` expressed in origin choice 1, the setting diffpy uses.

    ``setting`` is the origin choice the coordinate is written in (1 or 2). A
    coordinate already in choice 1, or a group with only one origin, is returned
    unchanged — so this is a no-op for 206 of the 230 groups.
    """
    r = np.asarray(xyz, dtype=np.float64)
    if int(setting) != 2 or not has_two_origin_choices(space_group):
        return r
    shift = np.asarray(ORIGIN_SHIFT_TO_CHOICE_1[int(space_group)], dtype=np.float64)
    return (r - shift) % 1.0


def warn_if_setting_unverified(space_group: int, setting: int, source: str = "") -> None:
    """Say so when a structure is in the configuration that produced bad masters.

    A file in one of the 24 two-origin groups that declares setting 1 is either
    genuinely in choice 1, or was written by a tool that did not record the
    choice at all — and the second case is what silently produced silicon at
    twice its density. We cannot tell the two apart from the coordinates (a
    16c/16d cell is a perfectly valid structure), so this does not guess: it
    says the answer is unverified and names the phase.
    """
    if not has_two_origin_choices(space_group):
        return
    if int(setting) == 2:
        # Not silence: the two writers we have derive this field from spglib's
        # ``origin_shift``, which is the translation from the input cell to the
        # standardised one — a different quantity entirely. Measured: an FCC
        # aluminium cell translated by (0.1,0.2,0.3) gets stamped "setting 2"
        # although sg 225 has only one origin. No file in the library says 2
        # today, so this is latent; but a file that did would make us shift
        # coordinates that were never in choice 2.
        logger.warning(
            "%s declares origin choice 2 for space group %d. Check that this is "
            "the ITA origin choice and not spglib's standardisation shift — the "
            "converters have been writing the latter into this field, and the "
            "two are unrelated.",
            source or "This structure", int(space_group),
        )
        return
    logger.warning(
        "%s is in space group %d, which the International Tables publish in two "
        "origin choices, and the file declares choice 1. If its coordinates in "
        "fact came from a database CIF they are almost certainly choice 2, and "
        "expanding them here puts the atoms on the wrong Wyckoff site — the "
        "fault that made silicon come out at twice its density. Check the "
        "source, or regenerate the .xtal with the origin choice recorded.",
        source or "This structure", int(space_group),
    )
