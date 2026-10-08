"""Point-group symmetry of monoclinic phases in Orienta's crystal frame.

Orienta expresses every orientation in the crystal frame of orix and EMsoft:
``X || a`` and ``Z || c*`` (``Y`` completes a right-handed set). For a
monoclinic crystal whose cell has its unique axis along ``b`` (the standard
setting of the International Tables, beta != 90 degrees), the two-fold axis is
therefore along ``Y``.

orix, however, knows exactly one monoclinic point group per class, the one with
the two-fold axis along ``Z`` (``C2h`` "2/m", ``C2`` "2" and ``Cs`` "m" in
``orix.quaternion.symmetry`` are all unique-axis-c). Reducing the orientation of
a b-unique phase with that group reports two orientations related by the true
two-fold axis as 180 degrees apart, and hands them different IPF colours.

The indexers do not have this problem: PyEBSDIndex's monoclinic quaternions put
the axis along ``b`` and the library ``.sht`` files carry ``z_rot = 1``. So the
stored orientations are right and only the symmetry that reduces them is wrong.
This module is the one place that decides which symmetry a monoclinic phase
gets:

* :func:`frame_symmetry` chooses the unique axis from the space group (the
  direction of its two-fold axis or mirror normal), falls back to the lattice
  angles, and finally to the convention of the indexers, ``b``.
* Every other crystal system is returned exactly as orix returns it.
* The b-unique centrosymmetric group keeps the name ``"2/m"``. Tables keyed by
  point-group name (TSL codes, Laue classes, ``.sht`` headers, saved results)
  therefore keep working, and a ``"2/m"`` stored by an earlier version means the
  same thing it meant then. The axis is carried by the generators and is
  visible as the name of the proper subgroup (``"121"`` for b, orix's own name
  for a rotation about ``Y``); :func:`unique_axis` reads it back.
* :class:`FramePhase` is an orix ``Phase`` whose ``point_group`` is decided
  here. It keeps its space group, because orix derives the point group from the
  space group number and would otherwise ignore any symmetry assigned to it.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from orix.crystal_map import Phase, PhaseList
from orix.quaternion import Symmetry
from orix.quaternion.symmetry import (
    C2x,
    C2y,
    C2z,
    Ci,
    Csx,
    Csy,
    Csz,
    get_point_group,
)

logger = logging.getLogger(__name__)

#: Unique axis assumed when nothing says otherwise: the standard setting of the
#: International Tables and the axis PyEBSDIndex and the library ``.sht`` files
#: use.
DEFAULT_UNIQUE_AXIS = "b"

#: Names under which a monoclinic point group reaches Orienta without an axis.
_AMBIGUOUS = {"2", "m", "2/m"}

#: Monoclinic point-group classes by space-group number (standard 3..15).
_PG2 = range(3, 6)
_PGM = range(6, 10)
_PG2M = range(10, 16)

#: ``unique axis -> (rotation, mirror)`` in the crystal frame X||a, Z||c*.
_GENERATORS = {
    "a": (C2x, Csx),
    "b": (C2y, Csy),
    "c": (C2z, Csz),
}


#: Attribute that marks a symmetry built here: it names its unique axis, so it
#: is never mistaken for the axis-less object orix hands out.
_TAG = "_orienta_unique_axis"


def _tagged(sym: Symmetry, name: str, axis: str) -> Symmetry:
    out = Symmetry(sym)
    out.improper = np.asarray(sym.improper, dtype=bool).copy()
    out.name = name
    setattr(out, _TAG, axis)
    return out


def _build() -> dict:
    """The nine monoclinic groups (three classes x three axes), built once.

    Names are orix's own (``"211"``, ``"121"``, ``"112"``, ``"m11"``, ...)
    except the centrosymmetric class, which keeps ``"2/m"`` for every axis (see
    the module doc). They are copies; orix's module-level objects are not
    touched.
    """
    table = {}
    for axis in ("a", "b", "c"):
        rot, mir = _GENERATORS[axis]
        centro = Symmetry.from_generators(rot, Ci)
        table[axis] = {
            "2": _tagged(rot, rot.name, axis),
            "m": _tagged(mir, mir.name, axis),
            "2/m": _tagged(centro, "2/m", axis),
        }
    return table


_TABLE = _build()
_EXPLICIT_AXIS = {"211": "a", "m11": "a", "121": "b", "1m1": "b",
                  "112": "c", "11m": "c"}
_EXPLICIT_CLASS = {"211": "2", "121": "2", "112": "2",
                   "m11": "m", "1m1": "m", "11m": "m"}


def _class_of(point_group) -> Optional[str]:
    """``"2"``, ``"m"`` or ``"2/m"`` for a monoclinic point group, else None."""
    name = getattr(point_group, "name", point_group)
    name = str(name).strip() if name is not None else ""
    if name in _AMBIGUOUS:
        return name
    return _EXPLICIT_CLASS.get(name)


def unique_axis(point_group) -> Optional[str]:
    """Unique axis ``"a"``, ``"b"`` or ``"c"`` of a monoclinic symmetry.

    ``"b"`` (:data:`DEFAULT_UNIQUE_AXIS`) for a name that does not say, as
    :func:`frame_symmetry` would resolve it; None for anything that is not
    monoclinic.
    """
    if _class_of(point_group) is None:
        return None
    tag = getattr(point_group, _TAG, None)
    if tag is not None:
        return tag
    name = str(getattr(point_group, "name", point_group)).strip()
    return _EXPLICIT_AXIS.get(name, DEFAULT_UNIQUE_AXIS)


def _space_group_axis(space_group_number: int) -> Optional[str]:
    """Unique axis of a monoclinic space group, from its own symmetry operations.

    The direction of the two-fold axis (class 2 and 2/m) or of the mirror normal
    (class m) in the setting the space-group number denotes: standard numbers
    3..15 are b-unique, the alternative settings of the International Tables
    (e.g. ``A12/a1`` = 4015, ``B112/m`` = 1012) carry their own axis.
    """
    try:
        from diffpy.structure.spacegroups import GetSpaceGroup

        group = GetSpaceGroup(int(space_group_number))
    except Exception:
        return None
    for op in group.iter_symops():
        rot = np.asarray(op.R, dtype=float)
        det = np.linalg.det(rot)
        trace = np.trace(rot)
        if abs(det - 1.0) < 1e-9 and abs(trace + 1.0) < 1e-9:      # two-fold
            wanted = 1.0
        elif abs(det + 1.0) < 1e-9 and abs(trace - 1.0) < 1e-9:    # mirror
            wanted = -1.0
        else:
            continue
        values, vectors = np.linalg.eig(rot)
        vec = vectors[:, int(np.argmin(np.abs(values - wanted)))].real
        idx = int(np.argmax(np.abs(vec)))
        if abs(vec[idx]) > 0.99:
            return "abc"[idx]
    return None


def _lattice_axis(lattice) -> Optional[str]:
    """Unique axis from the cell angles: the one angle that is not 90 degrees."""
    if lattice is None:
        return None
    try:
        if hasattr(lattice, "alpha"):
            angles = (lattice.alpha, lattice.beta, lattice.gamma)
        else:
            angles = tuple(lattice)[3:6]
        off = [abs(float(a) - 90.0) > 1e-2 for a in angles]
    except Exception:
        return None
    if sum(off) != 1:
        return None
    return "abc"[off.index(True)]


def monoclinic_class_of_space_group(space_group_number) -> Optional[str]:
    """``"2"``, ``"m"``, ``"2/m"`` for a monoclinic space group number, else None.

    Alternative settings (``n % 1000``) are the same class.
    """
    try:
        n = int(space_group_number) % 1000
    except (TypeError, ValueError):
        return None
    if n in _PG2:
        return "2"
    if n in _PGM:
        return "m"
    if n in _PG2M:
        return "2/m"
    return None


def frame_symmetry(point_group=None, *, space_group=None, lattice=None):
    """THE decision: the point-group symmetry of a phase in the crystal frame.

    Parameters
    ----------
    point_group
        An orix ``Symmetry``, a point-group name or None.
    space_group
        Space-group number (1..230, or an alternative setting such as 4015).
        When given it decides the point group, as it does in orix.
    lattice
        Lattice (diffpy ``Lattice`` or ``(a, b, c, alpha, beta, gamma)``). Only
        consulted for a monoclinic phase whose space group says nothing.

    Returns
    -------
    The symmetry orix would return, except for a monoclinic phase without an
    explicit unique axis, which gets the group whose two-fold axis (or mirror
    normal) lies along the axis its cell has: ``Y`` for the usual b-unique
    setting. Phases of every other crystal system are returned untouched.
    """
    if space_group is not None:
        cls = monoclinic_class_of_space_group(space_group)
        if cls is None:
            return get_point_group(int(space_group))
        axis = (_space_group_axis(space_group) or _lattice_axis(lattice)
                or DEFAULT_UNIQUE_AXIS)
        return _TABLE[axis][cls]
    if point_group is None:
        return None
    cls = _class_of(point_group)
    if cls is None:
        return point_group
    if getattr(point_group, _TAG, None) is not None:
        return point_group
    name = str(getattr(point_group, "name", point_group)).strip()
    axis = _EXPLICIT_AXIS.get(name) or _lattice_axis(lattice) or DEFAULT_UNIQUE_AXIS
    return _TABLE[axis][cls]


class FramePhase(Phase):
    """An orix ``Phase`` whose point group is decided by :func:`frame_symmetry`.

    orix derives ``point_group`` from the space group number whenever there is
    one, so a symmetry assigned afterwards is ignored. This subclass asks
    :func:`frame_symmetry` instead, with the space group and the lattice of the
    structure, and keeps everything else of ``Phase`` (including
    ``from_cif``, which builds ``cls`` and therefore a ``FramePhase``).
    """

    @property
    def point_group(self):
        sg = self.space_group
        lattice = None
        structure = getattr(self, "structure", None)
        if structure is not None:
            lattice = getattr(structure, "lattice", None)
        if sg is not None:
            return frame_symmetry(space_group=sg.number, lattice=lattice)
        return frame_symmetry(self._point_group, lattice=lattice)

    @point_group.setter
    def point_group(self, value) -> None:
        Phase.point_group.fset(self, value)


def frame_phase(phase):
    """``phase`` as a :class:`FramePhase` (same object if it already is one)."""
    if phase is None or isinstance(phase, FramePhase):
        return phase
    new = FramePhase(
        name=phase.name,
        space_group=phase.space_group,
        point_group=getattr(phase, "_point_group", None),
        color=phase.color,
        structure=phase.structure,
    )
    return new


def frame_phase_list(phases) -> PhaseList:
    """A ``PhaseList`` with the same ids whose phases are all FramePhases."""
    if phases is None:
        return phases
    try:
        items = {pid: frame_phase(p) for pid, p in phases}
    except TypeError:
        return phases
    if not items:
        return phases
    return PhaseList(phases=items)


def frame_xmap(xmap):
    """Give ``xmap`` frame-aware phases (in place) and return it."""
    if xmap is None or getattr(xmap, "phases", None) is None:
        return xmap
    phases = xmap.phases
    if all(isinstance(p, FramePhase) for _, p in phases):
        return xmap
    xmap.phases = frame_phase_list(phases)
    return xmap


def phase_from_cif(path):
    """``Phase.from_cif`` returning a :class:`FramePhase`."""
    return FramePhase.from_cif(path)


__all__ = [
    "DEFAULT_UNIQUE_AXIS",
    "FramePhase",
    "frame_phase",
    "frame_phase_list",
    "frame_symmetry",
    "frame_xmap",
    "monoclinic_class_of_space_group",
    "phase_from_cif",
    "unique_axis",
]
