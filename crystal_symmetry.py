"""Point-group symmetry of monoclinic and trigonal phases in Orienta's crystal frame.

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
* Every other crystal system is returned exactly as orix returns it, except
  seven trigonal space groups (see below).
* The b-unique centrosymmetric group keeps the name ``"2/m"``. Tables keyed by
  point-group name (TSL codes, Laue classes, ``.sht`` headers, saved results)
  therefore keep working, and a ``"2/m"`` stored by an earlier version means the
  same thing it meant then. The axis is carried by the generators and is
  visible as the name of the proper subgroup (``"121"`` for b, orix's own name
  for a rotation about ``Y``); :func:`unique_axis` reads it back.
* :class:`FramePhase` is an orix ``Phase`` whose ``point_group`` is decided
  here. It keeps its space group, because orix derives the point group from the
  space group number and would otherwise ignore any symmetry assigned to it.

Trigonal phases have the same problem in another form. For each of the Laue
class -3m orix knows one setting of the point group, the one with the two-fold
axes (or the mirror normals) along ``a``: ``D3`` "32", ``C3v`` "3m", ``D3d``
"-3m". The space groups 149, 151, 153 (point group 312), 157, 159 (31m) and 162,
163 (-31m) have them perpendicular to ``a``, along ``Y``, 30 degrees about ``c``
further on. (The hexagonal groups 187-190 are not affected: their Laue class
6/mmm holds both.) For these seven :func:`frame_symmetry` returns a group that

* is named after its setting: ``"312"`` for 149, 151, 153 (orix knows that name
  and reads it as ``D3y``, the true operators), and orix' own ``"3m"`` and
  ``"-3m"`` for 157, 159 and 162, 163, because orix has no name for the other
  setting of those two (the classes 31m and -31m). orix writes and reads a
  phase's point group by name (H5, TSL codes, the tables keyed by name) and
  fails on a name it does not know. The group of 149, 151, 153 must not be
  called ``"32"``: orix reads that name as ``D3x`` (named ``"321"``) and, as
  its name differs from the one derived from the space group, drops the space
  group;
* has the operators of the true setting and carries a tag, ``"312"``, ``"31m"``
  or ``"-31m"``, that :func:`point_group_key` reads for code that has to pass a
  group on as text;
* answers ``laue`` and ``fundamental_sector`` for that setting. orix would build
  the Laue group without the tag and then pick the sector from the name, 30
  degrees off.

The tag lives on the object only: orix' H5 file stores the name. For 157, 159,
162 and 163 a phase with a space group gets its group back from the space group
(:func:`frame_xmap`); a phase without one keeps the name, which is the other
setting, and loses the setting when it is saved. For 149, 151 and 153 the name
itself says the setting: orix reads ``"312"`` as ``D3y``, :func:`frame_xmap`
turns that into the marked group, and the operators survive the file. orix
warns that it sets the space group to None (the name of the point group derived
from it is ``"32"``), and the space group number is gone after the load.
"""
from __future__ import annotations

import copy
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
from orix.crystal_map import Phase, PhaseList
from orix.quaternion import Symmetry
from orix.quaternion.symmetry import (
    C2x,
    C2y,
    C2z,
    C3,
    Ci,
    Csx,
    Csy,
    Csz,
    D3y,
    S6,
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


class _MonoclinicSymmetry(Symmetry):
    """A monoclinic group that knows its unique axis.

    Two things orix gets wrong or loses for such a group. ``laue`` is rebuilt as
    a plain ``Symmetry`` without the axis. ``fundamental_sector`` switches on
    the name "2/m" and returns the normals {Z, Y}, which is a fundamental domain
    for the Z and Y axes but not for X (of the images of a direction, half have
    none inside it and half have two); the X-axis group needs {Z, X}.
    """

    @property
    def laue(self):
        return _TABLE[getattr(self, _TAG)]["2/m"]

    @property
    def fundamental_sector(self):
        if getattr(self, _TAG, None) == "a" and self.name == "2/m":
            from orix.vector import FundamentalSector, Vector3d

            fs = FundamentalSector(
                Vector3d([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])).flatten().unique()
            fs._center = None
            return fs
        return super().fundamental_sector


def _tagged(sym: Symmetry, name: str, axis: str) -> Symmetry:
    out = _MonoclinicSymmetry(sym)
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


@lru_cache(maxsize=None)
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


#: Space groups whose point group has its two-fold axes or mirror normals along
#: ``Y`` in the crystal frame, with the setting of that point group. orix returns
#: the setting with them along ``X`` for all of these (and for 150, 152, 154, 155,
#: 156, 158, 160, 161, 164-167, where that is right). ``phase_cell`` takes its set
#: of space groups with a per-reflection structure factor from here.
_TRIGONAL_Y_SETTING = {
    149: "312", 151: "312", 153: "312",
    157: "31m", 159: "31m",
    162: "-31m", 163: "-31m",
}
TRIGONAL_Y_SETTING_SPACE_GROUPS = frozenset(_TRIGONAL_Y_SETTING)

#: Attribute that marks a trigonal group built here: it names the setting.
_TRIGONAL_TAG = "_orienta_trigonal_setting"


class _TrigonalSymmetry(Symmetry):
    """A trigonal group that knows its setting (two-fold axes along ``Y``).

    orix names a trigonal group by its Laue class and builds ``laue`` from the
    name alone, and ``fundamental_sector`` turns the sector by 30 degrees for the
    names "3m" and "-3m" (right for mirrors with the normal along X, wrong for
    these groups). Both are answered here for the setting. A copy orix makes
    (slicing, ``Symmetry(...)``) is built without the tag; it falls back to
    orix' own answer instead of failing.
    """

    @property
    def laue(self):
        if getattr(self, _TRIGONAL_TAG, None) is None:
            return super().laue
        return _TRIGONAL["-31m"]

    @property
    def fundamental_sector(self):
        if getattr(self, _TRIGONAL_TAG, None) is None:
            return super().fundamental_sector
        # orix switches on the name and turns the normals by 30 degrees for
        # "3m" and "-3m", which would put the sector 30 degrees off for these
        # groups. The normals follow from the operators alone (the same rule orix
        # applies to a name it has no case for), so they are built here.
        from orix.vector import FundamentalSector, Vector3d

        vz = Vector3d.zvector()
        if any(vz.angle_with(self.outer(vz)) > np.pi / 2):
            n = vz
        else:
            n = Vector3d.empty()
        angle = 2 * np.pi * (1 + n.size) / self.size
        new_v = Vector3d.from_polar(
            azimuth=[np.pi / 2, angle - np.pi / 2], polar=[np.pi / 2, np.pi / 2])
        n = Vector3d(np.vstack([n.data, new_v.data]))
        fs = FundamentalSector(n).flatten().unique()
        fs._center = None
        return fs


def _tagged_trigonal(sym: Symmetry, name: str, setting: str) -> Symmetry:
    out = _TrigonalSymmetry(sym)
    out.improper = np.asarray(sym.improper, dtype=bool).copy()
    out.name = name
    setattr(out, _TRIGONAL_TAG, setting)
    return out


def _build_trigonal() -> dict:
    """The three groups of :data:`_TRIGONAL_Y_SETTING`, built once.

    Copies, named "312" (orix' name of ``D3y``), "3m" and "-3m" with the setting
    in the tag; orix' module-level objects are not touched.
    """
    return {
        "312": _tagged_trigonal(D3y, "312", "312"),
        "31m": _tagged_trigonal(Symmetry.from_generators(C3, Csy), "3m", "31m"),
        "-31m": _tagged_trigonal(Symmetry.from_generators(S6, Csy), "-3m", "-31m"),
    }


_TRIGONAL = _build_trigonal()


def _trigonal_from_text(point_group) -> Optional[Symmetry]:
    """The group of the text ``"312"``, ``"31m"`` or ``"-31m"``, else None."""
    if isinstance(point_group, str):
        return _TRIGONAL.get(point_group.strip())
    return None


def _is_plain_312(point_group) -> bool:
    """An unmarked orix object named "312" (``D3y``): the group of 149, 151, 153.

    Its ``laue`` has the Y operators but orix picks the sector from the name,
    which is the untagged answer 30 degrees off. The check is by operators, so a
    group that only carries the name is left alone.
    """
    if isinstance(point_group, str) or not isinstance(point_group, Symmetry):
        return False
    if getattr(point_group, _TRIGONAL_TAG, None) is not None:
        return False
    if getattr(point_group, "name", None) != "312" or point_group.size != 6:
        return False
    def rows(sym):
        q = np.asarray(sym.data, dtype=float).reshape(-1, 4)
        out = set()
        for r in q:
            first = r[np.argmax(np.abs(r) > 1e-6)]      # q and -q are one rotation
            out.add(tuple(np.round(-r if first < 0 else r, 6) + 0.0))
        return out

    return rows(point_group) == rows(_TRIGONAL["312"])


def point_group_key(point_group) -> str:
    """The text that names a point group unambiguously.

    ``"312"``, ``"31m"`` or ``"-31m"`` for a group of this module's trigonal
    settings, otherwise the group's ``name`` (or the text that was passed). It is
    for code that hands a group on as text: :func:`frame_symmetry` turns the key
    back into the same group. Empty for None.
    """
    if point_group is None:
        return ""
    setting = getattr(point_group, _TRIGONAL_TAG, None)
    if setting is not None:
        return setting
    return str(getattr(point_group, "name", point_group)).strip()


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
    The symmetry orix would return, except for

    * a monoclinic phase without an explicit unique axis, which gets the group
      whose two-fold axis (or mirror normal) lies along the axis its cell has:
      ``Y`` for the usual b-unique setting;
    * the trigonal space groups 149, 151, 153, 157, 159, 162 and 163 (and the
      texts ``"312"``, ``"31m"`` and ``"-31m"``), which get the group with the
      two-fold axes or mirror normals along ``Y``. Its name is "312", "3m" and
      "-3m"; :func:`point_group_key` gives the text that tells it apart.

    Phases of every other crystal system, and the names "32", "3m" and "-3m",
    are returned untouched. A group that already carries a setting is returned
    as it is.
    """
    if space_group is not None:
        cls = monoclinic_class_of_space_group(space_group)
        if cls is None:
            # Only the exact number: an alternative-setting number (1162, 2149)
            # is no space group of diffpy and goes to orix as it did before.
            setting = _TRIGONAL_Y_SETTING.get(int(space_group))
            if setting is not None:
                return _TRIGONAL[setting]
            return get_point_group(int(space_group))
        axis = (_space_group_axis(space_group) or _lattice_axis(lattice)
                or DEFAULT_UNIQUE_AXIS)
        return _TABLE[axis][cls]
    if point_group is None:
        return None
    trigonal = _trigonal_from_text(point_group)
    if trigonal is not None:
        return trigonal
    if _is_plain_312(point_group):
        return _TRIGONAL["312"]
    cls = _class_of(point_group)
    if cls is None:
        return point_group
    if getattr(point_group, _TAG, None) is not None:
        return point_group
    name = str(getattr(point_group, "name", point_group)).strip()
    axis = _EXPLICIT_AXIS.get(name) or _lattice_axis(lattice) or DEFAULT_UNIQUE_AXIS
    return _TABLE[axis][cls]


def phase_symmetry_arguments(point_group, space_group=None) -> dict:
    """Keyword arguments that make a ``Phase`` carry the symmetry of a master.

    A master pattern names its point group as text. For the seven trigonal
    space groups that text (``"312"``, ``"31m"``, ``"-31m"``) is the only place
    the setting is written, and orix stores a phase's point group by name, which
    would turn it into the other setting on the next load. Those get the space
    group instead, from which :class:`FramePhase` and :func:`frame_xmap` restore
    the group; every other phase gets the point group alone, as before.
    """
    try:
        sg = int(space_group)
    except (TypeError, ValueError):
        sg = None
    if (sg in TRIGONAL_Y_SETTING_SPACE_GROUPS
            and point_group_key(point_group) == _TRIGONAL_Y_SETTING[sg]):
        return {"space_group": sg}
    return {"point_group": point_group}


def copy_phase_arguments(phase) -> dict:
    """Keyword arguments for a :class:`FramePhase` that has the symmetry of ``phase``.

    For a phase of the seven trigonal space groups (149, 151, 153, 157, 159,
    162, 163) whatever kind it is: the space group alone, because a plain orix
    phase names such a group "32", "3m" or "-3m", and passing that next to the
    space group makes orix drop the space group and leaves the other setting.
    For every other phase ``point_group`` and ``space_group`` as the phase holds
    them, which is what callers passed before.
    """
    if has_trigonal_y_setting(phase):
        return phase_symmetry_arguments(
            phase_point_group(phase), phase.space_group.number)
    return {"point_group": phase.point_group,
            "space_group": getattr(phase, "space_group", None)}


def phase_point_group(phase):
    """The frame-aware point group of any orix phase (None if it has none).

    The same answer :class:`FramePhase` gives, for a phase that is not one:
    the space group decides, the cell angles and the indexers' convention
    follow. Use it where a plain orix phase from outside (a kikuchipy master
    pattern, say) is about to be reduced or sampled.
    """
    if phase is None:
        return None
    structure = getattr(phase, "structure", None)
    lattice = getattr(structure, "lattice", None) if structure is not None else None
    sg = getattr(phase, "space_group", None)
    if sg is not None:
        return frame_symmetry(space_group=sg.number, lattice=lattice)
    return frame_symmetry(getattr(phase, "_point_group", None), lattice=lattice)


def phase_point_group_key(phase) -> Optional[str]:
    """:func:`point_group_key` of :func:`phase_point_group` (None without a group).

    For code that passes a phase's point group on as text (the pseudo-symmetry
    helpers cache by it): ``.name`` is the same for the two settings of a
    trigonal class, this is not.
    """
    pg = phase_point_group(phase)
    return None if pg is None else point_group_key(pg)


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
        return phase_point_group(self)

    @point_group.setter
    def point_group(self, value) -> None:
        # orix' setter looks a text up among its own names and refuses "31m"
        # and "-31m". The three trigonal settings are turned into their group
        # first; every other value goes to orix as before.
        trigonal = _trigonal_from_text(value)
        Phase.point_group.fset(self, value if trigonal is None else trigonal)


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


def has_trigonal_y_setting(phase) -> bool:
    """True if ``phase`` has one of the space groups of :data:`TRIGONAL_Y_SETTING_SPACE_GROUPS`.

    These are the groups whose true point group differs from the one orix
    derives from the space group number. For every other phase the answer is
    False, however its point group was set.
    """
    number = getattr(getattr(phase, "space_group", None), "number", None)
    try:
        return int(number) in TRIGONAL_Y_SETTING_SPACE_GROUPS
    except (TypeError, ValueError):
        return False


def trigonal_frame_phase(phase):
    """``phase`` as a :class:`FramePhase` if its space group is one of the seven
    trigonal groups of :data:`TRIGONAL_Y_SETTING_SPACE_GROUPS`, else ``phase`` itself.

    For code that builds reflector families or symmetry classes from
    ``phase.point_group`` (diffsims' ``unique`` and ``symmetrise``) and may be
    handed a plain orix ``Phase`` from outside: for these seven groups orix'
    point group has the two-fold axes or mirror normals along X, the space group
    has them along Y, and the families of the plain phase hold reflections that
    are not equivalent. Every other phase is returned untouched (the same
    object), so nothing changes for the other 223 space groups.

    The result is a shallow copy of ``phase`` of class :class:`FramePhase`: it
    shares the structure (and every attribute) of the phase it came from, so the
    atoms, their coordinates and anything a caller normalises in place are the
    very same, and the reflector lists computed from it carry no rounding noise
    that a phase built as a ``FramePhase`` from the start would not carry.
    """
    if phase is None or isinstance(phase, FramePhase) or not has_trigonal_y_setting(phase):
        return phase
    out = copy.copy(phase)
    out.__class__ = FramePhase
    return out


def trigonal_frame_phase_list(phases):
    """``phases`` with every phase of the seven trigonal groups made a
    :class:`FramePhase` (:func:`trigonal_frame_phase`), same ids; the same
    object when no phase needs it."""
    if phases is None:
        return phases
    try:
        pairs = list(phases)
        items = {pid: trigonal_frame_phase(p) for pid, p in pairs}
    except TypeError:
        return phases
    if not items or all(items[pid] is p for pid, p in pairs):
        return phases
    return PhaseList(phases=items)


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
    "copy_phase_arguments",
    "FramePhase",
    "frame_phase",
    "frame_phase_list",
    "frame_symmetry",
    "frame_xmap",
    "has_trigonal_y_setting",
    "monoclinic_class_of_space_group",
    "phase_from_cif",
    "phase_point_group",
    "phase_point_group_key",
    "phase_symmetry_arguments",
    "point_group_key",
    "TRIGONAL_Y_SETTING_SPACE_GROUPS",
    "trigonal_frame_phase",
    "trigonal_frame_phase_list",
    "unique_axis",
]
