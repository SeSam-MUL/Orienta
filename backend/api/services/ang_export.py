"""The EDAX TSL ``.ang`` file for an indexing result, and the header facts that
go with it.

Why this is not just ``orix.io.save(xmap)``
-------------------------------------------
orix writes whatever the CrystalMap holds. The CrystalMaps the indexers return
are not export-ready:

* the spherical and consensus (EDS-prior, multi-phase Hough) maps carry PIXEL
  coordinates, so orix wrote ``XSTEP 1.0`` and x/y columns that count pixels;
* a phase built from a symmetry alone (the spherical maps) has no structure, so
  orix wrote its placeholder lattice ``1 1 1 90 90 90``.

Neither is a number anyone measured. This module builds the map the file is
written from out of what the result really knows (the step size, the grid, the
selection mask, and the file each phase came from), refuses when a lattice
cannot be found, and states in the header what a reader needs to know to use
the orientations: the Euler frame, where the grid sits in the original scan,
the acquisition geometry of the source scan, and whether the phase assignment
used the EDS chemistry prior.

The same helpers feed the ``.ctf`` of the batch exporter and the ``.h5``
exports, so every format says the same thing.
"""
from __future__ import annotations

import logging
import math
import os
import re
import types
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.api.services.load_choices import LoadChoiceError, measurement_groups
from backend.api.services.scan_geometry import ScanGeometry

logger = logging.getLogger(__name__)

#: orix's lattice for a phase that has no structure. No crystal has it: a cell
#: edge of 1 angstrom does not exist, so equality with it means "not known".
PLACEHOLDER_LATTICE = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)

#: Header keys written by this module (all ``# KEY: value`` comment lines, which
#: orix, MTEX and OIM skip or keep as free header text).
KEY_FRAME = "ORIENTA_EULER_FRAME"
KEY_ORIGIN = "ORIENTA_SCAN_ORIGIN"
KEY_ACQ = "ORIENTA_ACQUISITION"
KEY_ASSIGN = "ORIENTA_PHASE_ASSIGNMENT"
KEY_INDEXING = "ORIENTA_INDEXING"
KEY_PC = "ORIENTA_PATTERN_CENTRE_BRUKER"
KEY_NOTE = "ORIENTA_NOTE"
KEY_SCAN_GRID = "ORIENTA_SCAN_GRID"
KEY_CRYSTAL_FRAME = "ORIENTA_CRYSTAL_FRAME"
#: Machine-readable state of the three x/y/z-star lines when they hold no PC.
KEY_PC_STATE = "ORIENTA_PC"


class ExportRefused(ValueError):
    """The result lacks something a file must not be written without."""


class LatticeUnknown(ExportRefused):
    """A phase's lattice constants could not be found. Never a silent 1.000."""


class SymmetryUnknown(ExportRefused):
    """A phase has no point group, or one with no TSL symmetry code."""


# ---------------------------------------------------------------------------
# lattice constants
# ---------------------------------------------------------------------------

def is_placeholder_lattice(lattice) -> bool:
    """Is this orix's lattice for a phase that has none? Decided by the three edges alone: a
    phase made from a hexagonal or trigonal space group carries ``1 1 1 90 90 120`` (edges
    0.9999999999996 after the round trip), and no crystal has a cell edge of 1 angstrom."""
    try:
        edges = [float(a) for a in list(lattice)[:3]]
        return len(edges) == 3 and all(abs(a - 1.0) < 1e-9 for a in edges)
    except (TypeError, ValueError):
        return False


def to_angstrom(lattice, unit: str = "angstrom") -> Tuple[float, ...]:
    """``(a, b, c, alpha, beta, gamma)`` with the edges in angstrom.

    ``unit`` is the unit the SOURCE declares for the edges, never guessed from
    their size: a CIF read by diffpy / orix is in ``"angstrom"`` (Al is
    ``4.049``), a structure inside a kikuchipy master pattern or dictionary is
    in ``"nm"`` (Al is ``0.4049``). A result whose edges are implausible for
    the declared unit (no crystal has an edge below 1 angstrom) is refused
    instead of rescaled -- that is a unit mix-up, not a lattice.
    """
    a, b, c, al, be, ga = (float(v) for v in lattice)
    if unit == "nm":
        a, b, c = a * 10.0, b * 10.0, c * 10.0
    elif unit != "angstrom":
        raise ValueError(f"unknown length unit {unit!r}")
    if min(a, b, c) < 1.0:
        raise LatticeUnknown(
            f"lattice edges {a:.3g} {b:.3g} {c:.3g} angstrom (declared unit "
            f"{unit!r}) are not those of a crystal: the unit of this structure "
            "is wrong or the structure is not a real one")
    return (a, b, c, al, be, ga)


def lattice_from_structure(phase, unit: str = "angstrom") -> Optional[Tuple[float, ...]]:
    """The lattice a phase carries, in angstrom; ``None`` if it carries none.

    ``unit`` is the unit of the structure's edges (see :func:`to_angstrom`).
    """
    structure = getattr(phase, "structure", None)
    lat = getattr(structure, "lattice", None)
    if lat is None:
        return None
    try:
        raw = tuple(float(v) for v in lat.abcABG())
    except Exception:
        return None
    if is_placeholder_lattice(raw):
        return None
    return to_angstrom(raw, unit)


def lattice_from_sht(path) -> Tuple[float, ...]:
    """Lattice of the crystal an ``.sht`` master was simulated from (angstrom).

    The ``.sht`` binary stores the crystal data the simulation ran on, in
    nanometres and degrees.
    """
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master

    s = read_sht_master(str(path), device="cpu")
    a, b, c, al, be, ga = (float(v) for v in s.crystal_lattice)
    return (a * 10.0, b * 10.0, c * 10.0, al, be, ga)


def lattice_from_cif(path) -> Tuple[float, ...]:
    """Lattice of a CIF (angstrom)."""
    from orix.crystal_map import Phase

    from ebsd_utils import sanitize_cif

    phase = Phase.from_cif(sanitize_cif(str(path)))
    lat = lattice_from_structure(phase)
    if lat is None:
        raise LatticeUnknown(f"{Path(str(path)).name} carries no lattice")
    return lat


def lattice_from_master_h5(path) -> Optional[Tuple[float, ...]]:
    """Lattice stored in a kikuchipy master / dictionary file, if it has one."""
    import h5py

    found: List[Tuple[float, ...]] = []

    def visit(name, obj):
        if not found and name.endswith("structure/lattice/abcABG"):
            found.append(tuple(float(v) for v in np.asarray(obj).ravel()[:6]))

    try:
        with h5py.File(str(path), "r") as f:
            f.visititems(visit)
    except Exception:
        return None
    if not found or is_placeholder_lattice(found[0]):
        return None
    # kikuchipy writes the structure of a master / dictionary in nanometres
    return to_angstrom(found[0], "nm")


def lattice_from_phase_file(path) -> Optional[Tuple[float, ...]]:
    """Dispatch on the file a phase came from; ``None`` if it cannot be read."""
    if not path:
        return None
    p = Path(str(path))
    suffix = p.suffix.lower()
    try:
        if suffix == ".sht":
            return lattice_from_sht(p)
        if suffix == ".cif":
            return lattice_from_cif(p)
        if suffix in (".h5", ".hdf5"):
            return lattice_from_master_h5(p)
    except Exception as exc:
        logger.warning("lattice from %s failed: %s", p.name, exc)
    return None


def resolve_phase_lattice(phase, *, phase_file=None, recorded=None,
                          structure_unit: str = "angstrom"):
    """``(lattice_angstrom, source)`` for one phase, or raise ``LatticeUnknown``.

    Order: what the phase itself carries (a Hough map carries the CIF
    structure, in ``structure_unit``), what the run recorded (always angstrom),
    then the file the phase came from.
    """
    got = lattice_from_structure(phase, structure_unit)
    if got is not None:
        return got, "phase structure"
    if recorded is not None:
        rec = tuple(float(v) for v in recorded)
        if len(rec) == 6 and not is_placeholder_lattice(rec):
            return to_angstrom(rec, "angstrom"), "recorded with the run"
    got = lattice_from_phase_file(phase_file)
    if got is not None:
        return got, Path(str(phase_file)).name
    name = getattr(phase, "name", "") or "(unnamed)"
    where = (f" The phase file {Path(str(phase_file)).name!r} could not be read."
             if phase_file else " The result does not record which file the phase came from.")
    raise LatticeUnknown(
        f"The lattice constants of phase {name!r} are not known, so the file "
        f"would carry a placeholder (1 1 1 90 90 90).{where} Load the phase's "
        "CIF or .sht again, or re-run the indexing with the phase file "
        "available, then export again."
    )


# ---------------------------------------------------------------------------
# acquisition geometry of the source scan
# ---------------------------------------------------------------------------

def read_source_geometry(source_path, variant=None) -> Dict[str, object]:
    """Acquisition facts from an Oxford h5oina header, in degrees / mm / kV.

    Keys that the file does not have are absent. Returns ``{}`` for a source
    that is not an h5oina (EDAX files keep these elsewhere); nothing is guessed.

    ``variant`` (a ``LoadVariant``; None = the first measurement with a header, as before) names
    the measurement the result was indexed on: the header of that measurement is read, and
    ``header_path`` says which. A file without the measurement refuses with the coded
    ``LoadChoiceError``.
    """
    out: Dict[str, object] = {}
    if not source_path:
        return out
    try:
        import h5py

        if not h5py.is_hdf5(str(source_path)):
            return out
        with h5py.File(str(source_path), "r") as f:
            for entry in measurement_groups(f, variant):
                hdr = f.get(f"{entry}/EBSD/Header")
                if hdr is None or not hasattr(hdr, "keys"):
                    continue
                out["header_path"] = f"/{entry}/EBSD/Header"

                def get(key):
                    if key not in hdr:
                        return None
                    try:
                        arr = np.asarray(hdr[key][()], dtype=float).ravel()
                    except Exception:
                        return None
                    return arr if arr.size else None

                def deg(key):
                    v = get(key)
                    return None if v is None else np.degrees(v)

                # The three values MTEX and Aztec users ask for, verbatim
                # (radians, exactly as the h5oina header stores them).
                raw: Dict[str, object] = {}
                for key in ("Scanning Rotation Angle", "Specimen Orientation Euler",
                            "Tilt Angle"):
                    v = get(key)
                    if v is not None:
                        raw[key] = float(v[0]) if v.size == 1 else [float(x) for x in v]
                if raw:
                    out["h5oina_header"] = raw
                scan_rot = deg("Scanning Rotation Angle")
                if scan_rot is not None:
                    out["scanning_rotation_angle_deg"] = float(scan_rot[0])
                tilt = deg("Tilt Angle")
                if tilt is not None:
                    out["sample_tilt_deg"] = float(tilt[0])
                spec = deg("Specimen Orientation Euler")
                if spec is not None and spec.size >= 3:
                    out["specimen_orientation_euler_deg"] = [float(v) for v in spec[:3]]
                det = deg("Detector Orientation Euler")
                if det is not None and det.size >= 3:
                    out["detector_orientation_euler_deg"] = [float(v) for v in det[:3]]
                wd = get("Working Distance")
                if wd is not None:
                    out["working_distance_mm"] = float(wd[0])
                kv = get("Beam Voltage")
                if kv is not None:
                    out["beam_voltage_kv"] = float(kv[0])
                break
    except LoadChoiceError:
        raise
    except Exception as exc:
        logger.warning("could not read the acquisition geometry of %s: %s",
                       source_path, exc)
    return out


def source_label(geometry: Optional[Dict[str, object]]) -> str:
    """Where the geometry came from, without claiming a vendor format."""
    path = (geometry or {}).get("header_path")
    return f"source file header {path}" if path else ""


def acquisition_line(geometry: Dict[str, object],
                     detector_tilt_deg: Optional[float] = None) -> Optional[str]:
    """One ``key=value`` line of acquisition facts, or ``None`` if there are none."""
    parts: List[str] = []
    g = geometry or {}
    if "scanning_rotation_angle_deg" in g:
        parts.append(f"scanning_rotation_angle_deg={g['scanning_rotation_angle_deg']:.3f}")
    if "sample_tilt_deg" in g:
        parts.append(f"sample_tilt_deg={g['sample_tilt_deg']:.3f}")
    if detector_tilt_deg is not None:
        parts.append(f"detector_tilt_deg={float(detector_tilt_deg):.3f}")
    if "specimen_orientation_euler_deg" in g:
        v = g["specimen_orientation_euler_deg"]
        parts.append("specimen_orientation_euler_deg=" + ",".join(f"{x:.3f}" for x in v))
    if "detector_orientation_euler_deg" in g:
        v = g["detector_orientation_euler_deg"]
        parts.append("detector_orientation_euler_deg=" + ",".join(f"{x:.3f}" for x in v))
    if "beam_voltage_kv" in g:
        parts.append(f"beam_voltage_kv={g['beam_voltage_kv']:.1f}")
    return " ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# symmetry, space group and crystal frame
# ---------------------------------------------------------------------------

#: TSL "Symmetry" code per Laue group: the digits of the Laue group's rotation
#: group (432 -> 43, 622 -> 62, 222 -> 22, ...), the eleven codes every EDAX
#: .ang format has (see MTEX's documentation of ``TSL2pointGroup``). Monoclinic
#: is 20 (what TSL writes for 2/m; MTEX's laueGroups treats 2 as an older alias
#: of it). orix reads 20 as its rotation group 121 (2-fold axis along y, the
#: b-unique setting of e.g. Fe4Al13) and 2 as 2/m with the axis along z, so a
#: phase whose unique axis is c (gamma != 90, alpha = beta = 90) is written 2.
_LAUE_TO_TSL = {
    "-1": "1",
    "2/m": "20", "112/m": "20",
    "mmm": "22",
    "4/m": "4", "4/mmm": "42",
    "-3": "3", "-3m": "32", "-3m1": "32", "-31m": "32",
    "6/m": "6", "6/mmm": "62",
    "m-3": "23", "m-3m": "43",
}

#: Said once, used in every file that holds orientations of non-cubic phases.
CRYSTAL_FRAME = (
    "X||a, Z||c* (Y completes a right-handed set): the crystal frame of orix "
    "and EMsoft, in which every orientation Orienta computes (Hough, "
    "dictionary and spherical indexing alike) is expressed. MTEX's default "
    "crystal frame is X||a*, Z||c, so a crystalSymmetry built for these "
    "orientations needs 'X||a','Z||c*'. The two conventions give the same "
    "frame for cubic, tetragonal and orthorhombic phases and differ for "
    "hexagonal, trigonal, monoclinic and triclinic ones. Euler angles copied "
    "from an Aztec solution are not part of any Orienta result and are never "
    "converted."
)


#: The Laue groups of the crystal systems whose frame differs between Orienta (X||a, Z||c*) and
#: MTEX's default (X||a*, Z||c): triclinic, monoclinic, trigonal and hexagonal. The other systems
#: have the same frame in both, and a file of those phases states nothing.
_FRAME_LAUE_GROUPS = frozenset({
    "-1", "2/m", "112/m", "-3", "-3m", "-3m1", "-31m", "6/m", "6/mmm"})

#: What a ``.ang`` or ``.ctf`` of a non-cubic phase says about its Euler angles: the file has no field
#: for the crystal frame, and Orienta converts nothing to a vendor's own setting.
CRYSTAL_FRAME_STATEMENT = (
    "Crystal frame of the Euler angles of non-cubic phases: X||a, Z||c* (Y completes a right-handed "
    "set) in every system; not converted to any vendor setting, and this format has no field for it.")

#: The operators that lie along Y (not along X, as in orix' own groups) for the three point groups of
#: :data:`crystal_symmetry.TRIGONAL_Y_SETTING_SPACE_GROUPS`.
_Y_SETTING_OPERATORS = {
    "312": "two-fold axes",
    "31m": "mirror normals",
    "-31m": "two-fold axes and mirror normals",
}


def _header_safe_name(text) -> str:
    """A phase name as plain text for a header comment: letters, digits, space and ``_ . + ( ) / -``.

    White space becomes one space and every other character ``?``. A name is free text, and orix
    reads EVERY header line of a ``.ang`` for ``# Phase``, ``# MaterialName``, ``# Formula``,
    ``# Symmetry`` and ``# LatticeConstants`` (anywhere in the line) and for the vendor footprint
    ``Column names: phi1, Phi, phi2``; without ``#`` and ``:`` a name cannot form one of them."""
    return re.sub(r"[^A-Za-z0-9 _.+()/-]", "?", " ".join(str(text).split()))


def map_phases(xmap) -> list:
    """The phases ``xmap`` holds, without the "not indexed" one, in the order of the file."""
    return [phase for pid, phase in xmap.phases if int(pid) >= 0]


def crystal_frame_texts(phases, *, space_group: bool = True) -> List[str]:
    """What a file with these ``phases`` states about the crystal frame of its Euler angles.

    Nothing (``[]``) unless a phase is triclinic, monoclinic, trigonal or hexagonal: for cubic,
    tetragonal and orthorhombic phases Orienta's frame and MTEX's default are the same. Then
    :data:`CRYSTAL_FRAME_STATEMENT`, and for each phase of the setting with its two-fold axes or
    mirror normals along Y (the space groups 149, 151, 153, 157, 159, 162, 163) one more text with
    its point group (and, for a ``.ang``, its space group): the TSL symmetry code 32 of a ``.ang``
    and the Laue number of a ``.ctf`` are the same for both settings and cannot say it.
    The texts must not hold what orix' header reader searches every line for (``# Symmetry``,
    ``# Phase``, ``# MaterialName``, ``# Formula``, ``# LatticeConstants``): they contain no ``#``.
    ``phases`` are orix phases; a plain ``Phase`` is read through ``phase_point_group``, which takes
    the space group into account, so the phase handed in must still hold it."""
    from crystal_symmetry import phase_point_group, point_group_key

    frame = False
    y_texts: List[str] = []
    for phase in phases or ():
        group = phase_point_group(phase)
        if group is None:
            continue
        laue = getattr(getattr(group, "laue", None), "name", None)
        if laue not in _FRAME_LAUE_GROUPS:
            continue
        frame = True
        key = point_group_key(group)
        operators = _Y_SETTING_OPERATORS.get(key)
        if operators is None:
            continue
        number = getattr(getattr(phase, "space_group", None), "number", None)
        where = f"space group {int(number)}, " if space_group and number is not None else ""
        cannot = "the TSL Symmetry code 32" if space_group else "the Laue class"
        y_texts.append(
            f"Phase {_header_safe_name(getattr(phase, 'name', ''))}: {where}point group {key}: "
            f"{operators} lie along Y, which {cannot} cannot state.")
    return [CRYSTAL_FRAME_STATEMENT] + y_texts if frame else []


#: ``/Indexing@orientation_reference_frame`` of the .h5 exports.
FRAME_LABEL_VENDOR = (
    "vendor_stored: the Euler angles as the source system's own solution holds "
    "them; header values such as the Scanning Rotation Angle are not applied "
    "(see /Acquisition)")
FRAME_LABEL_NATIVE = "native (EMsoft/kikuchipy common)"


def tsl_symmetry_code(point_group, lattice=None) -> str:
    """The TSL ``# Symmetry`` code of a phase. Raises when it has none.

    ``lattice`` (a, b, c, alpha, beta, gamma) only matters for monoclinic
    phases: 20 (unique axis b) unless the cell is c-unique, then 2.
    """
    if point_group is None:
        raise ValueError(
            "the phase has no point group, so no TSL symmetry code can be "
            "written for it")
    try:
        laue = point_group.laue.name
    except Exception:
        laue = None
    code = _LAUE_TO_TSL.get(laue)
    if code is None:
        raise ValueError(
            f"point group {getattr(point_group, 'name', point_group)!r} (Laue "
            f"group {laue!r}) has no TSL symmetry code in this exporter's table")
    if code == "20" and lattice is not None:
        _a, _b, _c, al, be, ga = (float(v) for v in lattice)
        if abs(ga - 90.0) > 1e-3 and abs(al - 90.0) < 1e-3 and abs(be - 90.0) < 1e-3:
            code = "2"
    return code


def space_group_from_sht(path) -> Optional[int]:
    """Space-group number stored in an ``.sht``."""
    from backend.spherical_gpu.pipeline.sht_io import read_sht_master

    return int(read_sht_master(str(path), device="cpu").space_group)


def space_group_symbol(number) -> str:
    try:
        from diffpy.structure.spacegroups import GetSpaceGroup

        return str(GetSpaceGroup(int(number)).short_name).replace(" ", "")
    except Exception:
        return ""


def phase_table_attrs(phase, *, sht_path=None, phase_file=None,
                      recorded_lattice=None, recorded_space_group=None,
                      structure_unit: str = "angstrom") -> Dict[str, object]:
    """Attributes for ``/Indexing/Phases/<k>`` beyond name and point group.

    Space group (number and symbol), lattice constants (a, b, c in angstrom,
    alpha, beta, gamma in degrees) and the crystal frame the stored Euler
    angles refer to. A value that cannot be found is left out and said so
    (``lattice_status``); nothing is invented.
    """
    out: Dict[str, object] = {}
    sg = None
    try:
        sgo = getattr(phase, "space_group", None)
        if sgo is not None:
            sg = int(getattr(sgo, "number", sgo))
    except Exception:
        sg = None
    if sg is None and recorded_space_group is not None:
        sg = int(recorded_space_group)
    if sg is None and sht_path:
        try:
            sg = space_group_from_sht(sht_path)
        except Exception as exc:
            logger.warning("space group from %s failed: %s", sht_path, exc)
    if sg is not None:
        out["space_group"] = int(sg)
        sym = space_group_symbol(sg)
        if sym:
            out["space_group_symbol"] = sym
    try:
        lattice, source = resolve_phase_lattice(
            phase, phase_file=phase_file or sht_path, recorded=recorded_lattice,
            structure_unit=structure_unit)
        out["lattice_constants"] = np.asarray(lattice, dtype=float)
        out["lattice_length_unit"] = "angstrom"
        out["lattice_source"] = source
    except LatticeUnknown as exc:
        logger.warning("phase table: %s", exc)
        out["lattice_status"] = "unknown"
    out["crystal_reference_frame"] = CRYSTAL_FRAME
    return out


# ---------------------------------------------------------------------------
# provenance of the result
# ---------------------------------------------------------------------------

def assignment_provenance(steps: Optional[Sequence[dict]]) -> Dict[str, object]:
    """Did the EDS influence which phase a pixel got?

    ``steps`` is the citation trail of the result (``provenance.get_steps``).
    ``None`` means "a batch checkpoint": the batch gives each pixel the phase
    with the highest confidence index and never reads EDS.
    """
    steps = list(steps) if steps is not None else []
    eds = [s for s in steps if str(s.get("key", "")).startswith("eds.")]
    prior = next((s for s in eds if s.get("key") == "eds.chemistry_prior"), None)
    rescue = next((s for s in eds if s.get("key") == "eds.particle_rescue"), None)
    bits: List[str] = []
    if prior is not None:
        p = prior.get("params") or {}
        strengths = p.get("strength_by_phase") or {}
        sv = ", ".join(f"{k} {float(v):g}" for k, v in strengths.items())
        n = p.get("n_adjusted")
        detail = "; ".join(x for x in (
            f"strengths {sv}" if sv else "",
            f"{int(n)} px adjusted" if n is not None else "") if x)
        bits.append("EDS chemistry prior: yes" + (f" ({detail})" if detail else ""))
    else:
        bits.append("EDS chemistry prior: no")
    if rescue is not None:
        n = (rescue.get("params") or {}).get("n_changed")
        bits.append("EDS particle rescue: yes"
                    + (f" ({int(n)} px reassigned)" if n is not None else ""))
    text = ". ".join(bits) + "."
    if eds:
        text += (" The phase of each pixel was decided with the EDS composition, "
                 "so the phase fractions are not independent of the EDS data.")
    else:
        text += " The phase of each pixel was decided by the diffraction pattern alone."
    return {"eds_used": bool(eds), "eds_prior": prior is not None, "text": text}


def euler_frame_text(vendor: str, *, unit: str = "radians",
                     angles: str = "phi1, Phi, phi2",
                     first: str = "phi1") -> List[str]:
    """Plain-words statement of the Euler frame of a ``.ang`` / ``.ctf``."""
    v = (vendor or "").lower()
    lines = [
        f"{angles} are Bunge Euler angles in {unit} in the EDAX TSL sample "
        "frame, the frame orix and kikuchipy use; orix reads this file without "
        "any conversion."
    ]
    if v in ("oxford", "bruker"):
        lines.append(
            "The scan came from an Oxford Instruments / Bruker system. The .h5 "
            "exports of this result hold the Euler angles in the Oxford/Aztec "
            "frame, which is rotated by 90 degrees about the sample normal: "
            f"{first}(.h5) = {first}(this file) - 90 degrees (mod 360); the "
            f"other two angles are identical. Subtract 90 degrees from {first} "
            "here to compare with Aztec or with the .h5.")
    elif v in ("edax", "tsl", "ametek"):
        lines.append(
            "The scan came from an EDAX system, whose own frame this is: the .h5 "
            "exports of this result hold the same Euler angles as this file.")
    else:
        lines.append(
            "The source system was not recorded; the .h5 exports of this result "
            "hold the Euler angles in the frame of the source system, which for "
            f"Oxford/Bruker differs from this file by 90 degrees in {first}.")
    return lines


def scan_origin_text(scan_row_offset: int, scan_col_offset: int,
                     scan_shape, step_um: float, step_y_um: Optional[float] = None,
                     resampled: bool = False,
                     geometry: Optional[ScanGeometry] = None) -> str:
    """Where the first pixel of the file sits in the original scan.

    ``step_um`` is the step along x. ``step_y_um`` is the distance between rows when it is not the
    same (a hexagonal scan: ``step_x * sqrt(3) / 2``); without it rows are ``step_um`` apart.

    ``resampled`` is for a file whose grid is not the grid of the scan (a hexagonal scan resampled to a
    square one): row and column of the original no longer name its first pixel, so the sentence gives
    the place in um only. ``step_um`` and ``step_y_um`` are then the steps of the original scan.

    ``geometry`` is the hexagonal geometry of the file's own grid (the crop's, as the result stamps
    it). A crop that starts on an odd row of the original has the offset of its odd rows reversed
    (``ScanGeometry.cropped``), and its first pixel lies half a step to the side of ``column * step``:
    the position in um then includes that half step. A file whose odd rows lie to the left
    (offset -0.5) has its x raised by half a step (``hex_x_lift``), so x = 0 of the file is half a step
    left of its first pixel; the sentence says so. Without ``geometry`` the positions are
    ``column * step_um`` and ``row * row step``, as for a square scan.
    """
    r0, c0 = int(scan_row_offset or 0), int(scan_col_offset or 0)
    row_step = step_um if step_y_um is None else step_y_um
    # x of the first pixel in the original scan: the odd rows of the original are shifted, and the
    # file's geometry has the shift of the original unless the crop starts on an odd row
    x_first = c0 * step_um
    if geometry is not None and geometry.is_hex and r0 % 2 == 1:
        x_first += -float(geometry.odd_row_shift) * step_um
    if resampled:
        tail = ("The scan was resampled from a hexagonal grid, so no row or column of the "
                "original applies.")
        if r0 == 0 and c0 == 0:
            return ("the first pixel of this file is at the origin of the scan (x 0.000 um, "
                    f"y 0.000 um; not a crop). {tail}")
        return (f"the first pixel of this file lies at x {x_first:.3f} um, "
                f"y {r0 * row_step:.3f} um of the original scan. Coordinates in this file start "
                f"at 0, at that pixel. {tail}")
    if r0 == 0 and c0 == 0:
        return ("the first pixel of this file is row 0, column 0 of the scan "
                "(not a crop).")
    shape = ""
    if scan_shape is not None:
        try:
            shape = f" {int(scan_shape[0])} x {int(scan_shape[1])}"
        except Exception:
            shape = ""
    lift = hex_x_lift(geometry) if geometry is not None and geometry.is_hex else 0.0
    if lift > 0.0:
        return (
            f"the first pixel of this file is row {r0}, column {c0} of the original"
            f"{shape} scan (0-based); it lies at x {x_first:.3f} um, y {r0 * row_step:.3f} um of "
            f"the original scan. The odd rows of this file lie half a step to the left, so x in "
            f"this file is raised by half a step ({lift:.3f} um): its first pixel is at x "
            f"{lift:.3f} um, y 0.000 um, and the origin of its coordinates lies at "
            f"x {x_first - lift:.3f} um, y {r0 * row_step:.3f} um of the original scan."
        )
    return (
        f"the first pixel of this file is row {r0}, column {c0} of the original"
        f"{shape} scan (0-based). Coordinates in this file start at 0; the "
        f"origin lies at x {x_first:.3f} um, y {r0 * row_step:.3f} um of the "
        "original scan."
    )


# ---------------------------------------------------------------------------
# pattern centre in the EDAX convention
# ---------------------------------------------------------------------------

def pc_tsl_from_bruker(pc, detector_shape) -> Optional[Tuple[float, float, float]]:
    """PC in the EDAX TSL convention from a Bruker-convention PC.

    kikuchipy's PC is Bruker; the TSL one is NOT the same numbers:
    ``y_T = 1 - y_B`` and ``z_T = z_B * nrows / min(nrows, ncols)`` (the
    installed kikuchipy's ``EBSDDetector.pc_tsl``, which agrees with
    PyEBSDIndex's EDAX branch). Needs the detector shape; returns ``None``
    without it.
    """
    if pc is None or detector_shape is None:
        return None
    try:
        import kikuchipy as kp

        pc3 = np.asarray(pc, dtype=float).ravel()[:3]
        if pc3.size < 3:
            return None
        det = kp.detectors.EBSDDetector(
            shape=tuple(int(v) for v in detector_shape), pc=pc3,
            convention="bruker")
        x, y, z = det.pc_tsl()[0]
        return float(x), float(y), float(z)
    except Exception as exc:
        logger.warning("could not convert the pattern centre to TSL: %s", exc)
        return None


# ---------------------------------------------------------------------------
# building the map the file is written from
# ---------------------------------------------------------------------------

def _best_match_column(values, n_points: int) -> np.ndarray:
    """One value per map point from a per-point property.

    A property is ``(n_points,)`` for one match per point and
    ``(n_points, k)`` when the indexer kept ``k`` matches (best first). The
    export keeps the best match, so this returns column 0.
    """
    arr = np.asarray(values)
    if arr.size == 0 or arr.size % n_points:
        raise ValueError(
            f"property with {arr.size} values does not fit {n_points} map points")
    return arr.reshape(n_points, -1)[:, 0]


def build_export_xmap(
    xmap,
    original_shape,
    selection_mask,
    step_um: float,
    confidence_rows=None,
    *,
    phase_files: Optional[Dict[int, str]] = None,
    recorded_lattices: Optional[Dict[int, Sequence[float]]] = None,
    structure_unit: str = "angstrom",
    step_y_um: Optional[float] = None,
):
    """A CrystalMap on the full grid, in micrometres, with real lattices.

    Rotations stay in the frame the indexer produced (the EDAX TSL / kikuchipy
    frame); only coordinates, phase table and ids are rebuilt. Returns
    ``(xmap, tsl_codes)`` where ``tsl_codes`` maps the phase number written in
    the file to its TSL ``# Symmetry`` code.

    The map stays an index-space rectangle: x = column * ``step_um``, y = row *
    ``step_y_um`` (``step_um`` when not given). orix reads its ``XSTEP`` and
    ``YSTEP`` from it. The positions of a hexagonal scan are not in this map;
    :func:`write_ang` takes them from the scan geometry.
    """
    from diffpy.structure import Lattice, Structure
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation

    from crystal_symmetry import FramePhase, copy_phase_arguments
    from backend.api.services.result_exporter import (
        confidence_rows_for_export,
        map_phase_ids_for_export,
        place_rows_on_grid,
        xmap_phase_write_table,
    )

    if not (step_um and step_um > 0):
        raise ValueError(f"step size must be positive, got {step_um!r}")
    if step_y_um is not None and not step_y_um > 0:
        raise ValueError(f"row step must be positive, got {step_y_um!r}")
    row_step = float(step_um if step_y_um is None else step_y_um)
    n_rows, n_cols = int(original_shape[0]), int(original_shape[1])
    n = n_rows * n_cols

    quats = np.asarray(xmap.rotations.data, dtype=float)
    if quats.ndim == 3:                       # top-N matches: keep the best
        quats = quats[:, 0, :]
    # Points on the map, not rotations: with several matches per point
    # (``rotations.shape == (n_points, k)``) the rotations outnumber the points,
    # and a ``Rotation`` has no ``len()``. Every per-point property has one row
    # per point.
    n_points = int(quats.shape[0])
    flat_q = np.zeros((n, 4))
    flat_q[:, 0] = 1.0
    placed_q = place_rows_on_grid(quats, original_shape, selection_mask,
                                  fill=0.0).reshape(n, 4)
    mask = place_rows_on_grid(np.ones(quats.shape[0], dtype=bool), original_shape,
                              selection_mask, fill=False, dtype=bool).reshape(n)
    flat_q[mask] = placed_q[mask]

    table, id_map = xmap_phase_write_table(xmap)
    written = place_rows_on_grid(
        map_phase_ids_for_export(xmap.phase_id, xmap), original_shape,
        selection_mask, fill=0, dtype=np.uint8).reshape(n).astype(int)
    pid = written - 1                          # 0-based, -1 = not indexed

    written_to_actual = {w: a for a, w in id_map.items()}
    phases = []
    for written_id, phase in table:
        actual = written_to_actual.get(written_id)
        lattice, source = resolve_phase_lattice(
            phase,
            phase_file=(phase_files or {}).get(actual),
            recorded=(recorded_lattices or {}).get(actual),
            structure_unit=structure_unit,
        )
        logger.info("ang export: phase %s lattice from %s: %s",
                    getattr(phase, "name", "?"), source, lattice)
        name = getattr(phase, "name", "") or f"phase{written_id}"
        structure = Structure(title=name, lattice=Lattice(*lattice))
        # the space group stays with the phase: a .ctf names it, and the point group follows from it
        phases.append(FramePhase(
            name=name, structure=structure, **copy_phase_arguments(phase)))
    phase_list = PhaseList(phases)

    ys, xs = np.mgrid[0:n_rows, 0:n_cols]
    prop = {}
    if confidence_rows is not None:
        ci = place_rows_on_grid(
            confidence_rows_for_export(confidence_rows, original_shape),
            original_shape, selection_mask, fill=0.0).reshape(n)
        prop["ci"] = np.asarray(ci, dtype=np.float32)
    else:
        for key in ("ci", "scores"):
            if key in getattr(xmap, "prop", {}):
                col = _best_match_column(xmap.prop[key], n_points)
                prop["ci"] = np.asarray(place_rows_on_grid(
                    col, original_shape, selection_mask, fill=0.0),
                    dtype=np.float32).reshape(n)
                break
    for key in ("fit", "patternfit"):
        if key in getattr(xmap, "prop", {}):
            try:
                col = _best_match_column(xmap.prop[key], n_points)
                prop["fit"] = np.asarray(place_rows_on_grid(
                    col, original_shape, selection_mask, fill=0.0),
                    dtype=np.float32).reshape(n)
            except ValueError:
                pass
            break

    xmap_out = CrystalMap(
        rotations=Rotation(flat_q),
        phase_id=pid,
        x=(xs.ravel() * float(step_um)).astype(float),
        y=(ys.ravel() * row_step).astype(float),
        phase_list=phase_list,
        prop=prop,
        scan_unit="um",
    )
    # numbered by the map that is written: the phase list above may name a phase no cell takes,
    # which orix leaves out of the map, and the phases after it move up by one
    from backend.api.services.result_exporter import _tsl_codes_of_map
    return xmap_out, _tsl_codes_of_map(xmap_out)


# ---------------------------------------------------------------------------
# header editing
# ---------------------------------------------------------------------------

def _header_field(name: str, value: float) -> str:
    # orix writes "TEM_PIXperUM           1.000000": key padded to 23 columns
    return f"# {name:<23}{value:.6f}"


def edit_ang_header(path, *, replace: Optional[Dict[str, str]] = None,
                    comments: Optional[Sequence[str]] = None,
                    symmetry_codes: Optional[Dict[int, str]] = None) -> None:
    """Replace header lines by prefix and add ``# KEY: text`` comment lines.

    ``replace`` maps a line prefix (``"# x-star"``) to the whole new line, or to a list
    of lines that replace it (the last one stays where the old line was).
    ``comments`` are inserted just above the ``# Column names`` line, i.e. at the
    end of the header. Line endings of the file are kept. ``symmetry_codes``
    maps a phase number to the TSL code its ``# Symmetry`` line is rewritten to
    (orix writes the name of the rotation group, e.g. ``112``, which is not a
    TSL code).
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(eol)
    out: List[str] = []
    in_header = True
    insert_at = None
    phase_no = None
    for ln in lines:
        if in_header and ln.strip() and not ln.lstrip().startswith("#"):
            in_header = False
        if in_header:
            m = re.match(r"^#\s*Phase\s+(\d+)\s*$", ln)
            if m:
                phase_no = int(m.group(1))
            elif (symmetry_codes and phase_no in symmetry_codes
                  and re.match(r"^#\s*Symmetry", ln)):
                ln = f"# Symmetry    {symmetry_codes[phase_no]}"
            for prefix, new in (replace or {}).items():
                if ln.startswith(prefix):
                    if isinstance(new, (list, tuple)):
                        out.extend(new[:-1])
                        ln = new[-1]
                    else:
                        ln = new
                    break
            if ln.lstrip().startswith("# Column names") and insert_at is None:
                insert_at = len(out)
        out.append(ln)
    if comments:
        if insert_at is None:                      # no Column-names line: end of header
            insert_at = 0
            for i, ln in enumerate(out):
                if ln.lstrip().startswith("#"):
                    insert_at = i + 1
        out[insert_at:insert_at] = list(comments)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(eol.join(out))


def header_comments(
    *,
    method: Optional[str],
    vendor: str,
    provenance: Dict[str, object],
    scan: Optional[Dict[str, object]] = None,
    step_um: float,
    geometry: Optional[Dict[str, object]] = None,
    detector_tilt_deg: Optional[float] = None,
    pc_bruker=None,
    notes: Sequence[str] = (),
    grid: Optional[ScanGeometry] = None,
    resampled_from: Optional[ScanGeometry] = None,
    phases: Optional[Sequence] = None,
) -> List[str]:
    """The ``# KEY: text`` lines every ``.ang`` of this application carries.

    ``phases`` are the orix phases of the file: for a triclinic, monoclinic, trigonal or hexagonal one
    the header states the crystal frame of the Euler angles (:func:`crystal_frame_texts`); a file of
    cubic, tetragonal and orthorhombic phases, or without ``phases``, is written as before.

    ``grid`` is the geometry of a hexagonal scan (``None`` for a square one): the file then
    states its grid and the points where it differs from OIM's own hexagonal files.
    ``resampled_from`` is the hexagonal geometry of a scan that this file holds resampled to a
    square grid: the scan origin is then stated in um (``scan_origin_text``), and the caller adds
    the note of the resampling (``resampling_note``) to ``notes``. A file is one or the other, so
    both together raise ``ValueError``."""
    hexagonal = grid is not None and grid.is_hex
    if hexagonal and resampled_from is not None:
        raise ValueError("a file is either a hexagonal one or a resampled one, not both")
    lines: List[str] = []
    if method:
        lines.append(f"# {KEY_INDEXING}: method={method}; software=Orienta")
    for t in euler_frame_text(vendor):
        lines.append(f"# {KEY_FRAME}: {t}")
    for t in crystal_frame_texts(phases):
        lines.append(f"# {KEY_CRYSTAL_FRAME}: {t}")
    if scan is None and hexagonal and hex_x_lift(grid) > 0.0:
        lines.append(f"# {KEY_ORIGIN}: not recorded by this export; x in this file is raised by half a "
                     f"step ({hex_x_lift(grid):.3f} um) because the odd rows lie half a step to the left: "
                     "the first pixel of the file is at that x, y 0.")
    elif scan is None:
        lines.append(f"# {KEY_ORIGIN}: not recorded by this export; coordinates "
                     "start at 0 at the first pixel of the file.")
    elif resampled_from is not None:
        lines.append("# %s: %s" % (KEY_ORIGIN, scan_origin_text(
            scan.get("scan_row_offset", 0), scan.get("scan_col_offset", 0),
            scan.get("scan_shape"), resampled_from.step_x,
            step_y_um=resampled_from.step_y, resampled=True, geometry=resampled_from)))
    else:
        lines.append("# %s: %s" % (KEY_ORIGIN, scan_origin_text(
            scan.get("scan_row_offset", 0), scan.get("scan_col_offset", 0),
            scan.get("scan_shape"), step_um,
            step_y_um=grid.step_y if hexagonal else None,
            geometry=grid if hexagonal else None)))
    if hexagonal:
        lines.append(f"# {KEY_SCAN_GRID}: {hex_grid_text(grid)}")
        notes = list(hex_grid_notes(grid)) + list(notes)
    mtex = mtex_line(vendor, geometry)
    if mtex:
        lines.append(f"# ORIENTA_MTEX: {mtex}")
    acq = acquisition_line(geometry or {}, detector_tilt_deg)
    if acq:
        lines.append(f"# {KEY_ACQ}: {acq}")
    if pc_bruker is not None:
        v = np.asarray(pc_bruker, dtype=float).ravel()[:3]
        lines.append(f"# {KEY_PC}: x={v[0]:.6f} y={v[1]:.6f} z={v[2]:.6f}")
    lines.append(f"# {KEY_ASSIGN}: {provenance['text']}")
    for n in notes:
        lines.append(f"# {KEY_NOTE}: {n}")
    return lines


def ctf_frame_text(vendor: str) -> str:
    """Plain-words statement of the Euler frame of a ``.ctf``.

    A Channel Text File is an Oxford format (MTEX applies one fixed correction
    to every .ctf), so it holds the angles the .h5 holds: the source system's
    frame, not the TSL frame of the .ang.
    """
    v = (vendor or "").lower()
    if v in ("oxford", "bruker"):
        return ("Euler1, Euler2, Euler3 are Bunge Euler angles in degrees in the "
                "Oxford/Aztec frame of the source scan, the same Euler angles as "
                "the .h5 exports of this result. The .ang of this result holds the "
                "EDAX TSL frame instead (orix / kikuchipy): Euler1(.ang) = "
                "Euler1(this file) + 90 degrees (mod 360).")
    if v in ("edax", "tsl", "ametek"):
        return ("Euler1, Euler2, Euler3 are Bunge Euler angles in degrees in the "
                "EDAX TSL frame of the source scan (the frame of orix and "
                "kikuchipy), the same Euler angles as the .h5 and .ang exports of "
                "this result.")
    return ("Euler1, Euler2, Euler3 are Bunge Euler angles in degrees in the "
            "EDAX TSL frame (orix / kikuchipy); the source system was not "
            "recorded, so no conversion to its own frame was made.")


def mtex_line(vendor: str, geometry: Optional[Dict[str, object]]) -> Optional[str]:
    """The MTEX correction that reproduces MTEX's own h5oina import of the scan.

    Derived from MTEX's source, not run in MATLAB: MTEX's h5oina import turns the
    header's Scanning Rotation Angle S into an EulerCorrection (rotation about
    the surface normal by S) on the raw Aztec angles; the Aztec angles are the
    native (TSL-frame) ones rotated by -90 degrees about the normal, so the
    equivalent correction for THIS file is a rotation by S - 90 degrees. Only
    stated for an Oxford source whose header gives S.
    """
    if (vendor or "").lower() not in ("oxford", "bruker"):
        return None
    s = (geometry or {}).get("scanning_rotation_angle_deg")
    if s is None:
        return None
    corr = float(s) - 90.0
    return (
        f"to get in MTEX the orientations MTEX gives the source h5oina (Scanning "
        f"Rotation Angle of {float(s):g} degrees), load this file with "
        f"'EulerCorrection', rotation.byAxisAngle(zvector,{corr:g}*degree); the "
        "default .ang correction (setting 2) does not reproduce them. Derived "
        "from the MTEX source (loadEBSD_h5.m, loadEBSD_ang.m), not run in MATLAB. "
        "orix, kikuchipy and PyEBSDIndex read this file as it is.")


def ctf_project_note(*, vendor: str, provenance: Dict[str, object],
                     scan: Optional[Dict[str, object]], step_um: float,
                     geometry: Optional[Dict[str, object]] = None,
                     notes: Sequence[str] = (),
                     resampled_from: Optional[ScanGeometry] = None,
                     phases: Optional[Sequence] = None) -> str:
    """The same facts as the ``.ang`` comments, as one line for the CTF ``Prj`` field.

    A Channel Text File has no comment lines (parsers reject unknown ones), but
    its project field is free text. ``notes`` are the ``# ORIENTA_NOTE`` texts of the ``.ang``;
    each is appended as ``Note: <text>``. ``resampled_from`` and ``phases`` as for
    :func:`header_comments`; the space group of a phase is a field of the ``.ctf``, so the
    statement for the Y setting names the point group alone.
    """
    parts = ["Euler angles: " + ctf_frame_text(vendor)]
    parts.extend(crystal_frame_texts(phases, space_group=False))
    if scan is None:
        parts.append("Scan origin: not recorded by this export.")
    elif resampled_from is not None:
        parts.append("Scan origin: " + scan_origin_text(
            scan.get("scan_row_offset", 0), scan.get("scan_col_offset", 0),
            scan.get("scan_shape"), resampled_from.step_x,
            step_y_um=resampled_from.step_y, resampled=True, geometry=resampled_from))
    else:
        parts.append("Scan origin: " + scan_origin_text(
            scan.get("scan_row_offset", 0), scan.get("scan_col_offset", 0),
            scan.get("scan_shape"), step_um))
    acq = acquisition_line(geometry or {})
    if acq:
        parts.append("Acquisition: " + acq + ".")
    parts.append("Phase assignment: " + str(provenance["text"]))
    for n in notes:
        parts.append("Note: " + str(n))
    return " ".join(parts)


def resampling_note(resampling, geometry: ScanGeometry) -> str:
    """The note both files of a resampled hexagonal scan carry: one ASCII line, no tab.

    It states the rule (nearest neighbour), the step of the square grid, how many measured points the
    scan had, how many cells are written and how many measured points no cell takes, and that nothing
    was interpolated. When the direction of the row offset of ``geometry`` is not confirmed by any file,
    it says that this is an assumption. ``resampling`` is a ``hex_resample.SquareResampling``; its step
    is the one every coordinate of the file comes from."""
    text = (f"Resampled from a hexagonal scan to a square grid by nearest neighbour at step "
            f"{resampling.step:.10g} um (source points {resampling.n_source}, points written "
            f"{resampling.n_out}, source points unused {resampling.n_unused}); "
            f"nothing was interpolated.")
    if not geometry.shift_confirmed:
        side = "right" if float(geometry.odd_row_shift) > 0 else "left"
        text += (f" The direction of the row offset is an assumption (no file states it): the odd "
                 f"rows are placed half a step to the {side} of the even rows.")
    return text


def header_replacements(*, geometry: Optional[Dict[str, object]] = None,
                        pc_tsl=None, pc_state: Optional[Tuple[str, str]] = None) -> Dict[str, object]:
    """Real values for the placeholder fields orix writes as 0.

    ``pc_state`` is ``(machine text, plain words)`` for a result whose x/y/z-star hold no PC:
    the three lines stay (the format needs them, and a NaN breaks parsers) at 0, and
    directly above them stand ``# ORIENTA_PC: <machine text>`` and a note, so nobody takes
    the zeros for a measured PC.
    """
    rep: Dict[str, object] = {}
    g = geometry or {}
    if "working_distance_mm" in g:
        rep["# WorkingDistance"] = _header_field("WorkingDistance", g["working_distance_mm"])
    if pc_tsl is not None:
        rep["# x-star"] = _header_field("x-star", pc_tsl[0])
        rep["# y-star"] = _header_field("y-star", pc_tsl[1])
        rep["# z-star"] = _header_field("z-star", pc_tsl[2])
    elif pc_state is not None:
        rep["# x-star"] = [f"# {KEY_PC_STATE}: {pc_state[0]}",
                           f"# {KEY_NOTE}: {pc_state[1]}",
                           _header_field("x-star", 0.0)]
    return rep


def write_ang(path, xmap, *, replace=None, comments=None,
              symmetry_codes=None, geometry: Optional[ScanGeometry] = None) -> None:
    """Write the map with orix and edit its header. Replaces an existing file.

    ``geometry`` is the scan geometry of a hexagonal scan: the file is then a ``HexGrid`` file
    whose x/y columns are the positions of the geometry (see :func:`hex_export_positions`),
    written by a private copy of orix's writer, and ``xmap`` is the index-space rectangle as for
    a square scan. For ``None`` or a square geometry nothing differs from before."""
    if geometry is None or not geometry.is_hex:
        from orix.io import save

        save(str(path), xmap, overwrite=True)
        edit_ang_header(path, replace=replace, comments=comments,
                        symmetry_codes=symmetry_codes)
        return

    save_hex_ang(path, xmap, geometry)
    edit_ang_header(path, replace=replace, comments=comments, symmetry_codes=symmetry_codes)


#: The header line orix writes for every map, and what a hexagonal file says instead.
HEX_GRID_REPLACE = {"# GRID: SqrGrid": "# GRID: HexGrid"}


def save_hex_ang(path, xmap, geometry: ScanGeometry) -> None:
    """Write ``xmap`` (the index-space rectangle) as the ``.ang`` of a hexagonal scan: orix's
    header with its ``# GRID:`` line replaced by ``HexGrid`` (:data:`HEX_GRID_REPLACE`).

    The file is written under a temporary name beside ``path`` and moved onto it only when it is
    complete, so a refused write ("Nothing was written") leaves an existing file as it was and
    no temporary file behind. ``IOError`` for a name that is not ``.ang``, as ``orix.io.save``;
    ``ExportRefused`` when the written header has no ``HexGrid`` line (a changed orix)."""
    target = Path(str(path))
    if target.suffix.lower() != ".ang":
        raise IOError(f"'{target.suffix.lstrip('.')}' does not correspond to any supported "
                      "format for a hexagonal scan: the file name must end in '.ang'.")
    writer = orix_ang_writer_with(*hex_export_positions(geometry))
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.ang")
    try:
        writer(str(tmp), xmap)
        edit_ang_header(tmp, replace=HEX_GRID_REPLACE)
        with open(tmp, "r", encoding="utf-8", newline="") as fh:
            grid_lines = []
            for ln in fh:
                if not ln.startswith("#"):
                    break
                if ln.startswith("# GRID:"):
                    grid_lines.append(ln.rstrip("\r\n"))
        if grid_lines != ["# GRID: HexGrid"]:
            raise ExportRefused(
                "The installed orix writes the grid line of the header in a way this export "
                f"does not know (found {grid_lines!r}), so the file would not say it is "
                "hexagonal. Nothing was written.")
        os.replace(tmp, target)
    except BaseException:
        # a half-written file with square positions must not stay behind
        if tmp.exists():
            tmp.unlink()
        raise


# ---------------------------------------------------------------------------
# hexagonal scans
# ---------------------------------------------------------------------------

#: Relative difference of the step a result states and the step of its stamped geometry that is
#: still the same step (a float32 step read back from a file).
HEX_STEP_RTOL = 1e-4


def hex_geometry_for_export(stamp, original_shape, step_size_um=None, *,
                            hex_resampled=None) -> Optional[ScanGeometry]:
    """The hexagonal geometry a result is exported with, or ``None`` for a square file.

    ``stamp`` is the geometry stamped on the result when it ran (the dict of
    ``ScanGeometry.to_dict``); it is the only source, never the dataset that is active now.
    ``None`` for a result without a stamp, a square stamp, or a map that was resampled to a square
    grid from a hexagonal file (``hex_resampled``). For a hexagonal stamp ``ExportRefused`` when
    its grid is not the result's ``original_shape``, when ``step_size_um`` (the step the result
    states, ``None`` when it states none) is not the stamp's ``step_x``, or when the stamp cannot
    be read: a file with the wrong positions is worse than none.
    """
    if not isinstance(stamp, dict) or stamp.get("kind") != "hex" or hex_resampled is True:
        return None
    try:
        geometry = ScanGeometry.from_dict(stamp)
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportRefused(
            f"The scan geometry stored with this result cannot be used ({exc}). "
            "Nothing was written.") from exc
    if not geometry.steps_known:
        raise ExportRefused(
            "The step sizes of this hexagonal scan are not known, so its positions cannot be "
            "written. Nothing was written.")
    shape = (int(original_shape[0]), int(original_shape[1]))
    if geometry.n_cols < 2:
        raise ExportRefused(
            "A hexagonal scan of one column cannot be written as an .ang file: orix takes such "
            "a map for a one-dimensional one and writes no 2-D grid. Nothing was written.")
    if geometry.n_rows < 2:
        raise ExportRefused(
            "A hexagonal scan of one row cannot be written as an .ang file: orix takes such "
            "a map for a one-dimensional one, takes its y step from the map and writes no 2-D "
            "grid. Nothing was written.")
    if geometry.grid_shape != shape:
        raise ExportRefused(
            f"The hexagonal scan geometry of this result is {geometry.n_rows} x "
            f"{geometry.n_cols} points, its grid is {shape[0]} x {shape[1]}. Nothing was written.")
    if step_size_um is not None:
        try:
            stated = float(step_size_um)
        except (TypeError, ValueError):
            stated = float("nan")
        if not math.isclose(stated, geometry.step_x, rel_tol=HEX_STEP_RTOL):
            raise ExportRefused(
                f"The step size stored with this result ({step_size_um!r} um) is not the step "
                f"of its hexagonal scan geometry ({geometry.step_x:g} um). Nothing was written.")
    return geometry


def hex_x_lift(geometry: ScanGeometry) -> float:
    """How far the x column of a hexagonal ``.ang`` is raised, in um.

    With odd rows at -0.5 step_x the geometry has x = -step_x/2 in them, and OIM files carry no
    negative x: every x is raised by half a step. A scan of one row has no odd row and no lift."""
    if float(geometry.odd_row_shift) < 0 and geometry.n_rows > 1:
        return 0.5 * float(geometry.step_x)
    return 0.0


def hex_export_positions(geometry: ScanGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` of the x/y columns of the ``.ang`` of a hexagonal scan: two
    ``(n_rows, n_cols)`` arrays in um, row-major (the order of the data lines)."""
    x, y = geometry.positions()
    return x + hex_x_lift(geometry), y


def hex_grid_text(grid: ScanGeometry) -> str:
    """The value of the ``# ORIENTA_SCAN_GRID`` line."""
    return (f"HexGrid; rows={grid.n_rows}; columns={grid.n_cols} in every row; "
            f"odd rows at {float(grid.odd_row_shift):+.1f} step_x; "
            f"direction={'confirmed' if grid.shift_confirmed else 'assumed'}")


def hex_grid_notes(grid: ScanGeometry) -> List[str]:
    """The ``# ORIENTA_NOTE`` texts of a hexagonal file, in plain words."""
    notes = [
        f"Every row of this HexGrid file has {grid.n_cols} points (equal row length, "
        f"{grid.n_points} points in all); OIM's own hexagonal files have one point less in "
        "every second row.",
    ]
    if hex_x_lift(grid) > 0:
        notes.append(
            "x is raised by half a step: the odd rows lie half a step to the left of the even "
            "rows, and no x in this file is negative.")
    if not grid.shift_confirmed:
        side = "right" if float(grid.odd_row_shift) > 0 else "left"
        notes.append(
            f"The direction of the row offset is an assumption (no file states it): the odd "
            f"rows are placed half a step to the {side} of the even rows.")
    return notes


def orix_ang_writer_with(x, y):
    """A copy of orix's ``.ang`` writer that writes ``x`` and ``y`` as the x/y columns.

    orix writes the same positions for every map: it recomputes them from the shape and the step
    sizes with its module-level ``create_coordinate_arrays``. This builds a private copy of
    ``file_writer`` whose global of that name is a function returning ``x`` and ``y``; nothing in
    orix is patched, so no other thread or file sees the change. ``x`` and ``y`` are the
    ``(n_rows, n_cols)`` arrays of :func:`hex_export_positions`.

    The returned function, ``write(filename, xmap, **kwargs)``, writes ``filename`` exactly as
    orix would (``orix.io.save`` is bypassed: it does not overwrite, see :func:`write_ang`). It
    raises ``ExportRefused`` when orix no longer asks for the coordinates exactly once, or asks
    for another shape (``xmap`` is not the ``x.shape`` rectangle, or has one column, which orix
    collapses to a 1-D map): the positions would not be those of the file.
    """
    from orix.io.plugins import ang as orix_ang

    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"x and y must be two 2-D arrays of one shape, got {x.shape} and {y.shape}")
    if x.shape[1] < 2:
        raise ExportRefused(
            "A hexagonal scan of one column cannot be written as an .ang file: orix takes such "
            "a map for a one-dimensional one and writes no 2-D grid. Nothing was written.")
    if x.shape[0] < 2:
        raise ExportRefused(
            "A hexagonal scan of one row cannot be written as an .ang file: orix takes such "
            "a map for a one-dimensional one, takes its y step from the map and writes no 2-D "
            "grid. Nothing was written.")
    original = orix_ang.file_writer
    if (not isinstance(original, types.FunctionType) or original.__closure__ is not None
            or "create_coordinate_arrays" not in original.__code__.co_names):
        raise ExportRefused(
            "The installed orix writes the x/y columns in a way this export does not know. "
            "Nothing was written.")
    calls: List[tuple] = []

    def coordinates(shape=None, step_sizes=None):
        calls.append(tuple(shape) if shape else ())
        if len(calls) != 1 or tuple(int(v) for v in calls[0]) != x.shape:
            raise ExportRefused(
                f"The installed orix asked for the coordinates of a map of shape {calls[-1]} "
                f"(call {len(calls)}); the hexagonal scan is {x.shape}. Nothing was written.")
        return {"x": x.ravel().copy(), "y": y.ravel().copy()}, x.size

    private = types.FunctionType(
        original.__code__, {**original.__globals__, "create_coordinate_arrays": coordinates},
        original.__name__, original.__defaults__)
    private.__kwdefaults__ = None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)

    def write(filename, xmap, **kwargs):
        calls.clear()
        private(filename, xmap, **kwargs)
        if len(calls) != 1:
            raise ExportRefused(
                "The installed orix did not ask for the coordinates exactly once, so the x/y "
                "columns would not be those of the scan. Nothing was written.")

    return write
