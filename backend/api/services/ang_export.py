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
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

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
    try:
        return all(abs(float(a) - b) < 1e-9
                   for a, b in zip(lattice, PLACEHOLDER_LATTICE))
    except (TypeError, ValueError):
        return False


def to_angstrom(lattice) -> Tuple[float, ...]:
    """``(a, b, c, alpha, beta, gamma)`` with the edges in angstrom.

    kikuchipy keeps the structure of a master pattern in NANOMETRES (Al is
    ``0.4049``), orix and diffpy read a CIF in angstrom (``4.049``). A cell edge
    below 2 angstrom does not occur in a crystal -- the smallest primitive edges
    are about 2.3 -- so a lattice whose smallest edge is below 2 is in
    nanometres and is scaled by ten. The rule is unit-blind on purpose: the
    structure of a phase does not say which unit it was written in.
    """
    a, b, c, al, be, ga = (float(v) for v in lattice)
    if min(a, b, c) < 2.0:
        a, b, c = a * 10.0, b * 10.0, c * 10.0
    return (a, b, c, al, be, ga)


def lattice_from_structure(phase) -> Optional[Tuple[float, ...]]:
    """The lattice a phase carries, in angstrom; ``None`` if it carries none."""
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
    return to_angstrom(raw)


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
    return to_angstrom(found[0])


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


def resolve_phase_lattice(phase, *, phase_file=None, recorded=None):
    """``(lattice_angstrom, source)`` for one phase, or raise ``LatticeUnknown``.

    Order: what the phase itself carries (a Hough map carries the CIF
    structure), what the run recorded, then the file the phase came from.
    """
    got = lattice_from_structure(phase)
    if got is not None:
        return got, "phase structure"
    if recorded is not None:
        rec = tuple(float(v) for v in recorded)
        if len(rec) == 6 and not is_placeholder_lattice(rec):
            return to_angstrom(rec), "recorded with the run"
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

def read_source_geometry(source_path) -> Dict[str, object]:
    """Acquisition facts from an Oxford h5oina header, in degrees / mm / kV.

    Keys that the file does not have are absent. Returns ``{}`` for a source
    that is not an h5oina (EDAX files keep these elsewhere); nothing is guessed.
    """
    out: Dict[str, object] = {}
    if not source_path:
        return out
    try:
        import h5py

        if not h5py.is_hdf5(str(source_path)):
            return out
        with h5py.File(str(source_path), "r") as f:
            for entry in f.keys():
                hdr = f.get(f"{entry}/EBSD/Header")
                if hdr is None or not hasattr(hdr, "keys"):
                    continue

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
    except Exception as exc:
        logger.warning("could not read the acquisition geometry of %s: %s",
                       source_path, exc)
    return out


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
#: is written as 2, the code orix's reader maps to its "2/m" (unique axis along
#: z). EDAX also has a code 20 for the unique axis along y; orix has no such
#: Laue group, so it is not written.
_LAUE_TO_TSL = {
    "-1": "1",
    "2/m": "2", "112/m": "2",
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


def tsl_symmetry_code(point_group) -> str:
    """The TSL ``# Symmetry`` code of a phase. Raises when it has none."""
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
                      recorded_lattice=None, recorded_space_group=None) -> Dict[str, object]:
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
            phase, phase_file=phase_file or sht_path, recorded=recorded_lattice)
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
                     scan_shape, step_um: float) -> str:
    r0, c0 = int(scan_row_offset or 0), int(scan_col_offset or 0)
    if r0 == 0 and c0 == 0:
        return ("the first pixel of this file is row 0, column 0 of the scan "
                "(not a crop).")
    shape = ""
    if scan_shape is not None:
        try:
            shape = f" {int(scan_shape[0])} x {int(scan_shape[1])}"
        except Exception:
            shape = ""
    return (
        f"the first pixel of this file is row {r0}, column {c0} of the original"
        f"{shape} scan (0-based). Coordinates in this file start at 0; the "
        f"origin lies at x {c0 * step_um:.3f} um, y {r0 * step_um:.3f} um of the "
        "original scan."
    )


# ---------------------------------------------------------------------------
# pattern centre in the EDAX convention
# ---------------------------------------------------------------------------

def pc_tsl_from_bruker(pc, detector_shape) -> Optional[Tuple[float, float, float]]:
    """PC in the EDAX TSL convention from a Bruker-convention PC.

    kikuchipy's PC is Bruker; the TSL one is NOT the same numbers:
    ``y_T = 1 - y_B`` and ``z_T = z_B * nrows / min(nrows, ncols)`` (the
    installed kikuchipy's ``EBSDDetector.pc_tsl``; its docstring writes the
    aspect ratio the other way round, the code is what runs). Needs the
    detector shape; returns ``None`` without it.
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

def build_export_xmap(
    xmap,
    original_shape,
    selection_mask,
    step_um: float,
    confidence_rows=None,
    *,
    phase_files: Optional[Dict[int, str]] = None,
    recorded_lattices: Optional[Dict[int, Sequence[float]]] = None,
):
    """A CrystalMap on the full grid, in micrometres, with real lattices.

    Rotations stay in the frame the indexer produced (the EDAX TSL / kikuchipy
    frame); only coordinates, phase table and ids are rebuilt. Returns
    ``(xmap, tsl_codes)`` where ``tsl_codes`` maps the phase number written in
    the file to its TSL ``# Symmetry`` code.
    """
    from diffpy.structure import Lattice, Structure
    from orix.crystal_map import CrystalMap, Phase, PhaseList
    from orix.quaternion import Rotation

    from backend.api.services.result_exporter import (
        confidence_rows_for_export,
        map_phase_ids_for_export,
        place_rows_on_grid,
        xmap_phase_write_table,
    )

    if not (step_um and step_um > 0):
        raise ValueError(f"step size must be positive, got {step_um!r}")
    n_rows, n_cols = int(original_shape[0]), int(original_shape[1])
    n = n_rows * n_cols

    quats = np.asarray(xmap.rotations.data, dtype=float)
    if quats.ndim == 3:                       # top-N matches: keep the best
        quats = quats[:, 0, :]
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
    tsl_codes: Dict[int, str] = {}
    for written_id, phase in table:
        actual = written_to_actual.get(written_id)
        lattice, source = resolve_phase_lattice(
            phase,
            phase_file=(phase_files or {}).get(actual),
            recorded=(recorded_lattices or {}).get(actual),
        )
        logger.info("ang export: phase %s lattice from %s: %s",
                    getattr(phase, "name", "?"), source, lattice)
        name = getattr(phase, "name", "") or f"phase{written_id}"
        pg = getattr(phase, "point_group", None)
        try:
            tsl_codes[written_id] = tsl_symmetry_code(pg)
        except ValueError as exc:
            raise SymmetryUnknown(f"phase {name!r}: {exc}") from exc
        structure = Structure(title=name, lattice=Lattice(*lattice))
        phases.append(Phase(
            name=name,
            point_group=None if pg is None else getattr(pg, "name", str(pg)),
            structure=structure,
        ))
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
                col = np.asarray(xmap.prop[key]).reshape(len(xmap.rotations), -1)[:, 0]
                prop["ci"] = np.asarray(place_rows_on_grid(
                    col, original_shape, selection_mask, fill=0.0),
                    dtype=np.float32).reshape(n)
                break
    for key in ("fit", "patternfit"):
        if key in getattr(xmap, "prop", {}):
            col = np.asarray(xmap.prop[key]).reshape(len(xmap.rotations), -1)[:, 0]
            try:
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
        y=(ys.ravel() * float(step_um)).astype(float),
        phase_list=phase_list,
        prop=prop,
        scan_unit="um",
    )
    return xmap_out, tsl_codes


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

    ``replace`` maps a line prefix (``"# x-star"``) to the whole new line.
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
) -> List[str]:
    """The ``# KEY: text`` lines every ``.ang`` of this application carries."""
    lines: List[str] = []
    if method:
        lines.append(f"# {KEY_INDEXING}: method={method}; software=Orienta")
    for t in euler_frame_text(vendor):
        lines.append(f"# {KEY_FRAME}: {t}")
    if scan is None:
        lines.append(f"# {KEY_ORIGIN}: not recorded by this export; coordinates "
                     "start at 0 at the first pixel of the file.")
    else:
        lines.append("# %s: %s" % (KEY_ORIGIN, scan_origin_text(
            scan.get("scan_row_offset", 0), scan.get("scan_col_offset", 0),
            scan.get("scan_shape"), step_um)))
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


def ctf_project_note(*, vendor: str, provenance: Dict[str, object],
                     scan: Optional[Dict[str, object]], step_um: float,
                     geometry: Optional[Dict[str, object]] = None) -> str:
    """The same facts as the ``.ang`` comments, as one line for the CTF ``Prj`` field.

    A Channel Text File has no comment lines (parsers reject unknown ones), but
    its project field is free text.
    """
    parts = ["Euler angles: " + " ".join(euler_frame_text(
        vendor, unit="degrees", angles="Euler1, Euler2, Euler3", first="Euler1"))]
    if scan is None:
        parts.append("Scan origin: not recorded by this export.")
    else:
        parts.append("Scan origin: " + scan_origin_text(
            scan.get("scan_row_offset", 0), scan.get("scan_col_offset", 0),
            scan.get("scan_shape"), step_um))
    acq = acquisition_line(geometry or {})
    if acq:
        parts.append("Acquisition: " + acq + ".")
    parts.append("Phase assignment: " + str(provenance["text"]))
    return " ".join(parts)


def header_replacements(*, geometry: Optional[Dict[str, object]] = None,
                        pc_tsl=None) -> Dict[str, str]:
    """Real values for the placeholder fields orix writes as 0."""
    rep: Dict[str, str] = {}
    g = geometry or {}
    if "working_distance_mm" in g:
        rep["# WorkingDistance"] = _header_field("WorkingDistance", g["working_distance_mm"])
    if pc_tsl is not None:
        rep["# x-star"] = _header_field("x-star", pc_tsl[0])
        rep["# y-star"] = _header_field("y-star", pc_tsl[1])
        rep["# z-star"] = _header_field("z-star", pc_tsl[2])
    return rep


def write_ang(path, xmap, *, replace=None, comments=None,
              symmetry_codes=None) -> None:
    """Write the map with orix and edit its header. Replaces an existing file."""
    from orix.io import save

    save(str(path), xmap, overwrite=True)
    edit_ang_header(path, replace=replace, comments=comments,
                    symmetry_codes=symmetry_codes)
