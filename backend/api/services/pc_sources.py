"""Read the pattern centre another file holds ("Take PC from...").

A PC can be taken from a file that is not the one being worked on: the ``.ang`` OIM wrote for a
scan, the ``.osc`` beside an EDAX pattern file, an EDAX or Oxford ``.h5``, an Orienta export. Each
states it in its own convention and some do not state the pattern SHAPE it belongs to, which the
conversion needs (``pc_convention``). This module reads the numbers, converts them to Bruker, and
says where every assumption was made (``notes``, ``shape_assumed``). It never applies anything;
``routes/pc_calibrations`` shows the proposal to the user first.
"""
from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from backend.api.services import pc_convention
from backend.api.services.pc_search import PHYSICAL_HIGH, PHYSICAL_LOW   # numpy only: the cost imports lazily


class PcSourceError(ValueError):
    """A file that gives no usable PC. ``code`` is what the interface translates on."""

    def __init__(self, code: str, message: str, **params):
        super().__init__(message)
        self.code = code
        self.params = params


class Note(str):
    """A sentence about how a PC was read, with a stable ``code`` and ``params`` the interface translates on.

    It is the English sentence for everything that treats it as text (``in``, ``startswith``, ``==``, json);
    the code is for a caller that wants to say it in another language."""

    code: str
    params: dict

    def __new__(cls, text: str, code: str, **params):
        obj = super().__new__(cls, text)
        obj.code = code
        obj.params = params
        return obj


@dataclass
class FilePc:
    pc_bruker: Tuple[float, float, float]
    source_kind: str                  # "ang" | "osc" | "edax_h5" | "oxford_h5" | "orienta_export"
    convention: str                   # what the file states its numbers in
    stated: Tuple[float, float, float]
    shape: Optional[Tuple[int, int]] = None
    shape_assumed: bool = False       # the file does not state its pattern shape: the caller's was used
    per_pixel: bool = False           # the file holds a map; the PC is its mean
    notes: List[str] = field(default_factory=list)

    def note_items(self) -> List[dict]:
        """Every note as ``{code, params, text}``; a plain sentence without a code gets ``other``."""
        return [{"code": getattr(n, "code", "other"), "params": dict(getattr(n, "params", {})), "text": str(n)}
                for n in self.notes]

    def to_dict(self) -> dict:
        return {"pc": [float(v) for v in self.pc_bruker], "source_kind": self.source_kind,
                "convention": self.convention, "stated": [float(v) for v in self.stated],
                "shape": None if self.shape is None else [int(v) for v in self.shape],
                "shape_assumed": self.shape_assumed, "per_pixel": self.per_pixel, "notes": [str(n) for n in self.notes],
                "note_items": self.note_items()}


def _finite3(values) -> Tuple[float, float, float]:
    out = tuple(float(v) for v in values)
    if len(out) != 3 or not all(math.isfinite(v) for v in out):
        raise PcSourceError("noPcInFile", "the file holds no usable pattern centre")
    return out


def _whole_shape(rows, cols) -> Optional[Tuple[int, int]]:
    """``(rows, columns)`` when both are finite, positive whole numbers, else ``None``: 127.5 is
    not cut to 127 (``pc_convention`` refuses it, and so does this module)."""
    try:
        r, c = float(rows), float(cols)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(r) and math.isfinite(c)) or r <= 0 or c <= 0 or r != int(r) or c != int(c):
        return None
    return int(r), int(c)


def _caller_shape(caller) -> Optional[Tuple[int, int]]:
    """The caller's shape if it is exactly (rows, columns); a stack shape (N, H, W) is not read from its first two."""
    try:
        if caller is None or isinstance(caller, (str, bytes)) or len(caller) != 2:
            return None
        return _whole_shape(caller[0], caller[1])
    except TypeError:
        return None


def _shape_or_error(stated, caller):
    """``(shape, assumed)``: the shape the file states, else the caller's, else an error."""
    if stated is not None:
        return (int(stated[0]), int(stated[1])), False
    shape = _caller_shape(caller)
    if shape is not None:
        return shape, True
    raise PcSourceError(
        "needsShape", "this file does not state its pattern size, and the conversion needs it")


def _stated_pc(values) -> Tuple[float, float, float]:
    """Three finite numbers that are not all 0: three zeros are what a file that was never calibrated holds."""
    stated = _finite3(values)
    if stated == (0.0, 0.0, 0.0):
        raise PcSourceError("noPcInFile", "the three numbers of the pattern centre are all 0: the file states no pattern centre")
    return stated


def _to_bruker(stated, convention, shape):
    """The conversion service, its refusals (an overflow, a non-finite result) as 'the file has no PC'."""
    try:
        return pc_convention.to_bruker(stated, convention, shape)
    except pc_convention.ConventionError as exc:
        raise PcSourceError("noPcInFile", f"the pattern centre the file states cannot be converted: {exc}") from exc


#: What of the converted PC depends on the shape, by convention (``pc_convention``): TSL's y never does and its z
#: only when the pattern is taller than wide; Oxford's y and z both do (they are in units of the width).
_SHAPE_DEPENDENCE = {
    "tsl": "only z of the converted PC depends on it, and only for a pattern taller than wide (rows > columns)",
    "oxford": "y and z of the converted PC depend on it",
}


def _assumed_shape_note(what: str, shape: Tuple[int, int], convention: str) -> Note:
    return Note(f"{what} does not state a usable pattern size: {shape[0]} x {shape[1]} (rows x columns), the size of "
                f"the file it is applied to, was assumed; {_SHAPE_DEPENDENCE[convention]}.",
                "assumedShape", rows=int(shape[0]), cols=int(shape[1]), convention=convention)


def _unreadable_as_code(reader):
    """A file whose content is not what its format says (an empty dataset, a group where a number should be, a
    string, a file that cannot be opened) is ``unreadableFile``, not a raw exception; a code already set runs through."""
    @functools.wraps(reader)
    def wrapped(path: Path, shape):
        try:
            return reader(path, shape)
        except PcSourceError:
            raise
        except (OSError, ValueError, TypeError, IndexError, KeyError) as exc:
            raise PcSourceError("unreadableFile", f"cannot read {path.name}: {exc}") from exc
    return wrapped


def _range_notes(pc) -> List[Note]:
    """A PC outside what a detector can be is still only proposed; the note says so (limits of ``pc_search``)."""
    outside = [axis for k, axis in enumerate("xyz") if not (PHYSICAL_LOW[k] <= pc[k] <= PHYSICAL_HIGH[k])]
    if not outside:
        return []
    limits = ", ".join(f"{axis} {PHYSICAL_LOW[k]}-{PHYSICAL_HIGH[k]}" for k, axis in enumerate("xyz"))
    return [Note(f"The pattern centre is outside what a detector can be ({limits}); it is only a proposal, "
                 f"check {', '.join(outside)}.", "outsideDetectorRange", axes=list(outside), limits=limits)]


# ------------------------------------------------------------------ .ang

_ANG_STAR = re.compile(r"^#\s*(x-star|y-star|z-star)\s+(\S+)\s*$")
_ANG_STATE = re.compile(r"^#\s*ORIENTA_PC\s*:\s*(.+?)\s*$")
#: The header is read with fixed limits, never the rest of the file: a line is cut at 4096 characters, the
#: header ends after 400 lines. A .ang without line breaks is one line of its whole size.
_ANG_MAX_LINE = 4096
_ANG_MAX_LINES = 400


@_unreadable_as_code
def _read_ang(path: Path, shape) -> FilePc:
    stars, state = {}, None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for _ in range(_ANG_MAX_LINES + 1):
            line = handle.readline(_ANG_MAX_LINE + 1)
            if not line or (line.strip() and not line.startswith("#")):
                break
            if len(line) > _ANG_MAX_LINE and not line.endswith("\n"):
                raise PcSourceError("noPcInFile", f"a header line of the .ang is longer than {_ANG_MAX_LINE} characters")
            m = _ANG_STAR.match(line)
            if m:
                try:
                    stars[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
            s = _ANG_STATE.match(line)
            if s:
                state = s.group(1)
    if state is not None:
        raise PcSourceError("noPcInFile", f"this Orienta export states no pattern centre ({state})", state=state)
    if set(stars) != {"x-star", "y-star", "z-star"}:
        raise PcSourceError("noPcInFile", "the .ang header has no x-star / y-star / z-star")
    stated = _stated_pc((stars["x-star"], stars["y-star"], stars["z-star"]))
    sh, assumed = _shape_or_error(None, shape)
    pc = _to_bruker(stated, "tsl", sh)
    return FilePc(pc, "ang", "tsl", stated, sh, assumed, False, [_assumed_shape_note("The .ang header", sh, "tsl")])


# ------------------------------------------------------------------ .osc

def _read_osc(path: Path, shape) -> FilePc:
    try:                                  # edax_up1 swallows a read error and reports 'no PC'
        with open(path, "rb"):
            pass
    except OSError as exc:
        raise PcSourceError("unreadableFile", f"cannot open {path.name}: {exc}") from exc
    from edax_up1 import read_osc_metadata
    meta = read_osc_metadata(str(path))
    if meta.pc is None:
        raise PcSourceError("noPcInFile", "the .osc file holds no plausible pattern centre")
    stated = _stated_pc(meta.pc)
    sh, assumed = _shape_or_error(None, shape)
    return FilePc(_to_bruker(stated, "tsl", sh), "osc", "tsl", stated, sh, assumed, False,
                  [_assumed_shape_note("The .osc file", sh, "tsl")])


# ------------------------------------------------------------------ HDF5

def _scalar(group, key):
    return float(np.asarray(group[key][()]).reshape(-1)[0]) if key in group else None


def _scan_groups(h5):
    import h5py
    return [k for k in h5.keys() if isinstance(h5[k], h5py.Group)]


def _stated_shape(group):
    """``(rows, columns)`` from ``Pattern Height`` / ``Pattern Width`` of a scan's header, or ``None``."""
    if "EBSD/Header" not in group:
        return None
    header = group["EBSD/Header"]
    return _whole_shape(_scalar(header, "Pattern Height"), _scalar(header, "Pattern Width"))


def _oxford_map(data) -> Tuple[np.ndarray, int]:
    """``(3, n)`` per-pixel PC of the pixels whose three values are finite and above 0, and the number left out.
    An unindexed pixel holds 0; it must not pull the mean towards 0."""
    comps = [np.asarray(data[k][()], dtype=float).ravel()
             for k in ("Pattern Center X", "Pattern Center Y", "Detector Distance")]
    stack = np.vstack(comps)               # ValueError (-> unreadableFile) when the three differ in size
    valid = np.all(np.isfinite(stack) & (stack > 0), axis=0)
    if not valid.any():
        raise PcSourceError("noPcInFile", "no pixel of the map holds a pattern centre (all are 0 or not finite)")
    return stack[:, valid], int(valid.size - valid.sum())


@_unreadable_as_code
def _read_h5(path: Path, shape) -> FilePc:
    import h5py
    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise PcSourceError("unreadableFile", f"cannot open {path.name}: {exc}") from exc
    with handle as f:
        # The manufacturer's header wins over /Detector/pc of an Orienta export, as in the loader
        # (safe_loader._restore_pc_from_orienta_export): the file proposes the PC it is loaded with.
        header_seen = False
        for key in _scan_groups(f):
            group = f[key]
            # 1. EDAX: the calibration group of the scan header (TSL numbers)
            cal_path = "EBSD/Header/Pattern Center Calibration"
            if cal_path in group:
                header_seen = True
                cal = group[cal_path]
                vals = [_scalar(cal, k) for k in ("x-star", "y-star", "z-star")]
                if all(v is not None and math.isfinite(v) for v in vals) and any(v != 0 for v in vals):
                    sh, assumed = _shape_or_error(_stated_shape(group), shape)
                    stated = tuple(vals)
                    notes = [_assumed_shape_note("The file's header", sh, "tsl")] if assumed else []
                    return FilePc(_to_bruker(stated, "tsl", sh), "edax_h5", "tsl", stated, sh,
                                  assumed, False, notes)
            # 2. Oxford: a per-pixel map (Oxford numbers); its mean over the pixels that hold a PC
            if "EBSD/Data/Pattern Center X" in group:
                header_seen = True
                good, left_out = _oxford_map(group["EBSD/Data"])
                with np.errstate(all="ignore"):
                    mean, spread = good.mean(axis=1), good.std(axis=1)
                stated = _finite3([float(v) for v in mean])
                sh, assumed = _shape_or_error(_stated_shape(group), shape)
                per_pixel = bool(good.shape[1] > 1 and np.any(np.ptp(good, axis=1) > 0))
                notes = []
                if left_out:
                    notes.append(Note(f"{left_out} of {left_out + good.shape[1]} pixels of the map were left out of "
                                      "the mean: their pattern centre is not finite or not above 0 (unindexed points).",
                                      "pixelsLeftOut", left_out=int(left_out), total=int(left_out + good.shape[1])))
                if per_pixel:
                    notes.append(Note("The mean over the map of the file's per-pixel pattern centre.", "meanOfMap"))
                    if np.all(np.isfinite(spread)):    # values near 1e154 square to inf: no spread to report
                        notes.append(Note("Standard deviation over the map (Oxford numbers): "
                                          + ", ".join(f"{axis} {float(s):.3g}" for axis, s in zip("xyz", spread))
                                          + ".", "stdOverMap", **{axis: float(s) for axis, s in zip("xyz", spread)}))
                if assumed:
                    notes.append(_assumed_shape_note("The file's header", sh, "oxford"))
                return FilePc(_to_bruker(stated, "oxford", sh), "oxford_h5", "oxford", stated, sh,
                              assumed, per_pixel, notes)
        if header_seen:
            raise PcSourceError("noPcInFile", f"the header of {path.name} states no usable pattern centre "
                                "(the loader lets the header win over /Detector/pc, so this is not used)")
        # 3. an Orienta export states the PC its run used, in Bruker's convention
        if "Detector" in f and "pc" in f["Detector"] and not bool(f["Detector"].attrs.get("pc_missing", False)):
            pc = _stated_pc(np.asarray(f["Detector/pc"][()], dtype=float).ravel())   # exactly three values
            recorded = f["Detector"].attrs.get("shape")
            stated_shape = None if recorded is None else _caller_shape(np.asarray(recorded).ravel().tolist())
            sh = stated_shape or _caller_shape(shape)
            return FilePc(pc, "orienta_export", "bruker", pc, sh, stated_shape is None and sh is not None, False,
                          [Note("An Orienta export: the PC its run used.", "orientaExport")])
    raise PcSourceError("noPcInFile", f"{path.name} holds no pattern centre Orienta can read")


# ------------------------------------------------------------------ entry point

_READERS = {".ang": _read_ang, ".osc": _read_osc, ".h5": _read_h5, ".hdf5": _read_h5, ".h5oina": _read_h5}


def read_pc(path, *, shape=None) -> FilePc:
    """The pattern centre of ``path`` in Bruker's convention.

    ``shape`` is the (rows, columns) of the patterns the PC will be applied to: used when the file
    does not state its own (``.ang``, ``.osc``), and then ``shape_assumed`` is set."""
    p = Path(path)
    if not p.is_file():
        raise PcSourceError("fileNotFound", f"{p.name} does not exist")
    reader = _READERS.get(p.suffix.lower())
    if reader is None:
        raise PcSourceError("unsupportedFile", f"{p.name}: no pattern centre can be read from a {p.suffix or 'file'} file",
                            suffix=p.suffix.lower())
    result = reader(p, shape)
    result.notes.extend(_range_notes(result.pc_bruker))
    return result
