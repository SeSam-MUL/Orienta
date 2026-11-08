"""The pattern-centre conventions of the three vendors, and the conversion between them.

Orienta keeps every PC in the BRUKER convention (kikuchipy's own): ``PCx`` from the left edge
of the pattern, ``PCy`` from the TOP edge, ``PCz`` in units of the pattern height. A number a
person reads off another program, or types into the page, is in that program's convention:

* ``tsl`` (EDAX OIM / TSL, also called ``edax``): ``y*`` counts from the BOTTOM edge (as a
  fraction of the height), and ``z*`` is in units of the SHORTER pattern side.
* ``oxford`` (Oxford AZtec): ``y`` counts from the BOTTOM edge and ``y`` and ``z`` are both in
  units of the pattern WIDTH (so ``y = (1 - PCy) * rows / cols`` and ``z = PCz * rows / cols``).

The numbers are not guessed here. Both directions go through kikuchipy's own detector class
(``EBSDDetector(..., convention=...)``, ``pc_tsl()``, ``pc_oxford()``), the code that loads
the vendors' files, so a PC typed in a vendor's convention is read the way the same vendor's
file would have been read. Which convention a number is in depends on the pattern SHAPE: every
function here needs it, and refuses to guess one.

EMsoft's convention is not offered: it is in pixels and needs the pixel size and the binning,
which the page does not hold reliably.
"""
from __future__ import annotations

import math
import numbers
from typing import Sequence, Tuple

import numpy as np

#: What the page and the API call each convention. ``edax`` is an alias of ``tsl``.
CONVENTIONS: Tuple[str, ...] = ("bruker", "tsl", "oxford")
_ALIASES = {"bruker": "bruker", "tsl": "tsl", "edax": "tsl", "oxford": "oxford"}


class ConventionError(ValueError):
    """A convention name or a shape that cannot be used. ``code`` is what the interface
    translates on."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonical(name) -> str:
    """``"bruker" | "tsl" | "oxford"`` for a name (case-insensitive, ``edax`` = ``tsl``)."""
    key = str(name or "").strip().lower()
    if key not in _ALIASES:
        raise ConventionError(
            "unknownConvention",
            f"unknown pattern-centre convention {name!r}; use one of {', '.join(CONVENTIONS)}")
    return _ALIASES[key]


def _shape(shape) -> Tuple[int, int]:
    try:
        if isinstance(shape, (str, bytes)) or len(shape) != 2:    # a stack (N, H, W) is not read from its first two
            raise ValueError("not (rows, columns)")
        rows, cols = (int(shape[0]), int(shape[1]))
        if rows != float(shape[0]) or cols != float(shape[1]):      # 60.5 is not cut to 60
            raise ValueError("not whole numbers")
    except (TypeError, ValueError, IndexError, KeyError, OverflowError):
        rows = cols = 0
    if rows <= 0 or cols <= 0:
        raise ConventionError(
            "needsShape", "a pattern-centre conversion needs the pattern shape (rows, columns)")
    return rows, cols


def _triple(pc) -> Tuple[float, float, float]:
    try:
        items = list(pc)
        # plain numbers only: a bool would become 1.0 / 0.0, a string or a nested list is no number
        if len(items) != 3 or not all(
                isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)) for v in items):
            raise ValueError("not three numbers")
        vals = [float(v) for v in items]
    except (TypeError, ValueError, OverflowError):
        raise ConventionError("badPc", "a pattern centre is three finite numbers") from None
    if not all(math.isfinite(v) for v in vals):
        raise ConventionError("badPc", "a pattern centre is three finite numbers")
    return vals[0], vals[1], vals[2]


def convert(pc: Sequence[float], source: str, target: str, shape) -> Tuple[float, float, float]:
    """``pc`` given in convention ``source`` for patterns of ``shape`` (rows, columns), in
    convention ``target``. A conversion to the same convention returns the numbers unchanged."""
    src, dst = canonical(source), canonical(target)
    triple = _triple(pc)
    rows, cols = _shape(shape)
    if src == dst:
        return triple
    from kikuchipy.detectors import EBSDDetector
    with np.errstate(over="ignore", invalid="ignore"):      # an overflow is refused below, not warned about
        det = EBSDDetector(shape=(rows, cols), pc=triple, convention=src)   # stored as Bruker
        if dst == "bruker":
            out = det.pc_bruker()
        elif dst == "tsl":
            out = det.pc_tsl()
        else:
            out = det.pc_oxford()
    flat = [float(v) for v in out.reshape(-1)[:3]]
    if not all(math.isfinite(v) for v in flat):       # a finite but huge PC can overflow in the conversion
        raise ConventionError("badPc", "the converted pattern centre is not a finite number")
    return flat[0], flat[1], flat[2]


def to_bruker(pc, source: str, shape) -> Tuple[float, float, float]:
    return convert(pc, source, "bruker", shape)


def from_bruker(pc, target: str, shape) -> Tuple[float, float, float]:
    return convert(pc, "bruker", target, shape)
