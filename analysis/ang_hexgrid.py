"""Reading an EDAX TSL ``.ang`` file whose header says ``GRID: HexGrid``.

orix reads such a file as if its x/y columns were the positions of a square map: with the odd rows
half a step to one side it finds twice as many distinct x values as the scan has columns and
builds a ``(rows, 2 * columns)`` map with holes. Every grain and neighbour calculation on that map
is wrong without any message. This module reads the header first and then either

* hands a ``SqrGrid`` file (or a file that states no grid) to orix as before, with no scan grid,
* rebuilds a ``HexGrid`` file with the same number of points in every row as a map in index space
  (``x = column * step_x``, ``y = row * step_y``, the order of the data lines) and returns the grid
  it found as a plain dict, or
* refuses by name, with ``HexAngRefused``, any other ``HexGrid`` file (OIM's own files with one
  point less in every second row, a wrong point count, positions that are not those of a regular
  hexagonal lattice).

``grid`` is a plain dict (``n_rows``, ``n_cols``, ``step_x``, ``step_y``, ``shift``, ``confirmed``)
because this package imports nothing from ``backend``; the caller turns it into a scan geometry.
``shift`` is the offset of the odd rows in steps of ``step_x`` (+0.5 or -0.5). It is read from the
x column: the constant added to every x (a file written with the odd rows to the left lifts all x
by half a step so that none is negative) cancels in the difference of an odd and an even row.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Optional, Tuple

import numpy as np

#: step_y / step_x of a regular hexagonal lattice (the row spacing of a hexagonal scan).
HEX_ROW_RATIO = math.sqrt(3.0) / 2.0
#: Relative tolerance of the positions against the steps the header states, and of the step ratio
#: (the same value as ``edax_hex._POS_RTOL``).
POS_RTOL = 1e-3
#: Slack per index step on top of ``POS_RTOL``: the header states a step with six decimals and the
#: positions have five, so the position of the 2000th row can differ from ``2000 * YSTEP`` by 1e-3
#: um however regular the file is.
ROUNDING_PER_INDEX = 1e-6

_HEADER_KEYS = ("GRID", "XSTEP", "YSTEP", "NCOLS_ODD", "NCOLS_EVEN", "NROWS", "ORIENTA_SCAN_GRID")
_HEADER_LINE = re.compile(r"^#\s*(" + "|".join(_HEADER_KEYS) + r")\s*:\s*(.*?)\s*$")
_DIRECTION = re.compile(r"\bdirection=(\w+)")
# orix stops reading a header after this many lines
_MAX_HEADER_LINES = 1000


class AngGridRefused(ValueError):
    """An ``.ang`` file this package refuses by name. ``code`` is the stable identifier the
    interface translates; the message is the English sentence."""

    code = ""


class HexAngRefused(AngGridRefused):
    """The file says ``HexGrid`` but is not a hexagonal scan with equal rows that is read back."""

    code = "hex_ang_unreadable"


class HexNeighbourhoodUnsupported(AngGridRefused):
    """A calculation that works with the four neighbours of a square map was asked for a file
    with a hexagonal grid."""

    code = "hex_neighbourhood_unsupported"


def read_ang_grid_header(path) -> Dict[str, str]:
    """The grid facts of the header of an ``.ang`` file, as the text after the colon.

    Keys present in the header: ``GRID``, ``XSTEP``, ``YSTEP``, ``NCOLS_ODD``, ``NCOLS_EVEN``,
    ``NROWS`` and ``ORIENTA_SCAN_GRID`` (the first line of each). Only the header is read, up to
    the first line that does not start with ``#``."""
    found: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= _MAX_HEADER_LINES or not line.startswith("#"):
                break
            m = _HEADER_LINE.match(line.rstrip("\r\n"))
            if m and m.group(1) not in found:
                found[m.group(1)] = m.group(2)
    return found


def _refuse(reason: str) -> HexAngRefused:
    return HexAngRefused(
        "This .ang file states a hexagonal grid (GRID: HexGrid), but " + reason + " Orienta reads "
        "hexagonal .ang files that have the same number of points in every row, as Orienta writes "
        "them. It does not read this file, because reading it as a square map would give wrong "
        "neighbours and wrong grains. Convert the scan to a square grid in OIM Analysis, or load "
        "the scan's .h5 file instead.")


def _int_field(header: Dict[str, str], key: str) -> int:
    try:
        return int(header[key])
    except (KeyError, ValueError):
        raise _refuse(f"its header has no usable {key} line.") from None


def _float_field(header: Dict[str, str], key: str) -> float:
    try:
        value = float(header[key])
    except (KeyError, ValueError):
        raise _refuse(f"its header has no usable {key} line.") from None
    if not (math.isfinite(value) and value > 0):
        raise _refuse(f"its {key} is {header[key]!r}, which is not a step.")
    return value


def _tolerance(step_x: float, step_y: float, n: int) -> np.ndarray:
    return POS_RTOL * max(step_x, step_y) + ROUNDING_PER_INDEX * np.asarray(n, dtype=np.float64)


def _shape_of(header: Dict[str, str]) -> Tuple[int, int]:
    """``(n_rows, n_cols)`` of a hexagonal file with equal rows, or ``HexAngRefused``."""
    n_odd, n_even = _int_field(header, "NCOLS_ODD"), _int_field(header, "NCOLS_EVEN")
    n_rows = _int_field(header, "NROWS")
    if n_odd != n_even:
        raise _refuse(f"the rows are not of equal length (NCOLS_ODD {n_odd}, NCOLS_EVEN {n_even}).")
    if n_rows < 2 or n_even < 2:
        raise _refuse(f"it has {n_rows} rows of {n_even} points, and the offset of the odd rows "
                      "cannot be read from fewer than two rows and two columns.")
    return n_rows, n_even


def _grid_of(header: Dict[str, str], x, y) -> dict:
    """The grid dict of the positions ``x``, ``y`` (flat, in file order) under ``header``, or
    ``HexAngRefused``."""
    n_rows, n_cols = _shape_of(header)
    step_x, step_y = _float_field(header, "XSTEP"), _float_field(header, "YSTEP")
    if not math.isclose(step_y, step_x * HEX_ROW_RATIO, rel_tol=POS_RTOL):
        raise _refuse(f"its steps (XSTEP {step_x:g}, YSTEP {step_y:g}) are not those of a regular "
                      "hexagonal lattice (YSTEP = XSTEP * sqrt(3) / 2).")
    if x is None or y is None:
        raise _refuse("it has no x and y columns.")
    if x.size != n_rows * n_cols:
        raise _refuse(f"it holds {x.size} points, not the {n_rows} x {n_cols} = {n_rows * n_cols} "
                      "that its header states.")

    xs = np.asarray(x, dtype=np.float64).reshape(n_rows, n_cols)
    ys = np.asarray(y, dtype=np.float64).reshape(n_rows, n_cols)
    # a comparison with NaN is False, so a non-finite position would pass every check below
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise _refuse("some of its x or y positions are not finite numbers.")
    rows = np.arange(n_rows, dtype=np.float64)[:, None]
    cols = np.arange(n_cols, dtype=np.float64)[None, :]

    # y: row r lies at r * YSTEP in every column
    if np.any(np.abs(ys - rows * step_y) > _tolerance(step_x, step_y, rows)):
        raise _refuse("the y positions of its points are not rows at multiples of YSTEP.")
    # x: along a row the points are XSTEP apart, and all even rows start at one x, all odd rows at
    # another (this subtraction is the offset of the row, with any constant lift left in)
    dev = xs - cols * step_x
    start_even, start_odd = dev[0, 0], dev[1, 0]
    expected = np.where(rows % 2 == 1, start_odd, start_even)
    if np.any(np.abs(dev - expected) > _tolerance(step_x, step_y, cols)):
        raise _refuse("the x positions of its points are not rows of points XSTEP apart, "
                      "with the even rows alike and the odd rows alike.")
    shift = (start_odd - start_even) / step_x
    if abs(abs(shift) - 0.5) > POS_RTOL:
        raise _refuse(f"its odd rows are offset by {shift:+.3f} steps from the even rows, not "
                      "by half a step.")

    direction = _DIRECTION.search(header.get("ORIENTA_SCAN_GRID", ""))
    return {
        "n_rows": int(n_rows), "n_cols": int(n_cols),
        "step_x": float(step_x), "step_y": float(step_y),
        "shift": 0.5 if shift > 0 else -0.5,
        # Orienta writes which way it knew the direction; a file without that line has the
        # direction in its x column
        "confirmed": not (direction is not None and direction.group(1) == "assumed"),
    }


def _index_space_map(xmap, grid: dict):
    """``xmap`` with the coordinates of a regular rectangle in index space: the same rotations,
    phase ids, properties and phase list (the very phases orix read), the order of the data lines."""
    from orix.crystal_map import CrystalMap

    ys, xs = np.mgrid[0:grid["n_rows"], 0:grid["n_cols"]]
    return CrystalMap(
        rotations=xmap.rotations,
        phase_id=np.array(xmap.phase_id, copy=True),
        x=(xs.ravel() * grid["step_x"]).astype(float),
        y=(ys.ravel() * grid["step_y"]).astype(float),
        phase_list=xmap.phases,
        prop={key: np.asarray(xmap.prop[key]) for key in xmap.prop},
        scan_unit=xmap.scan_unit,
    )


def load_ang_xmap(path) -> Tuple[object, Optional[dict]]:
    """``(xmap, grid)`` of an ``.ang`` file.

    ``grid`` is ``None`` for every file that does not state ``GRID: HexGrid`` (the map is what
    ``orix.io.load`` returns, unchanged). For a hexagonal file with equal rows it is the dict
    described in the module docstring and the map sits in index space, shape ``(n_rows, n_cols)``.
    Raises ``HexAngRefused`` for any other hexagonal file. An unreadable file raises what
    ``orix.io.load`` raises."""
    from orix.io import load as orix_load

    try:
        header = read_ang_grid_header(path)
    except OSError:
        header = {}            # orix reports the missing or unreadable file in its own words
    if header.get("GRID", "").lower() != "hexgrid":
        return orix_load(str(path)), None

    # the header is checked before the data is read: a refusal costs no time
    _shape_of(header)
    try:
        xmap = orix_load(str(path))
    except (ValueError, OverflowError) as exc:
        # orix builds the map from the positions: a NaN position stops it with ValueError and an
        # infinite one with OverflowError (``int(np.around(position / step))`` in
        # ``CrystalMap._data_slices_from_coordinates``, orix 0.14 and 0.15)
        raise _refuse(f"orix could not build a map from its positions ({type(exc).__name__}: {exc}).") from exc
    grid = _grid_of(header, xmap.x, xmap.y)
    return _index_space_map(xmap, grid), grid
