"""One definition of "neighbour" for every evaluation that compares a scan point with the points around it.

A square scan has four neighbours (left, right, above, below). A hexagonal scan stored as the index rectangle
it is has six, and which two of the adjacent rows' points they are depends on the parity of the row and on
which way the odd rows are offset (``ScanGeometry.odd_row_shift``, reversed by a crop that starts on an odd
row). Evaluations that looked at "the four" of a hexagonal scan used to be refused (``hex_guard``); each one
that is made correct takes an optional ``topology`` argument whose default ``None`` is the code that existed
before, and gets it from ``hex_topology(geometry)``, which is ``None`` for every square or unknown scan. So
this module is the only place that knows what a hexagonal neighbour is.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from backend.api.services.scan_geometry import refuse_masked_stamp

#: up, down, left, right: the order of the inline code this replaces (variant_unification.py).
_SQUARE_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Topology:
    kind: str                  # "square" | "hex"
    n_rows: int
    n_cols: int
    odd_row_shift: float = 0.0   # +0.5 / -0.5 for hex, 0 for square

    def __post_init__(self) -> None:
        # The same contract as ScanGeometry: a wrong lattice must not be built quietly.
        if self.kind not in ("square", "hex"):
            raise ValueError(f"scan kind must be one of ('square', 'hex'), got {self.kind!r}")
        if self.kind == "hex" and abs(float(self.odd_row_shift)) != 0.5:
            raise ValueError(f"a hex scan shifts odd rows by +-0.5 steps, got {self.odd_row_shift}")

    @property
    def n_points(self) -> int:
        return self.n_rows * self.n_cols

    def _diagonal(self, row: int) -> int:
        """Column step (+1 or -1) of the second neighbour in the rows above and below ``row``."""
        return 1 if ((row % 2 == 1) == (self.odd_row_shift > 0)) else -1

    def offsets(self, row: int) -> Tuple[Tuple[int, int], ...]:
        if self.kind == "square":
            return _SQUARE_OFFSETS
        d = self._diagonal(row)
        return ((0, -1), (0, 1), (-1, 0), (-1, d), (1, 0), (1, d))

    def disc_offsets(self, row: int, rings: int) -> Tuple[Tuple[int, int], ...]:
        """(drow, dcol) of every position within ``rings`` steps of a point in ``row``, the centre first.

        Square: the (2 rings + 1)^2 block (what a box filter covers). Hexagonal: breadth first over the
        lattice, so ``rings = 2`` is the 19 points a 5 x 5 window stands for on a square scan."""
        if rings < 1:
            raise ValueError("rings must be at least 1")
        if self.kind == "square":
            rest = [(dr, dc) for dr in range(-rings, rings + 1) for dc in range(-rings, rings + 1)
                    if (dr, dc) != (0, 0)]
            return ((0, 0),) + tuple(rest)
        seen = {(0, 0)}
        frontier = [(0, 0)]
        for _ in range(rings):
            nxt = []
            for dr, dc in frontier:
                for a, b in self.offsets(row + dr):          # the parity of the frontier point's own row
                    p = (dr + a, dc + b)
                    if p not in seen:
                        seen.add(p)
                        nxt.append(p)
            frontier = nxt
        return ((0, 0),) + tuple(sorted(seen - {(0, 0)}))

    def xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """The position of every point in units of the step: two ``(n_rows, n_cols)`` float arrays ``(x, y)``.

        The rows of a hexagonal scan are ``sqrt(3) / 2`` steps apart (``ScanGeometry`` enforces it) and the odd
        rows are shifted by ``odd_row_shift`` steps. Used where "nearest" must mean nearest on the sample, which
        the distance in index space is not on a hexagonal scan."""
        rows = np.arange(self.n_rows, dtype=np.float64)[:, None]
        cols = np.arange(self.n_cols, dtype=np.float64)[None, :]
        if self.kind == "square":
            return cols + 0.0 * rows, rows + 0.0 * cols
        shift = np.where(rows % 2 == 1, float(self.odd_row_shift), 0.0)
        return cols + shift, rows * (np.sqrt(3.0) / 2.0) + 0.0 * cols

    def neighbours(self, flat: int) -> List[int]:
        row, col = divmod(int(flat), self.n_cols)
        out = []
        for dr, dc in self.offsets(row):
            r, c = row + dr, col + dc
            if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                out.append(r * self.n_cols + c)
        return out

    def table(self, rings: int = 1, include_self: bool = False) -> np.ndarray:
        """``(n_points, K)`` flat indices of the neighbours of every point, -1 where outside the scan."""
        if rings < 1:
            raise ValueError("rings must be at least 1")
        return _table(self, rings, include_self)

    def pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Every unordered neighbouring pair once, as flat indices ``a < b``."""
        return _pairs(self)


#: How many tables of each kind stay in memory. A table of a 1000 x 1200 scan is 58 MB for one ring and
#: grows with the square of the rings, so eight of them held more than a gigabyte after the scan that
#: needed them was gone. A scan can ask for more than four keys (the first ring, the windows of 3, 5 and 7
#: with and without ``include_self``, the diagnostics, the pairs); beyond four the least recently used is dropped and
#: built again when it is asked for, which costs time and nothing else.
#: ``clear_caches`` empties them when a file is loaded (``scan_geometry.clear_all``).
_CACHE_SIZE = 4


def clear_caches() -> None:
    """Drop every table that was built: the next call builds it again."""
    _table.cache_clear()
    _pairs.cache_clear()


@lru_cache(maxsize=_CACHE_SIZE)
def _table(t: Topology, rings: int, include_self: bool) -> np.ndarray:
    n_rows, n_cols = t.n_rows, t.n_cols
    flat = np.arange(t.n_points, dtype=np.int64).reshape(n_rows, n_cols)
    cols = np.arange(n_cols, dtype=np.int64)[None, :]

    def offsets_of(parity):
        if rings == 1 and not include_self:
            return t.offsets(parity)
        disc = t.disc_offsets(parity, rings)
        return disc if include_self else disc[1:]

    by_parity = [offsets_of(0), offsets_of(1)]
    out = np.full((t.n_points, len(by_parity[0])), -1, dtype=np.int64)
    for parity in (0, 1):
        rows = np.arange(parity, n_rows, 2, dtype=np.int64)
        if rows.size == 0:
            continue
        for k, (dr, dc) in enumerate(by_parity[parity]):
            rr, cc = rows[:, None] + dr, cols + dc
            ok = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
            out[flat[rows].ravel(), k] = np.where(ok, rr * n_cols + cc, -1).ravel()
    out.flags.writeable = False
    return out


@lru_cache(maxsize=_CACHE_SIZE)
def _pairs(t: Topology) -> Tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = t.n_rows, t.n_cols
    flat = np.arange(t.n_points, dtype=np.int64).reshape(n_rows, n_cols)
    a = [flat[:, :-1].ravel(), flat[:-1, :].ravel()]            # right, below (the same column)
    b = [flat[:, 1:].ravel(), flat[1:, :].ravel()]
    if t.kind == "hex":
        cols = np.arange(n_cols, dtype=np.int64)
        for parity in (0, 1):
            rows = np.arange(parity, n_rows - 1, 2, dtype=np.int64)
            d = t._diagonal(parity)
            keep = cols[(cols + d >= 0) & (cols + d < n_cols)]
            if rows.size and keep.size:
                a.append(flat[rows][:, keep].ravel())
                b.append(flat[rows + 1][:, keep + d].ravel())
    lo, hi = np.concatenate(a), np.concatenate(b)
    lo.flags.writeable = False
    hi.flags.writeable = False
    return lo, hi


def topology_for(geometry) -> Optional[Topology]:
    """The topology of a ``ScanGeometry``, of the dict the API reports for one, or None for anything else."""
    if geometry is None:
        return None
    get = geometry.get if isinstance(geometry, dict) else (lambda k, d=None: getattr(geometry, k, d))
    kind = get("kind")
    if kind not in ("square", "hex"):
        return None
    # A hex dict without the key (or with 0) is refused by Topology, not read as one of the two lattices.
    shift = float(get("odd_row_shift", 0.0)) if kind == "hex" else 0.0
    return Topology(kind, _whole_number(get("n_rows")), _whole_number(get("n_cols")), shift)


def _whole_number(value) -> int:
    """``int(value)``, but a number with a fractional part (4.7) is a ``ValueError``, not a silent 4."""
    whole = int(value)                        # None: TypeError; "x", nan: ValueError; inf: OverflowError
    if not isinstance(value, (str, bytes)) and whole != value:
        raise ValueError(f"{value!r} is not a whole number")
    return whole


def _stamp_kind(stamp):
    """The ``kind`` of a geometry or of the dict stored for one (``None`` for anything else)."""
    if stamp is None:
        return None
    return stamp.get("kind") if isinstance(stamp, dict) else getattr(stamp, "kind", None)


def topology_of_stamp(stamp, shape=None) -> Optional[Topology]:
    """The HEXAGONAL topology of a scan stamp stored on a result (``None`` for a square, unknown or missing
    one: the code as it was), checked against the grid it must describe. A stamp of another grid is a
    ``ValueError``, not a guess: the hexagonal rule on a grid it does not describe is wrong in a plausible way.
    A hexagonal stamp that cannot be read (a missing, non-numeric, infinite or fractional field) is a
    ``ValueError`` too, whose text always starts "The scan geometry stamped on this result cannot be read". A square
    stamp is always ``None``, complete or not: on a square scan only the index grid counts, so a damaged
    square stamp costs a square run nothing."""
    try:
        if _stamp_kind(stamp) == "square":
            return None
        if _stamp_kind(stamp) == "hex":
            refuse_masked_stamp(stamp)      # the rule of the routes that draw the map (hex_raster)
        topology = hex_topology(stamp)
    except (TypeError, KeyError, AttributeError, ValueError, OverflowError) as exc:
        cause = f"{type(exc).__name__}: {exc}"
        cause = cause if len(cause) <= 100 else cause[:97] + "..."
        raise ValueError(f"The scan geometry stamped on this result cannot be read ({cause})") from exc
    if topology is not None and shape is not None:
        rows, cols = (int(v) for v in shape)
        if (topology.n_rows, topology.n_cols) != (rows, cols):
            raise ValueError(
                f"the scan geometry stamped on this result is {topology.n_rows} x {topology.n_cols}, "
                f"but the map is {rows} x {cols}")
    return topology


def hex_topology(geometry) -> Optional[Topology]:
    """The topology of a HEXAGONAL scan, else None: None is how a consumer says "the code as it was"."""
    t = topology_for(geometry)
    return t if t is not None and t.kind == "hex" else None
