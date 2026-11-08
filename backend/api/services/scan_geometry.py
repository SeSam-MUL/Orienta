"""Where every scan point sits on the sample, and how a crop changes that.

A scan is a rectangle of ``n_rows x n_cols`` points in the index space the
patterns are stored in. On a square scan point (row, col) sits at
``(col * step_x, row * step_y)``. On a hexagonal scan every second row is shifted
by half a step along x: with ``odd_row_shift = +0.5`` the points of the odd rows
(row index 1, 3, ...) sit half a step to the right, with ``-0.5`` half a step to
the left, and the rows are ``step_y = step_x * sqrt(3) / 2`` apart. This module
is the one place that knows that. Everything else asks a ``ScanGeometry``.

The registry at the bottom holds one geometry per dataset name and has the same
lifecycle as ``crop_window``: it is cleared on a file load, copied on a deepcopy,
composed on a crop, dropped on a delete and carried through the per-file stash.
It is a module of its own because ``crop_window`` holds crops only; ``get_crop(name)
is None`` means "this dataset is not a crop" and must keep meaning that.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: step_y / step_x of a regular hexagonal lattice.
HEX_ROW_RATIO = math.sqrt(3.0) / 2.0
#: How far a stated step_y may be from step_x * sqrt(3)/2 before the file is refused.
HEX_RATIO_RTOL = 1e-3

KINDS = ("square", "hex")

#: Which way the odd rows of a hexagonal scan are offset, in steps: +0.5 = to the right. NO FILE
#: STATES IT (``HexGrid`` and ``IsScanSuccessful`` of an APEX file are unreliable and a UP header
#: has no such field), so this is OIM's convention. ``ScanGeometry.hex`` keeps
#: ``shift_confirmed = False`` as its default: a hexagonal geometry from any other source is
#: still an assumption. An APEX measurement is confirmed by measurement (the direction was read
#: from the patterns of EDAX hexagonal scans: the odd rows sit half a step to the right) and passes
#: ``shift_confirmed=True``.
#: It is written here and nowhere else: ``ScanGeometry.hex`` and the APEX loader both take it
#: from this constant, and ``tests/test_scan_geometry.py`` fails if the literal appears in
#: another source file. If a reference ever contradicts it, this is the one line that changes.
OIM_ODD_ROW_SHIFT = 0.5


def refuse_masked_stamp(stamp) -> None:
    """Raise ``ValueError`` for a stamp (the dict form of a geometry, as a result or a dataset stores
    it) that says it has a valid mask.

    The dict carries no mask, so a geometry read back from it would be a different scan than the one
    stamped: the points the mask excluded would count. Every reader of a stamp calls this, so the
    routes that draw a map (``hex_raster.geometry_from_stamp``) and the grain tools
    (``scan_neighbours.topology_of_stamp``) refuse the same stamps, until a producer of such a stamp
    exists that also carries the mask."""
    if isinstance(stamp, dict) and stamp.get("has_valid_mask"):
        raise ValueError(
            "the valid mask cannot travel in the dict form (it is mask-free, like the API "
            "format); hand the geometry over with set_geometry instead")


@dataclass(frozen=True)
class ScanGeometry:
    """The index-space rectangle of a scan and the physical position of its points."""

    kind: str                       # "square" | "hex"
    n_rows: int
    n_cols: int
    step_x: float                   # um between neighbours within a row
    step_y: float                   # um between rows
    odd_row_shift: float = 0.0      # in units of step_x: 0 (square), +0.5 or -0.5 (hex)
    #: False while the direction of the row offset is an assumption (no file states it).
    shift_confirmed: bool = True
    #: False when the steps are placeholders (1.0), not measurements.
    steps_known: bool = True
    #: (n_rows, n_cols) bool, True = a measured point; None = every point is measured.
    #: Not part of equality (an array would make the generated __eq__ ambiguous).
    valid_mask: Optional[np.ndarray] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"scan kind must be one of {KINDS}, got {self.kind!r}")
        if int(self.n_rows) <= 0 or int(self.n_cols) <= 0:
            raise ValueError(f"a scan has at least one point, got {self.n_rows}x{self.n_cols}")
        if not (float(self.step_x) > 0 and float(self.step_y) > 0):
            raise ValueError(f"steps must be positive, got ({self.step_x}, {self.step_y})")
        if self.kind == "square" and float(self.odd_row_shift) != 0.0:
            raise ValueError("a square scan has no row offset")
        if self.kind == "hex":
            if abs(float(self.odd_row_shift)) != 0.5:
                raise ValueError(f"a hex scan shifts odd rows by +-0.5 steps, got {self.odd_row_shift}")
            if self.steps_known and not math.isclose(
                    float(self.step_y), float(self.step_x) * HEX_ROW_RATIO, rel_tol=HEX_RATIO_RTOL):
                raise ValueError(
                    f"rows of a hex scan are step_x*sqrt(3)/2 apart; got step_x={self.step_x}, "
                    f"step_y={self.step_y}")
        if self.valid_mask is not None:
            mask = np.array(self.valid_mask, dtype=bool, copy=True)
            if mask.shape != (int(self.n_rows), int(self.n_cols)):
                raise ValueError(f"valid_mask is {mask.shape}, the scan is {self.n_rows}x{self.n_cols}")
            mask.flags.writeable = False
            object.__setattr__(self, "valid_mask", mask)

    # ------------------------------------------------------------------ shape

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (int(self.n_rows), int(self.n_cols))

    @property
    def n_points(self) -> int:
        return int(self.n_rows) * int(self.n_cols)

    @property
    def n_valid(self) -> int:
        return self.n_points if self.valid_mask is None else int(np.count_nonzero(self.valid_mask))

    @property
    def is_hex(self) -> bool:
        return self.kind == "hex"

    # --------------------------------------------------------------- position

    def positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Physical (x, y) in um of every point, two ``(n_rows, n_cols)`` arrays."""
        rows = np.arange(self.n_rows, dtype=np.float64)[:, None]
        cols = np.arange(self.n_cols, dtype=np.float64)[None, :]
        shift = np.where(rows % 2 == 1, float(self.odd_row_shift) * float(self.step_x), 0.0)
        x = cols * float(self.step_x) + shift
        y = np.broadcast_to(rows * float(self.step_y), (self.n_rows, self.n_cols)).copy()
        return x, y

    def nearest(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """The (row, col) of the point closest to each physical position.

        Positions outside the scan give the closest edge point. An exact tie goes to
        the lower row and, within a row, to the lower column, which follows OIM's lower-column
        tie-break when it resamples a hex scan. Distances here are float64; OIM itself compares
        float32 distances, so a float64 recomputation does not reproduce every one of its
        choices (it differs on 3.32 % of the map of an OIM hexagonal scan).
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        sx, sy = float(self.step_x), float(self.step_y)
        last_r, last_c = self.n_rows - 1, self.n_cols - 1

        def in_row(r):
            shift = np.where(r % 2 == 1, float(self.odd_row_shift) * sx, 0.0)
            c = np.clip(np.ceil((x - shift) / sx - 0.5).astype(np.int64), 0, last_c)
            d2 = (x - (c * sx + shift)) ** 2 + (y - r * sy) ** 2
            return c, d2

        # The two rows bracketing y, and the rows next to them: past a corner both brackets clip
        # to the same edge row, and the row beside it can be closer in x (the half-step offset).
        r_lo = np.clip(np.floor(y / sy).astype(np.int64), 0, last_r)
        candidates = (np.clip(r_lo - 1, 0, last_r), r_lo,
                      np.clip(r_lo + 1, 0, last_r), np.clip(r_lo + 2, 0, last_r))
        best_r, (best_c, best_d) = candidates[0], in_row(candidates[0])
        for r in candidates[1:]:                  # ascending rows: a strict '<' keeps the lower on a tie
            c, d2 = in_row(r)
            better = d2 < best_d
            best_r = np.where(better, r, best_r)
            best_c = np.where(better, c, best_c)
            best_d = np.where(better, d2, best_d)
        return best_r, best_c

    # ------------------------------------------------------------------- crop

    def cropped(self, row0: int, col0: int, rows: int, cols: int) -> "ScanGeometry":
        """The geometry of the window ``[row0:row0+rows, col0:col0+cols]``.

        The window's own row 0 is the parent's row ``row0``. When ``row0`` is odd, the
        parent's shifted rows are the window's even rows, so measured from the window's
        first row the shifted ones are the window's odd rows moved the other way:
        the offset reverses.
        """
        row0, col0, rows, cols = int(row0), int(col0), int(rows), int(cols)
        if row0 < 0 or col0 < 0 or rows <= 0 or cols <= 0 \
                or row0 + rows > self.n_rows or col0 + cols > self.n_cols:
            raise ValueError(
                f"window rows {row0}-{row0 + rows}, cols {col0}-{col0 + cols} does not fit "
                f"a {self.n_rows}x{self.n_cols} scan")
        shift = float(self.odd_row_shift)
        if row0 % 2 == 1 and shift != 0.0:
            shift = -shift
        mask = None
        if self.valid_mask is not None:
            mask = np.array(self.valid_mask[row0:row0 + rows, col0:col0 + cols], copy=True)
        return replace(self, n_rows=rows, n_cols=cols, odd_row_shift=shift, valid_mask=mask)

    # ------------------------------------------------------------------- wire

    def to_dict(self) -> dict:
        """JSON-safe summary. The mask itself never crosses the wire."""
        return {
            "kind": self.kind,
            "n_rows": int(self.n_rows),
            "n_cols": int(self.n_cols),
            "grid_shape": [int(self.n_rows), int(self.n_cols)],
            "step_x": float(self.step_x),
            "step_y": float(self.step_y),
            "steps_known": bool(self.steps_known),
            "odd_row_shift": float(self.odd_row_shift) + 0.0,     # + 0.0: never "-0.0"
            "shift_confirmed": bool(self.shift_confirmed),
            "has_valid_mask": self.valid_mask is not None,
            "n_valid": self.n_valid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScanGeometry":
        refuse_masked_stamp(d)
        return cls(
            kind=str(d["kind"]),
            n_rows=int(d["n_rows"]),
            n_cols=int(d["n_cols"]),
            step_x=float(d["step_x"]),
            step_y=float(d["step_y"]),
            odd_row_shift=float(d.get("odd_row_shift", 0.0)),
            shift_confirmed=bool(d.get("shift_confirmed", True)),
            steps_known=bool(d.get("steps_known", True)),
        )

    # ----------------------------------------------------------- constructors

    @classmethod
    def square(cls, n_rows, n_cols, step_x, step_y=None, *, steps_known=True) -> "ScanGeometry":
        return cls("square", int(n_rows), int(n_cols), float(step_x),
                   float(step_x if step_y is None else step_y), 0.0, True, bool(steps_known))

    @classmethod
    def hex(cls, n_rows, n_cols, step_x, step_y=None, *, odd_row_shift=OIM_ODD_ROW_SHIFT,
            shift_confirmed=False) -> "ScanGeometry":
        """A hexagonal scan with rows of equal length. ``shift_confirmed`` stays False
        until a reference file states which way the odd rows are offset."""
        sy = float(step_x) * HEX_ROW_RATIO if step_y is None else float(step_y)
        return cls("hex", int(n_rows), int(n_cols), float(step_x), sy,
                   float(odd_row_shift), bool(shift_confirmed), True)


# ----------------------------------------------------------------- HDF5 attributes

#: ``grid_type`` as EDAX/OIM name it: the value stored for each scan kind.
GRID_TYPE_OF_KIND = {"square": "SqrGrid", "hex": "HexGrid"}
_KIND_OF_GRID_TYPE = {v: k for k, v in GRID_TYPE_OF_KIND.items()}
_HEX_ATTRS = ("odd_row_shift", "shift_confirmed", "step_x", "step_y")


def geometry_to_attrs(geometry: ScanGeometry) -> dict:
    """The attribute form of a geometry, as a checkpoint (``/metadata``) and the indexing result
    (``/Indexing``) store it: ``grid_type`` (``SqrGrid`` or ``HexGrid``), ``odd_row_shift``,
    ``shift_confirmed``, ``step_x`` and ``step_y``. The grid shape is stored beside it by the
    file, not here; the valid mask is not stored."""
    return {
        "grid_type": GRID_TYPE_OF_KIND[geometry.kind],
        "odd_row_shift": float(geometry.odd_row_shift) + 0.0,       # + 0.0: never -0.0
        "shift_confirmed": bool(geometry.shift_confirmed),
        "step_x": float(geometry.step_x),
        "step_y": float(geometry.step_y),
    }


def _attr_text(value) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def geometry_from_attrs(attrs, grid_shape) -> ScanGeometry:
    """The geometry stored by :func:`geometry_to_attrs`, for a scan of ``grid_shape`` (n_rows, n_cols).

    ``attrs`` is a mapping or an h5py attribute manager. Raises ``ValueError`` for an unknown
    ``grid_type``, for a ``HexGrid`` that lacks any of its attributes (a missing row offset is never
    defaulted: the wrong direction would shift every odd row of the export by a whole step), for a
    ``SqrGrid`` without steps, and when ``attrs`` also states a ``grid_shape`` that is not
    ``grid_shape``."""
    n_rows, n_cols = (int(v) for v in grid_shape)
    if "grid_shape" in attrs:
        stated = tuple(int(v) for v in np.asarray(attrs["grid_shape"]).ravel())
        if stated != (n_rows, n_cols):
            raise ValueError(f"the stamp states grid_shape {stated}, the scan is {(n_rows, n_cols)}")
    if "grid_type" not in attrs:
        raise ValueError("the stamp has no grid_type")
    grid_type = _attr_text(attrs["grid_type"])
    kind = _KIND_OF_GRID_TYPE.get(grid_type)
    if kind is None:
        raise ValueError(f"unknown grid_type {grid_type!r}, expected one of {sorted(_KIND_OF_GRID_TYPE)}")
    required = _HEX_ATTRS if kind == "hex" else ("step_x", "step_y")
    missing = [name for name in required if name not in attrs]
    if missing:
        raise ValueError(f"a {grid_type} stamp lacks {', '.join(missing)}")
    if kind == "square" and "odd_row_shift" in attrs and float(attrs["odd_row_shift"]) != 0.0:
        raise ValueError(f"a SqrGrid stamp states odd_row_shift {float(attrs['odd_row_shift'])}")
    return ScanGeometry(
        kind=kind, n_rows=n_rows, n_cols=n_cols,
        step_x=float(attrs["step_x"]), step_y=float(attrs["step_y"]),
        odd_row_shift=float(attrs["odd_row_shift"]) if kind == "hex" else 0.0,
        shift_confirmed=bool(attrs["shift_confirmed"]) if kind == "hex" else True,
    )


# --------------------------------------------------------------------- signal

_UNDEFINED_UNITS = ("", "<undefined>", "undefined", "None")


class GeometryMismatchError(ValueError):
    """The geometry a loader attached does not describe the grid the signal has."""


def plain(value):
    """A hyperspy metadata sub-tree as a plain dict; anything else unchanged."""
    return value.as_dictionary() if hasattr(value, "as_dictionary") else value


def apex_of_signal(signal) -> Optional[dict]:
    """The APEX measurement a loader paired with ``signal`` (``Signal.apex``), as plain data.

    This record is the ``apex`` of the ``/load`` answer, of a loaded-files entry, of the batch helper's
    memory and of the batch quick-load answer: ``{"h5_path": absolute path of the metadata file,
    "measurement_id": GUID, "measurement_name": "<area> / <map>", "pairing": "guid" | "count"}`` (how the
    pairing was proven). ``None`` when the file was loaded without APEX metadata."""
    try:
        value = plain(signal.metadata.get_item("Signal.apex", None))
    except Exception:  # noqa: BLE001 - a stand-in signal without metadata
        return None
    return dict(value) if isinstance(value, dict) else None


def _axes_grid(signal):
    """(n_rows, n_cols) of the signal's navigation axes, or None when it has no axes."""
    try:
        nav = list(signal.axes_manager.navigation_axes)       # hyperspy order: x (fast), then y
    except Exception:  # noqa: BLE001 - a stand-in signal
        return None
    if len(nav) >= 2:
        return int(nav[1].size), int(nav[0].size)
    if len(nav) == 1:
        return 1, int(nav[0].size)
    return 1, 1


def geometry_of_signal(signal) -> ScanGeometry:
    """The geometry a loader attached to ``signal`` (``Signal.scan_geometry``), else a
    square scan read off the navigation axes.

    This is the ONLY reader of the signal's copy: the loader hands the geometry to the
    route through the signal, the route registers it, and from then on everyone asks
    ``geometry_for(dataset)``.
    """
    try:
        mirror = plain(signal.metadata.get_item("Signal.scan_geometry", None))
    except Exception:  # noqa: BLE001 - a stand-in signal without metadata
        mirror = None
    if isinstance(mirror, dict):
        geometry = ScanGeometry.from_dict(mirror)
        grid = _axes_grid(signal)
        if grid is not None and grid != geometry.grid_shape:
            raise GeometryMismatchError(
                f"the loader's scan geometry is {geometry.grid_shape} but the signal's "
                f"navigation grid is {grid}")
        return geometry

    nav = list(signal.axes_manager.navigation_axes)       # hyperspy order: x (fast), then y
    if len(nav) >= 2:
        x_ax, y_ax = nav[0], nav[1]
        n_cols, n_rows = int(x_ax.size), int(y_ax.size)
    elif len(nav) == 1:
        x_ax = y_ax = nav[0]
        n_cols, n_rows = int(nav[0].size), 1
    else:
        return ScanGeometry.square(1, 1, 1.0, steps_known=False)
    sx, sy = float(x_ax.scale), float(y_ax.scale)
    units = str(getattr(x_ax, "units", ""))
    if units in _UNDEFINED_UNITS or not (sx > 0 and sy > 0):
        return ScanGeometry.square(n_rows, n_cols, 1.0, 1.0, steps_known=False)
    return ScanGeometry.square(n_rows, n_cols, sx, sy)


# ------------------------------------------------------------------- registry

_lock = threading.RLock()
_geometries: Dict[str, ScanGeometry] = {}


def set_geometry(name: str, geometry: ScanGeometry) -> None:
    with _lock:
        _geometries[name] = geometry
    logger.info("scan geometry for %r: %s %dx%d", name, geometry.kind,
                geometry.n_rows, geometry.n_cols)


def geometry_for(name: str) -> Optional[ScanGeometry]:
    """The geometry of dataset ``name``, or None when nothing is registered for it."""
    with _lock:
        return _geometries.get(name)


def drop_signal_mirror(signal) -> None:
    """Remove the loader's copy of the geometry from ``signal``'s metadata.

    The copy is a TRANSPORT from the loader to the registry. Left on the signal it goes stale the
    moment the signal is cropped (``inav``) or deep-copied: it keeps describing the whole scan, and
    the next ``geometry_of_signal`` on the crop would raise ``GeometryMismatchError``. So whoever
    reads it, drops it. (hyperspy's metadata tree has no ``remove_item``; ``del`` on the node does
    it.) A stand-in without that metadata has nothing to drop."""
    try:
        del signal.metadata.Signal.scan_geometry
    except Exception:  # noqa: BLE001 - no mirror, or a stand-in signal
        pass


def register_from_signal(name: str, signal) -> Optional[ScanGeometry]:
    """Register the geometry of a freshly loaded ``signal`` under ``name``, and take the loader's
    copy off the signal (see :func:`drop_signal_mirror`)."""
    try:
        geometry = geometry_of_signal(signal)
    except GeometryMismatchError:
        raise                       # a loader that disagrees with its own signal is a bug: fail loud
    except Exception:  # noqa: BLE001 - a stand-in signal: no geometry is "unknown", not an error
        logger.warning("could not read the scan geometry of %r", name, exc_info=True)
        return None
    drop_signal_mirror(signal)
    set_geometry(name, geometry)
    return geometry


def derive(name: str, parent_name: str) -> Optional[ScanGeometry]:
    """A deepcopy sits exactly where its parent does."""
    with _lock:
        parent = _geometries.get(parent_name)
        if parent is None:
            logger.warning("derive(%r): parent %r has no geometry", name, parent_name)
            return None
        _geometries[name] = parent
        return parent


def derive_cropped(name: str, parent_name: str, window) -> Optional[ScanGeometry]:
    """A crop: the parent's geometry cut by ``window``, a ``CropWindow`` on the PARENT's
    grid (``row0``, ``col0``, ``rows``, ``cols``)."""
    with _lock:
        parent = _geometries.get(parent_name)
        if parent is None:
            logger.warning("derive_cropped(%r): parent %r has no geometry", name, parent_name)
            return None
        child = parent.cropped(window.row0, window.col0, window.rows, window.cols)
        _geometries[name] = child
        return child


def clear(name: str) -> None:
    with _lock:
        _geometries.pop(name, None)


def clear_all() -> None:
    """Forget every registered geometry, and with them the tables derived from geometries: the
    neighbour tables (``scan_neighbours``) and the cell index of the hexagonal raster (``hex_raster``)
    are big arrays that nothing but the scan just closed would ask for again. Both modules are
    imported here, not above: ``hex_raster`` imports this one."""
    with _lock:
        _geometries.clear()
    from backend.api.services import hex_raster, scan_neighbours
    scan_neighbours.clear_caches()
    hex_raster.clear_caches()


def snapshot(names) -> Dict[str, ScanGeometry]:
    """The geometries of ``names``, for the per-file stash. Frozen objects: no copy needed."""
    with _lock:
        return {n: _geometries[n] for n in names if n in _geometries}


def restore(snap: Dict[str, ScanGeometry], skip: Optional[str] = None) -> None:
    """Put stashed geometries back. A name that has one already keeps it (the freshly
    loaded raw dataset, or a live dataset), and ``skip`` is never touched."""
    with _lock:
        for name, geometry in (snap or {}).items():
            if name != skip and name not in _geometries:
                _geometries[name] = geometry
