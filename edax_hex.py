"""EDAX HexGrid support — read hexagonally sampled scans as square maps.

OIM Analysis can acquire on a hexagonal grid, and nothing downstream of us
can work with one: kikuchipy refuses outright (``OSError: Only square grids
are supported, not HexGrid``) and our own fallback loader trips over the
padding EDAX writes. This module makes those files openable by resampling
them onto a square grid the same way OIM Analysis does it itself.

How EDAX stores a hex scan (measured on real files, see
``tasks/edax-hexgrid-format-facts.md``):

* Rows alternate. A *long* row holds ``nColumns`` points starting at x=0; a
  *short* row holds ``nColumns - 1`` points starting at x=step_x/2. Rows are
  ``step_y = step_x * sqrt(3)/2`` apart.
* The per-point arrays (CI, IQ, Euler angles, positions …) are padded to a
  full ``nRows x nColumns`` rectangle. Padding slots carry
  ``x = y = -1111111.0``; the header says so in its own ``Notes`` field.
* The ``Pattern`` dataset holds **only the real points**, with no padding.
  So a pattern's index is *not* its index in the padded grid — a plain
  reshape silently shifts everything from the first short row onwards.

The conversion is nearest neighbour, which is the only honest choice for
diffraction patterns: an interpolated pattern is an invented measurement.
That is also exactly what OIM does — verified byte for byte against a square
file OIM itself produced from one of these scans.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# EDAX marks a grid slot with no measurement as x = y = -1111111.0.
_SENTINEL_LIMIT = -1.0e5

# Relative tolerance when checking measured positions against the step sizes
# the header claims. Positions are stored as float32.
_POS_RTOL = 1e-3


@dataclass(frozen=True)
class HexGridInfo:
    """What the file header says about a hexagonal scan."""

    path: str
    scan_group: str
    n_cols: int          # points in a long row (header "nColumns")
    n_rows: int          # number of rows (header "nRows")
    step_x: float        # µm between points within a row
    step_y: float        # µm between rows
    sy: int              # pattern height
    sx: int              # pattern width
    n_patterns: int      # patterns actually stored (real points only)

    @property
    def n_points_padded(self) -> int:
        return self.n_rows * self.n_cols


@dataclass(frozen=True)
class HexLayout:
    """The row structure as *measured* from the stored point positions."""

    n_cols: int
    n_rows: int
    step_x: float
    step_y: float
    long_row_parity: int      # 0 -> even rows are the long ones
    x0_long: float            # x of the first point of a long row
    x0_short: float           # x of the first point of a short row
    valid: np.ndarray         # (n_rows*n_cols,) bool, padded order
    pattern_of_point: np.ndarray  # (n_rows*n_cols,) int64, -1 on padding
    x: np.ndarray             # (n_rows*n_cols,) stored x, padding left as-is
    y_of_row: np.ndarray      # (n_rows,) stored y of each row

    @property
    def n_padding(self) -> int:
        return int((~self.valid).sum())

    def points_in_row(self, rows):
        """Number of measured points in the given row(s)."""
        long_row = (np.asarray(rows) % 2) == self.long_row_parity
        return np.where(long_row, self.n_cols, self.n_cols - 1)


@dataclass(frozen=True)
class SquareMap:
    """Nearest-neighbour mapping from a square output grid onto hex points."""

    n_rows: int
    n_cols: int
    step: float
    point_index: np.ndarray    # (n_rows*n_cols,) int64 into the padded arrays
    pattern_index: np.ndarray  # (n_rows*n_cols,) int64 into the Pattern dataset


# --------------------------------------------------------------------------
# Reading the file
# --------------------------------------------------------------------------

def _scan_group(h5file) -> str:
    """Name of the top-level group holding the scan (EDAX names it freely)."""
    for key in h5file.keys():
        try:
            group = h5file[key]
        except Exception:  # pragma: no cover - unreadable link
            continue
        if hasattr(group, "keys") and "EBSD" in group:
            return key
    raise ValueError("No top-level group with an 'EBSD' subgroup found")


def _header_value(header, key, default=None):
    if key not in header:
        return default
    value = header[key][()]
    value = np.ravel(value)
    if value.size == 0:
        return default
    value = value[0]
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def is_edax_hex_file(path) -> bool:
    """True if *path* is an EDAX h5 scan acquired on a hexagonal grid."""
    path = str(path)
    try:
        import h5py
    except Exception:  # pragma: no cover - h5py is a hard dependency
        return False
    try:
        if not h5py.is_hdf5(path):
            return False
        with h5py.File(path, "r") as f:
            scan = _scan_group(f)
            grid = _header_value(f[scan]["EBSD/Header"], "Grid Type")
            return str(grid) == "HexGrid"
    except Exception:
        # Not an EDAX h5ebsd file, unreadable, or truncated — not our business.
        return False


def read_hex_grid_info(path) -> HexGridInfo:
    """Read the hex-scan header. Raises ``ValueError`` if it is not one."""
    import h5py

    path = str(path)
    with h5py.File(path, "r") as f:
        scan = _scan_group(f)
        header = f[scan]["EBSD/Header"]
        grid = str(_header_value(header, "Grid Type"))
        if grid != "HexGrid":
            raise ValueError(f"{Path(path).name} is not a HexGrid scan (Grid Type={grid!r})")

        n_cols = int(_header_value(header, "nColumns"))
        n_rows = int(_header_value(header, "nRows"))
        step_x = float(_header_value(header, "Step X", 1.0))
        step_y = float(_header_value(header, "Step Y", 1.0))
        sy = int(_header_value(header, "Pattern Height"))
        sx = int(_header_value(header, "Pattern Width"))
        n_patterns = int(f[scan]["EBSD/Data/Pattern"].shape[0])

    if n_cols < 2 or n_rows < 2:
        raise ValueError(
            f"{Path(path).name}: implausible hex grid {n_cols}x{n_rows}")
    if not (step_x > 0 and step_y > 0):
        raise ValueError(
            f"{Path(path).name}: implausible step sizes ({step_x}, {step_y})")

    return HexGridInfo(path, scan, n_cols, n_rows, step_x, step_y,
                       sy, sx, n_patterns)


# --------------------------------------------------------------------------
# Measuring the layout (never assumed — always read from the file)
# --------------------------------------------------------------------------

def measure_hex_layout(x, y, n_cols, n_rows, step_x, step_y) -> HexLayout:
    """Derive the hex row structure from the stored point positions.

    Everything is checked rather than assumed, and anything that does not fit
    the alternating-row model raises instead of being papered over: a wrong
    layout produces a plausible-looking but wholly misaligned map, which is
    far worse than a refusal to open the file.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size != n_rows * n_cols or y.size != n_rows * n_cols:
        raise ValueError(
            f"Position arrays hold {x.size} points but the header says "
            f"{n_rows}x{n_cols} = {n_rows * n_cols}")

    xg = x.reshape(n_rows, n_cols).astype(np.float64)
    yg = y.reshape(n_rows, n_cols).astype(np.float64)
    valid = (xg > _SENTINEL_LIMIT) & (yg > _SENTINEL_LIMIT)

    counts = valid.sum(axis=1)
    # Every row must be a filled prefix — padding sits at the end of a row.
    prefix = np.arange(n_cols)[None, :] < counts[:, None]
    if not np.array_equal(valid, prefix):
        bad = int(np.argmax((valid != prefix).any(axis=1)))
        raise ValueError(
            f"Hex row {bad} has holes in the middle: the padding of an EDAX "
            f"hex scan is expected at the end of a row only")

    lengths = set(int(c) for c in counts)
    if lengths - {n_cols, n_cols - 1}:
        raise ValueError(
            f"Hex rows must hold {n_cols} or {n_cols - 1} points, found "
            f"lengths {sorted(lengths)}")
    if n_cols - 1 not in lengths:
        raise ValueError(
            "No short rows found — this is not a hexagonal grid (every row "
            f"holds all {n_cols} points)")

    is_long = counts == n_cols
    long_row_parity = 0 if is_long[0] else 1
    expected_long = (np.arange(n_rows) % 2) == long_row_parity
    if not np.array_equal(is_long, expected_long):
        bad = int(np.argmax(is_long != expected_long))
        raise ValueError(
            f"Hex rows must alternate long/short; row {bad} breaks the pattern")

    x0_long = float(np.median(xg[expected_long, 0]))
    x0_short = float(np.median(xg[~expected_long, 0]))
    offset = abs(x0_short - x0_long)
    if not math.isclose(offset, step_x / 2, rel_tol=1e-2, abs_tol=step_x * 1e-2):
        raise ValueError(
            f"Short rows are offset by {offset:.6g} µm, expected half a step "
            f"({step_x / 2:.6g} µm)")

    # Spot-check that positions really follow the step sizes claimed.
    _check_positions(xg, yg, valid, counts, expected_long,
                     x0_long, x0_short, step_x, step_y)

    flat_valid = valid.reshape(-1)
    pattern_of_point = np.full(flat_valid.size, -1, dtype=np.int64)
    pattern_of_point[flat_valid] = np.arange(int(flat_valid.sum()), dtype=np.int64)

    return HexLayout(
        n_cols=n_cols, n_rows=n_rows, step_x=float(step_x), step_y=float(step_y),
        long_row_parity=long_row_parity, x0_long=x0_long, x0_short=x0_short,
        valid=flat_valid, pattern_of_point=pattern_of_point,
        x=xg.reshape(-1), y_of_row=yg[:, 0].copy(),
    )


def _check_positions(xg, yg, valid, counts, is_long,
                     x0_long, x0_short, step_x, step_y) -> None:
    """Verify stored positions against the header's step sizes."""
    tol = max(step_x, step_y) * _POS_RTOL

    rows = np.arange(xg.shape[0])
    y_expected = rows * step_y
    y_measured = np.array([yg[r, 0] for r in rows])
    if np.max(np.abs(y_measured - y_expected)) > tol:
        worst = int(np.argmax(np.abs(y_measured - y_expected)))
        raise ValueError(
            f"Row {worst} sits at y={y_measured[worst]:.6g} µm but a spacing "
            f"of {step_y:.6g} µm puts it at {y_expected[worst]:.6g} µm")

    cols = np.arange(xg.shape[1])
    for r in rows:
        n = int(counts[r])
        x0 = x0_long if is_long[r] else x0_short
        expected = x0 + cols[:n] * step_x
        if np.max(np.abs(xg[r, :n] - expected)) > tol:
            raise ValueError(
                f"Points in row {r} do not sit on a {step_x:.6g} µm raster")


# --------------------------------------------------------------------------
# Hex -> square nearest neighbour
# --------------------------------------------------------------------------

def build_square_map(layout: HexLayout, out_step: Optional[float] = None) -> SquareMap:
    """Map every point of a square grid onto its nearest hex measurement.

    The output step defaults to the hex in-row spacing, which is what OIM
    Analysis uses, and the output extent covers the scan
    (``round(span / step) + 1`` points per axis, again matching OIM).

    Distances are measured against the positions **as stored in the file**
    (float32) rather than against an idealised raster. That is not
    pedantry: a square point often falls exactly half way between two
    points of a staggered row, and which of the two wins is decided by
    float32 rounding. Recomputing the raster in double precision turns
    those decisions into exact ties and picks differently from OIM on
    ~3 % of the map — measured, not feared.
    """
    step = float(out_step) if out_step else float(layout.step_x)
    if step <= 0:
        raise ValueError(f"Output step must be positive, got {step}")

    x_max = layout.x0_long + (layout.n_cols - 1) * layout.step_x
    y_max = (layout.n_rows - 1) * layout.step_y
    n_cols = int(round(x_max / step)) + 1
    n_rows = int(round(y_max / step)) + 1

    # Generate the output coordinates the way OIM writes them: single
    # precision, index times step (verified bit-identical to the X/Y
    # Position arrays in OIM's own square export).
    step32 = np.float32(step)
    x_axis = (np.arange(n_cols, dtype=np.float32) * step32).astype(np.float64)
    y_axis = (np.arange(n_rows, dtype=np.float32) * step32).astype(np.float64)

    rr, cc = np.divmod(np.arange(n_rows * n_cols, dtype=np.int64), n_cols)
    xs, ys = x_axis[cc], y_axis[rr]

    # The nearest point always lies in one of the two rows bracketing ys:
    # any other row is at least 1.5 * step_y away in y, which is further
    # than the worst case within an adjacent row.
    lower = np.clip(np.floor(ys / layout.step_y).astype(np.int64),
                    0, layout.n_rows - 1)
    upper = np.clip(lower + 1, 0, layout.n_rows - 1)

    cand_lo, d2_lo = _nearest_in_row(lower, xs, ys, layout)
    cand_hi, d2_hi = _nearest_in_row(upper, xs, ys, layout)
    # Ties go to the lower row, mirroring a flat row-major argmin.
    point_index = np.where(d2_lo <= d2_hi, cand_lo, cand_hi)

    pattern_index = layout.pattern_of_point[point_index]
    if np.any(pattern_index < 0):  # pragma: no cover - guarded by construction
        raise ValueError("Square grid mapped onto a padding slot")

    return SquareMap(n_rows=n_rows, n_cols=n_cols, step=step,
                     point_index=point_index, pattern_index=pattern_index)


def _nearest_in_row(rows, xs, ys, layout: HexLayout):
    """Closest point within the given hex row, plus its squared distance.

    Both neighbours bracketing ``xs`` are evaluated against their stored
    positions and the closer one wins; an exact tie goes to the lower
    column, which is the choice OIM makes.
    """
    base = rows * layout.n_cols
    last = layout.points_in_row(rows) - 1
    x0 = layout.x[base]

    left = np.clip(np.floor((xs - x0) / layout.step_x).astype(np.int64), 0, last)
    right = np.clip(left + 1, 0, last)

    dx_left = np.abs(layout.x[base + left] - xs)
    dx_right = np.abs(layout.x[base + right] - xs)
    take_left = dx_left <= dx_right

    cols = np.where(take_left, left, right)
    dx = np.where(take_left, dx_left, dx_right)
    dy = ys - layout.y_of_row[rows]
    return base + cols, dx * dx + dy * dy


# --------------------------------------------------------------------------
# Cached per-file maps
# --------------------------------------------------------------------------

_MAP_CACHE: "dict[tuple, tuple[HexGridInfo, HexLayout, SquareMap]]" = {}
_MAP_CACHE_MAX = 4


def _cache_key(path: str, out_step: Optional[float]):
    try:
        stat = os.stat(path)
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None
    return (os.path.abspath(path), stamp, out_step)


def square_map_for_file(path, out_step: Optional[float] = None):
    """Return ``(HexGridInfo, SquareMap)`` for a hex file, cached per file."""
    info, _, smap = _hex_bundle(path, out_step)
    return info, smap


def _hex_bundle(path, out_step: Optional[float] = None):
    import h5py

    path = str(path)
    key = _cache_key(path, out_step)
    cached = _MAP_CACHE.get(key)
    if cached is not None:
        return cached

    info = read_hex_grid_info(path)
    with h5py.File(path, "r") as f:
        data = f[info.scan_group]["EBSD/Data"]
        missing = [n for n in ("X Position", "Y Position") if n not in data]
        if missing:
            raise ValueError(
                f"{Path(path).name}: a hex scan needs {' and '.join(missing)} "
                f"to work out which measurement sits where; the file has "
                f"neither a square grid nor those coordinates")
        x = np.asarray(data["X Position"])
        y = np.asarray(data["Y Position"])

    layout = measure_hex_layout(x, y, info.n_cols, info.n_rows,
                                info.step_x, info.step_y)
    n_valid = int(layout.valid.sum())
    if n_valid != info.n_patterns:
        raise ValueError(
            f"{Path(path).name}: {n_valid} measured points but "
            f"{info.n_patterns} patterns stored — cannot line them up")

    smap = build_square_map(layout, out_step)

    if len(_MAP_CACHE) >= _MAP_CACHE_MAX:
        _MAP_CACHE.clear()
    _MAP_CACHE[key] = (info, layout, smap)
    return info, layout, smap


def remap_point_values(path, values, out_step: Optional[float] = None):
    """Resample a padded per-point channel (CI, IQ, …) onto the square grid.

    Returns a ``(n_rows, n_cols)`` array, or ``None`` if *values* does not
    have one entry per grid slot.
    """
    info, _, smap = _hex_bundle(path, out_step)
    values = np.asarray(values).reshape(-1)
    if values.size != info.n_points_padded:
        return None
    return values[smap.point_index].reshape(smap.n_rows, smap.n_cols)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _gather_eager(dset, pattern_index, n_rows, n_cols):
    """Read the mapped patterns row by row so peak memory stays bounded."""
    sy, sx = dset.shape[1], dset.shape[2]
    out = np.empty((n_rows * n_cols, sy, sx), dtype=dset.dtype)
    for r in range(n_rows):
        idx = pattern_index[r * n_cols:(r + 1) * n_cols]
        uniq, inverse = np.unique(idx, return_inverse=True)
        out[r * n_cols:(r + 1) * n_cols] = dset[uniq][inverse]
    return out.reshape(n_rows, n_cols, sy, sx)


def _gather_lazy(dset, pattern_index, n_rows, n_cols):
    """Same mapping as :func:`_gather_eager`, as one dask array per output row."""
    import dask.array as da

    sy, sx = dset.shape[1], dset.shape[2]
    source = da.from_array(dset, chunks=("auto", sy, sx))
    rows = [source[pattern_index[r * n_cols:(r + 1) * n_cols]]
            for r in range(n_rows)]
    return da.stack(rows, axis=0)


def load_edax_hex(path, lazy: bool = False, verbose: bool = True):
    """Load an EDAX HexGrid scan as a square-gridded kikuchipy EBSD signal.

    The signal is assembled exactly the way kikuchipy assembles a native
    square EDAX scan — same metadata, same axis names, same detector with
    the pattern centre from the file header — so everything downstream sees
    an ordinary square map.
    """
    import h5py
    from kikuchipy.io._io import _dict2signal
    from kikuchipy.io.plugins._h5ebsd import H5EBSDReader, _hdf5group2dict
    from kikuchipy.detectors import EBSDDetector
    from orix.crystal_map import CrystalMap

    path = str(path)
    file_name = Path(path).name
    info, layout, smap = _hex_bundle(path)

    if verbose:
        logger.info(
            "Loading EDAX hex scan %s (%s): %dx%d hex points -> %dx%d square "
            "at %.4g µm", file_name, "lazy" if lazy else "eager",
            info.n_cols, info.n_rows, smap.n_cols, smap.n_rows, smap.step)

    f = h5py.File(path, "r")
    try:
        group = f[info.scan_group]
        header = _hdf5group2dict(group["EBSD/Header"], recursive=True)
        sem = (_hdf5group2dict(group["SEM-PRIAS Images/Header"])
               if "SEM-PRIAS Images" in group else {})
        # Read this here, not below: on the eager path the file is closed by
        # the time the metadata is assembled, and a membership test on a
        # closed h5py file quietly answers False instead of raising.
        version = _safe_str(f["Version"][()]) if "Version" in f else None
        dset = group["EBSD/Data/Pattern"]

        if lazy:
            data = _gather_lazy(dset, smap.pattern_index, smap.n_rows, smap.n_cols)
        else:
            data = _gather_eager(dset, smap.pattern_index, smap.n_rows, smap.n_cols)
    finally:
        # A lazy dask array keeps reading from this dataset, so the file has
        # to stay open in that case — the signal owns it from here on.
        if not lazy:
            f.close()

    ny, nx = smap.n_rows, smap.n_cols
    sy, sx = info.sy, info.sx
    px_size = 1.0

    pc_cal = header.get("Pattern Center Calibration", {}) or {}
    scan_dict = {
        "data": data,
        "axes": H5EBSDReader.get_axes_list((ny, nx, sy, sx),
                                           (smap.step, smap.step, px_size)),
        "metadata": {
            "Acquisition_instrument": {
                "SEM": {
                    "working_distance": header.get("Working Distance"),
                    "magnification": sem.get("Mag"),
                },
            },
            "General": {"original_filename": file_name, "title": Path(path).stem},
            "Signal": {"signal_type": "EBSD", "record_by": "image"},
        },
        "original_metadata": {
            "manufacturer": "EDAX",
            "version": version,
            "hex_to_square": {
                "source_grid": "HexGrid",
                "hex_shape": (info.n_rows, info.n_cols),
                "hex_step": (info.step_y, info.step_x),
                "square_step": smap.step,
                "resampling": "nearest",
            },
            **header,
        },
        "xmap": CrystalMap.empty(shape=(ny, nx), step_sizes=(smap.step, smap.step)),
        "detector": EBSDDetector(
            shape=(sy, sx),
            px_size=px_size,
            tilt=header.get("Camera Elevation Angle", 0.0),
            azimuthal=header.get("Camera Azimuthal Angle", 0.0),
            sample_tilt=header.get("Sample Tilt", 70.0),
            pc=(pc_cal.get("x-star", 0.5),
                pc_cal.get("y-star", 0.5),
                pc_cal.get("z-star", 0.5)),
            convention="edax",
        ),
    }

    signal = _dict2signal(scan_dict, lazy=lazy)
    try:
        signal.metadata.set_item("Signal.hex_resampled", True)
    except Exception:  # pragma: no cover - metadata is best effort
        logger.debug("Could not tag hex_resampled on %s", file_name, exc_info=True)
    return signal


def _safe_str(value):
    value = np.ravel(value)
    if value.size == 0:
        return None
    value = value[0]
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)
