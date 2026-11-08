"""The display raster of a hexagonal scan.

A hexagonal scan is stored as the index rectangle it is (rows of equal length, odd rows offset), and a map
drawn on that rectangle is wrong in both axes. A map layer of a hexagonal scan is therefore rendered, for
display only, on a square raster of step_x / factor per pixel: every pixel takes the value of the nearest
scan point and is drawn only inside that point's hexagon, so the outside of the scan stays transparent.
Nothing reads the raster back as data; selections, masks and probes are always in index space.

ScanGeometry.nearest is the rule (nearest point, a tie to the lower row, then the lower column). It is held
against a plain argmin over every point (tests/hex_raster_oracle.py) and mirrored operation for operation
by frontend/src/services/hexGeometry.js (tests/fixtures/hex_raster_cases.json holds the two together).
The tie rule holds for exact ties, which a step that is exact in binary (2.0, 0.5, ...) produces at factor 1.
With a step carrying rounding noise (1.7, say) a pixel on the border between two cells can fall to either
one, in the last bit of the distances; the raster is for display and does not depend on it. What it does
depend on is that such a pixel falls into one of the two and not into none, which is what EDGE_TOLERANCE
in the inside test is for.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from backend.api.services.scan_geometry import ScanGeometry

#: Tried in this order; the first whose raster has at most MAX_RASTER_PIXELS pixels is used.
FACTORS = (4, 2, 1)
MAX_RASTER_PIXELS = 16_000_000
#: Raster pixels handled at a time when the cell index is built (bounds the temporaries, whatever the
#: width of the raster); a chunk is a whole number of rows, at least one.
CHUNK_PIXELS = 1_000_000
#: A pixel centre on the border of a hexagon belongs to it, and the border is only known to the last bits of
#: the positions (a step of 1.7 is not exact in binary). The inside test therefore allows this fraction of a
#: step beyond the border: far above the rounding noise (about 1e-16 of the largest coordinate) and far below
#: the distance a pixel centre keeps from a border it does not lie on, so it takes in the pixels the borders
#: pass through and nothing else. Without it a
#: pixel on the border between two cells can fall into none (factor 1 only, where centres reach borders).
#: Mirrored by EDGE_TOLERANCE in frontend/src/services/hexGeometry.js.
EDGE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RasterSpec:
    factor: int
    height: int
    width: int
    step_um: float          # side of a display pixel, micrometres
    x0: float               # position of the raster's left edge on the sample, micrometres
    y0: float               # ... and of its top edge

    def to_dict(self) -> dict:
        return {"kind": "hex_raster", "factor": self.factor, "shape": [self.height, self.width],
                "step_um": self.step_um, "x0": self.x0, "y0": self.y0}


def _require_hex(geometry) -> None:
    if geometry is None or geometry.kind != "hex":
        raise ValueError("only a hexagonal scan has a display raster")


def _extents(geometry, factor: int) -> RasterSpec:
    a, b, s = float(geometry.step_x), float(geometry.step_y), float(geometry.odd_row_shift)
    offsets = (0.0, s * a) if geometry.n_rows >= 2 else (0.0,)
    x_lo = min(offsets) - a / 2
    x_hi = (geometry.n_cols - 1) * a + max(offsets) + a / 2
    rho = 2.0 * b / 3.0
    y_lo, y_hi = -rho, (geometry.n_rows - 1) * b + rho
    d = a / factor
    return RasterSpec(factor, math.ceil((y_hi - y_lo) / d - 1e-9), math.ceil((x_hi - x_lo) / d - 1e-9),
                      d, x_lo, y_lo)


def choose_factor(geometry) -> int:
    _require_hex(geometry)
    for factor in FACTORS:
        spec = _extents(geometry, factor)
        if spec.height * spec.width <= MAX_RASTER_PIXELS:
            return factor
    return FACTORS[-1]


def raster_spec(geometry, factor: Optional[int] = None) -> RasterSpec:
    _require_hex(geometry)
    return _extents(geometry, factor or choose_factor(geometry))


def _assign(geometry, x, y):
    """(row, col, inside) for physical positions ``x``, ``y`` (arrays or 0-d): the nearest point and
    whether the position lies inside that point's hexagon."""
    a, b = float(geometry.step_x), float(geometry.step_y)
    row, col = geometry.nearest(x, y)
    shift = np.where(row % 2 == 1, float(geometry.odd_row_shift) * a, 0.0)
    dx = np.abs(x - (col * a + shift))
    dy = np.abs(y - row * b)
    rho = 2.0 * b / 3.0
    tol = EDGE_TOLERANCE * a
    return row, col, (dx <= a / 2 + tol) & (dy <= rho * (1.0 - dx / a) + tol)


def _build(geometry, spec: RasterSpec) -> np.ndarray:
    out = np.empty((spec.height, spec.width), dtype=np.int32)
    xs = spec.x0 + (np.arange(spec.width, dtype=np.float64) + 0.5) * spec.step_um
    chunk_rows = max(1, CHUNK_PIXELS // spec.width)
    for i0 in range(0, spec.height, chunk_rows):
        i1 = min(spec.height, i0 + chunk_rows)
        ys = spec.y0 + (np.arange(i0, i1, dtype=np.float64) + 0.5) * spec.step_um
        x, y = np.meshgrid(xs, ys)
        row, col, inside = _assign(geometry, x, y)
        out[i0:i1] = np.where(inside, row * geometry.n_cols + col, -1)
    out.flags.writeable = False
    return out


@lru_cache(maxsize=4)
def _cell_index_cached(geometry, factor: int) -> np.ndarray:
    """Keyed by the (frozen, hashable) geometry: the valid mask is not part of its identity."""
    return _build(geometry, _extents(geometry, factor))


def clear_caches() -> None:
    """Drop the cell indexes that were built (``scan_geometry.clear_all`` calls this)."""
    _cell_index_cached.cache_clear()


def cell_index(geometry, spec: Optional[RasterSpec] = None) -> np.ndarray:
    _require_hex(geometry)
    if spec is None:
        spec = raster_spec(geometry)
    elif spec != raster_spec(geometry, factor=spec.factor):
        raise ValueError("the raster spec belongs to another scan geometry")
    return _cell_index_cached(geometry, spec.factor)


def to_display(values, geometry, *, fill=0, spec: Optional[RasterSpec] = None) -> np.ndarray:
    _require_hex(geometry)
    values = np.asarray(values)
    if values.ndim < 2 or tuple(values.shape[:2]) != tuple(geometry.grid_shape):
        raise ValueError(f"values are {tuple(values.shape[:2])}, the scan is {tuple(geometry.grid_shape)}")
    idx = cell_index(geometry, spec)
    inside = idx >= 0
    flat = values.reshape((geometry.n_points,) + values.shape[2:])
    out = np.full(idx.shape + values.shape[2:], fill, dtype=values.dtype)
    out[inside] = flat[idx[inside]]
    return out


def png_to_display(png_base64: str, geometry, spec: Optional[RasterSpec] = None) -> str:
    """An image of the index grid (any PNG mode) as an RGBA PNG of the raster, transparent outside."""
    from PIL import Image
    rgba = np.asarray(Image.open(io.BytesIO(base64.b64decode(png_base64))).convert("RGBA"))
    buf = io.BytesIO()
    Image.fromarray(to_display(rgba, geometry, spec=spec)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def cell_at_pixel(geometry, spec: RasterSpec, i: int, j: int) -> Optional[Tuple[int, int]]:
    x = spec.x0 + (j + 0.5) * spec.step_um
    y = spec.y0 + (i + 0.5) * spec.step_um
    row, col, inside = _assign(geometry, np.float64(x), np.float64(y))
    return (int(row), int(col)) if bool(inside) else None


def pixel_of_cell(geometry, spec: RasterSpec, row: int, col: int) -> Tuple[int, int]:
    """The (i, j) of the pixel that holds the centre of the cell, meant for placing a marker there.
    At factor 1 that pixel can lie in a neighbouring cell or in none (12 of the 36 cells of a 6 x 6 scan);
    whoever needs the cell a pixel belongs to asks ``cell_at_pixel``."""
    a = float(geometry.step_x)
    shift = float(geometry.odd_row_shift) * a if row % 2 == 1 else 0.0
    x, y = col * a + shift, row * float(geometry.step_y)
    return (math.floor((y - spec.y0) / spec.step_um), math.floor((x - spec.x0) / spec.step_um))


def display_info(geometry) -> Optional[dict]:
    """What a response says about the raster it drew, or None for a scan that has none."""
    if geometry is None or geometry.kind != "hex":
        return None
    return raster_spec(geometry).to_dict()


def geometry_from_stamp(stamp) -> Optional[ScanGeometry]:
    """The geometry of a hexagonal stamp (``result.metadata["scan_geometry"]``, an analysis dataset's),
    or None for a missing, unknown or square one. A stamp that says it has a valid mask raises
    ValueError: a geometry with a valid mask has no place in the dict, which carries no mask, and the
    error message says so."""
    if not isinstance(stamp, dict) or stamp.get("kind") != "hex":
        return None
    return ScanGeometry.from_dict(stamp)
