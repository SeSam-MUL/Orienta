"""Resample a hexagonal scan to a square grid, point for point as OIM Analysis does it.

A ``.ctf`` file and a square ``.ang`` file hold a square grid, so a hexagonal scan has to be
resampled before it can be written to them. The rule is OIM's own: a square grid with the in-row step,
``round(span / step) + 1`` cells along each axis, and every cell takes the nearest scan point, with an
exact tie going to the lower row and then the lower column. It is the rule ``edax_hex.build_square_map``
applies to OIM's hexagonal files (``tests/test_edax_hex.py`` holds it against OIM's own square export);
this module feeds that function the geometry of a ``ScanGeometry`` instead of the positions stored in a
file, so a scan that never came from OIM, an Orienta ``.h5`` for one, is resampled the same way.

``ScanGeometry.nearest`` is not used for this. It compares float64 distances, where OIM decides on
float32 ones: on a real scan the two choose differently for a few per cent of the cells.

The shape of the grid: a hexagonal scan of ``n_rows x n_cols`` index points has rows of ``n_cols`` points
whose odd rows are shifted by half a step. The shifted rows reach half a step past the last column, and
that half step gets no column (the grid has ``n_cols`` columns, as OIM's does). Rows are ``step_y`` apart,
so the grid has ``round((n_rows - 1) * step_y / step_x) + 1`` rows.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np

import edax_hex
from backend.api.services.scan_geometry import ScanGeometry


@dataclass(frozen=True)
class RectHexLayout(edax_hex.HexLayout):
    """A ``HexLayout`` whose rows have the lengths they are given, not "long" and "short".

    ``edax_hex.HexLayout.points_in_row`` knows two lengths, ``n_cols`` and ``n_cols - 1``, which is what
    OIM's own files have. A geometry from another source can state any length per row (a mask padded at
    the row ends); this subclass carries them. ``HexLayout`` itself, and with it the OIM load path, is
    unchanged."""

    row_lengths: np.ndarray          # (n_rows,) int64, measured points of each row

    def points_in_row(self, rows):
        return self.row_lengths[np.asarray(rows)]


@dataclass(frozen=True)
class SquareResampling:
    """Where each cell of the square grid takes its value from."""

    step: float                 # step of the square grid, um
    n_rows: int                 # rows of the square grid
    n_cols: int                 # columns of the square grid
    point_index: np.ndarray     # (n_out,) int64: flat index into the (n_rows * n_cols) hex rectangle
    x_axis: np.ndarray          # (n_cols,) float64, um
    y_axis: np.ndarray          # (n_rows,) float64, um
    n_points: int               # slots of the hex rectangle (a gathered array must have this many rows)
    n_source: int               # measured hex points (= the geometry's n_valid)
    n_unused: int               # measured hex points no cell takes

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n_rows, self.n_cols)

    @property
    def n_out(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def axes(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.x_axis, self.y_axis)

    def gather(self, arr) -> np.ndarray:
        """``arr`` resampled to the square grid: row ``i`` of the result is the row of ``arr`` that cell
        ``i`` (row-major) takes. The first axis of ``arr`` must be one entry per slot of the hex
        rectangle; anything else raises ``ValueError`` (``edax_hex.remap_point_values`` answers ``None``
        there, which a writer could take for "no data")."""
        arr = np.asarray(arr)
        if arr.ndim == 0 or arr.shape[0] != self.n_points:
            raise ValueError(
                f"an array to resample must have one entry per scan point on its first axis "
                f"({self.n_points} points); got shape {arr.shape}")
        return arr[self.point_index]


def oim_step_geometry(geometry: ScanGeometry) -> ScanGeometry:
    """The geometry with its step as the instrument states it: the shortest decimal that reads back as
    the same float32. ``step_y`` keeps its ratio to ``step_x``.

    OIM's ties between two scan points depend on that decimal step (0.15, say). The same step held as a
    float32 widened to float64 (0.15000000596046448, which is what a float32 header value becomes) flips
    thousands of them: on a reference scan 6 600 of 40 592 cells took another point. A step that is
    already the decimal is returned unchanged."""
    step = float(str(np.float32(geometry.step_x)))
    if step == float(geometry.step_x):
        return geometry
    return replace(geometry, step_x=step, step_y=float(geometry.step_y) * step / float(geometry.step_x))


def float32_positions(geometry: ScanGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """Float32 ``(n_rows, n_cols)`` x and y of the points, as OIM stores them, for the resampling.

    x is the geometry's position in float64 rounded to float32; when the odd rows are shifted to the
    left, half a step is added to every x first, so the smallest row start is 0 (as when they are
    shifted to the right). y is the row index times
    the float32 step, which is how OIM computes it: rounding the float64 product instead differs from the
    stored value in the last bit for many rows. ``ScanGeometry.positions`` itself is shared with the
    raster and the neighbour tools and stays float64. The step is the one ``oim_step_geometry`` gives."""
    geometry = oim_step_geometry(geometry)
    x, _ = geometry.positions()
    if float(geometry.odd_row_shift) < 0:
        x = x + 0.5 * float(geometry.step_x)
    x32 = x.astype(np.float32)
    rows = np.arange(geometry.n_rows, dtype=np.float32)[:, None]
    y32 = np.broadcast_to(rows * np.float32(geometry.step_y), (geometry.n_rows, geometry.n_cols)).copy()
    return x32, y32


def layout_from_geometry(geometry: ScanGeometry) -> RectHexLayout:
    """The row structure of a hexagonal geometry, the way ``edax_hex.measure_hex_layout`` states it for
    a file. Raises ``ValueError`` for a square geometry, for a mask that is not a prefix of every row
    (padding is expected at the row ends only) and for a row without any point."""
    if not geometry.is_hex:
        raise ValueError(f"only a hexagonal scan is resampled; this geometry is {geometry.kind!r}")
    geometry = oim_step_geometry(geometry)
    n_rows, n_cols = geometry.grid_shape
    mask = (np.ones((n_rows, n_cols), dtype=bool) if geometry.valid_mask is None
            else np.asarray(geometry.valid_mask, dtype=bool))
    counts = mask.sum(axis=1).astype(np.int64)
    prefix = np.arange(n_cols)[None, :] < counts[:, None]
    if not np.array_equal(mask, prefix):
        bad = int(np.argmax((mask != prefix).any(axis=1)))
        raise ValueError(
            f"the valid mask of row {bad} is not a prefix of the row: padding is expected at the end "
            f"of a row only")
    if int(counts.min()) == 0:
        raise ValueError(f"row {int(np.argmin(counts))} of the scan has no measured point")

    x32, y32 = float32_positions(geometry)
    flat_valid = mask.reshape(-1)
    pattern_of_point = np.full(flat_valid.size, -1, dtype=np.int64)
    pattern_of_point[flat_valid] = np.arange(int(flat_valid.sum()), dtype=np.int64)

    step_x = float(geometry.step_x)
    # After the shift every row starts at 0 or at half a step; the rows that start at 0 are the odd
    # ones for -0.5 and the even ones for +0.5.
    long_row_parity = 1 if float(geometry.odd_row_shift) < 0 else 0
    return RectHexLayout(
        n_cols=n_cols, n_rows=n_rows, step_x=step_x, step_y=float(geometry.step_y),
        long_row_parity=long_row_parity, x0_long=0.0, x0_short=0.5 * step_x,
        valid=flat_valid, pattern_of_point=pattern_of_point,
        x=x32.astype(np.float64).reshape(-1), y_of_row=y32[:, 0].astype(np.float64),
        row_lengths=counts)


def resample_xmap(xmap, resampling: SquareResampling):
    """The ``CrystalMap`` ``xmap`` of a hexagonal scan on the square grid of ``resampling``.

    ``xmap`` is the map on the index-space rectangle: one point per slot of the hexagonal rectangle
    (``resampling.n_points``). Every cell of the result carries the rotations, the phase id and the
    value of every property of the point it takes; a point that is not indexed stays not indexed.
    Rotations with several matches per point (``rotations.shape == (n_points, k)``) keep all ``k``
    matches. The coordinates are the axes of the resampling and the scan unit is ``"um"``. The phase list is the one of
    ``xmap`` (its phases keep their class and space group; a phase that no cell takes drops out of the
    list, as orix does for any map). ``xmap`` itself is not changed.

    ``ValueError`` when ``xmap`` does not have ``resampling.n_points`` points, or when a property has
    another first axis than one row per point: such an array cannot be resampled."""
    from orix.crystal_map import CrystalMap
    from orix.quaternion import Rotation

    rotations = xmap.rotations
    if rotations.shape[0] != resampling.n_points:
        raise ValueError(
            f"the map has {rotations.shape[0]} points, the scan {resampling.n_points} points: "
            f"it is not the map of this scan")
    prop = {}
    for key in xmap.prop.keys():
        # the dict itself: ``xmap.prop[key]`` masks by ``is_in_data`` and raises a bare IndexError for
        # an array of another length. The rotations have all slots (checked above), so nothing is masked.
        values = np.asarray(dict.__getitem__(xmap.prop, key))
        if values.ndim == 0 or values.shape[0] != resampling.n_points:
            raise ValueError(
                f"the property {key!r} has shape {values.shape}, not one row per scan point "
                f"({resampling.n_points} points): it cannot be resampled")
        prop[key] = resampling.gather(values)

    x_axis, y_axis = resampling.axes
    return CrystalMap(
        rotations=Rotation(resampling.gather(rotations.data)),
        phase_id=resampling.gather(xmap.phase_id),
        x=np.tile(x_axis, resampling.n_rows),
        y=np.repeat(y_axis, resampling.n_cols),
        phase_list=xmap.phases,
        prop=prop,
        scan_unit="um",
    )


def square_resampling(geometry: ScanGeometry, out_step: Optional[float] = None) -> SquareResampling:
    """Map the square grid OIM would resample ``geometry`` to onto its points.

    ``out_step`` defaults to the in-row step. See the module docstring for the shape of the grid."""
    layout = layout_from_geometry(geometry)
    smap = edax_hex.build_square_map(layout, out_step)
    x_axis, y_axis = edax_hex.square_axes(smap.n_cols, smap.n_rows, smap.step)
    n_source = int(layout.valid.sum())
    n_unused = n_source - int(np.unique(smap.point_index).size)
    return SquareResampling(
        step=float(smap.step), n_rows=int(smap.n_rows), n_cols=int(smap.n_cols),
        point_index=smap.point_index, x_axis=x_axis, y_axis=y_axis,
        n_points=int(geometry.n_points), n_source=n_source, n_unused=n_unused)
