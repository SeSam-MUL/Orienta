"""Frame averaging (averaging each pattern with its neighbours) on any scan topology.

kikuchipy's ``average_neighbour_patterns`` correlates ONE window with the grid of patterns. On a hexagonal scan
the window of a point is not the same for every row (which two points of the adjacent rows are its neighbours
depends on the parity of the row), so the same window cannot be laid over the index grid. This is the same
operation written band by band on the neighbour offsets of a ``Topology`` (``scan_neighbours``): the sum of the
patterns of the point and its neighbours within ``rings`` steps, divided by how many of them lie inside the scan
(the border is clipped, as kikuchipy's zero-padded border divided by the window sum is), and rescaled
individually to the full range of the data type, which is what kikuchipy does after averaging.

Memory. The result is a second array of the size of ``data``; apart from that the function holds a few arrays
of ``BAND_BYTES`` (float32) and nothing that grows with the number of points or with ``rings``: the neighbour
indices of a band are computed from the offsets of the two row parities, not stored for the whole scan (a table
of every point's neighbours is ``8 * n_points * (3 * rings * (rings + 1) + 1)`` bytes, 1.6 GB for a million
points and 7 rings).
"""
from __future__ import annotations

import numpy as np

from backend.api.services.scan_neighbours import Topology

#: Size of one band in float32: the temporary arrays of a step are a few times this.
BAND_BYTES = 32 * 1024 * 1024
#: Upper bound of how many such arrays one step holds at once, on top of the result array. Measured as growth of
#: the process's memory on a full-size band: 4.5 (uint8: 144 MiB), 5 (uint16: 160 MiB) and 6 (float32: 192 MiB)
#: bands, so 6.
BAND_ARRAYS = 6

#: Frame averaging of a hexagonal scan holds the patterns twice (the signal and the averaged copy, written back at
#: the end), so the patterns may be 2 GiB, half of the 4 GiB the viewer allows for materialising a dataset: the
#: peak is then that ceiling plus a few band arrays (``working_bytes``) and does not grow with the window.
HEX_MAX_BYTES = 2 * 1024 ** 3
#: Largest odd window on a hexagonal scan (24 rings, 1801 points per neighbourhood). The work grows with the square
#: of the window, and the dialog stops at 50.
HEX_MAX_WINDOW = 49


def dtype_range(dtype) -> tuple[float, float]:
    """The range kikuchipy rescales to: ``[-1, 1]`` for floating point, the full integer range otherwise."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return -1.0, 1.0
    info = np.iinfo(dtype)
    return float(info.min), float(info.max)


def working_bytes() -> int:
    """The bytes the temporary arrays of one step can hold at most (on top of the result array)."""
    return BAND_ARRAYS * BAND_BYTES


def average_neighbour_patterns(data: np.ndarray, topology: Topology, rings: int,
                               valid: np.ndarray | None = None) -> np.ndarray:
    """``data`` ``(n_rows, n_cols, h, w)`` averaged with the points within ``rings`` steps; a new array, same dtype.

    A pattern that is constant after averaging has no range to rescale and is set to the lowest value of the data
    type (kikuchipy divides by zero there).

    ``valid`` is the ``(n_rows, n_cols)`` mask of measured points (``ScanGeometry.valid_mask``): a point that was
    not measured has no pattern to average, so it counts neither as a neighbour nor in the divisor, and its own
    pattern is copied unchanged (there is nothing to average it with and nothing to rescale)."""
    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError(f"patterns must be (n_rows, n_cols, h, w), got shape {data.shape}")
    n_rows, n_cols, h, w = data.shape
    if (n_rows, n_cols) != (topology.n_rows, topology.n_cols):
        raise ValueError(f"the topology is for a {topology.n_rows} x {topology.n_cols} grid, "
                         f"but the patterns are {n_rows} x {n_cols}")
    if rings < 1:
        raise ValueError("rings must be at least 1")
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (n_rows, n_cols):
            raise ValueError(f"the mask of measured points is {valid.shape}, the grid is {n_rows} x {n_cols}")
    # (drow, dcol) of the centre and its neighbours, one row of offsets per row parity.
    disc = [np.asarray(topology.disc_offsets(parity, rings), dtype=np.int64) for parity in (0, 1)]
    omin, omax = dtype_range(data.dtype)
    out = np.empty(data.shape, dtype=data.dtype)                     # C order whatever the order of ``data``
    out_flat = out.reshape(n_rows * n_cols, h, w)                    # a view, because ``out`` is a C array
    band = max(1, int(BAND_BYTES // (h * w * 4)))
    for lo in range(0, n_rows * n_cols, band):
        hi = min(lo + band, n_rows * n_cols)
        flat = np.arange(lo, hi, dtype=np.int64)
        row, col = flat // n_cols, flat % n_cols
        odd = (row % 2).astype(bool)
        total = np.zeros((hi - lo, h, w), dtype=np.float32)
        count = np.zeros(hi - lo, dtype=np.int32)
        for k in range(disc[0].shape[0]):
            rr = row + np.where(odd, disc[1][k, 0], disc[0][k, 0])
            cc = col + np.where(odd, disc[1][k, 1], disc[0][k, 1])
            ok = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
            rr, cc = np.clip(rr, 0, n_rows - 1), np.clip(cc, 0, n_cols - 1)
            if valid is not None:
                ok &= valid[rr, cc]
            gathered = data[rr, cc].astype(np.float32)               # (band, h, w)
            if not ok.all():
                gathered[~ok] = 0.0
            total += gathered
            count += ok
        mean = total
        mean /= np.maximum(count, 1).astype(np.float32)[:, None, None]
        imin = mean.min(axis=(1, 2), keepdims=True)
        imax = mean.max(axis=(1, 2), keepdims=True)
        has_range = imax > imin
        mean -= imin
        mean /= np.where(has_range, imax - imin, 1.0).astype(np.float32)
        mean *= np.float32(omax - omin)
        mean += np.float32(omin)
        mean[~has_range[:, 0, 0]] = omin
        result = mean.astype(data.dtype)
        if valid is not None:
            unmeasured = ~valid[row, col]
            if unmeasured.any():
                result[unmeasured] = data[row[unmeasured], col[unmeasured]]
        out_flat[lo:hi] = result
    return out
