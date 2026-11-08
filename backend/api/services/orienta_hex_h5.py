"""The two HDF5 blocks that make an Orienta ``.h5`` hold a hexagonal scan natively.

A hexagonal scan is stored in index order (``n_rows * n_cols`` entries, no padding), and two
places in the file say how those entries sit on the sample:

* the EDAX-style header of a fresh rich export (``Scan1/EBSD/Header``), written by
  :func:`write_header_block`: ``Grid Type`` ``HexGrid``, ``nRows``, ``nColumns``, ``Step X``,
  ``Step Y``, ``Odd Row Shift`` (+0.5 or -0.5 steps) and ``Shift Confirmed``;
* the attributes of the ``/Indexing`` group, written by :func:`write_index_block`: the five
  attributes of :func:`scan_geometry.geometry_to_attrs` plus ``grid_shape``, so a result and a
  checkpoint carry the same stamp.

A square scan gets neither block: both writers return ``False`` and touch nothing.

The readers never guess. A hexagonal stamp that lacks a key, shifts the odd rows by anything but
+-0.5 steps, spaces its rows by anything but ``step_x * sqrt(3) / 2``, or states a shape the stored
data does not have raises ``ValueError``; a hexagonal file is never read back as a square one.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import h5py
import numpy as np

from backend.api.services.scan_geometry import (
    GeometryMismatchError, ScanGeometry, geometry_from_attrs, geometry_to_attrs,
)

logger = logging.getLogger(__name__)

#: Where the EDAX-style header lives in a fresh rich export.
HEADER_PATH = "Scan1/EBSD/Header"
PATTERN_PATH = "Scan1/EBSD/Data/Pattern"
#: The ``Version`` string that marks a file Orienta wrote itself.
RICH_EXPORT_VERSION = "Orienta rich export"

#: Attribute that carries the sentence below when the direction of the row offset is an assumption.
ASSUMED_ATTR = "odd_row_shift_assumed"

# Header keys of a hexagonal scan, in the order they are written.
_KEY_GRID = "Grid Type"
_KEY_ROWS = "nRows"
_KEY_COLS = "nColumns"
_KEY_STEP_X = "Step X"
_KEY_STEP_Y = "Step Y"
_KEY_SHIFT = "Odd Row Shift"
_KEY_CONFIRMED = "Shift Confirmed"

# Attributes that only a stamped block has: their presence without ``grid_type`` is a damaged stamp.
_STAMP_ATTRS = ("odd_row_shift", "shift_confirmed", "step_x", "step_y", ASSUMED_ATTR)


def assumed_sentence(geometry: ScanGeometry) -> str:
    """The sentence stored beside an unconfirmed row offset."""
    side = "right" if float(geometry.odd_row_shift) > 0 else "left"
    return ("The direction of the row offset is an assumption (no file states it): the odd rows "
            f"are placed half a step to the {side} of the even rows.")


def _put(group, key, data) -> None:
    if key in group:
        del group[key]
    group.create_dataset(key, data=data)


# ------------------------------------------------------------------------------- writers

def write_header_block(hdr, geometry: ScanGeometry) -> bool:
    """Write the hexagonal keys into the open header group ``hdr``.

    Returns ``False`` and writes nothing for a square geometry. Every value is a one-element
    array, the way an EDAX header stores it; the steps and the offset are float64 so the stamp
    reads back bit-exact."""
    if not geometry.is_hex:
        return False
    attrs = geometry_to_attrs(geometry)
    _put(hdr, _KEY_GRID, np.bytes_(attrs["grid_type"]))
    _put(hdr, _KEY_ROWS, np.array([geometry.n_rows], np.int32))
    _put(hdr, _KEY_COLS, np.array([geometry.n_cols], np.int32))
    _put(hdr, _KEY_STEP_X, np.array([attrs["step_x"]], np.float64))
    _put(hdr, _KEY_STEP_Y, np.array([attrs["step_y"]], np.float64))
    _put(hdr, _KEY_SHIFT, np.array([attrs["odd_row_shift"]], np.float64))
    _put(hdr, _KEY_CONFIRMED, np.array([1 if attrs["shift_confirmed"] else 0], np.int32))
    if ASSUMED_ATTR in hdr.attrs:
        del hdr.attrs[ASSUMED_ATTR]
    if not geometry.shift_confirmed:
        hdr.attrs[ASSUMED_ATTR] = assumed_sentence(geometry)
    return True


def write_index_block(idx, geometry: ScanGeometry) -> bool:
    """Write the geometry attributes into the open ``/Indexing`` group ``idx``.

    Returns ``False`` and writes nothing for a square geometry. ``step_size_um`` is set to
    ``step_x`` so it cannot disagree with the stamp."""
    if not geometry.is_hex:
        return False
    for key, value in geometry_to_attrs(geometry).items():
        idx.attrs[key] = value
    idx.attrs["grid_shape"] = [int(geometry.n_rows), int(geometry.n_cols)]
    idx.attrs["step_size_um"] = float(geometry.step_x)
    if ASSUMED_ATTR in idx.attrs:
        del idx.attrs[ASSUMED_ATTR]
    if not geometry.shift_confirmed:
        idx.attrs[ASSUMED_ATTR] = assumed_sentence(geometry)
    return True


def signal_patterns_3d(signal):
    """(n_patterns, sy, sx) uint8/uint16 array from an EBSD signal.

    Materialises lazy/memmap data. Returns (patterns, (ny, nx, sy, sx)).
    """
    nav = signal.axes_manager.navigation_shape          # (nx, ny)
    sig = signal.axes_manager.signal_shape              # (sx, sy)
    nx = int(nav[0]); ny = int(nav[1]) if len(nav) > 1 else 1
    sx = int(sig[0]); sy = int(sig[1]) if len(sig) > 1 else int(sig[0])
    data = np.ascontiguousarray(np.asarray(signal.data))
    pats = data.reshape(ny * nx, sy, sx)
    if pats.dtype not in (np.uint8, np.uint16):
        from backend.api.services.load_choices import to_uint8_chunked
        pats = to_uint8_chunked(pats)
    return np.ascontiguousarray(pats), (ny, nx, sy, sx)


def write_fresh_file(path, patterns, scan_geometry: ScanGeometry, pc_bruker=None, tilts=None) -> None:
    """Write a minimal EDAX-format ``.h5`` (``Manufacturer`` EDAX, ``/Scan1/EBSD/...``).

    ``patterns`` is the ``(n_points, sy, sx)`` array in index order, uint8 or uint16, and
    ``scan_geometry`` the grid it sits on; the two must agree on the number of points
    (``GeometryMismatchError`` otherwise). ``pc_bruker`` is the pattern centre to state (Bruker
    convention); it goes into the header in the EDAX TSL convention, which is what kikuchipy's EDAX
    reader takes it as. ``None`` writes no pattern centre.

    ``tilts`` is a dict of already resolved angles in degrees: ``sample_tilt`` (default 70.0),
    ``tilt`` (the camera elevation) and ``azimuthal``; a missing or ``None`` camera angle is not
    written (the reader then sets it to 0 and says nothing, so the caller logs why)."""
    tilts = tilts or {}
    pats = np.asarray(patterns)
    if pats.ndim != 3:
        raise ValueError(f"patterns must be (n_points, sy, sx), got shape {pats.shape}")
    if pats.shape[0] != scan_geometry.n_points:
        raise GeometryMismatchError(
            f"{pats.shape[0]} patterns do not fit the {scan_geometry.n_rows} x "
            f"{scan_geometry.n_cols} scan ({scan_geometry.n_points} points)")
    _, sy, sx = pats.shape
    sample_tilt = tilts.get("sample_tilt")
    sample_tilt = 70.0 if sample_tilt is None else float(sample_tilt)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("Manufacturer", data=np.bytes_("EDAX"))
        f.create_dataset("Version", data=np.bytes_(RICH_EXPORT_VERSION))
        hdr = f.create_group(HEADER_PATH)
        if not scan_geometry.is_hex:
            hdr.create_dataset("Grid Type", data=np.bytes_("SqrGrid"))
        hdr.create_dataset("Pattern Height", data=np.array([sy], np.int32))
        hdr.create_dataset("Pattern Width", data=np.array([sx], np.int32))
        hdr.create_dataset("Pattern Bit Depth", data=np.array([8], np.int32))
        hdr.create_dataset("Sample Tilt", data=np.array([sample_tilt], np.float32))
        if scan_geometry.is_hex:
            # Grid Type, nRows, nColumns, Step X/Y (float64), Odd Row Shift, Shift Confirmed.
            write_header_block(hdr, scan_geometry)
        else:
            hdr.create_dataset("Step X", data=np.array([scan_geometry.step_x], np.float32))
            hdr.create_dataset("Step Y", data=np.array([scan_geometry.step_y], np.float32))
            hdr.create_dataset("nColumns", data=np.array([scan_geometry.n_cols], np.int32))
            hdr.create_dataset("nRows", data=np.array([scan_geometry.n_rows], np.int32))
        # The geometry the reader needs to rebuild the detector: camera tilt (as the
        # header names it), and the PC in the TSL convention of that header.
        if tilts.get("tilt") is not None:
            hdr.create_dataset("Camera Elevation Angle", data=np.array([float(tilts["tilt"])], np.float64))
        if tilts.get("azimuthal") is not None:
            hdr.create_dataset("Camera Azimuthal Angle", data=np.array([float(tilts["azimuthal"])], np.float64))
        if pc_bruker is not None:
            from backend.api.services import ang_export
            tsl = ang_export.pc_tsl_from_bruker(pc_bruker, (sy, sx))
            if tsl is not None:
                grp = hdr.create_group("Pattern Center Calibration")
                for key, value in zip(("x-star", "y-star", "z-star"), tsl):
                    grp.create_dataset(key, data=np.array([value], np.float64))
        d = f.create_dataset(PATTERN_PATH, data=pats, compression="gzip", chunks=(1, sy, sx))
        d.attrs["patterns_processed"] = True


# ------------------------------------------------------------------------------- readers

def _scalar(group, key):
    """The first value of the header entry ``key``; ``ValueError`` when it is missing or empty."""
    if key not in group:
        raise ValueError(f"the hexagonal header has no {key!r}")
    flat = np.ravel(group[key][()])
    if flat.size == 0:
        raise ValueError(f"the hexagonal header entry {key!r} is empty")
    value = flat[0]
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def read_header_geometry(path) -> Optional[ScanGeometry]:
    """The geometry stamped in the header of ``path``.

    Returns a :class:`ScanGeometry` for a ``HexGrid`` header with every Orienta key, and ``None``
    for a ``SqrGrid`` header. Raises ``ValueError`` for a file without ``Scan1/EBSD/Header``; for
    any other ``Grid Type``; for a ``HexGrid`` header that lacks a key (so an OIM-written hexagonal
    header, which has no ``Odd Row Shift`` or ``Shift Confirmed``, raises: it states no direction);
    for a shift other than +-0.5, a bad row ratio, or ``nRows * nColumns`` different from the number
    of stored patterns.

    Callers dispatch with :func:`is_native_hex_file` first: it is True exactly for the files this
    reader is meant for, and a True file that this reader refuses is damaged."""
    with h5py.File(str(path), "r") as f:
        if HEADER_PATH not in f:
            raise ValueError(f"{path} has no {HEADER_PATH}")
        hdr = f[HEADER_PATH]
        grid = str(_scalar(hdr, _KEY_GRID))
        if grid == "SqrGrid":
            return None
        if grid != "HexGrid":
            raise ValueError(f"unknown Grid Type {grid!r} in the header, expected HexGrid or SqrGrid")
        attrs = {"grid_type": grid}
        for key, name in ((_KEY_SHIFT, "odd_row_shift"), (_KEY_CONFIRMED, "shift_confirmed"),
                          (_KEY_STEP_X, "step_x"), (_KEY_STEP_Y, "step_y")):
            if key in hdr:
                attrs[name] = _scalar(hdr, key)
        shape = (int(_scalar(hdr, _KEY_ROWS)), int(_scalar(hdr, _KEY_COLS)))
        if PATTERN_PATH in f:
            stored = int(f[PATTERN_PATH].shape[0])
            if stored != shape[0] * shape[1]:
                raise ValueError(
                    f"the header states {shape[0]} x {shape[1]} = {shape[0] * shape[1]} points, "
                    f"the file stores {stored} patterns")
        return geometry_from_attrs(attrs, shape)


#: Datasets directly under ``/Indexing`` whose shape is the scan grid, in the order they are tried:
#: the light export's ``X``, and the single-phase rich export's ``selection_mask`` and
#: ``confidence_index``.
_GRID_MAPS = ("X", "selection_mask", "confidence_index")


def _stored_data_shape(idx) -> Optional[Tuple[int, int]]:
    """The (rows, cols) of a map the ``/Indexing`` group stores, independent of its attributes.

    The first 2-D dataset of ``X``, ``selection_mask`` and ``confidence_index`` under the group,
    else the 2-D ``confidence_index`` of the first phase under ``PerPhase``. A 1-D array (a
    per-phase confidence index stored flat) is never a source, as it carries no row length.
    ``None`` if the group stores no such map."""
    for name in _GRID_MAPS:
        ds = idx.get(name)
        if isinstance(ds, h5py.Dataset) and ds.ndim == 2:
            return (int(ds.shape[0]), int(ds.shape[1]))
    per_phase = idx.get("PerPhase")
    if per_phase is not None:
        for name in per_phase:
            group = per_phase[name]
            ci = group.get("confidence_index") if hasattr(group, "get") else None
            if isinstance(ci, h5py.Dataset) and ci.ndim == 2:
                return (int(ci.shape[0]), int(ci.shape[1]))
    return None


def read_index_geometry(path) -> Optional[ScanGeometry]:
    """The geometry stamped on ``/Indexing`` of ``path``.

    Returns a :class:`ScanGeometry` for a stamped block, and ``None`` for a block without a stamp
    (``/Indexing`` has none of ``grid_type``, ``odd_row_shift``, ``shift_confirmed``, ``step_x``,
    ``step_y``, ``odd_row_shift_assumed``): a square or legacy file, whatever other attributes it
    carries. Raises ``ValueError`` for a file without ``/Indexing``; for a stamp attribute without
    ``grid_type``; for a ``grid_type`` without ``grid_shape``; and for everything
    :func:`scan_geometry.geometry_from_attrs` refuses (unknown ``grid_type``, a missing key, a shift
    other than +-0.5, a bad row ratio, a ``grid_shape`` that is not the shape of the stored data).

    The shape handed to ``geometry_from_attrs`` is the shape of the data the group stores (see
    :func:`_stored_data_shape`), not the ``grid_shape`` attribute it is compared with. A group that
    stores no 2-D map has only the attribute to go by, and then nothing is compared.

    Callers dispatch with :func:`is_native_hex_file` first; this reader serves every writer's
    ``/Indexing``, whether the file is native or not."""
    with h5py.File(str(path), "r") as f:
        if "Indexing" not in f:
            raise ValueError(f"{path} has no /Indexing group")
        idx = f["Indexing"]
        attrs = idx.attrs
        if "grid_type" not in attrs:
            leftover = [name for name in _STAMP_ATTRS if name in attrs]
            if leftover:
                raise ValueError(f"/Indexing has {', '.join(leftover)} but no grid_type")
            return None
        if "grid_shape" not in attrs:
            raise ValueError(f"{path} has a grid_type but no grid_shape")
        shape = _stored_data_shape(idx)
        if shape is None:
            shape = tuple(int(v) for v in np.asarray(attrs["grid_shape"]).ravel())
        return geometry_from_attrs(attrs, shape)


def is_native_hex_file(path) -> bool:
    """Whether ``path`` is an HDF5 file Orienta wrote as a hexagonal scan: ``Version`` is
    ``Orienta rich export``, ``Manufacturer`` is ``EDAX`` and the header's ``Grid Type`` is
    ``HexGrid``.

    It keys on the grid type, not on ``Odd Row Shift``: a file Orienta wrote whose stamp lost a key
    is still native, and :func:`read_header_geometry` then raises for it instead of the file being
    read as an OIM scan in the padded layout. A padded OIM copy has OIM's ``Version`` and is
    False, and so is a square header whatever else it holds.

    Never raises: anything unreadable is not a native hexagonal file."""
    try:
        if not h5py.is_hdf5(str(path)):
            return False
        with h5py.File(str(path), "r") as f:
            if "Version" not in f or "Manufacturer" not in f or HEADER_PATH not in f:
                return False
            if str(_scalar(f, "Version")) != RICH_EXPORT_VERSION:
                return False
            if str(_scalar(f, "Manufacturer")) != "EDAX":
                return False
            return str(_scalar(f[HEADER_PATH], _KEY_GRID)) == "HexGrid"
    except Exception:
        return False


# ------------------------------------------------------------------------------- loader

def _gather_lazy(dset, n_rows: int, n_cols: int):
    """The ``(n_points, sy, sx)`` dataset as one dask array per scan row, stacked to
    ``(n_rows, n_cols, sy, sx)``. Pattern ``[r, c]`` is stored row ``r * n_cols + c``: the file is in
    index order, so no index map is needed."""
    import dask.array as da

    sy, sx = dset.shape[1], dset.shape[2]
    source = da.from_array(dset, chunks=("auto", sy, sx))
    return da.stack([source[r * n_cols:(r + 1) * n_cols] for r in range(n_rows)], axis=0)


def load_native_hex(path, lazy: bool = False, verbose: bool = True):
    """Load a native hexagonal Orienta ``.h5`` (see :func:`is_native_hex_file`) as the index
    rectangle of its scan.

    The signal has ``n_rows x n_cols`` navigation points in the file's own order, nothing is
    resampled, and the geometry that says where each point lies is attached as
    ``Signal.scan_geometry`` for the route to register (as an APEX-described UP scan does). The
    detector is built the way :func:`edax_hex.load_edax_hex` builds it, from the EDAX header: pattern
    centre in the TSL convention (``convention="edax"``), camera elevation and azimuth, sample tilt.
    A header without a pattern centre gets kikuchipy's placeholder; the caller settles that
    (``safe_loader._restore_pc_from_orienta_export``).

    Raises ``ValueError`` when the header's stamp is damaged (:func:`read_header_geometry`), when it
    describes a square scan, or when the file stores no patterns of the stated count; a hexagonal
    file is never read as a square one."""
    import logging
    from pathlib import Path

    from kikuchipy.detectors import EBSDDetector
    from kikuchipy.io._io import _dict2signal
    from kikuchipy.io.plugins._h5ebsd import _hdf5group2dict
    from orix.crystal_map import CrystalMap

    path = str(path)
    file_name = Path(path).name
    geometry = read_header_geometry(path)
    if geometry is None:
        raise ValueError(f"{file_name} states a square grid, not a hexagonal one")
    n_rows, n_cols = geometry.n_rows, geometry.n_cols

    f = h5py.File(path, "r")
    # A lazy dask array keeps reading from this dataset: when a lazy signal is returned the file
    # stays open and the signal owns it. In every other case, a refusal included, it is closed here
    # (an open HDF5 file cannot be renamed or opened for writing on Windows).
    signal_owns_file = False
    try:
        if PATTERN_PATH not in f:
            raise ValueError(f"{file_name} has no {PATTERN_PATH}")
        dset = f[PATTERN_PATH]
        if dset.ndim != 3 or int(dset.shape[0]) != n_rows * n_cols:
            raise ValueError(
                f"{file_name} stores patterns of shape {tuple(dset.shape)}, expected "
                f"({n_rows * n_cols}, rows, cols) for {n_rows} x {n_cols} points")
        sy, sx = int(dset.shape[1]), int(dset.shape[2])
        header = _hdf5group2dict(f[HEADER_PATH], recursive=True)
        version = str(_scalar(f, "Version"))
        if lazy:
            data = _gather_lazy(dset, n_rows, n_cols)
        else:
            data = np.asarray(dset[()]).reshape(n_rows, n_cols, sy, sx)

        if verbose:
            logging.getLogger(__name__).info(
                "Loading hexagonal Orienta file %s (%s): %d x %d points, %d x %d patterns",
                file_name, "lazy" if lazy else "eager", n_rows, n_cols, sy, sx)

        pc_cal = header.get("Pattern Center Calibration", {}) or {}
        axes = [
            {"size": n_rows, "index_in_array": 0, "name": "y", "scale": geometry.step_y,
             "offset": 0.0, "units": "um"},
            {"size": n_cols, "index_in_array": 1, "name": "x", "scale": geometry.step_x,
             "offset": 0.0, "units": "um"},
            {"size": sy, "index_in_array": 2, "name": "dy", "scale": 1.0, "offset": 0.0, "units": "um"},
            {"size": sx, "index_in_array": 3, "name": "dx", "scale": 1.0, "offset": 0.0, "units": "um"},
        ]
        scan = {
            "data": data,
            "axes": axes,
            "metadata": {
                "General": {"original_filename": file_name, "title": Path(path).stem},
                "Signal": {"signal_type": "EBSD", "record_by": "image"},
            },
            "original_metadata": {"manufacturer": "EDAX", "version": version, **header},
            "xmap": CrystalMap.empty(shape=(n_rows, n_cols),
                                     step_sizes=(geometry.step_y, geometry.step_x)),
            "detector": EBSDDetector(
                shape=(sy, sx),
                px_size=1.0,
                tilt=header.get("Camera Elevation Angle", 0.0),
                azimuthal=header.get("Camera Azimuthal Angle", 0.0),
                sample_tilt=header.get("Sample Tilt", 70.0),
                pc=(pc_cal.get("x-star", 0.5), pc_cal.get("y-star", 0.5), pc_cal.get("z-star", 0.5)),
                convention="edax",
            ),
        }
        signal = _dict2signal(scan, lazy=lazy)
        signal.metadata.set_item("Signal.scan_geometry", geometry.to_dict())
        signal_owns_file = lazy
        return signal
    finally:
        if not signal_owns_file:
            f.close()
