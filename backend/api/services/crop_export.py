"""Write a crop window out as a standalone h5oina-like file.

Everything else about a crop lives in memory and dies with the backend
process. This module is what makes a cut-out shareable: it walks the source
file and writes a second one that a fresh app session opens like any other
measurement.

h5oina is regular enough for one generic rule: under the scan's own data
groups, anything whose first axis is as long as the scan has points is
per-pixel data and gets cut; everything else is copied verbatim. That covers
the patterns, every EBSD channel, every EDS element and the per-pixel spectra
without naming any of them, and a dataset the rule cannot recognise is copied
rather than mangled. Copying is the safe direction: a full-size dataset in a
cropped file is visibly odd, a wrongly-cut one is not.

The rule has one unsafe direction, stated here because it cannot be tested
away: a dataset under those groups that is NOT per-pixel but whose first axis
happens to equal the point count -- a lookup table with one row per scan point
by coincidence -- would be cut. Both files in ``Test_data/`` were enumerated
and every dataset under ``/EBSD/Data/`` and ``/EDS/Data/`` is genuinely
per-pixel -- 30 of them on SampleB, 28 on HIgh_MG -- so nothing misclassifies
today; a format that puts something else there would need the rule tightened,
not this comment widened.

Three things the generic rule alone would get wrong, each handled explicitly:

* **Reads are per-row, never whole-dataset.** The window's flat indices form
  one contiguous run per row, so a cut is ``rows`` slice reads of ``cols``
  entries. Reading the source dataset and then fancy-indexing it would pull
  9.7 GB for ``Processed Patterns`` alone on the 27 GB file this feature
  exists to serve.

* **The electron image is not on the scan grid.** Measured on the two files
  in ``Test_data/``, it is 1024x768 at 0.059 um against a 120x90 scan at
  0.5 um. Row and column numbers mean nothing across areas, so the window is
  carried over in micrometres by ``crop_window.project_to_area``. It is also
  stored FLAT in a real file -- ``(786432,)``, not ``(768, 1024)`` -- so it
  is reshaped through its own header, cut, and written back flat.

* **Only the areas that were cut get their size headers rewritten.** The
  electron image carries its own ``X Cells`` / ``Y Cells`` (1024 x 768) and a
  blanket "rewrite every X Cells" rule would overwrite them with the scan
  crop's, describing a 1024-pixel-wide image as four pixels wide.

Limitation, deliberate: SQUARE grids only. A hex scan's file holds the padded
hex rectangle while the crop window is defined on the resampled square display
grid, so a subset of the file is not well defined. We refuse instead of
writing something wrong.

Deliberate, not an oversight: the written file mixes two coordinate frames.
``Relative Offset`` / ``Relative Size`` and ``Bounding Box Size`` describe the
cut-out relative to itself (measured on a real export: offset ``[0, 0]``, box
``[20, 15]`` um), while the per-pixel ``EBSD/Data/X`` and ``Y`` keep their
ABSOLUTE stage coordinates (10.0 -> 13.5 um for the same crop). Neither is
rebased, for three reasons: absolute stage coordinates are the honest record
of where the beam was, and inventing new ones throws that away; every area
keeps agreeing with every other, which is the property ``project_to_area``
checks when the file is opened again, and rewriting one area's offset without
the others would arm that guard against the file's own electron image; and
cross-area co-registration survives because both areas were cut to the same
physical region. ``CropProvenance`` records where the crop sat, exactly, in
scan pixels -- so nothing is lost, it just is not in the site frame.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

from backend.api.services.crop_window import CropWindow, project_to_area

logger = logging.getLogger(__name__)

FORMAT_VERSION = "1.0"


def _looks_like_hex(source_path) -> bool:
    """Whether the FILE says it is a hexagonally sampled EDAX scan.

    Imported lazily, like every other consumer of ``edax_hex`` in this code
    base (``pattern_quality``, ``h5_viewer_backend``, ``safe_loader``). It was
    a module-level import until a port to the public repository — which ships
    without ``edax_hex.py`` — turned "this build cannot auto-detect hex" into
    "importing this module raises", i.e. the crop export vanished with a
    ModuleNotFoundError at the moment the user pressed Save.

    Returns False when the detector is unavailable, and says so once in the
    log. That is not a silent weakening of the guard: a build without
    ``edax_hex`` also has no hex support anywhere, so it never resamples a hex
    scan onto a square display grid — and the square-vs-padded mismatch this
    guard exists to catch cannot arise. The ``is_hex`` parameter stays as the
    override for a caller that knows better.
    """
    try:
        from edax_hex import is_edax_hex_file
    except ImportError:
        logger.warning(
            "edax_hex is unavailable, so a hex scan cannot be detected from the "
            "file; relying on the caller's is_hex flag"
        )
        return False
    return bool(is_edax_hex_file(source_path))

#: Data groups whose per-pixel datasets get cut. Everything else is copied.
#: Slash-delimited on both sides and matched against ``"/" + path + "/"``, so
#: a group called ``EBSD-Schichtbild`` cannot pass for ``EBSD``.
_PER_PIXEL_GROUPS = ("/EBSD/Data/", "/EDS/Data/")

#: Areas measured on the scan grid, i.e. the ones a crop shrinks directly.
_SCAN_HEADERS = ("/EBSD/Header/", "/EDS/Header/")

_ELECTRON_DATA = "/Electron Image/Data/"
_ELECTRON_HEADER = "/Electron Image/Header/"

#: Header fields that describe the grid size and must be rewritten.
_COL_FIELDS = ("X Cells", "XCells", "nColumns")
_ROW_FIELDS = ("Y Cells", "YCells", "nRows")
#: The same extent in micrometres: [width, height].
_BBOX_FIELD = "Bounding Box Size"

#: The dataset that holds "the patterns as the app currently has them".
#: ``Unprocessed Patterns`` is deliberately NOT here -- it is the raw
#: material, honestly labelled, and is cut from the source like any other
#: channel even when processed patterns are supplied.
_PATTERN_LEAVES = ("Processed Patterns", "Pattern")

#: Below this many elements a dataset is written uncompressed: gzip on a
#: handful of numbers costs more in chunk overhead than it saves.
_COMPRESS_ABOVE = 1024


def write_cropped_h5oina(
    source_path: str,
    out_path: str,
    window: CropWindow,
    patterns: Optional[np.ndarray] = None,
    is_hex: bool = False,
) -> dict:
    """Write ``window`` of ``source_path`` to ``out_path`` as its own file.

    Parameters
    ----------
    patterns
        The patterns to write in place of the source's, already cut to the
        window -- either ``(rows * cols, H, W)`` or ``(rows, cols, H, W)``.
        Supply these whenever the dataset has been processed (background
        removal, frame averaging): what gets saved must be what the work was
        done on, not the raw material it started from.
    is_hex
        Force the hex refusal. The file is checked anyway
        (``edax_hex.is_edax_hex_file``, the same ``Grid Type == "HexGrid"``
        header ``H5OINADataExtractor.hex_map`` reads), so this only exists for
        a caller that knows something the file does not say. Leaving it False
        is safe.

    Returns
    -------
    dict
        ``{"path", "n_points", "datasets_cut", "datasets_copied",
        "electron_images_cut", "electron_images_full"}``.
    """
    # Asked of the FILE, not only of the caller. A default of False on a
    # parameter is a wrong answer waiting for someone to forget it, and this
    # is the one failure mode of this module a caller cannot see: a hex scan
    # written as if it were square opens cleanly and is silently sheared.
    if is_hex or _looks_like_hex(source_path):
        raise ValueError(
            "Cropped file export supports square scan grids only. This is a hex "
            "scan: the file stores the padded hex rectangle while the crop window "
            "is defined on the resampled square grid, so a subset of the file is "
            "not well defined. The in-memory crop works normally."
        )

    src_path = Path(source_path)
    dst_path = Path(out_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    n_points_full = int(window.original_shape[0]) * int(window.original_shape[1])
    n_cols_full = int(window.original_shape[1])
    n_points = int(window.rows * window.cols)

    report = {
        "path": str(dst_path), "n_points": n_points,
        "datasets_cut": 0, "datasets_copied": 0,
        "electron_images_cut": 0, "electron_images_full": 0,
    }

    _check_window_fits(src_path, window)

    try:
        _write(src_path, dst_path, window, patterns, report,
               n_points_full, n_cols_full, n_points)
    except BaseException:
        # A half-written .h5oina on disk is worse than none: it opens far
        # enough to look like a measurement. Take it away with the failure.
        try:
            dst_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove the partial file %s", dst_path)
        raise

    logger.info(
        "Wrote cropped file %s: %d points, %d datasets cut, %d copied, "
        "%d electron images cut (%d full)",
        dst_path, report["n_points"], report["datasets_cut"],
        report["datasets_copied"], report["electron_images_cut"],
        report["electron_images_full"],
    )
    return report


def _write(src_path, dst_path, window, patterns, report,
           n_points_full, n_cols_full, n_points) -> None:
    """The walk itself. Split out so the caller owns the partial-file cleanup."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        elec_grid = _electron_grid(src)
        elec_rect = None
        if elec_grid is not None:
            elec_rect = project_to_area(
                window, _area_geom(src, ("EBSD", "EDS")),
                _area_geom(src, ("Electron Image",)), elec_grid,
            )
            if elec_rect is None:
                logger.warning(
                    "electron images written in full: the scan and the electron "
                    "image cannot place a window between them (missing step "
                    "size, or the two areas do not share an origin)"
                )

        supplied = _checked_patterns(patterns, window) if patterns is not None else None
        supplied_written = []

        def _visit(node: h5py.Group, parent_path: str) -> None:
            for name in node:
                item = node[name]
                path = f"{parent_path}/{name}" if parent_path else name
                if isinstance(item, h5py.Group):
                    grp = dst.require_group(path)
                    for key, value in item.attrs.items():
                        grp.attrs[key] = value
                    _visit(item, path)
                    continue
                _write_dataset(path, item)

        def _write_dataset(path: str, item: h5py.Dataset) -> None:
            marked = f"/{path}/"
            leaf = path.rsplit("/", 1)[-1]

            replacement = _header_replacement(marked, leaf, item)
            if replacement is not None:
                _create(path, replacement, item)
                return

            if _ELECTRON_DATA in marked:
                _write_electron_image(path, item)
                return

            if any(g in marked for g in _PER_PIXEL_GROUPS) \
                    and item.ndim >= 1 and item.shape[0] == n_points_full:
                if supplied is not None and leaf in _PATTERN_LEAVES:
                    if supplied.shape[1:] != item.shape[1:]:
                        raise ValueError(
                            f"supplied patterns are {supplied.shape[1:]} per point "
                            f"but {path} holds {item.shape[1:]}"
                        )
                    _create(path, supplied, item)
                    supplied_written.append(path)
                else:
                    _cut_per_pixel(path, item)
                report["datasets_cut"] += 1
                return

            # Anything the rule does not recognise is copied verbatim, with its
            # dtype, attributes and filters -- h5py's own copy, so a
            # variable-length string or an enum survives intact.
            parent = _parent_of(path)
            src.copy(item, dst.require_group(parent) if parent else dst,
                     name=leaf)
            report["datasets_copied"] += 1

        def _header_replacement(marked: str, leaf: str, item: h5py.Dataset):
            """The new value for a size header, or ``None`` to leave it be."""
            if leaf not in _COL_FIELDS and leaf not in _ROW_FIELDS \
                    and leaf != _BBOX_FIELD:
                return None

            group = item.parent
            if any(h in marked for h in _SCAN_HEADERS):
                # Only if this area really is the scan grid we are cutting.
                if _declared_cells(group) != tuple(window.original_shape):
                    return None
                new_rows, new_cols = window.rows, window.cols
            elif _ELECTRON_HEADER in marked:
                if elec_rect is None or _declared_cells(group) != elec_grid:
                    return None
                new_rows, new_cols = elec_rect["rows"], elec_rect["cols"]
            else:
                return None

            if leaf in _COL_FIELDS:
                return _like(item, new_cols)
            if leaf in _ROW_FIELDS:
                return _like(item, new_rows)

            steps = _steps_of(group)
            if steps is None:
                return None
            return np.asarray([new_cols * steps[0], new_rows * steps[1]],
                              dtype=np.asarray(item[()]).dtype)

        def _cut_per_pixel(path: str, item: h5py.Dataset) -> None:
            """One contiguous slice read per window row -- see the docstring."""
            out = _create(path, None, item,
                          shape=(n_points,) + item.shape[1:])
            for r in range(window.rows):
                start = (window.row0 + r) * n_cols_full + window.col0
                out[r * window.cols:(r + 1) * window.cols] = \
                    item[start:start + window.cols]

        def _write_electron_image(path: str, item: h5py.Dataset) -> None:
            cut = _cut_electron(item, elec_grid, elec_rect)
            if cut is None:
                # Not placeable: hand the whole image over without reading it.
                parent = _parent_of(path)
                src.copy(item, dst.require_group(parent) if parent else dst,
                         name=path.rsplit("/", 1)[-1])
                report["electron_images_full"] += 1
                return
            report["electron_images_cut"] += 1
            _create(path, cut, item)

        def _create(path: str, data, like: h5py.Dataset, shape=None):
            """Create a dataset, carrying the source's attributes across."""
            if data is not None:
                data = np.asarray(data)
                shape = data.shape
            n_elements = int(np.prod(shape)) if len(shape) else 1
            kwargs = {}
            if n_elements > _COMPRESS_ABOVE:
                kwargs["compression"] = "gzip"
            if data is not None:
                ds = dst.create_dataset(path, data=data, **kwargs)
            else:
                ds = dst.create_dataset(path, shape=shape, dtype=like.dtype,
                                        **kwargs)
            for key, value in like.attrs.items():
                ds.attrs[key] = value
            return ds

        _visit(src, "")

        if supplied is not None and not supplied_written:
            # The provenance would claim "processed" over patterns that were
            # never written. Refuse rather than mislabel the file.
            raise ValueError(
                "processed patterns were supplied but this file has no pattern "
                f"dataset to put them in (looked for {list(_PATTERN_LEAVES)} "
                "under the scan's Data group)"
            )

        prov = dst.create_group("CropProvenance")
        prov.attrs["format_version"] = FORMAT_VERSION
        prov.attrs["source_file"] = str(src_path)
        prov.attrs["row0"] = int(window.row0)
        prov.attrs["col0"] = int(window.col0)
        prov.attrs["rows"] = int(window.rows)
        prov.attrs["cols"] = int(window.cols)
        prov.attrs["original_shape"] = [int(v) for v in window.original_shape]
        prov.attrs["shape_kind"] = str(window.shape_kind)
        prov.attrs["n_selected"] = int(window.n_selected)
        prov.attrs["patterns"] = "processed" if patterns is not None else "as-in-source"
        if window.nav_mask is not None:
            prov.create_dataset("nav_mask",
                                data=np.asarray(window.nav_mask, dtype=bool))


# --- helpers ------------------------------------------------------------------


def _check_window_fits(src_path: Path, window: CropWindow) -> None:
    """Refuse a window that was cut from a different grid than this file has.

    Every rule below keys on the scan's point count, so a mismatched window
    would cut nothing, rewrite nothing, and hand back a verbatim copy of the
    source under the name of a crop -- the silent kind of wrong. The file may
    legitimately state no cell counts at all, and then there is nothing to
    check against.
    """
    with h5py.File(src_path, "r") as src:
        for area in ("EBSD", "EDS"):
            group = _header_group(src, area)
            if group is None:
                continue
            cells = _declared_cells(group)
            if cells is None:
                continue
            if cells != tuple(int(v) for v in window.original_shape):
                raise ValueError(
                    f"the crop window was cut from a "
                    f"{tuple(window.original_shape)} grid but {src_path.name} "
                    f"states {cells} in its {area} header"
                )
            return


def _parent_of(path: str) -> str:
    parent = path.rsplit("/", 1)[0]
    return "" if parent == path else parent


def _like(dataset: h5py.Dataset, value) -> np.ndarray:
    """A replacement array shaped and typed like the header field it replaces.

    Oxford writes its scalars as ``(1,)`` arrays and readers index into them,
    so a bare scalar here would break them.
    """
    arr = np.asarray(dataset[()])
    if arr.shape:
        return np.full(arr.shape, value, dtype=arr.dtype)
    return np.array(value, dtype=arr.dtype)


def _read_one(group: h5py.Group, key: str):
    """The first element of a header field, or ``None``."""
    if key not in group or not isinstance(group[key], h5py.Dataset):
        return None
    try:
        flat = np.ravel(group[key][()])
        return flat[0] if flat.size else None
    except (TypeError, ValueError, IndexError):
        return None


def _declared_cells(group: h5py.Group) -> Optional[Tuple[int, int]]:
    """``(rows, cols)`` an area's header claims for itself, or ``None``."""
    cols = next((v for v in (_read_one(group, k) for k in _COL_FIELDS)
                 if v is not None), None)
    rows = next((v for v in (_read_one(group, k) for k in _ROW_FIELDS)
                 if v is not None), None)
    if cols is None or rows is None:
        return None
    try:
        rows_i, cols_i = int(rows), int(cols)
    except (TypeError, ValueError):
        return None
    if rows_i < 1 or cols_i < 1:
        return None
    return (rows_i, cols_i)


def _steps_of(group: h5py.Group) -> Optional[Tuple[float, float]]:
    """``(x_step, y_step)`` in um, or ``None``."""
    sx = _read_one(group, "X Step")
    sy = _read_one(group, "Y Step")
    try:
        x = float(sx) if sx is not None else 0.0
        y = float(sy) if sy is not None else x
    except (TypeError, ValueError):
        return None
    if x <= 0 or y <= 0:
        return None
    return (x, y)


def _header_group(h5file: h5py.File, area: str) -> Optional[h5py.Group]:
    """The ``Header`` group of one acquisition area, under any root key."""
    for root in h5file:
        if not isinstance(h5file[root], h5py.Group):
            continue
        path = f"{root}/{area}/Header"
        if path in h5file and isinstance(h5file[path], h5py.Group):
            return h5file[path]
    return None


def _area_geom(h5file: h5py.File, areas: Tuple[str, ...]) -> Optional[dict]:
    """Geometry of the first of ``areas`` that states one.

    Shaped like one entry of ``H5OINADataExtractor.get_pixel_sizes()`` because
    that is what ``project_to_area`` reads -- including ``relative_offset``,
    without which its refusal to carry a window between areas that start at
    different physical points could never fire.

    ``("EBSD", "EDS")`` for the scan side mirrors ``get_grid_dimensions()``:
    an Aztec "Elementverteilungsdaten" acquisition has no EBSD group at all
    and its geometry lives in the EDS header.
    """
    for area in areas:
        group = _header_group(h5file, area)
        if group is None:
            continue
        steps = _steps_of(group)
        if steps is None:
            continue
        return {"x": steps[0], "y": steps[1], "units": "um", "source": "step",
                "relative_offset": _raw_offset(group)}
    return None


def _raw_offset(group: h5py.Group):
    if "Relative Offset" not in group:
        return None
    try:
        flat = np.asarray(group["Relative Offset"][()]).ravel()
        if flat.size < 2:
            return None
        return [float(flat[0]), float(flat[1])]
    except (TypeError, ValueError):
        return None


def _electron_grid(h5file: h5py.File) -> Optional[Tuple[int, int]]:
    """``(rows, cols)`` of the electron-image area.

    The header is authoritative -- a real file stores the image flat, so the
    array shape says nothing. Where there is no header (the older exports and
    the fixtures that mimic them) a 2-D image describes its own grid.
    """
    group = _header_group(h5file, "Electron Image")
    if group is not None:
        cells = _declared_cells(group)
        if cells is not None:
            return cells
    for root in h5file:
        if not isinstance(h5file[root], h5py.Group):
            continue
        path = f"{root}{_ELECTRON_DATA}".rstrip("/")
        if path not in h5file:
            continue
        found: list = []

        def _first_2d(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                found.append(obj.shape[:2])
                return True
            return None

        h5file[path].visititems(_first_2d)
        if found:
            return (int(found[0][0]), int(found[0][1]))
    return None


def _cut_electron(item: h5py.Dataset, grid, rect) -> Optional[np.ndarray]:
    """The window's part of one electron image, or ``None`` to keep it whole.

    Returns the cut in the source's own layout: flat in, flat out. Like the
    per-pixel channels this never reads the whole dataset -- a 2-D image is one
    hyperslab, a flat one is a slice per image row.
    """
    if rect is None or grid is None:
        return None
    rows, cols = grid
    r0, c0 = rect["row0"], rect["col0"]
    n_rows, n_cols = rect["rows"], rect["cols"]
    if item.ndim == 1:
        if item.shape[0] != rows * cols:
            return None
        runs = [item[(r0 + r) * cols + c0:(r0 + r) * cols + c0 + n_cols]
                for r in range(n_rows)]
        return np.concatenate(runs) if runs else np.empty(0, dtype=item.dtype)
    if item.ndim >= 2:
        if tuple(item.shape[:2]) != (rows, cols):
            return None
        return item[r0:r0 + n_rows, c0:c0 + n_cols]
    return None


def _checked_patterns(patterns: np.ndarray, window: CropWindow) -> np.ndarray:
    """The supplied patterns as ``(rows * cols, ...)``, or a loud failure.

    A silent reshape of the wrong array would write a file whose patterns
    belong to different pixels than its channels -- valid-looking and wrong,
    which is the one outcome worth crashing to avoid.
    """
    arr = np.asarray(patterns)
    n_points = int(window.rows * window.cols)
    if arr.ndim >= 4 and tuple(arr.shape[:2]) == (window.rows, window.cols):
        arr = arr.reshape((n_points,) + arr.shape[2:])
    if arr.ndim < 1 or arr.shape[0] != n_points:
        raise ValueError(
            f"supplied patterns have shape {tuple(np.shape(patterns))} but this "
            f"crop window is {window.rows}x{window.cols} = {n_points} points"
        )
    return arr
