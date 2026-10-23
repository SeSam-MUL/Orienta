"""Where a cropped dataset sits inside its source scan.

A crop never nests. Cropping a crop composes the two windows into ONE window
relative to the original grid (``compose``), so every consumer only ever sees
a single ``(row0, col0, rows, cols)`` and no one has to walk a chain.

The window is defined on the DISPLAY grid — the grid
``H5OINADataExtractor.get_grid_dimensions()`` reports and the EBSD signal's
navigation shape uses. On a hex scan that is the resampled square grid, not
the padded rectangle the file stores.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropWindow:
    """A rectangular window on the original display grid, plus an optional
    navigation mask for the non-rectangular shapes.

    The mask NEVER removes data: patterns outside the drawn shape stay in the
    dataset. The mask only says which pixels the user selected, and indexing
    honours it.
    """

    source_file: str
    row0: int
    col0: int
    rows: int
    cols: int
    original_shape: Tuple[int, int]
    shape_kind: str = "rect"           # "rect" | "ellipse" | "lasso"
    # Window-local (rows, cols) bool. Excluded from comparison: an array field
    # would make the generated __eq__ raise "truth value of an array is
    # ambiguous" and __hash__ fail on the unhashable array.
    nav_mask: Optional[np.ndarray] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Reject a window that cannot be cut, at the point it is built.

        Numpy slicing truncates a window that runs off the grid and WRAPS a
        negative origin to the opposite edge — both silently, both producing
        data from the wrong place. The window is authoritative for every
        consumer, so it has to be impossible to build a wrong one.

        ``shape_kind`` is deliberately not validated: the label is descriptive,
        and a new selection tool should not have to edit this module.
        """
        grid_rows, grid_cols = (int(self.original_shape[0]),
                                int(self.original_shape[1]))
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(
                f"a crop window must select at least one pixel, got "
                f"rows={self.rows}, cols={self.cols}"
            )
        if self.row0 < 0 or self.col0 < 0:
            raise ValueError(
                f"a crop window starts inside the grid, got row0={self.row0}, "
                f"col0={self.col0}"
            )
        if self.row0 + self.rows > grid_rows:
            raise ValueError(
                f"crop window covers rows {self.row0}-{self.row0 + self.rows} "
                f"but the grid has {grid_rows} rows"
            )
        if self.col0 + self.cols > grid_cols:
            raise ValueError(
                f"crop window covers cols {self.col0}-{self.col0 + self.cols} "
                f"but the grid has {grid_cols} cols"
            )
        if self.nav_mask is not None:
            mask_shape = tuple(np.shape(self.nav_mask))
            if mask_shape != (self.rows, self.cols):
                raise ValueError(
                    f"nav_mask has shape {mask_shape} but this crop window is "
                    f"{(self.rows, self.cols)}"
                )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def n_selected(self) -> int:
        if self.nav_mask is None:
            return int(self.rows * self.cols)
        return int(np.count_nonzero(self.nav_mask))

    def apply(self, arr) -> np.ndarray:
        """Cut an array whose first two axes are the ORIGINAL display grid.

        Trailing axes (RGB, the 3 of a PC map) are kept untouched.
        """
        a = np.asarray(arr)
        if a.ndim < 2 or tuple(a.shape[:2]) != tuple(self.original_shape):
            raise ValueError(
                f"array has grid {tuple(a.shape[:2])} but this crop window was "
                f"cut from {tuple(self.original_shape)}"
            )
        return a[self.row0:self.row0 + self.rows,
                 self.col0:self.col0 + self.cols]

    def to_dict(self) -> dict:
        """JSON-safe summary for API responses (the mask itself is not sent —
        it can be thousands of entries; the UI draws its own outline)."""
        return {
            "source_file": str(self.source_file),
            "row0": int(self.row0),
            "col0": int(self.col0),
            "rows": int(self.rows),
            "cols": int(self.cols),
            "original_shape": [int(v) for v in self.original_shape],
            "shape": str(self.shape_kind),
            "n_selected": int(self.n_selected),
            "has_mask": bool(self.nav_mask is not None),
        }


def compose(outer: CropWindow, inner: CropWindow) -> CropWindow:
    """Fold a window cut FROM a cropped dataset back onto the original grid."""
    if tuple(inner.original_shape) != tuple(outer.shape):
        raise ValueError(
            f"the inner window was cut from {tuple(inner.original_shape)} but "
            f"the outer dataset is {tuple(outer.shape)}"
        )

    mask = None
    if outer.nav_mask is not None:
        mask = np.array(inner.apply(outer.nav_mask), dtype=bool)
    if inner.nav_mask is not None:
        inner_mask = np.asarray(inner.nav_mask, dtype=bool)
        mask = inner_mask.copy() if mask is None else (mask & inner_mask)

    # The shape label describes how the SELECTION was drawn. An inner
    # rectangle over a lasso is still a lasso selection.
    kind = inner.shape_kind if inner.shape_kind != "rect" else outer.shape_kind

    return CropWindow(
        source_file=outer.source_file,
        row0=int(outer.row0 + inner.row0),
        col0=int(outer.col0 + inner.col0),
        rows=int(inner.rows),
        cols=int(inner.cols),
        original_shape=tuple(outer.original_shape),
        shape_kind=kind,
        nav_mask=mask,
    )


# The header's own size figure and the one derived from step x cell count may
# differ by the odd percent (the Oxford reference scan: 60.39 um image field against a 60.0 um
# map, 0.65 %); beyond this the two disagree about where the map is, and no
# placement is claimed.
_SIZE_TOL = 0.03


def _vec2(v) -> Optional[Tuple[float, float]]:
    if v is None:
        return None
    try:
        flat = [float(x) for x in v]
    except (TypeError, ValueError):
        return None
    if len(flat) < 2:
        return None
    return flat[0], flat[1]


def _steps(geom) -> Optional[Tuple[float, float]]:
    if not geom:
        return None
    try:
        sx = float(geom.get("x") or 0.0)
        sy = float(geom.get("y") or sx)
    except (TypeError, ValueError, AttributeError):
        return None
    if sx <= 0 or sy <= 0:
        return None
    return sx, sy


def electron_footprint(
    src_geom: Optional[dict],
    dst_geom: Optional[dict],
    scan_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    """Where the whole scan grid sits inside an electron image's pixels.

    Returns ``(footprint, reason)``. ``footprint`` is ``{"mode", "x0", "y0",
    "sx", "sy"}``: the scan's top-left corner in image pixels and the number of
    image pixels per scan pixel on each axis. ``mode`` is

    * ``"header"`` -- placed by the file's own ``Relative Offset`` /
      ``Relative Size``;
    * ``"assumed"`` -- the scan states no offset, so it is taken to start where
      the image starts and to be sized by step ratio alone (the pre-registration
      behaviour; nothing in the file backs it).

    ``(None, reason)`` when the two areas cannot be placed against each other.

    The frame, measured on every Oxford file we have (an Oxford reference scan and five
    scans of an aluminium alloy): the electron image states ``Relative Offset [0, 0]`` and
    ``Relative Size [1, 1]``, and the scan states its position and extent as
    FRACTIONS OF THE IMAGE's bounding box -- e.g. one of the alloy scans, offset
    [0.0696, 0.2550], size [0.7882, 0.3014], and 0.7882 x 17.25 um = 13.6 um =
    136 cells x 0.1 um. A scan whose offset differs from the image's is
    therefore not an inconsistency to refuse, it is the ordinary case of a
    map acquired inside a wider SE field, and subtracting the image's own
    offset (and dividing by its size, which generalises the frame to an image
    that is itself a sub-field) is the whole correction.

    The step x cells figure is the cross-check on ``Relative Size``: where they
    disagree by more than ``_SIZE_TOL`` the header contradicts itself.
    """
    src = _steps(src_geom)
    dst = _steps(dst_geom)
    if src is None or dst is None:
        return None, "one of the areas has no step size"
    rows, cols = int(scan_shape[0]), int(scan_shape[1])
    img_h, img_w = int(dst_shape[0]), int(dst_shape[1])
    if min(rows, cols, img_h, img_w) <= 0:
        return None, "an area has no pixels"

    off_m = _vec2((src_geom or {}).get("relative_offset"))
    if off_m is None:
        return ({"mode": "assumed", "x0": 0.0, "y0": 0.0,
                 "sx": src[0] / dst[0], "sy": src[1] / dst[1]}, None)

    off_i = _vec2((dst_geom or {}).get("relative_offset")) or (0.0, 0.0)
    size_i = _vec2((dst_geom or {}).get("relative_size")) or (1.0, 1.0)
    if size_i[0] <= 0 or size_i[1] <= 0:
        return None, "the electron image states a non-positive Relative Size"

    # The scan's extent in micrometres, from step x cells, and the image's.
    um_w, um_h = cols * src[0], rows * src[1]
    img_um_w, img_um_h = img_w * dst[0], img_h * dst[1]

    size_m = _vec2((src_geom or {}).get("relative_size"))
    if size_m is None:
        fw, fh = um_w / img_um_w, um_h / img_um_h
    else:
        if size_m[0] <= 0 or size_m[1] <= 0:
            return None, "the scan states a non-positive Relative Size"
        fw, fh = size_m[0] / size_i[0], size_m[1] / size_i[1]
        if (abs(fw * img_um_w - um_w) > _SIZE_TOL * um_w
                or abs(fh * img_um_h - um_h) > _SIZE_TOL * um_h):
            return None, (
                "the scan's Relative Size and its step x cell count give "
                "different extents (%.2f x %.2f um against %.2f x %.2f um)"
                % (fw * img_um_w, fh * img_um_h, um_w, um_h))

    fx0 = (off_m[0] - off_i[0]) / size_i[0]
    fy0 = (off_m[1] - off_i[1]) / size_i[1]
    return ({"mode": "header",
             "x0": fx0 * img_w, "y0": fy0 * img_h,
             "sx": fw * img_w / cols, "sy": fh * img_h / rows}, None)


def project_to_area(
    window: CropWindow,
    src_geom: Optional[dict],
    dst_geom: Optional[dict],
    dst_shape: Tuple[int, int],
) -> Optional[Dict[str, object]]:
    """Carry a window of the scan grid into an electron image's pixels.

    Electron images do not share the scan grid -- measured on the Oxford reference scan file
    in this project, 1024x768 at 0.05897 um against 120x90 at 0.5 um -- and
    they usually cover a different field of view. Row and column numbers
    therefore mean nothing across areas; the header geometry does, see
    ``electron_footprint``.

    Native pixels only, never resampled: the box is rounded to whole image
    pixels, which is at most half an image pixel at either edge (a small
    fraction of a scan pixel, since the image is the finer grid).

    Returns ``None`` when the areas cannot place the window. Otherwise a dict:

    ``row0, col0, rows, cols``  the part of the box that lies inside the image
    ``box``                      ``(y0, y1, x0, x1)`` -- the whole box, which in
                                 ``header`` mode may run past the image; the
                                 caller then pads that part transparent, so the
                                 returned picture still has the window's shape
    ``mode``                     ``"header"`` or ``"assumed"``
    ``exact``                    False only in ``assumed`` mode, where the box
                                 is clamped to the image instead of padded
    """
    fp, reason = electron_footprint(
        src_geom, dst_geom, tuple(window.original_shape), dst_shape)
    if fp is None:
        logger.warning("electron image cannot be registered: %s", reason)
        return None

    img_h, img_w = int(dst_shape[0]), int(dst_shape[1])
    x0 = int(round(fp["x0"] + window.col0 * fp["sx"]))
    x1 = int(round(fp["x0"] + (window.col0 + window.cols) * fp["sx"]))
    y0 = int(round(fp["y0"] + window.row0 * fp["sy"]))
    y1 = int(round(fp["y0"] + (window.row0 + window.rows) * fp["sy"]))
    if x1 <= x0 or y1 <= y0:
        return None

    cx0, cx1 = max(x0, 0), min(x1, img_w)
    cy0, cy1 = max(y0, 0), min(y1, img_h)
    if cx1 <= cx0 or cy1 <= cy0:
        # The window lies wholly outside the image: no cut-out at all, and
        # clamping would hand back a rectangle from the wrong place.
        return None

    clamped = (cx0, cx1, cy0, cy1) != (x0, x1, y0, y1)
    exact = True
    if clamped and fp["mode"] != "header":
        # Without header backing there is nothing to say where the missing
        # part is: keep the real data, shorter, and say so.
        exact = False
        box = (cy0, cy1, cx0, cx1)
    else:
        box = (y0, y1, x0, x1)
    return {"row0": cy0, "col0": cx0, "rows": cy1 - cy0, "cols": cx1 - cx0,
            "exact": exact, "mode": fp["mode"], "box": box}


# --- Per-dataset registry -------------------------------------------------
# In-memory like every other session registry in this app. Keyed by the
# dataset name used in ebsd_viewer._raw_signals.

_lock = threading.RLock()
_windows: Dict[str, CropWindow] = {}

# Masks switched OFF are parked here, keyed by dataset name, so "off" does not
# destroy what the user drew: the window on file loses its nav_mask (the whole
# bounding box counts as selected again) while the drawn shape waits here to be
# handed back verbatim when the mask is switched on again.
_stashed_masks: Dict[str, np.ndarray] = {}


def set_crop(name: str, window: CropWindow) -> None:
    with _lock:
        _windows[name] = window
    logger.info(
        "crop window for %r: rows %d-%d, cols %d-%d of %s (%s, %d selected)",
        name, window.row0, window.row0 + window.rows,
        window.col0, window.col0 + window.cols,
        window.original_shape, window.shape_kind, window.n_selected,
    )


def get_crop(name: str) -> Optional[CropWindow]:
    with _lock:
        return _windows.get(name)


def set_mask_enabled(name: str, enabled: bool) -> CropWindow:
    """Switch a crop's navigation mask on or off, keeping it either way.

    Off means the whole bounding box counts as selected — the crop itself,
    including its ``shape_kind`` label, is untouched. Raises ``KeyError`` for a
    dataset that has no crop window: there is nothing to toggle, and answering
    silently would let a caller believe it had changed something.
    """
    with _lock:
        window = _windows[name]
        if enabled:
            mask = _stashed_masks.pop(name, None)
            if mask is None:
                if window.nav_mask is None and window.shape_kind != "rect":
                    # The window still calls itself a lasso/ellipse but neither
                    # a live nor a parked mask exists — the drawn shape is gone.
                    # Answering "success, nothing changed" told the user their
                    # shape was back when it was not.
                    raise KeyError(
                        f"the navigation mask drawn for {name!r} "
                        f"({window.shape_kind}) is no longer held, so it "
                        f"cannot be switched back on — draw the shape again"
                    )
                return window          # already on, or there never was one
            updated = replace(window, nav_mask=mask)
        else:
            if window.nav_mask is None:
                return window
            _stashed_masks[name] = window.nav_mask
            updated = replace(window, nav_mask=None)
        _windows[name] = updated
    logger.info("navigation mask for %r: %s (%d selected)",
                name, "on" if enabled else "off", updated.n_selected)
    return updated


def snapshot_stashed_masks(names) -> Dict[str, np.ndarray]:
    """The parked (switched-off) masks for ``names``.

    A switched-off mask lives HERE, not on the window, so anything that saves
    "the crop" by saving ``get_crop(name)`` saves the window WITHOUT it. The
    per-file dataset stash did exactly that, and the load path calls
    ``clear_all()``, which wipes this table — so a file A -> B -> A round trip
    restored a maskless window whose ``shape_kind`` still said "lasso", and
    switching the mask back on returned success having changed nothing. The
    drawn shape was gone for good.
    """
    with _lock:
        return {n: _stashed_masks[n] for n in names if n in _stashed_masks}


def restore_stashed_masks(masks: Dict[str, np.ndarray]) -> None:
    """Put parked masks back, as saved by :func:`snapshot_stashed_masks`.

    Restores only names that have no parked mask already: a live session's
    answer beats a restored one, the same rule the surrounding registry
    restore follows for the freshly re-read raw dataset.
    """
    with _lock:
        for name, mask in (masks or {}).items():
            _stashed_masks.setdefault(name, mask)


def clear_crop(name: str) -> None:
    with _lock:
        _windows.pop(name, None)
        # A parked mask outlives its window otherwise, and would be handed to
        # whatever crop next claims the same dataset name.
        _stashed_masks.pop(name, None)


def clear_all() -> None:
    with _lock:
        _windows.clear()
        _stashed_masks.clear()
