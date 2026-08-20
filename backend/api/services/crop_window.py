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


# Two ``Relative Offset`` values closer than this count as the same origin.
# The datum is a fraction of the site, so on the widest file measured here
# (604 um) this is 6e-4 um — far below one pixel of any area, in either
# reading of the unit — while any offset a real sub-region would carry is
# orders of magnitude larger. It exists only to absorb float noise, not to
# tolerate a difference.
_SAME_ORIGIN_TOL = 1e-6


def project_to_area(
    window: CropWindow,
    src_geom: Optional[dict],
    dst_geom: Optional[dict],
    dst_shape: Tuple[int, int],
) -> Optional[Dict[str, object]]:
    """Carry a crop window from one acquisition area into another, via um.

    Electron images do not share the scan grid — measured on the SampleB file
    in this project, 1024x768 at 0.05897 um against 120x90 at 0.5 um, a factor
    of 8.478 on a slightly wider field of view. Row and column numbers
    therefore mean nothing across areas; physical size does.

    Pixel size alone only places the window if the two areas START at the same
    physical point. That is a fact about the file, not an assumption to make:
    each area's ``Relative Offset`` header says where its field of view begins
    inside the site, and ``get_pixel_sizes()`` reports it. Where both areas
    state an offset and the two DISAGREE, this returns ``None`` — the caller
    then shows the full image and says why. Mapping a sub-region inside a wider
    SE field is a routine Aztec workflow, and correcting for the offset is a
    harder problem than refusing to guess at it.

    Where one or both areas do NOT state an offset the window is projected as
    before: an absent datum is no evidence of disagreement, and refusing on it
    would strand every older export that omits the key.

    Each axis uses ITS OWN step ratio: X Step and Y Step are separate numbers
    and sharing one of them would stretch the window on the other axis.

    Returns ``{"row0","col0","rows","cols","exact"}`` or ``None`` when the two
    areas cannot place the window between them. ``exact`` is False when the
    projected rectangle had to be clamped to the destination — the cut-out is
    then real data but no longer the window's aspect ratio, which matters
    because the layer compositor stretches every layer to the map's size.
    """
    def _steps(geom):
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

    src = _steps(src_geom)
    dst = _steps(dst_geom)
    if src is None or dst is None:
        return None

    if not _same_origin(src_geom, dst_geom):
        logger.warning(
            "acquisition areas start at different points (%r vs %r); a window "
            "cannot be carried between them by pixel size alone",
            (src_geom or {}).get("relative_offset"),
            (dst_geom or {}).get("relative_offset"),
        )
        return None

    src_x, src_y = src
    dst_x, dst_y = dst
    dst_rows, dst_cols = int(dst_shape[0]), int(dst_shape[1])

    col0 = int(round(window.col0 * src_x / dst_x))
    row0 = int(round(window.row0 * src_y / dst_y))
    cols = int(round(window.cols * src_x / dst_x))
    rows = int(round(window.rows * src_y / dst_y))

    # No lower clamp: ``CropWindow`` validates row0/col0 as non-negative and
    # both ratios are strictly positive, so the projected origin cannot be
    # negative.
    exact = True
    if col0 >= dst_cols or row0 >= dst_rows:
        # The window starts past the far edge of the other area: there is no
        # cut-out at all, and clamping would hand back a rectangle from the
        # wrong place.
        return None
    if row0 + rows > dst_rows:
        rows = dst_rows - row0
        exact = False
    if col0 + cols > dst_cols:
        cols = dst_cols - col0
        exact = False

    if rows <= 0 or cols <= 0:
        return None

    return {"row0": row0, "col0": col0, "rows": rows, "cols": cols,
            "exact": exact}


def _same_origin(src_geom: Optional[dict], dst_geom: Optional[dict]) -> bool:
    """Do these two areas' fields of view start at the same physical point?

    True unless BOTH state a ``Relative Offset`` and the two differ. An absent
    datum is not a claim that the area starts at the origin, so it is not
    evidence of disagreement either — see ``project_to_area``.
    """
    a = (src_geom or {}).get("relative_offset")
    b = (dst_geom or {}).get("relative_offset")
    if a is None or b is None:
        return True
    try:
        av = [float(v) for v in a]
        bv = [float(v) for v in b]
    except (TypeError, ValueError):
        # Unreadable is not the same as different; fall back to projecting.
        return True
    if len(av) != len(bv):
        return True
    return all(abs(x - y) <= _SAME_ORIGIN_TOL for x, y in zip(av, bv))


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
