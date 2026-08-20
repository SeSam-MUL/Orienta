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
from dataclasses import dataclass
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
    nav_mask: Optional[np.ndarray] = None   # (rows, cols) bool, window-local

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


# --- Per-dataset registry -------------------------------------------------
# In-memory like every other session registry in this app. Keyed by the
# dataset name used in ebsd_viewer._raw_signals.

_lock = threading.RLock()
_windows: Dict[str, CropWindow] = {}


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


def clear_crop(name: str) -> None:
    with _lock:
        _windows.pop(name, None)


def clear_all() -> None:
    with _lock:
        _windows.clear()
