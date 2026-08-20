"""A view of H5OINADataExtractor restricted to a crop window.

Every piece of side data in this app — EDS element maps, Band Contrast,
electron images, pixel geometry — is read through H5OINADataExtractor. That
makes it the one place where a crop can be honoured: this proxy cuts what
comes out, and the ~20 call sites that read "the active dataset's side data"
need no change at all.

What it does NOT do: apply the navigation mask. The mask says which pixels
the user selected, not which data exists. A value outside the lasso is still
readable; only computations that mean "selected" ask the mask.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class CroppedExtractor:
    """Wraps an extractor so every grid-shaped read is cut to ``window``.

    Cut arrays are numpy VIEWS of what the wrapped extractor returned: these
    are read paths, and copying an EDS map on every request would be waste.
    Callers that mutate a returned map must copy it first — the same rule the
    raw extractor already imposes for arrays it caches.
    """

    def __init__(self, extractor, window):
        # Bypass __setattr__ indirection: plain attributes on this object.
        object.__setattr__(self, "_ext", extractor)
        object.__setattr__(self, "_window", window)
        object.__setattr__(self, "_crop_status", {})

    # --- identity -------------------------------------------------------
    @property
    def raw(self):
        """The underlying uncropped extractor."""
        return self._ext

    @property
    def crop_window(self):
        return self._window

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes this class does not define.
        return getattr(self._ext, name)

    def __repr__(self) -> str:
        w = self._window
        return (f"<CroppedExtractor rows {w.row0}-{w.row0 + w.rows}, "
                f"cols {w.col0}-{w.col0 + w.cols} of {w.original_shape}>")

    # --- grid -----------------------------------------------------------
    def get_grid_dimensions(self):
        return (int(self._window.rows), int(self._window.cols))

    @property
    def flat_point_count(self) -> int:
        return int(self._window.rows * self._window.cols)

    def _flat_to_map(self, arr):
        """Reshape a per-point channel of the FULL scan, then cut it."""
        return self._window.apply(self._ext._flat_to_map(arr))

    # --- EDS ------------------------------------------------------------
    def get_element_map(self, element_name):
        """Flat per-point counts — of the CROP, row-major over its grid.

        The flat form has to agree with ``get_grid_dimensions``: consumers
        compute ``row * n_cols + col`` against it.
        """
        data = self._ext.get_element_map(element_name)
        if data is None:
            return None
        two_d = self._to_cropped_grid(data)
        if two_d is None:
            return data      # not on the scan grid — nothing to place
        return two_d.reshape(-1, *two_d.shape[2:])

    def get_element_map_2d(self, element_name):
        data = self._ext.get_element_map(element_name)
        if data is None:
            return None
        two_d = self._to_cropped_grid(data)
        return data if two_d is None else two_d

    def get_band_contrast_map(self):
        bc = self._ext.get_band_contrast_map()
        if bc is None:
            return None
        return self._window.apply(bc)

    def _to_cropped_grid(self, data):
        """Full-scan flat or 2D array -> cropped 2D array, or None if the
        array is not on the scan grid at all."""
        arr = np.asarray(data)
        if arr.ndim >= 2 and tuple(arr.shape[:2]) == tuple(self._window.original_shape):
            return self._window.apply(arr)
        if arr.ndim >= 1 and arr.shape[0] == self._ext.flat_point_count:
            return self._window.apply(self._ext._flat_to_map(arr))
        return None

    # --- patterns -------------------------------------------------------
    def get_pattern_at_index(self, index, pattern_type: str = "processed"):
        """Translate a flat index in the CROP into the original flat index."""
        w = self._window
        idx = int(index)
        if not (0 <= idx < w.rows * w.cols):
            raise IndexError(
                f"pattern index {idx} is outside the crop ({w.rows}x{w.cols})")
        local_row, local_col = divmod(idx, w.cols)
        orig_row = w.row0 + local_row
        orig_col = w.col0 + local_col
        orig_index = orig_row * int(w.original_shape[1]) + orig_col
        return self._ext.get_pattern_at_index(orig_index, pattern_type)

    # --- feature report -------------------------------------------------
    def detect_available_features(self) -> Dict[str, Any]:
        features = dict(self._ext.detect_available_features())
        features["grid_shape"] = self.get_grid_dimensions()
        features["cropped"] = True
        features["crop_window"] = self._window.to_dict()
        return features

    # --- provenance for things that could not be cut --------------------
    def last_crop_status(self, key: str) -> Dict[str, Any]:
        """Whether the last read of ``key`` could be cut, and why not.

        Filled by the electron-image path (Task 12); returns "cropped" for
        anything that never had a problem.
        """
        return self._crop_status.get(key, {"cropped": True, "reason": None})
