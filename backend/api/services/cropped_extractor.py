"""A view of H5OINADataExtractor restricted to a crop window.

EDS element maps, Band Contrast and the per-pixel patterns all reach the app
through H5OINADataExtractor. That makes it the one place where a crop can be
honoured: this proxy cuts what comes out, and the call sites that read those
need no change at all.

A method needs this proxy's attention for one of two independent reasons: it
RETURNS a grid-shaped array, or it TAKES a per-pixel flat index. Grouping by
return shape alone is how ``get_eds_spectrum`` was missed once already, so the
list below is split by both. Check both columns before wiring anything to a
CroppedExtractor.

RETURNS a grid-shaped array — CUT to the window:
    get_grid_dimensions, flat_point_count, get_pattern_count, _flat_to_map,
    get_element_map, get_element_map_2d, get_band_contrast_map,
    detect_available_features

PROJECTED — cut, but not by row and column number:
    get_electron_image. An electron image is a separate acquisition area on
    its own, finer grid over a slightly different field of view, so the window
    travels into it through MICRONS (``crop_window.project_to_area``). Where
    the file does not carry both areas' geometry, or says they start at
    different points, the FULL image comes back with
    ``last_crop_status(name)["cropped"] is False`` and a reason. A cut that
    ran off the far edge and had to be clamped keeps ``cropped`` True but sets
    ``exact`` False — real data, no longer the window's aspect ratio.

TAKES a per-pixel index — TRANSLATED (a crop-local index goes in, the
corresponding original pixel is read out):
    get_pattern_at_index, get_aztec_pixel, get_eds_spectrum,
    index_to_position, position_to_index

NEITHER: everything else forwards through ``__getattr__``, binds to the raw
extractor and knows nothing about the window — while this object advertises a
cropped grid. As of Task 12 the untreated ones are all in the returns-a-grid
column: ``get_scalar_map``, ``get_phase_map`` and ``compute_ipf_map``. No
index-taker is left untranslated, and a test holds that line. Do not put one of
the untreated reads behind a cropped route until its own task has landed.

What this proxy does NOT do: apply the navigation mask. The mask says which
pixels the user selected, not which data exists. A value outside the lasso is
still readable; only computations that mean "selected" ask the mask.
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

    def get_pattern_count(self) -> int:
        """Kept equal to ``prod(get_grid_dimensions())``, as on the raw class.

        A consumer that walks ``range(get_pattern_count())`` into
        ``get_pattern_at_index`` has to stay inside the crop.
        """
        return int(self._window.rows * self._window.cols)

    def _flat_to_map(self, arr):
        """Reshape a per-point channel of the FULL scan, then cut it.

        The argument is on the ORIGINAL grid — that is what the wrapped
        extractor's own ``_flat_to_map`` expects; the result is on the crop's.
        """
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

    # --- coordinates ----------------------------------------------------
    # Everything below speaks the CROP's grid on the way in and on the way
    # out. Handing back a coordinate on a grid other than the one
    # get_grid_dimensions() advertises is silently wrong data.

    def _original_flat_index(self, index) -> int:
        """Crop-local flat index -> flat index on the original grid."""
        w = self._window
        idx = int(index)
        if not (0 <= idx < w.rows * w.cols):
            raise IndexError(
                f"index {idx} is outside the crop ({w.rows}x{w.cols})")
        local_row, local_col = divmod(idx, w.cols)
        return ((w.row0 + local_row) * int(w.original_shape[1])
                + w.col0 + local_col)

    def index_to_position(self, index):
        """Crop-local flat index -> crop-local (row, col)."""
        w = self._window
        idx = int(index)
        if not (0 <= idx < w.rows * w.cols):
            raise IndexError(
                f"index {idx} is outside the crop ({w.rows}x{w.cols})")
        return (idx // int(w.cols), idx % int(w.cols))

    def position_to_index(self, row, col):
        """Crop-local (row, col) -> crop-local flat index."""
        w = self._window
        r, c = int(row), int(col)
        if not (0 <= r < w.rows and 0 <= c < w.cols):
            raise IndexError(
                f"position ({r}, {c}) is outside the crop "
                f"({w.rows}x{w.cols})")
        return r * int(w.cols) + c

    # --- per-pixel reads ------------------------------------------------
    def get_pattern_at_index(self, index, pattern_type: str = "processed"):
        """Translate a flat index in the CROP into the original flat index."""
        return self._ext.get_pattern_at_index(
            self._original_flat_index(index), pattern_type)

    def get_aztec_pixel(self, index):
        """The Aztec record of the pixel at a CROP-local flat index.

        The raw method takes a flat index over its own display grid — it
        bounds the index against ``get_grid_dimensions()`` and reads the
        per-point datasets at that offset, with no Aztec-specific numbering
        involved — so the same translation ``get_pattern_at_index`` uses is
        all it needs.
        """
        return self._ext.get_aztec_pixel(self._original_flat_index(index))

    def get_eds_spectrum(self, index):
        """The EDS spectrum of the pixel at a CROP-local flat index.

        Same family as ``get_aztec_pixel``: the raw method takes a flat index
        over the display grid and reads the per-point datasets at that offset,
        so only the index needs translating.
        """
        return self._ext.get_eds_spectrum(self._original_flat_index(index))

    # --- feature report -------------------------------------------------
    def detect_available_features(self) -> Dict[str, Any]:
        """The raw report with every grid-derived field restated for the crop.

        ``pattern_count`` has to be overwritten here as well as on the method:
        the dict was built by the raw extractor against the raw grid, and one
        report describing two different grids is worse than no report.
        """
        features = dict(self._ext.detect_available_features())
        features["grid_shape"] = self.get_grid_dimensions()
        features["pattern_count"] = self.get_pattern_count()
        features["cropped"] = True
        features["crop_window"] = self._window.to_dict()
        return features

    # --- electron images ------------------------------------------------
    def get_electron_image(self, image_name):
        """Cut an electron image to the crop — through um, not row numbers.

        Three cases, in order:
        1. The image is already on the scan grid -> cut by index.
        2. Both areas expose their step size -> project and cut.
        3. Anything else -> hand back the FULL image and record why. A
           silently mis-cut image would misplace every feature on it.

        The navigation mask is deliberately NOT applied: an electron image has
        no pixel-for-pixel correspondence with the scan grid, so there is no
        honest way to blank "the pixels outside the lasso" on it.
        """
        from backend.api.services.crop_window import project_to_area

        img = self._ext.get_electron_image(image_name)
        if img is None:
            return None
        arr = np.asarray(img)

        if (arr.ndim >= 2
                and tuple(arr.shape[:2]) == tuple(self._window.original_shape)):
            self._crop_status[image_name] = {
                "cropped": True, "exact": True, "reason": None}
            return self._window.apply(arr)

        geoms: Dict[str, Any] = {}
        try:
            geoms = self._ext.get_pixel_sizes() or {}
        except Exception:
            logger.debug("get_pixel_sizes failed for %r", image_name,
                         exc_info=True)

        # The scan grid this window is cut from is the EBSD grid where there
        # is one and the EDS grid otherwise — the same precedence
        # ``get_grid_dimensions`` uses, so an EDS-only acquisition (no EBSD
        # header at all) still places its window.
        scan_geom = geoms.get("ebsd") or geoms.get("eds")

        rect = None
        if arr.ndim >= 2:
            rect = project_to_area(
                self._window, scan_geom, geoms.get("electron_image"),
                arr.shape[:2],
            )
        if rect is None:
            reason = (
                "the file does not give the scan and the electron image a "
                "common geometry: one of them has no step size or bounding "
                "box, or the two areas start at different points"
            )
            self._crop_status[image_name] = {
                "cropped": False, "exact": False, "reason": reason}
            logger.warning("electron image %r not cropped: %s",
                           image_name, reason)
            return arr

        if not rect["exact"]:
            # Handing back the FULL image instead would be worse, not safer:
            # the compositor stretches every layer to the map's size either
            # way, and the full image is further from the window than the
            # clamped cut-out is. So keep the cut-out and say it is short.
            logger.warning(
                "electron image %r cropped short: the window runs past the "
                "edge of that area, so the cut-out is not the window's shape",
                image_name)
        self._crop_status[image_name] = {
            "cropped": True,
            "exact": bool(rect["exact"]),
            "reason": None if rect["exact"] else (
                "the window runs past the edge of the electron image, so the "
                "cut-out had to be clamped and is not the window's shape"),
        }
        return arr[rect["row0"]:rect["row0"] + rect["rows"],
                   rect["col0"]:rect["col0"] + rect["cols"]]

    # --- provenance for things that could not be cut --------------------
    def last_crop_status(self, key: str) -> Dict[str, Any]:
        """Whether the last read of ``key`` could be cut, and why not.

        Filled by the electron-image path (Task 12); returns "cropped and
        exact" for anything that never had a problem.
        """
        return self._crop_status.get(
            key, {"cropped": True, "exact": True, "reason": None})
