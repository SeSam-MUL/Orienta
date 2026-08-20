"""Per-pixel EDS atomic-% for the active dataset (shared service).

Extracted verbatim from crystal_hint._get_pixel_at_pct so both the Crystal
Hint route and the Single-Pixel Phase Test route can use it. Returns None on
any failure so chemistry weighting fails soft to pattern-only ranking.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _is_open() -> bool:
    from backend.api.services.h5_session import is_open
    return is_open()


def _get_extractor():
    # The pixel index this module is handed is a flat index over the ACTIVE
    # dataset's grid (it is cross-checked against the EBSD nav shape below),
    # so the extractor has to be the active dataset's view. Off the crop path
    # get_active_extractor() is the raw extractor, so nothing changes there.
    from backend.api.services.h5_session import get_active_extractor
    return get_active_extractor()


def _get_active_signal():
    from backend.api.routes import ebsd_viewer as _ev
    return _ev._get_active_signal()


def get_pixel_at_pct(pixel_index: int) -> Optional[dict[str, float]]:
    """Per-pixel EDS atomic-% for the active dataset, or None."""
    try:
        if not _is_open():
            return None
        ext = _get_extractor()
        elements = ext.get_available_elements()
        if not elements:
            return None
        n_rows, n_cols = ext.get_grid_dimensions()
        if pixel_index < 0 or pixel_index >= n_rows * n_cols:
            return None
        try:
            _sig = _get_active_signal()
            if _sig is not None:
                _nav = _sig.axes_manager.navigation_shape
                _ebsd_total = int(np.prod([int(x) for x in _nav])) if _nav else 0
                if _ebsd_total and _ebsd_total != n_rows * n_cols:
                    logger.warning(
                        "EDS grid (%d) != EBSD nav grid (%d) — skipping EDS "
                        "chemistry to avoid misaligned pixels",
                        n_rows * n_cols, _ebsd_total)
                    return None
        except Exception:
            pass  # can't cross-check → proceed (fail-soft)
        from eds_utils import (parse_element_name, counts_to_weight_pct,
                               weight_pct_to_atomic_pct)
        counts: dict[str, np.ndarray] = {}
        for el_name in elements:
            data = ext.get_element_map(el_name)
            if data is not None and pixel_index < len(data):
                counts[parse_element_name(el_name)] = np.array(
                    [float(data[pixel_index])])
        if not counts:
            return None
        at = weight_pct_to_atomic_pct(counts_to_weight_pct(counts))
        return {el: float(v[0]) for el, v in at.items()}
    except Exception as exc:
        logger.warning("EDS At%% lookup failed for pixel %d: %s", pixel_index, exc)
        return None
