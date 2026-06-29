"""EDS multi-element overlay renderer.

Business logic for compositing multiple EDS element maps into a single
RGBA overlay array. Uses eds_utils.py for unit conversion.

No PyQt5 dependencies — this is a business logic module.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color string to (r, g, b) floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def render_overlay(
    eds_counts: dict,
    active_elements: List[str],
    display_mode: str,
    color_config,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Composite multiple element layers into a single RGBA overlay.

    Parameters
    ----------
    eds_counts : dict
        ``{'counts': {element: 1D_array}, 'n_rows': int, 'n_cols': int}``
    active_elements : list of str
        Elements to render (checked in panel).
    display_mode : str
        ``"Counts"``, ``"Wt%"``, or ``"At%"``
    color_config
        ``EDSColorConfig`` instance with color/opacity settings.
    n_rows, n_cols : int
        Target grid dimensions.

    Returns
    -------
    np.ndarray
        Shape ``(n_rows, n_cols, 4)``, dtype float32, values [0, 1].
    """
    result = np.zeros((n_rows, n_cols, 4), dtype=np.float32)

    if not active_elements:
        return result

    eds_rows = eds_counts.get("n_rows", n_rows)
    eds_cols = eds_counts.get("n_cols", n_cols)
    need_resize = (eds_rows, eds_cols) != (n_rows, n_cols)

    for element in active_elements:
        raw = eds_counts["counts"].get(element)
        if raw is None:
            continue

        # 1. Reshape to 2D
        data_2d = raw.astype(np.float32).reshape(eds_rows, eds_cols)

        # 2. Convert units if needed
        if display_mode == "Wt%":
            from eds_utils import counts_to_weight_pct
            wt = counts_to_weight_pct({element: raw})
            data_2d = wt[element].astype(np.float32).reshape(eds_rows, eds_cols)
        elif display_mode == "At%":
            from eds_utils import counts_to_all
            _, _, at = counts_to_all({element: raw})
            data_2d = at[element].astype(np.float32).reshape(eds_rows, eds_cols)

        # 3. Resize if EDS grid differs from target
        if need_resize:
            from scipy.ndimage import zoom
            zoom_y = n_rows / eds_rows
            zoom_x = n_cols / eds_cols
            data_2d = zoom(data_2d, (zoom_y, zoom_x), order=1)

        # 4. Normalize to [0, 1]
        positive = data_2d[data_2d > 0]
        vmax = float(np.percentile(positive, 99)) if len(positive) > 0 else 1.0
        vmax = max(vmax, 1e-10)
        normalized = np.clip(data_2d / vmax, 0.0, 1.0)

        # 5. Create RGBA layer
        r, g, b = hex_to_rgb(color_config.get_color(element))
        opacity = color_config.get_opacity(element)

        layer = np.zeros((n_rows, n_cols, 4), dtype=np.float32)
        layer[..., 0] = r
        layer[..., 1] = g
        layer[..., 2] = b
        layer[..., 3] = normalized * opacity

        # 6. Alpha compositing (over operation)
        alpha_new = layer[..., 3:4]
        alpha_old = result[..., 3:4]
        result[..., :3] = (
            layer[..., :3] * alpha_new + result[..., :3] * (1 - alpha_new)
        )
        result[..., 3:4] = alpha_new + alpha_old * (1 - alpha_new)

    return result
