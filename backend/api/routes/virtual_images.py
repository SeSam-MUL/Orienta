"""
Virtual EBSD-derived images for the EDS Analysis page.

Two products:

- **Virtual BSE** (back-scatter electron) — the per-pixel sum of the
  diffraction pattern over a rectangular detector ROI. With the
  default ROI (whole detector) this is essentially the "total
  intensity" image, which gives a structural-contrast picture similar
  to a real BSE detector and is excellent for comparing EDS element
  distributions against grain morphology.

- **Band Contrast** — pattern quality. Oxford H5OINA files store this
  natively at ``/<scan>/EBSD/Data/Band Contrast`` (uint8, 0..255), so
  we read that first; if no native field is present we fall back to
  kikuchipy's ``get_image_quality`` which computes a normalised quality
  metric from the pattern FFTs.

Both endpoints return a Base64 PNG ready for an ``<img>`` tag plus the
raw value range so the UI can show the colour scale.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.api.services.h5_session import get_extractor, is_open
from backend.api.services.image_utils import (
    colormap_array_to_base64,
    element_color_overlay_to_base64,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _signal_data():
    """Return the active EBSD signal's pattern data array, or raise 400."""
    from backend.api.routes.ebsd_viewer import _get_active_signal
    sig = _get_active_signal()
    if sig is None:
        raise HTTPException(
            status_code=400,
            detail="No EBSD data loaded — open a file in the EBSD Viewer first.",
        )
    data = getattr(sig, "data", None)
    if data is None:
        raise HTTPException(
            status_code=400,
            detail="Active EBSD signal has no pattern data.",
        )
    if data.ndim != 4:
        raise HTTPException(
            status_code=400,
            detail=(
                "Virtual BSE / Band Contrast need a 4D pattern stack "
                f"(n_rows, n_cols, det_h, det_w); got {data.ndim}D."
            ),
        )
    return sig, data


def _render_2d_image(arr_2d: np.ndarray, cmap: str, color: str) -> str:
    """Single-color overlay if `color` set, else matplotlib heatmap."""
    if color:
        return element_color_overlay_to_base64(arr_2d.astype(np.float64), color)
    return colormap_array_to_base64(arr_2d.astype(np.float64), cmap)


@router.get("/virtual-bse")
async def virtual_bse(
    cmap: str = "gray",
    color: str = "",
    roi: Optional[str] = None,
):
    """Return the virtual BSE image (per-pixel detector ROI sum).

    Args:
        cmap: matplotlib colormap when no ``color`` is supplied. Defaults
            to ``gray`` because virtual BSE is structural-contrast, not
            chemistry — a hot/viridis lookup misleads people into reading
            it as a heatmap.
        color: hex element-style colour. Pass ``""`` for the default
            grayscale heatmap.
        roi: optional ``"row_min,row_max,col_min,col_max"`` rectangle in
            detector pixel coordinates (inclusive low / exclusive high,
            same as numpy slicing). Defaults to the full detector.
    """
    sig, data = _signal_data()
    n_rows, n_cols, det_h, det_w = data.shape

    if roi:
        try:
            parts = [int(p) for p in roi.split(",")]
            if len(parts) != 4:
                raise ValueError("roi must be four comma-separated ints")
            r0, r1, c0, c1 = parts
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        r0 = max(0, r0); r1 = min(det_h, r1)
        c0 = max(0, c0); c1 = min(det_w, c1)
        if r0 >= r1 or c0 >= c1:
            raise HTTPException(
                status_code=400,
                detail=f"Empty ROI after clamping: {(r0, r1, c0, c1)}",
            )
    else:
        r0, r1, c0, c1 = 0, det_h, 0, det_w

    # Compute per-pixel sum across the detector ROI. ``data.sum`` works
    # for both numpy and dask arrays — kikuchipy lazy signals stay lazy
    # until the ``np.asarray`` call. That materialisation reads the whole
    # pattern stack off disk (multi-GB), so run it off the event loop or
    # the entire backend freezes while it streams.
    def _compute_bse():
        b = data[..., r0:r1, c0:c1].sum(axis=(-2, -1))
        return np.asarray(b, dtype=np.float64)

    bse = await asyncio.to_thread(_compute_bse)

    image_b64 = _render_2d_image(bse, cmap, color)
    return {
        "image": image_b64,
        "shape": [int(n_rows), int(n_cols)],
        "roi": [int(r0), int(r1), int(c0), int(c1)],
        "detector_shape": [int(det_h), int(det_w)],
        "min_val": float(np.nanmin(bse)),
        "max_val": float(np.nanmax(bse)),
        "source": "computed",
    }


def _read_native_band_contrast() -> Optional[np.ndarray]:
    """Best-effort native Band Contrast read from the open H5OINA file.

    Returns a 2D ``(n_rows, n_cols)`` array, or ``None`` if the file
    has no Band Contrast dataset (EDAX, processed-only signals,
    or anywhere the user has stripped the metadata).
    """
    if not is_open():
        return None
    try:
        from backend.api.services.h5_session import get_h5_file
        h5 = get_h5_file()
    except Exception:
        return None

    # Walk top-level scan groups (Oxford uses '1', '2', ... per scan).
    for scan_key in list(h5.keys()):
        try:
            grp = h5[scan_key]
        except Exception:
            continue
        try:
            ds = grp["EBSD/Data/Band Contrast"]
        except Exception:
            continue
        try:
            bc = np.asarray(ds[...])
        except Exception:
            continue
        # Reshape from flat (n_pix,) using the header dims if needed.
        if bc.ndim == 1:
            try:
                hdr = grp["EBSD/Header"]
                n_cols = int(hdr["X Cells"][()])
                n_rows = int(hdr["Y Cells"][()])
                if bc.size == n_rows * n_cols:
                    bc = bc.reshape(n_rows, n_cols)
            except Exception:
                # Without the header we can't reliably reshape.
                return None
        return bc.astype(np.float64)
    return None


def _read_native_band_contrast_from(path) -> Optional[np.ndarray]:
    """Read native Band Contrast directly from a SPECIFIC H5OINA file path.

    Unlike ``_read_native_band_contrast`` (which reads the shared EDS
    h5_session), this opens its OWN read-only handle and closes it again, so a
    phase map can show BC for the result it is displaying even when a DIFFERENT
    file is open in the viewer/EDS page — WITHOUT mutating the shared session
    (the exact cross-page coupling that has bitten this app before). Returns a
    2D ``(n_rows, n_cols)`` array, or None if the file has no Band Contrast
    dataset (EDAX, processed-only, stripped metadata) or can't be opened.
    """
    if not path:
        return None
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(path, "r") as h5:
            for scan_key in list(h5.keys()):
                try:
                    grp = h5[scan_key]
                    bc = np.asarray(grp["EBSD/Data/Band Contrast"][...])
                except Exception:
                    continue
                if bc.ndim == 1:
                    try:
                        hdr = grp["EBSD/Header"]
                        n_cols = int(hdr["X Cells"][()])
                        n_rows = int(hdr["Y Cells"][()])
                        if bc.size == n_rows * n_cols:
                            bc = bc.reshape(n_rows, n_cols)
                        else:
                            continue
                    except Exception:
                        continue
                return bc.astype(np.float64)
    except Exception:
        return None
    return None


@router.get("/band-contrast")
async def band_contrast(
    cmap: str = "gray",
    color: str = "",
):
    """Return the Band Contrast image.

    Tries the native Oxford H5OINA dataset first, then falls back to
    ``kikuchipy.signals.EBSD.get_image_quality`` which is comparable in
    spirit but normalised to ``[0, 1]``.
    """
    if not is_open():
        raise HTTPException(status_code=400, detail="No HDF5 file is open")

    bc = _read_native_band_contrast()
    if bc is not None:
        image_b64 = _render_2d_image(bc, cmap, color)
        return {
            "image": image_b64,
            "shape": [int(bc.shape[0]), int(bc.shape[1])],
            "min_val": float(np.nanmin(bc)),
            "max_val": float(np.nanmax(bc)),
            "source": "h5oina",
            "label": "Band Contrast (native)",
        }

    # Fallback — compute image quality via kikuchipy. This is slower
    # (FFT-per-pattern) but works on any 4D signal.
    sig, data = _signal_data()
    try:
        # ``get_image_quality`` runs an FFT per pattern over the whole
        # stack — heavy + lazy. Materialise off the event loop so the
        # backend stays responsive.
        def _compute_iq():
            return np.asarray(sig.get_image_quality(), dtype=np.float64)

        iq = await asyncio.to_thread(_compute_iq)
    except Exception as exc:
        logger.exception("get_image_quality failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "No native Band Contrast in this file and computing "
                f"image_quality failed: {exc}. The pattern stack may be "
                "non-numeric or the signal axes incorrectly tagged."
            ),
        )

    image_b64 = _render_2d_image(iq, cmap, color)
    return {
        "image": image_b64,
        "shape": [int(iq.shape[0]), int(iq.shape[1])],
        "min_val": float(np.nanmin(iq)),
        "max_val": float(np.nanmax(iq)),
        "source": "computed",
        "label": "Image Quality (kikuchipy fallback)",
    }
