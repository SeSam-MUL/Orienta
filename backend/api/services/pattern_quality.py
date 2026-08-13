"""Single source of truth for the per-pixel pattern-quality map.

Precedence: native Oxford "Band Contrast" (from the file, uint8 0..255) →
``xmap.prop['bc']`` carried from a light-h5 export → kikuchipy FFT
``get_image_quality`` (0..1). No std/mean CoV, no CI×255 surrogate. The
returned ``source`` lets every caller show whether the value is measured
(native) or computed, and never fakes a value it doesn't have.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

LABEL_NATIVE = "Band Contrast (native)"
LABEL_NATIVE_IQ = "Image Quality (native)"
LABEL_COMPUTED = "Pattern Quality (computed)"


@dataclass
class QualityMap:
    array: np.ndarray            # 2D (n_rows, n_cols), RAW scale
    source: str                  # "native" | "computed"
    metric: str                  # "band_contrast" | "image_quality"
    label: str
    value_range: tuple


def read_native_band_contrast(
    source_file: Optional[str],
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Read native Band Contrast from an explicit H5OINA path via a PRIVATE
    read-only handle (never touches the shared h5_session). Returns a 2D
    float array or ``None`` (EDAX / stripped / unreadable)."""
    if not source_file:
        return None
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(source_file, "r") as h5:
            for scan_key in list(h5.keys()):
                try:
                    grp = h5[scan_key]
                    bc = np.asarray(grp["EBSD/Data/Band Contrast"][...])
                except Exception:
                    continue
                if bc.ndim == 1:
                    try:
                        hdr = grp["EBSD/Header"]
                        c = int(hdr["X Cells"][()])
                        r = int(hdr["Y Cells"][()])
                    except Exception:
                        if n_rows and n_cols:
                            r, c = n_rows, n_cols
                        else:
                            continue
                    if bc.size == r * c:
                        bc = bc.reshape(r, c)
                    else:
                        continue
                return bc.astype(np.float64)
    except Exception:
        logger.debug("native BC read failed for %s", source_file, exc_info=True)
        return None
    return None


def read_native_image_quality(
    source_file: Optional[str],
    n_rows: Optional[int] = None,
    n_cols: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Read the EDAX-style ``IQ`` channel, the same way as Oxford's BC.

    Oxford writes "Band Contrast", EDAX writes "IQ" — different quantities
    (band sharpness in the Hough transform vs. a pattern-quality index) but
    both measured by the vendor and stored in the file. A file that has IQ and
    no BC used to yield nothing at all: the map refused to draw, and the
    honest reading "this file has no band contrast" was of no use to someone
    holding a perfectly good quality channel.

    Returns the raw values (EDAX IQ has no fixed range) or ``None``.
    """
    if not source_file:
        return None
    try:
        import h5py
    except Exception:
        return None
    try:
        with h5py.File(source_file, "r") as h5:
            for scan_key in list(h5.keys()):
                try:
                    grp = h5[scan_key]
                    iq = np.asarray(grp["EBSD/Data/IQ"][...])
                except Exception:
                    continue
                if iq.ndim == 1:
                    r = c = None
                    try:
                        hdr = grp["EBSD/Header"]
                        c = int(np.ravel(hdr["nColumns"][()])[0])
                        r = int(np.ravel(hdr["nRows"][()])[0])
                    except Exception:
                        r = c = None
                    if not (r and c and iq.size == r * c) and n_rows and n_cols:
                        r, c = n_rows, n_cols
                    if not (r and c and iq.size == r * c):
                        continue
                    iq = iq.reshape(r, c)
                return iq.astype(np.float64)
    except Exception:
        logger.debug("native IQ read failed for %s", source_file, exc_info=True)
        return None
    return None


def _prop_bc(xmap, n_rows, n_cols) -> Optional[np.ndarray]:
    """Real native BC carried in ``xmap.prop['bc']`` (from a light-h5)."""
    try:
        prop = getattr(xmap, "prop", None)
        if not prop or "bc" not in prop:
            return None
        arr = np.asarray(prop["bc"], dtype=np.float64).ravel()
        if arr.size != n_rows * n_cols:
            return None
        return arr.reshape(n_rows, n_cols)
    except Exception:
        return None


def compute_image_quality(signal) -> np.ndarray:
    """kikuchipy FFT image quality (0..1). Heavy (FFT per pattern)."""
    return np.asarray(signal.get_image_quality(), dtype=np.float64)


def get_quality_map(
    n_rows: int,
    n_cols: int,
    *,
    source_file: Optional[str] = None,
    signal=None,
    xmap=None,
    allow_compute: bool = True,
) -> Optional[QualityMap]:
    """Return the per-pixel quality map by precedence, or ``None`` when no
    source is available (caller decides whether that is an error or a skip)."""
    native = read_native_band_contrast(source_file, n_rows, n_cols)
    if native is None:
        native = _prop_bc(xmap, n_rows, n_cols)
    if native is not None:
        return QualityMap(native, "native", "band_contrast", LABEL_NATIVE, (0, 255))

    # The vendor's own quality channel, when it is EDAX rather than Oxford.
    # Measured data beats anything computed here, so it comes before the FFT.
    iq_native = read_native_image_quality(source_file, n_rows, n_cols)
    if iq_native is not None:
        lo = float(np.nanmin(iq_native))
        hi = float(np.nanmax(iq_native))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        return QualityMap(iq_native, "native", "image_quality", LABEL_NATIVE_IQ, (lo, hi))

    if allow_compute and signal is not None:
        iq = compute_image_quality(signal)
        iq = np.asarray(iq, dtype=np.float64).reshape(n_rows, n_cols)
        return QualityMap(iq, "computed", "image_quality", LABEL_COMPUTED, (0, 1))

    return None
