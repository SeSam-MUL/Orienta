"""Single source of truth for the per-pixel pattern-quality map.

Precedence: native Oxford "Band Contrast" (from the file, uint8 0..255) →
``xmap.prop['bc']`` carried from a light-h5 export → kikuchipy FFT
``get_image_quality`` (0..1). No std/mean CoV, no CI×255 surrogate. The
returned ``source`` lets every caller show whether the value is measured
(native) or computed, and never fakes a value it doesn't have.

The two file-based readers are CROP-AWARE: when the active dataset is a cut-out
of the file being read, and the caller asked for the cut-out's grid, the
channel comes back cut to the same window — see ``_cut_to_active_crop``.
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


def _cut_to_active_crop(arr, source_file, n_rows, n_cols):
    """Cut a FULL-SCAN channel to the active crop window, when it applies.

    The two other native-BC readers were made crop-aware
    (``virtual_images._native_band_contrast_for_active_dataset`` and
    ``eds._bc_grid_for``); this one was not. It reshapes to the FILE header's
    ``X Cells`` / ``Y Cells`` — correct, the file always holds the whole scan —
    and so always answered with the full scan. Its callers guard by shape and
    drop it, and in ``get_quality_map`` the native branch RETURNS that full-scan
    array, shadowing its own computed fallback: on a crop of an Oxford file the
    BC layer ended up with no file-based source at all and the endpoint 400'd
    with "this file has neither Band Contrast (Oxford) nor IQ (EDAX), and no
    patterns are loaded" — three claims, none of them true.

    Cut only when every one of these holds, so nothing else changes:

    * the caller named a grid (all real callers do);
    * a crop window is active;
    * it was cut from THIS file — the caller may be reading a result's own
      source file, which need not be the one that is open;
    * the array really is the full scan the window describes;
    * and the grid asked for is the window's. A result indexed BEFORE the crop
      asks for the full grid and must keep getting the full scan.

    A failed lookup returns ``None``: a read may degrade to "no value", but it
    may not degrade to "here is the full scan", which is the outcome this
    exists to prevent. ``None`` sends ``get_quality_map`` on to the computed
    fallback, which is derived from the ACTIVE signal and cannot show the wrong
    region.
    """
    if arr is None or not n_rows or not n_cols:
        return arr
    try:
        from backend.api.routes.ebsd_viewer import (
            get_active_crop_window, _canonical_path,
        )
    except ImportError:
        logger.debug("ebsd_viewer not importable — no crop window", exc_info=True)
        return arr
    try:
        window = get_active_crop_window()
    except Exception:
        logger.warning("could not read the active crop window — reporting no "
                       "native quality channel rather than the full scan",
                       exc_info=True)
        return None
    if window is None:
        return arr

    src = str(getattr(window, "source_file", "") or "")
    if not src or not source_file:
        return arr
    if _canonical_path(src) != _canonical_path(str(source_file)):
        return arr
    if tuple(arr.shape[:2]) != tuple(int(v) for v in window.original_shape):
        return arr
    if (int(window.rows), int(window.cols)) != (int(n_rows), int(n_cols)):
        return arr
    return np.asarray(window.apply(arr))


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
                return _cut_to_active_crop(
                    bc.astype(np.float64), source_file, n_rows, n_cols)
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
                return _cut_to_active_crop(
                    iq.astype(np.float64), source_file, n_rows, n_cols)
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


# ---------------------------------------------------------------------------
# Is this data good enough for template matching?
# ---------------------------------------------------------------------------

# Median FFT image quality below which dictionary (template) indexing cannot
# work. The "image quality" column is what THIS function reports — i.e. after
# the static + dynamic background removal it performs. An earlier version of
# this table quoted the RAW figures (0.374 / 0.037), which are not what the
# thresholds below are compared against; re-measured through the shipped
# function 2026-08-14:
#
#   dataset                 image quality   raw    best NCC
#   LoGainNi.h5                  0.3616     0.379   0.469   works
#   HiGainNi.h5                  0.0285     0.036   0.115   noise
#   AL_SI_x3000 (raw)            ~0.022             0.045   noise
#   AL_SI_x3000 (3x3 averaged)   ~0.082             0.075   noise
#
# The LoGainNi NCC is from the full-map 2 deg acceptance run against Hough
# (28,086 px, 100,347 orientations); the AL_SI figures predate this function
# and are raw-IQ, kept as an order-of-magnitude marker rather than a
# calibration point.
#
# Caveat when reading a "marginal" verdict: preprocessing moves this number.
# On LoGainNi, dynamic background alone gives 0.358, but the app's own
# "Recommended Pipeline" (dynamic + CLAHE) gives 0.272 — most of the headroom
# above the 0.25 threshold. A dataset can be flagged for how it was processed
# rather than for what it is.
#
# Hough and spherical indexing tolerate this data because they integrate over
# band positions; template matching correlates pixel-by-pixel and cannot.
# The band between the two thresholds is "expect a poor result", below the
# lower one it is "the map will be meaningless".
DICT_IQ_GOOD = 0.25
DICT_IQ_MINIMUM = 0.15


def assess_for_template_matching(signal, n_sample: int = 64) -> dict:
    """Can dictionary indexing work on these patterns?

    Samples up to ``n_sample`` patterns spread over the scan, removes the
    static (sample mean) and dynamic background — the same preparation
    dictionary indexing needs — and reports the median FFT image quality.

    Returns a dict with ``image_quality``, ``verdict`` ("good" | "marginal" |
    "too_noisy" | "unknown") and a human-readable ``detail``. Never raises:
    a failed assessment must not block indexing, only inform it.
    """
    import kikuchipy as kp

    out = {"image_quality": None, "verdict": "unknown", "detail": ""}
    try:
        data = signal.data
        if getattr(data, "ndim", 0) != 4:
            return out
        n_rows, n_cols = int(data.shape[0]), int(data.shape[1])
        step_r = max(n_rows // 8, 1)
        step_c = max(n_cols // 8, 1)
        pats = []
        for r in range(0, n_rows, step_r):
            for c in range(0, n_cols, step_c):
                pats.append(np.asarray(data[r, c], dtype=np.float32))
                if len(pats) >= n_sample:
                    break
            if len(pats) >= n_sample:
                break
        if len(pats) < 4:
            return out

        stack = np.stack(pats)
        # Static background is not stored for EDAX files, so subtract the
        # sample mean — that is what static removal amounts to here.
        s = kp.signals.EBSD((stack - stack.mean(0)).astype(np.float32))
        s.remove_dynamic_background()
        iq = float(np.median(np.asarray(s.get_image_quality()).ravel()))
    except Exception:
        logger.warning("Could not assess pattern quality", exc_info=True)
        return out

    out["image_quality"] = round(iq, 4)
    if iq >= DICT_IQ_GOOD:
        out["verdict"] = "good"
        out["detail"] = f"pattern quality {iq:.3f} — fine for template matching"
    elif iq >= DICT_IQ_MINIMUM:
        out["verdict"] = "marginal"
        out["detail"] = (
            f"pattern quality {iq:.3f} is marginal for template matching "
            f"(good from {DICT_IQ_GOOD:.2f}). Expect low correlation scores; "
            "frame averaging or a longer exposure would help."
        )
    else:
        out["verdict"] = "too_noisy"
        out["detail"] = (
            f"pattern quality {iq:.3f} is far below what template matching "
            f"needs ({DICT_IQ_MINIMUM:.2f} minimum, {DICT_IQ_GOOD:.2f} for a "
            "good result). Dictionary indexing correlates whole patterns "
            "pixel-by-pixel and cannot recover bands from noise; it will "
            "return a near-uniform map with meaningless orientations. "
            "Hough and Spherical indexing integrate over band positions and "
            "still work on this data — use one of those, or acquire with a "
            "longer exposure / more frame averaging."
        )
    return out
