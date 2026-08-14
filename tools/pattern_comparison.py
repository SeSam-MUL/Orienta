"""
Pattern comparison utilities for dictionary indexing quality assessment.

Provides functions to:
- Extract per-pixel NCC score maps for full-scan heatmaps
- Retrieve the best-matching simulated pattern for any pixel
- Compute element-wise NCC image (3rd panel in comparison view)
- Compute scalar NCC/R score

No PyQt5 / GUI dependencies — pure numpy + orix.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score map
# ---------------------------------------------------------------------------

def get_ncc_score_map(result) -> Optional[np.ndarray]:
    """Return 2D NCC score map (n_rows, n_cols) from an IndexingResult.

    Works for both Dictionary and Hough indexing results.

    Returns
    -------
    np.ndarray or None
        Float array in [0, 1] or None if no scores available.
    """
    if result is None:
        return None

    n_rows, n_cols = result.original_shape
    scores = result.confidence_scores

    if scores is None:
        return None

    scores_flat = np.asarray(scores).ravel()

    # If scores cover only selected pixels, embed them into the full grid
    n_selected = int(result.selection_mask.sum())
    n_total = n_rows * n_cols

    if scores_flat.size == n_selected and n_selected < n_total:
        full = np.full(n_total, np.nan)
        flat_mask = result.selection_mask.ravel()
        full[flat_mask] = scores_flat
        return full.reshape(n_rows, n_cols)

    if scores_flat.size == n_total:
        return scores_flat.reshape(n_rows, n_cols)

    # Fallback: try to reshape directly
    try:
        return scores_flat[:n_total].reshape(n_rows, n_cols)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lazy single-pattern fetch from dictionary file
# ---------------------------------------------------------------------------

def _load_pattern_from_dict_file(
    result, dict_path: str, row: int, col: int, rank: int,
) -> Optional[np.ndarray]:
    """Read a single dictionary pattern from its source HDF5 file.

    Used when the full dictionary tensor was released from memory after
    indexing (GPU path) but we still need the best-match pattern for the
    pattern-match dialog. The dataset supports random row access, so we
    only pay for one row's worth of I/O (~80 KB for 128×156 float32).

    Returns None if the file is unreachable, the index is missing, or the
    simulation_indices array doesn't carry a valid index for this pixel.
    """
    from pathlib import Path
    if not dict_path or not Path(dict_path).is_file():
        logger.debug(f"dict_path not usable: {dict_path!r}")
        return None

    # Locate the xmap index for this pixel (same translation as Strategy 1)
    n_rows, n_cols = result.original_shape
    flat_idx = row * n_cols + col
    flat_mask = result.selection_mask.ravel()
    if not flat_mask[flat_idx]:
        return None
    xmap = result.xmap
    n_total = n_rows * n_cols
    n_xmap = xmap.rotations.size
    xmap_idx = flat_idx if n_xmap == n_total else int(flat_mask[:flat_idx].sum())

    if not hasattr(xmap, "prop") or "simulation_indices" not in xmap.prop:
        logger.debug("No simulation_indices on xmap — cannot identify best match")
        return None
    sim_indices = xmap.prop["simulation_indices"]
    try:
        if sim_indices.ndim == 2:
            r = min(rank, sim_indices.shape[1] - 1)
            best_idx = int(sim_indices[xmap_idx, r])
        else:
            best_idx = int(sim_indices[xmap_idx])
    except (IndexError, TypeError) as e:
        logger.warning(f"simulation_indices lookup failed for ({row},{col}): {e}")
        return None
    if best_idx < 0:
        return None

    # The dictionary file is a kikuchipy-written EBSD signal HDF5 — the
    # patterns live at /Scan 1/EBSD/Data/patterns (shape: n_dict, h, w).
    try:
        import h5py
        with h5py.File(dict_path, "r") as f:
            # Walk the scan groups to find the patterns dataset rather than
            # hardcoding "Scan 1" — Oxford-style files use numeric scan ids.
            patterns_ds = None
            for top_key in f.keys():
                grp = f[top_key]
                if hasattr(grp, "keys") and "EBSD" in grp:
                    ds = grp.get("EBSD/Data/patterns")
                    if ds is not None:
                        patterns_ds = ds
                        break
            # Legacy raw dict-cache shape: /patterns at top level
            if patterns_ds is None:
                for k in ("patterns", "dictionary", "data"):
                    if k in f:
                        patterns_ds = f[k]
                        break
            if patterns_ds is None:
                logger.warning(f"No patterns dataset found in {dict_path}")
                return None
            if not (0 <= best_idx < patterns_ds.shape[0]):
                logger.warning(
                    f"simulation_indices[{xmap_idx}]={best_idx} out of range "
                    f"(dict has {patterns_ds.shape[0]} patterns)"
                )
                return None
            return np.asarray(patterns_ds[best_idx], dtype=np.float32)
    except Exception as e:
        logger.warning(f"Failed to read pattern {best_idx} from {dict_path}: {e}")
        return None


def _read_pattern_row(dict_path: str, index: int) -> Optional[np.ndarray]:
    """Read one pattern row out of a dictionary HDF5 file."""
    from pathlib import Path
    if not dict_path or not Path(dict_path).is_file():
        logger.debug(f"dict_path not usable: {dict_path!r}")
        return None
    try:
        import h5py
        with h5py.File(dict_path, "r") as f:
            patterns_ds = None
            for top_key in f.keys():
                grp = f[top_key]
                if hasattr(grp, "keys") and "EBSD" in grp:
                    ds = grp.get("EBSD/Data/patterns")
                    if ds is not None:
                        patterns_ds = ds
                        break
            if patterns_ds is None:
                for k in ("patterns", "dictionary", "data"):
                    if k in f:
                        patterns_ds = f[k]
                        break
            if patterns_ds is None:
                logger.warning(f"No patterns dataset found in {dict_path}")
                return None
            if not (0 <= index < patterns_ds.shape[0]):
                logger.warning(
                    f"pattern index {index} out of range "
                    f"(dict has {patterns_ds.shape[0]} patterns)"
                )
                return None
            return np.asarray(patterns_ds[index], dtype=np.float32)
    except Exception as e:
        logger.warning(f"Failed to read pattern {index} from {dict_path}: {e}")
        return None


def _load_pattern_multi_phase(result, sources: dict, row: int, col: int, rank: int):
    """Best-match pattern for a merged multi-phase Dictionary result.

    Each phase was indexed against its own dictionary file, so the merged
    result cannot carry a single ``dict_path``. ``per_phase_match_sources``
    holds ``{phase_name: {dict_path, simulation_indices}}`` with the indices
    already lifted onto the full navigation grid; the winner at this pixel
    comes from the merged xmap's ``phase_id``.
    """
    n_rows, n_cols = result.original_shape
    flat_idx = row * n_cols + col

    mask = getattr(result, "selection_mask", None)
    if mask is not None:
        flat_mask = np.asarray(mask).ravel()
        if flat_idx < flat_mask.size and not flat_mask[flat_idx]:
            return None

    xmap = result.xmap
    try:
        phase_ids = np.asarray(xmap.phase_id).ravel()
        pid = int(phase_ids[flat_idx])
    except Exception as e:
        logger.debug(f"multi-phase: no phase_id for ({row},{col}): {e}")
        return None
    if pid < 0:
        return None  # pixel not assigned to any phase

    try:
        phase_name = str(xmap.phases[pid].name)
    except Exception as e:
        logger.debug(f"multi-phase: phase {pid} not in phase list: {e}")
        return None

    src = sources.get(phase_name)
    if not src:
        logger.debug(
            f"multi-phase: no match source recorded for phase {phase_name!r} "
            f"(have: {sorted(sources)})"
        )
        return None

    sim = np.asarray(src.get("simulation_indices"))
    if sim.ndim != 2 or flat_idx >= sim.shape[0]:
        return None
    r = min(rank, sim.shape[1] - 1)
    best_idx = int(sim[flat_idx, r])
    if best_idx < 0:
        return None

    return _read_pattern_row(src.get("dict_path"), best_idx)


# ---------------------------------------------------------------------------
# Best-match simulated pattern
# ---------------------------------------------------------------------------

def get_best_match_pattern(
    result,
    row: int,
    col: int,
    rank: int = 0,
) -> Optional[np.ndarray]:
    """Return the best-matching simulated pattern for pixel (row, col).

    Strategy (in order of preference):
      1. Use ``xmap.prop['simulation_indices']`` — direct dictionary index
         stored by kikuchipy's ``dictionary_indexing()``.  Fast and correct.
      2. Fallback: misorientation search against dictionary rotations.

    Parameters
    ----------
    result : IndexingResult
        Must contain metadata['dictionary'] and metadata['signal'].
    row, col : int
        Pixel coordinates in the original scan.
    rank : int
        0 = best match, 1 = 2nd best, etc. (requires keep_n > 1 during indexing)

    Returns
    -------
    np.ndarray or None
        2D simulated pattern, or None if not available.
    """
    if result is None:
        return None

    dictionary = result.metadata.get('dictionary')
    dict_path = result.metadata.get('dict_path')

    # Strategy 0a: multi-phase run. The merged result has neither the
    # dictionary nor a single dict_path — each phase was indexed against its
    # own file. Resolve which phase won this pixel and read from that file.
    if dictionary is None and not dict_path:
        sources = result.metadata.get('per_phase_match_sources')
        if sources:
            return _load_pattern_multi_phase(result, sources, row, col, rank)

    # Strategy 0: lazy single-pattern fetch from disk.
    # GPU-path dict indexing releases the dictionary tensor from VRAM after
    # the run (it's multiple GB), so result.metadata['dictionary'] is None.
    # But we stored the source file path under 'dict_path' and the matched
    # indices under xmap.prop['simulation_indices'] — together those let
    # us read just the one pattern the dialog needs, no full reload.
    if dictionary is None and dict_path:
        return _load_pattern_from_dict_file(result, dict_path, row, col, rank)

    if dictionary is None:
        logger.debug(
            "No dictionary in result.metadata and no dict_path — "
            "cannot retrieve simulated pattern"
        )
        return None

    # Guard against EBSDMasterPattern being stored instead of a sim-pattern EBSD
    # signal. A master pattern has .data of shape (2, N_phi, 2*N_phi) — indexing
    # into that returns a Lambert-projection slice that *looks* like a pattern
    # but matches the experimental one poorly (R ~= 0), producing the infamous
    # "wrong best match" dialog. Detect and bail out cleanly so the UI can show
    # "no match available" instead of garbage.
    dict_type_name = type(dictionary).__name__
    if 'MasterPattern' in dict_type_name:
        logger.warning(
            f"Dictionary is a {dict_type_name}, not an EBSD signal of pre-generated "
            f"patterns — cannot retrieve per-pixel simulated pattern. "
            f"Generate a dictionary first (kikuchipy.signals.EBSDMasterPattern.get_patterns)."
        )
        return None

    n_rows, n_cols = result.original_shape
    n_total = n_rows * n_cols
    flat_idx = row * n_cols + col

    # Check pixel was indexed
    flat_mask = result.selection_mask.ravel()
    if not flat_mask[flat_idx]:
        logger.debug(f"Pixel ({row}, {col}) was not indexed (outside selection mask)")
        return None

    xmap = result.xmap

    # Determine the correct index into xmap arrays.
    # If the xmap covers the full grid (n_total entries), flat_idx works directly.
    # If the xmap is partial (only indexed pixels), we need to translate
    # flat_idx → the subset index (how many True values precede this pixel).
    n_xmap = xmap.rotations.size
    if n_xmap == n_total:
        xmap_idx = flat_idx
    else:
        xmap_idx = int(flat_mask[:flat_idx].sum())

    # --- Strategy 1: simulation_indices (preferred) -------------------------
    # kikuchipy stores the dictionary index of the best match directly.
    if hasattr(xmap, 'prop') and 'simulation_indices' in xmap.prop:
        try:
            sim_indices = xmap.prop['simulation_indices']
            if sim_indices.ndim == 2:
                # Shape (n_pixels, keep_n) — pick requested rank
                r = min(rank, sim_indices.shape[1] - 1)
                best_dict_idx = int(sim_indices[xmap_idx, r])
            else:
                best_dict_idx = int(sim_indices[xmap_idx])
            if 0 <= best_dict_idx < len(dictionary.data):
                logger.debug(
                    f"Pixel ({row}, {col}) rank={rank}: simulation_indices -> dict[{best_dict_idx}]"
                )
                return np.asarray(dictionary.data[best_dict_idx], dtype=np.float32)
            else:
                logger.warning(
                    f"simulation_indices[{xmap_idx}, {rank}] = {best_dict_idx} "
                    f"out of range (dict size {len(dictionary.data)})"
                )
        except (IndexError, KeyError, TypeError) as e:
            logger.warning(f"simulation_indices lookup failed for ({row}, {col}): {e}")

    # --- Strategy 2: misorientation search (fallback) -----------------------
    try:
        matched_rotation = xmap.rotations[xmap_idx]
    except (IndexError, AttributeError) as e:
        logger.warning(f"Could not get matched rotation for pixel ({row}, {col}): {e}")
        return None

    best_idx = _find_closest_dictionary_index(matched_rotation, dictionary)
    if best_idx is None:
        return None

    pat = np.asarray(dictionary.data[best_idx], dtype=np.float32)
    return pat


def _find_closest_dictionary_index(matched_rotation, dictionary) -> Optional[int]:
    """Find dictionary index whose orientation is closest to matched_rotation."""
    try:
        from orix.quaternion import Rotation

        # Try to get dictionary rotations from its xmap attribute
        dict_rotations = None
        if hasattr(dictionary, 'xmap') and dictionary.xmap is not None:
            dict_rotations = dictionary.xmap.rotations
        elif hasattr(dictionary, 'orientations'):
            dict_rotations = dictionary.orientations

        if dict_rotations is None:
            # No rotation info — fall back to index 0
            logger.debug("Dictionary has no xmap.rotations — returning pattern at index 0")
            return 0

        # Compute misorientation angles (in radians) between matched and all dict entries
        angles = matched_rotation.angle_with(dict_rotations)
        best_idx = int(np.argmin(np.asarray(angles)))
        return best_idx

    except Exception as e:
        logger.warning(f"Could not find closest dictionary orientation: {e}")
        return 0


# ---------------------------------------------------------------------------
# NCC image and scalar
# ---------------------------------------------------------------------------

def compute_ncc_image(
    experimental: np.ndarray,
    simulated: np.ndarray,
) -> np.ndarray:
    """Element-wise NCC correlation image.

    Formula: ((E - μ_E) / σ_E) * ((S - μ_S) / σ_S)

    Each pixel value shows local agreement between the two patterns.
    Range is approximately [-1, 1]; positive = agreement, negative = anti-correlation.

    Parameters
    ----------
    experimental, simulated : np.ndarray
        2D pattern arrays (same shape).

    Returns
    -------
    np.ndarray
        2D correlation image, same shape as inputs.
    """
    e = np.asarray(experimental, dtype=np.float64)
    s = np.asarray(simulated, dtype=np.float64)

    # Pattern shapes must match. If they don't (e.g. simulated dictionary
    # was generated for a different detector size than the loaded EBSD),
    # broadcasting in `e_norm * s_norm` would either silently produce a
    # garbage result or raise a confusing ValueError. Return zeros + log so
    # the Pattern Match dialog renders a clean "no correlation" state.
    if e.shape != s.shape:
        logger.warning(
            "compute_ncc_image: shape mismatch experimental=%s simulated=%s — returning zeros",
            e.shape, s.shape,
        )
        return np.zeros_like(e, dtype=np.float32)

    e_std = e.std()
    s_std = s.std()

    if e_std < 1e-10 or s_std < 1e-10:
        return np.zeros_like(e)

    e_norm = (e - e.mean()) / e_std
    s_norm = (s - s.mean()) / s_std

    return (e_norm * s_norm).astype(np.float32)


def compute_ncc_scalar(
    experimental: np.ndarray,
    simulated: np.ndarray,
) -> float:
    """Compute scalar NCC (R) value between two patterns.

    R = mean of the element-wise NCC image, equivalent to the
    normalized cross-correlation at zero lag.

    Returns
    -------
    float
        R value in [-1, 1]. Typical good values: 0.2–0.8.
    """
    ncc_img = compute_ncc_image(experimental, simulated)
    return float(np.mean(ncc_img))


# ---------------------------------------------------------------------------
# Circular detector aperture (EDAX-style)
# ---------------------------------------------------------------------------

def detect_circular_aperture(experimental: np.ndarray) -> bool:
    """Heuristically decide whether ``experimental`` was recorded through a
    circular detector aperture (EDAX-style: a circle inscribed in a square
    frame, black corners outside the circle).

    Detection: if all four 8x8 corner blocks are uniformly near-zero — mean
    of each block below 5 % of the pattern's max — the corners lie outside a
    physical aperture, so the detector is circular. Full-frame detectors
    (Bruker / Oxford) carry real signal in the corners and return False.

    Parameters
    ----------
    experimental : np.ndarray
        2D experimental pattern.

    Returns
    -------
    bool
        True if a circular aperture is present.
    """
    e = np.asarray(experimental, dtype=np.float64)
    if e.ndim != 2:
        return False
    h, w = e.shape
    if h < 16 or w < 16:
        return False
    e_max = float(e.max())
    if e_max <= 0:
        return False
    threshold = 0.05 * e_max
    blk = 8
    corners = [
        e[:blk, :blk], e[:blk, -blk:],
        e[-blk:, :blk], e[-blk:, -blk:],
    ]
    return all(float(c.mean()) < threshold for c in corners)


def circular_mask(shape: Tuple[int, int], radius_frac: float = 1.0) -> np.ndarray:
    """Boolean mask for a centred circle of a ``(H, W)`` frame.

    Center is ``((H-1)/2, (W-1)/2)``. The radius is ``radius_frac`` times the
    largest inscribed-circle radius ``min(H, W) / 2``. Pixels inside (or on)
    the circle are True.

    Parameters
    ----------
    shape : (H, W)
    radius_frac : float
        Fraction of the inscribed-circle radius, clamped to ``[0.1, 1.0]``.
        ``1.0`` (default) = the full inscribed circle. Smaller values shrink
        the aperture so a too-tight phosphor edge can be excluded too.
    """
    h, w = int(shape[0]), int(shape[1])
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    frac = float(min(max(radius_frac, 0.1), 1.0))
    radius = (min(h, w) / 2.0) * frac
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    return ((rows - cy) ** 2 + (cols - cx) ** 2) <= radius ** 2


def apply_circular_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``arr`` with everything outside ``mask`` set to 0.

    Works for 2D grayscale and 3D RGB(A) arrays (mask broadcast over the
    last axis).
    """
    out = np.array(arr, copy=True)
    if out.ndim == 2:
        out[~mask] = 0
    elif out.ndim == 3:
        out[~mask, ...] = 0
    return out


def compute_ncc_scalar_masked(
    experimental: np.ndarray,
    simulated: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Scalar NCC computed only over pixels where ``mask`` is True.

    Unlike :func:`compute_ncc_scalar`, the (matching black) corner regions
    outside a circular aperture are excluded so they neither inflate nor
    deflate the correlation.
    """
    e = np.asarray(experimental, dtype=np.float64)
    s = np.asarray(simulated, dtype=np.float64)
    if e.shape != s.shape or e.shape != mask.shape:
        logger.warning(
            "compute_ncc_scalar_masked: shape mismatch e=%s s=%s mask=%s",
            e.shape, s.shape, mask.shape,
        )
        return 0.0
    ev = e[mask]
    sv = s[mask]
    if ev.size == 0:
        return 0.0
    e_std = ev.std()
    s_std = sv.std()
    if e_std < 1e-10 or s_std < 1e-10:
        return 0.0
    e_norm = (ev - ev.mean()) / e_std
    s_norm = (sv - sv.mean()) / s_std
    return float(np.mean(e_norm * s_norm))


# ---------------------------------------------------------------------------
# Experimental pattern retrieval
# ---------------------------------------------------------------------------

# Small cache of source-file signals, keyed by absolute path. Lets the Pattern
# Match dialog read a result's OWN experimental patterns even when a DIFFERENT
# file is currently active (e.g. two results from two files in the gallery),
# without re-opening the (possibly multi-GB) file on every pixel click. Lazy
# signals, so the open handle + a few chunks are the only cost. Capped small.
_source_signal_cache: "dict[str, object]" = {}
_SOURCE_CACHE_MAX = 3


def _get_source_signal(source_file):
    """Lazily load (and cache) the EBSD signal for a result's source file."""
    import os
    if not source_file:
        return None
    key = os.path.abspath(str(source_file))
    cached = _source_signal_cache.get(key)
    if cached is not None:
        return cached
    if not os.path.isfile(key):
        logger.warning("Result source file not found on disk: %s", key)
        return None
    try:
        from safe_loader import load_ebsd_safe
        sig = load_ebsd_safe(key, verbose=False)
    except Exception:
        logger.warning("Could not load result source file %s for experimental "
                       "pattern", key, exc_info=True)
        return None
    # Bounded FIFO eviction — this dialog only ever compares a handful of files.
    if len(_source_signal_cache) >= _SOURCE_CACHE_MAX:
        _source_signal_cache.pop(next(iter(_source_signal_cache)))
    _source_signal_cache[key] = sig
    return sig


def get_experimental_pattern(result, row: int, col: int) -> Optional[np.ndarray]:
    """Return experimental pattern at (row, col) from the indexed signal.

    Resolution order:
      1. ``result.metadata['signal']`` — Dictionary indexing stores the exact
         signal it matched against.
      2. The result's OWN ``source_file`` — when it differs from the currently
         active file, read the pattern from there (cached lazy load). This is
         the fix for the cross-file bug: with two results from two files in the
         gallery, the Pattern Match dialog used to show whichever file was
         *active* (wrong patterns + aperture-shape mismatch → unmasked/garbage
         match) instead of the result's real patterns.
      3. The currently-active EBSD signal — the correct source when the result
         WAS indexed on the loaded file (and the only option for older results
         with no ``source_file`` tag).

    Parameters
    ----------
    result : IndexingResult
    row, col : int

    Returns
    -------
    np.ndarray or None
        2D float32 pattern, or None if signal not available.
    """
    md = result.metadata if result.metadata else {}
    signal = md.get('signal')
    if signal is None:
        source_file = md.get('source_file')
        active_signal = active_path = None
        try:
            from backend.api.routes.ebsd_viewer import (
                _get_active_signal, _ebsd_file_path)
            active_signal = _get_active_signal()
            active_path = _ebsd_file_path
        except Exception:
            pass

        import os
        differs = bool(
            source_file and active_path
            and os.path.abspath(str(source_file)) != os.path.abspath(str(active_path))
        )
        if differs:
            # This result belongs to a file that is NOT the one on screen.
            # Read its patterns from its own source so we never show a
            # different file's patterns under this result. If the source can't
            # be loaded (deleted/moved/unreadable) we return None — NOT the
            # active signal: showing the wrong file's patterns is exactly the
            # bug this method fixes, and a silent wrong pattern is worse than an
            # empty panel. Fail loud so the reason is visible in the log.
            signal = _get_source_signal(source_file)
            if signal is None:
                logger.warning(
                    "Pattern Match: result's source file %r is not the active "
                    "file and could not be loaded — returning no experimental "
                    "pattern instead of the active file's (wrong) patterns.",
                    source_file)
                return None
        else:
            signal = active_signal
    if signal is None:
        return None
    try:
        pat = np.asarray(signal.data[row, col], dtype=np.float32)
        return pat
    except (IndexError, AttributeError) as e:
        logger.debug(f"signal.data[{row},{col}] failed ({e}), trying signal.inav[{col},{row}]")
    # Fallback: kikuchipy uses (col, row) navigation axes in some signal types
    try:
        pat = np.asarray(signal.inav[col, row].data, dtype=np.float32)
        return pat
    except Exception as e2:
        logger.warning(f"Could not get experimental pattern at ({row}, {col}): {e2}")
        return None


def ncc_diff_to_png_b64(experimental, simulated, mask=None) -> str:
    """Per-pixel NCC agreement image as a base64 PNG (blue→white→red).

    Uses :func:`compute_ncc_image` (range ~[-1, 1]), maps to a diverging
    colormap, and paints out-of-mask pixels neutral grey (128,128,128) so
    the circular-aperture corners neither display nor mislead. Returns a
    base64 PNG string (no data-URI prefix).
    """
    import base64, io
    from PIL import Image

    ncc = compute_ncc_image(experimental, simulated).astype(np.float64)
    t = np.clip((ncc + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] -> [0,1]
    # Diverging blue(0) -> white(0.5) -> red(1): vectorised, no matplotlib.
    r = np.where(t < 0.5, 2.0 * t, 1.0)
    b = np.where(t < 0.5, 1.0, 2.0 * (1.0 - t))
    g = np.where(t < 0.5, 2.0 * t, 2.0 * (1.0 - t))
    rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    if mask is not None and mask.shape == ncc.shape:
        rgb[~mask] = (128, 128, 128)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
