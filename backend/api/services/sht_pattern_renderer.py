"""HTTP-layer wrapper for SHT-based forward pattern simulation.

Singleton ``PatternRenderer`` + module-level LRU phase cache so that
successive ``/api/indexing/pattern-match`` requests reuse loaded SHT grids.

Fail-loud policy (per spec section 7): missing SHT files, OOM during render,
or invalid metadata raise :class:`SHTRenderError` rather than silently
substituting placeholders. The API layer converts these into explicit
error messages in the JSON response.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from backend.spherical_gpu.exceptions import SHTReadError
from backend.spherical_gpu.pipeline.forward import LambertGrid, PatternRenderer

logger = logging.getLogger(__name__)


class SHTRenderError(RuntimeError):
    """Raised when the service cannot produce a simulated pattern."""


# ---------------------------------------------------------------------------
# Singleton + cache
# ---------------------------------------------------------------------------

_RENDERER: Optional[PatternRenderer] = None
_PHASE_CACHE: "OrderedDict[str, LambertGrid]" = OrderedDict()
_CACHE_CAPACITY: int = 8


def _device_from_env() -> torch.device:
    """Honor SHT_FORWARD_DEVICE=cpu|cuda override; default = cuda if available.

    Per spec section 7, this is an *explicit* opt-in; we never silently fall
    back from GPU to CPU on errors.
    """
    forced = os.environ.get("SHT_FORWARD_DEVICE", "").strip().lower()
    if forced == "cpu":
        return torch.device("cpu")
    if forced == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_renderer() -> PatternRenderer:
    """Return the process-singleton PatternRenderer (lazily constructed)."""
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = PatternRenderer(device=_device_from_env())
        logger.info("SHT PatternRenderer singleton on %s", _RENDERER.device)
    return _RENDERER


def load_or_get_phase(
    sht_path: str,
    max_bandwidth: Optional[int] = None,
) -> LambertGrid:
    """Return a cached LambertGrid for ``sht_path`` (resolved to absolute).

    The cache key includes ``max_bandwidth`` because the same SHT loaded
    at different truncation bandwidths produces different grids and is
    a different cache entry.

    Raises
    ------
    SHTRenderError
        If the file does not exist on disk.
    """
    path_key = str(Path(sht_path).resolve())
    bw_key = -1 if max_bandwidth is None else int(max_bandwidth)
    key = (path_key, bw_key)
    if key in _PHASE_CACHE:
        _PHASE_CACHE.move_to_end(key)
        return _PHASE_CACHE[key]
    if not Path(path_key).is_file():
        raise SHTRenderError(f"SHT file not found: {path_key}")
    try:
        grid = get_renderer().load_phase(path_key, max_bandwidth=max_bandwidth)
    except SHTReadError as e:
        raise SHTRenderError(f"failed to read {path_key}: {e}") from e
    _PHASE_CACHE[key] = grid
    while len(_PHASE_CACHE) > _CACHE_CAPACITY:
        _PHASE_CACHE.popitem(last=False)
    return grid


def release_gpu_caches() -> int:
    """Drop the cached per-phase ``LambertGrid`` objects and the ``PatternRenderer``
    singleton so their GPU tensors are released by the next ``empty_cache()``.

    These caches (a high-bandwidth grid is multiple GB) are NOT touched by a bare
    ``torch.cuda.empty_cache()`` because they hold live references, so without this
    the "Release GPU" button reports "0 MiB freed". Call this BEFORE empty_cache().
    Returns the number of cached phases dropped. Safe to call on a CPU-only box.
    """
    global _RENDERER
    n = len(_PHASE_CACHE)
    _PHASE_CACHE.clear()
    _RENDERER = None
    return n


def compute_forward_ncc_map(
    result,
    max_bandwidth: Optional[int] = 256,
    chunk_size: int = 128,
    progress_callback=None,
) -> np.ndarray:
    """Compute the forward-NCC quality map for a spherical indexing result.

    For every indexed pixel:
      1. Render the SHT-forward simulated pattern at the indexed
         orientation and PC.
      2. Compute the normalized cross-correlation (NCC) against the
         experimental pattern.

    The result is stored on ``result.metadata["forward_ncc_map"]`` and
    also returned.

    Parameters
    ----------
    result : IndexingResult
        Must be a spherical-indexing result with
        ``metadata["sht_paths_by_phase"]`` populated.
    max_bandwidth : Optional[int]
        Bandwidth truncation for the SHT inverse (default 256 -- "Standard"
        sharpness). bw=128 is faster, bw=384 is sharpest.
    chunk_size : int
        Patterns per GPU launch.
    progress_callback : callable, optional
        Called as ``progress_callback(frac)`` with frac in [0, 1].

    Returns
    -------
    np.ndarray
        Shape ``(n_rows, n_cols)``, dtype float32. Non-indexed pixels
        are NaN. Indexed pixels carry the forward NCC value in [-1, 1].

    Raises
    ------
    SHTRenderError
        If the result is not spherical, or sht_paths_by_phase is missing,
        or any phase SHT file is unreadable.
    """
    if result is None:
        raise SHTRenderError("no indexing result given")
    metadata = result.metadata or {}
    if metadata.get("indexing_method") != "spherical":
        raise SHTRenderError(
            f"forward-NCC map is only available for spherical results "
            f"(got {metadata.get('indexing_method')!r})"
        )
    sht_map = metadata.get("sht_paths_by_phase") or {}
    if not sht_map:
        raise SHTRenderError(
            "result.metadata['sht_paths_by_phase'] is empty -- "
            "re-run indexing to populate it."
        )
    det = metadata.get("detector_geometry")
    if not det:
        raise SHTRenderError(
            "result.metadata['detector_geometry'] is missing -- "
            "re-run indexing to populate it."
        )

    n_rows, n_cols = result.original_shape
    pat_h = int(det["pat_height"])
    pat_w = int(det["pat_width"])
    pixel_size_um = float(det.get("pixel_size", 70.0))
    sample_tilt = float(det.get("sample_tilt", 70.0))

    # Circular-detector mask for EDAX/TSL (dark corners). See
    # forward_diagnostics.build_circular_include_mask for the rationale —
    # without it the corner mismatch against the full-square SHT render
    # collapses the NCC. Oxford/Bruker full-frame detectors stay unmasked
    # (bit-identical to before).
    from backend.api.services.forward_diagnostics import (
        build_circular_include_mask,
    )
    include_mask = None
    # Consult both 'vendor' (hardcoded 'Bruker' on the spherical path) and
    # 'source_vendor' (the real detected vendor) — see indexing.py:603/657.
    _vendor_u = (str(det.get("vendor", "")) + " "
                 + str(det.get("source_vendor", ""))).upper()
    if (bool(det.get("circular_mask", False))
            or det.get("mask_radius_fraction") is not None
            or "EDAX" in _vendor_u or "TSL" in _vendor_u):
        _rf = float(det.get("mask_radius_fraction", 1.0))
        include_mask = build_circular_include_mask(pat_h, pat_w, _rf)
        if not include_mask.any():
            raise SHTRenderError(
                f"circular mask excludes all pixels (mask_radius_fraction={_rf})"
            )

    from backend.spherical_gpu.pipeline.detector import convert_pc_to_emsoft
    xpc, ypc, L_um = convert_pc_to_emsoft(
        pc=(float(det["pc_x"]), float(det["pc_y"]), float(det["pc_z"])),
        vendor=str(det.get("vendor", "Bruker")),
        pat_width=pat_w, pat_height=pat_h,
        pixel_size=pixel_size_um, binning=int(det.get("binning", 1)),
    )

    # Output map seeded with NaN
    ncc_map = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    # Build the full per-pixel index ↔ flat-idx mapping once.
    xmap = result.xmap
    n_xmap = int(xmap.rotations.size)
    n_total = n_rows * n_cols
    flat_mask = result.selection_mask.ravel()
    if n_xmap == n_total:
        # xmap covers full grid; px_idx is just the flat idx.
        def px_idx_of_flat(flat: int) -> int:
            return flat
        valid_flat = np.arange(n_total)
    else:
        # xmap covers only selected pixels; need cumulative-sum mapping.
        cumsum = np.cumsum(flat_mask) - 1  # px_idx for each True
        def px_idx_of_flat(flat: int) -> int:
            return int(cumsum[flat])
        valid_flat = np.where(flat_mask)[0]

    # Group pixels by phase id
    from orix.quaternion import Rotation
    phase_ids = np.asarray(xmap.phase_id).ravel()
    eulers_all = np.asarray(xmap.rotations.to_euler())              # (n_xmap, 3)

    # For each phase, render its pixels' patterns and compute NCC
    from tools.pattern_comparison import get_experimental_pattern  # signal accessor
    total_indexed = int(len(valid_flat))
    done = 0

    for phase_id_int, sht_path in sht_map.items():
        phase_id = int(phase_id_int)
        sht_resolved = str(Path(sht_path).resolve())
        if not Path(sht_resolved).is_file():
            logger.warning("SHT not found for phase %s: %s -- skipping", phase_id, sht_resolved)
            continue

        # Pixels in this phase, within indexed region
        phase_pixels_flat = []
        phase_orient_idx = []
        for flat in valid_flat:
            try:
                pid_at_pixel = int(phase_ids[px_idx_of_flat(int(flat))])
            except Exception:
                continue
            if pid_at_pixel != phase_id:
                continue
            phase_pixels_flat.append(int(flat))
            phase_orient_idx.append(px_idx_of_flat(int(flat)))
        if not phase_pixels_flat:
            continue
        phase_pixels_flat = np.asarray(phase_pixels_flat, dtype=np.int64)
        phase_orient_idx = np.asarray(phase_orient_idx, dtype=np.int64)

        # Load grid
        grid = load_or_get_phase(sht_resolved, max_bandwidth=max_bandwidth)

        # Orientation quaternions for these pixels
        phase_eulers = eulers_all[phase_orient_idx]                   # (M, 3)
        phase_quats_np = Rotation.from_euler(phase_eulers).data       # (M, 4) numpy
        phase_quats = torch.tensor(np.asarray(phase_quats_np)[:, :4], dtype=torch.float64)

        # Process in chunks to bound CPU/GPU memory
        M = phase_quats.shape[0]
        for chunk_start in range(0, M, chunk_size):
            chunk_end = min(chunk_start + chunk_size, M)
            chunk_quats = phase_quats[chunk_start:chunk_end]          # (B, 4)
            chunk_flat  = phase_pixels_flat[chunk_start:chunk_end]    # (B,)

            sims = get_renderer().render_batch(
                grid=grid,
                orientation_quats=chunk_quats,
                pc_emsoft=(float(xpc), float(ypc), float(L_um)),
                detector_shape=(pat_h, pat_w),
                pixel_size_um=pixel_size_um,
                tilt_deg=sample_tilt,
                chunk_size=chunk_size,
            ).numpy()                                                 # (B, H, W)

            # Read experimental patterns for these pixels
            sims_flat = sims.reshape(sims.shape[0], -1).astype(np.float64)
            if include_mask is not None:
                sims_flat = sims_flat[:, include_mask]   # inside-circle only
            sims_flat -= sims_flat.mean(axis=1, keepdims=True)
            sims_norm = np.linalg.norm(sims_flat, axis=1, keepdims=True)
            sims_norm[sims_norm < 1e-12] = 1.0
            sims_flat /= sims_norm                                    # (B, masked) unit-norm

            for local_i, flat_idx in enumerate(chunk_flat):
                r = int(flat_idx // n_cols)
                c = int(flat_idx % n_cols)
                exp = get_experimental_pattern(result, r, c)
                if exp is None:
                    continue
                exp_v = np.asarray(exp, dtype=np.float64).ravel()
                if include_mask is not None:
                    exp_v = exp_v[include_mask]
                exp_v -= exp_v.mean()
                exp_n = np.linalg.norm(exp_v)
                if exp_n < 1e-12:
                    continue
                exp_v /= exp_n
                ncc_map[r, c] = float(np.dot(exp_v, sims_flat[local_i]))

            done += (chunk_end - chunk_start)
            if progress_callback is not None and total_indexed > 0:
                try:
                    progress_callback(min(1.0, done / total_indexed))
                except Exception:
                    pass

    # Cache on the result so downstream endpoints can reuse without recompute
    if metadata is not None:
        metadata["forward_ncc_map"] = ncc_map
        metadata["forward_ncc_bandwidth"] = int(max_bandwidth) if max_bandwidth else None

    return ncc_map


def render_pattern_to_png_b64(
    sht_path: str,
    orientation_quat: torch.Tensor,
    pc_emsoft: Tuple[float, float, float],
    detector_shape: Tuple[int, int],
    pixel_size_um: float,
    tilt_deg: float = 70.0,
    det_tilt_deg: float = 0.0,
    max_bandwidth: Optional[int] = None,
) -> str:
    """Render one pattern and return base64-encoded PNG bytes.

    Parameters
    ----------
    tilt_deg : float
        SAMPLE tilt angle in degrees (typically 70 for EBSD).
    det_tilt_deg : float
        DETECTOR tilt angle in degrees (typically 0 for Oxford, often
        non-zero for EDAX — LoGainNi.h5 has det_tilt=10). The Tier1
        indexer uses ``geom.tilt_deg`` here; the renderer MUST receive
        the same value or the simulated pattern is rotated by
        ``sample_tilt - 0 - (sample_tilt - det_tilt) = det_tilt`` away
        from the experimental, producing R = 0 visual mismatch even
        when the orientation is correct. Bug history 2026-05-22:
        previously this defaulted to 0 with no caller able to override,
        which broke /api/indexing/pattern-match and /api/pc/render-preview
        on every EDAX dataset with non-zero detector tilt.
    max_bandwidth : Optional[int]
        Override the default truncation bandwidth (128). 256 gives
        noticeably sharper patterns at ~1.7 GB VRAM; 384 the sharpest
        at ~5.7 GB.

    Raises
    ------
    SHTRenderError
        Wraps both file-not-found errors and underlying CUDA/Torch failures
        so the API layer can present a single clean error path.
    """
    grid = load_or_get_phase(sht_path, max_bandwidth=max_bandwidth)
    try:
        pat = get_renderer().render(
            grid=grid,
            orientation_quat=orientation_quat,
            pc_emsoft=pc_emsoft,
            detector_shape=detector_shape,
            pixel_size_um=pixel_size_um,
            tilt_deg=tilt_deg,
            det_tilt_deg=det_tilt_deg,
        )
    except torch.cuda.OutOfMemoryError as e:                # type: ignore[attr-defined]
        raise SHTRenderError(
            "GPU out of memory during SHT pattern render. "
            "Set SHT_FORWARD_DEVICE=cpu and restart backend, or reduce "
            "detector size."
        ) from e
    except RuntimeError as e:
        # PyTorch raises plain RuntimeError for many low-level issues;
        # surface them loudly instead of swallowing.
        raise SHTRenderError(f"render failed: {e}") from e

    arr = pat.numpy().astype(np.float32)
    rng = float(np.ptp(arr))
    if rng < 1e-12:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        norm = ((arr - arr.min()) / rng * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(norm, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
