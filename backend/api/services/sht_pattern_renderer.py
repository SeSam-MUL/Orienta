"""HTTP-layer wrapper for SHT-based forward pattern simulation.

Singleton ``PatternRenderer`` + module-level LRU phase cache so that
successive ``/api/indexing/pattern-match`` requests reuse loaded SHT grids.

Fail-loud policy: missing SHT files, OOM during render,
or invalid metadata raise :class:`SHTRenderError` rather than silently
substituting placeholders. The API layer converts these into explicit
error messages in the JSON response.
"""
from __future__ import annotations

import base64
import errno
import io
import logging
import os
import traceback
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


class SHTOutOfMemoryError(SHTRenderError):
    """The grid did not fit: refused up front, or a real out-of-memory error.

    A subclass, so every existing ``except SHTRenderError`` keeps catching it;
    callers that want to tell "choose a lower quality" from "broken file" can
    catch this one first.
    """


#: Substrings of the messages the runtimes use for an allocation that failed.
#: "out of memory": CUDA (also inside a TorchScript trace). "not enough memory":
#: torch's CPU allocator as raised by torch 2.11 on Windows
#: ("DefaultCPUAllocator: not enough memory: you tried to allocate N bytes").
#: "can't allocate memory": the Linux wording of the same allocator. The cuBLAS
#: status is what a failed cuBLAS workspace allocation reports (also matched by
#: ``spherical_gpu.runtime.is_oom_error``).
_OOM_TEXT_MARKERS = (
    "out of memory",
    "not enough memory",
    "can't allocate memory",
    "cublas_status_alloc_failed",
)


#: Windows error codes for a failed allocation: ERROR_NOT_ENOUGH_MEMORY (8) and
#: ERROR_OUTOFMEMORY (14), as they arrive on an OSError.
_WIN_OOM_ERRORS = (8, 14)


def is_out_of_memory(exc: BaseException) -> bool:
    """True when ``exc`` (or the error it was explicitly raised from) is an
    out-of-memory error.

    A CUDA OOM arrives as ``torch.cuda.OutOfMemoryError``, but inside a
    TorchScript call it can surface as a plain ``RuntimeError`` whose text is
    the whole script trace -- so the text is checked too. CPU exhaustion is a
    ``MemoryError`` or a ``DefaultCPUAllocator`` error; a failed cuBLAS
    workspace is ``CUBLAS_STATUS_ALLOC_FAILED``; the operating system reports
    ``ENOMEM`` / Windows error 8 or 14 on an ``OSError``. Our own
    :class:`SHTOutOfMemoryError` counts by type: its text is a fixed sentence
    that does not repeat torch's wording.

    Only ``__cause__`` (``raise ... from``) is followed, not the implicit
    ``__context__``: an unrelated error raised while an OOM was being handled
    would otherwise be reported as one.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (MemoryError, SHTOutOfMemoryError)):
            return True
        oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
        if isinstance(exc, OSError) and (
            getattr(exc, "winerror", None) in _WIN_OOM_ERRORS or exc.errno == errno.ENOMEM
        ):
            return True
        # The wording is only trusted on a RuntimeError, which is what torch
        # raises: any other exception that merely MENTIONS "not enough memory"
        # (a file name, a message about something else) is not an OOM.
        if isinstance(exc, RuntimeError):
            text = str(exc).lower()
            if any(marker in text for marker in _OOM_TEXT_MARKERS):
                return True
        exc = exc.__cause__
    return False


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


#: VRAM a render grid needs per bandwidth, in GiB (what ``mem_get_info`` reports).
#:
#: Measured (Windows, one machine with an RTX 4070, one process per run, the
#: reserved memory capped with
#: ``set_per_process_memory_fraction``): at bw 384 a budget of 5.8 GiB failed and
#: 5.85 GiB ran; at bw 256 1.75 GiB failed and 1.8 GiB ran. The figures below are
#: the smallest budgets that ran (5.85 and 1.8) rounded up to 0.1 (5.9, 1.8);
#: the tensors alone are 5.56 and 1.66 GiB, and the
#: allocator keeps up to 6.48 and 1.95 GiB when there is room, which is comfort,
#: not need. One card only. The peak of about 7 GiB in ``nvidia-smi`` is the
#: whole machine and not what the grid needs. These are also the figures shown
#: next to the Quality presets. 128 is small enough never to be the problem.
_VRAM_NEEDED_GB = {256: 1.8, 384: 5.9}
_VRAM_HEADROOM = 1.15


def _vram_needed_gb(max_bandwidth) -> float:
    if max_bandwidth is None:
        return 0.0
    bw = int(max_bandwidth)
    best = 0.0
    for k, v in _VRAM_NEEDED_GB.items():
        if bw >= k:
            best = max(best, v)
    return best


def _check_vram_for_bandwidth(max_bandwidth) -> None:
    """Refuse a render grid the GPU cannot hold, instead of letting it crawl.

    At bw=384 on a 12 GB card that already had 11.5 GB in use, the load did
    not fail -- it thrashed for five minutes at 100 % GPU with the server
    blocked, and the process then died (2026-09-07). Refusing up front with
    both numbers in the message is the honest outcome: the user picks 256 or
    128 and keeps working. On the CPU the same question is asked of the RAM
    (:func:`_check_ram_for_bandwidth`). No-op when the memory cannot be read.
    """
    need = _vram_needed_gb(max_bandwidth)
    if need <= 0:
        return
    try:
        import torch
        dev = get_renderer().device
    except Exception:  # noqa: BLE001 -- cannot tell, do not refuse
        return
    if getattr(dev, "type", str(dev)) != "cuda" and "cuda" not in str(dev):
        # No VRAM here: the CPU path lives in RAM. Asked outside any ``try``, so
        # a refusal is not swallowed by the blanket ``except`` of a measurement.
        _check_ram_for_bandwidth(max_bandwidth)
        return
    try:
        # ``mem_get_info`` asks the driver, and the driver counts every block
        # PyTorch has cached after an earlier render as used: measured 4.3 GB
        # "free" against 10.3 GB once the cache was returned. Return the free
        # blocks first, so the figure is what a new grid can actually get.
        # Live tensors (the cached phases) are not touched by this.
        # Cost, measured on an RTX 4070: 0.02 ms with an empty cache, 47 ms with
        # 2 GB cached, ~290 ms with kernels still in flight (empty_cache waits
        # for them). It runs only here, i.e. for bw >= 256 on CUDA (``need`` is
        # 0 below that, and the CPU is checked against RAM above), once per
        # cache miss -- but it can briefly stall a parallel GPU user of this
        # process.
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info(dev if dev.index is not None else 0)
    except Exception:  # noqa: BLE001 -- cannot measure, do not refuse
        return
    free_gb = free_b / 2**30
    if free_gb < need * _VRAM_HEADROOM:
        raise SHTOutOfMemoryError(
            f"Quality bw={int(max_bandwidth)} needs about {need:.1f} GB of GPU memory "
            f"and only {free_gb:.1f} GB is free (of {total_b / 2**30:.1f} GB). "
            f"Choose 256 or 128, or free the GPU (other programs, or Release "
            f"GPU memory in Settings) and try again."
        )


#: RAM a CPU render grid needs per bandwidth, and the headroom asked for on top.
#:
#: Measured (Windows 11, torch 2.11.0+cu126 running on the CPU with CUDA hidden,
#: the 50 KB hP28 .sht of the library, first request in a
#: fresh backend, process RSS before -> after):
#:   bw 256: 1.1 -> 3.3 GB (+2.2) with 24 threads; 0.65 -> 2.47 GB (+1.8) with 12.
#:   bw 384: 1.1 -> 8.0 GB (+6.9); 0.65 -> 6.30 GB (+5.65) and 0.65 -> 7.8 GB (+7.1)
#:           in two further runs; 4 threads: 1.0 -> 7.9 GB.
#: Six simultaneous bw-256 requests for different files peaked at 13.7 GB
#: (about 2.1 GB each), three bw-384 requests at 28.5 GB -- the reason the
#: sphere route now runs them one at a time. The figures below are the upper
#: ends of the single-request runs.
_RAM_NEEDED_GB = {256: 2.2, 384: 7.0}
_RAM_HEADROOM = 1.15

#: (The sphere preview now computes in the compute worker, where this ledger
#: starts empty in every fresh worker. The in-backend callers of
#: load_or_get_phase -- pattern match, forward NCC, refinement -- still need it.)
#:
#: RSS growth a finished CPU load left behind, per bandwidth. A freed grid's
#: memory stays with the process (the allocator keeps it; clearing every cache
#: the project owns did not return it, measured), and the next load of the
#: same size reuses it -- a second bw-384 file grew the process by 0.0 GB. That
#: memory counts as available for the next load, or a 16 GB machine would
#: refuse the second Max request that needs nothing new.
_RAM_RETAINED_GB: "dict[int, float]" = {}

#: The process RSS before the first tracked CPU load. What the process holds
#: above it is the most that a load could have left behind NOW: the OS may have
#: trimmed the working set since, or other programs taken the memory, and a
#: credit recorded earlier must not outlive that.
_RAM_RSS_BASELINE_GB: "Optional[float]" = None


def _ram_needed_gb(max_bandwidth) -> float:
    if max_bandwidth is None:
        return 0.0
    bw = int(max_bandwidth)
    best = 0.0
    for k, v in _RAM_NEEDED_GB.items():
        if bw >= k:
            best = max(best, v)
    return best


def _process_rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**30


def _check_ram_for_bandwidth(max_bandwidth) -> None:
    """Refuse a CPU render grid the machine has no RAM for, instead of letting it swap.

    Without a GPU the inverse SHT builds its tables in RAM: ~2.2 GB at bw 256,
    ~7 GB at bw 384 (see ``_RAM_NEEDED_GB``). On a 16 GB machine with other
    programs open that is the difference between a clear message and a computer
    that stops answering with no error at all. No-op when psutil cannot report.
    """
    need = _ram_needed_gb(max_bandwidth)
    if need <= 0:
        return
    try:
        import psutil
        vm = psutil.virtual_memory()
        available_gb = vm.available / 2**30
        total_gb = vm.total / 2**30
    except Exception:  # noqa: BLE001 -- cannot measure, do not refuse
        return
    bw = int(max_bandwidth)
    credit = max((g for b, g in _RAM_RETAINED_GB.items() if b >= bw), default=0.0)
    if credit > 0:
        # Only what the process still holds can be reused: cap the recorded
        # credit by the RSS above the baseline as it is NOW.
        try:
            held = max(0.0, _process_rss_gb() - _RAM_RSS_BASELINE_GB)
        except Exception:  # noqa: BLE001 -- unknown means no credit
            held = 0.0
        credit = min(credit, held)
    if available_gb + credit < need * _RAM_HEADROOM:
        raise SHTOutOfMemoryError(
            f"Quality bw={bw} needs about {need:.1f} GB of free RAM "
            f"and only {available_gb:.1f} GB is available (of {total_gb:.1f} GB). "
            f"Choose a lower quality, or close other programs and try again."
        )


def _release_cuda_cache() -> None:
    """Hand PyTorch's cached-but-free blocks back after a failed load."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- best effort, never mask the real error
        pass


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
    global _RAM_RSS_BASELINE_GB
    path_key = str(Path(sht_path).resolve())
    bw_key = -1 if max_bandwidth is None else int(max_bandwidth)
    key = (path_key, bw_key)
    if key in _PHASE_CACHE:
        _PHASE_CACHE.move_to_end(key)
        return _PHASE_CACHE[key]
    if not Path(path_key).is_file():
        raise SHTRenderError(f"SHT file not found: {path_key}")
    _check_vram_for_bandwidth(max_bandwidth)
    failed_oom = False
    track_ram = (max_bandwidth is not None and int(max_bandwidth) >= 256
                 and getattr(get_renderer().device, "type", "") == "cpu")
    rss_before = None
    if track_ram:
        try:
            rss_before = _process_rss_gb()
        except Exception:  # noqa: BLE001 -- bookkeeping only
            rss_before = None
        if rss_before is not None and _RAM_RSS_BASELINE_GB is None:
            _RAM_RSS_BASELINE_GB = rss_before
    try:
        grid = get_renderer().load_phase(path_key, max_bandwidth=max_bandwidth)
    except SHTReadError as e:
        raise SHTRenderError(f"failed to read {path_key}: {e}") from e
    except Exception as e:  # noqa: BLE001 -- only OOM is translated, the rest re-raised
        if not is_out_of_memory(e):
            raise
        # The full trace goes to the log; the caller gets a short sentence. The
        # trace is logged as TEXT: a LogRecord carrying ``exc_info`` keeps the
        # traceback -- and with it the frames and tensors of the failed load --
        # alive for as long as any handler holds the record.
        logger.error("out of memory loading %s at bw=%s:\n%s",
                     path_key, bw_key, traceback.format_exc())
        failed_oom = True
    if failed_oom:
        # Outside the handler: while it runs, the traceback still references the
        # frames -- and the tensors -- of the failed load, and empty_cache()
        # cannot return what is still referenced.
        _release_cuda_cache()
        raise SHTOutOfMemoryError(
            f"Not enough memory to load this phase at bandwidth {bw_key}."
        )
    if rss_before is not None:
        try:
            growth = _process_rss_gb() - rss_before
        except Exception:  # noqa: BLE001 -- bookkeeping only
            growth = 0.0
        # Process-wide RSS also counts whatever ELSE the process allocated while
        # this load ran (a parallel indexing job, a dataset being read): a
        # parallel 3 GB allocation once inflated the credit to 4.8 GB and let a
        # bw-256 load pass with 0.5 GB free. What a load can leave behind for
        # the next one of its size is at most what such a load needs.
        credited = min(growth, _ram_needed_gb(max_bandwidth))
        if credited > _RAM_RETAINED_GB.get(int(max_bandwidth), 0.0):
            _RAM_RETAINED_GB[int(max_bandwidth)] = credited
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
    global _RENDERER, _RAM_RSS_BASELINE_GB
    n = len(_PHASE_CACHE)
    _PHASE_CACHE.clear()
    _RENDERER = None
    # The memory those loads left with the process is what the RAM preflight
    # credits; once the caches are dropped on purpose, so is the credit.
    _RAM_RETAINED_GB.clear()
    _RAM_RSS_BASELINE_GB = None
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
    # ``tilt`` is the DETECTOR elevation and is a separate angle from the
    # sample tilt (the renderer uses alpha = 90 - sample_tilt + det_tilt).
    # Omitting it makes every simulated pattern come out rotated by exactly
    # det.tilt, which drives the whole forward-NCC map toward 0.
    det_tilt = float(det.get("tilt", 0.0))

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
                det_tilt_deg=det_tilt,
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
        noticeably sharper patterns at ~1.8 GB VRAM; 384 the sharpest
        at ~5.9 GB.

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
