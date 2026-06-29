"""On-demand forward-simulation diagnostics for an indexing result.

Produces 4 per-pixel maps and a summary statistics block, all stored on
``result.metadata["forward_diagnostics"]``:

- ``ncc_map``               Forward-NCC (experimental vs SHT-simulated)
- ``local_anomaly_map``     ``ncc - median_filter(ncc, 5)`` (post-process)
- ``pc_sensitivity_map``    L2 magnitude of NCC gradient under (1px, 1px, 50um) PC perturbations
- ``pattern_residual_map``  Mean per-pixel ``|I_exp - I_sim|`` after unit-normalization

See spec: docs/superpowers/specs/2026-05-13-forward-ncc-diagnose-layer-design.md
"""
from __future__ import annotations

import base64
import datetime as _dt
import io as _io
import logging
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from orix.quaternion import Rotation
from scipy.ndimage import generic_filter

from backend.api.services.sht_pattern_renderer import (
    SHTRenderError,
    get_renderer,
    load_or_get_phase,
)
from backend.spherical_gpu.pipeline.detector import convert_pc_to_emsoft

logger = logging.getLogger(__name__)


def compute_local_anomaly_map(
    ncc_map: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Return ``ncc - nanmedian(ncc, window x window)`` per pixel.

    NaN inputs propagate to NaN outputs. For maps smaller than the window,
    returns zeros (after logging an info message).

    Parameters
    ----------
    ncc_map
        (H, W) float32 array. NaN where pixel is not indexed.
    window
        Side length of the square neighborhood. Default 5.
    """
    if ncc_map.shape[0] < window or ncc_map.shape[1] < window:
        logger.info(
            "compute_local_anomaly_map: map %s smaller than window=%d, returning zeros",
            ncc_map.shape, window,
        )
        return np.zeros_like(ncc_map, dtype=np.float32)

    def _nanmedian(values: np.ndarray) -> float:
        # generic_filter passes a flat window; reduce to nanmedian
        return float(np.nanmedian(values)) if not np.all(np.isnan(values)) else np.nan

    median = generic_filter(
        ncc_map.astype(np.float32, copy=False),
        _nanmedian,
        size=window,
        mode="reflect",
    ).astype(np.float32)
    return (ncc_map - median).astype(np.float32)


def summary_stats(arr: np.ndarray, n_bins: int = 30) -> dict:
    """Per-metric scalar summary + 30-bin histogram.

    Excludes NaN values from all aggregates. If all values are NaN, all
    scalar fields become NaN and the histogram has zero counts.
    """
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {
            "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"),
            "p05": float("nan"), "p95": float("nan"),
            "histogram": {
                "bins": [0.0] * (n_bins + 1),
                "counts": [0] * n_bins,
            },
        }
    counts, bins = np.histogram(valid, bins=n_bins)
    return {
        "mean": float(np.mean(valid)),
        "std":  float(np.std(valid)),
        "min":  float(np.min(valid)),
        "max":  float(np.max(valid)),
        "p05":  float(np.percentile(valid, 5)),
        "p95":  float(np.percentile(valid, 95)),
        "histogram": {
            "bins":   [float(b) for b in bins],
            "counts": [int(c) for c in counts],
        },
    }


_METRIC_TO_KEY = {
    "ncc": "ncc_map",
    "local_anomaly": "local_anomaly_map",
    "pc_sensitivity": "pc_sensitivity_map",
    "pattern_residual": "pattern_residual_map",
}


def _xmap_flat_index_map(result) -> tuple[np.ndarray, np.ndarray]:
    """Return (flat_mask, cumsum) for full-grid <-> xmap index back-mapping.

    flat_mask: (H*W,) bool -- True where pixel is in the indexed selection.
    cumsum:    (H*W,) int -- xmap row index for each True position in flat_mask.
               (Values at False positions are not meaningful and must not be read.)
    """
    H, W = result.original_shape
    n_total = H * W
    flat_mask = (result.selection_mask if result.selection_mask is not None
                 else np.ones(n_total, dtype=bool)).ravel()
    if int(result.xmap.rotations.size) == n_total:
        cumsum = np.arange(n_total)
    else:
        cumsum = np.cumsum(flat_mask) - 1
    return flat_mask, cumsum


def build_phase_id_2d(result) -> np.ndarray:
    """Return an (H, W) int16 grid of phase ids; -1 for unindexed pixels.

    Used by the /browser route to display per-pixel phase names alongside
    metric values. Vectorised -- no Python loop.
    """
    H, W = result.original_shape
    flat_mask, cumsum = _xmap_flat_index_map(result)
    phase_ids_flat = np.asarray(result.xmap.phase_id).ravel()
    pid2d = np.full((H, W), -1, dtype=np.int16)
    pid2d.reshape(-1)[flat_mask] = phase_ids_flat[cumsum[flat_mask]]
    return pid2d


def top_n_pixels_with_count(
    diag: dict,
    phase_ids: np.ndarray,
    phase_names: dict,
    *,
    metric: str = "ncc",
    sort: str = "asc",
    n: int = 50,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> tuple[list[dict], int]:
    """Return the top-N pixels ranked by ``metric``, plus the unfiltered total.

    Parameters
    ----------
    diag
        The forward_diagnostics block from result.metadata.
    phase_ids
        (H, W) int array — pixel-wise phase id, -1 for unindexed.
    phase_names
        {phase_id: name}.
    metric
        One of ``ncc | local_anomaly | pc_sensitivity | pattern_residual``.
    sort
        ``"asc"`` (worst-first for NCC) or ``"desc"``.
    n
        Cap on returned items.
    range_min, range_max
        Optional inclusive filter on the chosen metric.

    Returns
    -------
    (items, total_matching) where items is at most ``n`` long.
    """
    if metric not in _METRIC_TO_KEY:
        raise ValueError(f"unknown metric: {metric!r}")
    if sort not in ("asc", "desc"):
        raise ValueError(f"sort must be 'asc' or 'desc', got {sort!r}")
    key = _METRIC_TO_KEY[metric]
    arr = diag[key]
    H, W = arr.shape

    flat = arr.ravel()
    mask = ~np.isnan(flat)
    if range_min is not None:
        mask &= (flat >= range_min)
    if range_max is not None:
        mask &= (flat <= range_max)
    idx = np.where(mask)[0]
    total = int(idx.size)
    if total == 0:
        return [], 0

    values = flat[idx]
    order = np.argsort(values)
    if sort == "desc":
        order = order[::-1]
    idx = idx[order][:n]

    items = []
    for rank, flat_i in enumerate(idx, start=1):
        r, c = int(flat_i // W), int(flat_i % W)
        pid = int(phase_ids[r, c]) if 0 <= r < phase_ids.shape[0] and 0 <= c < phase_ids.shape[1] else -1
        items.append({
            "rank": rank,
            "row": r, "col": c,
            "phase_id": pid,
            "phase_name": phase_names.get(pid, "?"),
            "ncc":              float(diag["ncc_map"][r, c]),
            "local_anomaly":    float(diag["local_anomaly_map"][r, c]),
            "pc_sensitivity":   float(diag["pc_sensitivity_map"][r, c]),
            "pattern_residual": float(diag["pattern_residual_map"][r, c]),
        })
    return items, total


def top_n_pixels(
    diag: dict,
    phase_ids: np.ndarray,
    phase_names: dict,
    *,
    metric: str = "ncc",
    sort: str = "asc",
    n: int = 50,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> list[dict]:
    """Thin wrapper around top_n_pixels_with_count returning only the items."""
    items, _ = top_n_pixels_with_count(
        diag, phase_ids, phase_names,
        metric=metric, sort=sort, n=n,
        range_min=range_min, range_max=range_max,
    )
    return items


def _render_chunk_four_variants(
    *,
    renderer,
    grid,
    quats: "torch.Tensor",
    pc_emsoft: tuple,
    detector_shape: tuple,
    pixel_size_um: float,
    tilt_deg: float,
    deltas: dict,
) -> tuple:
    """Render the same chunk at (PC, PC+dx, PC+dy, PC+dL).

    Returns (base, dx, dy, dL) each of shape (B, H, W) on CPU.
    """
    xpc, ypc, L = pc_emsoft
    dx_px = float(deltas["dx_px"])
    dy_px = float(deltas["dy_px"])
    dL_um = float(deltas["dL_um"])

    base = renderer.render_batch(
        grid=grid, orientation_quats=quats,
        pc_emsoft=(xpc, ypc, L),
        detector_shape=detector_shape,
        pixel_size_um=pixel_size_um, tilt_deg=tilt_deg,
        chunk_size=quats.shape[0],
    )
    dx = renderer.render_batch(
        grid=grid, orientation_quats=quats,
        pc_emsoft=(xpc + dx_px, ypc, L),
        detector_shape=detector_shape,
        pixel_size_um=pixel_size_um, tilt_deg=tilt_deg,
        chunk_size=quats.shape[0],
    )
    dy = renderer.render_batch(
        grid=grid, orientation_quats=quats,
        pc_emsoft=(xpc, ypc + dy_px, L),
        detector_shape=detector_shape,
        pixel_size_um=pixel_size_um, tilt_deg=tilt_deg,
        chunk_size=quats.shape[0],
    )
    dL_var = renderer.render_batch(
        grid=grid, orientation_quats=quats,
        pc_emsoft=(xpc, ypc, L + dL_um),
        detector_shape=detector_shape,
        pixel_size_um=pixel_size_um, tilt_deg=tilt_deg,
        chunk_size=quats.shape[0],
    )
    return base, dx, dy, dL_var


def build_circular_include_mask(
    pat_h: int, pat_w: int, radius_fraction: float = 1.0,
) -> np.ndarray:
    """Flat (H*W,) bool mask: True inside the inscribed circle.

    Mirrors backend.api.routes.ebsd_viewer._compute_include_mask so the
    forward-NCC diagnostic masks EDAX circular patterns the same way the
    viewer / indexing paths do. The inscribed-circle radius is
    ``min(H, W) / 2`` scaled by ``radius_fraction``.
    """
    cy = (pat_h - 1) / 2.0
    cx = (pat_w - 1) / 2.0
    r = max(0.0, float(radius_fraction)) * (min(pat_h, pat_w) / 2.0)
    yy, xx = np.ogrid[:pat_h, :pat_w]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= (r * r)).ravel()


def _unit_normalize(v: np.ndarray,
                    include_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean-subtract + L2-normalize. Returns zeros if norm is too small.

    When ``include_mask`` (a flat bool array over the flattened pattern) is
    given, only the masked-in pixels participate — used to exclude the dark
    corners of EDAX circular detectors, whose mismatch against the
    full-square SHT render would otherwise dominate the NCC variance.
    """
    v = v.astype(np.float64).ravel()
    if include_mask is not None:
        v = v[include_mask]
    v -= v.mean()
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _compute_pixel_metrics(
    I_exp: np.ndarray,
    I_base: np.ndarray,
    I_dx: np.ndarray,
    I_dy: np.ndarray,
    I_dL: np.ndarray,
    include_mask: Optional[np.ndarray] = None,
) -> tuple[float, float, float]:
    """Reduce one (exp, base, dx, dy, dL) tuple to (ncc, pc_sens, residual_L1).

    Patterns are unit-normalized (mean-subtract + L2). PC sensitivity is the
    L2 magnitude of the NCC gradient across the 3 perturbation axes.
    Residual is mean(|exp - base|) on the unit-normalized vectors.

    ``include_mask`` (flat bool over the flattened pattern) restricts every
    pattern to the inside-circle pixels before normalization — pass it for
    EDAX circular detectors.
    """
    e = _unit_normalize(I_exp, include_mask)
    b = _unit_normalize(I_base, include_mask)
    x = _unit_normalize(I_dx, include_mask)
    y = _unit_normalize(I_dy, include_mask)
    L = _unit_normalize(I_dL, include_mask)
    if np.linalg.norm(e) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    ncc_base = float(np.dot(e, b))
    ncc_dx   = float(np.dot(e, x))
    ncc_dy   = float(np.dot(e, y))
    ncc_dL   = float(np.dot(e, L))
    sens = float(np.sqrt(
        (ncc_dx - ncc_base) ** 2 +
        (ncc_dy - ncc_base) ** 2 +
        (ncc_dL - ncc_base) ** 2
    ))
    residual = float(np.mean(np.abs(e - b)))
    return ncc_base, sens, residual


# Default PC perturbation step sizes (validated 2026-05-11 on 7050 dataset)
_DEFAULT_DELTAS = {"dx_px": 1.0, "dy_px": 1.0, "dL_um": 50.0}


def _get_experimental_pattern(result, row: int, col: int):
    """Indirection so tests can mock pattern access."""
    from tools.pattern_comparison import get_experimental_pattern
    return get_experimental_pattern(result, row, col)


class _CancelledError(RuntimeError):
    """Internal sentinel raised when cancel_event was set."""


def compute_full_diagnostics(
    result,
    max_bandwidth: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    deltas: Optional[dict] = None,
    chunk_size: int = 128,
) -> None:
    """Run the on-demand diagnostics pass and write block atomically to result.metadata.

    See spec section 4.3 for the algorithm. Stores the block in
    ``result.metadata["forward_diagnostics"]`` only on successful completion;
    on any exception the metadata is unchanged.
    """
    metadata = getattr(result, "metadata", None) or {}
    method = metadata.get("indexing_method")
    if method not in ("spherical", "dictionary"):
        raise SHTRenderError(
            f"forward diagnostics require spherical or dictionary indexing (got {method!r})"
        )
    sht_map = metadata.get("sht_paths_by_phase") or {}
    if not sht_map:
        raise SHTRenderError(
            "result.metadata['sht_paths_by_phase'] is empty -- re-run indexing to populate SHT metadata"
        )
    det = metadata.get("detector_geometry") or {}
    required = {"pat_height", "pat_width", "pc_x", "pc_y", "pc_z"}
    if not required.issubset(det.keys()):
        raise SHTRenderError(
            f"result.metadata['detector_geometry'] missing keys: {required - set(det.keys())}"
        )

    deltas = dict(_DEFAULT_DELTAS) if deltas is None else dict(deltas)
    n_rows, n_cols = result.original_shape
    pat_h = int(det["pat_height"])
    pat_w = int(det["pat_width"])
    pixel_size_um = float(det.get("pixel_size", 70.0))
    sample_tilt = float(det.get("sample_tilt", 70.0))

    # Circular-detector mask. EDAX/TSL detectors record a circular pattern
    # with dark corners; the SHT renderer fills the full square, so an
    # unmasked NCC lets the corner mismatch dominate and collapses the
    # correlation. Mask both experimental and simulated patterns to the
    # inscribed circle before NCC. Activated when the detector geometry
    # opts in (circular_mask=True / a radius fraction) or the vendor is
    # EDAX/TSL. Oxford/Bruker full-frame detectors are left unmasked so
    # their results stay bit-identical.
    include_mask: Optional[np.ndarray] = None
    # The spherical indexing path hardcodes detector_geometry['vendor'] =
    # 'Bruker' and records the REAL detected vendor under 'source_vendor'
    # (indexing.py:603/657) — so we must consult both, or the mask never
    # fires on real EDAX data.
    vendor_u = (str(det.get("vendor", "")) + " "
                + str(det.get("source_vendor", ""))).upper()
    want_mask = bool(det.get("circular_mask", False)) \
        or det.get("mask_radius_fraction") is not None \
        or "EDAX" in vendor_u or "TSL" in vendor_u
    if want_mask:
        rf = float(det.get("mask_radius_fraction", 1.0))
        include_mask = build_circular_include_mask(pat_h, pat_w, rf)
        if not include_mask.any():
            raise SHTRenderError(
                f"circular mask excludes all pixels (mask_radius_fraction={rf})"
            )
        logger.info(
            "forward-diagnostics: circular mask ON (vendor=%r source_vendor=%r "
            "rf=%.2f, %d/%d px inside)", det.get("vendor"),
            det.get("source_vendor"), rf,
            int(include_mask.sum()), include_mask.size,
        )

    xpc, ypc, L_um = convert_pc_to_emsoft(
        pc=(float(det["pc_x"]), float(det["pc_y"]), float(det["pc_z"])),
        vendor=str(det.get("vendor", "Bruker")),
        pat_width=pat_w, pat_height=pat_h,
        pixel_size=pixel_size_um, binning=int(det.get("binning", 1)),
    )
    pc_emsoft = (float(xpc), float(ypc), float(L_um))

    # Per-pixel PC override from refined results. When present, rendering must
    # be serialised one pixel at a time (each pixel uses its own PC).
    # Shape: (H, W, 3) in EMsoft convention (xpc, ypc, L_um).
    pc_per_pixel = metadata.get("pc_per_pixel")
    if pc_per_pixel is not None:
        pc_per_pixel = np.asarray(pc_per_pixel, dtype=np.float64)
        if pc_per_pixel.shape != (n_rows, n_cols, 3):
            raise SHTRenderError(
                f"pc_per_pixel must have shape ({n_rows}, {n_cols}, 3) in EMsoft "
                f"convention; got {pc_per_pixel.shape}"
            )

    # Output arrays (NaN where unindexed)
    ncc_map      = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    pc_sens_map  = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    residual_map = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    missing_phases: list[dict] = []

    # xmap pixel <-> flat-idx mapping
    xmap = result.xmap
    n_xmap = int(xmap.rotations.size)
    n_total = n_rows * n_cols
    flat_mask = (result.selection_mask if result.selection_mask is not None
                 else np.ones(n_total, dtype=bool)).ravel()
    if n_xmap == n_total:
        cumsum = np.arange(n_total)
    else:
        cumsum = np.cumsum(flat_mask) - 1
    phase_ids = np.asarray(xmap.phase_id).ravel()
    eulers_all = np.asarray(xmap.rotations.to_euler())  # (n_xmap, 3)

    total_indexed = int(np.sum((phase_ids >= 0)))
    done = 0
    n_pixels_without_exp_pattern = 0
    renderer = get_renderer()

    def _lookup_phase(xmap_obj, pid: int):
        """Return the Phase object for ``pid`` or None.

        ``orix.crystal_map.PhaseList`` is dict-like via ``__getitem__`` but
        has no ``.get()``. Use a guarded subscript so the diagnostics code
        works against real orix versions and the mocks in the unit tests.
        """
        phases = getattr(xmap_obj, "phases", None)
        if phases is None:
            return None
        # Prefer .get if the test mock provides it.
        getter = getattr(phases, "get", None)
        if callable(getter):
            try:
                return getter(pid)
            except Exception:
                return None
        try:
            return phases[pid]
        except (KeyError, IndexError, TypeError):
            return None

    for phase_id_int, sht_path in sht_map.items():
        phase_id = int(phase_id_int)
        # Resolve SHT path
        try:
            grid = load_or_get_phase(str(sht_path), max_bandwidth=max_bandwidth)
        except Exception as e:
            logger.warning("SHT for phase %s unavailable: %s", phase_id, e)
            phase_obj = _lookup_phase(xmap, phase_id)
            missing_phases.append({
                "phase_id": phase_id,
                "phase_name": getattr(phase_obj, "name", str(phase_id)),
                "reason": str(e),
                "path": str(sht_path),
            })
            continue

        phase_name = "?"
        ph = _lookup_phase(xmap, phase_id)
        if ph is not None:
            phase_name = getattr(ph, "name", str(phase_id))

        # Collect pixel coordinates for this phase
        phase_pixels = []
        phase_orient_idx = []
        for flat in range(n_total):
            if not flat_mask[flat]:
                continue
            px_idx = int(cumsum[flat])
            if int(phase_ids[px_idx]) != phase_id:
                continue
            phase_pixels.append(flat)
            phase_orient_idx.append(px_idx)
        if not phase_pixels:
            continue
        phase_pixels = np.asarray(phase_pixels, dtype=np.int64)
        phase_orient_idx = np.asarray(phase_orient_idx, dtype=np.int64)

        eulers = eulers_all[phase_orient_idx]
        quats_np = Rotation.from_euler(eulers).data
        quats = torch.tensor(np.asarray(quats_np)[:, :4], dtype=torch.float64)

        M = quats.shape[0]
        for chunk_start in range(0, M, chunk_size):
            if cancel_event is not None and cancel_event.is_set():
                raise _CancelledError("compute cancelled by user")
            chunk_end = min(chunk_start + chunk_size, M)
            cq = quats[chunk_start:chunk_end]
            cf = phase_pixels[chunk_start:chunk_end]

            if pc_per_pixel is None:
                base, dx, dy, dL = _render_chunk_four_variants(
                    renderer=renderer, grid=grid, quats=cq,
                    pc_emsoft=pc_emsoft, detector_shape=(pat_h, pat_w),
                    pixel_size_um=pixel_size_um, tilt_deg=sample_tilt,
                    deltas=deltas,
                )
            else:
                # Per-pixel PC: serialise the chunk -- one render call per pixel.
                # Sacrifices batching for correctness; refined data is the
                # explicit user path so the 3-4x slowdown is acceptable.
                base_list, dx_list, dy_list, dL_list = [], [], [], []
                for local_i, flat_idx in enumerate(cf):
                    r_loc = int(flat_idx // n_cols)
                    c_loc = int(flat_idx % n_cols)
                    pc_ij = tuple(float(v) for v in pc_per_pixel[r_loc, c_loc])
                    b, dxv, dyv, dLv = _render_chunk_four_variants(
                        renderer=renderer, grid=grid,
                        quats=cq[local_i:local_i + 1],
                        pc_emsoft=pc_ij, detector_shape=(pat_h, pat_w),
                        pixel_size_um=pixel_size_um, tilt_deg=sample_tilt,
                        deltas=deltas,
                    )
                    base_list.append(b)
                    dx_list.append(dxv)
                    dy_list.append(dyv)
                    dL_list.append(dLv)
                base = torch.cat(base_list, dim=0)
                dx = torch.cat(dx_list, dim=0)
                dy = torch.cat(dy_list, dim=0)
                dL = torch.cat(dL_list, dim=0)
            base_np = base.numpy()
            dx_np   = dx.numpy()
            dy_np   = dy.numpy()
            dL_np   = dL.numpy()

            for local_i, flat_idx in enumerate(cf):
                r = int(flat_idx // n_cols)
                c = int(flat_idx % n_cols)
                I_exp = _get_experimental_pattern(result, r, c)
                if I_exp is None:
                    n_pixels_without_exp_pattern += 1
                    continue
                ncc, sens, resid = _compute_pixel_metrics(
                    np.asarray(I_exp, dtype=np.float32),
                    base_np[local_i], dx_np[local_i],
                    dy_np[local_i], dL_np[local_i],
                    include_mask=include_mask,
                )
                ncc_map[r, c]      = ncc
                pc_sens_map[r, c]  = sens
                residual_map[r, c] = resid

            done += (chunk_end - chunk_start)
            if progress_callback is not None and total_indexed > 0:
                try:
                    progress_callback(min(1.0, done / total_indexed), phase_name)
                except Exception:
                    pass

    # Post-process
    local_anom = compute_local_anomaly_map(ncc_map, window=5)

    # Summary
    summary = {
        "n_pixels_total":   int(n_rows * n_cols),
        "n_pixels_indexed": total_indexed,
        "n_pixels_without_exp_pattern": int(n_pixels_without_exp_pattern),
        "missing_phases":   missing_phases,
        "ncc":              summary_stats(ncc_map),
        "local_anomaly":    summary_stats(local_anom),
        "pc_sensitivity":   summary_stats(pc_sens_map),
        "pattern_residual": summary_stats(residual_map),
    }

    # Atomic assign at the end
    block = {
        "bandwidth": int(max_bandwidth),
        "computed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "ncc_map": ncc_map,
        "local_anomaly_map": local_anom,
        "pc_sensitivity_map": pc_sens_map,
        "pattern_residual_map": residual_map,
        "pc_perturbation_deltas": deltas,
        "summary": summary,
    }
    result.metadata["forward_diagnostics"] = block


JOB_REGISTRY_CAPACITY = 8
JOB_REGISTRY: "OrderedDict[str, DiagnosticsJob]" = OrderedDict()


class DiagnosticsJob:
    """Tracks one in-flight or finished compute_full_diagnostics call.

    Lives in the module-level JOB_REGISTRY (LRU cap 8). Each job owns its
    own cancel_event; the worker thread checks it between chunks.
    """

    def __init__(self, result_id: str, max_bandwidth: int):
        self.job_id: str = uuid.uuid4().hex
        self.result_id: str = result_id
        self.max_bandwidth: int = int(max_bandwidth)
        self.state: str = "running"     # running | completed | failed | cancelled
        self.progress: float = 0.0
        self.current_phase: str = ""
        self.pixels_done: int = 0
        self.pixels_total: int = 0
        self.error: Optional[str] = None
        self.cancel_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        _register_job(self)

    def set_progress(self, frac: float, current_phase: str = ""):
        self.progress = float(max(0.0, min(1.0, frac)))
        if current_phase:
            self.current_phase = current_phase

    def mark_completed(self):
        if self.cancel_event.is_set():
            self.state = "cancelled"
            return
        self.state = "completed"
        self.progress = 1.0

    def mark_failed(self, error_msg: str):
        if self.cancel_event.is_set():
            self.state = "cancelled"
            return
        self.state = "failed"
        self.error = error_msg

    def cancel(self):
        self.state = "cancelled"
        self.cancel_event.set()

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "progress": self.progress,
            "current_phase": self.current_phase,
            "pixels_done": self.pixels_done,
            "pixels_total": self.pixels_total,
            "error": self.error,
            "result_id": self.result_id,
        }


def _register_job(job: "DiagnosticsJob") -> None:
    JOB_REGISTRY[job.job_id] = job
    while len(JOB_REGISTRY) > JOB_REGISTRY_CAPACITY:
        JOB_REGISTRY.popitem(last=False)


def get_active_job_for_result(result_id: str) -> Optional["DiagnosticsJob"]:
    """Return a job for ``result_id`` whose state is 'running', if any."""
    for j in JOB_REGISTRY.values():
        if j.result_id == result_id and j.state == "running":
            return j
    return None


def _to_b64_png(arr: np.ndarray, size: int) -> str:
    """Min-max normalize a 2D array to uint8 and encode as base64 PNG (size x size).

    NaN-safe: NaN pixels are mapped to the bottom of the colormap. If the array
    is entirely NaN or constant, returns a uniform zero image.
    """
    from PIL import Image
    if not np.isfinite(arr).any():
        norm = np.zeros(arr.shape, dtype=np.uint8)
    else:
        arr_min = float(np.nanmin(arr))
        arr_max = float(np.nanmax(arr))
        rng = arr_max - arr_min
        if rng < 1e-12:
            norm = np.zeros(arr.shape, dtype=np.uint8)
        else:
            # Replace NaN with arr_min so it lands at the bottom of the colormap
            arr_safe = np.where(np.isnan(arr), arr_min, arr)
            norm = ((arr_safe - arr_min) / rng * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(norm)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def thumbnail_pair(result, row: int, col: int, size: int = 64) -> dict:
    """Return base64 PNGs of (experimental, simulated) at ``size`` x ``size``.

    Re-renders the simulated pattern fresh at the thumbnail size (separate
    from the stored compute pattern shape). Raises ``ValueError`` if the
    pixel is not indexed.
    """
    metadata = result.metadata or {}
    det = metadata.get("detector_geometry")
    if det is None:
        raise ValueError("result.metadata['detector_geometry'] missing -- re-run indexing")
    sht_map = metadata.get("sht_paths_by_phase") or {}
    if not sht_map:
        raise ValueError("result.metadata['sht_paths_by_phase'] is empty -- re-run indexing")
    n_rows, n_cols = result.original_shape

    # Find pixel's phase + orientation in xmap
    flat_mask = (result.selection_mask if result.selection_mask is not None
                 else np.ones(n_rows * n_cols, dtype=bool)).ravel()
    if n_rows * n_cols == int(result.xmap.rotations.size):
        cumsum = np.arange(n_rows * n_cols)
    else:
        cumsum = np.cumsum(flat_mask) - 1
    flat_i = row * n_cols + col
    if not flat_mask[flat_i]:
        raise ValueError(f"pixel ({row},{col}) is not indexed")
    px_idx = int(cumsum[flat_i])
    pid = int(np.asarray(result.xmap.phase_id).ravel()[px_idx])
    if pid < 0:
        raise ValueError(f"pixel ({row},{col}) is not indexed")
    sht_path = sht_map.get(pid)
    if sht_path is None:
        raise ValueError(f"no SHT registered for phase {pid}")

    # Render simulated at thumbnail size
    grid = load_or_get_phase(str(sht_path))
    eulers = np.asarray(result.xmap.rotations.to_euler())[px_idx]
    quat_np = Rotation.from_euler(eulers[None, :]).data[0, :4]
    quat = torch.tensor(quat_np, dtype=torch.float64)

    xpc, ypc, L_um = convert_pc_to_emsoft(
        pc=(float(det["pc_x"]), float(det["pc_y"]), float(det["pc_z"])),
        vendor=str(det.get("vendor", "Bruker")),
        pat_width=int(det["pat_width"]), pat_height=int(det["pat_height"]),
        pixel_size=float(det.get("pixel_size", 70.0)),
        binning=int(det.get("binning", 1)),
    )
    # Scale both PC pixel offsets AND pixel size by the same factor so that the
    # detector covers the same field of view at the thumbnail resolution.
    # NOTE: assumes square detector (pat_height == pat_width); non-square
    # detectors would need independent x/y scale factors.
    pat_h_int = int(det["pat_height"])
    pat_w_int = int(det["pat_width"])
    if pat_h_int != pat_w_int:
        raise ValueError(
            f"thumbnail_pair currently requires square detectors "
            f"(got {pat_h_int}x{pat_w_int}). Non-square support is planned for Phase B."
        )
    scale = size / pat_h_int
    xpc_t = float(xpc) * scale
    ypc_t = float(ypc) * scale
    pixel_size_thumb = float(det.get("pixel_size", 70.0)) / scale
    sim = get_renderer().render(
        grid=grid, orientation_quat=quat,
        pc_emsoft=(xpc_t, ypc_t, float(L_um)),
        detector_shape=(size, size),
        pixel_size_um=pixel_size_thumb,
        tilt_deg=float(det.get("sample_tilt", 70.0)),
    ).numpy().astype(np.float32)
    exp = _get_experimental_pattern(result, row, col)
    if exp is None:
        raise ValueError(f"pixel ({row},{col}) has no experimental pattern")
    exp = np.asarray(exp, dtype=np.float32)

    return {
        "row": row, "col": col,
        "experimental_b64": _to_b64_png(exp, size),
        "simulated_b64":    _to_b64_png(sim, size),
    }
