"""NLPAR pattern denoising for the EBSD Viewer: ``POST /api/ebsd/nlpar``.

Lives in its own module on purpose: ``ebsd_viewer.py`` is a hot file. The
route works on the viewer's ACTIVE signal like the other in-place pattern
operations (frame average, background removal, CLAHE): it validates, runs the
service in a worker thread, swaps the denoised patterns into the same signal
object, and flags the dataset dirty so that indexing and export use them.
Progress goes through the viewer's own ``_set_processing_progress`` and is
polled with the existing ``GET /api/ebsd/processing-progress/{request_id}``.

Provenance is a mark inside the signal's own metadata (``Orienta.nlpar``), so
a deepcopy and the per-file stash carry it without a second registry.
"""

import asyncio
import logging
import threading
import time
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.routes import ebsd_viewer as ev
from backend.api.services import nlpar as nlpar_service

logger = logging.getLogger(__name__)

router = APIRouter()

# request_id -> cancel event, for the runs that are in flight
_cancel_events: Dict[str, threading.Event] = {}
# names of datasets with a run in flight (one at a time per dataset)
_running: set = set()
_state_lock = threading.Lock()


class NlparRequest(BaseModel):
    radius: int = nlpar_service.DEFAULT_RADIUS     # search radius, patterns (1-3)
    request_id: Optional[str] = None               # progress / cancel handle
    force: bool = False                            # run again on an already denoised dataset
    lam: Optional[float] = None                    # fixed strength: tests only, the UI never sends it


def _is_hex_scan(path) -> bool:
    """EDAX hexagonal grids are stored as a padded rectangle that is not the
    neighbourhood NLPAR assumes. A build without ``edax_hex`` has no hex
    support at all, so there is nothing to detect."""
    if not path:
        return False
    try:
        from edax_hex import is_edax_hex_file
    except ImportError:
        return False
    try:
        return bool(is_edax_hex_file(path))
    except Exception:
        return False


def _is_hex_geometry() -> bool:
    """Does the geometry registry say the active dataset is a hexagonal scan, whatever file it
    came from (an EDAX APEX ``.up1`` is none of the OIM ``.h5`` files ``_is_hex_scan`` knows)?
    ``POST /nlpar`` is refused for it by ``hex_guard``; this lets the advice say so too."""
    from backend.api.services import scan_geometry
    geometry = scan_geometry.geometry_for(ev._active_dataset)
    return geometry is not None and geometry.kind == "hex"


def _available_ram() -> int:
    import psutil
    return int(psutil.virtual_memory().available)


@router.post("/nlpar")
async def nlpar(req: NlparRequest):
    """Denoise the active dataset's patterns with NLPAR, in place.

    A hexagonal scan is refused before this body runs: ``hex_guard.GUARDED`` lists the route, and
    the router carries the gate (``400``, code ``hex_neighbourhood_unsupported``). The check of an
    OIM hexagonal ``.h5`` below stays as the older, file-based guard."""
    signal = ev._get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")
    name = ev._active_dataset

    if not (nlpar_service.MIN_RADIUS <= req.radius <= nlpar_service.MAX_RADIUS):
        raise HTTPException(
            status_code=400,
            detail=f"Search radius must be {nlpar_service.MIN_RADIUS}-{nlpar_service.MAX_RADIUS}; got {req.radius}")
    if req.lam is not None and not req.lam > 0:
        raise HTTPException(status_code=400, detail="lam must be positive")

    nav = tuple(int(s) for s in signal.axes_manager.navigation_shape)
    if len(nav) != 2:
        raise HTTPException(status_code=400, detail="NLPAR needs a 2-D scan (rows x columns of patterns)")
    n_cols, n_rows = nav                      # hyperspy navigation_shape is (x, y)
    if nlpar_service.effective_radius(req.radius, n_rows, n_cols) < 1:
        raise HTTPException(status_code=400, detail="The scan is too small for NLPAR (at least 3 x 3 patterns are needed)")
    if _is_hex_scan(ev._ebsd_file_path):
        raise HTTPException(
            status_code=400,
            detail="NLPAR is not available for hexagonal EDAX scans: "
                   "the stored grid is not the neighbourhood NLPAR compares.")

    earlier = nlpar_service.applied_info(signal)
    if earlier is not None and not req.force:
        raise HTTPException(
            status_code=409,
            detail=f"This dataset was already denoised with NLPAR (search radius {earlier.get('radius')}). "
                   "Choose the original dataset, or run NLPAR on a copy.")

    sig_shape = tuple(int(s) for s in signal.axes_manager.signal_shape)    # (W, H)
    pattern_px = sig_shape[0] * sig_shape[1]
    n_patterns = n_rows * n_cols
    try:
        itemsize = int(signal.data.dtype.itemsize)
        available = _available_ram()
    except Exception as e:                                                    # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))
    problem = nlpar_service.capacity_problem(
        n_patterns, pattern_px, itemsize, available, ev._STASH_MATERIALIZE_MAX_BYTES,
        strip_bytes=nlpar_service.strip_working_bytes(n_rows, n_cols, pattern_px, req.radius))
    if problem:
        raise HTTPException(status_code=413, detail=problem)

    with _state_lock:
        if name in _running:
            raise HTTPException(status_code=409, detail="NLPAR is already running on this dataset")
        _running.add(name)
        cancel = threading.Event()
        if req.request_id:
            _cancel_events[req.request_id] = cancel

    t0 = time.perf_counter()
    started_at = time.time()
    rid = req.request_id
    ev._set_processing_progress(rid, done=0, total=1, fraction=0.0, stage="running",
                                phase="compile", started_at=started_at, n_patterns=n_patterns)
    logger.info("NLPAR start: %d patterns (%dx%d), radius %d", n_patterns, n_rows, n_cols, req.radius)

    def _progress(stage: str, fraction: float, done: int, total: int) -> None:
        ev._set_processing_progress(rid, done=done, total=total, fraction=fraction, stage="running",
                                    phase=stage, started_at=started_at, n_patterns=n_patterns)

    # The array the run starts from. If another request replaces `signal.data` meanwhile
    # (background removal, a crop in place), the result belongs to data that no longer is
    # the dataset's, and writing it back would silently undo that request.
    data0 = signal.data

    def _run():
        return nlpar_service.denoise(
            data0, radius=req.radius, lam=req.lam, progress=_progress, cancel=cancel)

    try:
        try:
            result = await asyncio.to_thread(_run)
        except nlpar_service.NlparCancelled:
            logger.info("NLPAR cancelled after %.1fs; dataset left unchanged", time.perf_counter() - t0)
            ev._set_processing_progress(rid, stage="error", error="cancelled", started_at=started_at)
            return {"success": False, "cancelled": True}
        except ValueError as e:
            ev._set_processing_progress(rid, stage="error", error=str(e), started_at=started_at)
            raise HTTPException(status_code=400, detail=str(e))
        except ImportError as e:
            ev._set_processing_progress(rid, stage="error", error=str(e), started_at=started_at)
            raise HTTPException(status_code=501, detail=f"NLPAR needs PyEBSDIndex with numba: {e}")
        except Exception as e:
            logger.exception("NLPAR failed after %.1fs", time.perf_counter() - t0)
            ev._set_processing_progress(rid, stage="error", error=str(e), started_at=started_at)
            raise HTTPException(status_code=500, detail=str(e))

        # The worker ran for a while. If the user switched dataset in between,
        # the patterns belong to a signal that is no longer the active one:
        # discard instead of writing into the wrong dataset.
        if ev._active_dataset != name or ev._get_active_signal() is not signal or signal.data is not data0:
            msg = "The dataset changed while NLPAR was running; the result was discarded."
            ev._set_processing_progress(rid, stage="error", error=msg, started_at=started_at)
            raise HTTPException(status_code=409, detail=msg)

        nlpar_service.swap_data(signal, result.data)
        nlpar_service.mark_applied(signal, result)
        ev._mark_active_dirty()

        elapsed = time.perf_counter() - t0
        logger.info("NLPAR done: %d patterns in %.1fs (%.0f patterns/s), radius %d, lambda %.3f, sigma %.2f",
                    n_patterns, elapsed, n_patterns / max(elapsed, 1e-6), result.radius, result.lam,
                    result.sigma_median)
        ev._set_processing_progress(rid, fraction=1.0, stage="complete", started_at=started_at,
                                    elapsed_seconds=round(elapsed, 2))
        return {
            "success": True,
            "radius": result.radius,
            "radius_requested": req.radius,
            "lam": round(result.lam, 4),
            "sigma_median": round(result.sigma_median, 3),
            "n_patterns": n_patterns,
            "n_strips": result.n_strips,
            "elapsed_seconds": round(elapsed, 2),
        }
    finally:
        with _state_lock:
            _running.discard(name)
            if rid:
                _cancel_events.pop(rid, None)


@router.post("/nlpar/cancel/{request_id}")
async def nlpar_cancel(request_id: str):
    """Ask a running NLPAR to stop at the next strip. The dataset stays as it was."""
    with _state_lock:
        event = _cancel_events.get(request_id)
    if event is None:
        return {"cancelled": False}
    event.set()
    return {"cancelled": True}


@router.get("/nlpar/advice")
async def nlpar_advice():
    """Should NLPAR be run on the active dataset? Cheap, from a sample of ~2000 patterns.

    ``recommendation`` is ``recommended`` (the neighbours differ about as much as noise
    explains: NLPAR will help), ``not_recommended`` (they differ far more: clean scan or
    structure finer than the search window, NLPAR then blurs boundaries), ``applied``
    (the dataset already carries the NLPAR mark: the control must be disabled, a second
    run is refused) or ``unavailable`` with a ``reason`` code. The user can still run it, except
    on a hexagonal scan (``reason`` ``hex``: an OIM hexagonal file, or any dataset whose registered
    geometry is hexagonal): ``POST /nlpar`` is refused there.
    """
    signal = ev._get_active_signal()
    if signal is None:
        raise HTTPException(status_code=400, detail="No EBSD data loaded")
    earlier = nlpar_service.applied_info(signal)
    if earlier is not None:
        return {"recommendation": "applied", "applied": True, "applied_radius": earlier.get("radius")}
    base = {"applied": False, "threshold": nlpar_service.LAMBDA_NOT_RECOMMENDED_FROM}
    nav = tuple(int(s) for s in signal.axes_manager.navigation_shape)
    if len(nav) != 2 or min(nav) < 5:
        return {**base, "recommendation": "unavailable", "reason": "too_small"}
    if _is_hex_scan(ev._ebsd_file_path) or _is_hex_geometry():
        return {**base, "recommendation": "unavailable", "reason": "hex"}
    try:
        est = await asyncio.to_thread(nlpar_service.estimate_noise, signal.data)
    except ValueError:
        return {**base, "recommendation": "unavailable", "reason": "uniform"}
    except Exception:
        logger.exception("NLPAR advice failed")
        return {**base, "recommendation": "unavailable", "reason": "error"}
    return {**base, "recommendation": nlpar_service.advice_for(est),
            "lam": round(est.lam, 3), "sigma": round(est.sigma, 2),
            "n_patterns": est.n_patterns, "seconds": round(est.seconds, 2)}
