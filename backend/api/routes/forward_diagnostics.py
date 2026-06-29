"""On-demand forward-NCC + PC sensitivity + pattern-residual diagnostics.

Spec: docs/superpowers/specs/2026-05-13-forward-ncc-diagnose-layer-design.md
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.api.routes.indexing import _result_registry
from backend.api.services import forward_diagnostics as svc

logger = logging.getLogger(__name__)
router = APIRouter()


class ComputeRequest(BaseModel):
    result_id: str
    max_bandwidth: int = 256
    force_recompute: bool = False


def _estimate_seconds(result, max_bandwidth: int) -> int:
    """Rough estimate based on pixel count and bandwidth."""
    H, W = result.original_shape
    indexed = int((H * W) * 0.95)   # rough fill estimate
    # 4 renders/pixel at bw=128 ~ 0.5ms each on RTX 4070; scale by (bw/128)**1.5
    ms_per_pixel = 4 * 0.5 * (max_bandwidth / 128.0) ** 1.5
    return int(max(5, indexed * ms_per_pixel / 1000))


@router.post("/compute")
def compute(req: ComputeRequest):
    result = _result_registry.get(req.result_id)
    if result is None:
        raise HTTPException(status_code=404,
                            detail={"error": f"no indexing result for id '{req.result_id}'"})
    if req.max_bandwidth not in (128, 256, 384):
        raise HTTPException(status_code=400,
                            detail={"error": f"max_bandwidth must be 128, 256, or 384 (got {req.max_bandwidth})"})

    method = (result.metadata or {}).get("indexing_method")
    if method not in ("spherical", "dictionary"):
        raise HTTPException(status_code=409,
                            detail={"error": f"forward diagnostics require spherical or dictionary indexing (got {method!r})"})
    if not (result.metadata or {}).get("sht_paths_by_phase"):
        raise HTTPException(status_code=409,
                            detail={"error": "result.metadata['sht_paths_by_phase'] is empty -- re-run indexing to populate SHT metadata"})

    # Cache check
    existing = (result.metadata or {}).get("forward_diagnostics")
    if existing and not req.force_recompute:
        if existing["bandwidth"] == req.max_bandwidth:
            return {
                "job_id": None,
                "already_computed": True,
                "existing_bandwidth": existing["bandwidth"],
                "computed_at": existing.get("computed_at"),
            }
        else:
            return {
                "job_id": None,
                "already_computed": True,
                "existing_bandwidth": existing["bandwidth"],
                "requested_bandwidth": req.max_bandwidth,
                "warning": f"Existing diagnostics at bw={existing['bandwidth']}; set force_recompute=true to redo at bw={req.max_bandwidth}",
            }

    # Conflict if a job is already running for this result
    running = svc.get_active_job_for_result(req.result_id)
    if running is not None:
        raise HTTPException(status_code=409, detail={
            "error": "job already running for this result",
            "running_job_id": running.job_id,
        })

    job = svc.DiagnosticsJob(result_id=req.result_id, max_bandwidth=req.max_bandwidth)

    def _worker():
        try:
            svc.compute_full_diagnostics(
                result,
                max_bandwidth=req.max_bandwidth,
                progress_callback=job.set_progress,
                cancel_event=job.cancel_event,
            )
            # Verify result still in registry (race with re-index)
            if _result_registry.get(req.result_id) is not result:
                logger.info("compute job %s: result replaced before write; discarding", job.job_id)
                job.mark_failed("result was replaced during compute")
                return
            # If user cancelled after compute_full_diagnostics finished but before we
            # got here, treat as cancelled: drop the just-written diagnostics block.
            if job.cancel_event.is_set():
                try:
                    (result.metadata or {}).pop("forward_diagnostics", None)
                except Exception:
                    pass
                job.state = "cancelled"
                return
            job.mark_completed()
        except svc._CancelledError:
            # cancel_event was set; partial data already discarded by compute_full_diagnostics
            job.state = "cancelled"
        except Exception as e:
            logger.exception("forward-diagnostics compute job failed")
            job.mark_failed(str(e))

    t = threading.Thread(target=_worker, daemon=True, name=f"forward-diag-{job.job_id[:8]}")
    job.thread = t
    t.start()

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "estimated_seconds": _estimate_seconds(result, req.max_bandwidth),
            "already_computed": False,
        },
    )


@router.get("/progress")
def progress(job_id: str):
    job = svc.JOB_REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": f"no job with id {job_id}"})
    return job.to_dict()


@router.post("/cancel")
def cancel(job_id: str):
    job = svc.JOB_REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": f"no job with id {job_id}"})
    job.cancel()
    return {"state": "cancelled"}


_VALID_SORTS = {
    "ncc_asc":              ("ncc",              "asc"),
    "ncc_desc":             ("ncc",              "desc"),
    "local_anomaly_desc":   ("local_anomaly",    "desc"),
    "local_anomaly_asc":    ("local_anomaly",    "asc"),
    "pc_sensitivity_desc":  ("pc_sensitivity",   "desc"),
    "pc_sensitivity_asc":   ("pc_sensitivity",   "asc"),
    "pattern_residual_desc":("pattern_residual", "desc"),
    "pattern_residual_asc": ("pattern_residual", "asc"),
}


@router.get("/summary")
def summary(result_id: str):
    result = _result_registry.get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"error": f"no indexing result for id '{result_id}'"})
    diag = (result.metadata or {}).get("forward_diagnostics")
    if diag is None:
        raise HTTPException(status_code=404, detail={"error": "forward diagnostics not yet computed for this result"})
    return {
        "computed": True,
        "bandwidth": diag["bandwidth"],
        "computed_at": diag["computed_at"],
        "n_pixels_indexed": diag["summary"]["n_pixels_indexed"],
        "missing_phases":   diag["summary"]["missing_phases"],
        "ncc":              diag["summary"]["ncc"],
        "local_anomaly":    diag["summary"]["local_anomaly"],
        "pc_sensitivity":   diag["summary"]["pc_sensitivity"],
        "pattern_residual": diag["summary"]["pattern_residual"],
    }


@router.get("/browser")
def browser(
    result_id: str,
    sort: str = "ncc_asc",
    n: int = 50,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
):
    if sort not in _VALID_SORTS:
        raise HTTPException(status_code=400, detail={
            "error": f"unknown sort '{sort}', valid: {sorted(_VALID_SORTS.keys())}",
        })
    result = _result_registry.get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"error": f"no indexing result for id '{result_id}'"})
    diag = (result.metadata or {}).get("forward_diagnostics")
    if diag is None:
        raise HTTPException(status_code=404, detail={"error": "forward diagnostics not yet computed"})

    n = max(1, min(int(n), 500))
    metric, direction = _VALID_SORTS[sort]
    pid2d = svc.build_phase_id_2d(result)
    phase_names = {}
    if hasattr(result.xmap, "phases"):
        for pid, ph in result.xmap.phases.items():
            phase_names[int(pid)] = getattr(ph, "name", str(pid))

    items, total = svc.top_n_pixels_with_count(
        diag, pid2d, phase_names,
        metric=metric, sort=direction, n=n,
        range_min=range_min, range_max=range_max,
    )
    return {
        "metric": metric,
        "sort": direction,
        "range": [range_min, range_max] if (range_min is not None or range_max is not None) else None,
        "total_matching": total,
        "items": items,
    }


@router.get("/thumbnail")
def thumbnail(result_id: str, row: int, col: int, size: int = 64):
    result = _result_registry.get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"error": f"no indexing result for id '{result_id}'"})
    size = max(16, min(int(size), 256))
    try:
        return svc.thumbnail_pair(result, row=row, col=col, size=size)
    except ValueError as e:
        raise HTTPException(status_code=404, detail={"error": str(e)})
    except Exception as e:
        logger.exception("thumbnail render failed")
        raise HTTPException(status_code=500, detail={"error": f"thumbnail render failed: {e}"})
