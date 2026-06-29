"""REST API for phase refinement.

Two sets of endpoints coexist on this router:

* Multiphase phase-refinement (legacy): ``/{file_stem}/summary``,
  ``/{file_stem}/pixel/{row}/{col}``, ``/{file_stem}/override`` and
  ``/{file_stem}/override-region`` — operate on ``_multiphase.h5``
  checkpoints from the batch-v2 phase-refinement UI.
* Joint R+PC refinement (Phase B): ``/compute``, ``/resmooth``, ``/cancel``,
  ``/progress`` and ``/summary`` — drive the new joint orientation +
  pattern-centre refinement orchestrator
  (``backend.api.services.refinement.compute_full_refinement``).

The two groups never share a path: the legacy endpoints all live below a
``{file_stem}`` path-parameter segment and never match the bare single-segment
joint-refinement endpoints registered below.

Spec: ``docs/superpowers/specs/2026-05-14-joint-r-pc-refinement-design.md``.
"""
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.api.services.checkpoint_writer import CheckpointWriter

logger = logging.getLogger(__name__)
router = APIRouter()


def _find_checkpoint(file_stem: str, search_dir: Optional[str] = None) -> CheckpointWriter:
    """Find the _multiphase.h5 file for a given source file stem."""
    if search_dir:
        path = os.path.join(search_dir, f"{file_stem}_multiphase.h5")
        if os.path.isfile(path):
            return CheckpointWriter(os.path.join(search_dir, f"{file_stem}.h5oina"))
    # Fallback: search in common locations
    for d in ["Test_data/batch_test", "."]:
        path = os.path.join(d, f"{file_stem}_multiphase.h5")
        if os.path.isfile(path):
            return CheckpointWriter(os.path.join(d, f"{file_stem}.h5oina"))
    raise HTTPException(404, f"No multiphase result found for '{file_stem}'")


@router.get("/{file_stem}/summary")
async def get_summary(file_stem: str, search_dir: Optional[str] = Query(None)):
    import h5py as _h5py
    cw = _find_checkpoint(file_stem, search_dir)
    validation = cw.validate()
    if not validation.valid:
        raise HTTPException(400, f"Corrupt checkpoint: {validation.errors}")
    with _h5py.File(cw.checkpoint_path, "r") as f:
        phases = list(f.get("phases", {}).keys())
        grid_shape = [int(x) for x in f["metadata"].attrs.get("grid_shape", [0, 0])]
        auto = {}
        if "auto_assignment" in f:
            best_ci = np.array(f["auto_assignment/best_ci"])
            uncertainty = np.array(f["auto_assignment/uncertainty"])
            auto = {
                "mean_best_ci": float(np.nanmean(best_ci)),
                "mean_uncertainty": float(np.nanmean(uncertainty)),
                "uncertain_pixels": int(int(np.sum(uncertainty < 0.3))),
            }
    return {"phases": phases, "grid_shape": grid_shape, "auto_assignment_stats": auto}


@router.get("/{file_stem}/pixel/{row}/{col}")
async def get_pixel_info(file_stem: str, row: int, col: int, search_dir: Optional[str] = Query(None)):
    cw = _find_checkpoint(file_stem, search_dir)
    all_cis = cw.read_all_phase_cis()
    result = []
    for name, ci_map in all_cis.items():
        if 0 <= row < ci_map.shape[0] and 0 <= col < ci_map.shape[1]:
            result.append({"name": name, "ci": float(ci_map[row, col])})
    result.sort(key=lambda x: x["ci"], reverse=True)
    return {"phases": result, "row": row, "col": col}


class OverrideRequest(BaseModel):
    pixels: List[List[int]]  # [[row, col, phase_idx], ...]
    source: str = "click"


@router.post("/{file_stem}/override")
async def set_override(file_stem: str, req: OverrideRequest, search_dir: Optional[str] = Query(None)):
    import h5py as _h5py
    cw = _find_checkpoint(file_stem, search_dir)
    with _h5py.File(cw.checkpoint_path, "r") as f:
        raw_shape = f["metadata"].attrs.get("grid_shape")
        if raw_shape is None or len(raw_shape) < 2:
            raise HTTPException(
                400,
                f"Checkpoint for '{file_stem}' is missing grid_shape — cannot apply overrides"
            )
        grid_shape = tuple(int(v) for v in raw_shape[:2])

    # Load existing or create new override maps. Use int16 (matches the
    # widened dtype in checkpoint_writer.write_manual_override, commit
    # f83e81f) so projects with > 127 phases don't silently overflow.
    override_map = np.full(grid_shape, -1, dtype=np.int16)
    source_map = np.zeros(grid_shape, dtype=np.uint8)

    if os.path.isfile(cw.checkpoint_path):
        import h5py as _h5py2
        with _h5py2.File(cw.checkpoint_path, "r") as f:
            if "manual_override/phase_id" in f:
                override_map = np.array(f["manual_override/phase_id"], dtype=np.int16)
                source_map = np.array(f["manual_override/override_source"])

    source_codes = {"auto": 0, "click": 1, "region": 2, "threshold": 3}
    src_code = source_codes.get(req.source, 1)

    applied = 0
    skipped_malformed = 0
    skipped_out_of_bounds = 0
    for pixel in req.pixels:
        # A malformed pixel (wrong arity) used to raise ValueError and return
        # 500 for the whole batch; now we just skip it and count applied.
        if len(pixel) != 3:
            skipped_malformed += 1
            continue
        r, c, phase_idx = pixel
        if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
            override_map[r, c] = phase_idx
            source_map[r, c] = src_code
            applied += 1
        else:
            skipped_out_of_bounds += 1

    cw.write_manual_override(override_map, source_map)
    return {
        "status": "ok",
        "overrides_applied": applied,
        "skipped_malformed": skipped_malformed,
        "skipped_out_of_bounds": skipped_out_of_bounds,
    }


class RegionOverrideRequest(BaseModel):
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    phase_idx: int


@router.post("/{file_stem}/override-region")
async def set_region_override(file_stem: str, req: RegionOverrideRequest, search_dir: Optional[str] = Query(None)):
    import h5py as _h5py
    cw = _find_checkpoint(file_stem, search_dir)
    with _h5py.File(cw.checkpoint_path, "r") as f:
        raw_shape = f["metadata"].attrs.get("grid_shape")
        if raw_shape is None or len(raw_shape) < 2:
            raise HTTPException(
                400,
                f"Checkpoint for '{file_stem}' is missing grid_shape — cannot apply overrides"
            )
        grid_shape = tuple(int(v) for v in raw_shape[:2])

    override_map = np.full(grid_shape, -1, dtype=np.int16)
    source_map = np.zeros(grid_shape, dtype=np.uint8)
    if os.path.isfile(cw.checkpoint_path):
        with _h5py.File(cw.checkpoint_path, "r") as f:
            if "manual_override/phase_id" in f:
                override_map = np.array(f["manual_override/phase_id"], dtype=np.int16)
                source_map = np.array(f["manual_override/override_source"])

    r1, r2 = max(0, req.row_start), min(grid_shape[0], req.row_end)
    c1, c2 = max(0, req.col_start), min(grid_shape[1], req.col_end)
    # Clamp to non-negative span — if the user sent reversed bounds
    # (row_start > row_end), the slice is empty but the previous code
    # multiplied two negative numbers and reported a positive count of
    # "applied" overrides that didn't actually happen.
    rows_applied = max(0, r2 - r1)
    cols_applied = max(0, c2 - c1)
    override_map[r1:r2, c1:c2] = req.phase_idx
    source_map[r1:r2, c1:c2] = 2  # region
    cw.write_manual_override(override_map, source_map)
    n = rows_applied * cols_applied
    return {"status": "ok", "overrides_applied": n}


# ============================================================================
# Joint R+PC refinement endpoints (Phase B).
#
# Spec: docs/superpowers/specs/2026-05-14-joint-r-pc-refinement-design.md
# These talk to backend.api.services.refinement.compute_full_refinement and
# operate on the in-memory indexing-result registry (not multiphase .h5 files).
# ============================================================================
from backend.api.routes.indexing import _result_registry  # noqa: E402
from backend.api.services import refinement as _svc  # noqa: E402


class ComputeRequest(BaseModel):
    result_id: str
    smoothness_lambda: float = 0.01
    max_outer_iter: int = 3
    force_full_recompute: bool = False


class ResmoothRequest(BaseModel):
    result_id: str
    smoothness_lambda: float


def _estimate_seconds(result, has_cache: bool) -> int:
    H, W = result.original_shape
    indexed = int((H * W) * 0.95)
    if has_cache:
        return max(30, int(indexed * 0.5 / 1000))
    return max(30, int(indexed * 2.0 / 1000))


@router.post("/compute")
def compute(req: ComputeRequest):
    result = _result_registry.get(req.result_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"no indexing result for id '{req.result_id}'"},
        )
    if not (req.smoothness_lambda >= 0 and req.smoothness_lambda < float("inf")):
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "smoothness_lambda must be >= 0 and finite "
                    f"(got {req.smoothness_lambda})"
                )
            },
        )
    method = (result.metadata or {}).get("indexing_method")
    if method not in ("spherical", "dictionary"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": (
                    f"refinement requires spherical or dictionary indexing "
                    f"(got {method!r})"
                )
            },
        )
    if not (result.metadata or {}).get("sht_paths_by_phase"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": (
                    "result.metadata['sht_paths_by_phase'] is empty -- "
                    "re-run indexing"
                )
            },
        )

    running = _svc.get_active_refinement_job(req.result_id)
    if running is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "job already running for this result",
                "running_job_id": running.job_id,
            },
        )

    job = _svc.RefinementJob(
        result_id=req.result_id, smoothness_lambda=req.smoothness_lambda
    )
    has_cache = (
        (result.metadata or {}).get("_refinement_stage1_cache") is not None
        and not req.force_full_recompute
    )

    def _worker():
        try:
            _svc.compute_full_refinement(
                result,
                smoothness_lambda=req.smoothness_lambda,
                progress_callback=job.set_progress,
                cancel_event=job.cancel_event,
                max_outer_iter=req.max_outer_iter,
                force_full_recompute=req.force_full_recompute,
            )
            if _result_registry.get(req.result_id) is not result:
                job.mark_failed("result was replaced during compute")
                return
            if job.cancel_event.is_set():
                # Cancel-vs-completed race: orchestrator returned normally but
                # a cancel landed before we mark the job. Drop the just-written
                # refinement metadata so the registry reflects the cancel.
                try:
                    (result.metadata or {}).pop("refinement", None)
                    (result.metadata or {}).pop("_refinement_stage1_cache", None)
                except Exception:
                    pass
                job.state = "cancelled"
                return
            job.mark_completed()
        except _svc._RefinementCancelledError:
            job.state = "cancelled"
        except Exception as e:
            logger.exception("refinement compute job failed")
            job.mark_failed(str(e))

    t = threading.Thread(
        target=_worker, daemon=True, name=f"refinement-{job.job_id[:8]}"
    )
    job.thread = t
    t.start()

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.job_id,
            "estimated_seconds": _estimate_seconds(result, has_cache),
            "stage_1_cache_reused": has_cache,
        },
    )


@router.post("/resmooth")
def resmooth(req: ResmoothRequest):
    result = _result_registry.get(req.result_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"no indexing result for id '{req.result_id}'"},
        )
    cache = (result.metadata or {}).get("_refinement_stage1_cache")
    if cache is None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": (
                    "Stage-1 cache not available -- use POST /compute with "
                    "force_full_recompute=true"
                )
            },
        )
    # Reuse compute path with force_full_recompute=False -- orchestrator skips Stage 1.
    return compute(
        ComputeRequest(
            result_id=req.result_id,
            smoothness_lambda=req.smoothness_lambda,
            force_full_recompute=False,
        )
    )


@router.post("/cancel")
def cancel(job_id: str):
    job = _svc.JOB_REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail={"error": f"no job with id {job_id}"}
        )
    job.cancel()
    return {"state": "cancelled"}


@router.get("/progress")
def progress(job_id: str):
    job = _svc.JOB_REGISTRY.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail={"error": f"no job with id {job_id}"}
        )
    return job.to_dict()


@router.get("/summary")
def summary(result_id: str):
    result = _result_registry.get(result_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"no indexing result for id '{result_id}'"},
        )
    block = (result.metadata or {}).get("refinement")
    if block is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "refinement not yet computed for this result"},
        )
    return {
        "computed": True,
        "refined_result_id": block["refined_result_id"],
        "smoothness_lambda": block["smoothness_lambda"],
        "computed_at": block["computed_at"],
        "stage_timings": {
            "stage_1": block["stage_1_seconds"],
            "stage_2": block["stage_2_seconds"],
            "stage_3": block["stage_3_seconds"],
        },
        "outer_iterations": block["outer_iterations"],
        "summary": block["summary"],
    }
