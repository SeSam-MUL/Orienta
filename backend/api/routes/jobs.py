"""/api/jobs: compute-worker jobs, watched and cancelled over HTTP."""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.api.problem import problem

router = APIRouter()

#: Jobs the interface may start directly. The sphere is NOT one of them: its
#: route keeps its own contract and runs it.
HTTP_JOBS: frozenset = frozenset()


class SubmitRequest(BaseModel):
    job: str
    args: dict = Field(default_factory=dict)


def _allowed(name: str) -> bool:
    if name in HTTP_JOBS:
        return True
    return os.environ.get("ORIENTA_WORKER_TEST_JOBS", "").strip() == "1" and name.startswith("_")


def _manager():
    from backend.worker import manager as wm
    return wm.get_manager()


def _not_found(job_id: str):
    return problem(404, "jobNotFound", f"No job with id {job_id}.")


@router.post("")
async def submit(req: SubmitRequest):
    from backend.worker import manager as wm
    if not wm.worker_enabled():
        raise problem(503, "workerDisabled", "The compute process is switched off (ORIENTA_NO_WORKER=1).")
    if not _allowed(req.job):
        raise problem(400, "unknownJob", f"Unknown job {req.job!r}.")
    try:
        job_id = _manager().submit(req.job, req.args)
    except wm.UnknownJob:
        raise problem(400, "unknownJob", f"Unknown job {req.job!r}.")
    return {"id": job_id}


@router.get("")
async def active(job: Optional[str] = None):
    return {"jobs": _manager().active(job)}


@router.get("/{job_id}")
async def get_job(job_id: str):
    snap = _manager().get(job_id)
    if snap is None:
        raise _not_found(job_id)
    return snap


@router.get("/{job_id}/wait")
async def wait_job(job_id: str, timeout: float = 30.0):
    m = _manager()
    if m.get(job_id) is None:
        raise _not_found(job_id)
    snap = await asyncio.to_thread(m.wait, job_id, min(max(timeout, 0.0), 30.0))
    return snap


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    m = _manager()
    if m.get(job_id) is None:
        raise _not_found(job_id)
    return {"cancelled": m.cancel(job_id)}
