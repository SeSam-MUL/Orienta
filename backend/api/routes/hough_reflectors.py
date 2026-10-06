"""Reflector families of a Hough phase, by CIF path (Indexing page).

``GET  /api/indexing/hough/reflectors?cif_path=``  the family table
``PUT  /api/indexing/hough/reflectors``            change the choice (null = default)
``POST /api/indexing/hough/reflectors/validate``   check one hand-typed family
``GET  /api/indexing/hough/reflectors/cost``       what the choice would cost

The PC Refinement page has the same four under ``/api/pc/phase/reflectors``,
for the phase loaded there.  Both go through ``hough_reflector_service`` and the
one registry in ``hough_reflectors``.
"""
import asyncio
import logging
from typing import List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.services import hough_reflector_service as svc

logger = logging.getLogger(__name__)
router = APIRouter()


class ReflectorChangeRequest(BaseModel):
    cif_path: str
    # null = back to the default construction.
    spec: Optional[dict] = None


class ReflectorValidateRequest(BaseModel):
    cif_path: str
    hkl: Union[str, List[float]]
    # The choice the page is working on, so a family it already holds is named.
    spec: Optional[dict] = None


def raise_http(exc: svc.ReflectorError):
    raise HTTPException(status_code=exc.status, detail=exc.as_detail())


@router.get("/reflectors")
async def get_reflectors(cif_path: str):
    """The families of this phase: which the default list holds, which the
    current choice holds, which PyEBSDIndex keeps."""
    def work():
        phase = svc.phase_for_path(cif_path)
        return svc.table(phase, cif_path)

    try:
        return await asyncio.to_thread(work)
    except svc.ReflectorError as exc:
        raise_http(exc)


@router.put("/reflectors")
async def put_reflectors(req: ReflectorChangeRequest):
    """Change the choice for this phase; every Hough build of it follows."""
    def work():
        phase = svc.phase_for_path(req.cif_path)
        return svc.change(phase, req.cif_path, req.spec)

    try:
        return await asyncio.to_thread(work)
    except svc.ReflectorError as exc:
        raise_http(exc)


@router.post("/reflectors/validate")
async def validate_reflector(req: ReflectorValidateRequest):
    """One hand-typed family ({hkl}, or {hkil} for a hexagonal phase)."""
    def work():
        phase = svc.phase_for_path(req.cif_path)
        return svc.check_family(phase, req.cif_path, req.hkl, req.spec)

    try:
        return await asyncio.to_thread(work)
    except svc.ReflectorError as exc:
        raise_http(exc)


@router.get("/reflectors/cost")
async def reflector_cost(cif_path: str, n_bands: int = 12):
    """Memory the band-triplet library of the stored choice needs."""
    def work():
        phase = svc.phase_for_path(cif_path)
        try:
            return svc.cost(phase, cif_path, n_bands)
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None)
            if code:
                raise svc.ReflectorError(code, str(exc), **getattr(exc, "params", {})) from exc
            raise

    try:
        return await asyncio.to_thread(work)
    except svc.ReflectorError as exc:
        raise_http(exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reflector cost failed for %s", cif_path, exc_info=True)
        raise HTTPException(status_code=400, detail={
            "code": "cost_failed", "message": str(exc), "params": {}})
