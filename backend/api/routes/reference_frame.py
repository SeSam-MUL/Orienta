"""Coordinate-system (reference-frame) + state-version endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.services import reference_frame_state as rfs
from backend.api.services import state_version

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["frame"])


class FrameBody(BaseModel):
    spec: dict


@router.get("/state-version")
def get_state_version() -> dict:
    active_result_id = None
    try:
        from backend.api.routes.indexing import _active_result_id
        active_result_id = _active_result_id
    except Exception:  # noqa: BLE001
        pass
    return {
        "version": state_version.get(),
        "active_result_id": active_result_id,
        "active_source_file": rfs.get_active_source_file(),
    }


@router.get("/frame")
def get_frame() -> dict:
    sf = rfs.get_active_source_file()
    spec = rfs.get_frame(sf)
    return {"source_file": sf, "spec": spec, "frame_sig": rfs.frame_signature(spec)}


@router.put("/frame")
def put_frame(body: FrameBody) -> dict:
    sf = rfs.get_active_source_file()
    if not sf:
        raise HTTPException(status_code=400, detail="No active dataset to attach a coordinate system to.")
    try:
        spec = rfs.set_frame(sf, body.spec)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    version = state_version.bump()
    return {"source_file": sf, "spec": spec,
            "frame_sig": rfs.frame_signature(spec), "version": version}
