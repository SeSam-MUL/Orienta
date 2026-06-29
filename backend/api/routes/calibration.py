"""
Calibration API Routes

Exposes the CalibrationStore for inspection and manual PC management.
Useful for debugging, batch workflows, and future frontend integration.
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.services.calibration_store import calibration_store

logger = logging.getLogger(__name__)
router = APIRouter()


class SetPCRequest(BaseModel):
    pc: List[float]  # [pcx, pcy, pcz]


@router.get("/")
async def list_calibrations():
    """List all calibration entries in the store."""
    entries = calibration_store.get_all()
    return {
        "count": len(entries),
        "entries": {name: entry.to_dict() for name, entry in entries.items()},
    }


@router.get("/{dataset}")
async def get_calibration(dataset: str):
    """Get calibration entry for a specific dataset."""
    entry = calibration_store.get_entry(dataset)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No calibration for '{dataset}'")
    return entry.to_dict()


@router.post("/{dataset}/pc")
async def set_pc(dataset: str, req: SetPCRequest):
    """Manually set PC for a dataset."""
    if len(req.pc) != 3:
        raise HTTPException(status_code=400, detail="PC must have exactly 3 values [pcx, pcy, pcz]")

    if not calibration_store.update_pc(dataset, req.pc, source="manual"):
        raise HTTPException(status_code=404, detail=f"No calibration entry for '{dataset}'")

    return {"success": True, "dataset": dataset, "pc": req.pc, "source": "manual"}


@router.delete("/{dataset}")
async def delete_calibration(dataset: str):
    """Remove calibration entry for a dataset."""
    if not calibration_store.remove(dataset):
        raise HTTPException(status_code=404, detail=f"No calibration for '{dataset}'")
    return {"success": True, "deleted": dataset}


@router.post("/{dataset}/propagate-to-parent")
async def propagate_pc_to_parent(dataset: str):
    """Propagate the refined PC from a derived dataset to its parent.

    Typical workflow: Load → Deepcopy → PC Refinement → Propagate back to original.
    """
    entry = calibration_store.get_entry(dataset)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No calibration for '{dataset}'")

    if not entry.parent_name:
        raise HTTPException(status_code=400, detail=f"'{dataset}' has no parent dataset")

    parent = calibration_store.get_entry(entry.parent_name)
    if parent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Parent '{entry.parent_name}' is no longer loaded",
        )

    calibration_store.update_pc(entry.parent_name, entry.pc_single, source="propagated")
    pc_list = list(float(v) for v in entry.pc_single)
    logger.info("Propagated PC %s from '%s' to parent '%s'", pc_list, dataset, entry.parent_name)

    return {
        "success": True,
        "parent": entry.parent_name,
        "pc": pc_list,
    }
