"""Pole-figure endpoint — renders the active indexing result's orientations."""
from __future__ import annotations

import base64
import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from backend.api.routes.indexing import get_last_indexing_result
from backend.api.services import pole_figure as pf
from backend.api.services import reference_frame_state as rfs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["pole-figure"])


def _parse_hkl(s: str | None) -> list[tuple[int, int, int]]:
    if not s:
        return [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    fams = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # accept "111" or "1 1 1" or "1,1,1" (already split on comma → "111"/"1 1 1")
        digits = tok.replace(" ", "")
        if len(digits) == 3 and all(c in "0123456789-" for c in digits.replace("-", "", 3)):
            # crude: only single-digit indices via compact form
            fams.append(tuple(int(c) for c in digits))
        else:
            fams.append(tuple(int(p) for p in tok.split()))
    return fams


@router.get("/pole-figure")
def pole_figure(phase_id: int = Query(...),
                hkl: str | None = Query(None),
                mode: str = Query("both"),
                subsample: int = Query(20000, ge=500, le=200000)) -> dict:
    res = get_last_indexing_result()
    if res is None:
        raise HTTPException(status_code=404, detail="No active indexing result.")
    spec = rfs.get_frame((res.metadata or {}).get("source_file"))
    r_user = rfs.resolve_r_user(spec)
    try:
        families = _parse_hkl(hkl)
        png = pf.compute_pole_figure(res.xmap, phase_id, families, r_user,
                                     spec["plot"], mode=mode, subsample=subsample)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    phase = res.xmap.phases[phase_id]
    return {
        "image": base64.b64encode(png).decode("ascii"),
        "phase_id": phase_id,
        "phase_name": phase.name or f"phase_{phase_id}",
        "frame_sig": rfs.frame_signature(spec),
        "n_pixels": int((res.xmap.phase_id == phase_id).sum()),
    }
