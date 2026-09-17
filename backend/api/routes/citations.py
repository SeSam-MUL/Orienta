"""What to cite for a given result."""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException

from backend.api.services.app_version import get_version_info
from backend.api.services.citations.provenance import get_steps
from backend.api.services.citations.render import (
    load_library,
    render_bibtex,
    render_methods,
    render_plain,
)
from backend.api.services.citations.steps import get_step

logger = logging.getLogger(__name__)
router = APIRouter()

# Always cited, whatever ran: the application and the stack under it.
_ALWAYS = ("orienta", "kikuchipy", "orix")


@router.get("/result/{result_id}")
def citations_for_result(result_id: str):
    from backend.api.routes.indexing import _result_registry

    result = _result_registry.get(result_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No result '{result_id}'. Load or re-run it first.",
        )

    library = load_library()
    steps = get_steps(result)

    ordered: List[str] = []
    undeclared: List[str] = []
    for cid in _ALWAYS:
        if cid in library and cid not in ordered:
            ordered.append(cid)
    for entry in steps:
        step = get_step(entry.get("key", ""))
        if step is None:
            undeclared.append(entry.get("key", ""))
            continue
        for cid in step.citation_ids:
            if cid in library and cid not in ordered:
                ordered.append(cid)

    entries = [library[cid] for cid in ordered]
    return {
        "bibtex": render_bibtex(entries),
        "methods": render_methods(steps),
        "plain": render_plain(entries),
        "steps": steps,
        "undeclared": undeclared,
        "app_version": get_version_info(),
    }
