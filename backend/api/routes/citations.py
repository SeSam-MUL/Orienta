"""What to cite for a given result, and for the grains of the Analysis page."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.services.citations.bundle import citation_bundle
from backend.api.services.citations.provenance import get_steps

router = APIRouter()


@router.get("/result/{result_id}")
def citations_for_result(result_id: str):
    from backend.api.routes.indexing import _result_registry

    result = _result_registry.get(result_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No result '{result_id}'. Results live in memory only: they "
                "are evicted once 12 newer ones exist, and a backend restart "
                "clears all of them. Load or re-run it first."
            ),
        )
    return citation_bundle(get_steps(result))


@router.get("/analysis")
def citations_for_analysis():
    """The same answer as ``/result/{id}``, for the grains reconstructed on the Analysis page.

    The map loaded there has no ``result_id``, so its one citable step (``analysis.grain_reconstruction``)
    is built from the parameters the current grains were reconstructed with and the grain-size
    statistics settings in force. 404 while no grains exist.
    """
    from backend.api.routes.analysis import grain_citation_step

    step = grain_citation_step()
    if step is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No grains have been reconstructed for the map on the Analysis page. "
                "Reconstruct the grains first."
            ),
        )
    return citation_bundle([step])
