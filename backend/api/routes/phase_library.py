"""The phase library's index endpoint (spec §3.1).

ONE route, ONE response: the phases and the library totals together.
Splitting them into two requests lets a number arrive before the list it
describes, and a header reading "36 phases" above an empty table is a worse lie
than a spinner (§2.8). The sidebar's own counters are not here: they narrow with
the active filter and belong to the page (`facets.js`).

The work is in `backend.api.services.phase_library`; this file is the seam.
"""
from __future__ import annotations

import logging

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.api.services.phase_library import build_index_document

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/index")
def phase_library_index():
    """Every phase in the local library, with the totals over that list.

    Reads file metadata by `stat` and answers the master question with an h5py
    key lookup, so the 44 MB `mLPNH` array is never touched.

    A failure is a 500 with the reason, not an empty list: the page's loader
    distinguishes "the backend answered, and the library is empty" from "we
    could not ask", and returning `{"phases": []}` on an error would collapse
    the two -- the shape that had the Indexing page saying "No phases found"
    for a library of 36.
    """
    try:
        return build_index_document()
    except Exception as exc:
        logger.exception("phase library index failed")
        raise HTTPException(status_code=500,
                            detail=f"Could not read the phase library: {exc}")


@router.get("/phase")
def phase_library_detail(key: str = Query(..., min_length=1)):
    """One phase's card (spec §2.3), keyed on the library stem.

    THE KEY IS A QUERY PARAMETER, NOT A PATH SEGMENT. Library keys contain
    spaces, dots, parentheses and Greek (`Al4Fe1.7Si (τ11)`, `α-(AlMnSi)`), and
    the collections routes already carry a structural test forbidding `{name}`
    segments: one reintroduced there shadowed a literal route depending on
    declaration order, and the symptom was a 404 naming a collection nobody
    asked for.

    Adds to the index row the two things too expensive to send for all 36:
    the master's namelist (read from its HEADER -- `mLPNH` is 44 MB and is never
    opened) and the `.sht`'s provenance, crystallography and parameters. Plus
    `deep_link`, so "show me this in the Database browser" does not need the page
    to parse a path.

    404 when the library has no such key: a card for a phase that does not exist
    is not an empty card, it is a wrong question, and answering `{}` would have
    the page render a blank phase rather than say so.
    """
    from backend.api.services.phase_library import build_phase_detail

    try:
        detail = build_phase_detail(key)
    except Exception as exc:
        logger.exception("phase library detail failed for %r", key)
        raise HTTPException(status_code=500,
                            detail=f"Could not read the phase: {exc}")
    if detail is None:
        raise HTTPException(status_code=404,
                            detail=f"No phase {key!r} in the local library")
    return detail


class NamesRequest(BaseModel):
    """A display name and/or search terms for one phase (spec §2.4).

    `null` MEANS "LEAVE IT ALONE"; `""` CLEARS. Both fields are partial updates,
    so a caller can set a display name without touching the search terms and the
    other way round. Pinned by `tests/test_phase_synonyms.py:117`.

    Spelled out here because f7's closing review found the frontend's `NameEditor`
    sending `display_name: null` to DELETE a name — which silently did nothing,
    the one outcome a user reads as "the app lost my edit". The convention is
    right; it just was not written where the next reader looks.
    """

    key: str
    #: `None` leaves the stored display name; `""` removes it.
    display_name: Optional[str] = None
    #: `None` leaves the stored terms; `[]` removes them all.
    search_terms: Optional[list[str]] = None
    author: str = ""


@router.put("/names")
def phase_library_set_names(req: NamesRequest):
    """Name a phase, or clear its names by sending them EMPTY — not null.

    `null` leaves a field as it is, `""` (or `[]`) clears it. See
    `NamesRequest`: these are partial updates, so omitting a field must not
    destroy it, and that is why "delete this name" has to say `""`.

    `author` is recorded and shown on the card. In a shared library a name
    without an author is a silent edit to someone else's work, so an empty one is
    stored as "unknown" rather than inherited from whoever wrote the last name.

    A display name stands in FRONT OF the key, never instead of it: the key is
    still in every payload and still searchable.
    """
    from backend.api.services.phase_synonyms import set_names
    from backend.api.services.crystal_hint_local_library import get_index

    if req.key not in get_index():
        raise HTTPException(status_code=404,
                            detail=f"No phase {req.key!r} in the local library")
    try:
        entry = set_names(req.key, display_name=req.display_name,
                          search_terms=req.search_terms, author=req.author)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        # The store lives in the library folder, which may be read-only or on a
        # disconnected share. Say that, rather than reporting success for a name
        # that was never written.
        logger.exception("could not write the synonym store")
        raise HTTPException(status_code=500,
                            detail=f"Could not save the name: {exc}")
    return {"key": req.key, **entry}
