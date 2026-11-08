"""What the application knows about scan geometry, for the interface to show.

``GET /hex-limits``  the list of what a hexagonal scan cannot do yet, grouped by when in the work
                     it is met (before indexing, after indexing). It is built from the table
                     that enforces the limits (``hex_guard.GUARDED``), so the text the interface
                     shows cannot go stale.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.services import hex_guard

router = APIRouter()


@router.get("/hex-limits")
async def hex_limits():
    """``{"groups": {"before_indexing": [...], "after_indexing": [...]},
    "categories": [...], "not_limited": [{"key", "what"}]}``.

    An item of a group is one guarded endpoint: ``{"key", "category", "method", "path", "what",
    "code", "message", "partial"}`` (``partial``: only some requests to it are refused, e.g. one
    layer or a cleanup).

    ``key`` is a stable, unique snake_case id: the interface translates the limit by it (``what``
    is only a fragment of the English refusal sentence). Routes that are one limit to the user share
    a key: the three crystal-hint routes are ``crystal_hint_averaging``, and the dialog groups by
    key. ``category`` is one of ``categories``, the groups the interface sorts the limits into.
    ``code`` names the sentence shape of ``message`` (the English sentence).

    For a ``partial`` item, ``affects`` lists the kinds of thing that are refused (no entry sets
    it today) and ``parameters`` the request parameters that trigger the refusal (the crystal
    hint's averaging radius), both read from the constants the gate itself uses, so the lists
    cannot drift from it; they are empty lists where they do not apply.

    ``not_limited`` names what works on a hexagonal scan although it might be expected not to:
    image export (PNG) and global pattern-centre refinement are NOT limited."""
    return hex_guard.limits()
