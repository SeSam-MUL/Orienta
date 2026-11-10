"""What to cite for a list of recorded steps, in the shape the citation endpoints answer with.

One builder for every place that shows or writes citations: ``GET /api/citations/result/{id}`` (an
indexing result's trail), ``GET /api/citations/analysis`` (the grains of the Analysis page) and the
"Citations" sheet of the Excel export. They all list the same entries in the same order for the same
steps: the application and the stack under it first, then each step's entries in the order the steps ran.
"""
from __future__ import annotations

import logging
from typing import List

from backend.api.services.app_version import get_version_info
from backend.api.services.citations.render import (
    load_library,
    render_bibtex,
    render_methods,
    render_plain,
)
from backend.api.services.citations.steps import get_step

logger = logging.getLogger(__name__)

# Always cited, whatever ran: the application and the stack under it.
ALWAYS_CITED = ("orienta", "kikuchipy", "orix")


def citation_bundle(steps: List[dict]) -> dict:
    """``bibtex``, ``methods``, ``plain``, ``steps``, ``undeclared``, ``warnings`` and ``app_version``
    for ``steps`` (``[{"key", "params"}]``)."""
    warnings: List[str] = []
    try:
        library = load_library()
    except Exception:
        logger.exception(
            "Failed to load the citation library (default library.json "
            "merged with CITATION.cff); continuing without it."
        )
        library = {}
        warnings.append(
            "The bibliography could not be loaded, so BibTeX and the plain "
            "reference list are unavailable. The methods description below "
            "is unaffected."
        )

    ordered: List[str] = []
    undeclared: List[str] = []
    for cid in ALWAYS_CITED:
        if cid in library and cid not in ordered:
            ordered.append(cid)
    for entry in steps:
        step = get_step(entry.get("key", ""))
        if step is None:
            # By key, once. A step appended twice to the same trail (the
            # sites that mutate an already-stored result do that) listed its
            # key twice here, and the panel keys its <li> on the key — a
            # duplicate React key, and a list that reads like two different
            # unknown steps ran. render_methods dedupes the same way.
            key = entry.get("key", "")
            if key not in undeclared:
                undeclared.append(key)
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
        "warnings": warnings,
        "app_version": get_version_info(),
    }
