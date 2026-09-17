"""Orienta's own citation, taken from the one file that already states it.

CITATION.cff is what GitHub and Zenodo read, so it is the authority for
Orienta's DOI and authors. Copying either into a second file would create a
place for them to disagree — and that has already happened once, between the
dev and beta trees. Generating means each tree is honest about itself.

The VERSION is deliberately not taken from here: CITATION.cff records the last
archived release, while the running build may be anything. app_version owns
that question.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Set

import yaml

# backend/api/services/citations/self_entry.py -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CFF = _PROJECT_ROOT / "CITATION.cff"

_DOI_RE = re.compile(r"10\.5281/zenodo\.\d+")


def stated_dois(text: str) -> Set[str]:
    """Every Zenodo DOI a document states."""
    return set(_DOI_RE.findall(text))


def orienta_entry_from_cff(cff_path: Optional[Path] = None) -> dict:
    """Build the CSL-JSON entry for Orienta itself from CITATION.cff."""
    path = Path(cff_path) if cff_path is not None else _DEFAULT_CFF
    if not path.exists():
        raise FileNotFoundError(f"CITATION.cff not found at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cff = yaml.safe_load(fh) or {}

    entry = {
        "id": "orienta",
        "type": "software",
        "title": cff.get("title", "Orienta"),
    }

    authors = []
    for person in cff.get("authors") or []:
        name = {}
        if person.get("family-names"):
            name["family"] = person["family-names"]
        if person.get("given-names"):
            name["given"] = person["given-names"]
        if not name and person.get("name"):
            name["literal"] = person["name"]
        if name:
            authors.append(name)
    if authors:
        entry["author"] = authors

    released = str(cff.get("date-released") or "")
    if released[:4].isdigit():
        entry["issued"] = {"date-parts": [[int(released[:4])]]}

    if cff.get("doi"):
        entry["DOI"] = cff["doi"]
    if cff.get("url"):
        entry["URL"] = cff["url"]
    if cff.get("repository-code"):
        entry["URL"] = cff["repository-code"]
    entry.setdefault("publisher", "Zenodo")
    return entry
