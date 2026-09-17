"""Render CSL-JSON entries to the formats a person actually pastes.

CSL-JSON is the source format because Zotero and pandoc already speak it, so
entries can be exported from a reference manager instead of typed. BibTeX is
RENDERED from it and never parsed — parsing BibTeX would need a dependency,
rendering it needs this file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_LIBRARY = Path(__file__).with_name("library.json")

# CSL type -> BibTeX entry type. Anything unlisted falls back to @misc, which
# is the honest choice: it renders without claiming a structure we do not know.
_BIBTEX_TYPE = {
    "article-journal": "article",
    "article": "article",
    "book": "book",
    "chapter": "incollection",
    "paper-conference": "inproceedings",
    "report": "techreport",
    "software": "misc",
    "dataset": "misc",
    "webpage": "misc",
}


def load_library(path: Optional[Path] = None) -> Dict[str, dict]:
    """Load a CSL-JSON library, keyed by entry id.

    The default library is merged with Orienta's own entry, which is generated
    from CITATION.cff rather than stored here — one source for the DOI. An
    explicit ``path`` (tests) skips the merge, so a fixture library is exactly
    what its file says.
    """
    src = Path(path) if path is not None else _DEFAULT_LIBRARY
    with open(src, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):           # CSL-JSON exports are often arrays
        library = {e["id"]: e for e in data}
    else:
        library = dict(data)
    if path is None:
        from .self_entry import orienta_entry_from_cff

        entry = orienta_entry_from_cff()
        library[entry["id"]] = entry
    return library


def _year(entry: dict) -> Optional[str]:
    parts = (entry.get("issued") or {}).get("date-parts") or []
    if parts and parts[0]:
        return str(parts[0][0])
    return None


def _authors(entry: dict) -> List[dict]:
    return entry.get("author") or []


# ONE pass, via translate: chained .replace() would re-escape the braces that
# the backslash replacement itself introduces, turning \textbackslash{} into
# \textbackslash\{\}. translate never rescans its own output.
_BIBTEX_ESCAPES = {
    ord("\\"): "\textbackslash{}",
    ord("{"): "\{",
    ord("}"): "\}",
}


def _escape(value: Any) -> str:
    """Escape for a BibTeX braced field."""
    return str(value).translate(_BIBTEX_ESCAPES)


def render_bibtex(entries: List[dict]) -> str:
    """Render entries as BibTeX, in the order given."""
    blocks = []
    for entry in entries:
        kind = _BIBTEX_TYPE.get(entry.get("type", ""), "misc")
        fields: List[tuple] = []

        if entry.get("title"):
            fields.append(("title", entry["title"]))
        authors = _authors(entry)
        if authors:
            names = " and ".join(
                ", ".join(p for p in (a.get("family"), a.get("given")) if p)
                or a.get("literal", "")
                for a in authors
            )
            fields.append(("author", names))
        year = _year(entry)
        if year:
            fields.append(("year", year))
        if entry.get("container-title"):
            fields.append(("journal", entry["container-title"]))
        for csl, bib in (("volume", "volume"), ("page", "pages"),
                         ("publisher", "publisher"), ("DOI", "doi"),
                         ("URL", "url")):
            if entry.get(csl):
                fields.append((bib, entry[csl]))
        if entry.get("version"):
            fields.append(("note", f"Version {entry['version']}"))

        body = ",\n".join(f"  {k} = {{{_escape(v)}}}" for k, v in fields)
        blocks.append(f"@{kind}{{{entry['id']},\n{body}\n}}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_plain(entries: List[dict]) -> str:
    """Author, year, title, journal, DOI as readable lines."""
    lines = []
    for entry in entries:
        authors = _authors(entry)
        if authors:
            who = ", ".join(
                f"{a.get('family', '')}, {(a.get('given') or '')[:1]}."
                if a.get("family") else a.get("literal", "")
                for a in authors
            )
        else:
            who = entry.get("publisher") or "Unknown author"
        bits = [f"{who} ({_year(entry) or 'n.d.'})", entry.get("title", "")]
        if entry.get("container-title"):
            journal = entry["container-title"]
            if entry.get("volume"):
                journal += f" {entry['volume']}"
            if entry.get("page"):
                journal += f", {entry['page']}"
            bits.append(journal)
        if entry.get("version"):
            bits.append(f"version {entry['version']}")
        if entry.get("DOI"):
            bits.append(f"https://doi.org/{entry['DOI']}")
        elif entry.get("URL"):
            bits.append(entry["URL"])
        lines.append(". ".join(b for b in bits if b))
    return "\n".join(lines) + ("\n" if lines else "")


class _NamingDict(dict):
    """format_map source that names what is missing instead of guessing it.

    Honesty rule 2: a methods paragraph that silently prints a default value
    describes a run that never happened.
    """

    def __missing__(self, key):  # noqa: D105
        return f"[{key} not recorded]"


def render_methods(steps: List[dict]) -> str:
    """Join the sentence of each step that ran, in order, with real params."""
    from .steps import get_step

    sentences = []
    for entry in steps:
        key = entry.get("key", "")
        params = entry.get("params") or {}
        step = get_step(key)
        if step is None:
            sentences.append(
                f"A step recorded as '{key}' ran; no citation declared for it."
            )
            continue
        sentences.append(step.sentence.format_map(_NamingDict(params)))
    return " ".join(sentences)
