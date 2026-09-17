"""Render CSL-JSON entries to the formats a person actually pastes.

CSL-JSON is the source format because Zotero and pandoc already speak it, so
entries can be exported from a reference manager instead of typed. BibTeX is
RENDERED from it and never parsed — parsing BibTeX would need a dependency,
rendering it needs this file.
"""
from __future__ import annotations

import json
import string
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


class _Missing:
    """Sentinel for a step param that was never recorded.

    Carries the key itself, because ``string.Formatter.format_field`` is
    handed only the value and the format spec, never the field name — so if
    "which key was missing" is going to reach the output, the value has to
    remember it.
    """

    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        self.key = key


class _MethodsFormatter(string.Formatter):
    """Turns recorded step params into prose, not Python's repr/str.

    Two defects, one formatter. (1) A missing param must be named even under
    a format spec: ``"{ncc:.2f}".format_map(a_dict_with___missing__)`` still
    raises ``ValueError`` — ``__missing__`` supplies a *value*, and
    ``str.format`` then tries to apply the float spec to it. Missingness is
    therefore decided in ``get_value`` and rendered in ``format_field``
    *before* ``format_spec`` is ever consulted for that value. (2) A
    composite value that IS present must still read as prose, not Python's
    ``repr()`` — a dict here is exactly the shape ``provenance.steps`` uses
    (and the ``.h5`` export and future add-ons will read), so it must stay
    machine-readable at the point it is recorded; humanising it happens here,
    at render time, once, for every call site.
    """

    def get_value(self, key, args, kwargs):
        if key not in kwargs:
            return _Missing(str(key))
        value = kwargs[key]
        # None is a recorded absence, not a recorded value — name it the
        # same way a missing key is named, instead of printing "None".
        if value is None:
            return _Missing(str(key))
        return value

    def format_field(self, value, format_spec):
        if isinstance(value, _Missing):
            # The spec is ignored on purpose: "[key not recorded]" is not a
            # float or a date, and applying ":.2f" to it is the crash this
            # class exists to prevent.
            return f"[{value.key} not recorded]"
        if isinstance(value, dict):
            return self._render_dict(value)
        if isinstance(value, (list, tuple)):
            return self._render_sequence(value)
        if isinstance(value, bool):          # bool is a subclass of int
            return "yes" if value else "no"
        if isinstance(value, float) and not format_spec:
            return self._trim_float(value)
        # No spec-less dict/list/bool/float branch matched, or the template
        # asked for an explicit spec on a plain value (e.g. "{n:.0f}") —
        # standard str.format behaviour, unchanged.
        return super().format_field(value, format_spec)

    def _render_scalar(self, value) -> str:
        """A scalar as it reads inside a list, or as a dict-value fallback."""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if value is None:
            return "not recorded"
        if isinstance(value, float):
            return self._trim_float(value)
        if isinstance(value, dict):
            return self._render_dict(value)
        if isinstance(value, (list, tuple)):
            return self._render_sequence(value)
        return str(value)

    def _render_dict(self, value: dict) -> str:
        """``{"Al": 0.0, "Si": 0.75}`` -> ``"Al 0%, Si 75%"``.

        Every dict param recorded today (``strength_by_phase``) holds a
        fraction in [0, 1] per key. A percentage reads as prose ("Si 75%");
        the bare fraction ("Si 0.75") reads as a leftover number that was
        supposed to be finished. A value outside [0, 1] (nothing today) falls
        back to a trimmed float instead of being multiplied by 100, so a
        future non-fractional dict param does not get silently mangled.
        """
        parts = []
        for k, v in value.items():
            if v is None:
                parts.append(f"{k} not recorded")
            elif isinstance(v, float) and 0.0 <= v <= 1.0:
                parts.append(f"{k} {self._as_percentage(v)}")
            else:
                parts.append(f"{k} {self._render_scalar(v)}")
        return ", ".join(parts)

    def _render_sequence(self, value) -> str:
        """``["Al", "Si", "Fe"]`` -> ``"Al, Si and Fe"``."""
        parts = [self._render_scalar(v) for v in value]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    @staticmethod
    def _trim_float(value: float) -> str:
        """Trimmed, not repr: no trailing ``.0``, capped at 3 decimals."""
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _as_percentage(value: float) -> str:
        text = f"{value * 100:.1f}".rstrip("0").rstrip(".")
        return f"{text or '0'}%"


_METHODS_FORMATTER = _MethodsFormatter()


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
        sentences.append(_METHODS_FORMATTER.vformat(step.sentence, (), params))
    return " ".join(sentences)
