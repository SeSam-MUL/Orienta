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

    ``__str__``/``__repr__`` return the same bracketed phrase as
    ``format_field`` does for this sentinel. ``string.Formatter.convert_field``
    runs *before* ``format_field`` and calls ``str()``/``repr()`` directly for
    a ``{x!s}``/``{x!r}`` conversion, which would otherwise bypass the
    sentinel check in ``format_field`` entirely and print
    ``<...Missing object at 0x...>`` — defining these here means the honesty
    guarantee holds for every syntax form that can reach a value, not just
    the plain ``{x}`` one a template happens to use today.
    """

    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        self.key = key

    def __str__(self) -> str:
        return f"[{self.key} not recorded]"

    def __repr__(self) -> str:
        return self.__str__()


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
            # class exists to prevent. str(value) is _Missing.__str__, the
            # same phrase a !s/!r conversion would already have produced.
            return str(value)
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
        """A scalar as it reads inside a list, or as a dict-value fallback.

        Deliberately no type-based heuristic (e.g. "a float in [0, 1] must be
        a fraction, so show it as a percentage") — a formatter cannot know
        whether a given dict's floats are fractions, radians, or counts, and
        guessing wrong would silently misstate a number in text destined for
        a manuscript. A float renders the same bare way everywhere: here,
        at the top level, and inside a list.
        """
        if isinstance(value, bool):
            return "yes" if value else "no"
        if value is None:
            return "[value not recorded]"
        if isinstance(value, float):
            return self._trim_float(value)
        if isinstance(value, dict):
            return self._render_dict(value)
        if isinstance(value, (list, tuple)):
            return self._render_sequence(value)
        return str(value)

    def _render_dict(self, value: dict) -> str:
        """``{"Al": 0.0, "Si": 0.75}`` -> ``"Al 0, Si 0.75"``.

        Bare numbers, not percentages: 0-1 is the scale the API itself uses
        for e.g. ``strength_by_phase``, and "%" would be this formatter's
        own invention rather than something the data says. A missing/None
        entry is named in the same bracketed form as a missing top-level
        param (``"[key not recorded]"``), not a second, unbracketed phrasing
        for the same condition.
        """
        parts = []
        for k, v in value.items():
            if v is None:
                parts.append(f"[{k} not recorded]")
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
