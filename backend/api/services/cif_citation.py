"""Reading and writing the citation of a CIF file, in one place.

Three code paths used to read a CIF's DOI and reference text on their own
(``cif_info``, ``cif_batch_info`` and the .xtal converter), with different
regular expressions: the info paths accepted a DOI without quotes, the
converter demanded quotes, and only the info paths knew ``_journal_paper_doi``
while only the converter knew ``_citation_journal_doi``. A CIF could therefore
show a DOI in the Crystal Database and still be reported as having none when
it was converted. Everything now goes through :func:`read_citation`.

The writer side (:func:`apply_citation`) is the text edit behind
``PATCH /cif/{name}/meta`` and behind the symmetrised copy, which has to keep
the citation of the file it came from. It edits every occurrence the reader
looks at, in place, so what was written is what is read back.

The phase card, the CIF info panel, the database builder and the converter ask
``phase_library.cif_source`` and ``set_cif_source``. Both stand on this module, so
a file gives one answer everywhere.

What counts as a DOI
--------------------
A value is a DOI only if, after removing a ``https://doi.org/`` / ``doi:``
prefix, it matches ``10.<3-9 digits>/<something>`` (:func:`normalise_doi`; the
Crystal Database dialog applies the same rule). The tag alone is not enough:
``loop_ _citation_id _citation_year _citation_doi`` is followed by the value of
a different column, and a tag may be followed by ``loop_`` or by a text field
that is not a DOI.

Tags are only recognised at the start of a line and outside text fields and
comments, so ``_x_publ_section_doi`` or a DOI mentioned inside a long text is
not mistaken for the real one.
"""

from __future__ import annotations

import bisect
import html as _html
import re
from dataclasses import dataclass, field
from typing import Optional

# Tags that carry a DOI. ``_citation_journal_doi`` is the form some COD / ICSD
# exports use; ``_journal_paper_doi`` is the older core-dictionary spelling.
DOI_TAGS = (
    "_publ_section_doi",
    "_citation_doi",
    "_journal_paper_doi",
    "_citation_journal_doi",
)

_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_DOI_BODY_RE = re.compile(r"^10\.\d{3,9}/\S+$")

# A text field: the opening and closing semicolons sit in column 1.
_TEXT_FIELD_RE = re.compile(r"^;.*?^;", re.MULTILINE | re.DOTALL)

_DOI_TAG_NAMES = frozenset(DOI_TAGS)

_DOI_TAG_RE = re.compile(
    r"^[ \t]*(" + "|".join(re.escape(t) for t in DOI_TAGS) + r")(?=[ \t\r\n])",
    re.MULTILINE | re.IGNORECASE,
)

# After the tag: blank or comment-only lines may come before the text field
# (valid CIF), but the opening ``;`` itself stays in column 1. So does the closing
# one, except that hand-edited files indent it. Indented, it must stand alone on its
# line: a text line that merely starts with " ;" is text (see ``apply_citation``).
_REFERENCE_RE = re.compile(
    r"^[ \t]*_publ_section_references(?:[ \t]*(?:#[^\r\n]*)?\r?\n)+;(.*?)"
    r"\r?\n(?:;|[ \t]+;[ \t]*(?=\r?\n|\Z))",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)

# The same tag with a one-line value instead of a text field.
_REFERENCE_INLINE_RE = re.compile(
    r"^[ \t]*_publ_section_references[ \t]+(?:'([^'\r\n]*)'|\"([^\"\r\n]*)\"|([^\s;'\"][^\s]*))",
    re.MULTILINE | re.IGNORECASE,
)


#: Longest DOI or citation text accepted from a person. The text travels into the
#: .xtal and from there into the header of every SHT file, where its length is a
#: signed 16-bit integer (``doiLen``, at most 32767 bytes); 2000 characters is
#: four times a long reference and, even at 3 bytes per character, a fifth of that.
MAX_CITATION_CHARS = 2000
#: The longest DOI there is any reason to accept (real ones are well under 100).
MAX_DOI_CHARS = 200


class CitationError(ValueError):
    """A DOI or reference that is refused. ``code`` is ``invalid_doi`` or ``too_long``."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalise_doi(value: str) -> str:
    """The bare DOI in ``value`` (``10.xxxx/...``), or '' if it is not one."""
    text = _DOI_PREFIX_RE.sub("", str(value or "").strip()).strip()
    return text if len(text) <= MAX_DOI_CHARS and _DOI_BODY_RE.match(text) else ""


@dataclass
class CifCitation:
    dois: list[str] = field(default_factory=list)
    reference: str = ""
    #: True when ``reference`` was composed from ``_journal_*`` and author fields
    #: because the file has no ``_publ_section_references`` of its own.
    composed: bool = False

    @property
    def doi(self) -> str:
        return self.dois[0] if self.dois else ""

    @property
    def primary(self) -> str:
        """What goes into ``useReference`` of a .xtal: the DOI, else the text."""
        return self.doi or self.reference

    @property
    def found(self) -> bool:
        return bool(self.primary)


# The accents of a CIF text field as the IUCr and TeX write them: ``\'e``, ``\`e``,
# ``\"u``, ``\^o``, ``\~n``, ``\=a``, ``\.z`` and the letter forms ``\v{C}``, ``\c c``,
# ``\u{g}``, ``\H{o}``, ``\k{a}``, ``\r{a}``. ``\<`` (as in ``\<Cern\'y``, the caron on
# the following letter) is the older IUCr spelling of ``\v``.
_ACCENTS = {
    "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303", '"': "\u0308",
    "=": "\u0304", ".": "\u0307", "<": "\u030c", "v": "\u030c", "c": "\u0327",
    "u": "\u0306", "H": "\u030b", "k": "\u0328", "r": "\u030a",
}
_LETTERS = {
    "ss": "\u00df", "o": "\u00f8", "O": "\u00d8", "ae": "\u00e6", "AE": "\u00c6",
    "aa": "\u00e5", "AA": "\u00c5", "l": "\u0142", "L": "\u0141", "i": "\u0131",
}
# A symbol accent takes the letter that follows at once (``\'e``, ``\'{e}``, ``\'{\i}``: no
# space, so that ``a \< b`` is not an accent); a letter accent (``\v``, ``\c`` ...) takes
# a space or a brace first (``\v{C}``, ``\v C``: ``\vector`` is not one).
_ACCENT_RE = re.compile(
    r"\{?\\(?:(['`^~\"=.<])\{?|([vcuHkr])(?:\s+\{?|\{))"
    r"(?:([A-Za-z])|\\([ij])(?![A-Za-z]))\}?\}?")
_LETTER_RE = re.compile(r"\{?\\(ss|ae|AE|aa|AA|[oOlLi])(?![A-Za-z])(?:\{\}|\s)?\}?")


def _tex_to_unicode(text: str) -> str:
    r"""TeX/IUCr accents to the letters they stand for: ``Universit\'e`` becomes
    Universite with an acute on the e, ``\<Cern\'y`` becomes Cerny with a caron on
    the C and an acute on the y. Only the accent and letter commands of the table
    above are translated; anything else is left as written."""
    import unicodedata

    def accent(m):
        return unicodedata.normalize("NFC", (m.group(3) or m.group(4)) + _ACCENTS[m.group(1) or m.group(2)])

    text = _ACCENT_RE.sub(accent, text)
    return _LETTER_RE.sub(lambda m: _LETTERS[m.group(1)], text)


def _clean_text(raw: str) -> str:
    # TeX escapes first: ``\<C`` carries a ``<`` that is not the start of a tag.
    plain = _tex_to_unicode(raw)
    plain = re.sub(r"<[^>]+>", "", plain)
    plain = _html.unescape(plain)
    return re.sub(r"\s+", " ", plain).strip()


def _spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _TEXT_FIELD_RE.finditer(text)]


def _inside(spans: list[tuple[int, int]], pos: int) -> bool:
    """Is ``pos`` inside one of the (sorted, disjoint) text-field spans? O(log n):
    a file with thousands of tags must not turn this into a quadratic scan."""
    k = bisect.bisect_right(spans, (pos, float("inf"))) - 1
    return k >= 0 and spans[k][0] <= pos < spans[k][1]


@dataclass
class _Occurrence:
    tag: str
    value: str
    replace_start: int   # where a replacement text goes ...
    replace_end: int     # ... and what it replaces
    in_loop: bool = False


_QUOTE_END = {"'": re.compile(r"'(?=[ \t\r\n]|$)"), '"': re.compile(r'"(?=[ \t\r\n]|$)')}
_TEXT_FIELD_END = re.compile(r"\r?\n;")
_BARE = re.compile(r"\S+")
_KEYWORDS = ("loop_", "data_", "save_", "global_", "stop_")


def _read_token(text: str, pos: int):
    """The CIF token that starts at or after ``pos`` (whitespace, newlines and
    comment lines skipped) as ``(value, start, end, is_bare)``, or None at the end
    of the text or for an unterminated quote or text field."""
    n = len(text)
    while pos < n:
        if text[pos] in " \t\r\n":
            pos += 1
        elif text[pos] == "#":
            nl = text.find("\n", pos)
            pos = n if nl < 0 else nl + 1
        else:
            break
    if pos >= n:
        return None
    ch = text[pos]
    if ch == ";" and (pos == 0 or text[pos - 1] == "\n"):
        close = _TEXT_FIELD_END.search(text, pos + 1)
        if not close:
            return None
        return text[pos + 1:close.start()].strip(), pos, close.end(), False
    if ch in "'\"":
        m = _QUOTE_END[ch].search(text, pos + 1)
        if not m:
            return None
        return text[pos + 1:m.start()], pos, m.end(), False
    m = _BARE.match(text, pos)
    return m.group(0), pos, m.end(), True


def _is_structure_word(token: str) -> bool:
    return token.startswith("_") or token.lower().startswith(_KEYWORDS)


_LONE_TAG_LINE_RE = re.compile(r"^[ \t]*_\S+[ \t]*$")


def _loop_columns(text: str, names=_DOI_TAG_NAMES) -> dict[int, tuple[int, int, int]]:
    """For every tag of ``names`` that is a column name of a ``loop_`` header, in ONE
    pass over the text: ``{line_start: (column_index, column_count, data_start)}``.
    Linear in the size of the file."""
    found: dict[int, tuple[int, int, int]] = {}
    header: Optional[list[tuple[int, str]]] = None
    in_text_field = False
    offset = 0

    def close(data_start: int) -> None:
        for index, (line_start, name) in enumerate(header):
            if name in names:
                found[line_start] = (index, len(header), data_start)

    for line in text.splitlines(keepends=True):
        start, offset = offset, offset + len(line)
        stripped = line.strip()
        if line.startswith(";"):
            if header is not None:
                close(start)
                header = None
            in_text_field = not in_text_field
            continue
        if in_text_field:
            continue
        if header is not None:
            if not stripped or stripped.startswith("#"):
                continue
            if _LONE_TAG_LINE_RE.match(line.rstrip("\r\n")):
                header.append((start, stripped.lower()))
                continue
            close(start)
            header = None
        if stripped.lower().startswith("loop_"):
            header = []
    if header is not None:
        close(offset)
    return found


def _first_loop_row(text: str, data_start: int, columns: int):
    """The first row of a loop as ``[(value, start, end), ...]``, or None when
    the loop has no complete row."""
    row = []
    pos = data_start
    while len(row) < columns:
        token = _read_token(text, pos)
        if token is None:
            return None
        value, start, end, bare = token
        if bare and _is_structure_word(value):
            return None
        row.append((value, start, end))
        pos = end
    return row


def _doi_occurrences(text: str) -> list[_Occurrence]:
    """Every DOI value in the file, in file order: a scalar after its tag, or the
    first row of a loop that has the tag as a column. Where a replacement goes is
    part of the answer, so reader and writer cannot disagree about which value
    is meant."""
    spans = _spans(text)
    loops = _loop_columns(text)
    found: list[_Occurrence] = []
    for m in _DOI_TAG_RE.finditer(text):
        if _inside(spans, m.start()):
            continue
        column = loops.get(m.start())
        if column is not None:
            index, count, data_start = column
            row = _first_loop_row(text, data_start, count)
            if row is not None:
                value, start, end = row[index]
                found.append(_Occurrence(m.group(1), value, start, end, True))
            continue
        token = _read_token(text, m.end())
        if token is None:
            continue
        value, _start, end, bare = token
        if bare and _is_structure_word(value):
            continue
        found.append(_Occurrence(m.group(1), value, m.end(), end))
    return found


def _reference_match(text: str):
    """``(span, raw_text)`` of the reference field, or None."""
    spans = _spans(text)
    for m in _REFERENCE_RE.finditer(text):
        if not _inside(spans, m.start()):
            return m.span(), m.group(1)
    for m in _REFERENCE_INLINE_RE.finditer(text):
        if not _inside(spans, m.start()):
            return m.span(), next(g for g in m.groups() if g is not None)
    return None


_AUTHOR_TAG = "_publ_author_name"
_JOURNAL_TAGS = ("_journal_name_full", "_journal_name_abbrev", "_journal_volume", "_journal_year",
                 "_journal_page_first", "_journal_page_last")


def _values_of(text: str, tag: str, spans, loops, all_rows: bool = False) -> list[str]:
    """The value(s) of ``tag``: a scalar after it, or the column of a loop (the
    first row, or every row with ``all_rows``). Placeholders (``?`` ``.``) are dropped."""
    pattern = re.compile(r"^[ \t]*" + re.escape(tag) + r"(?=[ \t\r\n])", re.MULTILINE | re.IGNORECASE)
    values: list[str] = []
    for m in pattern.finditer(text):
        if _inside(spans, m.start()):
            continue
        column = loops.get(m.start())
        if column is not None:
            index, count, pos = column
            while True:
                row = _first_loop_row(text, pos, count)
                if row is None:
                    break
                values.append(row[index][0])
                pos = row[-1][2]
                if not all_rows:
                    break
            continue
        token = _read_token(text, m.end())
        if token is None or (token[3] and _is_structure_word(token[0])):
            continue
        values.append(token[0])
    return [v.strip() for v in values if v.strip() not in ("", "?", ".")]


def _journal_citation(text: str) -> str:
    """A citation line from ``_journal_*`` and author fields, or ''.

    Raw COD and ICSD files, and Orienta's own COD downloads, carry the
    bibliography as fields (authors, journal, year, volume, pages) and no
    ``_publ_section_references``. They are published structures; treating them as
    "no reference" would offer to declare them unpublished. At least one journal
    field must be there: authors alone are not a citation."""
    spans = _spans(text)
    loops = _loop_columns(text, frozenset({_AUTHOR_TAG, *_JOURNAL_TAGS}))

    def one(*tags: str) -> str:
        for tag in tags:
            found = _values_of(text, tag, spans, loops)
            if found:
                return found[0]
        return ""

    journal = one("_journal_name_full", "_journal_name_abbrev")
    volume, year = one("_journal_volume"), one("_journal_year")
    first, last = one("_journal_page_first"), one("_journal_page_last")
    if not any((journal, volume, year, first)):
        return ""
    authors = _values_of(text, _AUTHOR_TAG, spans, loops, all_rows=True)
    head = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
    parts = [part for part in (
        journal, volume, f"({year})" if year else "",
        (f"{first}-{last}" if last and last != first else first),
    ) if part]
    body = " ".join(parts)
    return _clean_text(f"{head}: {body}" if head and body else head or body)


def read_citation(raw_text: str) -> CifCitation:
    """DOI(s) and reference text of a CIF, from its raw text. With no
    ``_publ_section_references`` the reference is composed from the journal and
    author fields, if the file has them (``composed``)."""
    dois: list[str] = []
    for occ in _doi_occurrences(raw_text):
        doi = normalise_doi(occ.value)
        if doi and doi not in dois:
            dois.append(doi)
    ref = ""
    composed = False
    found = _reference_match(raw_text)
    if found:
        ref = _clean_text(found[1])
        if ref.startswith("_"):
            ref = ""                 # the "text" is the next CIF tag, not a citation
    if not ref:
        ref = _journal_citation(raw_text)
        composed = bool(ref)
    # Text read from a file is capped like text typed by a person: it ends up in
    # the .xtal and in the header of every SHT file made from it.
    return CifCitation(dois=dois, reference=ref[:MAX_CITATION_CHARS], composed=composed)


def _one_line(value: str) -> str:
    # A line starting with ';' would end a CIF text field, so a pasted
    # citation must not keep its line breaks.
    return " ".join(value.split())


def apply_citation(content: str, doi: str = "", reference: str = "") -> tuple[str, bool]:
    """Write a DOI and/or reference text into CIF text. Empty arguments are
    left alone. Returns ``(new_text, changed)``.

    The DOI replaces every DOI value already in the file, each under its own tag:
    a COD file can carry two tags with different values, and a reader must not
    find the old one under the other. Only a file without any gets a new
    ``_publ_section_doi``. A ``doi`` that is not a DOI (see
    :func:`normalise_doi`) raises :class:`CitationError` ``invalid_doi``: it
    would be written and then never read. A reference over
    :data:`MAX_CITATION_CHARS` raises ``too_long``."""
    changed = False
    doi = doi.strip()
    if doi:
        if len(doi) > MAX_DOI_CHARS:
            raise CitationError("too_long", f"The DOI has {len(doi)} characters; at most {MAX_DOI_CHARS} are accepted.")
        bare = normalise_doi(doi)
        if not bare:
            raise CitationError("invalid_doi", f"Not a DOI: {doi!r}")
        existing = _doi_occurrences(content)
        if existing:
            # A scalar value is replaced after its tag; a loop cell in place. Last
            # first, so that the offsets of the earlier ones stay valid.
            for occ in reversed(existing):
                gap = "" if occ.in_loop else "  "
                content = f"{content[:occ.replace_start]}{gap}'{bare}'{content[occ.replace_end:]}"
        else:
            content += f"\n_publ_section_doi      '{bare}'\n"
        changed = True
    reference = reference.strip()
    if reference:
        reference = _one_line(reference)
        if len(reference) > MAX_CITATION_CHARS:
            raise CitationError(
                "too_long", f"The reference has {len(reference)} characters; at most {MAX_CITATION_CHARS} are accepted."
            )
        # The text sits on one line, and that line starts with a space: a
        # reference that begins with ';' would otherwise close the text field.
        block = f"_publ_section_references\n;\n {reference}\n;"
        found = _reference_match(content)
        if found:
            (a, b), _ = found
            content = f"{content[:a]}{block}{content[b:]}"
        else:
            content += f"\n{block}\n"
        changed = True
    return content, changed
