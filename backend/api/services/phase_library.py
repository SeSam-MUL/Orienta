"""The phase library's one payload: every phase, plus the library totals.

Serves spec §3.1 (2026-09-27-phase-library-design.md). The contract is
`scripts/build_phase_library_fixture.py` on `feat/phase-library`, which froze
the real library into `frontend/.../PhaseLibrary/__fixtures__/library.json` so
the search could be built before this endpoint existed. Field names here match
that fixture; a test compares the two record by record on the real library, so a
drift is a failure and not a surprise.

WHY ONE RESPONSE AND NOT TWO. The totals ship with the phases because two
requests let a number arrive before the list it describes: a header reading
"36 phases" above an empty table is a worse lie than a spinner. Spec §2.8.

`library_totals` IS NOT THE FACET COUNTER, and the name says so because the first
version of this module got the argument wrong. It reasoned that counting from the
returned list makes "the facet disagrees with the list" unreachable -- true, and
beside the point: the page's chips must answer "what would I see if I clicked
this NOW", i.e. against the list as currently FILTERED (§2.8, counters react to
filters), and that is `facets.js` on the frontend. The claim held inside this
payload and stopped exactly at the seam where the original contradiction
happened, where the search said `Cu` matched 6 phases and the facet said 4.

So b9's ruling: f7's rule is the truth, and these numbers are the LIBRARY TOTALS
-- what the header "N of 36" and the load state need, never a live counter. They
are still derived from the very list returned, so they cannot disagree with it;
they simply answer a different question than the sidebar does.

WHAT THIS DELIBERATELY DOES NOT READ. A master pattern is 210 MB and its `mLPNH`
array is 44 MB of that. Nothing here opens an array: the master question is
answered by an h5py KEY lookup (`'EMData/EBSDmaster/mLPNH' in f`), and file
facts come from `stat`.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from path_utils import sanitize_filename
from phase_metadata import (
    KNOWN_ELEMENTS, _SHT_REGEX, _cif_block_role, _iter_cif_blocks,
    _parse_cif_field,
)

logger = logging.getLogger(__name__)

SCHEMA = 1

#: Which block's cell to believe, best first. "standardized" is the conventional
#: cell in a standard setting -- the number a crystallographer quotes. The Niggli
#: cell is the primitive one and is a/sqrt(2) of it for fcc, which is how Al's
#: 4.049 becomes 2.8631: reporting that would be 29 % too small.
SETTING_ORDER = ("standardized_unitcell", "published_cell", "niggli_reduced_cell")

#: A multi-line citation block: `_publ_section_references` followed by a
#: semicolon-delimited text field. Reading only the first line severs the
#: sentence, which is exactly what a user test saw on screen. The terminating
#: `;` may be indented -- CIF permits it, and requiring column 0 silently
#: returned "no source" for such a file.
_REFS = re.compile(r"_publ_section_references\s*\n;(.*?)\n[ \t]*;", re.S)

#: A DOI as registered: `10.<registrant>/<suffix>`.
_DOI = re.compile(r"10\.\d{4,9}/[^\s\"'<>,;]+")

#: An http(s) URL anywhere in a string -- not anchored to a word boundary, see
#: :func:`read_reference`. Brackets are EXCLUDED from the match because a
#: citation writes the link inside them: `... mp-aaacqrcb
#: (https://materialsproject.org/materials/mp-aaacqrcb). A. Jain et al. ...`.
#: A class that swallowed the `)` left the `(` orphaned in mid-sentence, and the
#: card showed `... mp-aaacqrcb ( A. Jain et al. ...`. None of the library's URLs
#: contains a bracket.
_URL = re.compile(r"https?://[^\s\"'<>()\[\]]+")

#: A bracket pair left empty once its URL was removed.
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]")

_ENTITY = re.compile(r"&#x([0-9a-fA-F]+);")
_TAGS = re.compile(r"<[^>]+>")

#: Database ids as they appear in a FILE STEM. `mp` needs the letter guard and
#: the digit class: `r"mp-([0-9a-z]+)"` with `re.I` matches the `MP-` of
#: `Fe3Si_MP-mp-2199` and captures the literal string `mp`, so four phases
#: reported `{"mp": "mp"}` as a Materials Project identifier while the real id
#: sat two characters to the right.
_ID_PATTERNS = (("icsd", r"ICSD[-_ ]?(\d+)"),
                ("cod", r"COD[-_ ]?(\d+)"),
                ("mp", r"(?<![A-Za-z])mp-(\d[0-9a-z]*)"))


# --------------------------------------------------------------------------
# citation
# --------------------------------------------------------------------------

def read_doi(text: str) -> Optional[str]:
    """The bare DOI (`10.xxxx/...`) this CIF states, from any of three places.

    A bare DOI IS a valid source -- b9's decision, and the measurement behind it:
    10 of the 36 library phases carry a findable DOI, and **6 of them have no
    usable formatted citation**, so they were showing "no source" while the file
    held one. Reading only the references block would find ONE of the ten; the
    other nine sit in `_publ_section_doi` (8) and `_journal_paper_doi` (1).

    Normalised to the bare form, because the library writes both
    `'https://doi.org/10.1063/1.4812323'` and `'10.1107/S0108270197015989'` and
    those are one field, not two spellings. The card builds the link.
    """
    for tag in ("_journal_paper_doi", "_publ_section_doi"):
        raw = _parse_cif_field(text or "", tag)
        m = _DOI.search(raw or "")
        if m:
            return m.group(0).rstrip(".,;")
    block = _REFS.search(text or "")
    if block:
        m = _DOI.search(block.group(1))
        if m:
            return m.group(0).rstrip(".,;")
    return None


#: Separators a removed URL can leave behind. `)` and `]` are NOT here -- they
#: are handled by balance, and `.` never is: it ends sentences.
_JUNK_EDGE = " ,;:-_/(["


def _trim_junk(s: str) -> str:
    """Trim leftover separators, keeping brackets that have a partner.

    `(ICSD: icsd-56281, ..., icsd-157941)` must survive whole; a stray `)` with
    no opener must not.
    """
    s = s.strip(_JUNK_EDGE)
    while s and s[-1] in ")]":
        opener = "(" if s[-1] == ")" else "["
        if s.count(opener) >= s.count(s[-1]):
            break                       # balanced: the bracket belongs here
        s = s[:-1].rstrip(_JUNK_EDGE)
    return s


def read_url(text: str) -> Optional[str]:
    """The first http(s) URL the references block states, or None.

    Its own field, because a URL is neither a citation nor a DOI: the five
    SpringerMaterials entries carry one as the only place the source can be
    followed, and stripping it out of `reference` without keeping it would throw
    away the one link the card could offer.
    """
    m = _REFS.search(text or "")
    if not m:
        return None
    hit = _URL.search(m.group(1))
    return hit.group(0).rstrip(".,;)'\"") if hit else None


def read_reference(text: str) -> tuple[Optional[str], Optional[str]]:
    """``(citation, rejection_reason)`` from a CIF's text. At most one is set.

    Rather no citation than one that is plainly not a citation -- a user test
    saw `_publ_author_address` rendered as an author and a bare `https` as a
    source. But "this CIF names no source" and "the source was unusable" are
    two different sentences for the card, so the reason travels instead of both
    collapsing into one null.

    A CITATION THAT MERELY BEGINS WITH ITS DOI IS STILL A CITATION. The first
    version rejected anything starting with "http", which threw away
    `https://doi.org/10.1107/... Cooper M. J.: The structure of the
    intermetallic phase theta-(Al-Cu). Acta Crystallogr. B 28 (1972) 2910.` --
    a complete reference, discarded for its first four characters. The URLs are
    stripped first and what remains is judged on its own.

    (That old test also wore three hats for one head: of `("http://",
    "https://", "http", "https")` only `"http"` could ever match.)
    """
    m = _REFS.search(text or "")
    if not m:
        return None, None
    ref = _TAGS.sub("", m.group(1))
    ref = _ENTITY.sub(lambda x: chr(int(x.group(1), 16)), ref)
    ref = " ".join(ref.replace("&amp;", "&").split()).strip()
    if not ref:
        return None, "empty"
    if ref.startswith("_"):
        return None, "cif_tag"          # e.g. `_publ_author_address`

    # Strip URLs ANYWHERE, not just words that begin with one. The first version
    # dropped whole words starting with "http", which left
    # `mp-1190708_https://legacy.materialsproject.org/...` untouched -- it starts
    # with `mp` -- and returned that as this phase's citation.
    rest = _EMPTY_BRACKETS.sub(" ", _URL.sub(" ", ref))
    # Close the gap the removal leaves in front of punctuation, or the card shows
    # `... mp-aaacqrcb . A. Jain et al.` with a floating full stop.
    rest = re.sub(r"\s+([.,;:])", r"\1", rest)
    # A URL written without its scheme separator (`https` on its own was on a
    # card) leaves a token no strip would catch.
    rest = " ".join(w for w in rest.split()
                    if w.lower() not in ("http", "https", "http:", "https:")
                    and not w.lower().startswith("www."))
    # Trailing junk left by the removal, but NOT a full stop: the first version
    # stripped "." too and took the closing period off every citation that ended
    # in one. And NOT a closing bracket that has an opener: the same strip ran
    # unconditionally, so `Fe3Si_MP-mp-2199`, whose citation legitimately ends
    # `... (ICSD: icsd-56281, ..., icsd-157941)`, reached the card with the
    # bracket open. Four phases were affected and none of them had a URL, which
    # is what made it invisible: the strip was written for URL leftovers and ran
    # on everything.
    rest = _trim_junk(rest)
    if not rest:
        return None, "bare_url"
    # What survives must contain a WORD. `mp-1190708_` does not: after the URL
    # goes, its only letters are `mp`. An identifier and some punctuation is not
    # a reference, and printing it as one is how the doubled sentence with the
    # severed URL reached a user's screen.
    if not re.search(r"[A-Za-z]{3}", rest):
        return None, "ids_only"
    if len(rest) < 12:
        return None, "too_short"
    return rest, None


# --------------------------------------------------------------------------
# cell + setting
# --------------------------------------------------------------------------

def _num(raw: str) -> Optional[float]:
    m = re.match(r"^([\d.]+)", (raw or "").strip())
    return float(m.group(1)) if m else None      # `12.56(2)` -> 12.56


def cell_setting_from_cif(text: str) -> tuple[Optional[str], list[str]]:
    """``(setting whose cell we would quote, all settings the file declares)``.

    Uses the shared block reader, so the setting names come from the same code
    that stops one block's value contradicting another's.
    """
    settings: list[str] = []
    for name, body in _iter_cif_blocks(text or ""):
        role = _cif_block_role(name)
        if role and role not in settings and re.search(
                r"^_cell_length_a\s", body, re.MULTILINE):
            settings.append(role)
    best = next((s for s in SETTING_ORDER if s in settings), None)
    return best, sorted(settings)


def _cell_from_block(text: str, setting: str) -> dict:
    """The six parameters as the named block states them."""
    for name, body in _iter_cif_blocks(text or ""):
        if _cif_block_role(name) != setting:
            continue
        out = {k: _num(_parse_cif_field(body, "_cell_length_" + k))
               for k in ("a", "b", "c")}
        out.update({k: _num(_parse_cif_field(body, "_cell_angle_" + k))
                    for k in ("alpha", "beta", "gamma")})
        if out["a"] is not None:
            return out
    return {}


def _round_cell(cell: dict) -> dict:
    """Six decimals on a length, four on an angle.

    Rounded HERE so every consumer shows the same number: the index carries
    `4.651200000000001` and `119.99999999999999`, and leaving that to the card
    means each reader invents its own precision. Six decimals on an Angstrom is
    far finer than any of these files states (the most precise is 4 dp), so
    nothing is lost; the noise is float representation, not measurement.

    The ROUNDING IS NOT USED FOR THE SETTING CHECK -- `resolve_cell` compares the
    raw values, so a guard cannot be fooled by two numbers that round together.
    """
    out = {}
    for k, v in cell.items():
        if v is None:
            out[k] = None
        else:
            out[k] = round(float(v), 4 if k in ("alpha", "beta", "gamma") else 6)
    return out


def resolve_cell(entry, text: str) -> tuple[dict, Optional[str], list[str], Optional[str]]:
    """``(cell, setting, settings_available, conflict_reason)``.

    The NUMBERS come from the index -- i.e. from the one parsed structure that
    also produced `n_atoms` and the composition, so a card cannot describe two
    blocks at once. The SETTING NAME can only come from the block reader. That
    makes the label a claim about numbers it did not produce, so it is CHECKED:
    when the named block's `a` does not match the index's, the setting is
    reported as null with a reason instead of mislabelling the cell.

    Measured on the shipped library: for all 9 files whose blocks state
    genuinely different values of `a`, pymatgen reports the standardized
    block -- 0 exceptions. That is a property of today's files, not of the code,
    which is why it is verified per request rather than assumed once.
    """
    cell = {"a": entry.lattice_a_A, "b": entry.lattice_b_A, "c": entry.lattice_c_A,
            "alpha": entry.lattice_alpha_deg, "beta": entry.lattice_beta_deg,
            "gamma": entry.lattice_gamma_deg}
    setting, available = cell_setting_from_cif(text)
    if setting is None:
        return cell, None, available, None
    if cell["a"] is None:
        # No numbers to label. NOT a conflict -- the card must not read
        # "the setting disagrees" when the truth is "there is no cell".
        return cell, None, available, "cell_missing"
    stated = _cell_from_block(text, setting)
    if stated.get("a") is None:
        # The block declares `_cell_length_a` but its value is a CIF null or
        # unparsable. A valid setting would be hidden, so say which it was.
        return cell, None, available, "block_unreadable"

    # ALL SIX, not just `a`. The first version checked `a` alone, and `a` is
    # precisely the axis a primitive/conventional pair can SHARE -- monoclinic
    # C-centring, or any setting change that only swaps b and c. Demonstrated:
    # an entry carrying the Niggli cell's b, c and alpha was labelled
    # `standardized_unitcell` with no conflict reported, which is the exact
    # outcome the guard exists to prevent.
    #
    # The tolerance has a relative term, not a flat 0.01 A: on the 26.234 A axis
    # of `α-(AlFeSi)` a 0.0115 A re-derivation is 0.04 % and tripped the flat
    # figure, hiding a valid setting, while on a 2.8 A cell 0.01 A is 0.35 % and
    # far too loose. 5e-4 covers the measured case and is still three orders of
    # magnitude below the signal this guard is for -- a primitive/conventional
    # mix-up is 29 % or more.
    for axis in ("a", "b", "c"):
        got, want = cell[axis], stated.get(axis)
        if got is None or want is None:
            continue
        if abs(got - want) > max(0.01, 5e-4 * abs(want)):
            logger.warning(
                "phase library: %s reports %s=%.4f while its %s block states "
                "%.4f; not labelling the cell", entry.key, axis, got, setting, want)
            return cell, None, available, "setting_mismatch"
    for ang in ("alpha", "beta", "gamma"):
        got, want = cell[ang], stated.get(ang)
        if got is None or want is None:
            continue
        if abs(got - want) > 0.05:
            logger.warning(
                "phase library: %s reports %s=%.3f deg while its %s block "
                "states %.3f deg; not labelling the cell",
                entry.key, ang, got, setting, want)
            return cell, None, available, "setting_mismatch"
    return cell, setting, available, None


# --------------------------------------------------------------------------
# label side
# --------------------------------------------------------------------------

def elements_in_label(s: str, *, from_filename: bool = False) -> list[str]:
    """Element symbols as WRITTEN -- the label side, not the structure.

    TWO GUARDS, both of which a measurement put here.

    `from_filename` truncates at the first underscore, because a file stem is
    not a formula. `Al2Zn_MP-mp-aaacqrcb` read whole yields `['Al','M','P','Zn']`
    -- `MP` is Materials Project, read as manganese and phosphorus -- and that
    single phantom made `elements_disagree` true for a phase whose label and
    structure agree, i.e. one of the three badges on the real library was
    nonsense. `beta-AlFeSi` has no underscore, so the badge that MATTERS (its Si
    is in the name only) survives the truncation.

    And the result is filtered against the element table, because the regex
    alone happily returns `M`, which is not an element at all. The `T` exclusion
    was here before and stays: `T-phase` is a name, not tellurium... which the
    table would accept.
    """
    if not s:
        return []
    if from_filename:
        s = s.split("_", 1)[0]
    s = unicodedata.normalize("NFKD", s)
    found = set(re.findall(r"[A-Z][a-z]?", s)) - {"T"}
    return sorted(e for e in found if e in KNOWN_ELEMENTS)


def _ids_in_filename(key: str) -> Optional[dict]:
    """Database ids visible in the FILENAME: the namer's claim, not the file's.

    Kept apart from `icsd`/`cod`, which are read from the file, because that
    class of claim has already been wrong here -- `Al2Cu_mp-985806…` is named
    Al2Cu and holds an fcc superstructure.
    """
    out = {}
    for kind, pat in _ID_PATTERNS:
        m = re.search(pat, key, re.I)
        if m:
            out[kind] = m.group(1)
    return out or None


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------

def _file_facts(p: Optional[Path], root: Path) -> Optional[dict]:
    if p is None:
        return None
    try:
        st = p.stat()
    except OSError:
        return None
    try:
        rel = str(p.relative_to(root)).replace("\\", "/")
    except ValueError:
        rel = p.name
    return {"name": p.name, "rel": rel, "bytes": st.st_size,
            "date": date.fromtimestamp(st.st_mtime).isoformat()}


def _has_master_array(p: Path) -> bool:
    """Does this .h5 actually carry a master pattern?

    A KEY lookup, not a read: `mLPNH` is 44 MB of a 210 MB file.

    THE NAME CANNOT ANSWER THIS. Measured on the shipped library: 65 `.h5` under
    `EBSD_H5_Cache`, 19 with "master" in the name, but **20** carrying the array
    -- `pi-Al8FeMg3Si6_ICSD-27140_E20kV_sig70_n501_o0.h5` is a real master whose
    name is indistinguishable from the Monte-Carlo outputs beside it. A name rule
    loses it; a looser name rule goes the other way and unlocks 30 phases of
    which 14 have no master at all. Both mistakes were made before this comment.
    """
    try:
        import h5py  # noqa: PLC0415
        with h5py.File(str(p), "r") as f:
            return "EMData/EBSDmaster/mLPNH" in f
    except Exception:
        return False


def master_files(cache_dir: Path) -> list[Path]:
    """Every `.h5` under the cache that really carries a master pattern."""
    if not cache_dir.is_dir():
        return []
    return [p for p in sorted(cache_dir.rglob("*.h5")) if _has_master_array(p)]


def assign_files(files: list[Path], keys) -> dict[str, list[Path]]:
    """Give each file to the LONGEST library key that matches its stem.

    Winner takes all, and that is the whole point. Matching each key
    independently looked right and was not: `beta-AlFeSi` is a prefix of
    `beta-AlFeSi_withSi_COD-2107329`, so the shorter phase claimed the longer
    one's master -- and those are two DIFFERENT phases (one has no Si site at
    all). It would have offered Dictionary indexing against the wrong structure,
    the same harm that `capabilities.hough` was corrected for hours earlier, and
    it moved the agreed capability count from 16 to 17 which is how it was
    caught. `build_index` already walks SHT files longest-key-first for this
    reason; this is the same rule.
    """
    ranked = sorted((k for k in keys if k), key=len, reverse=True)
    out: dict[str, list[Path]] = {}
    for p in files:
        for key in ranked:
            if _stem_matches(p.stem, key):
                out.setdefault(key, []).append(p)
                break
    return out


def _stem_matches(stem: str, key: str) -> bool:
    for w in {key, sanitize_filename(key)}:
        if not w or not stem.startswith(w):
            continue
        rest = stem[len(w):]
        if rest == "" or rest[0] in "_-.":
            return True
    return False


def _files_for_key(files: list[Path], key: str) -> list[Path]:
    """The files that belong to `key`, matched on the FILE STEM.

    Only for a lookup with no library to rank against (the shipped
    `_master_h5_for_phase`, which is handed one phase). Prefer
    :func:`assign_files` whenever the full key set is available -- see its
    docstring for what independent matching gets wrong.

    NOT on the containing folder, and not by substring.

    *Folder* was the first attempt and it is wrong twice. `EBSD_H5_Cache/Al/`
    holds nine `.h5` belonging to EIGHT different phases, so a folder-keyed
    first-wins map would hand phase `Al` the file `Al13Fe4_master_....h5` the
    moment a second master lands there -- silently, since nothing compared the
    file's own name to the phase. And `Dictionary_Library/sd/` holds
    `sd_0302719_master_..._dict_....h5`, whose folder is `sd`, so the phase that
    HAS a pre-built dictionary was told it had none.

    *Substring* is how the shipped matcher does it: inside `Dictionary_Library`
    only, and rejecting tokens under four characters so `Al` and `Ni` can never
    find their own masters -- it answers 2 where the answer is 16.

    So: sanitise the key FORWARD with `path_utils.sanitize_filename` (the
    documented single source of truth, which also spells Greek out, `α-(AlMnSi)`
    -> `alpha-_AlMnSi`) and require it to be the stem's prefix at a boundary.
    The prefix must be followed by `_`, `-`, `.` or end-of-stem, or `Al` would
    claim `Al13Fe4`.
    """
    return [p for p in files if _stem_matches(p.stem, key)]


def _pick_one(cands: Optional[list[Path]]) -> Optional[Path]:
    """The single file the card shows, or None. Newest wins, explicitly.

    `Dictionary_Library/Ni/` holds a 2.0 deg and a 5.0 deg dictionary and the
    record has one slot, so the choice is stated rather than falling out of
    `sorted()`. (That the slot is one and the truth is a list is a known limit,
    recorded for f7: `files.dictionary` shows the newest.)
    """
    if not cands:
        return None
    return max(cands, key=lambda p: (p.stat().st_mtime, len(p.stem)))


def _one_file_for_key(files: list[Path], key: str) -> Optional[Path]:
    """One file for a single key, WITHOUT ranking against the library.

    Kept only for a caller that genuinely has no key set. It is the wrong tool
    whenever one exists: on this library it gives `beta-AlFeSi` the master of
    `beta-AlFeSi_withSi_COD-2107329`. Use :func:`master_for_key`.
    """
    return _pick_one(_files_for_key(files, key))


#: Assignment cache, keyed on what could change it: the file set (count and
#: newest mtime) and the library's key set. Rebuilding per call would re-open 65
#: h5 files for every phase -- 1.1 s for one `_master_map` pass over 36 keys.
_ASSIGN_CACHE: dict = {}


def master_map() -> dict[str, Path]:
    """``{phase key: its master .h5}`` for the whole library, computed once.

    THE SINGLE ANSWER to "can this phase do Dictionary indexing". The phase
    library card, `phase_collections.resolve(method="dictionary")` and the
    Phase-Tester picker all come here, because when they answered separately the
    card said yes for 16 phases and the resolver refused 14 of them.

    Batch, not per key, because the freshness check stats every `.h5` under the
    cache: 36 single lookups cost 1.45 s where one pass costs 0.04 s, and
    `list_collections` does exactly that loop.
    """
    from backend.api.services import crystal_hint_local_library as chl

    cache_dir = chl.PROJECT_ROOT / "Database" / "EBSD_H5_Cache"
    try:
        stamps = [p.stat().st_mtime
                  for p in cache_dir.rglob("*.h5")] if cache_dir.is_dir() else []
    except OSError:
        stamps = []
    keys = tuple(sorted(chl.get_index()))
    sig = (len(stamps), max(stamps, default=0.0), keys)
    hit = _ASSIGN_CACHE.get("master")
    if hit is None or hit[0] != sig:
        files = master_files(cache_dir)
        assigned = assign_files(files, keys)
        # The full file list is cached with the assignment, because
        # `unassigned_masters` needs "every master" and "the claimed ones" and
        # re-scanning would open all 65 files a second time.
        _ASSIGN_CACHE["master"] = (
            sig, {k: p for k, v in assigned.items() if (p := _pick_one(v))}, files)
    return _ASSIGN_CACHE["master"][1]


def all_master_files() -> list[Path]:
    """Every master `.h5` under the cache, from the same scan as :func:`master_map`."""
    master_map()
    return list(_ASSIGN_CACHE["master"][2])


def dictionary_map() -> dict[str, Path]:
    """``{phase key: its pre-built dictionary .h5}``, computed once.

    Same batching reason as :func:`master_map`, and the same ranking: the folder
    name is not the key (`Dictionary_Library/sd/` holds `sd_0302719`'s file), and
    a shorter key must not claim a longer one's.
    """
    from backend.api.services import crystal_hint_local_library as chl
    from path_utils import DATABASE_SUBFOLDERS

    d = chl.PROJECT_ROOT / "Database" / DATABASE_SUBFOLDERS["dictionary_library"]
    try:
        files = sorted(d.rglob("*.h5")) if d.is_dir() else []
        stamps = [p.stat().st_mtime for p in files]
    except OSError:
        files, stamps = [], []
    keys = tuple(sorted(chl.get_index()))
    sig = (len(stamps), max(stamps, default=0.0), keys)
    hit = _ASSIGN_CACHE.get("dict")
    if hit is None or hit[0] != sig:
        assigned = assign_files(files, keys)
        _ASSIGN_CACHE["dict"] = (
            sig, {k: p for k, v in assigned.items() if (p := _pick_one(v))})
    return _ASSIGN_CACHE["dict"][1]


def master_for_key(key: str) -> Optional[Path]:
    """The master, else a pre-built dictionary, for one phase. Either answers
    "can this phase do Dictionary indexing" with yes."""
    return master_map().get(key) or dictionary_map().get(key)


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------

def _synonyms() -> dict:
    """The synonym store, or empty. Never a reason to fail a payload."""
    try:
        from backend.api.services.phase_synonyms import load
        return load()
    except Exception:
        logger.warning("synonym store unavailable", exc_info=True)
        return {}


def _phase_record(key: str, entry, ctx: dict) -> dict:
    cif = Path(entry.cif_path) if entry.cif_path else None
    text = ""
    if cif is not None:
        try:
            text = cif.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""

    cell, setting, available, conflict = resolve_cell(entry, text)
    reference, rejected = read_reference(text)

    sht = ctx["shts"].get(key) or ctx["shts"].get(sanitize_filename(key))
    sht_fields = {}
    if sht is not None:
        m = _SHT_REGEX.match(sht.stem)
        if m:
            sht_fields = m.groupdict()

    master = _pick_one(ctx["master_of"].get(key))
    dictionary = _pick_one(ctx["dict_of"].get(key))

    label = _parse_cif_field(text, "_sm_phase_labels", source=cif.name if cif else "")
    els_structure = list(entry.elements)
    # The label side falls back to the SHT formula and then to the file STEM --
    # and which one it was is reported, because "the label says Si" and "the
    # FILENAME says Si" are different claims. `beta-AlFeSi` is the second kind:
    # its Si is in the name only, so its badge reads "naming vs file", not
    # "label vs structure".
    if label:
        els_label, source = elements_in_label(label), "sm_phase_labels"
    elif sht_fields.get("formula"):
        els_label, source = elements_in_label(sht_fields["formula"]), "sht_formula"
    else:
        els_label = elements_in_label(key, from_filename=True)
        source = "filename"

    return {
        "key": key,
        "formula_label": label or None,
        "formula_structure": entry.formula or None,
        "prototype": _parse_cif_field(text, "_sm_phase_prototype") or None,
        "pearson": (_parse_cif_field(text, "_sm_pearson_symbol")
                    or sht_fields.get("pearson") or None),
        "space_group_hm": entry.space_group or None,
        "space_group_it": entry.space_group_number,
        "crystal_system": entry.crystal_system,
        "n_atoms": entry.n_atoms,
        "elements_structure": els_structure,
        "elements_label": els_label,
        "elements_label_source": source,
        "elements_disagree": bool(els_label) and sorted(els_label) != sorted(els_structure),
        "cell": _round_cell(cell),
        "cell_setting": setting,
        "cell_settings_available": available,
        "cell_setting_conflict": conflict,
        "reference": reference,
        "reference_rejected": rejected,
        "doi": read_doi(text),
        "url": read_url(text),
        "icsd": _parse_cif_field(text, "_database_code_ICSD") or None,
        "cod": _parse_cif_field(text, "_cod_database_code") or None,
        "ids_from_filename": _ids_in_filename(key),
        "folder": sht.parent.name if sht is not None else None,
        "files": {
            "cif": _file_facts(cif, ctx["root"]),
            "xtal": _file_facts(Path(entry.xtal_path) if entry.xtal_path else None,
                                ctx["root"]),
            "master": _file_facts(master, ctx["root"]),
            "dictionary": _file_facts(dictionary, ctx["root"]),
            "sht": _file_facts(sht, ctx["root"]),
        },
        "capabilities": {
            # Hough computes its reflectors FROM THIS CIF, so an unreadable
            # structure is not a Hough capability. `resolve(method="hough")`
            # already refuses such a file, and a green tick the backend then
            # refuses is worse than an honest cross: on sd_1816951 the reader
            # Hough uses (orix/diffpy, not pymatgen) SUCCEEDS and returns
            # Mg 80 / Cu 20 at% where the truth is Cu 66.7 / Mg 33.3.
            "hough": cif is not None and not entry.parse_error,
            "spherical": sht is not None,
            # A master OR a pre-built dictionary. On this library the three
            # dictionaries all belong to phases that also have a master, so
            # either reading gives 16 -- but the question is "can you run
            # Dictionary", and a pre-built one answers yes on its own.
            "dictionary": master is not None or dictionary is not None,
        },
        "parse_error": entry.parse_error,
        # From the synonym store (§2.4). A display name stands IN FRONT OF the
        # key, never instead of it: `key` is still in this record and still
        # searchable, because a rename that hid the filename would leave two
        # people looking at one library unable to name the same phase.
        "display_name": ctx["synonyms"].get(key, {}).get("display_name"),
        "search_terms": list(ctx["synonyms"].get(key, {}).get("search_terms") or []),
        "name_author": ctx["synonyms"].get(key, {}).get("author"),
        "name_updated": ctx["synonyms"].get(key, {}).get("updated"),
    }


def _library_totals(phases: list[dict]) -> dict:
    """Totals over the whole library, derived from the returned list.

    NOT the sidebar's counters -- those narrow with the active filter and live in
    the page (`facets.js`). These answer "how big is the library", for the header
    "N of 36" and for the load state, so that no number is on screen before the
    list it describes.

    Every value is `len([p for p in phases if <predicate>])` over the SAME list
    the caller receives, so it cannot disagree with the payload it ships in.

    `elements` is the system bands (§3.4): a phase joins EVERY band whose element
    it contains, so the counts sum to more than the number of phases -- 93
    placements over 9 bands for 36 phases, with Al₂Cu in both Al and Cu.
    """
    def tally(values_of):
        out: dict[str, int] = {}
        for p in phases:
            for v in values_of(p):
                if v:
                    out[str(v)] = out.get(str(v), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "elements": tally(lambda p: set(p["elements_structure"])),
        "crystal_system": tally(lambda p: [p["crystal_system"]]),
        "capabilities": {k: sum(1 for p in phases if p["capabilities"][k])
                         for k in ("hough", "spherical", "dictionary")},
        "flags": {
            "elements_disagree": sum(1 for p in phases if p["elements_disagree"]),
            "parse_error": sum(1 for p in phases if p["parse_error"]),
            "no_reference": sum(1 for p in phases if not p["reference"]),
            "reference_rejected": sum(1 for p in phases if p["reference_rejected"]),
            "several_cell_settings": sum(
                1 for p in phases if len(p["cell_settings_available"]) > 1),
        },
    }


# --------------------------------------------------------------------------
# the Steckbrief: one phase, with what its simulation was told to do
# --------------------------------------------------------------------------

#: The namelist groups a master `.h5` carries, and what they are for. Read as a
#: HEADER: these are all scalars beside the 44 MB `mLPNH`, which is never opened.
_MASTER_NML_GROUPS = ("EBSDMasterNameList", "MCCLNameList", "BetheList")


def _scalar(ds):
    """One value out of an h5py scalar dataset, as JSON can carry it."""
    try:
        v = ds[()]
    except Exception:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace").strip().strip("'").strip() or None
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        if len(v) == 0:
            return None
        v = v[0]
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace").strip().strip("'").strip() or None
    if isinstance(v, (bool,)):
        return bool(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return int(f) if f.is_integer() and abs(f) < 2 ** 53 else f


def read_master_parameters(path: Path) -> dict:
    """What a master `.h5` says it was simulated with. Header only.

    The card promises "the simulation parameters behind Master and SHT" (§2.3),
    and this is the Master half: `NMLparameters/EBSDMasterNameList` (dmin, npx,
    Legendre grid, ...) and `.../MCCLNameList` (beam energy, depth, bin size).

    NOTHING HERE OPENS `EMData`. Every value read is a scalar in the namelist
    groups; `mLPNH` is 44 MB of the 210 MB file and the whole point of a
    "header-only route" is that a card costs a few kilobytes, not a master
    pattern. `test_the_detail_route_reads_no_bulk_array` pins it on the source.
    """
    out: dict = {}
    try:
        import h5py  # noqa: PLC0415
        with h5py.File(str(path), "r") as f:
            nml = f.get("NMLparameters")
            if nml is not None:
                for group in _MASTER_NML_GROUPS:
                    g = nml.get(group)
                    if g is None:
                        continue
                    vals = {k: _scalar(g[k]) for k in g
                            if getattr(g[k], "shape", None) is not None}
                    vals = {k: v for k, v in vals.items() if v is not None}
                    if vals:
                        out[group] = vals
            prog = f.get("EMheader/EBSDmaster/ProgramName")
            if prog is not None:
                out["program"] = _scalar(prog)
            ver = f.get("EMheader/EBSDmaster/Version")
            if ver is not None:
                out["version"] = _scalar(ver)
    except Exception:
        logger.debug("master header unreadable: %s", path, exc_info=True)
    return out


#: Which Database subfolder a file lives in, so the card can deep-link into the
#: Database browser without the frontend re-deriving it from a path (§3.5).
_CATEGORY_OF_SLOT = {"cif": "cif_library", "xtal": "xtal_library",
                     "master": "h5_cache", "dictionary": "dictionary_library",
                     "sht": "sht_database"}


def build_phase_detail(key: str) -> Optional[dict]:
    """One phase's full card, or None when the library has no such key.

    Everything the index row carries, plus the two things too expensive to send
    for all 36 at once:

    * `simulation.master` -- the namelist the master was built with, read from
      its header;
    * `simulation.sht` -- provenance, crystallography and parameters from the
      `.sht`, via `sht_provenance.build_sht_info`, whose own docstring explains
      why existence is measured on THIS machine and never taken from a sidecar.

    And `deep_link`, naming the Database-browser category of each file that
    exists, so "show me this in the browser" does not need the page to parse a
    path.
    """
    from backend.api.services import crystal_hint_local_library as chl

    idx = chl.get_index()
    entry = idx.get(key)
    if entry is None:
        return None

    root = chl.PROJECT_ROOT
    db = root / "Database"
    shts: dict[str, Path] = {}
    sht_dir = db / "EBSD_SHT_Database"
    for p in sorted(sht_dir.rglob("*.sht")) if sht_dir.is_dir() else []:
        m = _SHT_REGEX.match(p.stem)
        stem = (m.group("phase") or "").strip() if m else ""
        shts.setdefault(stem or p.stem, p)
    m_map, d_map = master_map(), dictionary_map()
    ctx = {"root": root, "shts": shts,
           "master_of": {key: [m_map[key]]} if key in m_map else {},
           "dict_of": {key: [d_map[key]]} if key in d_map else {},
           "synonyms": _synonyms()}

    record = _phase_record(key, entry, ctx)

    sim: dict = {}
    master = record["files"]["master"]
    if master:
        sim["master"] = read_master_parameters(root / master["rel"])
    sht = record["files"]["sht"]
    if sht:
        try:
            from backend.api.services.sht_provenance import build_sht_info
            info = build_sht_info(root / sht["rel"],
                                  xtal_dir=db / "XTAL_Library",
                                  cif_dir=db / "CIF_Library")
            sim["sht"] = {"parameters": info.get("parameters"),
                          "crystallography": info.get("crystallography"),
                          "provenance": info.get("provenance")}
        except Exception:
            logger.debug("sht info unreadable for %s", key, exc_info=True)

    record["simulation"] = sim
    record["deep_link"] = {
        slot: {"category": _CATEGORY_OF_SLOT[slot], "name": f["name"],
               "rel": f["rel"]}
        for slot, f in record["files"].items() if f
    }
    return record


def build_index_document() -> dict:
    """Everything the phase library page needs, in one answer."""
    from backend.api.services import crystal_hint_local_library as chl

    idx = chl.get_index()
    root = chl.PROJECT_ROOT
    db = root / "Database"
    shts: dict[str, Path] = {}
    for p in sorted((db / "EBSD_SHT_Database").rglob("*.sht")) \
            if (db / "EBSD_SHT_Database").is_dir() else []:
        m = _SHT_REGEX.match(p.stem)
        stem = (m.group("phase") or "").strip() if m else ""
        shts.setdefault(stem or p.stem, p)

    # Through `master_map`/`dictionary_map`, so the index and the per-phase card
    # share ONE assignment. Building it here separately meant the first card
    # after a page load paid the 65-file scan again: 1.9 s for `Al` against
    # 0.16 s once warm, while the index had already done the same work.
    m_map, d_map = master_map(), dictionary_map()
    ctx = {"root": root, "shts": shts,
           "master_of": {k: [v] for k, v in m_map.items()},
           "dict_of": {k: [v] for k, v in d_map.items()},
           "synonyms": _synonyms()}

    phases = [_phase_record(k, idx[k], ctx)
              for k in sorted(idx, key=str.lower)]

    # A master that belongs to no library phase is not nothing: one of the four
    # here is the S-phase master. The page shows them under the method chips as
    # "present, not assigned" (spec §2.7) rather than letting them be invisible
    # because no CIF happens to carry their name.
    claimed = {f["files"]["master"]["rel"] for f in phases if f["files"]["master"]}
    unassigned = [f for f in (_file_facts(p, root) for p in all_master_files())
                  if f and f["rel"] not in claimed]
    return {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Database/ on this machine, via crystal_hint_local_library",
        "phases": phases,
        "unassigned_masters": unassigned,
        "library_totals": _library_totals(phases),
    }
