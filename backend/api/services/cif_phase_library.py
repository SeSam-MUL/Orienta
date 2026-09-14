"""
CIF-based phase library for EDS phase suggestion.

Reads ``Database/crystal_database.xlsx`` (built by the CIF database tooling)
and returns a per-CIF phase library with normalized atomic-percent
compositions suitable for matching against measured EDS data.

The hardcoded ``DEFAULT_PHASE_LIBRARY`` in :mod:`eds_utils` covers ~30
canonical phases. This module replaces that for users who curate their
own CIFs — every CIF in the database becomes a candidate phase.

Composition parsing relies on :class:`pymatgen.core.composition.Composition`
so that parenthesised groups (e.g. ``Al2(FeSi)3``), decimal subscripts
(``Mn4.512Al127.296``), and Unicode subscripts in the Excel are all
handled the same way the rest of the project parses formulas.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.api.services.chemistry_score import (
    DEFAULT_REL_REQ, background_levels, infer_matrix_element,
    score_phase_ratio,
    score_phase_vectorised,
)
from backend.api.services.crystal_hint_phase_fit import _CHEM_IGNORE

logger = logging.getLogger(__name__)

# Reuse the project-wide subscript translation so the parsing matches
# whatever the database builder writes into the xlsx.
_SUB_TO_ASCII = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# ... and back, so a formula read straight from a CIF is displayed the same
# way as one the database builder wrote into the xlsx
# (cif_database_builder._format_formula_with_subscripts). Decimal points stay
# plain, exactly as there: "Mn5.268Al100.89" -> "Mn₅.₂₆₈Al₁₀₀.₈₉".
_ASCII_TO_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Two winners closer than this are a tie, not a decision.
#
# Deliberately equal to grain_phase_assignment.TIE_TOLERANCE, which is
# empirical and pinned by tests between a gap of 0.0030 ("must read as a
# tie") and 0.0187 ("must decide") on this same chemistry_fit score scale.
# An earlier version of this line said 0.02 and claimed to match it; 0.02 is
# above 0.0187, i.e. it would have flagged as an unresolvable tie exactly
# the case that module calibrated as a decision. Read the derivation there
# before touching this.
TIE_TOLERANCE = 0.01


@dataclass
class CifPhaseEntry:
    """One CIF entry projected onto the EDS-matching surface."""

    key: str                        # unique identifier (CIF filename or row index)
    cif_filename: str               # name as stored in the xlsx
    formula: str                    # Unicode formula, as displayed
    space_group: str
    space_group_number: Optional[int]
    crystal_system: str
    composition: Dict[str, float]   # element symbol -> at.% (sum 100)
    elements: List[str]             # sorted symbols, for fast subset checks
    #: Set by :func:`_suspect_rows` when this row came from a CIF the reader
    #: now refuses. The value in `composition` is then not trustworthy, and
    #: the flag has to travel WITH the entry rather than beside it so that
    #: whoever displays the phase can say so.
    suspect: bool = False
    suspect_reason: str = ""


# In-memory cache keyed by (xlsx path, CIF folder). The stored signature
# covers BOTH sources, so a rebuilt spreadsheet and a freshly downloaded CIF
# each invalidate it without a backend restart.
#: (xlsx, cif_dir) -> (folder signature, library, skipped files). The skips
#: are cached with the library because the suggestion panel asks on every
#: click: a list that only existed on the cold call would be there once and
#: gone for the rest of the session.
_LIBRARY_CACHE: Dict[
    Tuple[str, str],
    Tuple[tuple, Dict[str, CifPhaseEntry], List[Dict[str, str]]],
] = {}

# Reading one CIF costs ~0.25 s (pymatgen), and the folder is re-scanned on
# every suggest-phases click, so entries are memoised per
# (path, mtime, size) — only a file that actually changed is read again.
# The cached value is the (entry, reason) pair _entry_from_cif returns, so a
# file that failed to read is still reported by name on every later load.
_CIF_ENTRY_CACHE: Dict[
    Tuple[str, int, int], Tuple[Optional[CifPhaseEntry], str]] = {}
_CIF_ENTRY_CACHE_MAX = 4096


def _normalize_formula(raw: str) -> str:
    """Translate Unicode subscripts to ASCII digits and strip whitespace."""
    if not raw:
        return ""
    return raw.translate(_SUB_TO_ASCII).strip()


def parse_formula_to_at_pct(formula: str) -> Dict[str, float]:
    """Return ``{element: atomic_percent}`` for a chemical formula string.

    Returns an empty dict if pymatgen cannot parse the formula (e.g. when
    the database has an obviously broken composition entry). The caller
    decides how to log/handle that — silent fallback to "no match" is
    fine for phase suggestion, fail-loud is appropriate for validation
    paths.
    """
    norm = _normalize_formula(formula)
    if not norm:
        return {}
    try:
        from pymatgen.core.composition import Composition
    except ImportError:  # pragma: no cover — pymatgen is in the env
        logger.warning("pymatgen not installed — cannot parse formula %r", formula)
        return {}
    try:
        comp = Composition(norm)
        frac = comp.fractional_composition.as_dict()
    except Exception as exc:
        logger.warning("Failed to parse formula %r: %s", formula, exc)
        return {}
    return {str(el): float(frac_val) * 100.0 for el, frac_val in frac.items()}


class AmbiguousCifError(ValueError):
    """A CIF whose data blocks parse to DIFFERENT compositions.

    A SpringerMaterials CIF carries several blocks -- standardized, published,
    Niggli-reduced -- and pymatgen returns one structure per readable block.
    Normally they agree and ``[0]`` is harmless. When they do not, ``[0]`` means
    the composition of a phase is decided by data-block ORDER.

    `Database/CIF_Library/sd_1816951.cif` is the case that found this: MgCu2, a
    cubic Laves phase whose own atom-site table says Cu 16c + Mg 8b, i.e.
    Cu 66.7 / Mg 33.3, and whose .xtal (what EMsoft simulated) says the same.
    pymatgen returns Mg4Cu (40 sites) and Mg2Cu (24) because the Fd-3m origin
    choice is mis-expanded, and NEITHER is the compound. Taking ``[0]`` put
    Mg 80 / Cu 20 into the phase library, the Crystal-Hint suggestions and the
    spreadsheet the database builder writes.

    Carries both formulas, because a caller that cannot show them cannot let a
    user fix the file.
    """

    def __init__(self, source: str, formulas):
        self.source = str(source)
        self.formulas = sorted(str(f) for f in formulas)
        super().__init__(
            f"{self.source} parses to {len(self.formulas)} different "
            f"compositions ({', '.join(self.formulas)}); refusing it rather "
            f"than letting data-block order decide. Check the file.")


def one_structure(structures, source: str):
    """The single structure a CIF means, or :class:`AmbiguousCifError`.

    THE ONE COPY. Four places read a CIF and took ``structures[0]``: this
    module, ``crystal_hint_local_library._parse_cif`` (the Crystal-Hint
    suggestions), ``crystal_structure._load_cif`` (the 3-D viewer and the
    forward model) and ``cif_database_builder.parse_cif_file`` (which writes
    the spreadsheet everything else reads). Four copies of the same guard would
    have drifted; this is the guard.

    Refusing rather than guessing is the point: a phase missing from a list is
    an absence the user can see, a phase with an inverted composition is a
    confident wrong answer. Measured over the 36 CIFs of the shipped library
    (2026-09-13): 18 declare more than one ``data_`` block, ten return more
    than one structure, nine of those agree, and this raises on sd_1816951
    alone.

    Raises ``ValueError`` (plain) when there is no structure at all, so a
    caller can tell "unreadable" from "ambiguous" by the exception type.
    """
    if not structures:
        raise ValueError(f"no structure could be parsed from {source}")
    formulas = {str(s.composition.reduced_formula) for s in structures}
    if len(formulas) > 1:
        raise AmbiguousCifError(source, formulas)
    return structures[0]


def _parse_space_group_number(sg_str: str) -> Optional[int]:
    """Extract the IT space-group number from strings like ``'Pna2_1 (33)'``."""
    if not sg_str:
        return None
    match = re.search(r"\((\d+)\)", sg_str)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _cif_dir_for(xlsx_path: Path) -> Path:
    """Where the CIFs live, relative to the spreadsheet.

    Mirrors ``database._DB_SUBDIRS["cif"]`` (``Database/CIF_Library``). The
    name is repeated rather than imported because a service must not depend
    on the route layer.
    """
    return Path(xlsx_path).parent / "CIF_Library"


def _cif_paths(cif_dir: Path) -> List[Path]:
    """Every CIF under ``cif_dir``, ordered by filename.

    The suffix test is case-insensitive because the database builder accepts
    ``.CIF`` as well, and on a case-sensitive filesystem ``rglob("*.cif")``
    would quietly miss those.
    """
    if not cif_dir.is_dir():
        return []
    try:
        found = [q for q in cif_dir.rglob("*")
                 if q.is_file() and q.suffix.lower() == ".cif"]
    except OSError as exc:
        logger.warning("Could not scan CIF folder %s: %s", cif_dir, exc)
        return []
    return sorted(found, key=lambda q: (q.name, str(q)))


def _folder_signature(paths: List[Path]) -> tuple:
    """Fingerprint that changes on add, delete, rename or overwrite."""
    sig = []
    for q in paths:
        try:
            st = q.stat()
        except OSError:
            continue
        sig.append((q.name, st.st_mtime_ns, st.st_size))
    return tuple(sig)


def _entry_from_cif(path: Path):
    """Project one CIF file onto the EDS-matching surface.

    Read exactly the way ``cif_database_builder.parse_cif_file`` reads it —
    the non-primitive structure through :func:`one_structure`, then its reduced
    formula — so the entry produced here is the one the database builder would
    later write for the same file, not a second opinion about it. Both went
    through that guard on 2026-09-12; before then the parity claim was true and
    the shared behaviour was wrong, which is how Mg 80 / Cu 20 reached the
    spreadsheet.

    THE PARITY IS ABOUT THE CODE, NOT ABOUT THE SHIPPED FILE. A spreadsheet
    written by an older builder still holds what that builder produced. Those
    rows are found by :func:`load_cif_phase_library_with_skips`, which reports
    them as `suspect` beside the files it could not read at all.

    Returns ``(entry, reason)``. One of the two is always empty: an entry and
    ``""``, or ``None`` and the reason it could not be read.

    Surviving a broken download is right — it must not cost the user the rest
    of the library — but surviving it SILENTLY is what made the file
    invisible: it sits in ``Database/CIF_Library``, the suggestion panel does
    not offer it, and ``explain_no_match`` cannot name it either, because a
    file that never became an entry is not in the library it reasons about.
    The reason travels so the load can report it; see
    :func:`load_cif_phase_library_with_skips`.
    """
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:  # pragma: no cover — pymatgen is in the env
        logger.warning("pymatgen not installed — cannot read %s", path.name)
        return None, "the CIF reader (pymatgen) is not installed"

    try:
        parser = CifParser(str(path))
        # parse_structures superseded get_structures in pymatgen 2024.x;
        # the old name still works but warns.
        read = getattr(parser, "parse_structures", None) or parser.get_structures
        structures = read(primitive=False)
        structure = one_structure(structures, path.name)
    except AmbiguousCifError as exc:
        # The multi-block refusal travels on the SAME channel as every other
        # reason a file did not become a phase, so the suggestion panel shows
        # it without knowing this case exists.
        logger.warning("%s", exc)
        return None, str(exc)
    except Exception as exc:
        logger.warning("Could not read CIF %s: %s", path.name, exc)
        return None, f"unreadable: {exc}"

    formula = str(structure.composition.reduced_formula)
    composition = parse_formula_to_at_pct(formula)
    if not composition:
        logger.warning("CIF %s parsed to an empty composition (%r)",
                       path.name, formula)
        return None, (f"parsed to an empty composition ({formula!r}) — no "
                      "element to match chemistry against")
    formula = formula.translate(_ASCII_TO_SUB)

    # Metadata for the UI only — matching needs the composition alone, so a
    # structure whose symmetry cannot be determined still becomes a phase.
    space_group, sg_number, crystal_system = "", None, ""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=1e-5)
        sg_number = int(sga.get_space_group_number())
        space_group = f"{sga.get_space_group_symbol()} ({sg_number})"
        crystal_system = str(sga.get_crystal_system())
    except Exception as exc:
        logger.debug("No symmetry for %s: %s", path.name, exc)

    return CifPhaseEntry(
        key=path.name,
        cif_filename=path.name,
        formula=formula,
        space_group=space_group,
        space_group_number=sg_number,
        crystal_system=crystal_system,
        composition=composition,
        elements=sorted(composition.keys()),
    ), ""


def _cached_entry_from_cif(path: Path):
    """:func:`_entry_from_cif`, memoised on (path, mtime, size).

    Caches the ``(entry, reason)`` pair, so a file that failed to read is
    still reported by name on every later (cached) load rather than only on
    the cold one.
    """
    try:
        st = path.stat()
    except OSError as exc:
        return None, f"could not be opened: {exc}"
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    if key in _CIF_ENTRY_CACHE:
        return _CIF_ENTRY_CACHE[key]
    result = _entry_from_cif(path)
    if len(_CIF_ENTRY_CACHE) >= _CIF_ENTRY_CACHE_MAX:
        _CIF_ENTRY_CACHE.clear()
    _CIF_ENTRY_CACHE[key] = result
    return result


def load_cif_phase_library(
    xlsx_path: Path, cif_dir: Optional[Path] = None,
) -> Dict[str, CifPhaseEntry]:
    """The phase library alone — see :func:`load_cif_phase_library_with_skips`.

    Every existing caller wants the dict and nothing else; this is that dict,
    read off the same one load, not a second one.
    """
    return load_cif_phase_library_with_skips(xlsx_path, cif_dir)[0]


def load_cif_phase_library_with_skips(
    xlsx_path: Path, cif_dir: Optional[Path] = None,
):
    """Load and cache the phase library, plus the files it could not read.

    Returns ``(library, skipped)`` where ``skipped`` is
    ``[{"file": name, "reason": text}, ...]`` — one record per CIF on disk
    that did not become a phase. A file that cannot be read is not an error
    (it must not cost the user the rest of the library), but it must not be
    silent either: it is ON DISK, the user put it there, and until 2026-09-12
    the only trace was a log line — the suggestion panel simply did not offer
    the phase, and ``explain_no_match`` could not name it, because it reasons
    about entries and this file never became one.

    TWO SOURCES, ONE ANSWER. ``Database/CIF_Library`` decides WHICH phases
    exist; ``crystal_database.xlsx`` supplies curated metadata for the ones
    it happens to list. A CIF the spreadsheet does not know is read from the
    file itself and appended.

    Until 2026-09-12 the spreadsheet was the only source — and nothing that
    PUTS a CIF on disk writes a row into it. ``download-cif``, ``add-cif``
    and the server sync all leave it untouched; only the explicit "Build
    database" action rewrites it. A tester who downloaded his first CIF
    therefore got no phase suggestion for it in that session, and on the
    developer's own machine six CIFs added after the last build (2026-06-08)
    had been invisible to phase suggestion for three months.

    Rows keep their spreadsheet order and unlisted files are appended behind
    them, because a phase's position in this dict is the id written into
    :class:`PhaseMapStore`. Rows whose CIF is gone from disk are LEFT in
    place: pruning them belongs to the database builder, and doing it here
    would renumber every phase behind the gap.

    Returns an empty dict when there is neither a readable spreadsheet nor a
    CIF on disk.
    """
    p = Path(xlsx_path).resolve()
    folder = Path(cif_dir) if cif_dir is not None else _cif_dir_for(p)

    paths = _cif_paths(folder)
    try:
        xlsx_mtime = p.stat().st_mtime_ns if p.is_file() else None
    except OSError:
        xlsx_mtime = None
    signature = (xlsx_mtime, _folder_signature(paths))

    cache_key = (str(p), str(folder))
    cached = _LIBRARY_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1], cached[2]

    library = _entries_from_xlsx(p)
    skipped: List[Dict[str, str]] = []

    unlisted = [q for q in paths if q.name not in library]
    if unlisted:
        started = time.perf_counter()
        added = 0
        for q in unlisted:
            entry, reason = _cached_entry_from_cif(q)
            if entry is None:
                skipped.append({"file": q.name, "reason": reason})
                continue
            if entry.key in library:
                # Same filename in two material subfolders — the database
                # builder is name-keyed too, so the first one wins here.
                logger.warning("Duplicate CIF filename %s — %s ignored",
                               entry.key, q)
                skipped.append({
                    "file": str(q),
                    "reason": (f"another file is already named {entry.key} — "
                               "the library is keyed by filename, so only the "
                               "first one counts"),
                })
                continue
            library[entry.key] = entry
            added += 1
        logger.info(
            "CIF library: %d file(s) not listed in %s; %d read straight from "
            "disk in %.2f s (%d skipped)", len(unlisted), p.name, added,
            time.perf_counter() - started, len(skipped))

    # Rows the spreadsheet supplies from a CIF the reader refuses. They are
    # NOT skips -- the phase is offered -- but they belong on the same
    # channel, because both answer "why is this phase not what I expect".
    skipped.extend(_suspect_rows(library, paths))

    _LIBRARY_CACHE[cache_key] = (signature, library, skipped)
    return library, skipped


def _entries_from_xlsx(p: Path) -> Dict[str, CifPhaseEntry]:
    """The curated rows, in spreadsheet order. Empty if unreadable."""
    if not p.is_file():
        return {}

    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        logger.error("pandas not installed — cannot read %s", p)
        return {}

    try:
        df = pd.read_excel(str(p)).fillna("")
    except Exception:
        logger.exception("Failed to read CIF database at %s", p)
        return {}

    library: Dict[str, CifPhaseEntry] = {}
    seen_keys: Dict[str, int] = {}
    for idx, row in df.iterrows():
        formula = str(row.get("Composition", "")).strip()
        cif_file = str(row.get("CIF File Name", "")).strip()
        if not formula or not cif_file:
            continue
        composition = parse_formula_to_at_pct(formula)
        if not composition:
            continue
        # CIF filename is typically unique, but we suffix duplicates with
        # the row index so that two rows pointing at the same file (e.g.
        # accidental import) don't silently overwrite each other.
        base_key = cif_file
        suffix = seen_keys.get(base_key, 0)
        seen_keys[base_key] = suffix + 1
        key = base_key if suffix == 0 else f"{base_key}#{suffix}"

        sg_str = str(row.get("Space Group", "")).strip()
        sg_num = _parse_space_group_number(sg_str)
        crystal_system = str(row.get("Crystal System", "")).strip()

        library[key] = CifPhaseEntry(
            key=key,
            cif_filename=cif_file,
            formula=formula,
            space_group=sg_str,
            space_group_number=sg_num,
            crystal_system=crystal_system,
            composition=composition,
            elements=sorted(composition.keys()),
        )

    return library


def suggest_phases_from_cif_library(
    measured_at_pct: Dict[str, float],
    cif_library: Dict[str, CifPhaseEntry],
    tolerance: float = 15.0,
    min_score: float = 0.3,
    max_results: int = 20,
    matrix_element: Optional[str] = None,
    background: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Match measured At.% against the CIF library.

    Pre-filters by element subset: a phase whose elements are not all
    present in the measurement cannot be a real match (we'd be scoring
    against zeros, which both inflates the deviation and misranks
    plausible candidates). The remaining phases go through the same
    scoring routine that the legacy hardcoded library uses, and the
    survivors are enriched with CIF metadata for the UI.
    """
    if not cif_library or not measured_at_pct:
        return []

    candidates = candidates_for(
        cif_library, {el for el, v in measured_at_pct.items() if v > 0.0})
    if not candidates:
        return []

    # THE SAME SCORER THE MAP USES. These two used to disagree: the panel
    # ran eds_utils.suggest_phases (absolute deviation against a tolerance)
    # while the map ran the vetoed scorer, and on a Cu-rich pixel of the
    # 7050 file the panel ranked Al7FeCu2 at 0.415 while the map vetoed it
    # to 0.050. The user saw the right answer in one panel and a different
    # map beside it, with no way to reconcile them.
    #
    # `background` and `matrix_element` are properties of the WHOLE map, so
    # a caller scoring one pixel has to supply them; without them the median
    # of a single value is that value and the enrichment gate vetoes
    # everything.
    one = {el: np.array([float(v)], dtype=np.float64)
           for el, v in measured_at_pct.items()}
    scored = []
    for entry in candidates:
        s_val = float(score_phase_ratio(
            one, entry.composition,
            matrix_element=matrix_element, no_data_score=0.0,
            background=background,
        )[0])
        if s_val < min_score:
            continue
        scored.append((s_val, entry))
    scored.sort(key=lambda t: (-t[0], t[1].cif_filename))

    enriched: List[Dict] = [
        {
            "key": e.key,
            "cif_filename": e.cif_filename,
            "formula": e.formula,
            "space_group": e.space_group,
            "space_group_number": e.space_group_number,
            "crystal_system": e.crystal_system,
            "score": s_val,
            "expected": dict(e.composition),
            "elements": list(e.elements),
        }
        for s_val, e in scored
    ]
    return enriched[:max_results]


def explain_no_match(
    measured_at_pct: Dict[str, float],
    cif_library: Dict[str, CifPhaseEntry],
    matrix_element: Optional[str] = None,
    background: Optional[Dict[str, float]] = None,
    min_score: float = 0.3,
) -> Optional[Dict]:
    """Why :func:`suggest_phases_from_cif_library` returned nothing.

    Returns None when it did return something. Otherwise a dict with a
    stable ``code`` (for translation), the numbers behind it, and an
    English ``message`` as the fallback for a UI that has no string for
    that code yet.

    The case worth the code is ``not_enriched_over_background``. The panel
    scores ONE pixel with the WHOLE MAP's matrix element and background,
    and ``score_phase_ratio`` will not assign a phase whose defining
    element fails to sit 1.3x above that map's own median. For a particle
    in a matrix that is the right question. For a phase that FILLS the map
    it is backwards: measured on a map acquired with only Mg and Si — what
    an operator hunting Mg2Si would do — the median Si is 30 at%, the bar
    39 at%, and a 25 at% Si pixel of clean Mg2Si is "not enriched". The
    phase is then vetoed on every pixel of a map made of it, the route
    falls back to the hardcoded library, which has no Mg phase, and the
    user is shown an empty list with no reason. That is the 2026-08-25
    report.

    This function does NOT loosen the gate — its constants are calibrated
    on SampleB and the panel shares them with the map deliberately. It
    makes the silence speak.
    """
    if not measured_at_pct:
        return {"code": "no_chemistry",
                "message": "This pixel carries no EDS chemistry to match."}
    if not cif_library:
        return {"code": "empty_library",
                "message": ("No CIF phases available — put CIF files in "
                            "Database/CIF_Library or build the CIF database.")}

    measured_els = {el for el, v in measured_at_pct.items() if v > 0.0}
    candidates = candidates_for(cif_library, measured_els)
    if not candidates:
        shown = ", ".join(sorted(measured_els)) or "nothing"
        return {"code": "no_candidate_phase",
                "elements": sorted(measured_els),
                "message": (f"No phase in the library is built only from the "
                            f"elements this pixel shows ({shown}).")}

    one = {el: np.array([float(v)], dtype=np.float64)
           for el, v in measured_at_pct.items()}

    def _score(entry: CifPhaseEntry, bg) -> float:
        return float(score_phase_ratio(
            one, entry.composition, matrix_element=matrix_element,
            no_data_score=0.0, background=bg)[0])

    # An empty background dict makes every enrichment bar 0.0, i.e. the same
    # scorer WITHOUT the map — the only honest way to ask "would this have
    # matched if it were not for the map's own composition?".
    ranked = sorted(
        ((_score(e, background), _score(e, {}), e) for e in candidates),
        key=lambda t: (-t[1], t[2].cif_filename),
    )
    if any(gated >= min_score for gated, _free, _e in ranked):
        return None

    blocked = [(gated, free, e) for gated, free, e in ranked
               if free >= min_score and gated < min_score]
    if blocked:
        gated, free, entry = blocked[0]
        el, measured, bar, bg_level = _blocking_element(
            measured_at_pct, entry.composition, matrix_element, background)
        return {
            "code": "not_enriched_over_background",
            "phase": entry.cif_filename,
            "element": el,
            "measured_at_pct": round(measured, 1),
            "background_at_pct": round(bg_level, 1),
            "required_at_pct": round(bar, 1),
            "score_without_map": round(free, 3),
            "message": (
                f"{entry.cif_filename} fits this pixel ({free:.2f}) but was "
                f"rejected: {el} is {measured:.1f} at% here against a map "
                f"background of {bg_level:.1f} at%, and a phase requiring {el} "
                f"is only assigned above {bar:.1f} at%. A phase that makes up "
                f"most of the map cannot clear that bar."),
        }

    gated, free, entry = ranked[0]
    return {
        "code": "below_min_score",
        "phase": entry.cif_filename,
        "score": round(free, 3),
        # The bar belongs next to the number: "0.21" alone does not say
        # whether the phase was close or nowhere near.
        "min_score": round(float(min_score), 2),
        "message": (f"The closest phase in the library, {entry.cif_filename}, "
                    f"only scores {free:.2f} against this chemistry "
                    f"(minimum {min_score:.2f})."),
    }


def _blocking_element(
    measured_at_pct: Dict[str, float],
    phase_at_pct: Dict[str, float],
    matrix_element: Optional[str],
    background: Optional[Dict[str, float]],
) -> Tuple[str, float, float, float]:
    """Which defining element failed the enrichment bar, in at%.

    Repeats the arithmetic of ``score_phase_ratio``'s gate rather than
    importing a private helper for it; ``test_reason_agrees_with_the_scorer``
    is what stops the two from drifting apart.

    Returns ``(element, measured_at_pct, required_at_pct, background_at_pct)``
    for the element that misses its bar by the widest margin.
    """
    from backend.api.services.chemistry_score import _ENRICHMENT, _MAJOR_REQ

    def _norm(d):
        kept = {el: max(0.0, float(v)) for el, v in (d or {}).items()
                if el not in _CHEM_IGNORE and v is not None}
        total = sum(kept.values())
        return {el: v / total for el, v in kept.items()} if total > 1e-9 else {}

    p, q = _norm(measured_at_pct), _norm(phase_at_pct)
    bg = background or {}

    worst = None
    for el, frac in q.items():
        if frac < _MAJOR_REQ or el == matrix_element:
            continue
        bar = _ENRICHMENT * bg.get(el, 0.0)
        measured = p.get(el, 0.0)
        if bar > 0.0 and measured < bar:
            margin = bar - measured
            if worst is None or margin > worst[0]:
                worst = (margin, el, measured * 100.0, bar * 100.0,
                         bg.get(el, 0.0) * 100.0)
    if worst is None:          # pragma: no cover — callers check first
        return "", 0.0, 0.0, 0.0
    _margin, el, measured, bar, bg_level = worst
    return el, measured, bar, bg_level


#: How far a stored composition may sit from its CIF's own atom-site table
#: before it is worth telling the user, in at% on the worst element.
#:
#: Measured over the shipped 36-CIF library (2026-09-13): 15 files have a
#: readable site table, 13 of them land within 1 at% of the stored value, the
#: largest benign difference is 2.7 (sd_0302719, where the table lists a
#: mixed-occupancy cell and this reader sums Wyckoff multiplicities rather
#: than expanding the symmetry), and the one real disagreement is 46.7
#: (sd_1816951). 10.0 is 3.7x above the largest benign and 4.7x below the
#: real one.
_SUSPECT_AT_PCT = 10.0

_WYCKOFF_MULT = re.compile(r"(\d+)\s*[a-z]", re.I)
_SITE_SPECIES = re.compile(r"([0-9.]*)\s*([A-Z][a-z]?)")


def site_table_at_pct(path: Path) -> Optional[Dict[str, float]]:
    """At.% read straight off a CIF's ``_atom_site`` loop, or None.

    Wyckoff multiplicity x occupancy, summed per element. TEXT ONLY -- it
    never goes near pymatgen, which is the entire point: it is an independent
    opinion about what a CIF says, usable to check a composition that came
    FROM pymatgen.

    Independence matters more than it sounds. The obvious other reference, the
    ``.xtal`` EMsoft simulated from, is NOT independent: ``sd_1816951.xtal``
    holds Mg at (1/8, 3/8, 1/8) and Cu at (1/2, 0, 1/2) in Fd-3m, which
    expands to Mg32Cu8 -- the same Mg 80 / Cu 20 as the spreadsheet, because
    the converter was fed the same parse. The CIF text says ``Cu .16c`` and
    ``Mg .8b``: Cu 66.7 / Mg 33.3.

    Returns None when the loop has no Wyckoff column to read multiplicities
    from -- 21 of the 36 shipped CIFs (Materials Project, ICSD and COD exports
    list explicit coordinates instead). No reference then, and nothing is
    claimed.

    Mixed sites written as ``'0.884Al + 0.116Si'`` are split; the largest
    readable loop in a multi-block file wins.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    best: Optional[Dict[str, float]] = None
    for m in re.finditer(r"(?m)^\s*loop_\s*$", text):
        lines = [ln.strip() for ln in text[m.end():].splitlines()]
        tags, i = [], 0
        while i < len(lines) and (not lines[i] or lines[i].startswith("_")):
            if lines[i].startswith("_"):
                tags.append(lines[i].split()[0])
            i += 1
        if ("_atom_site_type_symbol" not in tags
                or "_atom_site_Wyckoff_symbol" not in tags):
            continue
        i_sym = tags.index("_atom_site_type_symbol")
        i_wyk = tags.index("_atom_site_Wyckoff_symbol")
        i_occ = (tags.index("_atom_site_occupancy")
                 if "_atom_site_occupancy" in tags else None)
        counts: Dict[str, float] = {}
        for ln in lines[i:]:
            if not ln or ln.startswith(("_", "loop_", "data_", "#", ";")):
                break
            toks = re.findall(r"'[^']*'|\"[^\"]*\"|\S+", ln)
            if len(toks) < len(tags):
                continue
            mult = _WYCKOFF_MULT.search(toks[i_wyk].strip("'\".").strip())
            if not mult:
                continue
            occ = 1.0
            if i_occ is not None:
                try:
                    occ = float(re.sub(r"\(.*\)", "", toks[i_occ]))
                except ValueError:
                    occ = 1.0
            for frac, el in _SITE_SPECIES.findall(toks[i_sym].strip("'\"")):
                if el:
                    counts[el] = (counts.get(el, 0.0)
                                  + int(mult.group(1)) * occ
                                  * (float(frac) if frac else 1.0))
        total = sum(counts.values())
        if total > 0 and (best is None or total > sum(best.values())):
            best = {e: 100.0 * v / total for e, v in counts.items()}
    return best


def _suspect_rows(
    library: Dict[str, CifPhaseEntry], paths: List[Path],
) -> List[Dict[str, str]]:
    """Spreadsheet rows whose composition its own CIF does not support.

    The skip list beside it answers "this file never became a phase". This
    answers the harder one: the file DID become a phase, from a
    ``crystal_database.xlsx`` row, and that row does not match the atom-site
    table of the CIF it names. The phase is offered, with a number the file it
    came from contradicts, and the row wins over the CIF everywhere.
    ``sd_1816951.cif`` is that today: MgCu2 stored as Mg 80 / Cu 20 against a
    site table that reads Cu 66.7 / Mg 33.3.

    WHAT IT CLAIMS IS WHAT IT CHECKED. An earlier version said the row "came
    from a refused parse" -- a provenance nothing here can know, and one that
    would have stayed on the screen after the user corrected the cell, since
    the CIF is refused either way. This compares two numbers and reports the
    two numbers, so fixing the cell clears it.

    Silence has two meanings and both are honest: the compositions agree, or
    the CIF has no Wyckoff column to read (21 of 36 shipped files) and there
    is no second opinion to be had.

    Costs a text scan -- 9.7 ms for the whole shipped library, measured -- so
    it runs on the load path for every caller rather than behind the one that
    happens to display it. An earlier version parsed each listed CIF to ask
    the same question and cost 7.2 s.
    """
    # Cleared on EVERY entry first, not on the ones this pass happens to
    # reach. The entries live in the library cache and this runs again on
    # every rebuild: a mark left on a row whose CIF has since left the folder,
    # or whose composition has been corrected, would otherwise never come off.
    for entry in library.values():
        entry.suspect = False
        entry.suspect_reason = ""

    out: List[Dict[str, str]] = []
    for q in paths:
        entry = library.get(q.name)
        if entry is None or not entry.composition:
            continue                     # the skip list covers those
        reference = site_table_at_pct(q)
        if not reference:
            continue                     # no second opinion to be had
        elements = set(reference) | set(entry.composition)
        worst_el = max(elements, key=lambda e: abs(
            reference.get(e, 0.0) - entry.composition.get(e, 0.0)))
        worst = abs(reference.get(worst_el, 0.0)
                    - entry.composition.get(worst_el, 0.0))
        if worst < _SUSPECT_AT_PCT:
            continue
        stored = _fmt_at_pct(entry.composition)
        table = _fmt_at_pct(reference)
        logger.warning(
            "%s is offered as a phase with the composition %s from the "
            "database spreadsheet, but its own atom-site table reads %s "
            "(%s differs by %.1f at%%). The spreadsheet wins, so that is the "
            "composition in use.", q.name, stored, table, worst_el, worst)
        out.append({
            "file": q.name,
            "code": "suspect_listed_row",
            "reason": (f"the phase database stores {stored} for this file, "
                       f"but its own atom-site table reads {table} "
                       f"({worst_el} differs by {worst:.1f} at%). The stored "
                       f"value is the one in use."),
        })
        entry.suspect = True
        entry.suspect_reason = out[-1]["reason"]
    return out


def _fmt_at_pct(at: Dict[str, float]) -> str:
    return " ".join(f"{el} {v:.1f}" for el, v in sorted(at.items()))


def candidates_for(
    cif_library: Dict[str, CifPhaseEntry], measured_elements,
    unmeasured=None,
) -> List[CifPhaseEntry]:
    """Library entries whose elements are all present in the measurement.

    A phase whose elements are not all measured cannot plausibly be
    assigned — we would be matching against zeros for the missing ones.

    THE ORDER OF THIS LIST IS LOAD-BEARING. Its index is the phase id
    written into :class:`PhaseMapStore`, persisted in the ``.npz`` sidecar,
    and handed to indexing by ``get_phase_masks``. Every caller must build
    the candidate list through this one function, or two code paths can
    silently disagree about what phase id 3 means.

    ``unmeasured`` names elements whose window could not be priced. They are
    not evidence against a phase -- nobody looked -- so a phase is kept when
    its remaining elements were measured. Without this, a scan whose Cu window
    cannot be priced loses every Cu-bearing phase from the candidate list
    before scoring begins, and the map simply does not contain them.
    """
    #
    # A phase made of NOTHING BUT unmeasured elements is dropped even so.
    # Keeping it was the opposite mistake: there is no evidence for or
    # against it, both scorers end up with an empty phase side, and the
    # fail-soft they return there is 1.0 -- the TOP of the range, so argmax
    # hands it every pixel and `min_score` cannot filter a perfect score.
    # Measured on a map of Fe 20 / Si 80 with Al unmeasured: `Al.cif` took
    # the whole map at 1.000 where Fe3Si had won before.
    blind = set(unmeasured or ())
    measured = set(measured_elements) | blind
    return [entry for entry in cif_library.values()
            if set(entry.elements).issubset(measured)
            and set(entry.elements) - blind]


def group_degenerate_entries(
    candidates: List[CifPhaseEntry], max_at_pct_sep: float = 3.0,
) -> List[List[int]]:
    """Group candidates the available chemistry cannot separate.

    Two entries are degenerate when their nominal compositions differ by at
    most ``max_at_pct_sep`` at% on every element. Grouping is transitive
    (single-linkage) and emitted in first-member order so group ids stay
    stable across runs.

    Why this exists: on the user's 22-candidate Al-alloy library, 11 pairs
    differ by less than 8 at% and ``Al6Fe_mp-570001`` vs ``beta-AlFeSi`` by
    1.1 at% — far below the error of standardless Cliff-Lorimer. Picking a
    winner between those by ``argmax`` is a coin flip decided by spreadsheet
    row order. Callers use the grouping to tell ties apart from real
    disagreements.
    """
    n = len(candidates)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(n):
        ci = candidates[i].composition
        for j in range(i + 1, n):
            cj = candidates[j].composition
            els = set(ci) | set(cj)
            if not els:
                continue
            if max(abs(ci.get(e, 0.0) - cj.get(e, 0.0)) for e in els) <= max_at_pct_sep:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [groups[k] for k in sorted(groups)]


def auto_classify_pixels(
    at_pct_per_element: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    cif_library: Dict[str, CifPhaseEntry],
    tolerance: float = 15.0,
    min_score: float = 0.3,
    rule_set=None,
    unmeasured=None,
) -> Tuple[np.ndarray, np.ndarray, List[CifPhaseEntry], np.ndarray]:
    """Classify every pixel against the CIF library, vectorised.

    Scores each candidate with :func:`chemistry_score.score_phase_ratio`
    — renormalised L1 over the metallic elements, with an absolute and a
    relative missing-major veto. The phase with the highest score wins;
    pixels whose best score stays below ``min_score`` are marked
    unclassified (-1).

    Changed 2026-08-19: the previous rule averaged the At.% deviation over
    the *phase's* elements and rescaled it by ``tolerance``. That let a large
    deviation on one element be diluted by small deviations on the others,
    with no veto for a missing major element, so a phase requiring 11.6 at%
    Fe was assigned to pixels measuring 3 at% Fe — 20.95 % of all pixels on
    SampleB. ``tolerance`` is retained for API compatibility and no longer
    affects scoring.

    Args:
        at_pct_per_element: pre-computed atomic-percent maps as 1D
            arrays of length ``n_rows * n_cols``. Element keys must be
            bare symbols (``"Fe"``), not H5OINA full names.
        n_rows, n_cols: scan grid shape (used only to reshape the
            output, not the inputs).
        cif_library: phases to consider (typically loaded with
            :func:`load_cif_phase_library`).
        tolerance: per-element At.% tolerance — same default as
            :func:`eds_utils.suggest_phases`.
        min_score: pixels whose best phase scores below this are
            marked unclassified rather than forced into the closest
            (often nonsense) bucket.
        rule_set: optional user-authored rules. Each decides which phases
            may COMPETE for a pixel - not how well they score - so a
            blocked phase loses even to a poorly-scoring one that is
            allowed. See :mod:`backend.api.services.phase_rules`.

    Returns:
        ``(phase_index_grid, score_grid, candidate_entries, ambiguous_grid)``:
        - ``phase_index_grid``: int32 (n_rows, n_cols), -1 where
          unclassified, else the index into ``candidate_entries``.
        - ``score_grid``: float32 (n_rows, n_cols), the winning
          phase's score per pixel (0..1).
        - ``candidate_entries``: list of :class:`CifPhaseEntry` whose
          index in the list matches the ids in ``phase_index_grid``.
          Empty if no phase passed the element-subset pre-filter.
        - ``ambiguous_grid``: bool (n_rows, n_cols), True where the
          runner-up from a *different* degeneracy group scored within
          :data:`TIE_TOLERANCE`. A tie between two members of the same
          group is not ambiguity — chemistry cannot tell them apart, so
          they are the same answer.
    """
    n_pixels = n_rows * n_cols
    if not cif_library or not at_pct_per_element:
        empty = np.full((n_rows, n_cols), -1, dtype=np.int32)
        return (empty, np.zeros((n_rows, n_cols), dtype=np.float32), [],
                np.zeros((n_rows, n_cols), dtype=bool))

    # `unmeasured` elements are not evidence against a phase -- nobody
    # looked. Without it a scan whose Cu window cannot be priced loses every
    # Cu-bearing phase here, before scoring begins, and the map simply does
    # not contain them.
    candidates = candidates_for(cif_library, at_pct_per_element.keys(),
                                unmeasured=unmeasured)
    if not candidates:
        empty = np.full((n_rows, n_cols), -1, dtype=np.int32)
        return (empty, np.zeros((n_rows, n_cols), dtype=np.float32), [],
                np.zeros((n_rows, n_cols), dtype=bool))

    # Score every candidate phase for every pixel. The grid stays
    # (n_candidates, n_pixels) since we argmax along the candidate axis
    # per pixel at the end.
    matrix_element = infer_matrix_element(at_pct_per_element)
    if rule_set is not None and rule_set.matrix_elements:
        # The user's declaration wins over the inference. `infer_matrix_element`
        # returns None below 40 at% and its own docstring warns that "this
        # choice inverts the whole metric if it is wrong" - so when the user
        # has said which element is the matrix, that is the answer.
        matrix_element = rule_set.matrix_elements[0]
    # Resolved ONCE here and handed to the scorer and the rules alike: cluster
    # mode passes a background explicitly while this path used to let the
    # scorer recompute one internally, and two definitions of "background" is
    # how the two modes drift apart.
    background = background_levels(at_pct_per_element)
    score_per_phase = np.zeros((len(candidates), n_pixels), dtype=np.float32)
    blocked_reasons = {}
    for i, entry in enumerate(candidates):
        # no_data_score=0.0: an unmeasured pixel must fall below min_score
        # and come out unclassified, never win argmax at a perfect 1.0.
        s_i = score_phase_ratio(
            at_pct_per_element, entry.composition,
            matrix_element=matrix_element, no_data_score=0.0,
            background=background, unmeasured=unmeasured,
        )
        # User rules decide ELIGIBILITY, before the argmax below. Same
        # evaluator as cluster mode; there it judges a cluster mean, here a
        # pixel.
        if rule_set is not None and not rule_set.is_empty:
            from backend.api.services.phase_rules import gate_scores
            s_i, outcome = gate_scores(
                s_i, rule_set.rule_for(entry.key), at_pct_per_element,
                background=background)
            if outcome is not None and not outcome.allowed.any():
                blocked_reasons[i] = outcome.reason
        score_per_phase[i] = s_i

    idx = np.arange(n_pixels)
    best_phase = np.argmax(score_per_phase, axis=0)
    best_score = score_per_phase[best_phase, idx]

    classified = best_score >= float(min_score)
    phase_grid = np.where(classified, best_phase, -1).astype(np.int32)

    # Ambiguity: a runner-up from a DIFFERENT degeneracy group scoring
    # within TIE_TOLERANCE means chemistry genuinely cannot decide, and
    # argmax would be breaking the tie on library row order.
    group_of = np.zeros(len(candidates), dtype=np.int32)
    for gid, members in enumerate(group_degenerate_entries(candidates)):
        for m in members:
            group_of[m] = gid

    ambiguous = np.zeros(n_pixels, dtype=bool)
    if len(candidates) > 1:
        masked = score_per_phase.copy()
        masked[best_phase, idx] = -np.inf          # exclude the winner
        same_group = group_of[:, None] == group_of[best_phase][None, :]
        masked[same_group] = -np.inf               # exclude its degenerate twins
        runner_up = masked.max(axis=0)
        ambiguous = np.isfinite(runner_up) & ((best_score - runner_up) <= TIE_TOLERANCE)
    ambiguous &= classified

    return (
        phase_grid.reshape(n_rows, n_cols),
        best_score.reshape(n_rows, n_cols).astype(np.float32),
        candidates,
        ambiguous.reshape(n_rows, n_cols),
    )
