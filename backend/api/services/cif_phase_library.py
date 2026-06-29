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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Reuse the project-wide subscript translation so the parsing matches
# whatever the database builder writes into the xlsx.
_SUB_TO_ASCII = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


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


# In-memory cache keyed by xlsx path. Invalidated on mtime change so that
# the user can rebuild the database without restarting the backend.
_LIBRARY_CACHE: Dict[str, Tuple[float, Dict[str, CifPhaseEntry]]] = {}


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


def load_cif_phase_library(xlsx_path: Path) -> Dict[str, CifPhaseEntry]:
    """Load and cache the CIF-derived phase library.

    The cache key is the absolute path of the xlsx; the cache is
    invalidated when the file's mtime changes so a freshly rebuilt
    database is picked up automatically. Returns an empty dict if the
    file does not exist or pandas cannot read it.
    """
    p = Path(xlsx_path).resolve()
    if not p.is_file():
        return {}

    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}

    cached = _LIBRARY_CACHE.get(str(p))
    if cached is not None and cached[0] == mtime:
        return cached[1]

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

    _LIBRARY_CACHE[str(p)] = (mtime, library)
    return library


def suggest_phases_from_cif_library(
    measured_at_pct: Dict[str, float],
    cif_library: Dict[str, CifPhaseEntry],
    tolerance: float = 15.0,
    min_score: float = 0.3,
    max_results: int = 20,
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

    measured_set = {el for el, val in measured_at_pct.items() if val > 0.0}
    simple_lib: Dict[str, Dict[str, float]] = {}
    meta_by_key: Dict[str, CifPhaseEntry] = {}
    for key, entry in cif_library.items():
        if not set(entry.elements).issubset(measured_set):
            continue
        simple_lib[key] = entry.composition
        meta_by_key[key] = entry

    if not simple_lib:
        return []

    # Lazy import keeps this module importable in test contexts that
    # don't ship eds_utils (e.g. mocked smoke tests).
    from eds_utils import suggest_phases  # noqa: PLC0415

    matches = suggest_phases(measured_at_pct, simple_lib, tolerance=tolerance)
    enriched: List[Dict] = []
    for m in matches:
        if m.score < min_score:
            continue
        meta = meta_by_key.get(m.phase_name)
        if meta is None:
            continue
        enriched.append({
            "key": meta.key,
            "cif_filename": meta.cif_filename,
            "formula": meta.formula,
            "space_group": meta.space_group,
            "space_group_number": meta.space_group_number,
            "crystal_system": meta.crystal_system,
            "score": float(m.score),
            "expected": dict(m.expected_composition),
            "elements": list(meta.elements),
        })
    return enriched[:max_results]


def auto_classify_pixels(
    at_pct_per_element: Dict[str, np.ndarray],
    n_rows: int,
    n_cols: int,
    cif_library: Dict[str, CifPhaseEntry],
    tolerance: float = 15.0,
    min_score: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, List[CifPhaseEntry]]:
    """Classify every pixel against the CIF library, vectorised.

    Computes, for each candidate phase, a per-pixel score from the
    deviation between measured and expected At.% (same scoring rule as
    :func:`eds_utils.suggest_phases` — average deviation rescaled by
    tolerance, rejected when any single element exceeds ``2*tolerance``).
    The phase with the highest score wins; pixels whose best score
    stays below ``min_score`` are marked unclassified (-1).

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

    Returns:
        ``(phase_index_grid, score_grid, candidate_entries)``:
        - ``phase_index_grid``: int32 (n_rows, n_cols), -1 where
          unclassified, else the index into ``candidate_entries``.
        - ``score_grid``: float32 (n_rows, n_cols), the winning
          phase's score per pixel (0..1).
        - ``candidate_entries``: list of :class:`CifPhaseEntry` whose
          index in the list matches the ids in ``phase_index_grid``.
          Empty if no phase passed the element-subset pre-filter.
    """
    n_pixels = n_rows * n_cols
    if not cif_library or not at_pct_per_element:
        empty = np.full((n_rows, n_cols), -1, dtype=np.int32)
        return empty, np.zeros((n_rows, n_cols), dtype=np.float32), []

    # Pre-filter: a phase whose elements aren't all measured cannot
    # plausibly be assigned (we'd be matching against zeros for the
    # missing elements, dragging bogus phases in front of real ones).
    measured_set = set(at_pct_per_element.keys())
    candidates: List[CifPhaseEntry] = [
        entry for entry in cif_library.values()
        if set(entry.elements).issubset(measured_set)
    ]
    if not candidates:
        empty = np.full((n_rows, n_cols), -1, dtype=np.int32)
        return empty, np.zeros((n_rows, n_cols), dtype=np.float32), []

    # Score every candidate phase for every pixel in one vectorised
    # loop. The grid stays (n_candidates, n_pixels) since we argmax
    # along the candidate axis per pixel at the end.
    score_per_phase = np.zeros((len(candidates), n_pixels), dtype=np.float32)
    for i, entry in enumerate(candidates):
        # Stack per-element deviations so we get max & mean in one go.
        dev_stack: List[np.ndarray] = []
        for el, expected_pct in entry.composition.items():
            measured = at_pct_per_element.get(el)
            if measured is None:
                dev_stack = []
                break  # subset check should have caught this
            dev_stack.append(np.abs(measured.astype(np.float32) - float(expected_pct)))
        if not dev_stack:
            continue
        dev_arr = np.stack(dev_stack, axis=0)
        avg_dev = dev_arr.mean(axis=0)
        max_dev = dev_arr.max(axis=0)

        # Same scoring rule as eds_utils.suggest_phases — keeps the
        # batch path consistent with the click-to-suggest path.
        score = np.maximum(0.0, 1.0 - avg_dev / float(tolerance))
        # Hard reject if any single element is way off.
        score = np.where(max_dev > tolerance * 2.0, 0.0, score)
        score_per_phase[i] = score

    best_phase = np.argmax(score_per_phase, axis=0)
    best_score = score_per_phase[best_phase, np.arange(n_pixels)]

    classified = best_score >= float(min_score)
    phase_grid = np.where(classified, best_phase, -1).astype(np.int32)

    return (
        phase_grid.reshape(n_rows, n_cols),
        best_score.reshape(n_rows, n_cols).astype(np.float32),
        candidates,
    )
