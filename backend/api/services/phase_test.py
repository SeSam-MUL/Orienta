"""Pure assembly logic for the Single-Pixel Phase Test (no GPU, no I/O).

- eds_prefilter: split library entries into tested vs EDS-excluded and
  compute each phase's chemistry-fit against the pixel's measured At%.
- combined_rank_score: how the ranked list orders phases per EDS mode.
"""
from __future__ import annotations

from typing import Optional

from backend.api.services.crystal_hint_phase_fit import (
    chemistry_fit, phase_nominal_at_pct,
)


def _chem_for(entry, pixel_at_pct: dict) -> Optional[float]:
    formula = getattr(entry, "formula", "") or ""
    phase_at = phase_nominal_at_pct(formula)
    if not phase_at:
        return None
    return chemistry_fit(pixel_at_pct, phase_at)


def eds_prefilter(
    entries: list,
    pixel_at_pct: Optional[dict],
    mode: str,
    threshold: float,
) -> tuple[list, list, dict, str]:
    """Return (tested, excluded, chem_by_key, effective_mode).

    - mode "off", or pixel_at_pct is None/empty: test everything; effective
      mode degrades to "off"; chem_by_key empty.
    - mode "soft": test everything (chemistry only re-ranks); chem_by_key set.
    - mode "filter": drop phases whose chemistry_fit < threshold into excluded.
    """
    if not pixel_at_pct or mode == "off":
        return list(entries), [], {}, "off"

    chem_by_key: dict[str, Optional[float]] = {}
    for e in entries:
        chem_by_key[e.key] = _chem_for(e, pixel_at_pct)

    if mode == "soft":
        return list(entries), [], chem_by_key, "soft"

    # mode == "filter"
    tested, excluded = [], []
    for e in entries:
        c = chem_by_key.get(e.key)
        # Unknown chemistry (unparseable formula) is NOT dropped — we can't
        # prove it's impossible, so keep it for the user to judge.
        if c is None or c >= threshold:
            tested.append(e)
        else:
            excluded.append(e)
    return tested, excluded, chem_by_key, "filter"


def combined_rank_score(
    r_score: Optional[float],
    chemistry_fit_val: Optional[float],
    mode: str,
) -> float:
    """Sort key (higher = better). None R sorts last."""
    if r_score is None or r_score != r_score:  # None or NaN
        return float("-inf")
    if mode == "soft" and chemistry_fit_val is not None:
        # Damp the pattern score by chemistry, but never below a floor weight
        # so a strong pattern match isn't fully erased by a soft chemistry miss.
        return max(r_score, 0.0) * (0.25 + 0.75 * float(chemistry_fit_val))
    return float(r_score)
