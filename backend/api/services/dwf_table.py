"""Debye-Waller lookup for the CIF -> .xtal converter (``DWF.xlsx``).

The table is keyed on the element, but a few elements have more than one
published B(300 K) value depending on the crystal structure they were measured
in (Peng et al. 1996, Table 1, lists Fe twice: BCC 0.3328 and FCC 0.5710 A^2).
Such an element has one row per structure, the structure in brackets after the
symbol: ``Fe (BCC)``, ``Fe (FCC)``.

Rule for which row a phase gets (``build_dwf_map``):

* an element with a single row: that row, always;
* an element with several rows and a phase that IS that element's own crystal
  (an elemental structure in the cubic F, cubic I, hcp or diamond space group
  of one of the labels): the row whose label names that structure;
* anything else (the element sits in a compound, an alloy or an unlabelled
  structure): the FIRST row of the element. The table lists the room-
  temperature stable structure first (``Fe (BCC)``), so this is the value used
  before the variant rows existed.

Nothing is guessed: a label that no row carries falls through to the first row.
"""
from __future__ import annotations

from typing import Iterable, Optional

#: Space group of an elemental crystal -> the structure label it carries in the
#: table. Only used when the phase contains exactly one element.
ELEMENTAL_STRUCTURE_BY_SPACE_GROUP = {
    225: "FCC",   # Fm-3m, cF4
    229: "BCC",   # Im-3m, cI2
    194: "HCP",   # P6_3/mmc, hP2
    227: "DIA",   # Fd-3m, cF8 (diamond)
}


def split_label(label) -> tuple[str, Optional[str]]:
    """``"Fe (BCC)"`` -> ``("Fe", "BCC")``; ``"Al"`` -> ``("Al", None)``."""
    text = str(label).strip()
    if not text:
        return "", None
    symbol = text.split()[0]
    variant = None
    if "(" in text and ")" in text:
        inner = text[text.index("(") + 1:text.rindex(")")].strip()
        variant = inner.upper() or None
    return symbol, variant


def build_dwf_map(
    rows: Iterable[tuple],
    *,
    spg_number: Optional[int] = None,
    n_elements: Optional[int] = None,
) -> dict[str, float]:
    """``{element symbol: B in nm^2}`` for one phase.

    Args:
        rows: ``(label, value)`` pairs in table order (``label`` as in the
            ``Element`` column, so possibly ``"Fe (FCC)"``).
        spg_number: space-group number of the phase being converted.
        n_elements: number of distinct elements in the phase; the structure-
            specific rows apply only when this is 1 (an elemental crystal).
    """
    by_symbol: dict[str, list[tuple[Optional[str], float]]] = {}
    for label, value in rows:
        symbol, variant = split_label(label)
        if symbol:
            by_symbol.setdefault(symbol, []).append((variant, float(value)))

    wanted = None
    if n_elements == 1 and spg_number is not None:
        wanted = ELEMENTAL_STRUCTURE_BY_SPACE_GROUP.get(int(spg_number))

    out: dict[str, float] = {}
    for symbol, candidates in by_symbol.items():
        chosen = candidates[0][1]
        if wanted is not None:
            for variant, value in candidates:
                if variant == wanted:
                    chosen = value
                    break
        out[symbol] = chosen
    return out
