/**
 * The system bands: one per element, a phase in every band it belongs to.
 *
 * Measured on the real library: 93 placements in 9 bands (Al 28 · Fe 19 ·
 * Si 18 · Mg 9 · Mn 8 · Zn 5 · Cu 4 · Ni 1 · O 1). Thirty-six phases, and
 * ninety-three cards, because an Al-Fe-Si phase is genuinely part of three
 * systems and filing it under one of them is a decision the library has no
 * business making for the person reading it.
 *
 * THAT IS WHY THE COUNT LINE SAYS BOTH NUMBERS. "93" alone reads as a
 * library three times the size it is; "36" alone leaves the reader counting
 * the same card three times and wondering. The spec asks for "N Phasen, M
 * Platzierungen" in as many words, and this is the reason.
 *
 * BANDS GROW BY THEMSELVES. One band per element that occurs in at least one
 * STRUCTURE -- so a new Ti CIF produces a Ti band with no configuration, and
 * an element that is only claimed by a label produces none. The second half
 * matters: `beta-AlFeSi` is named Si and its structure has none, and a Si
 * band containing it would be the search's "nur laut Etikett" caveat
 * silently dropped, because a band has nowhere to put it.
 *
 * Order: by card count, then alphabetically. A band with one card is still a
 * band -- Ni and O each have exactly one here, and hiding a band because it
 * is small would hide the phase in it.
 */

/**
 * @param {Array} phases the phases to lay out (already searched and filtered)
 * @returns {{bands: Array<{element, phases}>, phaseCount, placementCount}}
 */
export function systemBands(phases) {
  const byElement = new Map();
  for (const p of phases) {
    for (const e of p.elements_structure || []) {
      if (!byElement.has(e)) byElement.set(e, []);
      byElement.get(e).push(p);
    }
  }

  const bands = [...byElement.entries()]
    .map(([element, members]) => ({ element, phases: members }))
    .sort((a, b) => b.phases.length - a.phases.length
                    || a.element.localeCompare(b.element));

  return {
    bands,
    phaseCount: phases.length,
    placementCount: bands.reduce((n, b) => n + b.phases.length, 0),
  };
}

/**
 * Phases whose structure names no element at all.
 *
 * They would otherwise fall out of the band view entirely -- present in the
 * library, absent from the screen, which is the worst way for a list to be
 * wrong. Measured: none in this library today, and that is exactly why the
 * case needs writing down rather than discovering later.
 */
export function phasesWithoutBand(phases) {
  return phases.filter((p) => !(p.elements_structure || []).length);
}
