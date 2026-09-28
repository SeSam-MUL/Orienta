/**
 * The facets: which elements a phase contains, and which methods can use it.
 *
 * Both kinds live here because they share ONE counting rule, and a rule kept
 * in two files is a rule that drifts.
 *
 * WHAT A COUNT ON A CHIP MEANS, decided once here because the mockups got it
 * wrong in the way that makes facets worthless: a number that does not match
 * what clicking produces. Three of four personas clicked a count that did
 * nothing, and §2.8 is a whole section about counters that lie.
 *
 * The rule: a chip's count is WHAT YOU WOULD SEE IF YOU CLICKED IT NOW. It is
 * computed against the phases currently shown -- after the text search and
 * after every element already chosen. So the number never disagrees with the
 * click, including the case where it reaches 0, and a chip at 0 is shown and
 * disabled rather than hidden: a facet that vanishes takes with it the
 * information that the element exists at all.
 *
 * Elements combine with AND, not OR. `Al-Fe-Si` in the search box already
 * means "all three present" (spec §2.1), and a facet that meant something
 * else than the search box would be two different languages for one
 * question. Measured on the real library: Al 28, +Fe 18, +Si 13, +Mn 4 --
 * narrowing, which is what a person picking elements off a list expects.
 *
 * LABEL-ONLY MEMBERSHIP IS NOT A MATCH HERE. `beta-AlFeSi` is named Si and
 * its structure has none. The search says so in words ("nur laut Etikett");
 * a facet has no room for that, and a silicon facet that yields a phase with
 * no silicon is a wrong answer, not a nuanced one. The facets count
 * structures. The search remains the way to find a phase by the name it
 * carries.
 */

/** Elements present in a phase's STRUCTURE, as a Set. */
function structureElements(phase) {
  return new Set(phase.elements_structure || []);
}

/**
 * Every element the library contains, with the count each chip should show.
 *
 * @param {Array} allPhases   the whole library (defines which chips exist)
 * @param {Array} shownPhases what is on screen right now
 * @param {string[]} selected elements already chosen
 * @returns {Array<{symbol, count, selected, disabled}>}
 *   sorted by count descending, then alphabetically -- the order a person
 *   scans, with the elements that actually occur in this library at the top.
 */
export function elementFacets(allPhases, shownPhases, selected = []) {
  const chosen = new Set(selected);
  const symbols = new Set();
  for (const p of allPhases) for (const e of structureElements(p)) symbols.add(e);

  const rows = [...symbols].map((symbol) => {
    // Already chosen: every shown phase has it, so the honest number is the
    // size of what is shown -- clicking it again only removes it.
    const count = chosen.has(symbol)
      ? shownPhases.length
      : shownPhases.filter((p) => structureElements(p).has(symbol)).length;
    return { symbol, count, selected: chosen.has(symbol), disabled: count === 0 };
  });

  rows.sort((a, b) => b.count - a.count || a.symbol.localeCompare(b.symbol));
  return rows;
}

/** Phases whose STRUCTURE contains every chosen element. */
export function applyElementFacets(phases, selected = []) {
  if (!selected.length) return phases;
  return phases.filter((p) => {
    const have = structureElements(p);
    return selected.every((e) => have.has(e));
  });
}

/** Add or remove one element. Kept here so the page holds no set logic. */
export function toggleElement(selected, symbol) {
  return selected.includes(symbol)
    ? selected.filter((e) => e !== symbol)
    : [...selected, symbol];
}

/**
 * Above this many chips the facet gets its own search box (spec, task 4).
 *
 * This library has nine, so the box never appears on it -- the number is for
 * the libraries this app will meet, not for the one in front of us, and the
 * test constructs the case rather than pretending the fixture covers it.
 */
export const CHIP_SEARCH_THRESHOLD = 15;

/**
 * Narrow the chip list by a typed fragment: the symbol, or the element's
 * name in any language the table knows.
 *
 * A CHOSEN element is never filtered away. Hiding it would hide the only
 * control that undoes it, which is the §2.8 trap in miniature: the filter
 * that is biting has to stay reachable from inside the state it creates.
 */
export function filterChips(rows, text, nameLookup = () => []) {
  const q = String(text || '').trim().toLowerCase();
  if (!q) return rows;
  return rows.filter((r) => r.selected
    || r.symbol.toLowerCase().startsWith(q)
    || nameLookup(r.symbol).some((n) => n.toLowerCase().startsWith(q)));
}


// ---------------------------------------------------------------------------
// Capabilities (spec §2.7): three questions with three different answers
// ---------------------------------------------------------------------------

/**
 * Measured on the real library, and each number was wrong somewhere before:
 *
 *   Hough       needs a CIF                              36 / 36
 *   Spherical   needs an .sht                            29 / 36
 *   Dictionary  needs an EMsoft master OR a pre-built
 *               dictionary                               16 / 36, 3 pre-built
 *
 * Fassung 2 of the spec said "Master 4" -- it had counted the pre-built
 * dictionaries. The shipped app answers a third number, 2, because
 * `_master_h5_for_phase` matches on name fragments of four characters or
 * more and `Al` and `Ni` fall through. Three answers to one question is
 * exactly what these three facets exist against.
 *
 * The capability hangs on the CONTENT of the file, never on its name: 65
 * `.h5` sit under EBSD_H5_Cache, 19 have "master" in the name, and 20 carry
 * `EMData/EBSDmaster/mLPNH`. The contract carries the answer; this module
 * only reads it.
 */
export const CAPABILITIES = ['hough', 'spherical', 'dictionary'];

function can(phase, capability) {
  return Boolean((phase.capabilities || {})[capability]);
}

/** Same rule as the element chips: the count is what clicking gives you. */
export function capabilityFacets(allPhases, shownPhases, selected = []) {
  const chosen = new Set(selected);
  return CAPABILITIES.map((capability) => {
    const count = chosen.has(capability)
      ? shownPhases.length
      : shownPhases.filter((p) => can(p, capability)).length;
    return {
      capability,
      count,
      // How many of the WHOLE library can do this, and how big the library
      // is. Both, because "29" alone invites the reader to divide it by the
      // number in front of them, which is the filtered count.
      capableInLibrary: allPhases.filter((p) => can(p, capability)).length,
      libraryTotal: allPhases.length,
      selected: chosen.has(capability),
      disabled: count === 0,
    };
  });
}

/** Phases that can be indexed by every chosen method. */
export function applyCapabilityFacets(phases, selected = []) {
  if (!selected.length) return phases;
  return phases.filter((p) => selected.every((c) => can(p, c)));
}

/** Add or remove one capability. */
export function toggleCapability(selected, capability) {
  return selected.includes(capability)
    ? selected.filter((c) => c !== capability)
    : [...selected, capability];
}
