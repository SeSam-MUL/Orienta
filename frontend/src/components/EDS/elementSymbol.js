/**
 * The element symbol behind an Aztec line name.
 *
 * H5OINA names its EDS windows after the X-ray line: "Al Kα1", "C Kα1,2",
 * "Fe Ka1". Everything downstream of the quantification works in plain
 * symbols, because `eds_utils.parse_element_name` strips the line on the way
 * in — so the at% maps, the region compositions and every rule evaluated
 * against them are keyed "Al", never "Al Kα1".
 *
 * That mismatch is not cosmetic. A rule or region clause written as
 * "Al Kα1" names an element the dataset does not have under that key, and
 * `phase_rules._series` correctly refuses to guess: the clause becomes
 * undecidable and blocks. Found by driving the real page against a real
 * Oxford file — the element dropdown was offering line names, so a
 * hand-written element clause could never match anything.
 *
 * Same rule as the backend, deliberately: split on whitespace, take the
 * first token.
 */
export function elementSymbol(entry) {
  const raw = typeof entry === 'string'
    ? entry
    : (entry?.symbol || entry?.element || entry?.name || '');
  return String(raw).trim().split(/\s+/)[0] || '';
}

/** Distinct symbols, in the order the elements arrived. */
export function elementSymbols(entries) {
  const seen = new Set();
  const out = [];
  (entries || []).forEach((e) => {
    const sym = elementSymbol(e);
    if (sym && !seen.has(sym)) { seen.add(sym); out.push(sym); }
  });
  return out;
}
