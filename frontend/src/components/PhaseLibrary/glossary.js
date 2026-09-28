/**
 * The eight terms this page uses without explaining them (spec §2.3).
 *
 * They are not jargon to the person who wrote the library and they are
 * opaque to everyone else: a first-time-user agent read "cF4" and "Cu" in a
 * card about aluminium and concluded the entry was wrong. `Cu` there is the
 * STRUCTURE TYPE -- pure Al and pure Ni have it too -- and nothing on the
 * screen said so.
 *
 * Each term is explained in two places, because the two jobs are different:
 * on the term itself, for someone who has stopped at it, and in one
 * collapsible legend, for someone who wants to read the lot. The legend is
 * also the keyboard path: a tooltip that only appears on hover is not a path
 * at all (§2.9).
 *
 * `abbr: true` marks a real abbreviation, which gets the `<abbr>` element;
 * the others are terms and get `<span>`. The distinction is not decoration
 * -- a screen reader treats them differently.
 */
export const GLOSSARY = [
  { key: 'pearson', abbr: false },
  { key: 'prototype', abbr: false },
  { key: 'itNumber', abbr: true },
  { key: 'cif', abbr: true },
  { key: 'xtal', abbr: true },
  { key: 'master', abbr: false },
  { key: 'sht', abbr: true },
  { key: 'ht', abbr: true },
  // Added after the user loop. "Placements" is in the headline of the page
  // and was the one term of art the glossary did not define -- a reader
  // called that out in as many words. "by name only" is a match tag the
  // reader is expected to act on.
  { key: 'placements', abbr: false },
  { key: 'labelOnly', abbr: false },
];

export const GLOSSARY_KEYS = GLOSSARY.map((g) => g.key);

/** Look one up; unknown keys are a programming error, not a blank tooltip. */
export function glossaryEntry(key) {
  const found = GLOSSARY.find((g) => g.key === key);
  if (!found) {
    throw new Error(`no glossary entry for "${key}" -- add it to GLOSSARY`);
  }
  return found;
}
