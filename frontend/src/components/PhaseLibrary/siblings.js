/**
 * "There is another entry that looks like this one."
 *
 * THE FINDING THIS ANSWERS, from a reader given the ordinary task "use the
 * alpha phase": `sd_0302719` and `sd_1401510` are indistinguishable in a
 * list -- the same composition on the label (`Mn0.5Fe0.5Al5Si0.68`), the
 * same space group `Im-3`, the same Pearson symbol `cI168`, lattice
 * parameters 12.50 and 12.56 A. Typing `Al Fe Si` returns one of them
 * FIRST and the other LAST of fifteen hits. Clicking the first and
 * carrying on is the natural thing to do, and nothing anywhere says the
 * second exists.
 *
 * They are two structure models of one phase, measured thirty years apart
 * -- Barlock & Mondolfo 1975 against Cooper 1967 -- and which one you index
 * with is a decision. It cannot be made by somebody who does not know
 * there is one.
 *
 * WHAT COUNTS AS A SIBLING: the same space group AND the same compound.
 * Not the same Pearson symbol on its own (`cF4` is aluminium and nickel
 * both), not the same formula on its own (the same compound in two
 * structure types is a genuinely different entry and the space group says
 * so). Both together is narrow enough that this library has exactly one
 * such pair, which is the number a warning should have.
 *
 * AND IT NAMES WHAT DIFFERS, because "there is another one" without that
 * is a nuisance rather than help. The differences are computed, never
 * curated: a list of known duplicates goes stale the day somebody adds a
 * CIF.
 */
import { parseFormula, sameFormula, normaliseSpaceGroup } from './phaseSearch';

/** The compound a row states, from its label or, failing that, its cell. */
function compoundOf(p) {
  return parseFormula(p.formula_label) || parseFormula(p.formula_structure);
}

function spaceGroupOf(p) {
  return normaliseSpaceGroup(p.space_group_hm).core || null;
}

/**
 * How two entries that look alike are NOT alike.
 *
 * Ordered by how much it should weigh on the decision: what you can index
 * with first, because it is the only one that can make a phase unusable;
 * then what the structure is made of; then the paper; then the cell.
 */
export function differencesBetween(a, b) {
  const out = [];
  const caps = (p) => ['hough', 'spherical', 'dictionary']
    .filter((m) => (p.capabilities || {})[m]);
  const mine = caps(a);
  const theirs = caps(b);
  const extra = theirs.filter((m) => !mine.includes(m));
  const fewer = mine.filter((m) => !theirs.includes(m));
  if (extra.length) out.push({ kind: 'alsoIndexableWith', methods: extra });
  if (fewer.length) out.push({ kind: 'notIndexableWith', methods: fewer });

  const els = (p) => (p.elements_structure || []).join(' ');
  if (els(a) !== els(b)) {
    out.push({ kind: 'composition', theirs: els(b) || null });
  }
  // A citation is the thing that actually separated the two in practice.
  if ((a.reference || null) !== (b.reference || null) && b.reference) {
    out.push({ kind: 'reference', theirs: b.reference });
  }
  const aa = (a.cell || {}).a;
  const ba = (b.cell || {}).a;
  if (aa != null && ba != null && aa !== ba) {
    out.push({ kind: 'cellA', mine: aa, theirs: ba });
  }
  return out;
}

/**
 * Entries that share this one's space group and compound.
 *
 * Returns `[]` when the row has no space group or no readable formula --
 * "it might have a twin, we cannot tell" is not something to put on a
 * card as though it were a fact.
 */
export function siblingsOf(phases, key) {
  const list = Array.isArray(phases) ? phases : [];
  const self = list.find((p) => p.key === key);
  if (!self) return [];
  const sg = spaceGroupOf(self);
  const formula = compoundOf(self);
  if (!sg || !formula) return [];
  return list
    .filter((p) => p.key !== key
      && spaceGroupOf(p) === sg
      && sameFormula(formula, compoundOf(p)))
    .map((p) => ({
      key: p.key,
      display: p.display_name || p.formula_label || p.key,
      differences: differencesBetween(self, p),
    }));
}
