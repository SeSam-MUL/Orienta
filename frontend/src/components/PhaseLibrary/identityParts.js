/**
 * The identity line (spec §2.2):
 *
 * NAMED `identityParts`, NOT `identityLine`, because `IdentityLine.jsx` sits
 * beside it and Windows cannot tell the two apart -- the second time that
 * collision happened in this folder in one afternoon (the first was
 * `ElementFacets.jsx` / `elementFacets.js`). The convention here: a logic
 * module never shares a name with a component, case aside.
 *
 *
 *     [synonym] · formula · Pearson · space group · source      (key, small)
 *
 * THE FORMULA COMES BEFORE PEARSON AND IS NEVER LEFT OUT. Fassung 1 of the
 * spec made Pearson the distinguishing field; it is not.
 * `alpha-AlFeMnSi_ICSD-52623` and `α-(AlMnSi)` are both cP138 and both Pm-3,
 * and the only thing that separates them on a line is whether there is iron
 * in the formula.
 *
 * WHERE THE NAME CAME FROM IS VISIBLE. A name out of `_sm_phase_labels` is a
 * datum; one read off the file stem is a guess. Fifteen of the 36 have the
 * first kind. They must not look the same.
 *
 * A SOURCE IS CHECKED BEFORE IT APPEARS, and when it fails the check there
 * is NOTHING there -- "a missing field shows nothing" covers rubbish as well
 * as emptiness. The checking happens in the backend (`reference` is null
 * when it failed); this module only declines to invent a placeholder.
 */

/** Pearson's first two letters, decoded. */
const SYSTEM_BY_LETTER = {
  a: 'triclinic', m: 'monoclinic', o: 'orthorhombic',
  t: 'tetragonal', h: 'hexagonal', c: 'cubic',
};
const CENTRING_BY_LETTER = {
  P: 'primitive', I: 'bodyCentred', F: 'faceCentred',
  C: 'baseCentred', A: 'baseCentred', B: 'baseCentred', S: 'baseCentred',
  R: 'rhombohedral',
};

/**
 * `cF4` -> { system: 'cubic', centring: 'faceCentred', atoms: 4 }.
 *
 * Used for the structure-type parenthetical, and that is the point of it: a
 * first-time reader took "Prototype: Cu" on the aluminium row for copper IN
 * the aluminium. `Structure type Cu (face-centred cubic)` cannot be read
 * that way. The words come from THIS phase's Pearson symbol, which is sound
 * because a phase of structure type Cu has the Cu structure.
 *
 * Returns nulls rather than guesses for anything it does not recognise.
 */
export function pearsonParts(symbol) {
  const s = String(symbol || '').trim();
  const m = s.match(/^([amothc])([PIFCABSR])(\d+)?$/);
  if (!m) return { system: null, centring: null, atoms: null };
  return {
    system: SYSTEM_BY_LETTER[m[1]] || null,
    centring: CENTRING_BY_LETTER[m[2]] || null,
    atoms: m[3] ? Number(m[3]) : null,
  };
}

/**
 * A space group as it should be READ, not as the file happens to write it.
 *
 * This library writes them four ways -- `Fm-3m`, `P m -3`, `C 1 2/c 1`,
 * `A 1 2/a 1` -- and the last two are the monoclinic long form, where the
 * two `1`s are the axes the symmetry says nothing about. Dropping them is
 * the conventional short form, not a loss.
 */
export function displaySpaceGroup(hm) {
  const raw = String(hm || '').trim();
  if (!raw) return null;
  const tight = raw.replace(/\s+/g, '');
  // `C121/c1` -> `C2/c`: leading and trailing axis 1 of the long form.
  const short = tight.replace(/^([A-Z])1(.+?)1$/, '$1$2');
  return short || null;
}

/**
 * Where the name on the line came from: 'label', 'structure' or 'stem'.
 *
 * THE FIRST VERSION OF THIS READ THE WRONG FIELD. It keyed on
 * `elements_label_source === 'filename'`, on the assumption that some names
 * are read off the stem. Measured: on this library ALL FIFTEEN labels come
 * from `_sm_phase_labels`, and the three phases whose element source is the
 * filename have no label at all -- so that branch could never fire, and the
 * §2.2 requirement it was meant to serve would have gone unmet while
 * looking implemented.
 *
 * The guessed name a reader actually sees is the FILE STEM, which is what
 * the row falls back to when there is no label and no structure formula.
 * That is the case worth marking.
 *
 *   'label'     a recorded `_sm_phase_labels` value -- a datum
 *   'structure' computed from the structure in the file -- also a datum,
 *               but a different one, and the card shows both side by side
 *   'stem'      nothing but the file name -- a guess, and it says so
 */
export function nameProvenance(phase) {
  if (phase.formula_label) return 'label';
  if (phase.formula_structure) return 'structure';
  return 'stem';
}

/**
 * The short line, for a band row.
 *
 * WHAT IT HAS TO CARRY WAS MEASURED, not chosen. Two rows in this library
 * are both called `Mn0.5Fe0.5Al5Si0.68`, and every reader in the user loop
 * got stuck on them -- one said plainly "I'd click the first one and hope,
 * and I'd have a coin-flip in my results without knowing it". They share
 * their label, their Pearson symbol (cI168), their space group (Im-3) and
 * their atom count (168). A line of formula and Pearson would have left
 * them exactly as indistinguishable as before.
 *
 * What differs is the LATTICE PARAMETER -- 12.5 against 12.56 Å -- and the
 * file. So the short line is Pearson · a · key, and the lab head who asked
 * for "a, b, c on the line, because two competing models of one phase
 * differ by their cell" gets the first of those where it matters most.
 */
export function compactIdentityParts(phase) {
  const parts = [];
  if (phase.pearson) parts.push({ kind: 'pearson', text: phase.pearson });
  const a = phase.cell && phase.cell.a;
  if (typeof a === 'number') {
    parts.push({ kind: 'cellA', text: `a ${round(a)} Å`, value: a });
  }
  parts.push({ kind: 'key', text: phase.key });
  return parts;
}

/** Four decimals is a tenth of a picometre; no cell in this library needs
 *  more, and trailing zeros go so 12.5 stays 12.5. */
function round(value) {
  return Number(value.toFixed(4));
}

/**
 * The line, as an ordered list of parts. The component decides how to draw
 * them; this decides what is on it and in which order.
 *
 * @returns {Array<{kind: string, text: string, ...}>}
 */
export function identityParts(phase, { synonym = null } = {}) {
  const parts = [];
  if (synonym) parts.push({ kind: 'synonym', text: synonym });

  const provenance = nameProvenance(phase);
  if (provenance === 'label') {
    parts.push({ kind: 'formula', text: phase.formula_label, provenance });
  } else if (provenance === 'structure') {
    // No recorded label, so the structure's own formula stands in.
    parts.push({ kind: 'formula', text: phase.formula_structure, provenance });
  } else {
    // Neither. The row shows the file stem, and a file stem is a guess --
    // `sd_1816951` is a name only in the sense that somebody typed it.
    parts.push({ kind: 'formula', text: phase.key, provenance });
  }

  if (phase.pearson) {
    parts.push({ kind: 'pearson', text: phase.pearson, ...pearsonParts(phase.pearson) });
  }

  const sg = displaySpaceGroup(phase.space_group_hm);
  if (sg) parts.push({ kind: 'spaceGroup', text: sg, it: phase.space_group_it || null });

  // A source, or nothing. Never "unknown", never a placeholder.
  //
  // BOTH, WHEN THERE ARE BOTH. This was `else if`, so a DOI was shown only
  // for a phase with no citation -- and 4 of the 36 phases in this library
  // carry both, which meant the one thing a reader wants for a cited
  // phase, the link to the paper, was suppressed exactly where the paper
  // existed. The manual promised "the citation, with its DOI"; nothing on
  // the card could deliver it.
  if (phase.reference) parts.push({ kind: 'reference', text: phase.reference });
  if (phase.doi) parts.push({ kind: 'doi', text: phase.doi });

  return parts;
}
