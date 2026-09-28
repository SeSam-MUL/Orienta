/**
 * The phase library's search.
 *
 * This is the first thing built and everything downstream inherits it, so the
 * rules are written here once and measured against the real library.
 *
 * WHAT WENT WRONG BEFORE, all measured on the mockups and on the shipped app:
 *
 *   `Al`        -> 36 of 36, because "al" is inside "Journal" and "Metallkunde"
 *   `aluminium` -> 4 paper titles, and pure Al NOT among them
 *   `Ni`        -> 13, where the library holds exactly one Ni phase
 *   `Al-Fe-Si`  -> 1, where 13 phases contain all three elements
 *   `Al2CuMg`   -> 0, because the file writes the same formula as `MgCuAl2`
 *   `Im3`       -> 0, because the file writes `Im-3`
 *   `s phase`   -> 10, because "s" is inside almost every word
 *
 * A first-time user hit the fourth and fifth of those and said he would close
 * the program and ask a colleague for a list of filenames. So:
 *
 * TWO HIT CLASSES, AND THEY LOOK DIFFERENT ON SCREEN. A phase either IS the
 * thing you typed (its name, formula, elements, symmetry) or it is MENTIONED
 * by it (its citation). Mixing those is what produces "Al -> everything": the
 * citations are prose, and prose contains every short string. Class 2 is shown
 * under its own heading and never ranked among class 1.
 *
 * FOUR QUERY SHAPES, all typed into one box, decided by what the query looks
 * like rather than by a mode switch:
 *   element combination   `Al-Fe-Si`, `Al Fe Si`, `AlFeSi`
 *   formula as a multiset `Al2CuMg` finds the file that writes `MgCuAl2`
 *   space group           `Im3`, `I m -3`, `P63/mmc`, `P6_3/mmc`
 *   free text             everything else
 *
 * A query can match as several shapes at once; the union is returned, and each
 * hit records WHY it matched, because a hit with no visible reason reads as a
 * bug (a tester reported exactly that).
 */
import { ELEMENT_NAMES } from './elementNames';

// Greek is how this field writes phase names, and how three files in this very
// library are NAMED. A plain search for "alpha" finds none of them.
const GREEK = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ',
  epsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ',
  iota: 'ι', kappa: 'κ', lambda: 'λ', mu: 'μ',
  nu: 'ν', xi: 'ξ', pi: 'π', rho: 'ρ', sigma: 'σ',
  tau: 'τ', phi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω',
};
const GREEK_TO_WORD = Object.fromEntries(
  Object.entries(GREEK).map(([w, c]) => [c, w]));

const SYMBOL_BY_NAME = (() => {
  const m = new Map();
  for (const [sym, names] of Object.entries(ELEMENT_NAMES)) {
    for (const n of names) m.set(n, sym);
  }
  return m;
})();
const ALL_SYMBOLS = new Set(Object.keys(ELEMENT_NAMES));

/** Lower-case, decomposed, subscripts flattened, Greek spelled out. */
export function normalise(s) {
  if (s === null || s === undefined) return '';
  let out = String(s).normalize('NFKD').toLowerCase();
  out = out.replace(/[̀-ͯ]/g, '');          // combining marks
  out = out.replace(/[α-ω]/g, (c) => GREEK_TO_WORD[c] || c);
  return out;
}

/**
 * Space groups, folded to something comparable.
 *
 * The library and pymatgen between them write at least: `P63/mmc` and
 * `P6_3/mmc` (BOTH spellings for the same phase), `C 1 2/c 1` vs `C2/c`,
 * `Fd-3m O1`, `R -3 :H`, `P m -3`. Spaces and hyphens alone do not fold that,
 * which is why `P63/mmc` returned 0 or 4 depending on which source was indexed.
 *
 * The origin choice and setting suffix are SPLIT OFF rather than dropped: they
 * are a real difference (Si.cif says `Fd-3m` in one block and `Fd-3m O1` in
 * another) and the profile card shows them.
 */
export function normaliseSpaceGroup(s) {
  if (!s) return { core: '', suffix: null };
  let t = String(s).trim();
  let suffix = null;
  const m = t.match(/\s*[:]?\s*\b(O[12]|[HR])\s*$/i);
  if (m && /[:\s]/.test(t.slice(Math.max(0, m.index - 1), m.index + 1))) {
    suffix = m[1].toUpperCase();
    t = t.slice(0, m.index);
  }
  let core = t.toLowerCase().replace(/[\s_-]/g, '');
  // Monoclinic long form: `c121/c1` -> `c12/c1` is NOT enough; the redundant
  // axis 1s go entirely, so `C 1 2/c 1` meets `C2/c`.
  core = core.replace(/^([a-z])1(.+?)1$/, '$1$2');
  return { core, suffix };
}

/** `Al2CuMg` -> {al:2, cu:1, mg:1}. Element ORDER carries no information. */
export function parseFormula(s) {
  if (!s) return null;
  const t = String(s).normalize('NFKD')
    .replace(/[₀-₉]/g, (c) => String(c.charCodeAt(0) - 0x2080))
    // NOT the decimal point: this class used to hold `.` and `,`, so a cell
    // formula like `Al162.04Fe46Si30` became `Al162 04...` and parsed as 162.
    // Measured on the real library: that truncation is what made the one
    // proportional pair in it come out exact. `Al0.5Fe0.5` parsed as {al:0,fe:0}.
    .replace(/[()[\]{}·]/g, ' ')
    .replace(/,/g, ' ');
  const counts = {};
  let seen = false;
  const re = /([A-Z][a-z]?)(\d*\.?\d*)/g;
  let m;
  while ((m = re.exec(t)) !== null) {
    const sym = m[1];
    if (!ALL_SYMBOLS.has(sym)) return null;   // not a formula at all
    counts[sym.toLowerCase()] = (counts[sym.toLowerCase()] || 0)
      + (m[2] ? parseFloat(m[2]) : 1);
    seen = true;
  }
  if (!seen) return null;
  // Everything the regex skipped: if there is leftover text this is prose.
  if (t.replace(re, '').replace(/[\s\d.]/g, '').length) return null;
  return counts;
}

/** Element symbols a query names, whether written as symbols or as words. */
export function queryElements(query) {
  const raw = String(query).split(/[\s,\-–/]+/).filter(Boolean);
  const out = [];
  for (const w of raw) {
    const byName = SYMBOL_BY_NAME.get(normalise(w));
    if (byName) { out.push(byName); continue; }
    const cased = w[0].toUpperCase() + w.slice(1).toLowerCase();
    if (ALL_SYMBOLS.has(cased)) { out.push(cased); continue; }
    // `AlFeSi` written as one word
    const parts = w.match(/^([A-Z][a-z]?)+$/) ? w.match(/[A-Z][a-z]?/g) : null;
    if (parts && parts.every((p) => ALL_SYMBOLS.has(p))) { out.push(...parts); continue; }
    return null;                 // one unrecognised word -> not an element query
  }
  return out.length ? [...new Set(out)] : null;
}

/**
 * Fields that say what a phase IS, with the weight used for ranking.
 *
 * The third entry is a KEY, not a label. It was German prose until the
 * glossary went in -- `Raumgruppe`, `Strukturtyp`, `nur laut Etikett` --
 * rendered straight to the screen in an app that ships in four languages, so
 * an English or Japanese reader got the German word. The UI translates
 * these; `why.*` in the locale files.
 */
const IDENTITY_FIELDS = [
  ['display_name', 0, 'name'],
  ['synonyms', 0, 'synonym'],
  ['formula_label', 1, 'formula'],
  ['formula_structure', 1, 'formula'],
  ['elements', 1, 'element'],
  ['space_group', 2, 'spaceGroup'],
  ['pearson', 2, 'pearson'],
  ['key', 3, 'filename'],
];

/**
 * THE NAMES COME OFF THE ROW, and from nowhere else.
 *
 * This took a second argument for a while -- `buildIndex(phases, {synonyms})`
 * -- written before the endpoint carried names, and the only caller in the
 * program never passed it. Three tests injected synonyms themselves and
 * stayed green, so the search "supported" names that a running app could
 * never give it: the list showed `β-AlFeSi` while typing `β-AlFeSi` found
 * nothing. Two halves, each correct, and nothing standing where they meet --
 * the defect this branch has now shipped five times. One source.
 */
/**
 * What a phase is CALLED. One rule, one place.
 *
 * There were two: the band view asked `phaseLabel`, the hit lists used the
 * `display` the index had computed, and both happened to prefer the same
 * three fields in the same order. A mutation run found it -- unwiring the
 * index's names left the bands still showing them, so one of the two seam
 * tests passed over a broken search. Two spellings of one rule agree until
 * somebody edits one of them.
 *
 * A DISPLAY NAME STANDS IN FRONT OF THE KEY, NEVER INSTEAD OF IT. The key
 * is still in the row, still indexed, still searchable: the complaint this
 * page answers was "ich musste ewig suchen um das cif zu finden", and a
 * rename that hides the filename would bring it back.
 */
export function phaseLabel(phase) {
  return phase.display_name || phase.formula_label || phase.key;
}

export function buildIndex(phases) {
  return phases.map((p) => {
    const sg = normaliseSpaceGroup(p.space_group_hm);
    return {
      phase: p,
      key: p.key,
      display: phaseLabel(p),
      fields: {
        display_name: [p.display_name].filter(Boolean).map(normalise),
        synonyms: (p.search_terms || []).map(normalise),
        formula_label: [p.formula_label].filter(Boolean).map(normalise),
        formula_structure: [p.formula_structure].filter(Boolean).map(normalise),
        elements: (p.elements_structure || []).map((e) => e.toLowerCase()),
        // The Hermann-Mauguin symbol AND the International Tables
        // number. The number was in the payload and indexed nowhere, so
        // typing `225` -- the example the glossary itself offers -- found
        // nothing at all, while `Fm-3m` found six. A reader who follows
        // the one worked example on the page and gets "no phases match"
        // concludes the search is broken, and they are half right.
        space_group: [sg.core, p.space_group_it != null
          ? String(p.space_group_it) : null].filter(Boolean),
        pearson: [p.pearson].filter(Boolean).map(normalise),
        key: [normalise(p.key)],
      },
      // The structure type is NOT identity: `Cu` is the prototype of pure Al
      // and pure Ni, so indexing it as identity makes `Cu` return 6 phases
      // where the facet says 4 -- the very search/facet contradiction this
      // rewrite exists to remove. It gets its own labelled class.
      prototype: normalise(p.prototype),
      text: normalise([p.reference, p.icsd, p.cod,
        ...Object.values(p.ids_from_filename || {})].filter(Boolean).join(' ')),
      elements: new Set(p.elements_structure || []),
      elementsLabel: new Set(p.elements_label || []),
      formula: parseFormula(p.formula_label) || parseFormula(p.formula_structure),
    };
  });
}

/**
 * Two formulae are the same compound when their counts are proportional:
 * `Al2CuMg` and `Al4Cu2Mg2` are one phase, and a formula is a ratio.
 *
 * The tolerance is RELATIVE and loose (0.5 %) because the two numbers a
 * library holds for one phase are not the same kind of number. This library
 * writes both for the alpha phase: the label says `Fe23Al81Si15`, the
 * idealised stoichiometry, and the structure says `Al162.04Fe46Si30`, the
 * occupancy-weighted content of the cell. Measured deviation between them:
 * 2.5e-4. A tolerance of 1e-6 called them different phases; 0.5 % still
 * separates Al2Cu from Al3Cu, which differ by 50 %.
 */
export function sameFormula(a, b) {
  if (!a || !b) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  if (!(ka[0] in b) || !a[ka[0]] || !b[ka[0]]) return false;   // 0 counts -> no ratio
  const r = a[ka[0]] / b[ka[0]];
  return ka.every((k) => k in b && b[k] && Math.abs(a[k] / b[k] - r) <= 5e-3 * r);
}

/** A one-or-two-character token may only match a whole word. */
function fieldHit(values, token) {
  const whole = token.length <= 2;
  return values.some((v) => (whole
    ? new RegExp(`(^|[^a-z0-9])${token}([^a-z0-9]|$)`).test(v)
    : v.includes(token)));
}

/**
 * @returns {{identity: Array, prototype: Array, text: Array}} each entry
 *   `{key, phase, why: string[], weight: number}`
 *
 * THREE classes, not two. The structure type earns its own because putting it
 * with identity made `Cu` return 6 where the facet says 4 -- pure Al and pure
 * Ni have `Cu` as their prototype -- and 6-against-4 is the same
 * search-versus-facet contradiction, just smaller. Labelling the hits inside
 * one list was not enough: the count at the top of the page is what a user
 * compares against the facet.
 */
export function search(index, query) {
  const q = String(query || '').trim();
  if (!q) {
    return { identity: index.map((e) => ({ ...e, why: [], weight: 9 })),
             prototype: [], text: [] };
  }

  const tokens = normalise(q).split(/\s+/).filter(Boolean);
  const els = queryElements(q);
  const formula = parseFormula(q);
  const sg = normaliseSpaceGroup(q).core;

  const identity = [];
  const prototype = [];
  const text = [];
  for (const e of index) {
    const why = [];
    let weight = 9;

    if (els && els.length && els.every((s) => e.elements.has(s))) {
      why.push(els.length > 1 ? 'system' : 'element'); weight = Math.min(weight, 1);
    } else if (els && els.length && els.every((s) => e.elementsLabel.has(s))) {
      // Label-only: `beta-AlFeSi` is named Si and its structure has none.
      // Found, but marked -- the user who types Al-Fe-Si and does not find a
      // phase CALLED beta-AlFeSi concludes the search is broken.
      why.push('labelOnly'); weight = Math.min(weight, 4);
    }
    if (formula && sameFormula(formula, e.formula)) {
      why.push('formula'); weight = Math.min(weight, 1);
    }
    if (sg && sg.length > 1 && e.fields.space_group.includes(sg)) {
      why.push('spaceGroup'); weight = Math.min(weight, 2);
    }
    // Free text over identity fields: every token must hit somewhere.
    const perToken = tokens.map((t) => {
      for (const [name, w, label] of IDENTITY_FIELDS) {
        if (fieldHit(e.fields[name], t)) return { w, label };
      }
      return null;
    });
    if (perToken.every(Boolean)) {
      for (const h of perToken) { why.push(h.label); weight = Math.min(weight, h.w); }
    }
    if (why.length) {
      identity.push({ ...e, why: [...new Set(why)], weight });
    } else if (e.prototype && tokens.every((t) => fieldHit([e.prototype], t))) {
      prototype.push({ ...e, why: ['prototype'], weight: 5 });
    } else if (tokens.every((t) => t.length > 2 && e.text.includes(t))) {
      text.push({ ...e, why: ['citation'], weight: 8 });
    }
  }

  const byName = (a, b) => a.display.localeCompare(b.display, 'de');
  identity.sort((a, b) => a.weight - b.weight || byName(a, b));
  prototype.sort(byName);
  text.sort(byName);
  return { identity, prototype, text };
}
