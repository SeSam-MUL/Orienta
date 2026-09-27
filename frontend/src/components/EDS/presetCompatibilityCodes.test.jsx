// @vitest-environment node
/**
 * Every blocker/warning code the EDS frontend names must be a code the
 * BACKEND actually emits.
 *
 * WHY THIS FILE EXISTS. `PresetBar.test.jsx` asserted for months against a
 * blocker code `element_not_measured`. No backend line has ever produced that
 * string — the real one is `missing_element`. The test passed anyway, because
 * it mocks `edsExportApi.checkPreset` and therefore checks the mock against
 * itself: an invented code goes in, the same invented code comes out, green.
 * That is not a test of the contract, it is a test of the imagination of
 * whoever wrote the mock, and this repo has been bitten by exactly that before
 * ("23 green jsdom tests hadn't seen it — the mock was the data shape I'd
 * imagined").
 *
 * WHY IT MATTERS BEYOND A TYPO. `check_compatibility` deliberately returns
 * `{code, message, detail}` so the UI can translate on the CODE and never on
 * the English prose. A code that only exists in a mock is a translation key
 * that will never fire, a `switch` arm that is dead on arrival, and — worst —
 * a refusal the UI silently fails to recognise on the one scan where it
 * matters.
 *
 * WHERE THE TRUTH LIVES (re-check here when this test fails):
 *   backend/api/services/eds_presets.py  — `check_compatibility`, literals
 *                                          spelled `"code": "..."`
 *   backend/api/routes/eds_export.py     — same spelling
 *   backend/api/services/eds_export.py   — `_warn(warnings, "...", ...)`
 *   backend/api/routes/eds.py            — the CLASSIFY warnings, same
 *                                          spelling. Added 2026-08-27: these
 *                                          arrive on the `auto-classify`
 *                                          response in a `warnings: [...]`
 *                                          array, which is the exact shape
 *                                          this scanner already reads, so a
 *                                          mock naming one of them was being
 *                                          checked against a vocabulary that
 *                                          did not contain it.
 *
 * The list below is PINNED so this test still means something in a deployment
 * without the Python tree (the beta repo ships no backend/). When the backend
 * IS present it is parsed and becomes the authority, and the pin is checked
 * against it — so the pin cannot quietly rot into the very fiction it guards
 * against. A code ADDED to the backend needs no edit here; a code REMOVED or
 * RENAMED fails this test, which is the moment to re-read the files above.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

/**
 * Pinned copy of the backend vocabulary. Sorted; keep it that way.
 * Compatibility (eds_presets.py) first, export (eds_export.py) after.
 */
const PINNED_BACKEND_CODES = [
  // --- compatibility: blockers ---
  'matrix_mismatch',
  'missing_element',
  'missing_phase',
  // --- compatibility: warnings ---
  'absolute_window_not_portable',
  'element_set_differs',
  'matrix_not_checked',
  'phase_list_not_pinned',
  'phases_not_checked',
  'portability_not_checkable',
  'step_size_differs',
  // --- classify warnings (routes/eds.py) ---
  // Emitted by `_excluded_signal_warning` and `_degenerate_candidate_warnings`
  // on the `POST /auto-classify` response. All three describe the same
  // wrong-answer path: C and O are dropped before scoring, so an oxide is
  // scored on its metal alone and Al2O3 cannot be told from Al.
  'indistinguishable_after_excluding_c_and_o',
  'phase_has_no_scoreable_chemistry',
  'scoring_ignores_c_and_o',
  // --- export warnings ---
  'below_size_limit',
  'composition_implausible',
  'composition_implausible_summary',
  'definitions_eval_failed',
  'definitions_unavailable',
  'element_shape_mismatch',
  'map_png_failed',
  'no_step_size',
  'no_valid_eds',
  'nothing_classified',
  'particle_scale_failed',
  'particles_from_phases',
  'smoothing_failed',
  'xlsx_failed',
];

const REPO_ROOT = path.resolve(process.cwd(), '..');
const BACKEND_FILES = [
  'backend/api/services/eds_presets.py',
  'backend/api/routes/eds_export.py',
  'backend/api/services/eds_export.py',
  'backend/api/routes/eds.py',
];

/** Codes the backend emits, or `null` when the Python tree is not here. */
function readBackendCodes() {
  const found = new Set();
  let sawAny = false;
  for (const rel of BACKEND_FILES) {
    const p = path.join(REPO_ROOT, rel);
    if (!fs.existsSync(p)) continue;
    sawAny = true;
    const text = fs.readFileSync(p, 'utf8');
    // `"code": "missing_element"` — the dict literals.
    for (const m of text.matchAll(/"code":\s*"([a-z_]+)"/g)) found.add(m[1]);
    // `_warn(warnings, "no_step_size", ...)` — the export helper.
    for (const m of text.matchAll(/_warn\(\s*\w+,\s*"([a-z_]+)"/g)) found.add(m[1]);
  }
  return sawAny ? found : null;
}

const EDS_DIR = path.resolve(process.cwd(), 'src/components/EDS');
const SELF = 'presetCompatibilityCodes.test.jsx';

function edsSourceFiles() {
  return fs.readdirSync(EDS_DIR)
    // This file quotes both shapes in its own prose and in its own unit test;
    // scanning itself would only ever report its own examples.
    .filter((f) => f !== SELF && /\.(jsx?|tsx?)$/.test(f))
    .map((f) => path.join(EDS_DIR, f));
}

/**
 * Every `code: '...'` sitting inside a `blockers: [...]` / `warnings: [...]`
 * literal, plus every `blockers[0].code).toBe('...')` assertion.
 *
 * Scoped to those two shapes on purpose: `reason_code` (region-definition
 * feedback) is a DIFFERENT vocabulary, and one of its tests deliberately feeds
 * an unknown code to prove the fallback prose works.
 */
export function compatCodesIn(text) {
  const out = [];

  const opener = /\b(blockers|warnings)\s*:\s*\[/g;
  let m;
  while ((m = opener.exec(text))) {
    const start = opener.lastIndex - 1;      // sits on the '['
    let depth = 0;
    let end = start;
    for (; end < text.length; end++) {
      if (text[end] === '[') depth++;
      else if (text[end] === ']' && --depth === 0) break;
    }
    const body = text.slice(start, end + 1);
    for (const c of body.matchAll(/\bcode\s*:\s*['"`]([^'"`]+)['"`]/g)) {
      out.push(c[1]);
    }
  }

  const asserted = /\b(?:blockers|warnings)\s*\[\s*\d+\s*\]\s*\.code[^)]*\)\s*\.(?:toBe|toEqual)\(\s*['"`]([^'"`]+)['"`]/g;
  for (const c of text.matchAll(asserted)) out.push(c[1]);

  return out;
}

describe('EDS compatibility codes are the backend\'s codes', () => {
  const backend = readBackendCodes();
  const authority = backend ?? new Set(PINNED_BACKEND_CODES);

  it('the pinned list has not rotted against the backend', () => {
    if (!backend) {
      // No Python tree (slimmed deployment). Nothing to compare against; the
      // pin is then the only authority and the check below still runs.
      expect(PINNED_BACKEND_CODES.length).toBeGreaterThan(0);
      return;
    }
    // Subset, not equality: a code ADDED to the backend is harmless here and
    // must not fail an unrelated frontend change. A code that VANISHED is the
    // dangerous direction, because a UI branch keyed on it goes dead.
    const gone = PINNED_BACKEND_CODES.filter((c) => !backend.has(c));
    expect(gone, `pinned code(s) no longer emitted by ${BACKEND_FILES.join(', ')}`)
      .toEqual([]);
  });

  it('covers the classify warnings, not just the preset ones', () => {
    // These reach the user through `ClassifyWarnings` in PhaseMapPanel and
    // are mocked in `PhaseMapClassifyWarnings.test.jsx` inside a
    // `warnings: [...]` literal — the shape this scanner reads. Before
    // `routes/eds.py` joined BACKEND_FILES they were checked against a
    // vocabulary that could not contain them, i.e. against nothing.
    for (const code of ['scoring_ignores_c_and_o',
      'indistinguishable_after_excluding_c_and_o',
      'phase_has_no_scoreable_chemistry']) {
      expect([...authority], code).toContain(code);
    }
  });

  /**
   * `WRONG_ANSWER_WARNINGS` in PhaseMapPanel.jsx decides which classify
   * warnings get `role="alert"`. A code invented there is a branch that never
   * fires — the exact failure this file was written for, one level up: not a
   * mock checking itself, but a component promising prominence to a string no
   * backend line ever sends.
   *
   * Read as TEXT rather than imported: this suite is `environment: node` and
   * importing the panel would drag in React, the stores and the api client.
   */
  function wrongAnswerCodesInPanel() {
    const src = fs.readFileSync(path.join(EDS_DIR, 'PhaseMapPanel.jsx'), 'utf8');
    const m = src.match(/WRONG_ANSWER_WARNINGS\s*=\s*new Set\(\[([^\]]*)\]/);
    if (!m) return null;
    return [...m[1].matchAll(/['"`]([a-z_]+)['"`]/g)].map((x) => x[1]);
  }

  it('every code the panel promises to shout about is one the backend sends', () => {
    const codes = wrongAnswerCodesInPanel();
    expect(codes, 'WRONG_ANSWER_WARNINGS not found in PhaseMapPanel.jsx')
      .toBeTruthy();
    expect(codes.length, 'the set is empty — the scanner found nothing')
      .toBeGreaterThan(0);
    expect(codes.filter((c) => !authority.has(c)),
      'panel code(s) no backend line emits').toEqual([]);
  });

  it('covers the classify warnings, not just the preset ones', () => {
    // Both are load-bearing in PresetBar.test.jsx; `missing_element` is the
    // one an invented `element_not_measured` stood in for.
    expect([...authority]).toContain('missing_element');
    expect([...authority]).toContain('matrix_mismatch');
    expect([...authority]).not.toContain('element_not_measured');
  });

  it('every code used in the EDS frontend is one the backend emits', () => {
    const offenders = [];
    let seen = 0;
    for (const file of edsSourceFiles()) {
      for (const code of compatCodesIn(fs.readFileSync(file, 'utf8'))) {
        seen += 1;
        if (!authority.has(code)) offenders.push(`${path.basename(file)}: ${code}`);
      }
    }
    // A scanner that finds nothing would pass forever while proving nothing.
    expect(seen, 'the scanner found no blocker/warning codes at all')
      .toBeGreaterThan(3);
    expect(offenders, 'code(s) no backend line emits — see the file header')
      .toEqual([]);
  });

  it('the scanner reads the two shapes that occur, and only those', () => {
    expect(compatCodesIn(`blockers: [{ code: 'missing_element', message: 'x' }]`))
      .toEqual(['missing_element']);
    expect(compatCodesIn(
      `warnings: [\n  { code: "step_size_differs" },\n  { code: 'no_step_size' },\n]`,
    )).toEqual(['step_size_differs', 'no_step_size']);
    expect(compatCodesIn(`expect(r.blockers[0].code).toBe('matrix_mismatch');`))
      .toEqual(['matrix_mismatch']);
    // A nested array inside the literal must not end the scan early.
    expect(compatCodesIn(
      `blockers: [{ code: 'missing_element', detail: { clauses: ['a'] } }, { code: 'missing_phase' }]`,
    )).toEqual(['missing_element', 'missing_phase']);
    // `reason_code` is a different vocabulary and is deliberately untouched.
    expect(compatCodesIn(`reasonText(t, { reason_code: 'from-the-future' })`))
      .toEqual([]);
  });
});
