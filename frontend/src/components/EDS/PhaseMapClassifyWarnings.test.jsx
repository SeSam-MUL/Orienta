// @vitest-environment jsdom
/**
 * The classify warnings, and the hand-edit record, on their way to a user.
 *
 * WHY THIS FILE EXISTS. `POST /api/eds/auto-classify` has been returning a
 * `warnings` array — and not one line of frontend code read it. The array was
 * built, serialised, sent, and dropped on the floor. That is worse than not
 * having the warning: the backend author, the tests and the changelog all say
 * the user is warned, and the user is not.
 *
 * WHAT THE WARNING IS ABOUT, because it decides how loud it has to be. Phase
 * scoring drops carbon and oxygen before it compares anything. On a metal that
 * is a fair simplification. On an oxide it is the entire question: Al2O3 is
 * scored on its aluminium content alone and is therefore indistinguishable
 * from Al metal, so the classifier will confidently name a METAL where the
 * OXIDE is — with a good score and nothing on screen to doubt. A service lab
 * running batteries and ceramics hits this on the first scan. So:
 *
 *  - it renders where the eye lands after pressing Classify, not in a toast.
 *    Two testers have already said a message that vanishes in six seconds is
 *    not a warning;
 *  - the wrong-answer codes get `role="alert"`;
 *  - the text names the CONSEQUENCE ("a named metal may be sitting where the
 *    oxide is"), because "C and O are excluded from scoring" is a fact a
 *    reader can nod at without understanding what it costs them;
 *  - a run with no warnings renders NOTHING. An empty box after every clean
 *    run teaches the eye to skip the place the real one will appear;
 *  - the warnings outlive a colour click and a merge. `phaseMap` is
 *    overwritten by every later `GET /phase-map`, and none of those carry
 *    `warnings` — so a warning parked on the map object would disappear the
 *    first time the user touched anything.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  renderHook, act, render, cleanup, fireEvent, screen,
} from '@testing-library/react';

import en from '../../locales/en/eds.json';
import de from '../../locales/de/eds.json';
import ja from '../../locales/ja/eds.json';
import zh from '../../locales/zh/eds.json';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));
vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: '/tmp/x.h5oina' }),
}));

const MAP = { loaded: true, n_rows: 4, n_cols: 4, summary: [], all_phases: [], regions: [] };

const autoClassify = vi.fn();
const getPhaseMap = vi.fn();
const setPhaseColors = vi.fn();
const clearPhaseMap = vi.fn();
vi.mock('../../services/api', () => ({
  edsApi: {
    autoClassify: (...a) => autoClassify(...a),
    getPhaseMap: (...a) => getPhaseMap(...a),
    setPhaseColors: (...a) => setPhaseColors(...a),
    clearPhaseMap: (...a) => clearPhaseMap(...a),
    elements: vi.fn(() => Promise.resolve({ data: { elements: [] } })),
    cifPhases: vi.fn(() => Promise.resolve({ data: { phases: [] } })),
    assignRegionPhase: vi.fn(),
  },
}));

import {
  usePhaseMap, PhaseMapControls, ClassifyWarnings, HandEditSummary,
  WRONG_ANSWER_WARNINGS, warningParams, HAND_EDIT_KEYS,
} from './PhaseMapPanel';

/**
 * The three warnings exactly as `backend/api/routes/eds.py` builds them —
 * code, English prose, and the `detail` the messages interpolate from.
 * `_excluded_signal_warning` and `_degenerate_candidate_warnings` are the
 * source; `presetCompatibilityCodes.test.jsx` holds the codes against the
 * backend file so these three cannot drift into fiction.
 */
const W_EXCLUDED = {
  code: 'scoring_ignores_c_and_o',
  message: 'C and O carry 31.4 at% of the composition in this scan on average '
    + '(12.0 % of pixels carry at least 25 at%), and phase scoring drops them '
    + 'before it compares anything. ... a named metal may be sitting where the '
    + 'oxide is.',
  detail: {
    elements: ['C', 'O'], mean_at_pct: 31.42, max_at_pct: 61.8,
    frac_px_above: 0.12, px_threshold_at_pct: 25.0,
  },
};
const W_DEGENERATE = {
  code: 'indistinguishable_after_excluding_c_and_o',
  message: '2 candidate pair(s) become indistinguishable once carbon and '
    + 'oxygen are dropped for scoring: Al2O3.cif vs Al.cif; ...',
  detail: {
    pairs: [
      { a: 'Al2O3.cif', b: 'Al.cif', score_gap_bound: 0.0 },
      { a: 'SiO2.cif', b: 'Si.cif', score_gap_bound: 0.0 },
    ],
    tie_tolerance: 0.01,
  },
};
const W_NO_CHEM = {
  code: 'phase_has_no_scoreable_chemistry',
  message: '1 candidate phase(s) are made only of elements the scorer '
    + 'discards (C, O): Graphite.cif. ... it won by tie-break rather than by '
    + 'chemistry.',
  detail: { phases: ['Graphite.cif'] },
};

const flush = () => act(async () => {
  await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
});

/** The i18n mock's t, so a rendered key is legible in an assertion. */
const t = (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k);

/**
 * A `t` that resolves against the REAL English bundle.
 *
 * The mocked `t` proves the wiring — that the code is what gets translated.
 * It cannot prove the user is told anything, because the "message" is then an
 * i18n key. This one renders the sentence a user actually sees, which is the
 * only way to assert that the sentence names the consequence.
 */
const realT = (key, opts = {}) => {
  const val = key.split('.').reduce(
    (o, k) => (o && typeof o === 'object' ? o[k] : undefined), en);
  let s = typeof val === 'string' ? val : (opts.defaultValue ?? key);
  for (const [k, v] of Object.entries(opts)) {
    if (k === 'defaultValue') continue;
    s = s.split(`{{${k}}}`).join(String(v));
  }
  return s;
};

beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  autoClassify.mockResolvedValue({ data: { ...MAP, warnings: [] } });
  getPhaseMap.mockResolvedValue({ data: { loaded: false } });
  setPhaseColors.mockResolvedValue({ data: { loaded: false } });
  clearPhaseMap.mockResolvedValue({ data: {} });
});
afterEach(cleanup);

async function mountHook() {
  const r = renderHook(() => usePhaseMap({}));
  await flush();
  return r;
}

const warningEls = () =>
  [...document.querySelectorAll('[data-classify-warning]')];

// ---------------------------------------------------------------------------

describe('every code the backend emits reaches the screen', () => {
  for (const w of [W_EXCLUDED, W_DEGENERATE, W_NO_CHEM]) {
    it(`renders a visible message for ${w.code}`, () => {
      render(<ClassifyWarnings warnings={[w]} t={realT} />);
      const el = document.querySelector(`[data-classify-warning="${w.code}"]`);
      expect(el, 'the warning is not on screen at all').toBeTruthy();
      // Not a key, not a code, not an empty node: a sentence.
      expect(el.textContent.trim().length).toBeGreaterThan(40);
      expect(el.textContent).not.toContain(w.code);
      expect(el.textContent).not.toContain('phaseMap.warnings');
      // No un-substituted placeholder left behind.
      expect(el.textContent).not.toMatch(/\{\{/);
    });
  }

  it('all three carry alert prominence — these are wrong answers, not notes', () => {
    render(<ClassifyWarnings warnings={[W_EXCLUDED, W_DEGENERATE, W_NO_CHEM]}
                             t={realT} />);
    const els = warningEls();
    expect(els).toHaveLength(3);
    for (const el of els) expect(el.getAttribute('role')).toBe('alert');
  });

  it('translates on the CODE, never on the English prose', () => {
    render(<ClassifyWarnings warnings={[W_EXCLUDED]} t={t} />);
    const el = document.querySelector('[data-classify-warning]');
    expect(el.textContent).toContain('phaseMap.warnings.scoring_ignores_c_and_o');
    // The prose travels as the fallback, so an unknown code still says
    // something — but the translation is what wins when there is one.
    expect(el.textContent).toContain('defaultValue');
  });
});

describe('a run with nothing to say says nothing', () => {
  it('renders no node at all for an empty array', () => {
    const { container } = render(<ClassifyWarnings warnings={[]} t={realT} />);
    expect(container.innerHTML).toBe('');
    expect(document.querySelector('[data-classify-warnings]')).toBeNull();
  });

  it('renders nothing for a missing or malformed array either', () => {
    for (const bad of [undefined, null, 'warnings', {}]) {
      cleanup();
      const { container } = render(<ClassifyWarnings warnings={bad} t={realT} />);
      expect(container.innerHTML, String(bad)).toBe('');
    }
  });

  it('drops an entry with neither a code nor prose rather than printing a key', () => {
    render(<ClassifyWarnings warnings={[{ detail: {} }]} t={realT} />);
    expect(document.querySelector('[data-classify-warnings]')).toBeNull();
  });
});

describe('a code this build has never heard of', () => {
  const FUTURE = {
    code: 'quantification_saturated',
    message: 'The detector saturated on 8 % of pixels; those compositions are '
      + 'a floor, not a measurement.',
  };

  it('still reaches the user, in the backend\'s own words', () => {
    render(<ClassifyWarnings warnings={[FUTURE]} t={realT} />);
    const el = document.querySelector('[data-classify-warning="quantification_saturated"]');
    expect(el).toBeTruthy();
    expect(el.textContent).toContain('The detector saturated');
  });

  it('is a status, not an alert — we cannot claim it is a wrong answer', () => {
    render(<ClassifyWarnings warnings={[FUTURE]} t={realT} />);
    const el = document.querySelector('[data-classify-warning]');
    expect(el.getAttribute('role')).toBe('status');
    expect(WRONG_ANSWER_WARNINGS.has(FUTURE.code)).toBe(false);
  });
});

describe('the English text names the consequence, not just the fact', () => {
  /**
   * The backend prose does this and the translations have to keep the force.
   * "C and O are excluded from scoring" is a fact a reader nods at; "a named
   * metal may be sitting where the oxide is" is the thing they have to act on.
   */
  it('the exclusion warning says a metal may be standing in for an oxide', () => {
    const s = en.phaseMap.warnings.scoring_ignores_c_and_o;
    expect(s).toMatch(/Al₂O₃|Al2O3/);
    expect(s.toLowerCase()).toContain('may be sitting where the oxide is');
  });

  it('the tie warning says the NAME is decided by the tie-break', () => {
    const s = en.phaseMap.warnings.indistinguishable_after_excluding_c_and_o;
    expect(s.toLowerCase()).toContain('tie-break');
    expect(s.toLowerCase()).toContain('equally likely');
  });

  it('the unscoreable warning says the win was not chemistry', () => {
    const s = en.phaseMap.warnings.phase_has_no_scoreable_chemistry;
    expect(s.toLowerCase()).toContain('tie-break');
  });
});

describe('every language carries the warnings, with the consequence intact', () => {
  const LOCALES = { en, de, ja, zh };
  const KEYS = ['title', 'scoring_ignores_c_and_o',
    'indistinguishable_after_excluding_c_and_o',
    'phase_has_no_scoreable_chemistry'];

  it('has every string, non-empty', () => {
    for (const [lang, b] of Object.entries(LOCALES)) {
      for (const k of KEYS) {
        expect(typeof b.phaseMap.warnings[k], `${lang}.${k}`).toBe('string');
        expect(b.phaseMap.warnings[k].trim().length, `${lang}.${k}`)
          .toBeGreaterThan(0);
      }
    }
  });

  it('none is a copy of the English', () => {
    for (const [lang, b] of Object.entries(LOCALES)) {
      if (lang === 'en') continue;
      for (const k of KEYS) {
        expect(b.phaseMap.warnings[k], `${lang}.${k} untranslated`)
          .not.toBe(en.phaseMap.warnings[k]);
      }
    }
  });

  it('every language names Al2O3 vs Al — the example IS the argument', () => {
    // Drop it and the sentence becomes "some elements are ignored", which is
    // exactly the version nobody acts on.
    for (const [lang, b] of Object.entries(LOCALES)) {
      expect(b.phaseMap.warnings.scoring_ignores_c_and_o, `${lang}`)
        .toMatch(/Al₂O₃|Al2O3/);
    }
  });

  it('every language interpolates what the messages measure', () => {
    for (const [lang, b] of Object.entries(LOCALES)) {
      const w = b.phaseMap.warnings;
      expect(w.scoring_ignores_c_and_o, `${lang} elements`).toContain('{{elements}}');
      expect(w.scoring_ignores_c_and_o, `${lang} mean`).toContain('{{mean}}');
      expect(w.indistinguishable_after_excluding_c_and_o, `${lang} n`).toContain('{{n}}');
      expect(w.indistinguishable_after_excluding_c_and_o, `${lang} pairs`).toContain('{{pairs}}');
      expect(w.phase_has_no_scoreable_chemistry, `${lang} n`).toContain('{{n}}');
      expect(w.phase_has_no_scoreable_chemistry, `${lang} phases`).toContain('{{phases}}');
    }
  });

  it('no message uses {{count}}, which i18next would read as a plural selector', () => {
    // With `count` present i18next looks for `key_one`/`key_other`, finds
    // neither, and falls back past the sentence we wrote.
    for (const [lang, b] of Object.entries(LOCALES)) {
      for (const k of KEYS) {
        expect(b.phaseMap.warnings[k], `${lang}.${k}`).not.toContain('{{count}}');
      }
    }
  });
});

describe('warningParams reads the detail the backend actually sends', () => {
  it('joins the excluded elements and rounds the mean to one place', () => {
    expect(warningParams(W_EXCLUDED)).toEqual({ elements: 'C + O', mean: '31.4' });
  });

  it('counts pairs as `n` and names the first few', () => {
    const p = warningParams(W_DEGENERATE);
    expect(p.n).toBe(2);
    expect(p.pairs).toBe('Al2O3.cif / Al.cif; SiO2.cif / Si.cif');
  });

  it('truncates a long list rather than filling the panel', () => {
    const many = { code: 'phase_has_no_scoreable_chemistry',
      detail: { phases: ['a', 'b', 'c', 'd', 'e', 'f'] } };
    const p = warningParams(many);
    expect(p.n).toBe(6);
    expect(p.phases).toBe('a, b, c, d +2');
  });

  it('survives a warning with no detail at all', () => {
    expect(warningParams({ code: 'x' })).toEqual({});
    expect(warningParams(null)).toEqual({});
  });
});

// ---------------------------------------------------------------------------

describe('the warnings survive everything that is not a new run', () => {
  it('arrive from the classify response', async () => {
    autoClassify.mockResolvedValue({
      data: { ...MAP, warnings: [W_EXCLUDED, W_DEGENERATE] },
    });
    const { result } = await mountHook();
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(result.current.classifyWarnings.map((w) => w.code))
      .toEqual(['scoring_ignores_c_and_o',
        'indistinguishable_after_excluding_c_and_o']);
  });

  it('are empty before any run — a fresh panel claims nothing', async () => {
    const { result } = await mountHook();
    expect(result.current.classifyWarnings).toEqual([]);
  });

  /**
   * The trap this guards. `phaseMap` is replaced by every later
   * `GET /phase-map` answer — a colour click, a merge, a paint — and NONE of
   * those responses carry `warnings`. Reading them off `phaseMap` would have
   * shown the warning until the user's next click and then silently dropped
   * it, which is the same defect as not rendering it at all, only harder to
   * notice.
   */
  it('outlive a phase-map refresh that carries no warnings of its own', async () => {
    autoClassify.mockResolvedValue({ data: { ...MAP, warnings: [W_EXCLUDED] } });
    const { result } = await mountHook();
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(result.current.classifyWarnings).toHaveLength(1);

    // What a merge / a colour change writes back: the whole map, no warnings.
    await act(async () => { result.current.setPhaseMap({ ...MAP, n_locked: 12 }); });
    expect(result.current.classifyWarnings).toHaveLength(1);
  });

  it('are replaced, not merged, by the next run', async () => {
    autoClassify.mockResolvedValue({ data: { ...MAP, warnings: [W_EXCLUDED] } });
    const { result } = await mountHook();
    await act(async () => { await result.current.handleAutoClassify(); });

    autoClassify.mockResolvedValue({ data: { ...MAP, warnings: [W_NO_CHEM] } });
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(result.current.classifyWarnings.map((w) => w.code))
      .toEqual(['phase_has_no_scoreable_chemistry']);
  });

  it('a clean run clears the previous run\'s warnings', async () => {
    autoClassify.mockResolvedValue({ data: { ...MAP, warnings: [W_EXCLUDED] } });
    const { result } = await mountHook();
    await act(async () => { await result.current.handleAutoClassify(); });

    autoClassify.mockResolvedValue({ data: { ...MAP } });   // no key at all
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(result.current.classifyWarnings).toEqual([]);
  });

  it('go away with the map', async () => {
    autoClassify.mockResolvedValue({ data: { ...MAP, warnings: [W_EXCLUDED] } });
    const { result } = await mountHook();
    await act(async () => { await result.current.handleAutoClassify(); });
    await act(async () => { await result.current.handleClearMap(); });
    expect(result.current.classifyWarnings).toEqual([]);
  });
});

// ---------------------------------------------------------------------------

describe('the hand-edit record a sign-off has to read', () => {
  it('says nothing when the backend sent no record', () => {
    const { container } = render(<HandEditSummary handEdits={undefined} t={realT} />);
    expect(container.innerHTML).toBe('');
  });

  it('distinguishes "not recorded" from "none happened"', () => {
    // A map restored from a sidecar older than the edit log. Its counts are
    // null; printing 0 would be a claim, and the claim would be in somebody's
    // provenance.
    render(<HandEditSummary t={realT} handEdits={{
      tracked: false,
      counts: Object.fromEntries(HAND_EDIT_KEYS.map((k) => [k, null])),
      n_edits: null, region_grid_edited: null,
    }} />);
    const el = document.querySelector('[data-hand-edits]');
    expect(el.getAttribute('data-hand-edits-tracked')).toBe('false');
    expect(el.textContent).toContain('not the same as none');
    expect(el.textContent).not.toMatch(/\b0\b/);
  });

  it('states the zero when the zero is a measurement', () => {
    render(<HandEditSummary t={realT} handEdits={{
      tracked: true,
      counts: Object.fromEntries(HAND_EDIT_KEYS.map((k) => [k, 0])),
      n_edits: 0, region_grid_edited: false,
    }} />);
    const el = document.querySelector('[data-hand-edits]');
    expect(el.getAttribute('data-hand-edits-tracked')).toBe('true');
    expect(el.textContent).toContain('exactly as it was classified');
  });

  it('lists the counts the group leader asked for', () => {
    render(<HandEditSummary t={realT} handEdits={{
      tracked: true,
      counts: {
        regions_named: 3, merges: 12, splits: 1, paints: 40,
        grows: 0, edge_snaps: 0, phase_replacements: 0,
      },
      n_edits: 56, region_grid_edited: true,
    }} />);
    const text = document.querySelector('[data-hand-edits]').textContent;
    expect(text).toContain('3 regions named');
    expect(text).toContain('12 merges');
    expect(text).toContain('1 splits');
    expect(text).toContain('40 paints');
    // Zeros are not listed — a line of "0 grows · 0 edge snaps" buries the
    // twelve merges that matter.
    expect(text).not.toContain('0 grows');
  });

  it('warns that a boundary edit renumbered the particle ids', () => {
    render(<HandEditSummary t={realT} handEdits={{
      tracked: true, counts: { merges: 1 }, n_edits: 1,
      region_grid_edited: true,
    }} />);
    const el = document.querySelector('[data-hand-edits-regrid]');
    expect(el).toBeTruthy();
    expect(el.getAttribute('role')).toBe('status');
    expect(el.textContent).toContain('renumbered');
  });

  it('stays quiet about ids when the regions were left alone', () => {
    render(<HandEditSummary t={realT} handEdits={{
      tracked: true, counts: { paints: 5 }, n_edits: 5,
      region_grid_edited: false,
    }} />);
    expect(document.querySelector('[data-hand-edits-regrid]')).toBeNull();
  });

  it('makes no claim about ids on an untracked map', () => {
    // `region_grid_edited` is null there, not false: the map cannot say its
    // regions were left alone.
    render(<HandEditSummary t={realT} handEdits={{
      tracked: false, counts: {}, region_grid_edited: null,
    }} />);
    expect(document.querySelector('[data-hand-edits-regrid]')).toBeNull();
  });

  it('every language carries the hand-edit strings, non-empty and translated', () => {
    const LOCALES = { en, de, ja, zh };
    const KEYS = ['untracked', 'none', 'some', 'regionGridEdited',
      ...HAND_EDIT_KEYS];
    for (const [lang, b] of Object.entries(LOCALES)) {
      for (const k of KEYS) {
        expect(typeof b.phaseMap.handEdits[k], `${lang}.${k}`).toBe('string');
        expect(b.phaseMap.handEdits[k].trim().length, `${lang}.${k}`)
          .toBeGreaterThan(0);
        if (lang !== 'en') {
          expect(b.phaseMap.handEdits[k], `${lang}.${k} untranslated`)
            .not.toBe(en.phaseMap.handEdits[k]);
        }
      }
      expect(b.phaseMap.handEdits.some, `${lang}.some`).toContain('{{list}}');
      for (const k of HAND_EDIT_KEYS) {
        expect(b.phaseMap.handEdits[k], `${lang}.${k}`).toContain('{{n}}');
      }
    }
  });

  it('the counter keys are the ones the backend actually tallies', () => {
    // `_OP_COUNTER` in `backend/api/services/phase_map_store.py`. A key that
    // drifts here is a counter that silently stops being shown.
    expect([...HAND_EDIT_KEYS].sort()).toEqual([
      'edge_snaps', 'grows', 'merges', 'paints',
      'phase_replacements', 'regions_named', 'splits',
    ]);
  });
});

// ---------------------------------------------------------------------------

/**
 * The step that was actually missing.
 *
 * `ClassifyWarnings` rendering correctly in isolation proves nothing about the
 * defect this work fixes — the backend built the array perfectly too. What was
 * broken was the WIRE: nobody read `res.data.warnings`, and nobody put a
 * reader on screen. So this suite presses the real button on the real panel
 * and looks at the real DOM.
 */
describe('pressing Classify on the real panel puts the warning on screen', () => {
  const flushAll = () => act(async () => {
    for (let i = 0; i < 6; i += 1) await Promise.resolve();
  });

  function Harness() {
    const handle = usePhaseMap({});
    return <PhaseMapControls handle={handle} />;
  }

  it('shows nothing before a run, the warning after it, and nothing after Clear',
    async () => {
      autoClassify.mockResolvedValue({
        data: { ...MAP, n_classified: 10, n_unclassified: 0,
                warnings: [W_EXCLUDED, W_DEGENERATE] },
      });
      render(<Harness />);
      await flushAll();

      // Before: no map, no run, nothing claimed.
      expect(document.querySelector('[data-classify-warnings]')).toBeNull();

      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: 'phaseMap.autoClassify' }));
      });
      await flushAll();

      const box = document.querySelector('[data-classify-warnings]');
      expect(box, 'the warnings never reached the panel').toBeTruthy();
      const shown = [...box.querySelectorAll('[data-classify-warning]')]
        .map((el) => el.getAttribute('data-classify-warning'));
      expect(shown).toEqual(['scoring_ignores_c_and_o',
        'indistinguishable_after_excluding_c_and_o']);
      // Prominence, in the DOM the screen reader reads.
      expect(box.querySelectorAll('[role="alert"]')).toHaveLength(2);
      // The measured numbers travelled with it — an unqualified sentence
      // would leave the user unable to judge how bad this scan is.
      expect(box.textContent).toContain('"mean":"31.4"');
      expect(box.textContent).toContain('C + O');

      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: 'phaseMap.clear' }));
      });
      await flushAll();
      expect(document.querySelector('[data-classify-warnings]')).toBeNull();
    });

  it('a clean run leaves no box behind at all', async () => {
    autoClassify.mockResolvedValue({
      data: { ...MAP, n_classified: 10, n_unclassified: 0, warnings: [] },
    });
    render(<Harness />);
    await flushAll();
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'phaseMap.autoClassify' }));
    });
    await flushAll();
    expect(document.querySelector('[data-classify-warnings]')).toBeNull();
    expect(document.querySelector('[data-classify-warning]')).toBeNull();
  });

  it('the hand-edit counts appear beside the map, from GET /phase-map', async () => {
    autoClassify.mockResolvedValue({
      data: {
        ...MAP, n_classified: 10, n_unclassified: 0, warnings: [],
        hand_edits: {
          tracked: true,
          counts: { regions_named: 2, merges: 5, splits: 0, paints: 9,
                    grows: 0, edge_snaps: 0, phase_replacements: 0 },
          n_edits: 16, region_grid_edited: true,
        },
      },
    });
    render(<Harness />);
    await flushAll();
    // No map yet, so nothing to report about edits to it.
    expect(document.querySelector('[data-hand-edits]')).toBeNull();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'phaseMap.autoClassify' }));
    });
    await flushAll();

    const el = document.querySelector('[data-hand-edits]');
    expect(el, 'hand_edits is on the payload and still nowhere on screen')
      .toBeTruthy();
    // The counters are composed into one `some` line; the i18n mock nests
    // its JSON, so assert on the keys and the numbers rather than on a
    // double-escaped literal that says nothing about behaviour.
    expect(el.textContent).toContain('phaseMap.handEdits.some');
    for (const [key, n] of [['regions_named', 2], ['merges', 5], ['paints', 9]]) {
      expect(el.textContent, key).toContain(`phaseMap.handEdits.${key}`);
      expect(el.textContent, key).toContain(`n\\":${n}`);
    }
    // Counters at zero are left out; a line of "0 grows · 0 edge snaps"
    // buries the five merges that matter.
    expect(el.textContent).not.toContain('phaseMap.handEdits.grows');
    expect(document.querySelector('[data-hand-edits-regrid]')).toBeTruthy();
  });
});
