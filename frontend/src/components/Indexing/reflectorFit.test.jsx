// @vitest-environment jsdom
//
// The refusal's one-click answer: "Use the largest list that fits (N rows)".
//
// When the library of a phase's reflector list does not fit the memory, the card
// offers the largest default list that does (backend: largest_list_that_fits).
// Contract test: the payload is built from the keys the backend source really
// writes, and the routes are read from both sides, so a rename on either turns it
// red (the idea of syncResultContract.test.js).
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import ReflectorFamilies from './ReflectorFamilies';

const ROOT = path.resolve(__dirname, '../../../..');
const read = (...p) => fs.readFileSync(path.join(ROOT, ...p), 'utf8');

/** Keys of the dict `largest_list_that_fits` starts from (`out = {...}`). */
function backendFitKeys() {
  const src = read('hough_reflectors.py');
  const at = src.indexOf('out = {"fits"');
  expect(at).toBeGreaterThan(0);
  const body = src.slice(at, src.indexOf('}', at));
  return [...body.matchAll(/"([a-z_]+)":/g)].map((m) => m[1]);
}

function fitPayload(overrides = {}) {
  const values = {
    fits: false, rows: 50, families: 6, predicted_bytes: null, full_bytes: 9 * 1024 ** 3,
    full_rows: 70, budget_bytes: 5 * 1024 ** 3, reason: null,
  };
  const out = {};
  for (const k of backendFitKeys()) {
    expect(Object.keys(values)).toContain(k); // a new backend field: say what it is
    out[k] = values[k];
  }
  return { ...out, ...overrides };
}

const table = (extra = {}) => ({
  phase_name: 'Mg', mode: 'default', spec: null, hexagonal: true,
  rule: { min_d: 1, f_threshold: 0.1, max_rows: 70 },
  default_rule: { min_d: 1, f_threshold: 0.1, max_rows: 70 },
  families: [{ hkl: [1, 0, 0], hkl4: [1, 0, -1, 0], label: '{10-10}', d: 2.8, f: 5, rel_f: 0.5,
    mult: 6, selected: true, effective: true, parallel_with: [], dropped_by: null }],
  n_total: 1, truncated: false, n_selected: 1, n_effective: 1, effective: [], ...extra,
});

function makeApi(cost, fitImpl) {
  const calls = { fit: 0 };
  return {
    calls,
    api: {
      key: 'mg',
      load: () => Promise.resolve({ data: table() }),
      stored: () => Promise.resolve({ data: { specs: {} } }),
      cost: () => Promise.resolve({ data: cost }),
      change: () => Promise.resolve({ data: table() }),
      validate: () => Promise.resolve({ data: {} }),
      fit: fitImpl || (() => {
        calls.fit += 1;
        return Promise.resolve({ data: table({ mode: 'auto', fit: fitPayload() }) });
      }),
    },
  };
}
const over = (fit) => ({
  bytes: 9 * 1024 ** 3, fits: false, budget_bytes: 5 * 1024 ** 3, n_rows: 70, rows: 1,
  ...(fit === undefined ? {} : { fit }),
});
afterEach(() => cleanup());

const open = async () => {
  fireEvent.click(screen.getByRole('button', { name: /Reflector families/ }));
  await screen.findByTestId('family-{10-10}');
};

describe('the one-click answer', () => {
  it('is offered with the count the backend computed, and one click stores it', async () => {
    const { api, calls } = makeApi(over(fitPayload()));
    const changed = vi.fn();
    render(<ReflectorFamilies api={api} onChanged={changed} />);
    await open();
    const button = await screen.findByTestId('reflector-fit');
    expect(button.textContent).toBe('Use the largest list that fits (50 rows)');
    fireEvent.click(button);
    await waitFor(() => expect(calls.fit).toBe(1));
    await screen.findByText(/Using the 50 strongest rows of the default list/);
    expect(changed).toHaveBeenCalledTimes(1);
  });

  it('says what is left when nothing the ladder allows fits', async () => {
    const { api } = makeApi(over(fitPayload({ rows: null, families: null, reason: 'nothing_fits' })));
    render(<ReflectorFamilies api={api} />);
    await open();
    const none = await screen.findByTestId('reflector-fit-none');
    expect(none.textContent).toMatch(/40 rows and 4 families/);
    expect(none.textContent).toMatch(/Dictionary or Spherical/);
    expect(screen.queryByTestId('reflector-fit')).toBeNull();
  });

  it('says there is no shorter list for a monoclinic or triclinic phase, and offers no button', async () => {
    const { api } = makeApi(over(fitPayload({ rows: null, families: null, reason: 'low_symmetry' })));
    render(<ReflectorFamilies api={api} />);
    await open();
    const none = await screen.findByTestId('reflector-fit-none');
    expect(none.textContent).toMatch(/monoclinic or triclinic/);
    expect(none.textContent).toMatch(/lose orientations/);
    expect(none.textContent).toMatch(/Spherical or Dictionary/);
    expect(none.textContent).toMatch(/shorten the list by hand/);
    // not the text of the other refusal, which names a floor of rows
    expect(none.textContent).not.toMatch(/at least [0-9]+ rows/);
    expect(screen.queryByTestId('reflector-fit')).toBeNull();
  });

  it('is not offered when the list fits, or over a selection made by hand', async () => {
    for (const cost of [{ bytes: 1024 ** 2, fits: true, n_rows: 70 }, over(undefined)]) {
      const { api } = makeApi(cost);
      const { unmount } = render(<ReflectorFamilies api={api} />);
      await open();
      await waitFor(() => expect(screen.queryByText(/Library memory/)).toBeTruthy());
      expect(screen.queryByTestId('reflector-fit')).toBeNull();
      expect(screen.queryByTestId('reflector-fit-none')).toBeNull();
      unmount();
    }
  });

  it('shows the backend refusal in the user language', async () => {
    const { api } = makeApi(over(fitPayload()), () => Promise.reject({
      response: { data: { detail: { code: 'own_selection', message: 'x', params: {} } } },
    }));
    render(<ReflectorFamilies api={api} />);
    await open();
    fireEvent.click(await screen.findByTestId('reflector-fit'));
    await screen.findByText(/has a selection of its own/);
  });
});

describe('the contract with the backend', () => {
  it('the component reads fields the backend writes', () => {
    const src = read('frontend', 'src', 'components', 'Indexing', 'ReflectorFamilies.jsx');
    const keys = backendFitKeys();
    for (const field of ['rows', 'predicted_bytes']) {
      expect(keys).toContain(field);
      expect(src).toContain(`fit.${field}`);
    }
    // and the cost call carries it under the name the component reads
    expect(read('backend', 'api', 'services', 'hough_reflector_service.py')).toMatch(/out\["fit"\]/);
    expect(src).toContain('cost.fit');
  });

  it('both pages post to the routes the backend defines', () => {
    const api = read('frontend', 'src', 'services', 'api.js');
    expect(api).toContain("'/api/indexing/hough/reflectors/fit'");
    expect(api).toContain("'/api/pc/phase/reflectors/fit'");
    expect(read('backend', 'api', 'routes', 'hough_reflectors.py')).toContain('@router.post("/reflectors/fit")');
    expect(read('backend', 'api', 'routes', 'pcrefinement.py')).toContain('@router.post("/phase/reflectors/fit")');
    const adapters = read('frontend', 'src', 'components', 'Indexing', 'reflectorSpec.js');
    expect(adapters).toContain('fitHoughReflectors');
    expect(adapters).toContain('fitPhaseReflectors');
  });

  it('every text is translated into en, de, ja and zh', () => {
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      const j = JSON.parse(read('frontend', 'src', 'locales', lang, 'indexing.json')).reflectorFamilies;
      expect(j.fitButton, lang).toContain('{{rows}}');
      expect(j.fitTip, lang).toContain('{{rows}}');
      expect(j.fitDone, lang).toContain('{{rows}}');
      expect(j.fitDone, lang).toContain('{{size}}');
      expect(typeof j.fitNone, `${lang}.fitNone`).toBe('string');
      expect(typeof j.fitNoneLowSym, `${lang}.fitNoneLowSym`).toBe('string');
      for (const c of ['no_fit', 'already_fits', 'own_selection', 'low_symmetry']) {
        expect(typeof j.errors[c], `${lang}.errors.${c}`).toBe('string');
      }
    }
  });
});
