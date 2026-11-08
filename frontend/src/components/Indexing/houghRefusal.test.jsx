// @vitest-environment jsdom
//
// A refused Hough library, worded for the user with its one-click answer.
//
// Contract test: the payload is built from the keys the backend source writes
// (`_refusal_phases`, the raise in `create_indexer`), and both pages and both routes
// are read from the two sides, so a rename on either turns it red.
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import HoughRefusal, { REFUSAL_CODE, refusalFrom } from './HoughRefusal';

const ROOT = path.resolve(__dirname, '../../../..');
const read = (...p) => fs.readFileSync(path.join(ROOT, ...p), 'utf8');

/** Keys of the per-phase dict `_refusal_phases` builds (`entry = {...}`). */
function backendPhaseKeys() {
  const src = read('ebsd_utils.py');
  const at = src.indexOf('entry = {"name"');
  expect(at).toBeGreaterThan(0);
  return [...src.slice(at, src.indexOf('}', at)).matchAll(/"([a-z_]+)":/g)].map((m) => m[1]);
}

function phaseEntry(overrides = {}) {
  const values = {
    name: 'Al13Fe4', rows: 70, predicted_bytes: 7.15 * 1024 ** 3, fit_rows: 64, fit_families: 22,
    fit_reason: null, own_selection: false,
  };
  const out = {};
  for (const k of backendPhaseKeys()) {
    expect(Object.keys(values)).toContain(k); // a new backend field: say what it is
    out[k] = values[k];
  }
  return { ...out, ...overrides };
}

const problem = (phases) => ({
  code: REFUSAL_CODE, message: 'English text of the backend',
  params: { phases, budget_bytes: 4 * 1024 ** 3, free_bytes: 8 * 1024 ** 3, min_rows: 40,
    min_families: 4 },
});
afterEach(() => cleanup());

describe('the notice', () => {
  it('words the problem and offers the one-click answer with the backend count', async () => {
    const fit = vi.fn(() => Promise.resolve({ data: { fit: { rows: 64, predicted_bytes: null } } }));
    const fitted = vi.fn();
    render(<HoughRefusal problem={problem([phaseEntry()])} fit={fit} onFitted={fitted} />);
    const notice = screen.getByTestId('hough-refusal');
    expect(notice.textContent).toMatch(/Hough indexing did not start/);
    expect(notice.textContent).toMatch(/Nothing ran out of memory/);
    expect(notice.textContent).toMatch(
      /Al13Fe4: 70 reflector rows would need a band-triplet library of 7.15 GiB; the budget is 4.00 GiB/);
    expect(notice.textContent).not.toMatch(/English text of the backend/);
    // the card and the button are named by their translated labels
    expect(notice.textContent).toMatch(/Reflector families/);
    const button = screen.getByTestId('hough-refusal-fit-Al13Fe4');
    expect(button.textContent).toBe('Use the largest list that fits (64 rows)');
    fireEvent.click(button);
    await waitFor(() => expect(fit).toHaveBeenCalledWith('Al13Fe4'));
    await screen.findByText(/Using the 64 strongest rows of the default list/);
    expect(fitted).toHaveBeenCalledTimes(1);
  });

  it('offers no button for a phase with a selection of its own, and says what to do', () => {
    render(<HoughRefusal
      problem={problem([phaseEntry({ own_selection: true, fit_rows: null, fit_families: null })])}
      fit={vi.fn()} />);
    expect(screen.queryByTestId('hough-refusal-fit-Al13Fe4')).toBeNull();
    expect(screen.getByTestId('hough-refusal-Al13Fe4').textContent).toMatch(/selection of families of its own/);
  });

  it('says what is left when nothing the ladder allows fits', () => {
    render(<HoughRefusal
      problem={problem([phaseEntry({ fit_rows: null, fit_families: null, fit_reason: 'nothing_fits' })])}
      fit={vi.fn()} />);
    expect(screen.queryByTestId('hough-refusal-fit-Al13Fe4')).toBeNull();
    expect(screen.getByTestId('hough-refusal-Al13Fe4').textContent).toMatch(/Dictionary or Spherical/);
  });

  it('says there is no shorter list for a monoclinic or triclinic phase, with no button', () => {
    render(<HoughRefusal
      problem={problem([phaseEntry({ fit_rows: null, fit_families: null, fit_reason: 'low_symmetry' })])}
      fit={vi.fn()} />);
    expect(screen.queryByTestId('hough-refusal-fit-Al13Fe4')).toBeNull();
    const text = screen.getByTestId('hough-refusal-none-Al13Fe4').textContent;
    expect(text).toMatch(/monoclinic or triclinic/);
    expect(text).toMatch(/lose orientations/);
    expect(text).toMatch(/Spherical or Dictionary/);
    expect(text).toMatch(/shorten the list by hand in the Reflector families card/);
  });

  it('lists every phase of a multi-phase run, a button only where there is an answer', () => {
    render(<HoughRefusal
      problem={problem([phaseEntry(), phaseEntry({ name: 'Al', fit_rows: null, fit_families: null })])}
      fit={vi.fn()} />);
    expect(screen.getByTestId('hough-refusal-fit-Al13Fe4')).toBeTruthy();
    expect(screen.queryByTestId('hough-refusal-fit-Al')).toBeNull();
  });

  it('shows the refusal of the one-click itself', async () => {
    const fit = vi.fn(() => Promise.reject({
      response: { data: { detail: { code: 'own_selection', message: 'x', params: {} } } } }));
    render(<HoughRefusal problem={problem([phaseEntry()])} fit={fit} />);
    fireEvent.click(screen.getByTestId('hough-refusal-fit-Al13Fe4'));
    await screen.findByText(/has a selection of its own/);
  });

  it('is nothing for any other problem', () => {
    const { container } = render(<HoughRefusal problem={{ code: 'other', params: {} }} fit={vi.fn()} />);
    expect(container.textContent).toBe('');
  });
});

describe('refusalFrom', () => {
  it('finds the coded problem in a task status and in an HTTP error, and only that', () => {
    const p = problem([phaseEntry()]);
    expect(refusalFrom({ status: 'failed', problem: p })).toBe(p);
    expect(refusalFrom({ response: { data: { detail: p } } })).toBe(p);
    expect(refusalFrom({ response: { data: { detail: 'a string' } } })).toBeNull();
    expect(refusalFrom({ status: 'failed', error: 'x' })).toBeNull();
    expect(refusalFrom(null)).toBeNull();
  });
});

describe('the contract with the backend', () => {
  it('the code is the one the backend class carries', () => {
    expect(read('hough_reflectors.py')).toContain(`code = "${REFUSAL_CODE}"`);
  });

  it('the component reads fields the backend writes', () => {
    const src = read('frontend', 'src', 'components', 'Indexing', 'HoughRefusal.jsx');
    const keys = backendPhaseKeys();
    for (const field of ['name', 'rows', 'predicted_bytes', 'fit_rows', 'fit_reason', 'own_selection']) {
      expect(keys).toContain(field);
      expect(src).toContain(`p.${field}`);
    }
    const eu = read('ebsd_utils.py');
    for (const field of ['"phases"', '"budget_bytes"']) expect(eu).toContain(field);
    expect(src).toContain('params.phases');
    expect(src).toContain('params.budget_bytes');
  });

  it('both routes pass the problem on under the names the pages read', () => {
    expect(read('backend', 'api', 'routes', 'indexing.py')).toContain('["problem"] = e.as_problem()');
    expect(read('backend', 'api', 'routes', 'pcrefinement.py')).toContain('detail=e.as_problem()');
    const page = read('frontend', 'src', 'components', 'Indexing', 'IndexingPage.jsx');
    expect(page).toContain('refusalFrom(d)');
    expect(page).toContain('<HoughRefusal');
    expect(page).toContain('fitHoughReflectors');
    const pc = read('frontend', 'src', 'components', 'PCRefinement', 'PCRefinement.jsx');
    expect(pc).toContain('refusalFrom(err)');
    expect(pc).toContain('<HoughRefusal');
    expect(pc).toContain('fitPhaseReflectors');
  });

  it('every text is translated into en, de, ja and zh', () => {
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      const j = JSON.parse(read('frontend', 'src', 'locales', lang, 'indexing.json')).houghRefusal;
      for (const k of ['title', 'phase', 'own', 'other', 'lowSymmetry', 'runAgain', 'log']) {
        expect(typeof j[k], `${lang}.${k}`).toBe('string');
      }
      for (const v of ['{{name}}', '{{rows}}', '{{size}}', '{{budget}}']) expect(j.phase, lang).toContain(v);
      expect(j.own, lang).toContain('{{card}}');
      expect(j.other, lang).toContain('{{card}}');
      expect(j.lowSymmetry, lang).toContain('{{card}}');
    }
  });
});
