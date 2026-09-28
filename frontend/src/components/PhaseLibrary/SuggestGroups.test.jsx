// @vitest-environment jsdom
/**
 * The suggestion run.
 *
 * The fixtures are the real answer from the real library, measured by
 * calling `phase_collections.suggest()` on this machine: four proposals,
 * "Matrix and pure metals" (2), "Intermetallics in Al" (23), "Mg phases"
 * (6), "Zn phases" (1). Invented names would have hidden the thing that
 * matters about this screen -- that one proposal is 23 of the 36 phases,
 * and somebody may well want to untick exactly that one.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  render, screen, cleanup, fireEvent, waitFor, within,
} from '@testing-library/react';
import SuggestGroups, { reasonOf, untouchedBy } from './SuggestGroups';

afterEach(cleanup);

const REAL = [
  { name: 'Matrix and pure metals', keys: ['Al', 'Ni'] },
  { name: 'Intermetallics in Al', keys: Array.from({ length: 23 }, (_, i) => `p${i}`) },
  { name: 'Mg phases', keys: ['Mg2Si', 'Mg17Al12', 'a', 'b', 'c', 'd'] },
  { name: 'Zn phases', keys: ['Al2Zn_MP-mp-aaacqrcb'] },
];

function show(over = {}) {
  const api = {
    suggest: vi.fn(() => Promise.resolve({ data: { suggestions: REAL } })),
    applySuggest: vi.fn(() => Promise.resolve(
      { data: { created: ['Mg phases'], skipped: [] } })),
    ...over,
  };
  const onApplied = vi.fn();
  render(<SuggestGroups api={api} onApplied={onApplied} />);
  return { api, onApplied };
}

const run = () => fireEvent.click(screen.getByTestId('suggest-run'));

describe('proposing', () => {
  it('writes nothing until you accept, and says so', async () => {
    const { api } = show();
    run();
    await screen.findByText(/Nothing is created until you accept/);
    expect(api.applySuggest).not.toHaveBeenCalled();
  });

  it('lists every proposal with how many phases it would hold', async () => {
    show();
    run();
    await screen.findByText('Intermetallics in Al');
    expect(screen.getByText('23 phases')).toBeTruthy();
    expect(screen.getByText('1 phase')).toBeTruthy();
  });

  it('starts with everything ticked', async () => {
    // The proposal IS the answer to "what would you do"; making somebody
    // tick four boxes to accept an answer they asked for is a toll.
    show();
    run();
    await screen.findByText('Zn phases');
    const boxes = [...document.querySelectorAll('input[type=checkbox]')];
    expect(boxes.length).toBe(4);
    expect(boxes.every((b) => b.checked)).toBe(true);
  });

  it('the accept button counts what is ticked, and follows the ticks', async () => {
    show();
    run();
    await screen.findByText('Zn phases');
    expect(screen.getByText('Create 4 groups')).toBeTruthy();
    fireEvent.click(screen.getByText('Intermetallics in Al')
      .closest('label').querySelector('input'));
    expect(screen.getByText('Create 3 groups')).toBeTruthy();
  });

  it('sends exactly the names left ticked', async () => {
    const { api } = show();
    run();
    await screen.findByText('Zn phases');
    for (const name of ['Matrix and pure metals', 'Intermetallics in Al',
      'Zn phases']) {
      fireEvent.click(screen.getByText(name).closest('label')
        .querySelector('input'));
    }
    fireEvent.click(screen.getByText('Create 1 group'));
    await waitFor(() => expect(api.applySuggest)
      .toHaveBeenCalledWith(['Mg phases']));
  });

  it('accepting none writes nothing and closes', async () => {
    const { api } = show();
    run();
    await screen.findByText('Zn phases');
    for (const s of REAL) {
      fireEvent.click(screen.getByText(s.name).closest('label')
        .querySelector('input'));
    }
    fireEvent.click(screen.getByText('Create 0 groups'));
    expect(api.applySuggest).not.toHaveBeenCalled();
    expect(screen.getByTestId('suggest-run')).toBeTruthy();
  });

  it('cancel writes nothing', async () => {
    const { api } = show();
    run();
    await screen.findByText('Zn phases');
    fireEvent.click(screen.getByText('Cancel'));
    expect(api.applySuggest).not.toHaveBeenCalled();
    expect(screen.getByTestId('suggest-run')).toBeTruthy();
  });
});

describe('accepting', () => {
  it('says how many were made, and tells the page to re-read', async () => {
    const { onApplied } = show();
    run();
    await screen.findByText('Zn phases');
    fireEvent.click(screen.getByText('Create 4 groups'));
    await waitFor(() => expect(screen.getByTestId('suggest-created').textContent)
      .toBe('1 group created.'));
    expect(onApplied).toHaveBeenCalled();
  });

  it('SHOWS the names it left alone, rather than dropping them', async () => {
    // Three of nine quietly doing nothing reads as a broken suggestion,
    // not as "you already have those".
    show({ applySuggest: vi.fn(() => Promise.resolve({ data: {
      created: ['Zn phases'], skipped: ['Mg phases', 'Al-Fe'] } })) });
    run();
    await screen.findByText('Zn phases');
    fireEvent.click(screen.getByText('Create 4 groups'));
    const note = await screen.findByTestId('suggest-skipped');
    expect(note.textContent).toMatch(/Mg phases, Al-Fe/);
  });
});

describe('when it cannot', () => {
  it('a library with nothing to propose says so, and is not an error', async () => {
    show({ suggest: vi.fn(() => Promise.resolve({ data: { suggestions: [] } })) });
    run();
    await screen.findByTestId('suggest-empty');
    expect(screen.queryByTestId('suggest-problem')).toBe(null);
  });

  it('a refusal is named', async () => {
    show({ suggest: vi.fn(() => Promise.reject({
      response: { data: { detail: 'library folder is read-only' } } })) });
    run();
    const problem = await screen.findByTestId('suggest-problem');
    expect(problem.textContent).toMatch(/read-only/);
  });

  it('and so is a plain network failure', async () => {
    show({ suggest: vi.fn(() => Promise.reject(new Error('Network Error'))) });
    run();
    expect((await screen.findByTestId('suggest-problem')).textContent)
      .toMatch(/Network Error/);
  });

  it('a failure while applying leaves the proposal on screen', async () => {
    // Otherwise the ticks are gone and the run has to be repeated.
    show({ applySuggest: vi.fn(() => Promise.reject(new Error('boom'))) });
    run();
    await screen.findByText('Zn phases');
    fireEvent.click(screen.getByText('Create 4 groups'));
    await screen.findByTestId('suggest-problem');
    expect(screen.getByText('Zn phases')).toBeTruthy();
  });
});

describe('what a person is told a failure was', () => {
  it('prefers the server\'s own sentence', () => {
    expect(reasonOf({ response: { data: { detail: 'no such collection' } } }))
      .toBe('no such collection');
  });

  it('falls back to the message, and never to "[object Object]"', () => {
    expect(reasonOf(new Error('Network Error'))).toBe('Network Error');
    expect(reasonOf('plain string')).toBe('plain string');
  });
});

describe('a proposal you can actually look at', () => {
  // "I am being asked to accept a grouping of two thirds of my library
  // sight-unseen. A tick box next to a number is not enough to decide."
  it('the count is a button, and it opens the phases', async () => {
    render(<SuggestGroups api={{
      suggest: vi.fn(() => Promise.resolve({ data: { suggestions: REAL } })),
      applySuggest: vi.fn(),
    }} libraryTotal={36} labelFor={(k) => `name(${k})`} />);
    run();
    await screen.findByText('Zn phases');
    fireEvent.click(screen.getByTestId('suggest-peek-Zn phases'));
    expect(screen.getByTestId('suggest-phases-Zn phases').textContent)
      .toBe('name(Al2Zn_MP-mp-aaacqrcb)');
  });

  it('opening one closes the other, so the box stays readable', async () => {
    render(<SuggestGroups api={{
      suggest: vi.fn(() => Promise.resolve({ data: { suggestions: REAL } })),
      applySuggest: vi.fn(),
    }} libraryTotal={36} />);
    run();
    await screen.findByText('Zn phases');
    fireEvent.click(screen.getByTestId('suggest-peek-Zn phases'));
    fireEvent.click(screen.getByTestId('suggest-peek-Mg phases'));
    expect(screen.queryByTestId('suggest-phases-Zn phases')).toBe(null);
    expect(screen.getByTestId('suggest-phases-Mg phases')).toBeTruthy();
  });

  it('says how many phases the whole proposal leaves unfiled', async () => {
    // A reader added the rows up -- 2 + 23 + 6 + 1 = 32 against 36 -- and
    // could not tell whether the other four would be ungrouped or dropped.
    render(<SuggestGroups api={{
      suggest: vi.fn(() => Promise.resolve({ data: { suggestions: REAL } })),
      applySuggest: vi.fn(),
    }} libraryTotal={36} />);
    run();
    await screen.findByTestId('suggest-untouched');
    expect(screen.getByTestId('suggest-untouched').textContent)
      .toMatch(/4 phases in the library are in none of these/);
  });

  it('says nothing about a remainder when there is none', async () => {
    render(<SuggestGroups api={{
      suggest: vi.fn(() => Promise.resolve({ data: { suggestions: REAL } })),
      applySuggest: vi.fn(),
    }} libraryTotal={32} />);
    run();
    await screen.findByText('Zn phases');
    expect(screen.queryByTestId('suggest-untouched')).toBe(null);
  });
});

describe('counting what a proposal leaves out', () => {
  it('counts the union, not the sum -- a phase can be in two proposals', () => {
    expect(untouchedBy([{ keys: ['a', 'b'] }, { keys: ['b', 'c'] }], 10)).toBe(7);
  });

  it('never goes below zero', () => {
    expect(untouchedBy([{ keys: ['a', 'b', 'c'] }], 2)).toBe(0);
  });

  it('says nothing when the library size is unknown', () => {
    expect(untouchedBy([{ keys: ['a'] }], 0)).toBe(null);
  });

  it('copes with a proposal that has no keys', () => {
    expect(untouchedBy([{ name: 'x' }], 5)).toBe(5);
    expect(untouchedBy(null, 5)).toBe(5);
  });
});
