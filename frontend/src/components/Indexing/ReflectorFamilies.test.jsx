// @vitest-environment jsdom
//
// The reflector-family table against a fake backend that keeps the choice the
// way the real one does (per phase, default until changed).
//
// What the user must be able to do, and what must not happen:
//  * the table is NOT computed until the card is opened (a large unit cell takes
//    seconds) and is read again at every opening (the other page may have changed
//    the shared choice);
//  * a click on a box sends the whole new list of ticked families, nothing else;
//  * an error from the backend is shown in the user's words, and the table keeps
//    what the backend still holds;
//  * every column and every control has a hover text.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react';
import ReflectorFamilies from './ReflectorFamilies';

const FAMILIES = [
  { hkl: [1, 1, 1], hkl4: null, label: '{111}', d: 2.3377, f: 8.49, rel_f: 1.0, mult: 8 },
  { hkl: [2, 0, 0], hkl4: null, label: '{200}', d: 2.0245, f: 7.07, rel_f: 0.83, mult: 6 },
  { hkl: [2, 2, 0], hkl4: null, label: '{220}', d: 1.4315, f: 4.44, rel_f: 0.52, mult: 12 },
  { hkl: [3, 1, 1], hkl4: null, label: '{311}', d: 1.2208, f: 3.62, rel_f: 0.43, mult: 24 },
  { hkl: [2, 2, 2], hkl4: null, label: '{222}', d: 1.1688, f: 3.43, rel_f: 0.40, mult: 8 },
  { hkl: [4, 0, 0], hkl4: null, label: '{400}', d: 1.0122, f: 2.85, rel_f: 0.34, mult: 6 },
];
const PARALLEL = { '{222}': '{111}', '{400}': '{200}' };
const DEFAULT = ['{111}', '{200}', '{220}', '{311}', '{222}', '{400}'];

function makeBackend() {
  const b = {
    ticked: [...DEFAULT], mode: 'default', rule: { min_d: 1.0, f_threshold: 0.1, max_rows: 70 },
    calls: [], loads: 0, extLoads: [], failNext: null, stored: {}, version: 0,
  };
  b.table = () => {
    const sel = new Set(b.ticked);
    return {
      phase_name: 'Al', mode: b.mode, spec: null, rule: b.rule, hexagonal: false,
      default_rule: { min_d: 1.0, f_threshold: 0.1, max_rows: 70 },
      families: FAMILIES.map((f) => {
        const selected = sel.has(f.label);
        const dropper = PARALLEL[f.label];
        const dropped = selected && dropper && sel.has(dropper);
        const mates = Object.entries(PARALLEL).flatMap(([a, c]) => (a === f.label ? [c] : c === f.label ? [a] : []));
        return { ...f, selected, effective: selected && !dropped, parallel_with: mates,
                 dropped_by: dropped ? dropper : null };
      }),
      n_total: 6, truncated: false, n_selected: sel.size,
      n_effective: [...sel].filter((l) => !(PARALLEL[l] && sel.has(PARALLEL[l]))).length,
      effective: [],
    };
  };
  const labelOf = (hkl) => FAMILIES.find((f) => f.hkl.join(',') === hkl.join(','))?.label;
  const fail = () => {
    if (!b.failNext) return null;
    const detail = b.failNext;
    b.failNext = null;
    return Promise.reject({ response: { data: { detail } } });
  };
  b.api = {
    key: 'al',
    load: (ext = false) => { b.loads += 1; b.extLoads.push(ext); return Promise.resolve({ data: { ...b.table(), extended: !!ext } }); },
    stored: () => Promise.resolve({ data: { specs: b.stored } }),
    cost: () => Promise.resolve({ data: { bytes: 3 * 1024 ** 2, fits: true, n_rows: 64 } }),
    change: (spec, ext = false) => {
      b.calls.push(['change', spec]);
      b.lastChangeExt = ext;
      const f = fail();
      if (f) return f;
      if (spec === null) { b.ticked = [...DEFAULT]; b.mode = 'default'; }
      else if (spec.mode === 'custom') { b.ticked = spec.families.map(labelOf); b.mode = 'custom'; }
      else if (spec.mode === 'top_n') { b.ticked = ['{111}', '{200}', '{220}', '{311}'].slice(0, spec.n); b.mode = 'custom'; }
      else if (spec.mode === 'auto') { b.rule = { ...b.rule, ...spec.rule }; b.mode = 'auto'; }
      return Promise.resolve({ data: { ...b.table(), extended: !!ext } });
    },
    validate: (hkl, spec) => {
      b.calls.push(['validate', hkl, spec]);
      const f = fail();
      if (f) return f;
      const txt = String(hkl).replace(/\s+/g, ' ').trim();
      if (txt === '1 0 0') {
        return Promise.reject({ response: { data: { detail: {
          code: 'forbidden', message: 'forbidden', params: { hkl: [1, 0, 0] } } } } });
      }
      if (txt === '3 3 1') {
        return Promise.resolve({ data: { hkl: [3, 3, 1], label: '{331}', d: 0.9289, f: 2.1,
                                         rel_f: 0.25, mult: 24, parallel_with: [] } });
      }
      return Promise.resolve({ data: { hkl: [4, 4, 0], label: '{440}', d: 0.7, f: 1, rel_f: 0.1,
                                       mult: 12, parallel_with: ['{220}'] } });
    },
  };
  return b;
}

let b;
beforeEach(() => { b = makeBackend(); });
afterEach(() => cleanup());

const open = async () => {
  fireEvent.click(screen.getByRole('button', { name: /Reflector families/ }));
  await screen.findByTestId('family-{111}');
};
const box = (label) => within(screen.getByTestId(`family-${label}`)).getByRole('checkbox');

describe('opening', () => {
  it('computes nothing until it is opened, and reads again at each opening', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await waitFor(() => expect(screen.getByRole('button', { name: /Reflector families/ })).toBeTruthy());
    expect(b.loads).toBe(0);
    await open();
    expect(b.loads).toBe(1);
    fireEvent.click(screen.getByRole('button', { name: /Reflector families/ }));
    expect(screen.queryByTestId('family-{111}')).toBeNull();
    b.ticked = ['{111}', '{200}'];                   // the other page changed it meanwhile
    b.mode = 'custom';
    fireEvent.click(screen.getByRole('button', { name: /Reflector families/ }));
    await waitFor(() => expect(b.loads).toBe(2));
    await waitFor(() => expect(box('{220}').checked).toBe(false));
  });

  it('the closed card already says that this phase has a choice of its own', async () => {
    b.stored = { al: { mode: 'custom', families: [[1, 1, 1], [2, 0, 0]] } };
    render(<ReflectorFamilies api={b.api} />);
    await screen.findByText(/your selection/);
    expect(b.loads).toBe(0);
  });

  it('lists the families with the ones PyEBSDIndex ignores marked', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    expect(screen.getAllByRole('row').length).toBe(1 + 6);
    expect(within(screen.getByTestId('family-{222}')).getByText(/same pole as \{111\}/)).toBeTruthy();
    expect(within(screen.getByTestId('family-{400}')).getByText(/same pole as \{200\}/)).toBeTruthy();
    expect(screen.getByText(/6 selected, 4 used by PyEBSDIndex/)).toBeTruthy();
  });
});

describe('changing the choice', () => {
  it('unticking sends the whole list of the others, as an explicit choice', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    fireEvent.click(box('{111}'));
    await waitFor(() => expect(b.calls.length).toBe(1));
    const [, spec] = b.calls[0];
    expect(spec.mode).toBe('custom');
    expect(spec.families).toEqual([[2, 0, 0], [2, 2, 0], [3, 1, 1], [2, 2, 2], [4, 0, 0]]);
    await waitFor(() => expect(box('{111}').checked).toBe(false));
    expect(screen.getByText(/your selection/)).toBeTruthy();
  });

  it('reset goes back to the default and is off while it already is', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    const reset = screen.getByRole('button', { name: 'Reset to default' });
    expect(reset.disabled).toBe(true);
    fireEvent.click(box('{311}'));
    await waitFor(() => expect(reset.disabled).toBe(false));
    fireEvent.click(reset);
    await waitFor(() => expect(b.calls.at(-1)).toEqual(['change', null]));
    await waitFor(() => expect(box('{311}').checked).toBe(true));
  });

  it('the strongest N is a request for N, not a list computed here', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    fireEvent.change(screen.getByLabelText('Strongest'), { target: { value: '3' } });
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    await waitFor(() => expect(b.calls[0]).toEqual(['change', { mode: 'top_n', n: 3 }]));
    await waitFor(() => expect(box('{220}').checked).toBe(true));
    expect(box('{311}').checked).toBe(false);
  });

  it('the rule fields send a rule', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    fireEvent.click(screen.getByRole('button', { name: /Rule for the default list/ }));
    fireEvent.change(screen.getByLabelText('min d (Å)'), { target: { value: '0.8' } });
    fireEvent.change(screen.getByLabelText('|F| threshold'), { target: { value: '0.3' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply rule' }));
    await waitFor(() => expect(b.calls[0]).toEqual(['change', {
      mode: 'auto', rule: { min_d: 0.8, f_threshold: 0.3 } }]));
    await screen.findByText(/changed rule/);
  });

  it('tells the page, so it can refresh what it computed with the old list', async () => {
    const onChanged = vi.fn();
    render(<ReflectorFamilies api={b.api} onChanged={onChanged} />);
    await open();
    fireEvent.click(box('{311}'));
    await waitFor(() => expect(onChanged).toHaveBeenCalledTimes(1));
    expect(onChanged.mock.calls[0][0].mode).toBe('custom');
  });
});

describe('adding a family', () => {
  it('is checked by the backend first, then added to the ticked ones', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    fireEvent.change(screen.getByLabelText('Add family'), { target: { value: '3 3 1' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));
    await waitFor(() => expect(b.calls.map((c) => c[0])).toEqual(['validate', 'change']));
    expect(b.calls[0][2].families).toHaveLength(6);          // what it already holds, for duplicates
    expect(b.calls[1][1].families.at(-1)).toEqual([3, 3, 1]);
    await screen.findByText('Added {331}.');
  });

  it('a refused family is explained and nothing is changed', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    fireEvent.change(screen.getByLabelText('Add family'), { target: { value: '1 0 0' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));
    await screen.findByText(/\{1 0 0\} is not a reflection of this crystal/);
    expect(b.calls.map((c) => c[0])).toEqual(['validate']);
  });

  it('a parallel family is added with a plain warning', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    fireEvent.change(screen.getByLabelText('Add family'), { target: { value: '4 4 0' } });
    fireEvent.keyDown(screen.getByLabelText('Add family'), { key: 'Enter' });
    await screen.findByText(/same pole as \{220\}; PyEBSDIndex uses only the first/);
  });
});

describe('errors', () => {
  it('a refused change is worded for the user and the table keeps what is stored', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    b.failNext = { code: 'too_few', message: 'x', params: { kept: 1 } };
    fireEvent.click(box('{311}'));
    await screen.findByText(/needs at least two distinct families; this selection keeps 1/);
    expect(box('{311}').checked).toBe(true);
  });

  it('a change refused because a run is going is explained, and the table keeps its state', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    b.failNext = { code: 'run_in_progress', message: 'x', params: { runs: ['indexing'] } };
    fireEvent.click(box('{311}'));
    await screen.findByText(/An indexing run is going and uses the reflector selection/);
    expect(box('{311}').checked).toBe(true);
  });

  it('a stored choice that no longer fits the crystal is shown, not applied', async () => {
    const t = b.table();
    b.api.load = () => Promise.resolve({ data: {
      ...t, mode: 'custom',
      spec_error: { code: 'stale_spec', message: 'stale', params: {} } } });
    render(<ReflectorFamilies api={b.api} />);
    await open();
    expect(screen.getByRole('alert').textContent).toMatch(/made for a different crystal/);
    expect(screen.getByRole('button', { name: 'Reset to default' }).disabled).toBe(false);
  });

  it('a table that cannot be read says so instead of an empty card', async () => {
    b.api.load = () => Promise.reject({ response: { data: { detail: {
      code: 'not_found', message: 'nope', params: {} } } } });
    render(<ReflectorFamilies api={b.api} />);
    fireEvent.click(screen.getByRole('button', { name: /Reflector families/ }));
    await screen.findByText('The CIF file was not found.');
  });
});

describe('hover texts', () => {
  it('the card, every column, and every control carry one', async () => {
    render(<ReflectorFamilies api={b.api} />);
    expect(screen.getByRole('button', { name: /Reflector families/ }).title).toBeTruthy();
    await open();
    for (const col of ['Use', 'Family', 'd (Å)', '|F|', 'Mult.']) {
      expect(screen.getByText(col).title, col).toBeTruthy();
    }
    for (const name of ['Strongest', 'Add family']) {
      expect(screen.getByLabelText(name).title, name).toBeTruthy();
    }
    for (const name of ['Select', 'Add', 'Reset to default']) {
      expect(screen.getByRole('button', { name }).title, name).toBeTruthy();
    }
    fireEvent.click(screen.getByRole('button', { name: /Rule for the default list/ }));
    for (const name of ['min d (Å)', '|F| threshold']) {
      expect(screen.getByLabelText(name).title, name).toBeTruthy();
    }
    expect(screen.getByRole('button', { name: 'Apply rule' }).title).toBeTruthy();
  });

  it('shows the memory the library needs and says it fits', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    await screen.findByText(/Library memory: 3 MiB - fits/);
  });
});

describe('large cells', () => {
  it('shows a spinner with a status text and a running clock while the table is computed', async () => {
    let release;
    b.api.load = () => new Promise((res) => { release = () => res({ data: b.table() }); });
    render(<ReflectorFamilies api={b.api} />);
    fireEvent.click(screen.getByRole('button', { name: /Reflector families/ }));
    const status = await screen.findByRole('status');
    expect(status.textContent).toMatch(/Computing the reflector families/);
    expect(within(status).getByTestId('rf-spinner')).toBeTruthy();
    expect(status.textContent).toMatch(/\d+ s/);
    release();
    await screen.findByTestId('family-{111}');
    expect(screen.queryByTestId('rf-spinner')).toBeNull();
  });

  it('weaker families are asked for only when the user asks, and the choice keeps that view', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    expect(b.extLoads).toEqual([false]);
    fireEvent.click(screen.getByRole('button', { name: 'Show weaker families' }));
    await waitFor(() => expect(b.extLoads).toEqual([false, true]));
    await screen.findByRole('button', { name: 'Hide weaker families' });
    fireEvent.click(box('{311}'));
    await waitFor(() => expect(b.calls.length).toBe(1));
    expect(b.lastChangeExt).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Hide weaker families' }));
    await waitFor(() => expect(b.extLoads.at(-1)).toBe(false));
  });

  it('says what it is doing while a change is stored', async () => {
    render(<ReflectorFamilies api={b.api} />);
    await open();
    let release;
    b.api.change = () => new Promise((res) => { release = () => res({ data: b.table() }); });
    fireEvent.click(box('{311}'));
    expect((await screen.findByText('Working...'))).toBeTruthy();
    release();
    await waitFor(() => expect(screen.queryByText('Working...')).toBeNull());
  });
});
