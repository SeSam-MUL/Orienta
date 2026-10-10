// @vitest-environment jsdom
//
// The reflector controls of the PC Refinement page, through the real page and a
// fake backend. This page showed min_d, f_threshold, max_reflectors and nBands
// for months and none of them reached the backend (the page read only `method`
// and `searchLimit` from them). What these tests pin is that what is on the
// page is wired:
//  * the three dead numbers are gone and each loaded phase has the family table;
//  * a click in the table is a call that stores the choice for THAT phase;
//  * nBands reaches the backend when - and only when - the user changes it;
//  * the shown pattern is indexed again after either, because its bands were
//    drawn from the old indexer;
//  * every parameter's label and control carries its hover text.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react';

const backend = vi.hoisted(() => ({
  phases: [], reflectorCalls: [], paramCalls: [], indexCalls: [],
}));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const stem = (p) => p.split(/[\\/]/).pop().replace(/\.cif$/i, '');
  const summary = (name) => ({ name, space_group: 'Fm-3m', lattice: { a: 3.6, b: 3.6, c: 3.6 } });
  const fams = (name) => ([
    { hkl: [1, 1, 1], hkl4: null, label: '{111}', d: 2.3, f: 8, rel_f: 1, mult: 8,
      selected: true, effective: true, parallel_with: [], dropped_by: null },
    { hkl: [2, 0, 0], hkl4: null, label: '{200}', d: 2.0, f: 7, rel_f: 0.8, mult: 6,
      selected: true, effective: true, parallel_with: [], dropped_by: null },
    { hkl: [2, 2, 0], hkl4: null, label: '{220}', d: 1.4, f: 4, rel_f: 0.5, mult: 12,
      selected: true, effective: true, parallel_with: [], dropped_by: null },
  ].map((f) => ({ ...f, label: f.label })));
  const table = (name) => ({
    phase_name: name, mode: 'default', spec: null, hexagonal: false,
    rule: { min_d: 1, f_threshold: 0.1, max_rows: 70 },
    default_rule: { min_d: 1, f_threshold: 0.1, max_rows: 70 },
    families: fams(name), n_total: 3, truncated: false, n_selected: 3, n_effective: 3, effective: [],
  });
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: backend.phases.length > 0,
      phase_name: backend.phases[0] ?? null, phase_names: [...backend.phases],
      n_patterns: 2, pc: [0.5, 0.3, 0.8],
      patterns: [{ index: 0, row: 1, col: 2 }, { index: 1, row: 3, col: 4 }],
    }),
    addPhase: (path) => {
      backend.phases = [...backend.phases, stem(path)];
      return ok({ success: true, phase_name: stem(path), space_group: 'Fm-3m',
                  lattice: { a: 1, b: 1, c: 1 }, phases: backend.phases.map(summary),
                  n_phases: backend.phases.length });
    },
    loadPhase: (path) => {
      backend.phases = [stem(path)];
      return ok({ success: true, phase_name: stem(path), space_group: 'Fm-3m',
                  lattice: { a: 1, b: 1, c: 1 } });
    },
    removePhase: (name) => {
      backend.phases = backend.phases.filter((n) => n !== name);
      return ok({ success: true, phases: backend.phases.map(summary), n_phases: backend.phases.length });
    },
    phaseReflectors: (name) => { backend.reflectorCalls.push(['get', name]); return ok(table(name)); },
    setPhaseReflectors: (name, spec) => {
      backend.reflectorCalls.push(['put', name, spec]);
      return ok({ ...table(name), mode: spec ? 'custom' : 'default' });
    },
    validatePhaseReflector: () => ok({ hkl: [3, 3, 1], label: '{331}', d: 0.9, f: 1, rel_f: 0.2,
                                       mult: 24, parallel_with: [] }),
    phaseReflectorCost: () => ok({ bytes: 0, fits: true, n_rows: 64 }),
    updateParams: (p) => { backend.paramCalls.push(p); return ok({ success: true }); },
    indexPattern: (row, col) => {
      backend.indexCalls.push([row, col]);
      return ok({ ci: 0.5, phase_name: backend.phases[0], segments: [], n_bands: 7 });
    },
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });

  const indexApi = new Proxy({
    discoverFiles: () => ok({
      files: [
        { path: 'C:/lib/Al.cif', filename: 'Al.cif', formula: 'Al', element_group: 'Fe' },
        { path: 'C:/lib/Ni.cif', filename: 'Ni.cif', formula: 'Ni', element_group: 'Fe' },
      ],
      groups: ['Fe'],
    }),
    houghReflectorSpecs: () => ok({ specs: {} }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });

  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, indexApi, ebsdApi: quiet, calibrationApi: quiet };
});

import PCRefinement from './PCRefinement';

const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = function getContext() {
    this.__ctx = this.__ctx || fakeContext();
    return this.__ctx;
  };
  backend.phases = [];
  backend.reflectorCalls = [];
  backend.paramCalls = [];
  backend.indexCalls = [];
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => cleanup());

async function addPhases(...names) {
  fireEvent.click(await screen.findByText(/Load Phase\(s\) from CIF/));
  await screen.findByPlaceholderText(/paste the path/i);
  for (const n of names) {
    fireEvent.click(await screen.findByTitle(n));
    await waitFor(() => expect(backend.phases).toContain(n));
  }
}

describe('the indexing settings of the PC Refinement page', () => {
  it('shows nBands, the method and the search limit, and no longer the three dead numbers', async () => {
    render(<PCRefinement />);
    await screen.findByText('Indexing Settings');
    expect(screen.getByText('nBands:')).toBeTruthy();
    expect(screen.getByText('method:')).toBeTruthy();
    expect(screen.getByText('search_limit:')).toBeTruthy();
    for (const gone of ['min_d (A):', 'f_threshold:', 'max_reflectors:']) {
      expect(screen.queryByText(gone), gone).toBeNull();
    }
  });

  it('every parameter has its hover text on the label AND on the control', async () => {
    render(<PCRefinement />);
    await screen.findByText('Indexing Settings');
    const nBands = screen.getByText('nBands:');
    expect(nBands.title).toMatch(/Number of bands/);
    expect(nBands.title).toMatch(/PyEBSDIndex itself defaults to 9/);
    const input = nBands.parentElement.querySelector('input');
    expect(input.title).toBe(nBands.title);

    const method = screen.getByText('method:');
    expect(method.title).toMatch(/Nelder-Mead/);
    expect(method.parentElement.querySelector('select').title).toBe(method.title);

    const limit = screen.getByText('search_limit:');
    expect(limit.title).toMatch(/PSO only|Used by PSO only/);
    expect(limit.parentElement.querySelector('input').title).toBe(limit.title);

    const refl = screen.getByText('Reflectors:');
    expect(refl.title).toMatch(/reflector families/);
  });

  it('with no phase loaded it says what to do, and shows no table', async () => {
    render(<PCRefinement />);
    await screen.findByText('Indexing Settings');
    expect(screen.getByText('Load a phase to choose its reflector families.')).toBeTruthy();
    expect(screen.queryByTestId('reflector-families')).toBeNull();
  });
});

describe('the family table of a loaded phase', () => {
  it('a click stores the choice for THAT phase', async () => {
    render(<PCRefinement />);
    await addPhases('Al');
    const card = await screen.findByTestId('reflector-families');
    fireEvent.click(within(card).getByRole('button', { name: /Reflector families/ }));
    const row = await within(card).findByTestId('family-{220}');
    fireEvent.click(within(row).getByRole('checkbox'));
    await waitFor(() => expect(backend.reflectorCalls.filter((c) => c[0] === 'put')).toHaveLength(1));
    expect(backend.reflectorCalls.find((c) => c[0] === 'put')).toEqual([
      'put', 'Al', { mode: 'custom', families: [[1, 1, 1], [2, 0, 0]] }]);
  });

  it('two phases get a table each, named', async () => {
    render(<PCRefinement />);
    await addPhases('Al', 'Ni');
    const cards = await screen.findAllByTestId('reflector-families');
    expect(cards).toHaveLength(2);
    const holder = screen.getByTestId('pc-reflectors');
    expect(within(holder).getByText('Al')).toBeTruthy();
    expect(within(holder).getByText('Ni')).toBeTruthy();
    fireEvent.click(within(cards[1]).getByRole('button', { name: /Reflector families/ }));
    await waitFor(() => expect(backend.reflectorCalls).toEqual([['get', 'Ni']]));
  });
});

describe('nBands', () => {
  it('an untouched page sends nothing: the backend keeps its own default', async () => {
    render(<PCRefinement />);
    await addPhases('Al');
    await new Promise((r) => setTimeout(r, 50));
    expect(backend.paramCalls).toEqual([]);
  });

  it('a changed nBands is sent, once, and the shown pattern is indexed again', async () => {
    render(<PCRefinement />);
    await addPhases('Al');
    fireEvent.click(await screen.findByText(/pat 1,2/));          // select a calibration pattern
    const input = screen.getByText('nBands:').parentElement.querySelector('input');
    fireEvent.change(input, { target: { value: '9' } });
    await waitFor(() => expect(backend.paramCalls).toEqual([{ n_bands: 9 }]));
    await waitFor(() => expect(backend.indexCalls).toEqual([[1, 2]]));
  });

  it('a cleared field sends nothing', async () => {
    render(<PCRefinement />);
    await screen.findByText('Indexing Settings');
    const input = screen.getByText('nBands:').parentElement.querySelector('input');
    fireEvent.change(input, { target: { value: '' } });
    await new Promise((r) => setTimeout(r, 50));
    expect(backend.paramCalls).toEqual([]);
  });
});

describe('a change of the families', () => {
  it('indexes the selected pattern again, with the new list', async () => {
    render(<PCRefinement />);
    await addPhases('Al');
    fireEvent.click(await screen.findByText(/pat 3,4/));
    const card = await screen.findByTestId('reflector-families');
    fireEvent.click(within(card).getByRole('button', { name: /Reflector families/ }));
    const row = await within(card).findByTestId('family-{220}');
    fireEvent.click(within(row).getByRole('checkbox'));
    await waitFor(() => expect(backend.indexCalls).toEqual([[3, 4]]));
  });
});
