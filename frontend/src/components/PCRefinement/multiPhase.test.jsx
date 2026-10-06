// @vitest-environment jsdom
//
// PC Refinement with several phases, through the real page and a fake backend.
// The tester's duplex steel: austenite and ferrite ticked in the picker, every
// calibration pattern indexed against both, and the page says which phase each
// pattern is.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react';

const backend = vi.hoisted(() => ({ phases: [], paths: {}, calls: [], previews: [], failAdd: null }));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const stem = (p) => p.split(/[\\/]/).pop().replace(/\.cif$/i, '');
  const summary = (name) => ({ name, path: backend.paths[name], space_group: 'Fm-3m', lattice: { a: 3.6, b: 3.6, c: 3.6 } });
  const list = () => backend.phases.map(summary);
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true,
      has_phase: backend.phases.length > 0,
      phase_name: backend.phases[0] ?? null,
      phase_names: [...backend.phases],
      n_patterns: 2,
      pc: [0.5, 0.3, 0.8],
      patterns: [{ index: 0, row: 1, col: 2 }, { index: 1, row: 3, col: 4 }],
    }),
    loadPhase: (path) => {
      backend.calls.push(['load', path]);
      backend.phases = [stem(path)];
      backend.paths = { [stem(path)]: path };
      return ok({ success: true, phase_name: stem(path), space_group: 'Fm-3m', lattice: { a: 1, b: 1, c: 1 } });
    },
    addPhase: (path) => {
      backend.calls.push(['add', path]);
      if (path.endsWith('bad.cif')) {
        return Promise.reject({ response: { data: { detail: 'could not parse bad.cif' } } });
      }
      if (backend.failAdd) return Promise.reject({ response: { data: { detail: backend.failAdd } } });
      backend.phases = [...backend.phases, stem(path)];
      backend.paths[stem(path)] = path;
      return ok({ success: true, phase_name: stem(path), space_group: 'Fm-3m',
                  lattice: { a: 1, b: 1, c: 1 }, phases: list(), n_phases: backend.phases.length });
    },
    removePhase: (name) => {
      backend.calls.push(['remove', name]);
      backend.phases = backend.phases.filter((n) => n !== name);
      return ok({ success: true, phases: list(), n_phases: backend.phases.length });
    },
    optimize: () => ok({ task_id: 'task-1' }),
    getOptimizeStatus: () => ok({
      status: 'completed', progress: 1,
      result: {
        mean_pc: [0.51, 0.3, 0.8], pc_values: [[0.5, 0.3, 0.8], [0.52, 0.3, 0.8]],
        ci: 0.4, segments: [], phase_names: ['austenite', 'ferrite'],
        pattern_phases: [
          { index: 0, phase_name: 'ferrite', phase_index: 1, ci: 0.55 },
          { index: 1, phase_name: 'austenite', phase_index: 0, ci: 0.31 },
        ],
      },
    }),
    renderPreview: (args) => {
      backend.previews.push(args);
      return ok({ simulated_b64: null, ncc: 0.5, orientation_euler_deg: [0, 0, 0],
                  orientation_source: 'hough', indexed_phase: 'ferrite' });
    },
    indexAll: () => ok({
      success: true, global_ci: 0.4, n_indexed: 2,
      results: [
        { index: 0, row: 1, col: 2, ci: 0.5, phase_name: 'ferrite', segments: [], n_bands: 0 },
        { index: 1, row: 3, col: 4, ci: 0.3, phase_name: 'austenite', segments: [], n_bands: 0 },
      ],
    }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });

  const indexApi = new Proxy({
    discoverFiles: () => ok({
      files: [
        { path: 'C:/lib/austenite.cif', filename: 'austenite.cif', formula: 'austenite', element_group: 'Fe' },
        { path: 'C:/lib/ferrite.cif', filename: 'ferrite.cif', formula: 'ferrite', element_group: 'Fe' },
      ],
      groups: ['Fe'],
    }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });

  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, indexApi, ebsdApi: quiet, calibrationApi: quiet };
});

import PCRefinement from './PCRefinement';

// jsdom has no canvas: any 2D context call is a no-op.
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
  backend.paths = {};
  backend.calls = [];
  backend.previews = [];
  backend.failAdd = null;
  const sht = (n, p) => ({ path: `/sht/${n} (${n}) [${p}] {20kV}.sht`, filename: `${n} (${n}) [${p}] {20kV}.sht` });
  globalThis.fetch = vi.fn(() => Promise.resolve({
    json: () => Promise.resolve({ files: [sht('austenite', 'cF4'), sht('ferrite', 'cI2')] }),
  }));
});
afterEach(() => cleanup());

async function openPicker() {
  fireEvent.click(await screen.findByText(/Load Phase\(s\) from CIF/));
  return screen.findByPlaceholderText(/paste the path/i);
}

describe('PC refinement with several phases', () => {
  it('ticking two phases adds both, lists them, and keeps the picker open', async () => {
    render(<PCRefinement />);
    await openPicker();

    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toEqual(['austenite']));
    fireEvent.click(await screen.findByTitle('ferrite'));
    await waitFor(() => expect(backend.phases).toEqual(['austenite', 'ferrite']));

    expect(backend.calls).toEqual([
      ['add', 'C:/lib/austenite.cif'], ['add', 'C:/lib/ferrite.cif'],
    ]);
    const list = await screen.findByTestId('pc-phase-list');
    expect(within(list).getByText('austenite')).toBeTruthy();
    expect(within(list).getByText('ferrite')).toBeTruthy();
    expect(screen.getByText('austenite + ferrite')).toBeTruthy();       // header
    expect(screen.getByPlaceholderText(/paste the path/i)).toBeTruthy(); // still open
  });

  it('unticking a phase removes it and the list goes back to one phase', async () => {
    render(<PCRefinement />);
    await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toHaveLength(1));
    fireEvent.click(await screen.findByTitle('ferrite'));
    await waitFor(() => expect(backend.phases).toHaveLength(2));

    fireEvent.click(await screen.findByTitle('austenite'));      // untick
    await waitFor(() => expect(backend.phases).toEqual(['ferrite']));
    expect(backend.calls.at(-1)).toEqual(['remove', 'austenite']);
    await waitFor(() => expect(screen.queryByTestId('pc-phase-list')).toBeNull());
  });

  it('the × of the list removes that phase', async () => {
    render(<PCRefinement />);
    await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toHaveLength(1));
    fireEvent.click(await screen.findByTitle('ferrite'));
    const list = await screen.findByTestId('pc-phase-list');
    fireEvent.click(within(list).getByLabelText('Remove ferrite from the phases used for calibration'));
    await waitFor(() => expect(backend.phases).toEqual(['austenite']));
  });

  it('a phase typed by path is added through the same endpoint', async () => {
    render(<PCRefinement />);
    const input = await openPicker();
    fireEvent.change(input, { target: { value: 'D:\\mine\\sigma.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(backend.calls).toEqual([['add', 'D:\\mine\\sigma.cif']]));
    await waitFor(() => expect(screen.getByText('Added sigma')).toBeTruthy());
  });

  it('a refusal from the backend is shown, and nothing is listed', async () => {
    render(<PCRefinement />);
    const input = await openPicker();
    fireEvent.change(input, { target: { value: 'D:\\x\\bad.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(screen.getByText('could not parse bad.cif')).toBeTruthy());
    expect(screen.queryByTestId('pc-phase-list')).toBeNull();
  });

  it('after Index All every pattern shows the phase it was indexed as', async () => {
    render(<PCRefinement />);
    await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toHaveLength(1));
    fireEvent.click(await screen.findByTitle('ferrite'));
    await waitFor(() => expect(backend.phases).toHaveLength(2));

    fireEvent.click(await screen.findByText(/Index All/));
    await waitFor(() => expect(screen.getAllByTestId('pattern-phase')).toHaveLength(2));
    expect(screen.getAllByTestId('pattern-phase').map((n) => n.textContent))
      .toEqual(['ferrite', 'austenite']);
  });

  it('with ONE phase the pattern list shows no phase tags (unchanged page)', async () => {
    render(<PCRefinement />);
    await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toHaveLength(1));
    fireEvent.click(await screen.findByText(/Index All/));
    await waitFor(() => expect(screen.getAllByText(/CI 0\./).length).toBeGreaterThan(0));
    expect(screen.queryAllByTestId('pattern-phase')).toHaveLength(0);
    expect(screen.queryByTestId('pc-phase-list')).toBeNull();
  });

  it('a refine reports, per pattern, the phase it is indexed as at the refined PC', async () => {
    render(<PCRefinement />);
    await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toHaveLength(1));
    fireEvent.click(await screen.findByTitle('ferrite'));
    await waitFor(() => expect(backend.phases).toHaveLength(2));

    fireEvent.click(await screen.findByText(/Global PC Refine/));
    await waitFor(
      () => expect(screen.getAllByTestId('pattern-phase').map((n) => n.textContent))
        .toEqual(['ferrite', 'austenite']),
      { timeout: 4000 },
    );
    expect((await screen.findByTestId('pc-phase-summary')).textContent)
      .toBe('ferrite ×1, austenite ×1');
  }, 10000);

  it('the preview renders the master of the phase the selected pattern was indexed as', async () => {
    render(<PCRefinement />);
    await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toHaveLength(1));
    fireEvent.click(await screen.findByTitle('ferrite'));
    await waitFor(() => expect(backend.phases).toHaveLength(2));
    fireEvent.click(await screen.findByText(/Index All/));
    await waitFor(() => expect(screen.getAllByTestId('pattern-phase')).toHaveLength(2));

    // pattern 0 is ferrite, pattern 1 is austenite
    fireEvent.click(screen.getByText(/pat 1,2/));
    await waitFor(
      () => expect(backend.previews.at(-1)?.shtPath).toMatch(/ferrite/),
      { timeout: 4000 },
    );
    fireEvent.click(screen.getByText(/pat 3,4/));
    await waitFor(
      () => expect(backend.previews.at(-1)?.shtPath).toMatch(/austenite/),
      { timeout: 4000 },
    );
    expect(await screen.findByText('· phase ferrite')).toBeTruthy();
  }, 15000);

  // ---- phases are identified by FILE, errors are words ------------------------

  it('another file with the name of a loaded phase is refused, said so, and nothing is asked', async () => {
    render(<PCRefinement />);
    const input = await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));                // library austenite
    await waitFor(() => expect(backend.phases).toEqual(['austenite']));
    backend.calls = [];
    fireEvent.change(input, { target: { value: 'D:/mine/austenite.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await screen.findByText(/A different file named austenite is already in use \(C:\/lib\/austenite.cif\)/);
    expect(backend.calls).toEqual([]);                                     // not even asked
    expect(backend.phases).toEqual(['austenite']);
    expect(backend.paths.austenite).toBe('C:/lib/austenite.cif');
  });

  it('a library entry that shares its name with a loaded external file is NOT shown ticked, and ticking it is refused', async () => {
    render(<PCRefinement />);
    const input = await openPicker();
    fireEvent.change(input, { target: { value: 'D:/mine/austenite.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(backend.phases).toEqual(['austenite']));
    const lib = await screen.findByTitle('austenite');
    expect(lib.parentElement.querySelector('input[type="checkbox"]').checked).toBe(false);
    backend.calls = [];
    fireEvent.click(lib);
    await screen.findAllByText(/A different file named austenite is already in use \(D:\/mine\/austenite.cif\)/);
    expect(backend.calls).toEqual([]);
    expect(backend.phases).toEqual(['austenite']);                         // the external one stays
    expect(backend.paths.austenite).toBe('D:/mine/austenite.cif');
  });

  it('unticking removes the phase of THAT file only', async () => {
    render(<PCRefinement />);
    await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toHaveLength(1));
    fireEvent.click(await screen.findByTitle('ferrite'));
    await waitFor(() => expect(backend.phases).toHaveLength(2));
    fireEvent.click(await screen.findByTitle('ferrite'));                  // untick ferrite
    await waitFor(() => expect(backend.phases).toEqual(['austenite']));
    expect(backend.calls.at(-1)).toEqual(['remove', 'ferrite']);
  });

  it('a coded refusal from the backend is worded in the language of the page', async () => {
    render(<PCRefinement />);
    const input = await openPicker();
    backend.failAdd = { code: 'phase_unreadable', message: 'raw english', params: { reason: 'SyntaxError@char0' } };
    fireEvent.change(input, { target: { value: 'D:/x/junk.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await screen.findByText('The file could not be read as a phase: SyntaxError@char0');
    backend.failAdd = { code: 'phase_limit', message: 'x', params: { max: 8 } };
    fireEvent.change(input, { target: { value: 'D:/x/other.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await screen.findByText('At most 8 phases can be used together; remove one first.');
  });

  it('removing a phase takes the stale "phase per pattern" row away', async () => {
    render(<PCRefinement />);
    await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toHaveLength(1));
    fireEvent.click(await screen.findByTitle('ferrite'));
    await waitFor(() => expect(backend.phases).toHaveLength(2));
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    await screen.findByTestId('pc-phase-summary', {}, { timeout: 4000 });
    const list = await screen.findByTestId('pc-phase-list');
    fireEvent.click(within(list).getByLabelText('Remove ferrite from the phases used for calibration'));
    await waitFor(() => expect(backend.phases).toEqual(['austenite']));
    await waitFor(() => expect(screen.queryByTestId('pc-phase-summary')).toBeNull());
  }, 15000);

  it('a phase added the same way later also clears it', async () => {
    render(<PCRefinement />);
    await openPicker();
    fireEvent.click(await screen.findByTitle('austenite'));
    await waitFor(() => expect(backend.phases).toHaveLength(1));
    fireEvent.click(await screen.findByTitle('ferrite'));
    await waitFor(() => expect(backend.phases).toHaveLength(2));
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    await screen.findByTestId('pc-phase-summary', {}, { timeout: 4000 });
    const input = screen.getByPlaceholderText(/paste the path/i);
    fireEvent.change(input, { target: { value: 'D:/x/sigma.cif' } });
    fireEvent.click(screen.getByText('Add'));
    await waitFor(() => expect(backend.phases).toHaveLength(3));
    await waitFor(() => expect(screen.queryByTestId('pc-phase-summary')).toBeNull());
  }, 15000);
});
