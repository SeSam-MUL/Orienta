// @vitest-environment jsdom
//
// "Find PC" on the PC page, through the real page and a fake backend: nine calibration patterns are picked
// from the scan when fewer than five are on the list, the search runs without a starting value, Cancel works
// in every step, and the result panel says what the search says about itself, gives the PC to the file in the
// words every way of giving a PC uses and offers the one step back.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within } from '@testing-library/react';

const backend = vi.hoisted(() => ({
  patterns: [], phaseLoaded: true, source: 'header', pc: [0.5, 0.5, 0.5], calls: [], picked: [], result: {},
  dataset: 'scan_a', jobStatus: 'completed', pickGate: null, optimizeError: null, putBackError: null,
  putBackAnswer: null, previousPc: null, notes: [], pickError: null, perPixel: false, mapBeforeRun: false,
  updateAnswer: {}, polls: 0, unknown: [], parent: null,
  kind: null, refinedKind: 'find', calibration: null, convention: 'bruker', conventionKnown: false,
  phaseNames: ['Ni'], nSampled: 180, pickedNext: [], excludeError: null, convertError: null, statusGate: null,
}));

const coded = (status, code, detail = 'English sentence') => Object.assign(new Error(String(status)), {
  response: { status, data: { detail }, headers: { 'x-orienta-code': code } },
});

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    // a test can hold the status answer back (statusGate) to see what the page shows before the backend has said
    status: () => (backend.statusGate || Promise.resolve()).then(() => ({ data: {
      has_detector: true, has_phase: backend.phaseLoaded, phase_name: backend.phaseLoaded ? backend.phaseNames[0] : null,
      phase_names: backend.phaseLoaded ? backend.phaseNames : [], n_patterns: backend.patterns.length,
      pc: backend.pc, pc_basis: backend.perPixel ? 'mean' : 'single', pc_per_pixel: backend.perPixel,
      // where the PC in use came from, as the backend tells it: the kind follows the source unless a test sets it
      pc_in_use: {
        pc: backend.source === 'missing' ? null : backend.pc, per_pixel: backend.perPixel,
        kind: backend.kind || ({ missing: 'none', header: 'file', manual: 'typed', refined: backend.refinedKind,
          refined_map: 'grid' }[backend.source] || 'file'),
        calibration: backend.calibration, convention: backend.convention, convention_known: backend.conventionKnown,
      },
      pc_map_shape: backend.perPixel ? [2, 2] : null,
      patterns: backend.patterns.map((p, i) => ({ index: i, ...p })), pc_source: backend.source,
    } })),
    getPatternImage: () => ok({ image: 'AAAA' }),
    getPatternResult: () => ok({ success: true, segments: [[[0, 0], [10, 10]]], ci: 0.4 }),
    autoPick: async (replace, exclude) => {
      backend.calls.push(exclude ? ['autoPick', replace, exclude] : ['autoPick', replace]);
      if (backend.pickGate) await backend.pickGate;
      if (backend.pickError) throw backend.pickError;
      if (exclude && backend.excludeError) throw backend.excludeError;
      // "other patterns": the next list the test has prepared, else the same nine
      const picked = exclude && backend.pickedNext.length ? backend.pickedNext.shift() : backend.picked;
      backend.patterns = picked.map((p) => ({ ...p }));
      return { data: { success: true, n_patterns: picked.length, patterns: picked,
                       prepared: true, background: 'mean', n_sampled: backend.nSampled, stride: 1, notes: backend.notes } };
    },
    convert: (pc, source, target, shape) => {
      backend.calls.push(['convert', pc, source, target, shape]);
      if (backend.convertError) return Promise.reject(backend.convertError);
      // hand calculation for PC (0.5026, 0.3272, 0.8458) of a 128 x 156 pattern (K = 156 / 128 = 1.21875):
      // Oxford (x, (1 - y) / K, z / K) = (0.5026, 0.552041, 0.693994); EDAX/TSL (x, 1 - y, z) = (0.5026, 0.6728, 0.8458)
      return ok({ pc: target === 'oxford' ? [0.5026, 0.552041, 0.693994] : [0.5026, 0.6728, 0.8458], source, target });
    },
    optimize: async (indices, method, limit, mode) => {
      backend.calls.push(['optimize', indices, method, limit, mode]);
      if (backend.optimizeGate) await backend.optimizeGate;
      if (backend.optimizeError) throw backend.optimizeError;
      return { data: { task_id: 'task-1' } };
    },
    cancelOptimize: (id) => { backend.calls.push(['cancel', id]); return ok({ status: 'cancelling' }); },
    getOptimizeStatus: () => {
      backend.polls += 1;
      if (backend.jobStatus === 'completed') {
        // the search wrote its PC: the status now reports it (a map is replaced by that one PC)
        backend.pc = backend.result.mean_pc || backend.pc;
        backend.source = 'refined';
        backend.refinedKind = backend.result.mode === 'refine' ? 'refine' : 'find';
        if (backend.result.pc_map_action === 'replaced') backend.perPixel = false;
        return ok({ status: 'completed', progress: 1, result: backend.result });
      }
      return ok({ status: backend.jobStatus, progress: 0.3, error: 'The search could not be computed.' });
    },
    previous: () => ok({ dataset: backend.dataset, previous: backend.result.previous || null }),
    putBack: (dataset) => {
      backend.calls.push(['putBack', dataset]);
      if (backend.putBackError) return Promise.reject(backend.putBackError);
      const prev = backend.result.previous;
      backend.pc = prev.pc || [0.5, 0.5, 0.5];
      backend.source = prev.had_pc ? 'header' : 'missing';
      backend.perPixel = backend.mapBeforeRun;                         // the map comes back with the PC
      return ok(backend.putBackAnswer || {
        dataset, pc: prev.pc, had_pc: prev.had_pc, restored: [dataset], not_restored: [],
        pc_per_pixel: backend.perPixel });
    },
    updatePC: (...args) => {
      backend.calls.push(['updatePC', ...args.slice(0, 3)]);
      // the store takes the PC (and says so in the status) unless the test says it did not
      if (backend.updateAnswer.pc_stored !== false) { backend.pc = args.slice(0, 3); backend.source = 'manual'; }
      return ok({ success: true, ci: 0.2, segments: [], pc_map_action: 'none', pc_stored: true, ...backend.updateAnswer });
    },
  }, {
    // An unknown method is not answered with {}: a misspelled call (or one the page makes that this test
    // never meant to allow) throws, and is listed so the test cannot swallow it in a catch.
    get: (t, k) => {
      if (k in t || typeof k === 'symbol' || k === 'then') return t[k];
      return () => { backend.unknown.push(k); throw new Error(`pcApi.${String(k)} is not in this test's fake`); };
    },
  });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  const ebsdApi = new Proxy({ info: () => ok({ active_dataset: backend.dataset }) },
    { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const calibrationApi = new Proxy({ getEntry: () => ok({ parent_name: backend.parent }) },
    { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  return { ...orig, pcApi, indexApi: quiet, ebsdApi, calibrationApi };
});

import PCRefinement from './PCRefinement';
import useDataStore from '../../stores/useDataStore';

// a canvas that remembers what was drawn on it
const recordingContext = () => {
  const calls = [];
  return new Proxy({ calls }, {
    get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => { calls.push(k); return {}; })),
    set: (t, k, v) => { t[k] = v; return true; },
  });
};
class FakeImage {
  set src(_v) { this.width = 100; this.height = 100; queueMicrotask(() => this.onload && this.onload()); }
}

const NINE = Array.from({ length: 9 }, (_, i) => ({ row: i + 1, col: 2 * i + 1, quality: 0.5 }));
const FOUND = [0.5026, 0.3272, 0.8458];
const RESULT = {
  mean_pc: FOUND, pc_values: [[0.5, 0.33, 0.85]], ci: 0.4, global_ci: 0.4, segments: [[[0, 0], [10, 10]]],
  pattern_results: [], pc_map_action: 'none', dataset: 'scan_a', dataset_changed: false, mode: 'find',
  file_pc: [0.5026, 0.3273, 0.8458], file_pc_deviation: 0.0012, pattern_size: [128, 156],
  // What each of three patterns prefers on its own (fake values): the ranges are x 0.02, y 0.01, z 0.015, so with a
  // 128 x 156 pattern (x counts in 156 columns, y and z in 128 rows) 3.12, 1.28 and 1.92 pixels.
  search: {
    mode: 'find', recentred: 0, far_box_used: false, on_edge: [], second_position: null,
    confidence: { state: 'notJudged', reasons: [], contrast: 0.8 },
    per_pattern: [[0.50, 0.33, 0.85], [0.52, 0.34, 0.865], [0.51, 0.335, 0.855]],
  },
  quality: { contrast: 0.8, warning: null, axes: [], n_patterns: 9, contrast_before: null },
  previous: { pc: [0.5019, 0.3301, 0.842], had_pc: true },
  pc_warning: null, pc_warning_codes: [],
};
const ON_EDGE = {
  ...RESULT,
  search: { ...RESULT.search, on_edge: ['y', 'z'] },
  quality: { ...RESULT.quality, contrast: 0.4, warning: 'edge', axes: ['y', 'z'] },
  pc_warning: 'English fallback',
  pc_warning_codes: [{ code: 'pcOnBoxEdge', params: { axes: 'y, z' } }],
};
const NOT_MIN = {
  ...RESULT,
  search: { ...RESULT.search, confidence: { state: 'notJudged', reasons: ['notAMinimum'], contrast: 0.1 } },
  quality: { ...RESULT.quality, contrast: 0.1, warning: 'notAMinimum' },
  pc_warning: 'English fallback', pc_warning_codes: [{ code: 'pcNotAMinimum', params: {} }],
};
const OTHER = { pc: [0.3604, 0.4451, 0.7402], cost: 8.92 };
const TWO = {
  ...RESULT,
  search: { ...RESULT.search, second_position: OTHER },
  quality: { ...RESULT.quality, contrast: 0.1, warning: 'secondPosition', second_position: OTHER, best_cost: 8.45 },
  pc_warning: 'English fallback',
  pc_warning_codes: [{ code: 'pcSecondPosition', params: { pc: '0.360, 0.445, 0.740', cost: 8.92, best_cost: 8.45 } }],
};

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = function getContext() {
    this.__ctx = this.__ctx || recordingContext();
    return this.__ctx;
  };
  globalThis.Image = FakeImage;
  Object.assign(backend, {
    patterns: [], phaseLoaded: true, source: 'header', pc: [0.5019, 0.3301, 0.842], calls: [], picked: NINE,
    result: RESULT, dataset: 'scan_a', jobStatus: 'completed', pickGate: null, optimizeError: null,
    putBackError: null, putBackAnswer: null, notes: [], optimizeGate: null, pickError: null,
    perPixel: false, mapBeforeRun: false, updateAnswer: {}, polls: 0, unknown: [], parent: null,
    kind: null, refinedKind: 'find', calibration: null, convention: 'bruker', conventionKnown: false,
    phaseNames: ['Ni'], nSampled: 180, pickedNext: [], excludeError: null, convertError: null, statusGate: null,
  });
  useDataStore.setState({ gridShape: [90, 20] });
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => {
  cleanup();
  // every call the page made to a pcApi method the fake does not know failed the test, also when a catch ate it
  expect(backend.unknown).toEqual([]);
});

const optimizeCalls = () => backend.calls.filter((c) => c[0] === 'optimize');
const patternsOf = (n) => NINE.slice(0, n).map((p) => ({ row: p.row, col: p.col }));
const five = () => patternsOf(5);
const findButton = () => screen.findByRole('button', { name: 'Find PC' });
const ready = async () => { await screen.findByTestId('pc-file-head'); };
const waitResult = (timeout = 4000) => screen.findByTestId('pc-grant', {}, { timeout });
// A PC typed into the three Bruker fields and applied with "Use these numbers" (typing alone applies nothing).
const typeAndUse = (x = '0.51', y = '0.33', z = '0.84') => {
  for (const [name, v] of [['PC X:', x], ['PC Y:', y], ['PC Z:', z]]) {
    fireEvent.change(screen.getByLabelText(name), { target: { value: v } });
  }
  fireEvent.click(screen.getByRole('button', { name: 'Use these numbers' }));
};

describe('Find PC: the search', () => {
  it('picks nine patterns first when fewer than five are on the list, and searches over them in find mode', async () => {
    backend.patterns = [{ row: 0, col: 0 }, { row: 1, col: 1 }];
    render(<PCRefinement />);
    fireEvent.click(await findButton());
    await waitFor(() => expect(optimizeCalls()).toHaveLength(1));
    expect(backend.calls[0]).toEqual(['autoPick', true]);                         // replaces the list
    // flat index = row * columns + column; the grid has 20 columns (gridShape [rows, cols] = [90, 20])
    expect(optimizeCalls()[0][1]).toEqual(NINE.map((p) => p.row * 20 + p.col));
    expect(optimizeCalls()[0][4]).toBe('find');
  });

  it('does not pick when exactly five patterns are on the list, and searches over those five', async () => {
    backend.patterns = patternsOf(5);
    render(<PCRefinement />);
    expect(await screen.findAllByTitle(/^Pattern at row/)).toHaveLength(5);
    fireEvent.click(await findButton());
    await waitFor(() => expect(optimizeCalls()).toHaveLength(1));
    expect(backend.calls.some((c) => c[0] === 'autoPick')).toBe(false);
    expect(optimizeCalls()[0][1]).toEqual(NINE.slice(0, 5).map((p) => p.row * 20 + p.col));
  });

  it('picks when four patterns are on the list: four are one short of what the search needs', async () => {
    backend.patterns = patternsOf(4);
    render(<PCRefinement />);
    expect(await screen.findAllByTitle(/^Pattern at row/)).toHaveLength(4);
    fireEvent.click(await findButton());
    await waitFor(() => expect(optimizeCalls()).toHaveLength(1));
    expect(backend.calls[0]).toEqual(['autoPick', true]);
    expect(optimizeCalls()[0][1]).toEqual(NINE.map((p) => p.row * 20 + p.col));
  });

  it('the backend refusing the pick because a search runs is said in words, and no search starts', async () => {
    backend.patterns = patternsOf(2);
    backend.pickError = Object.assign(new Error('409'), { response: { status: 409, data: {
      detail: { code: 'pc_job_running', message: 'A PC optimization is running.' } } } });
    render(<PCRefinement />);
    fireEvent.click(await findButton());
    expect(await screen.findByText(/A PC refinement or grid calibration is running/)).toBeTruthy();
    expect(backend.calls).toEqual([['autoPick', true]]);
    expect(optimizeCalls()).toHaveLength(0);
    await waitFor(() => expect(screen.queryByTitle(/Cancel the running PC optimization/)).toBeNull());
  });

  it('leaving the page during a run stops asking the backend how it is going', async () => {
    backend.patterns = five();
    backend.jobStatus = 'running';
    const { unmount } = render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    await waitFor(() => expect(backend.polls).toBeGreaterThan(0), { timeout: 4000 });   // polled once a second
    unmount();
    const asked = backend.polls;
    await new Promise((r) => setTimeout(r, 2300));                  // two more seconds would have been two more polls
    expect(backend.polls).toBe(asked);
  }, 15000);

  it('Global PC Refine is still a refine', async () => {
    backend.patterns = [{ row: 1, col: 2 }];
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    await waitFor(() => expect(optimizeCalls()).toHaveLength(1));
    expect(optimizeCalls()[0][4]).toBe('refine');
    expect(optimizeCalls()[0][1]).toEqual([1 * 20 + 2]);
  });

  it('says its three steps, one under the other', async () => {
    backend.patterns = [];
    backend.jobStatus = 'running';
    render(<PCRefinement />);
    fireEvent.click(await findButton());
    const status = await screen.findByText(/Searching the pattern centre over 9 patterns\.\.\./);
    expect(status.textContent).toBe(
      '9 calibration patterns picked; background removed with the mean of the sampled patterns.\n'
      + 'Searching the pattern centre over 9 patterns...');
    backend.jobStatus = 'completed';
    await waitFor(() => expect(screen.getByText(/Search finished\./).textContent).toMatch(/\nSearch finished\.$/),
      { timeout: 4000 });
  }, 10000);

  it('names the pick notes the backend gives', async () => {
    backend.notes = ['meanFromFewPatterns'];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Pick patterns' }));
    await screen.findByText(/The mean background comes from only a few patterns/);
  });

  it('words a refusal of the search in the reader\'s language', async () => {
    backend.patterns = five();
    backend.optimizeError = coded(400, 'tooFewPatterns', 'Find PC needs at least 5 calibration patterns, got 4');
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    expect(await screen.findByText('Find PC needs at least 5 calibration patterns.')).toBeTruthy();
  });

  it('needs a phase: Find PC is off, the page says why, and nothing starts', async () => {
    backend.phaseLoaded = false;
    backend.patterns = five();
    render(<PCRefinement />);
    expect(await screen.findByText(/Load a phase first: the search compares the bands/)).toBeTruthy();
    const button = screen.getByRole('button', { name: 'Find PC' });
    expect(button.disabled).toBe(true);
    fireEvent.click(button);
    await new Promise((r) => setTimeout(r, 50));
    expect(backend.calls).toEqual([]);
  });

  it('Pick patterns needs no phase: it picks, tells how, and asks the page to refresh its list', async () => {
    backend.phaseLoaded = false;
    const seen = [];
    const listener = () => seen.push(1);
    window.addEventListener('pc-patterns-updated', listener);
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Pick patterns' }));
    await screen.findByText(/9 calibration patterns picked; background removed with the mean of the sampled patterns/);
    expect(backend.calls).toEqual([['autoPick', true]]);
    expect(seen.length).toBeGreaterThan(0);
    window.removeEventListener('pc-patterns-updated', listener);
  });
});

describe('Find PC: a search that fails', () => {
  it('says so where the run was started (there is no result panel to say it in)', async () => {
    backend.patterns = five();
    backend.jobStatus = 'failed';
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    expect(await screen.findByText('The search could not be computed.', {}, { timeout: 4000 })).toBeTruthy();
    expect(screen.queryByTitle(/Cancel the running PC optimization/)).toBeNull();
    expect(screen.queryByTestId('pc-grant')).toBeNull();
  }, 10000);
});

describe('Find PC: Cancel', () => {
  it('while the patterns are picked: nothing is searched afterwards, and the page says what is left', async () => {
    let open;
    backend.pickGate = new Promise((resolve) => { open = resolve; });
    render(<PCRefinement />);
    fireEvent.click(await findButton());
    const cancel = await screen.findByTitle(/Cancel the running PC optimization/);
    fireEvent.click(cancel);
    expect(await screen.findByText(/Cancelling/)).toBeTruthy();
    open();
    expect(await screen.findByText(/Cancelled before the search started\. The PC is unchanged/)).toBeTruthy();
    await waitFor(() => expect(screen.queryByTitle(/Cancel the running PC optimization/)).toBeNull());
    expect(optimizeCalls()).toHaveLength(0);
  });

  it('while the search runs: asks the backend to stop it and waits until it says "cancelled"', async () => {
    backend.patterns = five();
    backend.jobStatus = 'running';
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    const cancel = await screen.findByTitle(/Cancel the running PC optimization/);
    await waitFor(() => expect(optimizeCalls()).toHaveLength(1));
    fireEvent.click(cancel);
    await waitFor(() => expect(backend.calls).toContainEqual(['cancel', 'task-1']));
    expect(screen.getByTitle(/Cancel the running PC optimization/)).toBeTruthy();   // still running until it says so
    backend.jobStatus = 'cancelled';
    await waitFor(() => expect(screen.queryByTitle(/Cancel the running PC optimization/)).toBeNull(), { timeout: 4000 });
    expect(screen.getByText(/Cancelled\. The PC, the stored values and the patterns are unchanged\./)).toBeTruthy();
    expect(screen.queryByTestId('pc-grant')).toBeNull();
  }, 15000);

  it('while the job is being started: the job is stopped as soon as it has a name', async () => {
    let open;
    backend.optimizeGate = new Promise((resolve) => { open = resolve; });      // the request has not been answered
    backend.patterns = five();
    backend.jobStatus = 'cancelled';
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    await waitFor(() => expect(optimizeCalls()).toHaveLength(1));
    fireEvent.click(await screen.findByTitle(/Cancel the running PC optimization/));
    expect(await screen.findByText(/Cancelling/)).toBeTruthy();
    expect(backend.calls.some((c) => c[0] === 'cancel')).toBe(false);          // there is no job to name yet
    open();
    await waitFor(() => expect(backend.calls).toContainEqual(['cancel', 'task-1']));
    await waitFor(() => expect(screen.queryByTitle(/Cancel the running PC optimization/)).toBeNull(), { timeout: 4000 });
    expect(screen.getByText(/Cancelled\. The PC, the stored values and the patterns are unchanged\./)).toBeTruthy();
  }, 15000);
});

describe('Find PC: whose PC the result is', () => {
  it('a result for another file is not shown as the PC of the file in front of the user', async () => {
    backend.patterns = five();
    backend.result = { ...RESULT, dataset: 'scan_b', dataset_changed: true };
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    expect(await screen.findByText(/finished for 'scan_b' after you switched files/, {}, { timeout: 4000 })).toBeTruthy();
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    expect(screen.queryByText('Results')).toBeNull();
  }, 10000);

  it('the open file changing while the patterns are picked starts no search', async () => {
    let open;
    backend.pickGate = new Promise((resolve) => { open = resolve; });
    render(<PCRefinement />);
    fireEvent.click(await findButton());
    await waitFor(() => expect(backend.calls).toEqual([['autoPick', true]]));
    backend.dataset = 'scan_b';
    open();
    expect(await screen.findByText(/The open file changed while the patterns were picked/)).toBeTruthy();
    expect(optimizeCalls()).toHaveLength(0);
  });

  it('the head names the open file and the PC in use, and says "none" for a file without one', async () => {
    backend.source = 'missing';
    render(<PCRefinement />);
    const head = await screen.findByTestId('pc-file-head');
    await waitFor(() => expect(head.textContent).toBe('Open file: scan_a · PC in use: none'));
    cleanup();
    backend.source = 'header';
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5019, 0.3301, 0.8420 (Bruker) (from the file)'));
  });

  it('a file without a PC says so above the one Find PC button', async () => {
    backend.source = 'missing';
    render(<PCRefinement />);
    const hint = await screen.findByTestId('pc-missing-hint');
    expect(within(hint).getByText('No PC yet, indexing is blocked.')).toBeTruthy();
    expect(hint.textContent).toMatch(/This file has no pattern centre \(PC\); the fields show a placeholder \(0\.5, 0\.5, 0\.5\)\. Find one from your patterns/);
    expect(hint.textContent).toMatch(/type your numbers under Detector Settings/);
    expect(screen.getAllByRole('button', { name: 'Find PC' })).toHaveLength(1);
  });

  it('once something else has changed the PC, the result no longer claims to be the PC in use', async () => {
    backend.patterns = five();
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    await waitResult();
    backend.pc = [0.51, 0.31, 0.81];                       // a PC typed meanwhile
    window.dispatchEvent(new Event('pc-patterns-updated'));
    expect(await screen.findByText('The PC in use is no longer this result.')).toBeTruthy();
    expect(screen.queryByTestId('pc-grant')).toBeNull();
  }, 10000);
});

describe('Find PC: the result', () => {
  const run = async () => {
    backend.patterns = five();
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    return waitResult();
  };

  it('gives the PC to the file in the words every way of giving a PC uses, and the head follows', async () => {
    await run();
    // the numbers stand in the PC row above, so the green line does not repeat them
    expect(screen.getByTestId('pc-grant-text').textContent).toBe(
      'PC in use for scan_a. Kept until you close Orienta. The file on disk is not changed.');
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5026, 0.3272, 0.8458 (Bruker) (from Find PC)'));
    expect(screen.getByText('Previous PC: 0.5019, 0.3301, 0.8420')).toBeTruthy();
  }, 10000);

  it('says it cannot judge the result itself and asks for one look at the bands, and not that the patterns do not fix the PC', async () => {
    await run();
    expect(screen.getByTestId('pc-result-head').textContent)
      .toBe('Search ended normally. Orienta cannot judge the result itself, so check the bands once.');
    expect(screen.getByTestId('pc-good-enough').textContent)
      .toBe('Good enough means: the red lines lie on the bands of the pattern, also in the corners.');
    expect(screen.queryByText(/do not fix this PC clearly/)).toBeNull();
  }, 10000);

  it('words "unclear" with its number only when the backend sends that code', async () => {
    backend.result = { ...RESULT, pc_warning: 'English fallback',
      pc_warning_codes: [{ code: 'pcUnclear', params: { contrast: 0.1 } }] };
    await run();
    expect(screen.getByText(/do not fix this PC clearly/).textContent).toMatch(/fit contrast 0\.1/);
  }, 10000);

  it('shows the PC with the number of patterns, the fit contrast on its scale and how far the patterns disagree', async () => {
    await run();
    expect(screen.getByTestId('pc-result-pc').textContent).toBe('0.5026, 0.3272, 0.8458');
    expect(screen.getByText('from 9 patterns')).toBeTruthy();
    expect(screen.getByTestId('pc-contrast').textContent).toBe('Fit contrast 0.8 of 1, higher is better');
    // per axis the range of what three patterns prefer: x 0.02 (3.12 px of 156 columns), y 0.01 (1.28 px of 128 rows),
    // z 0.015 (1.92 px of 128 rows): the largest is 0.02, the largest in pixels 3
    expect(screen.getByTestId('pc-spread').textContent).toBe('Patterns disagree by up to 0.02 in x, y or z, about 3 pixels');
  }, 10000);

  it('keeps the file\'s own PC under Details, with the difference to four decimals', async () => {
    await run();
    expect(within(screen.getByTestId('pc-details')).getByTestId('pc-file-vs-result').textContent).toBe(
      'This file carries its own PC, 0.5026, 0.3273, 0.8458. The largest difference in x, y or z is 0.0012.');
  }, 10000);

  it('says when the search had to move its area or used the second one', async () => {
    backend.result = { ...RESULT, search: { ...RESULT.search, recentred: 1 } };
    await run();
    expect(screen.getByText('The search area was moved to find this PC.')).toBeTruthy();
    cleanup();
    backend.result = { ...RESULT, search: { ...RESULT.search, far_box_used: true, recentred: 1 } };
    await run();
    expect(screen.getByText(/A second search for a detector farther from the sample fitted the patterns much better/)).toBeTruthy();
    expect(screen.queryByText('The search area was moved to find this PC.')).toBeNull();
  }, 15000);

  it('"Show the bands over a pattern" opens the pattern with the bands drawn from this PC', async () => {
    await run();
    expect(screen.queryByTestId('pc-bands-pane')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Show the bands over a pattern' }));
    expect(await screen.findByTestId('pc-bands-pane')).toBeTruthy();
    expect(screen.getByTestId('pc-bands-canvas').getAttribute('data-segment-count')).toBe('1');
    expect(screen.queryByTestId('pc-decide')).toBeNull();
  }, 10000);

  it('puts the previous PC back after a question, loads the page from the backend and says what happened', async () => {
    backend.putBackAnswer = null;
    await run();
    fireEvent.click(screen.getByRole('button', { name: 'Put it back' }));
    expect(backend.calls.some((c) => c[0] === 'putBack')).toBe(false);          // asks first
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    expect(await screen.findByTestId('pc-undo-done')).toBeTruthy();
    expect(backend.calls).toContainEqual(['putBack', 'scan_a']);
    expect(screen.getByTestId('pc-undo-done').textContent).toBe(
      'Previous PC put back for scan_a: 0.5019, 0.3301, 0.8420. Kept until you close Orienta. The file on disk is not changed.');
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5019, 0.3301, 0.8420 (Bruker) (from the file)'));
  }, 10000);

  it('a file that had no PC goes back to none: "Remove it", and the hint is back', async () => {
    backend.result = { ...RESULT, previous: { pc: null, had_pc: false } };
    await run();
    expect(screen.getByText('This file had no PC before.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Remove this PC' }));
    expect(screen.getByText('Remove the PC of scan_a? It will have no PC again.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Yes, remove this PC' }));
    await waitFor(() => expect(screen.getByTestId('pc-undo-done').textContent).toMatch(/^PC removed\. scan_a has no PC again\./));
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent).toBe('Open file: scan_a · PC in use: none'));
    expect(await screen.findByTestId('pc-missing-hint')).toBeTruthy();
  }, 10000);

  it('names the copies that keep the new PC', async () => {
    backend.putBackAnswer = { dataset: 'scan_a', pc: [0.5019, 0.3301, 0.842], had_pc: true,
                              restored: ['scan_a'], not_restored: ['scan_a copy'], pc_per_pixel: false };
    await run();
    fireEvent.click(screen.getByRole('button', { name: 'Put it back' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    expect((await screen.findByTestId('pc-not-restored')).textContent)
      .toBe('Copies made since keep this PC: scan_a copy.');
  }, 10000);

  it('when a file was loaded since, the button goes and the page says why', async () => {
    backend.putBackError = coded(409, 'nothingToPutBack', 'There is no earlier PC of this dataset to put back.');
    await run();
    fireEvent.click(screen.getByRole('button', { name: 'Put it back' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    expect(await screen.findByText('This PC can no longer be put back: a file was loaded since.')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Put it back' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Yes, put it back' })).toBeNull();
  }, 10000);

  it('a step back refused because a search runs says so and keeps the button', async () => {
    backend.putBackError = Object.assign(new Error('409'), { response: { status: 409, data: {
      detail: { code: 'pc_job_running', message: 'A PC optimization is running.' } } } });
    await run();
    fireEvent.click(screen.getByRole('button', { name: 'Put it back' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    expect((await screen.findByRole('alert')).textContent).toMatch(/A PC refinement or grid calibration is running/);
    expect(screen.getByRole('button', { name: 'Put it back' })).toBeTruthy();
  }, 10000);
});

describe('Find PC: on a copy of a dataset', () => {
  it('the head names the copy that is open, not its parent, and follows the PC found for it', async () => {
    backend.dataset = 'scan_a copy';
    backend.parent = 'scan_a';
    backend.result = { ...RESULT, dataset: 'scan_a copy' };
    backend.patterns = five();
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a copy · PC in use: 0.5019, 0.3301, 0.8420 (Bruker) (from the file)'));
    fireEvent.click(await findButton());
    await waitResult();
    expect(screen.getByTestId('pc-grant-text').textContent).toMatch(/^PC in use for scan_a copy\. Kept/);
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a copy · PC in use: 0.5026, 0.3272, 0.8458 (Bruker) (from Find PC)'));
  }, 10000);
});

describe('Find PC: a per-pixel PC map', () => {
  const run = async () => {
    backend.patterns = five();
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    return waitResult();
  };

  it('the head says the PC in use is the mean of a map, not one PC', async () => {
    backend.perPixel = true;
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: per-pixel map, mean 0.5019, 0.3301, 0.8420 (Bruker) (from the file)'));
  });

  it('after Find PC replaced the map the head names one PC again', async () => {
    backend.perPixel = true;
    backend.result = { ...RESULT, pc_map_action: 'replaced' };
    await run();
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5026, 0.3272, 0.8458 (Bruker) (from Find PC)'));
  }, 10000);

  it('a put back brings the map back: the page stops saying it was replaced and the head says map again', async () => {
    backend.perPixel = true;
    backend.mapBeforeRun = true;
    backend.result = { ...RESULT, pc_map_action: 'replaced' };
    await run();
    expect(screen.getByTestId('pc-map-note').textContent).toMatch(/The PC map was replaced by the refined PC/);
    fireEvent.click(screen.getByRole('button', { name: 'Put it back' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    await screen.findByTestId('pc-undo-done');
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: per-pixel map, mean 0.5019, 0.3301, 0.8420 (Bruker) (from the file)'));
    const note = screen.getByTestId('pc-map-note').textContent;
    expect(note).toMatch(/A per-pixel PC map is active \(2 × 2 pixels\)/);
    expect(note).not.toMatch(/replaced/);
  }, 10000);
});

describe('Find PC: a PC the calibration store did not take', () => {
  const NOT_STORED = 'This PC is on this page only: indexing will not use it. Load the file again.';

  it('is said in the warning, and the page does not claim it is in use', async () => {
    backend.result = { ...RESULT, pc_stored: false, pc_warning: 'English fallback',
      pc_warning_codes: [{ code: 'pcNotStored', params: {} }] };
    backend.patterns = five();
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    expect(await screen.findByText(NOT_STORED, { exact: false }, { timeout: 4000 })).toBeTruthy();
    expect(screen.queryByTestId('pc-grant')).toBeNull();                     // no "PC in use for ..."
    expect(screen.queryByText('The PC in use is no longer this result.')).toBeNull();   // that would be another untruth
  }, 10000);

  it('a PC the store took (pc_stored true) is given as before', async () => {
    backend.result = { ...RESULT, pc_stored: true };
    backend.patterns = five();
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await findButton());
    await waitResult();
    expect(screen.queryByText(NOT_STORED, { exact: false })).toBeNull();
  }, 10000);

  it('a typed PC the store did not take says the same words', async () => {
    backend.updateAnswer = { pc_stored: false };
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    typeAndUse();
    expect((await screen.findByTestId('pc-not-stored', {}, { timeout: 4000 })).textContent).toBe(NOT_STORED);
    // not "in use": no grant line, and the badge does not claim it either
    expect(screen.queryByTestId('pc-typed-grant')).toBeNull();
    expect(screen.getByTestId('pc-typed-badge').textContent).toBe('Your numbers: not used yet');
  }, 10000);

  it('a typed PC the store took says nothing of it', async () => {
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    typeAndUse();
    await waitFor(() => expect(backend.calls.some((c) => c[0] === 'updatePC')).toBe(true), { timeout: 4000 });
    await screen.findByTestId('pc-typed-grant');           // the answer has been handled
    expect(screen.queryByTestId('pc-not-stored')).toBeNull();
  }, 10000);
});

// ---------------------------------------------------------------------------------------------------------------
// The page before the search (what the file needs, the phase, the patterns) and the Result panel after it
// ---------------------------------------------------------------------------------------------------------------

// nine patterns at other positions than NINE, for "Find PC again with other patterns" (k = 0, 1, ...)
const NEXT = (k) => Array.from({ length: 9 }, (_, i) => ({ row: 20 + 10 * k + i, col: i, quality: 0.4 }));
const flat = (list) => list.map((p) => p.row * 20 + p.col);   // 20 columns, as gridShape [90, 20]

// Opens the page with five patterns on the list, presses Find PC and waits for the result panel.
const runWith = async (result, props = {}) => {
  backend.result = result;
  backend.patterns = five();
  render(<PCRefinement {...props} />);
  await screen.findAllByTitle(/^Pattern at row/);
  fireEvent.click(await findButton());
  return screen.findByTestId('pc-result', {}, { timeout: 4000 });
};

describe('The page before the search: what the file needs, the phase, the patterns', () => {
  it('names the phase the search uses, one phase and several', async () => {
    render(<PCRefinement />);
    expect((await screen.findByTestId('pc-searches-with')).textContent).toBe('Searches with phase: Ni');
    cleanup();
    backend.phaseNames = ['Ni', 'Cu'];
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-searches-with').textContent).toBe('Searches with phases: Ni, Cu'));
  });

  it('names no phase while none is loaded', async () => {
    backend.phaseLoaded = false;
    render(<PCRefinement />);
    await ready();
    await new Promise((r) => setTimeout(r, 30));
    expect(screen.queryByTestId('pc-searches-with')).toBeNull();
  });

  it('lists the first two patterns of the list and says how many more there are', async () => {
    backend.patterns = patternsOf(5);
    render(<PCRefinement />);
    const used = await screen.findByTestId('pc-patterns-used');
    expect(used.textContent).toBe('Patterns used (row, column)(1, 1)(2, 3)and 3 more');
    cleanup();
    backend.patterns = patternsOf(2);
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-patterns-used').textContent).toBe('Patterns used (row, column)(1, 1)(2, 3)'));
  });

  it('has no list while there are no patterns', async () => {
    render(<PCRefinement />);
    await ready();
    expect(screen.queryByTestId('pc-patterns-used')).toBeNull();
  });

  it('has the wide Find PC button, Pick patterns, Take PC from… and Save as calibration…, and no comparison while no calibration is saved', async () => {
    const onNavigate = vi.fn();
    await runWith(RESULT, { onNavigate });
    expect(screen.getAllByRole('button', { name: 'Find PC' })).toHaveLength(1);
    expect(screen.getByRole('button', { name: 'Pick patterns' })).toBeTruthy();
    expect(screen.getAllByRole('button', { name: 'Take PC from…' })).toHaveLength(1);   // the hint of a file without a PC has none here
    expect(screen.getAllByRole('button', { name: 'Save as calibration…' })).toHaveLength(1);   // beside the line that gives the PC
    expect(screen.queryByRole('button', { name: /Compare with a saved calibration/ })).toBeNull();
  }, 10000);
});

describe('The head names where the PC in use came from', () => {
  it.each([
    ['file', null, '(from the file)'],
    ['find', null, '(from Find PC)'],
    ['refine', null, '(from the refine)'],
    ['typed', null, '(typed)'],
    ['grid', null, '(from the grid calibration)'],
    ['calibration', { id: 'c1', name: 'SEM 2026-10-14', version: 2 }, '(from SEM 2026-10-14, version 2)'],
  ])('kind %s', async (kind, calibration, words) => {
    backend.kind = kind;
    backend.calibration = calibration;
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe(`Open file: scan_a · PC in use: 0.5019, 0.3301, 0.8420 (Bruker) ${words}`));
  });

  it('names no origin for a file without a PC, and none for a calibration the backend did not name', async () => {
    backend.source = 'missing';
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent).toBe('Open file: scan_a · PC in use: none'));
    cleanup();
    backend.source = 'header';
    backend.kind = 'calibration';
    backend.calibration = null;
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-file-head-pc').textContent).toBe('0.5019, 0.3301, 0.8420'));
    expect(screen.queryByTestId('pc-file-head-origin')).toBeNull();
  });

  it('a Global PC Refine is "from the refine" in the head, also before the backend has confirmed it', async () => {
    backend.result = { ...RESULT, mode: 'refine' };
    backend.patterns = [{ row: 1, col: 2 }];
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    let release;
    backend.statusGate = new Promise((resolve) => { release = resolve; });    // the status answer is held back
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5026, 0.3272, 0.8458 (Bruker) (from the refine)'), { timeout: 4000 });
    release();
    await waitFor(() => expect(screen.getByTestId('pc-file-head-origin').textContent).toBe(' (from the refine)'));
  }, 10000);

  it('a Find PC is "from Find PC" in the head before the backend has confirmed it', async () => {
    backend.patterns = five();
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    backend.statusGate = new Promise(() => {});                                // the status answer never comes
    fireEvent.click(await findButton());
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toBe('Open file: scan_a · PC in use: 0.5026, 0.3272, 0.8458 (Bruker) (from Find PC)'), { timeout: 4000 });
  }, 10000);

  it('a typed PC is "typed" once the backend says so', async () => {
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    backend.source = 'manual';
    typeAndUse();
    await waitFor(() => expect(screen.getByTestId('pc-file-head-origin').textContent).toBe(' (typed)'), { timeout: 4000 });
  }, 10000);
});

describe('The Result panel: a result that ended normally', () => {
  it('has the bands as the main button, first on the left, then Go to indexing and Save as calibration…, and nothing else', async () => {
    await runWith(RESULT, { onNavigate: vi.fn() });
    const grant = screen.getByTestId('pc-grant');
    expect(within(grant).getAllByRole('button').map((b) => b.textContent))
      .toEqual(['Show the bands over a pattern', 'Go to indexing', 'Save as calibration…', 'Put it back']);
  }, 10000);

  it('"Go to indexing" goes through the app\'s own page switch', async () => {
    const onNavigate = vi.fn();
    await runWith(RESULT, { onNavigate });
    expect(onNavigate).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Go to indexing' }));
    expect(onNavigate).toHaveBeenCalledTimes(1);
    expect(onNavigate).toHaveBeenCalledWith('indexing');
  }, 10000);

  it('has no Go to indexing where the page cannot switch pages', async () => {
    await runWith(RESULT);
    expect(screen.queryByRole('button', { name: 'Go to indexing' })).toBeNull();
  }, 10000);

  it('"Show the bands over a pattern" shows the pattern with the bands under the result and hides them again', async () => {
    await runWith(RESULT);
    const button = screen.getByRole('button', { name: 'Show the bands over a pattern' });
    fireEvent.click(button);
    expect(await screen.findByTestId('pc-bands-pane')).toBeTruthy();
    expect(screen.getByText('The red lines are the bands drawn from this PC. They should lie on the bands of the pattern.')).toBeTruthy();
    fireEvent.click(button);
    expect(screen.queryByTestId('pc-bands-pane')).toBeNull();
  }, 10000);

  it('says the search ended normally in one line and has no warning, no decision box', async () => {
    await runWith(RESULT);
    expect(screen.getByText('Search ended normally.').tagName).toBe('B');
    expect(screen.queryByTestId('pc-decide')).toBeNull();
    expect(screen.queryByTestId('pc-result-note')).toBeNull();
  }, 10000);
});

describe('The Result panel: the facts under the PC', () => {
  it('shows no fit contrast when the backend has none', async () => {
    await runWith({ ...RESULT, quality: { ...RESULT.quality, contrast: null } });
    expect(screen.queryByTestId('pc-contrast')).toBeNull();
  }, 10000);

  it('shows no "patterns disagree" line when the search kept no PC per pattern', async () => {
    await runWith({ ...RESULT, search: { ...RESULT.search, per_pattern: null } });
    expect(screen.queryByTestId('pc-spread')).toBeNull();
  }, 10000);

  it('one pattern gives no disagreement either', async () => {
    await runWith({ ...RESULT, search: { ...RESULT.search, per_pattern: [[0.5, 0.33, 0.85]] } });
    expect(screen.queryByTestId('pc-spread')).toBeNull();
  }, 10000);

  it('says "pixel" for one pixel, "pixels" for more, and "under 1 pixel" where the rounded value would be 0', async () => {
    const spreadOf = (dz) => ({ ...RESULT, search: { ...RESULT.search, per_pattern: [[0.5, 0.33, 0.85], [0.5, 0.33, 0.85 + dz]] } });
    // only z differs; z counts in the 128 rows of the pattern: 0.0078125 x 128 = 1 pixel, 0.016 x 128 = 2.05, 0.002 x 128 = 0.26
    await runWith(spreadOf(0.0078125));
    expect(screen.getByTestId('pc-spread').textContent).toBe('Patterns disagree by up to 0.01 in x, y or z, about 1 pixel');
    cleanup();
    await runWith(spreadOf(0.016));
    expect(screen.getByTestId('pc-spread').textContent).toBe('Patterns disagree by up to 0.02 in x, y or z, about 2 pixels');
    cleanup();
    await runWith(spreadOf(0.002));
    expect(screen.getByTestId('pc-spread').textContent).toBe('Patterns disagree by up to 0.00 in x, y or z, under 1 pixel');
  }, 25000);

  it('leaves the pixels out when the pattern size is not known', async () => {
    await runWith({ ...RESULT, pattern_size: undefined });
    expect(screen.getByTestId('pc-spread').textContent).toBe('Patterns disagree by up to 0.02 in x, y or z');
  }, 10000);

  it('counts the pixels of y and z in rows and those of x in columns', async () => {
    // only z differs (0.05): 0.05 x 128 rows = 6.4 pixels; with x it would be 0.05 x 156 = 7.8
    await runWith({ ...RESULT, search: { ...RESULT.search, per_pattern: [[0.5, 0.33, 0.80], [0.5, 0.33, 0.85]] } });
    expect(screen.getByTestId('pc-spread').textContent).toBe('Patterns disagree by up to 0.05 in x, y or z, about 6 pixels');
    cleanup();
    // only x differs (0.05): 0.05 x 156 columns = 7.8 pixels
    await runWith({ ...RESULT, search: { ...RESULT.search, per_pattern: [[0.45, 0.33, 0.85], [0.5, 0.33, 0.85]] } });
    expect(screen.getByTestId('pc-spread').textContent).toBe('Patterns disagree by up to 0.05 in x, y or z, about 8 pixels');
  }, 15000);

  it('writes the PC in the open file\'s own convention, asked of the backend (Oxford)', async () => {
    backend.convention = 'oxford';
    backend.conventionKnown = true;
    await runWith(RESULT);
    expect((await screen.findByTestId('pc-in-convention')).textContent)
      .toBe('As Oxford (AZtec): PCx 0.5026, PCy 0.5520, DD 0.6940');
    expect(backend.calls).toContainEqual(['convert', FOUND, 'bruker', 'oxford', [128, 156]]);
  }, 10000);

  it('...and for an EDAX/TSL file', async () => {
    backend.convention = 'tsl';
    backend.conventionKnown = true;
    await runWith(RESULT);
    expect((await screen.findByTestId('pc-in-convention')).textContent)
      .toBe('As EDAX/TSL (OIM): x* 0.5026, y* 0.6728, z* 0.8458');
  }, 10000);

  it('has no such line for a Bruker file, for a convention the format does not state, or when the conversion fails', async () => {
    backend.convention = 'bruker';
    backend.conventionKnown = true;
    await runWith(RESULT);
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByTestId('pc-in-convention')).toBeNull();
    cleanup();
    backend.convention = 'oxford';
    backend.conventionKnown = false;
    await runWith(RESULT);
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByTestId('pc-in-convention')).toBeNull();
    expect(backend.calls.some((c) => c[0] === 'convert')).toBe(false);
    cleanup();
    backend.conventionKnown = true;
    backend.convertError = new Error('boom');
    await runWith(RESULT);
    await waitFor(() => expect(backend.calls.some((c) => c[0] === 'convert')).toBe(true));
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByTestId('pc-in-convention')).toBeNull();
    expect(screen.getByTestId('pc-result-pc').textContent).toBe('0.5026, 0.3272, 0.8458');
  }, 20000);
});

describe('The Result panel: the one bold warning line', () => {
  it('names the axes of a result at the edge, and asks what to do instead of handing the PC over', async () => {
    const result = await runWith(ON_EDGE);
    expect(screen.getByTestId('pc-result-head').textContent).toBe('The result lies at the edge of the searched area (in y and z).');
    expect(screen.getByText('The result lies at the edge of the searched area (in y and z).').tagName).toBe('B');
    expect(screen.getByTestId('pc-result-note').textContent).toBe(
      'The PC may lie outside it, or the patterns may not fix it there. Orienta cannot judge the result itself, so check the bands once.');
    expect(screen.queryByText('Search ended normally.')).toBeNull();
    expect(screen.getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_a. Kept until you close Orienta. The file on disk is not changed.');
    expect(within(result).getByTestId('pc-decide')).toBeTruthy();
    expect(screen.getByTestId('pc-contrast').textContent).toBe('Fit contrast 0.4 of 1, higher is better');
  }, 10000);

  it('names one axis alone and three axes with "and"', async () => {
    await runWith({ ...ON_EDGE, quality: { ...ON_EDGE.quality, axes: ['z'] } });
    expect(screen.getByTestId('pc-result-head').textContent).toBe('The result lies at the edge of the searched area (in z).');
    cleanup();
    await runWith({ ...ON_EDGE, quality: { ...ON_EDGE.quality, axes: ['x', 'y', 'z'] } });
    expect(screen.getByTestId('pc-result-head').textContent).toBe('The result lies at the edge of the searched area (in x, y, and z).');
  }, 15000);

  it('words a result that did not end at a lowest point, and asks the same question', async () => {
    await runWith(NOT_MIN);
    expect(screen.getByTestId('pc-result-head').textContent).toBe('The search did not end at a lowest point.');
    expect(screen.getByTestId('pc-result-note').textContent).toBe(
      'A PC next to this result fits the patterns better. Orienta cannot judge the result itself, so check the bands once.');
    expect(screen.getByTestId('pc-decide')).toBeTruthy();
  }, 10000);

  it('words two about equally good positions with the grey sentence, and has no decision box and no button for the other position', async () => {
    await runWith(TWO);
    expect(screen.getByText('Two about equally good positions.').tagName).toBe('B');
    expect(screen.getByTestId('pc-result-note').textContent).toBe('These patterns may not fix the PC well. Check the bands.');
    expect(screen.queryByTestId('pc-decide')).toBeNull();
    expect(screen.queryByRole('button', { name: /other position/i })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Keep this PC anyway' })).toBeNull();
    expect(screen.getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_a. Kept until you close Orienta. The file on disk is not changed.');
    // the main button is "Find PC again", the bands come beside it; the grey line has no buttons of its own
    const result = screen.getByTestId('pc-result');
    expect(within(result).getAllByRole('button').map((b) => b.textContent))
      .toEqual(['Find PC again with other patterns', 'Show the bands over a pattern', 'Put it back', 'Save as calibration…']);   // Save stands in Details
    expect(screen.getByTestId('pc-contrast').textContent).toBe('Fit contrast 0.1 of 1, higher is better');
  }, 10000);

  it('offers the step back after two equal positions when the file had a PC, and none when there is nothing to put back', async () => {
    await runWith(TWO);
    const undo = screen.getByTestId('pc-undo');
    expect(undo.textContent).toContain('Previous PC: 0.5019, 0.3301, 0.8420');
    fireEvent.click(within(undo).getByRole('button', { name: 'Put it back' }));
    fireEvent.click(within(undo).getByRole('button', { name: /^Yes, put/ }));
    await waitFor(() => expect(backend.calls).toContainEqual(['putBack', 'scan_a']));
    cleanup();
    await runWith({ ...TWO, previous: null });
    expect(screen.queryByTestId('pc-undo')).toBeNull();
  }, 20000);

  it('shows the bands of two about equally good positions where the result stands', async () => {
    await runWith(TWO);
    fireEvent.click(screen.getByRole('button', { name: 'Show the bands over a pattern' }));
    expect(await screen.findByTestId('pc-bands-pane')).toBeTruthy();
  }, 10000);

  it('says a PC the store did not take in the one bold line, without "ended normally"', async () => {
    await runWith({ ...RESULT, pc_stored: false, pc_warning: 'English fallback',
      pc_warning_codes: [{ code: 'pcNotStored', params: {} }] });
    expect(screen.getByTestId('pc-result-head').textContent)
      .toBe('This PC is on this page only: indexing will not use it. Load the file again.');
    expect(screen.queryByText('Search ended normally.')).toBeNull();
  }, 10000);

  it('says a PC the store did not take in the one bold line also when the backend sent no warning code', async () => {
    await runWith({ ...RESULT, pc_stored: false });
    expect(screen.getByTestId('pc-result-head').textContent)
      .toBe('This PC is on this page only: indexing will not use it. Load the file again.');
    expect(screen.queryByTestId('pc-grant')).toBeNull();
  }, 10000);

  it('says a warning it has no line for in the reader\'s words, never "ended normally"', async () => {
    await runWith({ ...RESULT, pc_warning: 'English fallback',
      pc_warning_codes: [{ code: 'pcDrift', params: { deviation: '0.082', refined: 'a', start: 'b' } }] });
    expect(screen.getByTestId('pc-result-head').textContent).toMatch(/^Refined PC moved 0\.082 from the starting PC/);
    expect(screen.queryByText('Search ended normally.')).toBeNull();
  }, 10000);
});

describe('The Result panel: warnings the bold line does not say', () => {
  const DRIFT = { code: 'pcDrift', params: { deviation: '0.082', refined: 'a', start: 'b' } };
  const PLACEHOLDER = { code: 'refineFromPlaceholder', params: {} };
  const extraLines = () => screen.queryAllByTestId('pc-result-warning').map((e) => e.textContent);
  const DRIFT_TEXT = /^Refined PC moved 0\.082 from the starting PC/;
  const PLACEHOLDER_TEXT = /^This file carries no pattern centre, so the refine started at the placeholder/;

  // Global PC Refine over the five patterns on the list
  const refineWith = async (result) => {
    backend.result = { ...result, mode: 'refine' };
    backend.patterns = five();
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    return screen.findByTestId('pc-result', {}, { timeout: 4000 });
  };

  it('a Global PC Refine that ends at the edge still says the drift and the placeholder, as grey lines under the bold one', async () => {
    await refineWith({ ...ON_EDGE, pc_warning_codes: [...ON_EDGE.pc_warning_codes, DRIFT, PLACEHOLDER] });
    expect(screen.getByTestId('pc-result-head').textContent).toBe('The result lies at the edge of the searched area (in y and z).');
    const lines = extraLines();
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatch(DRIFT_TEXT);
    expect(lines[1]).toMatch(PLACEHOLDER_TEXT);
    expect(screen.getByTestId('pc-decide')).toBeTruthy();
  }, 10000);

  it('says the edge only once: the code of the bold line gets no second line', async () => {
    await runWith(ON_EDGE);
    expect(extraLines()).toEqual([]);
  }, 10000);

  it('says two equal positions as a grey line under a result at the edge', async () => {
    await runWith({ ...ON_EDGE, pc_warning_codes: [...ON_EDGE.pc_warning_codes, { code: 'pcSecondPosition', params: {} }] });
    expect(extraLines()).toEqual(['Two about equally good positions. These patterns may not fix the PC well. Check the bands.']);
  }, 10000);

  it('does the same under a result without a lowest point and under two equal positions', async () => {
    await runWith({ ...NOT_MIN, pc_warning_codes: [...NOT_MIN.pc_warning_codes, DRIFT] });
    expect(extraLines()).toHaveLength(1);
    expect(extraLines()[0]).toMatch(DRIFT_TEXT);
    cleanup();
    await runWith({ ...TWO, pc_warning_codes: [...TWO.pc_warning_codes, PLACEHOLDER] });
    expect(extraLines()).toHaveLength(1);
    expect(extraLines()[0]).toMatch(PLACEHOLDER_TEXT);
  }, 20000);

  it('a PC the store did not take stays the bold line, and the edge is said under it', async () => {
    await runWith({ ...ON_EDGE, pc_stored: false, pc_warning_codes: [...ON_EDGE.pc_warning_codes, { code: 'pcNotStored', params: {} }] });
    expect(screen.getByTestId('pc-result-head').textContent)
      .toBe('This PC is on this page only: indexing will not use it. Load the file again.');
    const lines = extraLines();
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatch(/^The result lies at the edge of the area that was searched \(in y, z\)/);
  }, 10000);

  it('a PC the store did not take is not hidden by the lines under another bold line', async () => {
    // pc_stored true, so the bold line is the edge; a pcNotStored code alone would still be said, not dropped
    await runWith({ ...ON_EDGE, pc_warning_codes: [...ON_EDGE.pc_warning_codes, { code: 'pcNotStored', params: {} }] });
    expect(extraLines().some((l) => l.startsWith('This PC is on this page only'))).toBe(true);
  }, 10000);
});

describe('The Result panel: the decision box', () => {
  it('has Find PC again as the main button, "Keep this PC anyway" and the step back, in that order', async () => {
    const decide = await runWith(ON_EDGE).then(() => screen.getByTestId('pc-decide'));
    expect(within(decide).getAllByRole('button').map((b) => b.textContent))
      .toEqual(['Find PC again with other patterns', 'Keep this PC anyway', 'Put it back']);
    expect(within(decide).getByTestId('pc-bands-canvas')).toBeTruthy();
    expect(within(decide).getByText('What do you want to do?')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Show the bands over a pattern' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Index with this PC anyway' })).toBeNull();
  }, 10000);

  it('"Keep this PC anyway" starts nothing; it says the PC stays in use and then offers Go to indexing', async () => {
    const onNavigate = vi.fn();
    await runWith(ON_EDGE, { onNavigate });
    expect(screen.queryByRole('button', { name: 'Go to indexing' })).toBeNull();
    const before = backend.calls.length;
    fireEvent.click(screen.getByRole('button', { name: 'Keep this PC anyway' }));
    expect((await screen.findByTestId('pc-kept')).textContent).toBe(
      'This PC stays in use for scan_a. Kept until you close Orienta. To use it next time, save it as a calibration.Go to indexingSave as calibration…');
    expect(backend.calls.length).toBe(before);
    expect(onNavigate).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Go to indexing' }));
    expect(onNavigate).toHaveBeenCalledWith('indexing');
  }, 10000);

  it('"Keep this PC anyway" without a page switch says so and offers no Go to indexing', async () => {
    await runWith(ON_EDGE);
    fireEvent.click(screen.getByRole('button', { name: 'Keep this PC anyway' }));
    await screen.findByTestId('pc-kept');
    expect(screen.queryByRole('button', { name: 'Go to indexing' })).toBeNull();
  }, 10000);

  it('the Bands on/off switch really hides the bands', async () => {
    await runWith(ON_EDGE);
    const decide = screen.getByTestId('pc-decide');
    const canvas = within(decide).getByTestId('pc-bands-canvas');
    await waitFor(() => expect(canvas.__ctx.calls.filter((c) => c === 'lineTo').length).toBeGreaterThan(0));
    canvas.__ctx.calls.length = 0;
    fireEvent.click(within(decide).getByRole('checkbox', { name: /Bands on\/off/ }));
    await new Promise((r) => setTimeout(r, 30));
    expect(canvas.__ctx.calls.filter((c) => c === 'lineTo')).toHaveLength(0);
    fireEvent.click(within(decide).getByRole('checkbox', { name: /Bands on\/off/ }));
    await waitFor(() => expect(canvas.__ctx.calls.filter((c) => c === 'lineTo').length).toBeGreaterThan(0));
  }, 10000);

  it('"Remove this PC" for a file that had no PC asks, and puts the file back to none', async () => {
    await runWith({ ...ON_EDGE, previous: { pc: null, had_pc: false } });
    const decide = screen.getByTestId('pc-decide');
    fireEvent.click(within(decide).getByRole('button', { name: 'Remove this PC' }));
    expect(backend.calls.some((c) => c[0] === 'putBack')).toBe(false);          // asks first
    fireEvent.click(within(decide).getByRole('button', { name: 'Yes, remove this PC' }));
    await waitFor(() => expect(screen.getByTestId('pc-undo-done').textContent).toMatch(/^PC removed\./));
    expect(screen.queryByTestId('pc-decide')).toBeNull();
    expect(backend.calls).toContainEqual(['putBack', 'scan_a']);
  }, 10000);

  it('a file that had a PC gets "Put it back" with the PC it had, not a button that claims to remove', async () => {
    await runWith(ON_EDGE);
    const decide = screen.getByTestId('pc-decide');
    expect(within(decide).getByText('Previous PC: 0.5019, 0.3301, 0.8420')).toBeTruthy();
    expect(within(decide).queryByRole('button', { name: 'Remove this PC' })).toBeNull();
  }, 10000);

  it('"Find PC again with other patterns" leaves out the positions of the list, picks again and searches over the new ones', async () => {
    backend.pickedNext = [NEXT(0)];
    await runWith(ON_EDGE);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find PC again with other patterns' }).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: 'Find PC again with other patterns' }));
    await waitFor(() => expect(optimizeCalls()).toHaveLength(2), { timeout: 4000 });
    // the list was the five patterns (1,1) (2,3) (3,5) (4,7) (5,9): those are left out
    expect(backend.calls.filter((c) => c[0] === 'autoPick')).toEqual([['autoPick', true, [[1, 1], [2, 3], [3, 5], [4, 7], [5, 9]]]]);
    expect(optimizeCalls()[1][1]).toEqual(flat(NEXT(0)));
    expect(optimizeCalls()[1][4]).toBe('find');
  }, 15000);

  it('...and a second press leaves out the first list as well, so it does not come back to it', async () => {
    backend.pickedNext = [NEXT(0), NEXT(1)];
    await runWith(ON_EDGE);
    const again = () => screen.getByRole('button', { name: 'Find PC again with other patterns' });
    await waitFor(() => expect(again().disabled).toBe(false));
    fireEvent.click(again());
    await waitFor(() => expect(optimizeCalls()).toHaveLength(2), { timeout: 4000 });
    await waitFor(() => expect(again().disabled).toBe(false), { timeout: 4000 });
    fireEvent.click(again());
    await waitFor(() => expect(optimizeCalls()).toHaveLength(3), { timeout: 4000 });
    const picks = backend.calls.filter((c) => c[0] === 'autoPick');
    expect(picks[1][2]).toEqual([...five(), ...NEXT(0)].map((p) => [p.row, p.col]));
    expect(optimizeCalls()[2][1]).toEqual(flat(NEXT(1)));
  }, 25000);

  it('a new Find PC starts the count again: only the list at hand is left out', async () => {
    backend.pickedNext = [NEXT(0), NEXT(1)];
    await runWith(ON_EDGE);
    const again = () => screen.getByRole('button', { name: 'Find PC again with other patterns' });
    await waitFor(() => expect(again().disabled).toBe(false));
    fireEvent.click(again());
    await waitFor(() => expect(optimizeCalls()).toHaveLength(2), { timeout: 4000 });
    await waitFor(() => expect(again().disabled).toBe(false), { timeout: 4000 });
    fireEvent.click(screen.getByRole('button', { name: 'Find PC' }));        // plain: the nine on the list
    await waitFor(() => expect(optimizeCalls()).toHaveLength(3), { timeout: 4000 });
    await waitFor(() => expect(again().disabled).toBe(false), { timeout: 4000 });
    fireEvent.click(again());
    await waitFor(() => expect(optimizeCalls()).toHaveLength(4), { timeout: 4000 });
    const picks = backend.calls.filter((c) => c[0] === 'autoPick');
    expect(picks[1][2]).toEqual(NEXT(0).map((p) => [p.row, p.col]));
  }, 40000);

  it('says in words when no other pattern is left, and searches nothing', async () => {
    backend.excludeError = coded(400, 'noPatternLeft', 'No calibration pattern is left to pick');
    await runWith(ON_EDGE);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find PC again with other patterns' }).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: 'Find PC again with other patterns' }));
    expect(await screen.findByText(/No calibration pattern is left to pick: every sampled position/)).toBeTruthy();
    expect(optimizeCalls()).toHaveLength(1);
  }, 15000);

  it('says in words when the other patterns are fewer than the search needs', async () => {
    backend.excludeError = coded(400, 'tooFewPatterns', 'only 3 could be picked');
    await runWith(ON_EDGE);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find PC again with other patterns' }).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: 'Find PC again with other patterns' }));
    expect(await screen.findByText('Find PC needs at least 5 calibration patterns.')).toBeTruthy();
    expect(optimizeCalls()).toHaveLength(1);
  }, 15000);

  it('is off while a search or a pick runs', async () => {
    await runWith(ON_EDGE);
    backend.jobStatus = 'running';
    backend.pickedNext = [NEXT(0)];
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find PC again with other patterns' }).disabled).toBe(false));
    fireEvent.click(screen.getByRole('button', { name: 'Find PC again with other patterns' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Find PC again with other patterns' }).disabled).toBe(true));
    backend.jobStatus = 'cancelled';
  }, 15000);
});

describe('The Result panel: Details, the sentences that depend on the result', () => {
  const details = () => screen.getByTestId('pc-details');
  const sentence = (id) => within(details()).queryByTestId(id)?.textContent;

  it('always says how the fit contrast arises', async () => {
    await runWith(RESULT);
    expect(sentence('pc-explain-contrast')).toBe(
      'Fit contrast compares how well this PC fits the patterns with how well a typical PC of the wide search area fits them: 0 is no better, 1 is a perfect fit.');
  }, 10000);

  it('says a second position with both PCs\' costs when the search found one, and nothing when it did not', async () => {
    await runWith(TWO);
    expect(sentence('pc-second-position')).toBe(
      'A second position fits about as well: 0.3604, 0.4451, 0.7402, cost 8.92 (this result: cost 8.45, lower is better).');
    cleanup();
    await runWith(RESULT);
    expect(sentence('pc-second-position')).toBeUndefined();
  }, 15000);

  it('says the second position also under a warning at the edge', async () => {
    await runWith({ ...ON_EDGE, search: { ...ON_EDGE.search, second_position: OTHER },
      quality: { ...ON_EDGE.quality, second_position: OTHER, best_cost: 8.45 } });
    expect(screen.getByTestId('pc-decide')).toBeTruthy();
    expect(sentence('pc-second-position')).toMatch(/^A second position fits about as well: 0\.3604, 0\.4451, 0\.7402, cost 8\.92/);
  }, 10000);

  it('says the patterns are one from each block when the page picked them, with the real number of patterns', async () => {
    backend.result = { ...RESULT, quality: { ...RESULT.quality, n_patterns: 9 } };
    render(<PCRefinement />);
    fireEvent.click(await findButton());                          // no patterns on the list: the page picks nine
    await screen.findByTestId('pc-result', {}, { timeout: 4000 });
    expect(sentence('pc-explain-patterns')).toBe(
      'The 9 patterns are one from each of 3 × 3 blocks of the scan, the one with the best image quality in its block.');
  }, 10000);

  const again = () => screen.getByRole('button', { name: 'Find PC again with other patterns' });
  const findAgain = async (result) => {
    await runWith(result);
    await waitFor(() => expect(again().disabled).toBe(false));
    fireEvent.click(again());
    await waitFor(() => expect(optimizeCalls()).toHaveLength(2), { timeout: 4000 });
    await waitFor(() => expect(again().disabled).toBe(false), { timeout: 4000 });
  };

  it('after "other patterns" the blocks give the best one LEFT, not the best one', async () => {
    backend.pickedNext = [NEXT(0)];
    await findAgain(ON_EDGE);
    expect(sentence('pc-explain-patterns')).toBe(
      'The 9 patterns are one from each of 3 × 3 blocks of the scan, the best one left in its block.');
  }, 20000);

  it('with fewer than nine patterns it does not claim one from each block: the best in the blocks of a 3 × 3 division', async () => {
    backend.picked = NINE.slice(0, 7);
    backend.result = { ...RESULT, quality: { ...RESULT.quality, n_patterns: 7 } };
    render(<PCRefinement />);
    fireEvent.click(await findButton());
    await screen.findByTestId('pc-result', {}, { timeout: 4000 });
    expect(sentence('pc-explain-patterns')).toBe('The 7 patterns are the best in the blocks of a 3 × 3 division of the scan.');
  }, 10000);

  it('...and after "other patterns" it says the best LEFT in them', async () => {
    backend.pickedNext = [NEXT(0).slice(0, 7)];
    await findAgain({ ...ON_EDGE, quality: { ...ON_EDGE.quality, n_patterns: 7 } });
    expect(sentence('pc-explain-patterns')).toBe('The 7 patterns are the best left in the blocks of a 3 × 3 division of the scan.');
  }, 20000);

  it('...but does not claim the best image quality where the pick could not measure it', async () => {
    backend.notes = ['blockWithoutQuality'];
    render(<PCRefinement />);
    fireEvent.click(await findButton());
    await screen.findByTestId('pc-result', {}, { timeout: 4000 });
    expect(sentence('pc-explain-patterns')).toBe(
      'The 9 patterns were picked one from each of 3 × 3 blocks of the scan. In some blocks the image quality could not be measured, so those have no pattern.');
    expect(sentence('pc-explain-patterns')).not.toMatch(/best image quality/);
  }, 10000);

  it('says "These are the patterns you chose." for a list the user made', async () => {
    await runWith(RESULT);                                          // five patterns on the list, no pick
    expect(sentence('pc-explain-patterns')).toBe('These are the patterns you chose.');
  }, 10000);

  it('...also for a list the page picked that was changed before the search', async () => {
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Pick patterns' }));
    await screen.findByText(/9 calibration patterns picked/);
    backend.patterns = backend.patterns.slice(0, 7);                // two patterns taken off the list
    window.dispatchEvent(new Event('pc-patterns-updated'));
    await waitFor(() => expect(screen.getByTestId('pc-patterns-used').textContent).toMatch(/and 5 more/));
    fireEvent.click(screen.getByRole('button', { name: 'Find PC' }));
    await screen.findByTestId('pc-result', {}, { timeout: 4000 });
    expect(sentence('pc-explain-patterns')).toBe('These are the patterns you chose.');
  }, 10000);

  it('...and "the page picked them" for a list from Pick patterns that is searched as it is', async () => {
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Pick patterns' }));
    await screen.findByText(/9 calibration patterns picked/);
    await waitFor(() => expect(screen.getByTestId('pc-patterns-used').textContent).toMatch(/and 7 more/));
    fireEvent.click(screen.getByRole('button', { name: 'Find PC' }));
    await screen.findByTestId('pc-result', {}, { timeout: 4000 });
    expect(backend.calls.filter((c) => c[0] === 'autoPick')).toHaveLength(1);       // Find PC did not pick again
    expect(sentence('pc-explain-patterns')).toMatch(/^The 9 patterns are one from each of 3 × 3 blocks/);
  }, 10000);

  it('says a scan of only a few patterns was used completely, not split into blocks', async () => {
    backend.nSampled = 9;
    render(<PCRefinement />);
    fireEvent.click(await findButton());
    await screen.findByTestId('pc-result', {}, { timeout: 4000 });
    expect(sentence('pc-explain-patterns')).toBe('The scan has only 9 patterns, and all of them were used.');
  }, 10000);

  it('says the PC is one value for the whole map when the file has no per-pixel map', async () => {
    await runWith(RESULT);
    expect(sentence('pc-explain-map')).toBe(
      'The PC is one value for the whole map: indexing uses it at every map point, and the 9 patterns only fix it.');
  }, 10000);

  it('says the result shifted every point of a per-pixel map when it did', async () => {
    backend.perPixel = true;
    await runWith({ ...RESULT, pc_map_action: 'shifted' });
    expect(sentence('pc-explain-map')).toBe(
      'This file has a PC for every map point. The result moved all of them by the same amount, so the map keeps its shape, '
      + 'and indexing uses the moved values. The 9 patterns only fix that shift.');
  }, 10000);

  it('says a map from a grid calibration was replaced when it was', async () => {
    backend.perPixel = true;
    await runWith({ ...RESULT, pc_map_action: 'replaced' });
    expect(sentence('pc-explain-map')).toBe(
      'A PC map from a grid calibration was replaced: the PC is now one value for the whole map.');
  }, 10000);

  it('says the search area was moved, or that the second search gave the result, or neither', async () => {
    await runWith(RESULT);
    expect(sentence('pc-box-moved')).toBeUndefined();
    expect(sentence('pc-far-box')).toBeUndefined();
    cleanup();
    await runWith({ ...RESULT, search: { ...RESULT.search, recentred: 2 } });
    expect(sentence('pc-box-moved')).toBe('The search area was moved to find this PC.');
    expect(sentence('pc-far-box')).toBeUndefined();
    cleanup();
    await runWith({ ...RESULT, search: { ...RESULT.search, recentred: 2, far_box_used: true } });
    expect(sentence('pc-far-box')).toMatch(/^A second search for a detector farther from the sample/);
    expect(sentence('pc-box-moved')).toBeUndefined();
  }, 20000);

  it('names the PC the file carries only when it carries one', async () => {
    await runWith({ ...RESULT, file_pc: null, file_pc_deviation: null });
    expect(sentence('pc-file-vs-result')).toBeUndefined();
  }, 10000);
});
