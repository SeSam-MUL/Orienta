// @vitest-environment jsdom
//
// The calibration offer and "Take PC from..." on the PC page, through the real page, the real i18n and a fake
// backend that refuses every method it was not told about. The rule under test: NOTHING CHANGES WITHOUT A BUTTON.
// A fitting calibration is offered ("Use it"); one whose settings differ or cannot be checked needs the second
// naming ("Use anyway": the request names the file again); "Not now" asks nothing of the backend.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within, act } from '@testing-library/react';

const backend = vi.hoisted(() => ({
  patterns: [], source: 'header', pc: [0.5019, 0.3301, 0.842], calls: [], unknown: [], dataset: 'scan_a',
  kind: null, calibration: null, suggestions: [], allList: [], suggestError: null, allError: null,
  applyError: null, applyAnswer: null, fromFile: null, fromFileError: null, updateAnswer: {}, statusCalls: 0,
  previousPc: null, putBackError: null, optimizeResult: null, statusGate: null, current: null, unverified: false,
}));

const refusal = (code = 'pc_job_running') => Object.assign(new Error('409'), {
  response: { status: 409, data: { detail: { code, message: 'A PC optimization is running.' } } } });
const coded = (status, code, detail = 'English sentence') => Object.assign(new Error(String(status)), {
  response: { status, data: { detail }, headers: { 'x-orienta-code': code } } });

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const strict = (target) => new Proxy(target, {
    get: (t, k) => {
      if (k in t || typeof k === 'symbol' || k === 'then') return t[k];
      return () => { backend.unknown.push(String(k)); throw new Error(`${String(k)} is not in this test's fake`); };
    },
  });
  const pcApi = strict({
    status: () => {
      backend.statusCalls += 1;
      // a test can hold the answer back to see what the page does before it knows its PC
      return (backend.statusGate || Promise.resolve()).then(() => ({ data: {
        has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'], n_patterns: 0,
        pc: backend.pc, pc_basis: 'single', pc_per_pixel: false, pc_map_shape: null, patterns: backend.patterns,
        pc_source: backend.source,
        pc_in_use: {
          pc: backend.source === 'missing' ? null : backend.pc, per_pixel: false,
          kind: backend.kind || ({ missing: 'none', header: 'file', manual: 'typed', calibration: 'calibration' }[backend.source] || 'file'),
          calibration: backend.calibration, convention: 'bruker', convention_known: false,
        },
      } }));
    },
    getPatternImage: () => ok({ image: 'AAAA' }),
    getPatternResult: () => ok({ success: true, segments: [], ci: 0.4 }),
    autoPick: async (replace) => {
      backend.calls.push(['autoPick', replace]);
      return { data: { success: true, n_patterns: 0, patterns: [], prepared: true, background: 'mean', n_sampled: 1, stride: 1, notes: [] } };
    },
    optimize: async (indices, method, limit, mode) => {
      backend.calls.push(['optimize', mode]);
      return { data: { task_id: 'task-1' } };
    },
    getOptimizeStatus: () => ok({ status: 'running', progress: 0.1 }),
    cancelOptimize: () => ok({}),
    previous: () => ok({ previous: backend.previousPc }),
    putBack: (dataset) => {
      backend.calls.push(['putBack', dataset]);
      if (backend.putBackError) return Promise.reject(backend.putBackError);
      backend.pc = backend.previousPc.pc || [0.5, 0.5, 0.5];
      backend.source = backend.previousPc.had_pc ? 'header' : 'missing';
      backend.kind = null; backend.calibration = null;
      return ok({ dataset, pc: backend.previousPc.pc, had_pc: backend.previousPc.had_pc, restored: [dataset], not_restored: [], pc_per_pixel: false });
    },
    updatePC: (...args) => {
      backend.calls.push(['updatePC', ...args.slice(0, 3)]);
      const previous = { pc: backend.source === 'missing' ? null : backend.pc, had_pc: backend.source !== 'missing' };
      if (backend.updateAnswer.pc_stored !== false) { backend.pc = args.slice(0, 3); backend.source = 'manual'; backend.kind = null; backend.calibration = null; }
      return ok({ success: true, ci: 0.2, segments: [], pc_map_action: 'none', pc_stored: true, previous, ...backend.updateAnswer });
    },
  });
  const pcCalibrationsApi = strict({
    suggest: (dataset, all = false) => {
      backend.calls.push(all ? ['suggest', dataset, 'all'] : ['suggest', dataset]);
      const error = all ? backend.allError : backend.suggestError;
      if (error) return Promise.reject(error);
      return ok({
        dataset, suggestions: all ? backend.allList : backend.suggestions, current: backend.current,
        unverified: all ? false : backend.unverified,
      });
    },
    apply: (id, datasets, accept = [], version = null) => {
      backend.calls.push(['apply', id, datasets, accept, version]);
      if (backend.applyError) return Promise.reject(backend.applyError);
      const answer = backend.applyAnswer || {
        dataset: datasets[0], status: 'applied', pc: [0.5026, 0.3272, 0.8458], version: version || 1,
        mismatch_accepted: accept.length > 0, pc_map_action: 'none', pc_map_actions: {},
        previous: { pc: backend.source === 'missing' ? null : backend.pc, had_pc: backend.source !== 'missing' },
      };
      if (answer.status === 'applied') {
        backend.previousPc = answer.previous;
        backend.pc = answer.pc; backend.source = 'calibration'; backend.kind = 'calibration';
        backend.calibration = { id, name: 'SEM 2026-10-14', version: answer.version };
        backend.suggestions = [];                                   // it is on that version now
      }
      return ok({ calibration: id, results: [answer] });
    },
    // the page reads these for the stored results of the calibration in use and for the decision box (no result is stored here)
    list: () => ok({ calibrations: [], problems: [] }),
    results: (id) => ok({ calibration: id, latest: 2, results: [] }),
    fromFile: (path, dataset) => {
      backend.calls.push(['fromFile', path, dataset]);
      if (backend.fromFileError) return Promise.reject(backend.fromFileError);
      return ok(backend.fromFile);
    },
  });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  const ebsdApi = new Proxy({ info: () => ok({ active_dataset: backend.dataset }) }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const calibrationApi = new Proxy({ getEntry: () => ok({ parent_name: null }) }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  return { ...orig, pcApi, pcCalibrationsApi, indexApi: quiet, ebsdApi, calibrationApi };
});

import PCRefinement from './PCRefinement';
import useDataStore from '../../stores/useDataStore';

const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});
class FakeImage {
  set src(_v) { this.width = 100; this.height = 100; queueMicrotask(() => this.onload && this.onload()); }
}

// the answer of the load the tests start from: a load has its own request id and a file path
const LOAD_1 = { request_id: 'load-1', file_path: '/d/scan_a.h5oina', grid_shape: [90, 20] };

// --- the three kinds of offer (fake numbers) ---
const SETTINGS = { pattern_shape: [128, 156], binning: 8, camera_tilt: 5, sample_tilt: 70, working_distance_mm: 20, detector_insertion_mm: 55 };
const ALL_CHECKED = ['pattern_shape', 'binning', 'camera_tilt', 'sample_tilt', 'working_distance_mm', 'detector_insertion_mm'];
const A = {
  id: 'sem', name: 'SEM 2026-10-14', version: 2, pc: [0.5026, 0.3272, 0.8458], via_member: false, member_version: null,
  match: true, differences: [], checked: ALL_CHECKED, unknown: [], unverified: false,
  settings: { file: SETTINGS, calibration: SETTINGS },
  version_created: '2026-10-20T09:00:00+00:00', source_name: 'scan_x.h5oina', file_date: '2026-10-21',
  quality: { contrast: 0.8, warning: null, axes: [], n_patterns: 9 },
  replaces: { pc: [0.5019, 0.3301, 0.842], kind: 'file' },
};
const B = {
  ...A, version: 1, pc: [0.5019, 0.3301, 0.842], via_member: true, member_version: 1, match: false,
  version_created: '2026-10-14T09:00:00+00:00', source_name: 'scan_a.h5oina',
  differences: [{ key: 'working_distance_mm', calibration: 20, file: 15, tolerance: 0.1 }],
  settings: { file: { ...SETTINGS, working_distance_mm: 15 }, calibration: SETTINGS },
  quality: { contrast: 0.4, warning: 'edge', axes: ['y', 'z'], n_patterns: 9 },
};
const C = {
  ...A, match: false, via_member: true, member_version: 2, unverified: true,
  checked: ['pattern_shape'], unknown: ['binning', 'camera_tilt', 'sample_tilt', 'working_distance_mm', 'detector_insertion_mm'],
  settings: { file: { pattern_shape: [128, 156], binning: null, camera_tilt: null, sample_tilt: null, working_distance_mm: null, detector_insertion_mm: null }, calibration: SETTINGS },
};

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = function getContext() { this.__ctx = this.__ctx || fakeContext(); return this.__ctx; };
  globalThis.Image = FakeImage;
  Object.assign(backend, {
    patterns: [], source: 'header', pc: [0.5019, 0.3301, 0.842], calls: [], unknown: [], dataset: 'scan_a', kind: null,
    calibration: null, suggestions: [], allList: [], suggestError: null, allError: null, applyError: null, applyAnswer: null,
    fromFile: null, fromFileError: null, updateAnswer: {}, statusCalls: 0, previousPc: null, putBackError: null,
    statusGate: null, current: null, unverified: false,
  });
  useDataStore.setState({ gridShape: [90, 20], ebsdInfo: LOAD_1 });
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
  delete window.electronAPI;
});
afterEach(() => {
  cleanup();
  delete window.electronAPI;
  // every call to a method the fakes do not know failed the test, also when a catch ate it
  expect(backend.unknown).toEqual([]);
});

const applyCalls = () => backend.calls.filter((c) => c[0] === 'apply');
const offerBox = () => screen.findByTestId('calibration-offer');
const text = (el) => el.textContent.replace(/\s+/g, ' ').trim();

describe('the offer on load', () => {
  it('A: shows the calibration that fits with its facts, folds the settings and applies nothing', async () => {
    backend.suggestions = [A];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(box.dataset.kind).toBe('fits');
    const t = text(box);
    expect(t).toContain('A saved calibration has the same settings as this file.');
    expect(t).toContain('SEM 2026-10-14fits');
    expect(t).toContain('Version 2, updated 2026-10-20 from scan_x.h5oina');
    expect(t).toContain('PC 0.5026, 0.3272, 0.8458');
    expect(t).toContain('Fit contrast 0.8 of 1, higher is better');
    expect(t).toContain('All 6 settings same');
    // hand calculation: (0.0007 * 156, -0.0029 * 128, 0.0038 * 128) = (0.1092, -0.3712, 0.4864), length 0.6215
    expect(t).toContain('Replaces the PC from the file. Change: about 0.6 pixels');
    expect(t).toContain('Version 2 is from 2026-10-20; this file was acquired 2026-10-21.');
    expect(box.querySelector('details').open).toBe(false);
    expect(within(box).getByText('Show the settings')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Use it' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Not now' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Use anyway' })).toBeNull();
    // one question to the backend when the page opened (once the PC in use is known), no change
    expect(backend.calls).toEqual([['suggest', 'scan_a']]);
    expect(applyCalls()).toEqual([]);
  });

  it('A: the folded table has the six rows, the values of both sides and the grey sentence about the instrument', async () => {
    backend.suggestions = [A];
    render(<PCRefinement />);
    const box = await offerBox();
    const rows = [...box.querySelectorAll('tbody tr')].map((r) => [...r.children].map((c) => c.textContent));
    expect(rows).toEqual([
      ['pattern size', '128 × 156', '128 × 156', 'same'], ['binning', '8', '8', 'same'],
      ['camera tilt', '5.0°', '5.0°', 'same'], ['sample tilt', '70.0°', '70.0°', 'same'],
      ['working distance', '20 mm', '20 mm', 'same'], ['detector insertion', '55 mm', '55 mm', 'same'],
    ]);
    expect(within(box).getByTestId('calibration-instrument-note').textContent)
      .toBe('The file does not say which microscope or detector was used. Check that this calibration comes from the same one.');
  });

  it('A: "Use it" applies the shown version to this file only, refreshes the page like a put back and shows the grant', async () => {
    backend.suggestions = [A];
    render(<PCRefinement />);
    await offerBox();
    const before = backend.statusCalls;
    fireEvent.click(screen.getByRole('button', { name: 'Use it' }));
    const grant = await screen.findByTestId('pc-grant');
    expect(applyCalls()).toEqual([['apply', 'sem', ['scan_a'], [], 2]]);   // version 2 as shown, no accept_mismatch
    expect(screen.getByTestId('calibration-in-use').textContent).toBe('The PC of the saved calibration is in use for this file.');
    expect(screen.getByTestId('calibration-in-use-from').textContent).toBe('SEM 2026-10-14, version 2, updated 2026-10-20 from scan_x.h5oina');
    expect(within(grant).getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_a: 0.5026, 0.3272, 0.8458. Kept until you close Orienta. The file on disk is not changed.');
    expect(within(grant).getByText('Previous PC: 0.5019, 0.3301, 0.8420')).toBeTruthy();
    expect(within(grant).getByRole('button', { name: 'Put it back' })).toBeTruthy();
    expect(within(grant).getByRole('button', { name: 'Show the bands over a pattern' })).toBeTruthy();
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
    // the page asked the backend for its PC again, and its head names the calibration
    await waitFor(() => expect(backend.statusCalls).toBeGreaterThan(before));
    await waitFor(() => expect(screen.getByTestId('pc-file-head-pc').textContent).toBe('0.5026, 0.3272, 0.8458'));
    expect(screen.getByTestId('pc-file-head-origin').textContent).toBe(' (from SEM 2026-10-14, version 2)');
    // and asked for offers again, which now has none: the grant stays
    await waitFor(() => expect(backend.calls.filter((c) => c[0] === 'suggest').length).toBeGreaterThan(1));
    expect(screen.getByTestId('pc-grant')).toBeTruthy();
  });

  it('A: "Put it back" after the use asks, puts the previous PC back and the page follows', async () => {
    backend.suggestions = [A];
    render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Use it' }));
    const grant = await screen.findByTestId('pc-grant');
    fireEvent.click(within(grant).getByRole('button', { name: 'Put it back' }));
    expect(applyCalls()).toHaveLength(1);
    expect(backend.calls.some((c) => c[0] === 'putBack')).toBe(false);        // asked first
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    expect(await screen.findByTestId('pc-undo-done')).toBeTruthy();
    expect(backend.calls.filter((c) => c[0] === 'putBack')).toEqual([['putBack', 'scan_a']]);
    expect(screen.getByTestId('pc-undo-done').textContent)
      .toBe('Previous PC put back for scan_a: 0.5019, 0.3301, 0.8420. Kept until you close Orienta. The file on disk is not changed.');
    await waitFor(() => expect(screen.getByTestId('pc-file-head-pc').textContent).toBe('0.5019, 0.3301, 0.8420'));
  });

  it('a file that had no PC gets "Remove this PC" after the use', async () => {
    backend.source = 'missing';
    backend.suggestions = [A];
    render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Use it' }));
    const grant = await screen.findByTestId('pc-grant');
    expect(within(grant).getByText('This file had no PC before.')).toBeTruthy();
    expect(within(grant).getByRole('button', { name: 'Remove this PC' })).toBeTruthy();
  });

  it('A: no "Replaces" line for a file without a PC, none for a kind it does not know, and the kind is named', async () => {
    backend.suggestions = [{ ...A, replaces: { pc: null, kind: 'none' } }];
    const first = render(<PCRefinement />);
    expect(text(await offerBox())).not.toContain('Replaces');
    first.unmount();
    backend.suggestions = [{ ...A, replaces: { pc: [0.5019, 0.3301, 0.842], kind: 'find' } }];
    render(<PCRefinement />);
    expect(text(await offerBox())).toContain('Replaces the PC from Find PC. Change: about 0.6 pixels');
  });

  it('"Not now" applies nothing, hides the offer and says when it comes back; a new load brings it back', async () => {
    backend.suggestions = [A];
    render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
    expect(screen.getByTestId('calibration-later').textContent).toBe('The offer comes back when you load this file again.');
    expect(applyCalls()).toEqual([]);
    useDataStore.setState({ ebsdInfo: { ...LOAD_1, request_id: 'load-2' } });  // the file is loaded again: a load of its own
    expect(await offerBox()).toBeTruthy();
    expect(screen.queryByTestId('calibration-later')).toBeNull();
  });

  it('"Not now" stays when only the grid of the same load is edited (ebsdInfo is replaced, nothing was loaded)', async () => {
    backend.suggestions = [A];
    render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    const asked = () => backend.calls.filter((c) => c[0] === 'suggest').length;
    const before = asked();
    // what a grid edit does: a new ebsdInfo object of the same load and the same file
    await act(async () => { useDataStore.setState({ ebsdInfo: { ...LOAD_1, grid_shape: [45, 40] } }); });
    await new Promise((r) => setTimeout(r, 200));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
    expect(screen.getByTestId('calibration-later').textContent).toBe('The offer comes back when you load this file again.');
    expect(asked()).toBe(before);                                               // and nothing was asked again either
  });

  it('the grant line of "Use it" stays when only the grid of the same load is edited', async () => {
    backend.suggestions = [A];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Use it' }));
    expect(await screen.findByTestId('pc-grant')).toBeTruthy();
    await act(async () => { useDataStore.setState({ ebsdInfo: { ...LOAD_1, grid_shape: [45, 40] } }); });
    await new Promise((r) => setTimeout(r, 200));
    expect(screen.getByTestId('calibration-applied')).toBeTruthy();
    expect(screen.getByTestId('pc-grant')).toBeTruthy();
  });

  it('a PC change on the page does not bring a put-off offer back', async () => {
    backend.suggestions = [A];
    render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    backend.pc = [0.51, 0.33, 0.84]; backend.source = 'manual';
    window.dispatchEvent(new CustomEvent('page-changed', { detail: { page: 'pcrefinement' } }));
    await waitFor(() => expect(screen.getByTestId('pc-file-head-pc').textContent).toBe('0.5100, 0.3300, 0.8400'));
    await waitFor(() => expect(backend.calls.filter((c) => c[0] === 'suggest').length).toBeGreaterThan(1));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
  });

  it('B: a file that used a calibration before, with one setting different, needs "Use anyway" and says what differs', async () => {
    backend.suggestions = [B];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(box.dataset.kind).toBe('differs');
    const t = text(box);
    expect(t).toContain('This file used a saved calibration before, but one setting differs.');
    expect(t).toContain('SEM 2026-10-14does not fit');
    expect(t).toContain('Version 1, saved 2026-10-14');
    expect(t).toContain('PC 0.5019, 0.3301, 0.8420');
    expect(t).toContain('Fit contrast 0.4 of 1, higher is better');
    expect(t).toContain('When it was saved, the result lay at the edge of the searched area.');
    expect(within(box).getByTestId('calibration-difference').textContent).toBe('Other working distance: 15 mm, calibration 20 mm.');
    const row = box.querySelector('tr[data-row="working_distance_mm"]');
    expect([...row.children].map((c) => c.textContent)).toEqual(['working distance', '15 mm', '20 mm', 'different']);
    expect(t).not.toContain('All 6 settings same');
    expect(t).not.toContain('Replaces the PC');
    expect(screen.queryByRole('button', { name: 'Use it' })).toBeNull();
    expect(within(box).getByRole('button', { name: 'Find PC' })).toBeTruthy();
    // "Use anyway" is the danger button, the second naming: the request names the file again
    fireEvent.click(screen.getByRole('button', { name: 'Use anyway' }));
    await screen.findByTestId('pc-grant');
    expect(applyCalls()).toEqual([['apply', 'sem', ['scan_a'], ['scan_a'], 1]]);
    // the offer that differs ends in the grant line alone
    expect(screen.queryByTestId('calibration-in-use')).toBeNull();
  });

  const SHAPE_DIFF = (calibrationShape) => ({
    ...B,
    differences: [{ key: 'pattern_shape', calibration: calibrationShape, file: [128, 156] }],
    settings: { file: SETTINGS, calibration: { ...SETTINGS, pattern_shape: calibrationShape } },
  });

  it('a pattern shape of another height-to-width ratio says a PC does not carry over', async () => {
    backend.suggestions = [SHAPE_DIFF([156, 128])];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(within(box).getByTestId('calibration-difference').textContent)
      .toBe('Other pattern shape: a PC does not carry over to another height-to-width ratio.');
  });

  it('the same ratio at another size (binning) does not say that, and names both sizes as before', async () => {
    backend.suggestions = [SHAPE_DIFF([64, 78])];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(within(box).getByTestId('calibration-difference').textContent)
      .toBe('Other pattern size: 128 × 156, calibration 64 × 78.');
    expect(text(box)).not.toContain('height-to-width');
  });

  it('B: two differences are counted in the sentence and listed one by one', async () => {
    backend.suggestions = [{ ...B, differences: [
      B.differences[0], { key: 'camera_tilt', calibration: 5, file: 4.32, tolerance: 0.1 }] }];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(text(box)).toContain('This file used a saved calibration before, but 2 settings differ.');
    expect(within(box).getAllByTestId('calibration-difference').map((d) => d.textContent))
      .toEqual(['Other working distance: 15 mm, calibration 20 mm.', 'Other camera tilt: 4.3°, calibration 5.0°.']);
  });

  it('B: Find PC starts the page\'s own Find PC', async () => {
    backend.suggestions = [B];
    render(<PCRefinement />);
    const box = await offerBox();
    fireEvent.click(within(box).getByRole('button', { name: 'Find PC' }));
    await waitFor(() => expect(backend.calls.some((c) => c[0] === 'autoPick')).toBe(true));
    expect(applyCalls()).toEqual([]);                                          // starting a search is not using the calibration
  });

  it('C: settings that are not stated: says so, shows them as not stated and needs "Use anyway"', async () => {
    backend.suggestions = [C];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(box.dataset.kind).toBe('unstated');
    const t = text(box);
    expect(t).toContain('Some settings of this file are not stated, so Orienta cannot tell whether the saved calibration fits.');
    expect(t).toContain('SEM 2026-10-14not stated');
    expect(t).toContain('Use it only if you know it fits.');
    const row = box.querySelector('tr[data-row="binning"]');
    expect([...row.children].map((c) => c.textContent)).toEqual(['binning', 'not stated', '8', 'not stated']);
    expect(box.querySelector('tr[data-row="pattern_shape"]').dataset.check).toBe('same');
    expect(within(box).getByTestId('calibration-instrument-note')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Use anyway' }));
    await screen.findByTestId('pc-grant');
    expect(applyCalls()).toEqual([['apply', 'sem', ['scan_a'], ['scan_a'], 2]]);
  });

  it.each([
    ['notAMinimum', 'When it was saved, the search did not end at a lowest point.'],
    ['secondPosition', 'When it was saved, two positions fitted about equally well.'],
  ])('a calibration saved with the warning %s says so', async (warning, sentence) => {
    backend.suggestions = [{ ...B, quality: { contrast: 0.1, warning, axes: [], n_patterns: 9 } }];
    render(<PCRefinement />);
    expect(within(await offerBox()).getByTestId('calibration-saved-warning').textContent).toBe(sentence);
  });

  it('a calibration saved without a quality shows no contrast and no warning', async () => {
    backend.suggestions = [{ ...A, quality: null }];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(within(box).queryByTestId('calibration-contrast')).toBeNull();
    expect(within(box).queryByTestId('calibration-saved-warning')).toBeNull();
  });

  it('a calibration applied to a PC map says what happened to the map', async () => {
    backend.suggestions = [A];
    backend.applyAnswer = {
      dataset: 'scan_a', status: 'applied', pc: [0.5026, 0.3272, 0.8458], version: 2, mismatch_accepted: false,
      pc_map_action: 'replaced', pc_map_actions: { scan_a: 'replaced' }, previous: { pc: [0.5019, 0.3301, 0.842], had_pc: true },
    };
    render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Use it' }));
    expect((await screen.findByTestId('calibration-map-note')).textContent)
      .toBe('The PC map of this file was replaced by this PC: one PC now applies to the whole scan.');
  });

  it('a PC map that was only moved says so, and "none" says nothing', async () => {
    backend.suggestions = [A];
    backend.applyAnswer = {
      dataset: 'scan_a', status: 'applied', pc: [0.5026, 0.3272, 0.8458], version: 2, mismatch_accepted: false,
      pc_map_action: 'shifted', pc_map_actions: { scan_a: 'shifted' }, previous: { pc: [0.5019, 0.3301, 0.842], had_pc: true },
    };
    const first = render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Use it' }));
    expect((await screen.findByTestId('calibration-map-note')).textContent)
      .toBe('The PC map of this file was moved so that its mean is this PC. Its shape is unchanged.');
    first.unmount();
    backend.applyAnswer = null;
    backend.suggestions = [A];
    render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Use it' }));
    await screen.findByTestId('pc-grant');
    expect(screen.queryByTestId('calibration-map-note')).toBeNull();
  });

  it('the backend refusing during a search is said in words and nothing is shown as used', async () => {
    backend.suggestions = [A];
    backend.applyError = refusal();
    render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Use it' }));
    expect((await screen.findByTestId('calibration-error')).textContent)
      .toBe('A PC refinement or grid calibration is running. Wait until it has finished, or cancel it, then try again.');
    expect(screen.queryByTestId('pc-grant')).toBeNull();
    expect(screen.getByTestId('calibration-offer')).toBeTruthy();              // still there: the person can try again
  });

  it('a file that changed meanwhile (mismatch) is told, not shown as used', async () => {
    backend.suggestions = [A];
    backend.applyAnswer = { dataset: 'scan_a', status: 'mismatch', unverified: false, differences: [] };
    render(<PCRefinement />);
    await offerBox();
    fireEvent.click(screen.getByRole('button', { name: 'Use it' }));
    expect((await screen.findByTestId('calibration-error')).textContent)
      .toBe("The settings of this file differ from the calibration's, so nothing was changed. Press Use anyway to use it.");
    expect(screen.queryByTestId('pc-grant')).toBeNull();
  });

  it('a refusal of the offer request is said; an unreachable backend is not made into a message', async () => {
    backend.suggestError = coded(404, 'unknownDataset', 'scan_a is not loaded');
    const first = render(<PCRefinement />);
    expect((await screen.findByTestId('calibration-error')).textContent).toBe('This file is not loaded.');
    first.unmount();
    backend.suggestError = new Error('Network Error');
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    await waitFor(() => expect(backend.calls.filter((c) => c[0] === 'suggest').length).toBe(2));
    expect(screen.queryByTestId('calibration-error')).toBeNull();
  });

  it('the offers are not asked for before the page knows its PC in use: what an offer replaces is that PC', async () => {
    backend.suggestions = [A];
    let release;
    backend.statusGate = new Promise((resolve) => { release = resolve; });
    render(<PCRefinement />);
    await waitFor(() => expect(backend.statusCalls).toBeGreaterThan(0));
    await new Promise((r) => setTimeout(r, 80));                 // the dataset is known long before the status
    expect(backend.calls.filter((c) => c[0] === 'suggest')).toEqual([]);
    release();
    const box = await offerBox();
    expect(text(box)).toContain('Change: about 0.6 pixels');
    expect(backend.calls.filter((c) => c[0] === 'suggest')).toEqual([['suggest', 'scan_a']]);
  });

  it('nothing is asked and nothing shown when there is no calibration', async () => {
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    expect(screen.queryByTestId('calibration-panel')).toBeNull();
    expect(applyCalls()).toEqual([]);
  });
});

describe('the "Take PC from…" buttons', () => {
  it('stand in the hint of a file without a PC and in the row of Pick patterns; with a PC only in the row', async () => {
    backend.source = 'missing';
    const first = render(<PCRefinement />);
    await screen.findByTestId('pc-missing-hint');
    expect(screen.getAllByRole('button', { name: 'Take PC from…' })).toHaveLength(2);
    expect(within(screen.getByTestId('pc-missing-hint')).getByRole('button', { name: 'Take PC from…' })).toBeTruthy();
    first.unmount();
    backend.source = 'header';
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    expect(screen.getAllByRole('button', { name: 'Take PC from…' })).toHaveLength(1);
  });

  it('each opens the window, and Close ends it without a request that changes anything', async () => {
    backend.source = 'missing';
    render(<PCRefinement />);
    const hint = await screen.findByTestId('pc-missing-hint');
    fireEvent.click(within(hint).getByRole('button', { name: 'Take PC from…' }));
    expect(await screen.findByRole('dialog', { name: 'Take the pattern centre from…' })).toBeTruthy();
    expect(screen.getByText('Nothing changes until you press a button below.')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(screen.queryByRole('dialog', { name: 'Take the pattern centre from…' })).toBeNull();
    fireEvent.click(screen.getAllByRole('button', { name: 'Take PC from…' })[1]);
    expect(await screen.findByRole('dialog', { name: 'Take the pattern centre from…' })).toBeTruthy();
    expect(applyCalls()).toEqual([]);
  });
});

const D_FIT = A;
const D_NO = {
  ...B, id: 'sem-old', name: 'SEM 2026-09-02', version: 1, pc: [0.5011, 0.3305, 0.8402], version_created: '2026-09-02T09:00:00+00:00',
  differences: [{ key: 'working_distance_mm', calibration: 25, file: 20, tolerance: 0.1 }],
  settings: { file: SETTINGS, calibration: { ...SETTINGS, working_distance_mm: 25 } },
};
const D_UNSTATED = { ...C, id: 'sem-new', name: 'REM 2026-10-30', version: 1, version_created: '2026-10-30T09:00:00+00:00', quality: null };

const openDialog = async () => {
  fireEvent.click(await screen.findByRole('button', { name: 'Take PC from…' }));
  return screen.findByTestId('take-pc-dialog');
};
const rowOf = (id) => document.querySelector(`[data-testid="take-calibration-row"][data-id="${id}"]`);

describe('Take PC from…: a saved calibration', () => {
  it('lists every calibration with its badge; only the one that fits has the main button', async () => {
    backend.allList = [D_FIT, D_NO, D_UNSTATED];
    render(<PCRefinement />);
    await openDialog();
    await waitFor(() => expect(document.querySelectorAll('[data-testid="take-calibration-row"]')).toHaveLength(3));
    expect(backend.calls).toContainEqual(['suggest', 'scan_a', 'all']);
    const fit = text(rowOf('sem'));
    expect(fit).toContain('SEM 2026-10-14fits');
    expect(fit).toContain('Version 2, updated 2026-10-20 from scan_x.h5oina');
    expect(fit).toContain('PC 0.5026, 0.3272, 0.8458');
    expect(fit).toContain('Fit contrast 0.8 of 1, higher is better');
    const no = text(rowOf('sem-old'));
    expect(no).toContain('SEM 2026-09-02does not fit');
    expect(no).toContain('Version 1, saved 2026-09-02');
    expect(no).toContain('When it was saved, the result lay at the edge of the searched area.');
    expect(no).toContain('Other working distance: this file 20 mm, calibration 25 mm.');
    expect(text(rowOf('sem-new'))).toContain('REM 2026-10-30not stated');
    expect(within(rowOf('sem')).getByRole('button', { name: 'Use it' })).toBeTruthy();
    expect(within(rowOf('sem-old')).queryByRole('button', { name: 'Use it' })).toBeNull();
    expect(within(rowOf('sem-old')).getByRole('button', { name: 'Use anyway' })).toBeTruthy();
    expect(within(rowOf('sem-new')).getByRole('button', { name: 'Use anyway' })).toBeTruthy();
    expect(applyCalls()).toEqual([]);
  });

  it('a calibration of another height-to-width ratio says so in its row; one of the same ratio does not', async () => {
    backend.allList = [
      { ...D_NO, differences: [{ key: 'pattern_shape', calibration: [156, 128], file: [128, 156] }] },
      { ...D_NO, id: 'sem-binned', differences: [{ key: 'pattern_shape', calibration: [64, 78], file: [128, 156] }] },
    ];
    render(<PCRefinement />);
    await openDialog();
    await waitFor(() => rowOf('sem-binned'));
    expect(text(rowOf('sem-old'))).toContain('Other pattern shape: a PC does not carry over to another height-to-width ratio.');
    expect(text(rowOf('sem-binned'))).toContain('Other pattern size: this file 128 × 156, calibration 64 × 78.');
    expect(text(rowOf('sem-binned'))).not.toContain('height-to-width');
  });

  it('"Use it" on the one that fits applies it plainly and shows the grant inside the window', async () => {
    backend.allList = [D_FIT, D_NO];
    render(<PCRefinement />);
    await openDialog();
    await waitFor(() => rowOf('sem'));
    fireEvent.click(within(rowOf('sem')).getByRole('button', { name: 'Use it' }));
    const grant = await screen.findByTestId('take-calibration-grant');
    expect(applyCalls()).toEqual([['apply', 'sem', ['scan_a'], [], 2]]);
    expect(within(grant).getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_a: 0.5026, 0.3272, 0.8458. Kept until you close Orienta. The file on disk is not changed.');
    expect(within(grant).getByText('Previous PC: 0.5019, 0.3301, 0.8420')).toBeTruthy();
    await waitFor(() => expect(screen.getByTestId('pc-file-head-origin').textContent).toBe(' (from SEM 2026-10-14, version 2)'));
  });

  it('"Use anyway" on one that does not fit names the file a second time', async () => {
    backend.allList = [D_FIT, D_NO];
    render(<PCRefinement />);
    await openDialog();
    await waitFor(() => rowOf('sem-old'));
    fireEvent.click(within(rowOf('sem-old')).getByRole('button', { name: 'Use anyway' }));
    await screen.findByTestId('take-calibration-grant');
    expect(applyCalls()).toEqual([['apply', 'sem-old', ['scan_a'], ['scan_a'], 1]]);
  });

  it('a refusal during a search is said in the window and nothing is shown as used', async () => {
    backend.allList = [D_FIT];
    backend.applyError = refusal();
    render(<PCRefinement />);
    await openDialog();
    await waitFor(() => rowOf('sem'));
    fireEvent.click(within(rowOf('sem')).getByRole('button', { name: 'Use it' }));
    expect((await screen.findByTestId('take-calibration-error')).textContent)
      .toBe('A PC refinement or grid calibration is running. Wait until it has finished, or cancel it, then try again.');
    expect(screen.queryByTestId('take-calibration-grant')).toBeNull();
  });

  it('an empty list says so', async () => {
    render(<PCRefinement />);
    await openDialog();
    expect((await screen.findByTestId('take-list-empty')).textContent).toBe('There are no saved calibrations yet.');
  });

  it('a refused list says so, with the backend\'s reason', async () => {
    backend.allError = coded(404, 'unknownDataset', 'x');
    render(<PCRefinement />);
    await openDialog();
    expect((await screen.findByTestId('take-list-error')).textContent)
      .toBe('The saved calibrations could not be read: This file is not loaded.');
  });
});

const ANG = {
  pc: [0.5, 0.4, 0.7], source_kind: 'ang', convention: 'tsl', stated: [0.5, 0.6, 0.7], shape: null, shape_assumed: true,
  per_pixel: false, notes: ['English sentence of the backend'], target_shape: [128, 156], shape_matches: true, current_pc: null,
  note_items: [{ code: 'assumedShape', params: { rows: 128, cols: 156, convention: 'tsl' }, text: 'English sentence of the backend' }],
  dataset: 'scan_a',
};
const typePath = (value) => fireEvent.change(screen.getByLabelText('Path of the file'), { target: { value } });

describe('Take PC from…: another file', () => {
  it('Read without a path asks for one and asks the backend nothing', async () => {
    render(<PCRefinement />);
    await openDialog();
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    expect((await screen.findByTestId('take-path-error')).textContent).toBe('Type a path or use Browse first.');
    expect(backend.calls.some((c) => c[0] === 'fromFile')).toBe(false);
  });

  it('Browse uses the app\'s file picker and puts the path in the field', async () => {
    const openFile = vi.fn(() => Promise.resolve('D:\\scans\\sample_c.ang'));
    window.electronAPI = { openFile };
    render(<PCRefinement />);
    await openDialog();
    fireEvent.click(screen.getByRole('button', { name: 'Browse…' }));
    await waitFor(() => expect(screen.getByLabelText('Path of the file').value).toBe('D:\\scans\\sample_c.ang'));
    expect(openFile).toHaveBeenCalledTimes(1);
    expect(openFile.mock.calls[0][0].filters[0].extensions).toEqual(['ang', 'osc', 'h5', 'h5oina', 'hdf5']);
    expect(backend.calls.some((c) => c[0] === 'fromFile')).toBe(false);        // choosing is not reading
  });

  it('without the desktop app, Browse asks for the path in the themed prompt', async () => {
    render(<PCRefinement />);
    await openDialog();
    fireEvent.click(screen.getByRole('button', { name: 'Browse…' }));
    const prompt = await screen.findByPlaceholderText('Path of the file, for example sample_c.ang');
    fireEvent.change(prompt, { target: { value: '/data/sample_c.ang' } });
    fireEvent.keyDown(prompt, { key: 'Enter' });
    await waitFor(() => expect(screen.getByLabelText('Path of the file').value).toBe('/data/sample_c.ang'));
  });

  it('Read shows both conventions and the assumption in words, and changes nothing', async () => {
    backend.fromFile = ANG;
    render(<PCRefinement />);
    await openDialog();
    typePath('D:\\scans\\sample_c.ang');
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    const proposal = await screen.findByTestId('take-proposal');
    expect(backend.calls.filter((c) => c[0] === 'fromFile')).toEqual([['fromFile', 'D:\\scans\\sample_c.ang', 'scan_a']]);
    expect(text(within(proposal).getByTestId('take-stated'))).toBe('In the .ang file, EDAX/TSL (OIM) numbers: x*, y*, z* = 0.5000, 0.6000, 0.7000');
    expect(text(within(proposal).getByTestId('take-in-orienta'))).toBe('In Orienta (Bruker): x, y, z = 0.5000, 0.4000, 0.7000');
    expect(within(proposal).getByTestId('take-note').textContent).toBe(
      'This .ang file does not say its pattern size. Orienta used the pattern size of scan_a (128 rows × 156 columns). '
      + 'This matters only if the patterns were taller than wide: then z would differ.');
    expect(backend.calls.some((c) => c[0] === 'updatePC')).toBe(false);
    expect(screen.getByTestId('pc-file-head-pc').textContent).toBe('0.5019, 0.3301, 0.8420');
  });

  it('a note with a code this build does not know shows the backend\'s own sentence', async () => {
    backend.fromFile = { ...ANG, note_items: [{ code: 'somethingNew', params: {}, text: 'A sentence of a newer backend.' }] };
    render(<PCRefinement />);
    await openDialog();
    typePath('x.ang');
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    expect((await screen.findByTestId('take-note')).textContent).toBe('A sentence of a newer backend.');
  });

  it('a backend without note_items still shows its notes', async () => {
    const { note_items: _drop, ...older } = ANG;
    backend.fromFile = older;
    render(<PCRefinement />);
    await openDialog();
    typePath('x.ang');
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    expect((await screen.findByTestId('take-note')).textContent).toBe('English sentence of the backend');
  });

  it('an Oxford file is shown in Oxford\'s labels', async () => {
    backend.fromFile = { ...ANG, source_kind: 'oxford_h5', convention: 'oxford', stated: [0.5, 0.55, 0.69], shape: [128, 156], shape_assumed: false, notes: [], note_items: [] };
    render(<PCRefinement />);
    await openDialog();
    typePath('x.h5oina');
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    const proposal = await screen.findByTestId('take-proposal');
    expect(text(within(proposal).getByTestId('take-stated'))).toBe('In the Oxford .h5oina file, Oxford (AZtec) numbers: PCx, PCy, DD = 0.5000, 0.5500, 0.6900');
    expect(within(proposal).queryByTestId('take-note')).toBeNull();
  });

  it('a file of another pattern size says so', async () => {
    backend.fromFile = { ...ANG, shape: [100, 100], shape_matches: false, shape_assumed: false, note_items: [] };
    render(<PCRefinement />);
    await openDialog();
    typePath('x.ang');
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    expect((await screen.findByTestId('take-shape-differs')).textContent)
      .toBe("This file states a pattern size of 100 × 100 (rows × columns); the open file's is 128 × 156.");
  });

  it('a file that holds no PC is refused in words', async () => {
    backend.fromFileError = coded(400, 'noPcInFile', 'the .ang header has no x-star');
    render(<PCRefinement />);
    await openDialog();
    typePath('x.ang');
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    expect((await screen.findByTestId('take-path-error')).textContent).toBe('This file holds no usable pattern centre.');
    expect(screen.queryByTestId('take-proposal')).toBeNull();
  });

  it('"Use this PC" gives the file the Bruker numbers, says where they come from and offers the step back', async () => {
    backend.fromFile = ANG;
    render(<PCRefinement />);
    await openDialog();
    typePath('D:\\scans\\sample_c.ang');
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    await screen.findByTestId('take-proposal');
    const before = backend.statusCalls;
    fireEvent.click(screen.getByRole('button', { name: 'Use this PC' }));
    const grant = await screen.findByTestId('take-file-grant');
    expect(backend.calls.filter((c) => c[0] === 'updatePC')).toEqual([['updatePC', 0.5, 0.4, 0.7]]);   // Bruker, not the stated 0.6
    expect(screen.getByTestId('take-taken-from').textContent).toBe('The numbers come from sample_c.ang.');
    expect(within(grant).getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_a: 0.5000, 0.4000, 0.7000. Kept until you close Orienta. The file on disk is not changed.');
    expect(within(grant).getByText('Previous PC: 0.5019, 0.3301, 0.8420')).toBeTruthy();
    await waitFor(() => expect(backend.statusCalls).toBeGreaterThan(before));
    // the step back is the page's put back
    backend.previousPc = { pc: [0.5019, 0.3301, 0.842], had_pc: true };
    fireEvent.click(within(grant).getByRole('button', { name: 'Put it back' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    await waitFor(() => expect(backend.calls.filter((c) => c[0] === 'putBack')).toEqual([['putBack', 'scan_a']]));
  });

  it('editing the path after reading takes the proposal away: it belongs to the path that was read', async () => {
    backend.fromFile = ANG;
    render(<PCRefinement />);
    await openDialog();
    typePath('a.ang');
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    await screen.findByTestId('take-proposal');
    typePath('b.ang');
    expect(screen.queryByTestId('take-proposal')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Use this PC' })).toBeNull();
  });

  it('a PC the store did not take is not shown as in use', async () => {
    backend.fromFile = ANG;
    backend.updateAnswer = { pc_stored: false };
    render(<PCRefinement />);
    await openDialog();
    typePath('a.ang');
    fireEvent.click(screen.getByRole('button', { name: 'Read' }));
    await screen.findByTestId('take-proposal');
    fireEvent.click(screen.getByRole('button', { name: 'Use this PC' }));
    expect((await screen.findByTestId('take-not-stored')).textContent)
      .toBe('This PC is on this page only: indexing will not use it. Load the file again.');
    expect(screen.queryByTestId('take-file-grant')).toBeNull();
  });
});

// A file that states none of its settings (an .up1, an .ebsp): the plain question answers `unverified: true` with
// no suggestion, so the page asks again for all calibrations and offers the ones of the same pattern size as C.
describe('a file that states no settings: the calibrations of its size, as example C', () => {
  const OTHER_SIZE = {
    ...C, id: 'wide', name: 'Wide detector', via_member: false, member_version: null,
    // as the backend answers it: unverified is true here too, as only the pattern size could be compared
    differences: [{ key: 'pattern_shape', calibration: [128, 156], file: [60, 60], tolerance: null }], unverified: true,
    checked: ['pattern_shape'], unknown: C.unknown,
  };
  const OLDER = { ...C, id: 'old', name: 'SEM 2026-09-01', version: 1, via_member: false, member_version: null, created: '2026-09-01T09:00:00+00:00' };
  const NEWER = { ...C, id: 'new', name: 'SEM 2026-10-14', version: 2, via_member: false, member_version: null, created: '2026-10-14T09:00:00+00:00' };
  const NO_SETTINGS = { file: { pattern_shape: [60, 60], binning: null, camera_tilt: null, sample_tilt: null, working_distance_mm: null, detector_insertion_mm: null } };
  const sameSize = (c) => ({ ...c, settings: { ...c.settings, ...NO_SETTINGS } });
  const asks = () => backend.calls.filter((c) => c[0] === 'suggest');

  it('one calibration of the same size: example C appears and "Use anyway" sends accept_mismatch', async () => {
    backend.unverified = true;
    backend.allList = [sameSize(NEWER)];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(asks()).toEqual([['suggest', 'scan_a'], ['suggest', 'scan_a', 'all']]);
    expect(box.dataset.kind).toBe('unstated');
    const t = text(box);
    expect(t).toContain('Some settings of this file are not stated, so Orienta cannot tell whether the saved calibration fits.');
    expect(t).toContain('SEM 2026-10-14not stated');
    expect(t).toContain('Version 2, updated 2026-10-20 from scan_x.h5oina');
    expect(t).toContain('PC 0.5026, 0.3272, 0.8458');
    expect(t).toContain('Fit contrast 0.8 of 1, higher is better');
    expect(box.querySelector('tr[data-row="binning"]').dataset.check).toBe('notStated');
    expect(within(box).getByTestId('calibration-instrument-note')).toBeTruthy();
    expect(t).toContain('Use it only if you know it fits.');
    const names = [...box.querySelectorAll('button')].map((b) => b.textContent);
    expect(names).toEqual(['Find PC', 'Use anyway', 'Not now']);
    expect(applyCalls()).toEqual([]);
    fireEvent.click(screen.getByRole('button', { name: 'Use anyway' }));
    await screen.findByTestId('pc-grant');
    expect(applyCalls()).toEqual([['apply', 'new', ['scan_a'], ['scan_a'], 2]]);
  });

  it('a saved warning is shown', async () => {
    backend.unverified = true;
    backend.allList = [sameSize({ ...NEWER, quality: { contrast: 0.1, warning: 'edge', axes: ['y'], n_patterns: 9 } })];
    render(<PCRefinement />);
    expect((await screen.findByTestId('calibration-saved-warning')).textContent.length).toBeGreaterThan(0);
  });

  it('several of the size: the newest one is offered, as the first of the answer, and no other', async () => {
    backend.unverified = true;
    backend.allList = [sameSize(NEWER), sameSize(OLDER)];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(text(box)).toContain('SEM 2026-10-14not stated');
    expect(screen.getAllByTestId('calibration-offer')).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: 'Use anyway' }));
    await screen.findByTestId('pc-grant');
    expect(applyCalls()[0].slice(0, 2)).toEqual(['apply', 'new']);
  });

  it('a calibration that differs in size is skipped for the next one of the size', async () => {
    backend.unverified = true;
    backend.allList = [OTHER_SIZE, sameSize(NEWER)];
    render(<PCRefinement />);
    const box = await offerBox();
    expect(text(box)).toContain('SEM 2026-10-14not stated');
    expect(text(box)).not.toContain('Wide detector');
  });

  it('only calibrations of another size: nothing is offered', async () => {
    backend.unverified = true;
    backend.allList = [OTHER_SIZE];
    render(<PCRefinement />);
    await waitFor(() => expect(asks()).toHaveLength(2));
    await waitFor(() => expect(backend.statusCalls).toBeGreaterThan(0));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
    expect(screen.queryByTestId('calibration-panel')).toBeNull();
  });

  it('an entry that was compared by more than the size is not an "unstated" offer', async () => {
    backend.unverified = true;
    backend.allList = [A];
    render(<PCRefinement />);
    await waitFor(() => expect(asks()).toHaveLength(2));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
  });

  it('no calibration at all: nothing is offered', async () => {
    backend.unverified = true;
    backend.allList = [];
    render(<PCRefinement />);
    await waitFor(() => expect(asks()).toHaveLength(2));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
    expect(screen.queryByTestId('calibration-panel')).toBeNull();
  });

  it('the file is already on that very version: nothing is offered', async () => {
    backend.unverified = true;
    backend.current = { id: 'new', name: 'SEM 2026-10-14', version: 2 };
    backend.allList = [sameSize(NEWER)];
    render(<PCRefinement />);
    await waitFor(() => expect(asks()).toHaveLength(2));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
  });

  it('a file on an older version is offered the newer one', async () => {
    backend.unverified = true;
    backend.current = { id: 'new', name: 'SEM 2026-10-14', version: 1 };
    backend.allList = [sameSize(NEWER)];
    render(<PCRefinement />);
    expect((await offerBox()).dataset.kind).toBe('unstated');
  });

  it('a file that is not unverified never asks for the whole list', async () => {
    backend.suggestions = [];
    backend.allList = [sameSize(NEWER)];
    render(<PCRefinement />);
    await waitFor(() => expect(backend.statusCalls).toBeGreaterThan(0));
    await waitFor(() => expect(asks()).toHaveLength(1));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
  });

  it('"Not now" hides the offer and asks the backend nothing more', async () => {
    backend.unverified = true;
    backend.allList = [sameSize(NEWER)];
    render(<PCRefinement />);
    await offerBox();
    const before = backend.calls.length;
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    await screen.findByTestId('calibration-later');
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
    expect(backend.calls.length).toBe(before);
    expect(applyCalls()).toEqual([]);
  });

  it('the second question failing shows the backend\'s refusal, not an offer', async () => {
    backend.unverified = true;
    backend.allError = coded(500, 'calibration_unreadable', 'broken');
    render(<PCRefinement />);
    await screen.findByTestId('calibration-error');
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
  });
});
