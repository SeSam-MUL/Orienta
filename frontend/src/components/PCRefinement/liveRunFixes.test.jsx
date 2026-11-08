// @vitest-environment jsdom
//
// The PC page, through the real page, the real i18n and a fake backend that
// refuses every method it was not told about:
//   1  a file that runs on a calibration says so and offers "Apply to other files…" on every visit
//   2  "Show the bands over a pattern" is off, with the words "Load a phase first", while no phase is loaded
//   3  the status bar at the bottom follows every PC given to the file or put back
//   4  the question after a search: singular sentence, its sentences beside their buttons, no "-0.0000"
//   5  "Only this file" ends the offer of that calibration for this file until the file is loaded again
//   6  one main button while a result is shown
//   7  the sentence about the file's own PC, in all four languages
//   8  the grant line of a file with a per-pixel map says the numbers are the mean of its map
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within, act } from '@testing-library/react';

const backend = vi.hoisted(() => ({
  dataset: 'scan_a', filePath: '/d/scan_a.h5oina', files: {}, calls: [], unknown: [], saved: [], cands: [],
  applyAnswers: {}, result: null, phaseLoaded: true, patterns: [], suggestions: [], calInfo: null, versionAnswer: null,
  updateAnswer: {},
}));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const strict = (target, label) => new Proxy(target, {
    get: (t, k) => {
      if (k in t || typeof k === 'symbol' || k === 'then') return t[k];
      return () => { backend.unknown.push(`${label}.${String(k)}`); throw new Error(`${label}.${String(k)} is not in this test's fake`); };
    },
  });
  const cur = () => backend.files[backend.dataset];
  const pcApi = strict({
    status: () => {
      const f = cur();
      return ok({
        has_detector: true, has_phase: backend.phaseLoaded, phase_name: 'Ni', phase_names: ['Ni'], n_patterns: backend.patterns.length,
        pc: f.pc, pc_basis: f.perPixel ? 'map' : 'single', pc_per_pixel: !!f.perPixel, pc_map_shape: f.perPixel ? [2, 2] : null,
        patterns: backend.patterns.map((p, i) => ({ index: i, ...p })), pc_source: f.source,
        pc_in_use: {
          pc: f.source === 'missing' ? null : f.pc, per_pixel: !!f.perPixel, kind: f.kind || 'file', calibration: f.calibration || null,
          convention: 'bruker', convention_known: false,
        },
      });
    },
    getPatternImage: () => ok({ image: 'AAAA' }),
    getPatternResult: () => ok({ success: true, segments: [], ci: 0.4 }),
    autoPick: async () => ({ data: { success: true, patterns: [], n_patterns: 0, notes: [] } }),
    optimize: async (indices, method, limit, mode) => { backend.calls.push(['optimize', backend.dataset, mode]); return { data: { task_id: 'task-1' } }; },
    cancelOptimize: () => ok({}),
    getOptimizeStatus: () => {
      const f = cur();
      const r = { ...backend.result, dataset: backend.dataset };
      f.before = { pc: f.pc, had_pc: f.source !== 'missing' };
      f.pc = r.mean_pc; f.source = 'refined'; f.kind = r.mode === 'refine' ? 'refine' : 'find'; f.calibration = null;
      return ok({ status: 'completed', progress: 1, result: r });
    },
    previous: () => ok({ previous: null }),
    putBack: (dataset) => {
      backend.calls.push(['putBack', dataset]);
      const f = backend.files[dataset];
      f.pc = f.before.pc; f.source = f.before.had_pc ? 'header' : 'missing'; f.kind = 'file'; f.calibration = null;
      return ok({ dataset, pc: f.before.pc, had_pc: f.before.had_pc, restored: [dataset], not_restored: [], pc_per_pixel: false });
    },
    updatePC: (...args) => {
      backend.calls.push(['updatePC', ...args.slice(0, 3)]);
      const f = cur();
      const previous = { pc: f.pc, had_pc: f.source !== 'missing' };
      f.before = previous; f.pc = args.slice(0, 3); f.source = 'manual'; f.kind = 'typed'; f.calibration = null;
      return ok({ success: true, ci: 0.2, segments: [], pc_map_action: 'none', pc_stored: true, previous, ...backend.updateAnswer });
    },
  }, 'pcApi');
  const pcCalibrationsApi = strict({
    suggest: (dataset) => ok({ dataset, suggestions: backend.suggestions, current: null, unverified: false }),
    list: () => ok({ calibrations: backend.saved, problems: [] }),
    get: (id) => ok(backend.calInfo),
    candidates: () => ok({ candidates: backend.cands.map((c) => ({ ...c })) }),
    apply: (id, datasets, accept = [], version = null) => {
      backend.calls.push(['apply', id, datasets, accept, version]);
      datasets.forEach((d) => {
        const f = backend.files[d];
        if (f && !backend.applyAnswers[d]) {
          f.before = { pc: f.pc, had_pc: true }; f.pc = [0.5026, 0.3272, 0.8458]; f.source = 'calibration'; f.kind = 'calibration';
          f.calibration = { id, name: 'SEM 2026-10-14', version };
        }
      });
      return ok({ calibration: id, results: datasets.map((d) => backend.applyAnswers[d] || { dataset: d, status: 'applied', pc: [0.5026, 0.3272, 0.8458], version,
        mismatch_accepted: false, pc_map_action: 'none', pc_map_actions: {}, previous: { pc: [0.5019, 0.3301, 0.842], had_pc: true } }) });
    },
    newVersion: () => ok(backend.versionAnswer),
    results: (id) => ok({ calibration: id, latest: 2, results: [] }),
  }, 'pcCalibrationsApi');
  const ebsdApi = new Proxy({
    info: () => ok({ loaded: true, active_dataset: backend.dataset, file_path: backend.filePath, data_shape: [90, 20, 128, 156] }),
    // the metadata the status bar's detector comes from: the PC the store holds for the open file now
    getMetadata: () => { backend.calls.push(['getMetadata']); return ok({ detector: { has_detector: true, pc: [...cur().pc] }, step_size: null }); },
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  const calibrationApi = new Proxy({ getEntry: () => ok({ parent_name: null }) }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  return { ...orig, pcApi, pcCalibrationsApi, indexApi: quiet, ebsdApi, calibrationApi };
});

import PCRefinement from './PCRefinement';
import StatusBar from '../shared/StatusBar';
import useDataStore from '../../stores/useDataStore';
import useApplyFlow from './Calibration/applyFlow';
import i18n from '../../i18n';

class FakeImage { set src(_v) { this.width = 100; this.height = 100; queueMicrotask(() => this.onload && this.onload()); } }
const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});

const FOUND = [0.5026, 0.3272, 0.8458];
const OLD = [0.5019, 0.3301, 0.842];
const FIVE = Array.from({ length: 5 }, (_, i) => ({ row: i + 1, col: 2 * i + 1 }));
const SETTINGS = { pattern_shape: [128, 156], binning: 8, camera_tilt: 5, sample_tilt: 70, working_distance_mm: 20, detector_insertion_mm: 55 };
const ALL6 = ['pattern_shape', 'binning', 'camera_tilt', 'sample_tilt', 'working_distance_mm', 'detector_insertion_mm'];
const cand = (dataset, o = {}) => ({
  dataset, has_pc: true, pc: OLD, ref: null, settings: { file: SETTINGS }, match: true, unverified: false, differences: [],
  unknown: [], checked: ALL6, on_this_version: false, on_calibration: false, path: `/d/${dataset}.h5oina`, ...o,
});
// the offer that fits (fake numbers), version 2 of "SEM 2026-10-14"
const OFFER = {
  id: 'sem', name: 'SEM 2026-10-14', version: 2, pc: FOUND, via_member: false, member_version: null,
  match: true, differences: [], checked: ALL6, unknown: [], unverified: false,
  settings: { file: SETTINGS, calibration: SETTINGS },
  version_created: '2026-10-20T09:00:00+00:00', source_name: 'scan_x.h5oina', file_date: '2026-10-21',
  quality: { contrast: 0.8, warning: null, axes: [], n_patterns: 9 },
  replaces: { pc: OLD, kind: 'file' },
};
const RESULT = {
  mean_pc: FOUND, pc_values: [[0.5, 0.33, 0.85]], ci: 0.4, global_ci: 0.4, segments: [], pattern_results: [], pc_map_action: 'none',
  dataset_changed: false, mode: 'find', file_pc: null, pattern_size: [128, 156],
  search: { mode: 'find', recentred: 0, far_box_used: false, on_edge: [], second_position: null,
    confidence: { state: 'notJudged', reasons: [], contrast: 0.8 },
    per_pattern: [[0.50, 0.33, 0.85], [0.52, 0.34, 0.865], [0.51, 0.335, 0.855]] },
  quality: { contrast: 0.8, warning: null, axes: [], n_patterns: 9, contrast_before: null },
  previous: { pc: OLD, had_pc: true }, pc_warning: null, pc_warning_codes: [], calibration: { calibration: null, diverged: false, followers: [] },
};
const FOLLOWERS = (names) => names.map((dataset) => ({ dataset, per_pixel: false, map_kind: null }));
const refineWith = (names, mean = FOUND) => ({
  ...RESULT, mean_pc: mean, mode: 'refine', search: { ...RESULT.search, mode: 'refine' },
  quality: { contrast: 0.8, warning: null, axes: [], n_patterns: 9, contrast_before: 0.7 },
  calibration: { calibration: { id: 'sem', name: 'SEM 2026-10-14', version: 1, latest: 1 }, diverged: true, followers: FOLLOWERS(names) },
});
const OTHER = { pc: [0.6, 0.085, 0.877], cost: 8.9 };
const SECOND = {
  ...RESULT, search: { ...RESULT.search, second_position: OTHER },
  quality: { ...RESULT.quality, contrast: 0.1, warning: 'secondPosition', second_position: OTHER, best_cost: 8.45 },
};
const CAL_INFO = { id: 'sem', name: 'SEM 2026-10-14', version: 1, versions: [{ version: 1, pc: OLD }] };

const fileOf = (name, pc = OLD) => ({ path: `/d/${name}.h5oina`, pc: [...pc], source: 'header', kind: 'file', calibration: null, before: null });
const onCalibration = (version = 2) => ({
  ...fileOf('scan_a'), source: 'calibration', kind: 'calibration', calibration: { id: 'sem', name: 'SEM 2026-10-14', version },
});

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = function getContext() { this.__ctx = this.__ctx || fakeContext(); return this.__ctx; };
  globalThis.Image = FakeImage;
  Object.assign(backend, {
    dataset: 'scan_a', filePath: '/d/scan_a.h5oina', calls: [], unknown: [], saved: [], cands: [], applyAnswers: {}, result: RESULT,
    phaseLoaded: true, patterns: FIVE.map((p) => ({ ...p })), suggestions: [], calInfo: CAL_INFO, versionAnswer: null, updateAnswer: {},
    files: { scan_a: fileOf('scan_a'), scan_b: fileOf('scan_b') },
  });
  useApplyFlow.setState({ flow: null });
  // the status bar shows the detector the store holds; a load put the file's PC there
  useDataStore.setState({
    gridShape: [90, 20], isFileOpen: true, filePath: '/d/scan_a.h5oina', ebsdInfo: { request_id: 'load-1', file_path: '/d/scan_a.h5oina' },
    detector: { has_detector: true, pc: [...OLD] },
  });
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
  delete window.electronAPI;
});
afterEach(() => {
  cleanup();
  useApplyFlow.setState({ flow: null });
  expect(backend.unknown).toEqual([]);
});

const callsOf = (name) => backend.calls.filter((c) => c[0] === name);
const text = (el) => el.textContent.replace(/\s+/g, ' ').trim();
const runFind = async (result = RESULT) => {
  backend.result = result;
  const view = render(<><PCRefinement /><StatusBar /></>);
  await screen.findAllByTitle(/^Pattern at row/);
  fireEvent.click(await screen.findByRole('button', { name: 'Find PC' }));
  await screen.findByTestId('pc-result', {}, { timeout: 5000 });
  return view;
};
const barPc = () => {
  const m = document.body.textContent.match(/PC \((\d\.\d{3}), (\d\.\d{3}), (\d\.\d{3})\)/);
  return m ? m.slice(1).join(', ') : null;
};
// typing starts when the page knows the open file: the fields start again with it
const ready = async () => {
  const a = await screen.findByLabelText('PC X:');
  await waitFor(() => expect(screen.getByTestId('pc-file-head-file').textContent).toBe('scan_a'));
  return a;
};
const OLD_BAR = '0.502, 0.330, 0.842';
const FOUND_BAR = '0.503, 0.327, 0.846';
const refresh = () => act(async () => { window.dispatchEvent(new CustomEvent('page-changed', { detail: { page: 'pcrefinement' } })); });

describe('1: a file on a calibration says so and offers Apply to other files', () => {
  it('shows "This file is on <name>, version n." with the button on a page opened for such a file, and the window opens', async () => {
    backend.files.scan_a = onCalibration(2);
    backend.cands = [cand('scan_a', { on_this_version: true, on_calibration: true }), cand('scan_b')];
    render(<PCRefinement />);
    const line = await screen.findByTestId('calibration-on');
    expect(text(within(line).getByText(/^This file is on/))).toBe('This file is on SEM 2026-10-14, version 2.');
    fireEvent.click(within(line).getByRole('button', { name: 'Apply to other files…' }));
    const dialog = await screen.findByTestId('apply-dialog');
    expect(text(dialog)).toContain('Apply SEM 2026-10-14, version 2 (0.5019, 0.3301, 0.8420) to other files');
    expect(useApplyFlow.getState().flow.origin).toBe('scan_a');
  }, 15000);

  it('is there again when the page is opened anew for the same file (file switched away and back)', async () => {
    backend.files.scan_a = onCalibration(2);
    const first = render(<PCRefinement />);
    await screen.findByTestId('calibration-on');
    first.unmount();
    render(<PCRefinement />);
    expect(text(await screen.findByTestId('calibration-on'))).toContain('This file is on SEM 2026-10-14, version 2.');
  }, 15000);

  it('is not there for a file on no calibration', async () => {
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.queryByTestId('calibration-on')).toBeNull();
  }, 15000);

  it('right after "Use it" the grant has the one button; the line does not add a second one', async () => {
    backend.suggestions = [OFFER];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Use it' }));
    await screen.findByTestId('pc-grant');
    await waitFor(() => expect(screen.getByTestId('pc-file-head-origin').textContent).toBe(' (from SEM 2026-10-14, version 2)'));
    expect(screen.getAllByRole('button', { name: 'Apply to other files…' })).toHaveLength(1);
    expect(screen.queryByTestId('calibration-on')).toBeNull();
  }, 15000);
});

describe('2: no phase, no bands', () => {
  const useOffer = async () => {
    backend.suggestions = [OFFER];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Use it' }));
    return screen.findByTestId('pc-grant');
  };

  it('"Show the bands over a pattern" is off and the words say to load a phase; a click opens nothing', async () => {
    backend.phaseLoaded = false;
    const grant = await useOffer();
    const bands = within(grant).getByRole('button', { name: 'Show the bands over a pattern' });
    expect(bands.disabled).toBe(true);
    expect(within(grant).getByTestId('pc-bands-need-phase').textContent).toBe('Load a phase first: the bands are drawn from the loaded phase.');
    fireEvent.click(bands);
    expect(screen.queryByText(/The red lines are the bands/)).toBeNull();
  }, 15000);

  it('with a phase the button is on and opens the bands', async () => {
    const grant = await useOffer();
    const bands = within(grant).getByRole('button', { name: 'Show the bands over a pattern' });
    expect(bands.disabled).toBe(false);
    expect(within(grant).queryByTestId('pc-bands-need-phase')).toBeNull();
    fireEvent.click(bands);
    expect(await screen.findByText(/The red lines are the bands/)).toBeTruthy();
  }, 15000);

  it('the button of a result with two about equally good positions is off the same way', async () => {
    await runFind(SECOND);
    const res = screen.getByTestId('pc-result');
    expect(within(res).getByRole('button', { name: 'Show the bands over a pattern' }).disabled).toBe(false);
    backend.phaseLoaded = false;
    await refresh();
    await waitFor(() => expect(within(res).getByRole('button', { name: 'Show the bands over a pattern' }).disabled).toBe(true));
    expect(within(res).getByTestId('pc-bands-need-phase')).toBeTruthy();
  }, 20000);
});

describe('3: the status bar follows the PC given to the file', () => {
  it('"Use it" of a calibration', async () => {
    backend.suggestions = [OFFER];
    render(<><PCRefinement /><StatusBar /></>);
    await screen.findByRole('button', { name: 'Use it' });
    expect(barPc()).toBe(OLD_BAR);
    fireEvent.click(screen.getByRole('button', { name: 'Use it' }));
    await waitFor(() => expect(barPc()).toBe(FOUND_BAR));
    expect(callsOf('getMetadata')).toHaveLength(1);
  }, 15000);

  it('and putting it back', async () => {
    backend.suggestions = [OFFER];
    render(<><PCRefinement /><StatusBar /></>);
    fireEvent.click(await screen.findByRole('button', { name: 'Use it' }));
    const grant = await screen.findByTestId('pc-grant');
    await waitFor(() => expect(barPc()).toBe(FOUND_BAR));
    fireEvent.click(within(grant).getByRole('button', { name: 'Put it back' }));
    fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
    await screen.findByTestId('pc-undo-done');
    await waitFor(() => expect(barPc()).toBe(OLD_BAR));
  }, 15000);

  it('numbers typed in the fields', async () => {
    render(<><PCRefinement /><StatusBar /></>);
    fireEvent.change(await ready(), { target: { value: '0.51' } });
    fireEvent.change(screen.getByLabelText('PC Y:'), { target: { value: '0.34' } });
    fireEvent.change(screen.getByLabelText('PC Z:'), { target: { value: '0.84' } });
    expect(barPc()).toBe(OLD_BAR);                                   // typing alone changes nothing
    fireEvent.click(screen.getByRole('button', { name: 'Use these numbers' }));
    await waitFor(() => expect(barPc()).toBe('0.510, 0.340, 0.840'));
  }, 15000);

  it('a Find PC result', async () => {
    await runFind(RESULT);
    await waitFor(() => expect(barPc()).toBe(FOUND_BAR));
  }, 20000);

  it('asks for nothing while the bar already shows the PC in use', async () => {
    render(<><PCRefinement /><StatusBar /></>);
    await screen.findByTestId('pc-file-head-pc');
    await refresh();
    await new Promise((r) => setTimeout(r, 100));
    expect(callsOf('getMetadata')).toEqual([]);
    expect(barPc()).toBe(OLD_BAR);
  }, 15000);
});

describe('4: the question after a search (screen 7)', () => {
  beforeEach(() => {
    backend.cands = [cand('scan_a'), cand('scan_b'), cand('scan_c')];
  });

  it('says "moves" for one follower and "move" for two, in every language', async () => {
    await runFind(refineWith(['scan_b']));
    expect(text(await screen.findByTestId('question-new-version-note')))
      .toBe('Saves the new PC as version 2 of the calibration. scan_b moves to it.');
    cleanup();
    await runFind(refineWith(['scan_b', 'scan_c']));
    expect(text(await screen.findByTestId('question-new-version-note')))
      .toBe('Saves the new PC as version 2 of the calibration. scan_b and scan_c move to it.');
    const one = { count: 1, next: 2, files: 'scan_b' };
    const two = { count: 2, next: 2, files: 'scan_b und scan_c' };
    expect(i18n.getFixedT('de', 'pcrefinement')('question.newVersionNote', one))
      .toBe('Speichert das neue PC als Version 2 der Kalibrierung. scan_b wechselt darauf.');
    expect(i18n.getFixedT('de', 'pcrefinement')('question.newVersionNote', two))
      .toBe('Speichert das neue PC als Version 2 der Kalibrierung. scan_b und scan_c wechseln darauf.');
    for (const lng of ['ja', 'zh']) {
      const t = i18n.getFixedT(lng, 'pcrefinement');
      expect(t('question.newVersionNote', one)).toContain('scan_b');
      expect(t('question.newVersionNote', one)).not.toContain('question.');
      expect(t('question.newVersionNote', two)).toContain('2');
    }
  }, 30000);

  it('puts each sentence in one row with its button, the long one included', async () => {
    await runFind(refineWith(['scan_b', 'scan_c']));
    const q = await screen.findByTestId('follower-question');
    const rows = [
      ['Only this file', 'question-only-note'],
      ['New version for these 2 files too', 'question-new-version-note'],
      ['Save as a new calibration…', 'question-save-new-note'],
    ];
    for (const [name, noteId] of rows) {
      const note = within(q).getByTestId(noteId);
      const row = note.parentElement;
      expect(row.contains(within(q).getByRole('button', { name }))).toBe(true);
      expect(row.style.flexWrap).toBe('nowrap');                     // the sentence wraps inside the row, not under the button
      expect(note.style.flexGrow).toBe('1');
    }
  }, 20000);

  it('writes a change that rounds to zero without a sign', async () => {
    // the version holds [0.5019, 0.3301, 0.842]; x differs by -0.00001: four decimals show 0.0000
    await runFind(refineWith(['scan_b'], [0.50189, 0.3272, 0.8458]));
    await waitFor(() => expect(within(screen.getByTestId('follower-question')).getByTestId('question-change').textContent)
      .toMatch(/^Change: x 0\.0000, y -0\.0029, z \+0\.0038/));
  }, 20000);
});

describe('5: "Only this file" ends the offer of that calibration for this file', () => {
  it('no card of it any more, also when the page asks again; another calibration and a new load bring cards back', async () => {
    backend.suggestions = [OFFER];
    backend.cands = [cand('scan_a'), cand('scan_b')];
    await runFind(refineWith(['scan_b']));
    expect(await screen.findByTestId('calibration-offer')).toBeTruthy();          // the card is there before the answer
    const q = await screen.findByTestId('follower-question');
    fireEvent.click(within(q).getByRole('button', { name: 'Only this file' }));
    await screen.findByTestId('question-answered');
    await waitFor(() => expect(screen.queryByTestId('calibration-offer')).toBeNull());
    expect(screen.queryByTestId('calibration-later')).toBeNull();                 // no "comes back": the answer was no
    // the page asks again (the PC changed behind it): still no card
    backend.files.scan_a.pc = [0.51, 0.34, 0.84]; backend.files.scan_a.source = 'manual'; backend.files.scan_a.kind = 'typed';
    await refresh();
    await waitFor(() => expect(screen.getByTestId('pc-file-head-pc').textContent).toBe('0.5100, 0.3400, 0.8400'));
    await new Promise((r) => setTimeout(r, 150));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
    // another calibration is offered as before
    backend.suggestions = [{ ...OFFER, id: 'other', name: 'Other' }];
    backend.files.scan_a.pc = [0.52, 0.35, 0.85];
    await refresh();
    expect((await screen.findByTestId('calibration-offer')).textContent).toContain('Other');
    // and the calibration comes back when the file is loaded again
    backend.suggestions = [OFFER];
    backend.files.scan_a.pc = [0.53, 0.36, 0.86];
    await act(async () => { useDataStore.setState({ ebsdInfo: { request_id: 'load-2', file_path: '/d/scan_a.h5oina' } }); });
    await refresh();
    expect((await screen.findByTestId('calibration-offer')).textContent).toContain('SEM 2026-10-14');
  }, 30000);
});

describe('6: one main button while a result is shown', () => {
  const buttons = () => ({
    find: screen.getByRole('button', { name: 'Find PC' }),
    refine: screen.getByRole('button', { name: 'Global PC Refine' }),
    indexAll: screen.getByRole('button', { name: /^Index All/ }),
    plain: screen.getByRole('button', { name: 'Pick patterns' }),
  });

  it('Global PC Refine and Index All take the plain style when the result has its main button, and the colour back after a put back', async () => {
    render(<PCRefinement />);
    await screen.findAllByTitle(/^Pattern at row/);
    const before = buttons();
    expect(before.refine.style.background).toBe(before.find.style.background);     // the main colour
    expect(before.indexAll.style.background).not.toBe(before.plain.style.background);
    backend.result = RESULT;
    fireEvent.click(before.find);
    const res = await screen.findByTestId('pc-result', {}, { timeout: 5000 });
    const bandsButton = within(res).getByRole('button', { name: 'Show the bands over a pattern' });
    const after = buttons();
    expect(bandsButton.style.background).toBe(before.find.style.background);       // the result's main button
    expect(after.refine.style.background).toBe(after.plain.style.background);
    expect(after.indexAll.style.background).toBe(after.plain.style.background);
    expect(after.find.style.background).toBe(before.find.style.background);        // Find PC keeps its place on screen 1
    fireEvent.click(within(res).getByRole('button', { name: 'Put it back' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Yes, put it back' }));
    await screen.findByTestId('pc-undo-done');
    expect(buttons().refine.style.background).toBe(before.find.style.background);
    expect(buttons().indexAll.style.background).toBe(before.indexAll.style.background);
  }, 20000);
});

describe('7: the sentence about the PC the file carries', () => {
  it.each([
    ['en', 'This file carries its own PC, 0.5026, 0.3273, 0.8458. The largest difference in x, y or z is 0.0137.'],
    ['de', 'Diese Datei enthält ein eigenes PC, 0.5026, 0.3273, 0.8458. Die größte Abweichung in x, y oder z beträgt 0.0137.'],
    ['ja', 'このファイルには自身の PC、0.5026, 0.3273, 0.8458 があります。x、y、z の差の最大値は 0.0137 です。'],
    ['zh', '此文件带有自己的 PC，0.5026, 0.3273, 0.8458。x、y、z 中的最大差值为 0.0137。'],
  ])('%s', (lng, expected) => {
    expect(i18n.getFixedT(lng, 'pcrefinement')('result.fileHasPc', { pc: '0.5026, 0.3273, 0.8458', deviation: '0.0137' })).toBe(expected);
  });
});

describe('8: the grant line of a file with a per-pixel map', () => {
  const mapAnswer = (action) => ({
    dataset: 'scan_a', status: 'applied', pc: FOUND, version: 2, mismatch_accepted: false, pc_map_action: action,
    pc_map_actions: {}, previous: { pc: OLD, had_pc: true },
  });
  const grantText = async () => {
    backend.suggestions = [OFFER];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Use it' }));
    return (await screen.findByTestId('pc-grant-text')).textContent;
  };

  it('says the numbers are the mean of its map when the map was moved', async () => {
    backend.applyAnswers.scan_a = mapAnswer('shifted');
    expect(await grantText()).toBe(
      'PC in use for scan_a: 0.5026, 0.3272, 0.8458 (mean of its per-pixel map). Kept until you close Orienta. The file on disk is not changed.');
  }, 15000);

  it.each(['replaced', 'none'])('says nothing of a map when the answer is "%s" (one PC for the whole file now)', async (action) => {
    backend.applyAnswers.scan_a = mapAnswer(action);
    expect(await grantText()).toBe('PC in use for scan_a: 0.5026, 0.3272, 0.8458. Kept until you close Orienta. The file on disk is not changed.');
  }, 15000);

  it('also for numbers typed into the fields', async () => {
    backend.updateAnswer = { pc_map_action: 'shifted' };
    render(<PCRefinement />);
    fireEvent.change(await ready(), { target: { value: '0.51' } });
    fireEvent.change(screen.getByLabelText('PC Y:'), { target: { value: '0.34' } });
    fireEvent.change(screen.getByLabelText('PC Z:'), { target: { value: '0.84' } });
    fireEvent.click(screen.getByRole('button', { name: 'Use these numbers' }));
    expect((await screen.findByTestId('pc-grant-text')).textContent).toBe(
      'PC in use for scan_a: 0.5100, 0.3400, 0.8400 (mean of its per-pixel map). Kept until you close Orienta. The file on disk is not changed.');
  }, 15000);

  it('in every language', () => {
    for (const lng of ['en', 'de', 'ja', 'zh']) {
      const t = i18n.getFixedT(lng, 'pcrefinement');
      const plain = t('grant.inUseOne', { files: 'f', pc: 'P' });
      const map = t('grant.inUseOneMap', { files: 'f', pc: 'P' });
      expect(map).not.toBe(plain);
      expect(map).not.toContain('grant.');
    }
  });
});
