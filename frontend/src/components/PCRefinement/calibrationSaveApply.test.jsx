// @vitest-environment jsdom
//
// Save as calibration, Apply to other files and the question after a search at a file
// that was on a saved calibration, through the real page, the real i18n and a fake backend that refuses
// every method it was not told about. Expected numbers are worked out by hand in the comments.
//
// The rule under test: NOTHING IS PASSED ON WITHOUT A BUTTON. Saving writes a new calibration and changes no file;
// applying names the files ticked (and names a file whose settings differ a second time); the question changes
// nothing until a button answers it.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, within, act } from '@testing-library/react';

const backend = vi.hoisted(() => ({
  dataset: 'scan_a', filePath: '/d/scan_a.h5oina', files: {}, calls: [], unknown: [], saved: [], createError: null,
  created: null, cands: [], candsError: null, applyAnswers: {}, applyError: null, result: null, jobStatus: 'completed',
  phaseLoaded: true, patterns: [], older: [], calInfo: null, versionAnswer: null, newVersionError: null,
  switchError: null, switchErrors: [], loaded: [], suggestions: [], superseded: false, supersededQueue: [], switchGate: null,
  gates: [], copyOnSwitch: null, spelled: null, statusN: 0, target: null, putBackGate: null, putBackError: null,
  suggestAnswer: null, suggestCalls: [],
}));

const coded = (status, code, detail = 'English sentence') => Object.assign(new Error(String(status)), {
  response: { status, data: { detail }, headers: { 'x-orienta-code': code } },
});

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
      backend.statusN += 1;
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
    autoPick: async () => { backend.calls.push(['autoPick']); return { data: { success: true, patterns: [], n_patterns: 0, notes: [] } }; },
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
    putBack: async (dataset) => {
      backend.calls.push(['putBack', dataset]);
      if (backend.putBackGate) await backend.putBackGate;                         // a slow answer
      if (backend.putBackError) throw backend.putBackError;                       // a refusal
      const f = backend.files[dataset];
      f.pc = f.before.pc; f.source = f.before.had_pc ? 'header' : 'missing'; f.kind = 'file'; f.calibration = null;
      return ok({ dataset, pc: f.before.pc, had_pc: f.before.had_pc, restored: [dataset], not_restored: [], pc_per_pixel: false });
    },
    updatePC: (...args) => {
      backend.calls.push(['updatePC', ...args.slice(0, 3)]);
      const f = cur();
      const previous = { pc: f.pc, had_pc: f.source !== 'missing' };
      f.before = previous; f.pc = args.slice(0, 3); f.source = 'manual'; f.kind = 'typed';
      return ok({ success: true, ci: 0.2, segments: [], pc_map_action: 'none', pc_stored: true, previous });
    },
  }, 'pcApi');
  const pcCalibrationsApi = strict({
    suggest: (dataset) => {
      backend.suggestCalls.push(dataset);
      if (backend.suggestAnswer) return ok(backend.suggestAnswer);
      return ok({ dataset, suggestions: backend.suggestions, current: null, unverified: false,
        ...(backend.target ? { target: backend.target } : {}) });
    },
    list: () => ok({ calibrations: backend.saved, problems: [] }),
    create: (name, dataset) => {
      backend.calls.push(['create', name, dataset]);
      if (backend.createError) return Promise.reject(backend.createError);
      const made = backend.created || {};
      backend.saved = [...backend.saved, { id: 'rem-1', name }];
      return ok({ id: 'rem-1', name, version: 1, pc: [0.5026, 0.3272, 0.8458], version_created: '2026-11-01T10:00:00+00:00',
        source_name: `${dataset}.h5oina`, quality: null, ...made });
    },
    get: (id) => { backend.calls.push(['get', id]); return ok(backend.calInfo); },
    candidates: (id) => {
      backend.calls.push(['candidates', id]);
      return backend.candsError ? Promise.reject(backend.candsError) : ok({ candidates: backend.cands.map((c) => ({ ...c })) });
    },
    apply: (id, datasets, accept = [], version = null) => {
      backend.calls.push(['apply', id, datasets, accept, version]);
      if (backend.applyError) return Promise.reject(backend.applyError);
      datasets.forEach((d) => {
        const f = backend.files[d];
        if (f && !backend.applyAnswers[d]) { f.before = { pc: f.pc, had_pc: true }; f.pc = [0.5026, 0.3272, 0.8458]; f.source = 'calibration'; f.kind = 'calibration'; f.calibration = { id, name: 'SEM 2026-10-14', version }; }
      });
      return ok({ calibration: id, results: datasets.map((d) => backend.applyAnswers[d] || { dataset: d, status: 'applied', pc: [0.5026, 0.3272, 0.8458], version,
        mismatch_accepted: accept.includes(d), pc_map_action: 'none', pc_map_actions: {}, previous: { pc: [0.5019, 0.3301, 0.842], had_pc: true } }) });
    },
    newVersion: (id, dataset, applyTo, accept, fromVersion) => {
      backend.calls.push(['newVersion', id, dataset, applyTo, accept, fromVersion]);
      if (backend.newVersionError) return Promise.reject(backend.newVersionError);
      return ok(backend.versionAnswer);
    },
    results: (id) => { backend.calls.push(['results', id]); return ok({ calibration: id, latest: 2, results: backend.older }); },
  }, 'pcCalibrationsApi');
  const ebsdApi = new Proxy({
    info: () => ok({ loaded: true, active_dataset: backend.dataset, file_path: backend.filePath, data_shape: [90, 20, 128, 156] }),
    switchFile: async (path) => {
      backend.calls.push(['switchFile', path]);
      // what this call does is fixed when it is made: one gate, one refusal and one "replaced by another load" per call
      const gate = backend.gates.length ? backend.gates.shift() : backend.switchGate;
      const refusal = backend.switchErrors.length ? backend.switchErrors.shift() : backend.switchError;
      const replaced = backend.supersededQueue.length ? backend.supersededQueue.shift() : backend.superseded;
      if (gate) await gate;                                                       // a switch that takes its time
      if (refusal) return Promise.reject(refusal);
      // another load took the file's place: the backend stays on the file it had
      if (replaced) return { superseded: true, data: { loaded: true, file_path: backend.filePath } };
      const name = Object.keys(backend.files).find((n) => backend.files[n].path === path);
      // switching back restores the dataset that was active there last: a derived copy, or the file spelled otherwise
      backend.dataset = backend.copyOnSwitch || name; backend.filePath = backend.spelled || path;
      return { data: { loaded: true, file_path: backend.filePath } };
    },
    loadedFiles: () => ok({ files: backend.loaded.map((f) => ({ ...f, active: f.path === backend.filePath })) }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  const calibrationApi = new Proxy({ getEntry: () => ok({ parent_name: null }) }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  // the real one reads the load registry of the real axios instance, which a fake switch does not enter
  const isSupersededLoad = async (res) => res?.superseded === true;
  return { ...orig, pcApi, pcCalibrationsApi, indexApi: quiet, ebsdApi, calibrationApi, isSupersededLoad };
});

import PCRefinement from './PCRefinement';
import useDataStore from '../../stores/useDataStore';
import useApplyFlow from './Calibration/applyFlow';
import i18n from '../../i18n';

class FakeImage { set src(_v) { this.width = 100; this.height = 100; queueMicrotask(() => this.onload && this.onload()); } }
const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});

// The app mounts the PC page with the open file's path as its key (App.jsx): a new file is a new page.
function Host(props) {
  const filePath = useDataStore((s) => s.filePath);
  return <PCRefinement key={filePath || 'no-file'} {...props} />;
}

const FOUND = [0.5026, 0.3272, 0.8458];
const OLD = [0.5019, 0.3301, 0.842];
const FIVE = Array.from({ length: 5 }, (_, i) => ({ row: i + 1, col: 2 * i + 1 }));
const SETTINGS = { pattern_shape: [128, 156], binning: 8, camera_tilt: 5, sample_tilt: 70, working_distance_mm: 20, detector_insertion_mm: 55 };
const ALL6 = ['pattern_shape', 'binning', 'camera_tilt', 'sample_tilt', 'working_distance_mm', 'detector_insertion_mm'];
const cand = (dataset, o = {}) => ({
  dataset, has_pc: true, pc: OLD, ref: null, settings: { file: SETTINGS }, match: true, unverified: false, differences: [],
  unknown: [], checked: ALL6, on_this_version: false, on_calibration: false, path: `/d/${dataset}.h5oina`, ...o,
});
const FAR = cand('scan_far', {
  match: false, pc: [0.5004, 0.333, 0.8399], settings: { file: { ...SETTINGS, working_distance_mm: 15 } },
  differences: [{ key: 'working_distance_mm', file: 15, calibration: 20 }],
});
const OLDER_FILE = cand('scan_old', {
  match: false, unverified: true, pc: FOUND, settings: { file: { ...SETTINGS, working_distance_mm: null, binning: null } },
  unknown: ['binning', 'working_distance_mm'], checked: ['pattern_shape'],
});
const RESULT = {
  mean_pc: FOUND, pc_values: [[0.5, 0.33, 0.85]], ci: 0.4, global_ci: 0.4, segments: [], pattern_results: [], pc_map_action: 'none',
  dataset_changed: false, mode: 'find', file_pc: null, pattern_size: [128, 156],
  search: { mode: 'find', recentred: 0, far_box_used: false, on_edge: [], second_position: null,
    confidence: { state: 'notJudged', reasons: [], contrast: 0.8 },
    per_pattern: [[0.50, 0.33, 0.85], [0.52, 0.34, 0.865], [0.51, 0.335, 0.855]] },
  quality: { contrast: 0.8, warning: null, axes: [], n_patterns: 9, contrast_before: null },
  previous: { pc: OLD, had_pc: true }, pc_warning: null, pc_warning_codes: [], calibration: { calibration: null, diverged: false, followers: [] },
};
const ON_EDGE = {
  ...RESULT, search: { ...RESULT.search, on_edge: ['y', 'z'] },
  quality: { ...RESULT.quality, contrast: 0.4, warning: 'edge', axes: ['y', 'z'] },
  pc_warning: 'English fallback', pc_warning_codes: [{ code: 'pcOnBoxEdge', params: { axes: 'y, z' } }],
};
// a refine at scan_a, which was on version 1 of "SEM 2026-10-14"; scan_b and scan_c are still on it
const REFINE = {
  ...RESULT, mode: 'refine', search: { ...RESULT.search, mode: 'refine' },
  quality: { contrast: 0.8, warning: null, axes: [], n_patterns: 9, contrast_before: 0.7 },
  calibration: { calibration: { id: 'sem', name: 'SEM 2026-10-14', version: 1, latest: 1 }, diverged: true,
    followers: [{ dataset: 'scan_b', per_pixel: false, map_kind: null }, { dataset: 'scan_c', per_pixel: false, map_kind: null }] },
};
// the same, but scan_b holds a map from a grid calibration and scan_c one its file brought
const REFINE_MAPS = {
  ...REFINE,
  calibration: { ...REFINE.calibration, followers: [
    { dataset: 'scan_b', per_pixel: true, map_kind: 'grid' }, { dataset: 'scan_c', per_pixel: true, map_kind: 'header' }] },
};
const CAL_INFO = { id: 'sem', name: 'SEM 2026-10-14', version: 1, versions: [{ version: 1, pc: OLD }] };
const SAVED_CAL = { id: 'rem-1', name: 'REM A', version: 1, pc: FOUND, version_created: '2026-11-01T10:00:00+00:00', source_name: 'scan_a.h5oina' };

const fileOf = (name, pc = OLD) => ({ path: `/d/${name}.h5oina`, pc: [...pc], source: 'header', kind: 'file', calibration: null, before: null });

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = function getContext() { this.__ctx = this.__ctx || fakeContext(); return this.__ctx; };
  globalThis.Image = FakeImage;
  Object.assign(backend, {
    dataset: 'scan_a', filePath: '/d/scan_a.h5oina', calls: [], unknown: [], saved: [], createError: null, created: null,
    cands: [], candsError: null, applyAnswers: {}, applyError: null, result: RESULT, jobStatus: 'completed', phaseLoaded: true,
    patterns: FIVE.map((p) => ({ ...p })), older: [], calInfo: CAL_INFO, versionAnswer: null, newVersionError: null, switchError: null, suggestions: [],
    superseded: false, supersededQueue: [], switchGate: null, gates: [], copyOnSwitch: null, spelled: null, switchErrors: [],
    statusN: 0, target: null, putBackGate: null, putBackError: null, suggestAnswer: null, suggestCalls: [],
    files: { scan_a: fileOf('scan_a'), scan_far: fileOf('scan_far', [0.5, 0.5, 0.5]) },
    loaded: [{ path: '/d/scan_a.h5oina', name: 'scan_a.h5oina' }, { path: '/d/scan_far.h5oina', name: 'scan_far.h5oina' }],
  });
  useApplyFlow.setState({ flow: null });
  useDataStore.setState({ gridShape: [90, 20], isFileOpen: true, filePath: '/d/scan_a.h5oina', ebsdInfo: { mark: 1 } });
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
  delete window.electronAPI;
});
afterEach(() => {
  cleanup();
  useApplyFlow.setState({ flow: null });
  // a call to a method the fakes do not know failed the test, also when a catch ate it
  expect(backend.unknown).toEqual([]);
});

const callsOf = (name) => backend.calls.filter((c) => c[0] === name);
const text = (el) => el.textContent.replace(/\s+/g, ' ').trim();
const runFind = async (result = RESULT, props = {}) => {
  backend.result = result;
  render(<PCRefinement {...props} />);
  await screen.findAllByTitle(/^Pattern at row/);
  fireEvent.click(await screen.findByRole('button', { name: 'Find PC' }));
  return screen.findByTestId('pc-result', {}, { timeout: 5000 });
};
const openSave = async (result = RESULT) => {
  const res = await runFind(result);
  const where = result === RESULT ? within(await screen.findByTestId('pc-grant')) : within(res);
  fireEvent.click(where.getAllByRole('button', { name: 'Save as calibration…' })[0]);
  return screen.findByTestId('calibration-save-dialog');
};
const type = (el, value) => fireEvent.change(el, { target: { value } });

describe('Save as calibration (screen 5)', () => {
  it('stands beside the line that gives a Find PC result, after the other buttons', async () => {
    await runFind(RESULT, { onNavigate: vi.fn() });
    const grant = await screen.findByTestId('pc-grant');
    expect(within(grant).getAllByRole('button').map((b) => b.textContent))
      .toEqual(['Show the bands over a pattern', 'Go to indexing', 'Save as calibration…', 'Put it back']);
  }, 15000);

  it('shows the PC, the fit contrast of the result, keeps Save off until a name is typed, and saves under that name', async () => {
    backend.saved = [{ id: 'sem', name: 'SEM 2026-10-14' }];
    const dialog = await openSave();
    expect(within(dialog).getByRole('heading').textContent).toBe('Save as calibration');
    expect(text(dialog)).toContain('Saves the PC 0.5026, 0.3272, 0.8458 of this file with the settings it was acquired with');
    expect(text(dialog)).toContain('Nothing about this file changes.');
    expect(within(dialog).getByTestId('save-contrast').textContent).toBe('Fit contrast 0.8 of 1, higher is better');
    expect(within(dialog).queryByTestId('save-warning')).toBeNull();
    const save = within(dialog).getByRole('button', { name: 'Save' });
    expect(save.disabled).toBe(true);
    type(within(dialog).getByLabelText('Name'), '  REM A ');
    expect(save.disabled).toBe(false);
    expect(callsOf('create')).toEqual([]);                      // typing a name saves nothing
    fireEvent.click(save);
    await within(dialog).findByTestId('save-done');
    // the name is sent trimmed, for this file
    expect(callsOf('create')).toEqual([['create', 'REM A', 'scan_a']]);
    // no quality stored for this fake answer: the sentence holds the PC and the settings of the file only
    expect(within(dialog).getByTestId('save-done-text').textContent).toBe(
      'Calibration "REM A" saved 2026-11-01 (version 1). It holds the PC 0.5026, 0.3272, 0.8458 and the settings of scan_a.h5oina. '
      + 'The file itself is not changed. You can use the calibration again after you close Orienta.');
    expect(within(dialog).getByRole('button', { name: 'Apply to other files…' })).toBeTruthy();
    expect(within(dialog).getByRole('button', { name: 'Save' }).disabled).toBe(true);     // saved once
  }, 15000);

  it('says that a file with a PC for every map point is saved by their mean, and says nothing of the kind for a file with one PC', async () => {
    const plain = await openSave();
    expect(within(plain).queryByTestId('save-per-pixel')).toBeNull();
    cleanup();
    backend.files.scan_a.perPixel = true;
    const dialog = await openSave();
    expect(within(dialog).getByTestId('save-per-pixel').textContent)
      .toBe('This file has a PC for every map point. The calibration keeps their mean.');
  }, 20000);

  it('says in words that the name is taken, whatever the case and the spaces, and does not ask the backend', async () => {
    backend.saved = [{ id: 'sem', name: 'SEM 2026-10-14' }];
    const dialog = await openSave();
    await waitFor(() => expect(callsOf('create')).toEqual([]));
    type(within(dialog).getByLabelText('Name'), ' sem   2026-10-14');
    expect(await within(dialog).findByText('A calibration with this name exists already.')).toBeTruthy();
    expect(within(dialog).getByRole('button', { name: 'Save' }).disabled).toBe(true);
    type(within(dialog).getByLabelText('Name'), 'SEM 2026-10-15');
    expect(within(dialog).queryByText('A calibration with this name exists already.')).toBeNull();
    expect(within(dialog).getByRole('button', { name: 'Save' }).disabled).toBe(false);
  }, 15000);

  it.each([
    ['nameTaken', 'A calibration with this name exists already.'],
    ['badName', 'The name holds a character that cannot be saved.'],
    ['nameTooLong', 'The name is too long.'],
  ])('a refusal %s of the backend is said in words', async (code, words) => {
    const dialog = await openSave();
    backend.createError = coded(code === 'nameTaken' ? 409 : 400, code, 'English sentence');
    type(within(dialog).getByLabelText('Name'), 'Whatever');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
    expect((await within(dialog).findByTestId('save-error')).textContent).toBe(words);
    expect(within(dialog).queryByTestId('save-done')).toBeNull();
  }, 15000);

  it('a result at the edge: the window shows the warning, and the saved calibration says it keeps it', async () => {
    backend.created = { quality: { contrast: 0.4, warning: 'edge', axes: ['y', 'z'], n_patterns: 9 } };
    const dialog = await openSave(ON_EDGE);
    expect(within(dialog).getByTestId('save-contrast').textContent).toBe('Fit contrast 0.4 of 1, higher is better');
    expect(within(dialog).getByTestId('save-warning').textContent).toBe('The result lies at the edge of the searched area.');
    expect(text(dialog)).toContain('The calibration keeps this warning, so it is shown whenever the calibration is offered.');
    type(within(dialog).getByLabelText('Name'), 'Edge one');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
    expect((await within(dialog).findByTestId('save-done-text')).textContent).toContain(
      'It holds the PC 0.5026, 0.3272, 0.8458, the fit contrast 0.4, the warning about the edge of the searched area, and the settings of scan_a.h5oina.');
  }, 15000);

  it('a result at the edge has Save in Details with the sentence, and again after "Keep this PC anyway"', async () => {
    const res = await runFind(ON_EDGE, { onNavigate: vi.fn() });
    const details = within(res).getByTestId('pc-details-save');
    expect(text(details)).toBe('Save as calibration…The calibration keeps the warning above.');
    fireEvent.click(within(res).getByRole('button', { name: 'Keep this PC anyway' }));
    const kept = await screen.findByTestId('pc-kept');
    expect(text(kept)).toBe('This PC stays in use for scan_a. Kept until you close Orienta. To use it next time, save it as a calibration.Go to indexingSave as calibration…');
    fireEvent.click(within(kept).getByRole('button', { name: 'Save as calibration…' }));
    expect(await screen.findByTestId('save-warning')).toBeTruthy();
  }, 15000);

  it('"Compare with a saved calibration" is in the decision box only when one is saved, and opens the window of saved ones', async () => {
    const res = await runFind(ON_EDGE);
    expect(within(res).queryByRole('button', { name: 'Compare with a saved calibration' })).toBeNull();
    cleanup();
    backend.saved = [{ id: 'sem', name: 'SEM 2026-10-14' }];
    const res2 = await runFind(ON_EDGE);
    const compare = await within(res2).findByRole('button', { name: 'Compare with a saved calibration' });
    expect(screen.queryByTestId('take-pc-dialog')).toBeNull();
    fireEvent.click(compare);
    expect(await screen.findByTestId('take-pc-dialog')).toBeTruthy();
  }, 20000);

  it('stands beside a typed PC too, saving the PC that was typed', async () => {
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    for (const [name, v] of [['PC X:', '0.51'], ['PC Y:', '0.33'], ['PC Z:', '0.84']]) fireEvent.change(screen.getByLabelText(name), { target: { value: v } });
    fireEvent.click(screen.getByRole('button', { name: 'Use these numbers' }));
    const grant = await screen.findByTestId('pc-typed-grant');
    fireEvent.click(within(grant).getByRole('button', { name: 'Save as calibration…' }));
    const dialog = await screen.findByTestId('calibration-save-dialog');
    expect(text(dialog)).toContain('Saves the PC 0.5100, 0.3300, 0.8400 of this file');
    expect(within(dialog).queryByTestId('save-contrast')).toBeNull();       // a typed PC has no search behind it
  }, 15000);
});

describe('Save and Apply after a saved calibration was used', () => {
  it('"Use it" gives the grant Save as calibration… and Apply to other files…, which opens the window for the version used', async () => {
    backend.suggestions = [{
      id: 'sem', name: 'SEM 2026-10-14', version: 2, pc: FOUND, via_member: false, match: true, differences: [], unknown: [], unverified: false,
      checked: ALL6, settings: { file: SETTINGS, calibration: SETTINGS }, version_created: '2026-10-20T09:00:00+00:00', source_name: 'scan_x.h5oina',
      file_date: null, quality: null, replaces: { pc: OLD, kind: 'file' },
    }];
    backend.cands = [cand('scan_b')];
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Use it' }));
    const grant = await screen.findByTestId('calibration-grant');
    expect(callsOf('apply')).toEqual([['apply', 'sem', ['scan_a'], [], 2]]);
    fireEvent.click(within(grant).getByRole('button', { name: 'Save as calibration…' }));
    expect(await screen.findByTestId('calibration-save-dialog')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    fireEvent.click(within(grant).getByRole('button', { name: 'Apply to other files…' }));
    const dialog = await screen.findByTestId('apply-dialog');
    expect(within(dialog).getByRole('heading').textContent).toBe('Apply SEM 2026-10-14, version 2 (0.5026, 0.3272, 0.8458) to other files');
    await within(dialog).findByTestId('apply-table');
    expect(callsOf('candidates')).toEqual([['candidates', 'sem']]);
    // the applying is a second, separate step: only now was nothing else applied
    expect(callsOf('apply')).toHaveLength(1);
  }, 20000);

  const COPY_OFFER = {
    id: 'sem', name: 'SEM 2026-10-14', version: 2, pc: FOUND, via_member: false, match: true, differences: [], unknown: [], unverified: false,
    checked: ALL6, settings: { file: SETTINGS, calibration: SETTINGS }, version_created: '2026-10-20T09:00:00+00:00', source_name: 'scan_x.h5oina',
    file_date: null, quality: null, replaces: { pc: OLD, kind: 'file' },
  };

  it('on an open copy the offer is applied to the file the copy was made from, and a copy with a PC of its own is told so', async () => {
    backend.suggestions = [COPY_OFFER];
    backend.target = 'scan_root';
    backend.files.scan_root = fileOf('scan_root');
    backend.applyAnswers = { scan_root: { dataset: 'scan_root', status: 'applied', pc: FOUND, version: 2, mismatch_accepted: false,
      pc_map_action: 'none', pc_map_actions: { scan_root: 'single' }, previous: { pc: OLD, had_pc: true } } };
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Use it' }));
    const grant = await screen.findByTestId('calibration-grant');
    expect(callsOf('apply')).toEqual([['apply', 'sem', ['scan_root'], [], 2]]);                  // the file, not the open copy
    expect(within(grant).getByTestId('pc-grant-text').textContent).toContain('PC in use for scan_root');
    expect(within(grant).getByTestId('calibration-copy-kept').textContent)
      .toBe('The calibration was applied to scan_root, the file this copy was made from. scan_a has a PC of its own and keeps it.');
  }, 20000);

  it('on an open copy that the apply wrote too there is no such line', async () => {
    backend.suggestions = [COPY_OFFER];
    backend.target = 'scan_root';
    backend.files.scan_root = fileOf('scan_root');
    backend.applyAnswers = { scan_root: { dataset: 'scan_root', status: 'applied', pc: FOUND, version: 2, mismatch_accepted: false,
      pc_map_action: 'none', pc_map_actions: { scan_root: 'single', scan_a: 'single' }, previous: { pc: OLD, had_pc: true } } };
    render(<PCRefinement />);
    fireEvent.click(await screen.findByRole('button', { name: 'Use it' }));
    await screen.findByTestId('calibration-grant');
    expect(callsOf('apply')).toEqual([['apply', 'sem', ['scan_root'], [], 2]]);
    expect(screen.queryByTestId('calibration-copy-kept')).toBeNull();
  }, 20000);

  it('"Use it" refused because the calibration file is held says so, and the page asks the backend for its PC again', async () => {
    backend.suggestions = [{
      id: 'sem', name: 'SEM 2026-10-14', version: 2, pc: FOUND, via_member: false, match: true, differences: [], unknown: [], unverified: false,
      checked: ALL6, settings: { file: SETTINGS, calibration: SETTINGS }, version_created: '2026-10-20T09:00:00+00:00', source_name: 'scan_x.h5oina',
      file_date: null, quality: null, replaces: { pc: OLD, kind: 'file' },
    }];
    backend.applyAnswers = { scan_a: { dataset: 'scan_a', status: 'notWritable' } };
    render(<PCRefinement />);
    const use = await screen.findByRole('button', { name: 'Use it' });
    const before = backend.statusN;
    fireEvent.click(use);
    await screen.findByText(/calibration file could not be written/);
    await waitFor(() => expect(backend.statusN).toBeGreaterThan(before));
    expect(screen.queryByTestId('calibration-grant')).toBeNull();
  }, 20000);
});

describe('Apply to other files (screen 6)', () => {
  const openApply = async () => {
    const dialog = await openSave();
    type(within(dialog).getByLabelText('Name'), 'REM A');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
    fireEvent.click(await within(dialog).findByRole('button', { name: 'Apply to other files…' }));
    return screen.findByTestId('apply-dialog');
  };

  it('A: lists the files that fit with their working distance, old PC with version and new PC, ticks those that would change', async () => {
    backend.created = { version: 2, source_name: 'scan_x.h5oina', version_created: '2026-10-20T09:00:00+00:00' };
    backend.cands = [
      cand('scan_a'),                                                  // the file the PC is from: not a row
      cand('scan_b', { ref: { id: 'rem-1', name: 'REM A', version: 1 } }),
      cand('scan_c', { ref: { id: 'rem-1', name: 'REM A', version: 2 }, pc: FOUND, on_this_version: true, on_calibration: true }),
      cand('scan_new', { has_pc: false, pc: null }),
    ];
    const dialog = await openApply();
    expect(within(dialog).getByRole('heading').textContent).toBe('Apply REM A, version 2 (0.5026, 0.3272, 0.8458) to other files');
    await within(dialog).findByTestId('apply-table');
    expect(callsOf('candidates')).toEqual([['candidates', 'rem-1']]);
    expect(text(dialog)).toContain('Files with the same settings that would change are ticked.');
    expect(within(dialog).getByTestId('apply-version').textContent).toBe('REM A, version 2, updated 2026-10-20 from scan_x.h5oina');
    expect(text(dialog)).toContain('The files on disk are not changed.');
    const rows = [...dialog.querySelectorAll('tbody tr')].map((r) => [r.dataset.file, ...[...r.children].slice(2).map((c) => c.textContent),
      r.querySelector('input').checked]);
    expect(rows).toEqual([
      ['scan_b', '20 mm, same', '0.5019, 0.3301, 0.8420version 1', '0.5026, 0.3272, 0.8458', true],
      ['scan_c', '20 mm, same', '0.5026, 0.3272, 0.8458version 2', 'no changeversion 2', false],
      ['scan_new', '20 mm, same', 'none', '0.5026, 0.3272, 0.8458', true],
    ]);
    expect(within(dialog).getByRole('button', { name: 'Apply to 2 files' })).toBeTruthy();
    expect(within(dialog).getByRole('button', { name: 'Cancel' })).toBeTruthy();
    expect(callsOf('apply')).toEqual([]);                              // nothing is applied by opening the window
  }, 20000);

  it('A: Select all ticks every file that can change, the count follows the ticks, and with none ticked there is no Apply button', async () => {
    backend.cands = [cand('scan_b'), cand('scan_d'), cand('scan_c', { on_this_version: true, ref: { id: 'rem-1', name: 'REM A', version: 1 } })];
    const dialog = await openApply();
    await within(dialog).findByTestId('apply-table');
    expect(within(dialog).getByRole('button', { name: 'Apply to 2 files' })).toBeTruthy();      // scan_b and scan_d
    fireEvent.click(within(dialog).getByRole('checkbox', { name: 'scan_d' }));
    expect(within(dialog).getByRole('button', { name: 'Apply to 1 file' })).toBeTruthy();       // singular
    fireEvent.click(within(dialog).getByRole('checkbox', { name: 'Select all' }));
    expect(within(dialog).getByRole('button', { name: 'Apply to 2 files' })).toBeTruthy();      // not 3: scan_c has no change
    fireEvent.click(within(dialog).getByRole('checkbox', { name: 'Select all' }));
    expect(within(dialog).queryByRole('button', { name: /^Apply to \d+ file/ })).toBeNull();     // no dead "Apply to 0 files"
    expect(within(dialog).getByTestId('apply-tick-hint').textContent).toBe('Tick a file to apply the calibration to it.');
    expect(text(dialog)).toContain('No file is ticked.');
  }, 20000);

  it('Apply names the ticked files only, says per file what happened, and each applied file can be put back', async () => {
    backend.cands = [cand('scan_b'), cand('scan_c', { on_this_version: true }), cand('scan_new', { has_pc: false, pc: null })];
    backend.applyAnswers = {
      scan_b: { dataset: 'scan_b', status: 'applied', pc: FOUND, version: 1, mismatch_accepted: false, pc_map_action: 'shifted', pc_map_actions: {},
        previous: { pc: OLD, had_pc: true } },
      scan_new: { dataset: 'scan_new', status: 'applied', pc: FOUND, version: 1, mismatch_accepted: false, pc_map_action: 'none', pc_map_actions: {},
        previous: { pc: null, had_pc: false } },
    };
    backend.files.scan_b = { ...fileOf('scan_b'), before: { pc: OLD, had_pc: true } };
    const dialog = await openApply();
    await within(dialog).findByTestId('apply-table');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 2 files' }));
    const result = await within(dialog).findByTestId('apply-result');
    expect(callsOf('apply')).toEqual([['apply', 'rem-1', ['scan_b', 'scan_new'], [], 1]]);     // scan_c was not ticked, nothing accepted
    expect([...result.querySelectorAll('[data-testid="apply-line"]')].map((l) => l.textContent)).toEqual([
      'scan_b: applied.', 'scan_c: not applied (already on this version).', 'scan_new: applied.',
    ]);
    expect(within(result).getByTestId('apply-map-note').textContent)
      .toBe('scan_b: The PC map of this file was moved so that its mean is this PC. Its shape is unchanged.');
    expect(within(result).getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_b, scan_new: 0.5026, 0.3272, 0.8458. Kept until you close Orienta. The files on disk are not changed.');
    expect(within(dialog).queryByRole('button', { name: /^Apply to/ })).toBeNull();       // once only
    // the undo of one file uses the PC it had before (its `previous`)
    expect(within(result).getByText('scan_b: previous PC: 0.5019, 0.3301, 0.8420')).toBeTruthy();
    expect(within(result).getByText('scan_new had no PC before.')).toBeTruthy();
    fireEvent.click(within(result).getAllByRole('button', { name: 'Put it back' })[0]);
    fireEvent.click(await screen.findByRole('button', { name: 'Yes, put it back' }));
    await waitFor(() => expect(callsOf('putBack')).toEqual([['putBack', 'scan_b']]));
  }, 20000);

  it('B: a file of another height-to-width ratio says a PC does not carry over; a binned one of the same ratio is named by its sizes', async () => {
    backend.cands = [
      cand('scan_b'),
      cand('scan_tall', { match: false, differences: [{ key: 'pattern_shape', file: [156, 128], calibration: [128, 156] }],
        path: '/d/scan_tall.h5oina' }),
      cand('scan_binned', { match: false, differences: [{ key: 'pattern_shape', file: [64, 78], calibration: [128, 156] }],
        path: '/d/scan_binned.h5oina' }),
    ];
    const dialog = await openApply();
    await within(dialog).findByTestId('apply-table');
    const others = [...dialog.querySelectorAll('[data-testid="apply-difference"]')].map((d) => d.textContent);
    expect(others).toContain('Other pattern shape: a PC does not carry over to another height-to-width ratio.');
    expect(others).toContain('Other pattern size: 64 × 78, calibration 128 × 156.');
    expect(others).toHaveLength(2);
  }, 20000);

  it('B: a file with another working distance cannot be ticked; "Allow this file" only unlocks the box and applies nothing', async () => {
    backend.cands = [cand('scan_b'), FAR];
    const dialog = await openApply();
    const box = await within(dialog).findByTestId('apply-other');
    expect(box.dataset.file).toBe('scan_far');
    expect(within(box).getByRole('checkbox', { name: 'scan_far' }).disabled).toBe(true);
    expect(text(box)).toContain('Other working distance: 15 mm, calibration 20 mm.');
    expect(text(box)).toContain('Recommended: Find PC for it, then Save as calibration. The calibration is offered for files with the same settings.');
    expect(text(box)).toContain('Usually wrong: Allow this file, then tick it and press Apply. A PC from another working distance is usually wrong.');
    fireEvent.click(within(box).getByRole('button', { name: 'Allow this file' }));
    const tick = within(box).getByRole('checkbox', { name: 'scan_far' });
    expect(tick.disabled).toBe(false);
    expect(tick.checked).toBe(false);                                        // unlocked, not ticked
    expect(within(box).getByTestId('far-allowed').textContent).toBe('You allowed this file. It is not ticked yet. Then tick it and press Apply.');
    expect(callsOf('apply')).toEqual([]);
    expect(within(dialog).getByRole('button', { name: 'Apply to 1 file' })).toBeTruthy();    // scan_b only
    fireEvent.click(tick);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 2 files' }));
    await within(dialog).findByTestId('apply-result');
    // scan_far is named a second time (accept_mismatch)
    expect(callsOf('apply')).toEqual([['apply', 'rem-1', ['scan_b', 'scan_far'], ['scan_far'], 1]]);
  }, 20000);

  it('B: a file whose settings are not stated can be ticked, and is named a second time', async () => {
    backend.cands = [OLDER_FILE];
    const dialog = await openApply();
    const box = await within(dialog).findByTestId('apply-other');
    expect(text(box)).toContain('Its settings are not stated, so they cannot be compared. Tick it only if you know it fits.');
    expect(text(dialog)).toContain('No file is ticked.');
    fireEvent.click(within(box).getByRole('checkbox', { name: 'scan_old' }));
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 1 file' }));
    await within(dialog).findByTestId('apply-result');
    expect(callsOf('apply')).toEqual([['apply', 'rem-1', ['scan_old'], ['scan_old'], 1]]);
  }, 20000);

  it('B: a file that does not fit but already has this version cannot be ticked', async () => {
    backend.cands = [{ ...OLDER_FILE, on_this_version: true, on_calibration: true }];
    const dialog = await openApply();
    const box = await within(dialog).findByTestId('apply-other');
    expect(box.getAttribute('data-kind')).toBe('same');
    expect(text(box)).toContain('no change');
    expect(text(box)).not.toContain('Tick it only if you know it fits.');
    expect(within(box).getByRole('checkbox', { name: 'scan_old' }).disabled).toBe(true);
  }, 20000);

  it('a file the backend refuses is listed as not applied, with its reason', async () => {
    backend.cands = [cand('scan_b')];
    backend.applyAnswers = { scan_b: { dataset: 'scan_b', status: 'notLoaded' } };
    const dialog = await openApply();
    await within(dialog).findByTestId('apply-table');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 1 file' }));
    const line = await within(dialog).findByTestId('apply-line');
    expect(line.textContent).toBe('scan_b: not applied (not loaded any more).');
    expect(within(dialog).queryByTestId('pc-grant')).toBeNull();           // nothing was applied, so no "PC in use"
  }, 20000);

  it('a calibration file that could not be written is said in words, and the page asks the backend for its PC again', async () => {
    backend.cands = [cand('scan_b')];
    backend.applyAnswers = { scan_b: { dataset: 'scan_b', status: 'notWritable' } };
    const dialog = await openApply();
    await within(dialog).findByTestId('apply-table');
    const before = backend.statusN;
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 1 file' }));
    const line = await within(dialog).findByTestId('apply-line');
    expect(line.textContent).toBe('scan_b: not applied (calibration file not writable, nothing changed).');
    await waitFor(() => expect(backend.statusN).toBeGreaterThan(before));
  }, 20000);

  it('a request the backend refuses outright reloads the PC of the page too', async () => {
    backend.cands = [cand('scan_b')];
    backend.applyError = coded(400, 'unknownCalibration', 'no such calibration');
    const dialog = await openApply();
    await within(dialog).findByTestId('apply-table');
    const before = backend.statusN;
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 1 file' }));
    await waitFor(() => expect(backend.statusN).toBeGreaterThan(before));
  }, 20000);

  it('C: with no other file loaded it says so and only closes', async () => {
    backend.cands = [cand('scan_a')];
    const dialog = await openApply();
    expect((await within(dialog).findByTestId('apply-none')).textContent)
      .toBe('No other file is loaded. Add files on the Batch page (quick-load) or load them one by one.');
    expect(within(dialog).queryByRole('button', { name: /^Apply to/ })).toBeNull();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Close' }));
    expect(screen.queryByTestId('apply-dialog')).toBeNull();
    expect(useApplyFlow.getState().flow).toBeNull();
  }, 20000);

  it('Cancel closes the window and applies nothing', async () => {
    backend.cands = [cand('scan_b')];
    const dialog = await openApply();
    await within(dialog).findByTestId('apply-table');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByTestId('apply-dialog')).toBeNull();
    expect(callsOf('apply')).toEqual([]);
  }, 20000);

  it('an unreadable list is said in words', async () => {
    backend.candsError = coded(404, 'unknownCalibration');
    const dialog = await openApply();
    expect((await within(dialog).findByTestId('apply-list-error')).textContent).toBe('This calibration does not exist any more.');
  }, 20000);
});

describe('Find PC for a file that does not fit (screen 6, example B)', () => {
  const openFarRow = async (cands = [FAR]) => {
    backend.cands = cands;
    render(<Host />);
    await screen.findAllByTitle(/^Pattern at row/);
    // the calibration of this file is applied to others from the window; open it as "Use it" would
    await act(async () => { useApplyFlow.getState().open(SAVED_CAL, 'scan_a'); });
    return screen.findByTestId('apply-dialog');
  };

  it('opens the file through the app\'s file switch and starts Find PC once in the page that mounts for it', async () => {
    backend.result = { ...RESULT, mode: 'find' };
    const dialog = await openFarRow();
    const box = await within(dialog).findByTestId('apply-other');
    expect(optimizes()).toEqual([]);
    fireEvent.click(within(box).getByRole('button', { name: 'Find PC for it' }));
    // the page is keyed by the file: the first page goes, a second one mounts for scan_far and its window is back
    await waitFor(() => expect(callsOf('switchFile')).toEqual([['switchFile', '/d/scan_far.h5oina']]));
    await waitFor(() => expect(optimizes()).toEqual([['optimize', 'scan_far', 'find']]), { timeout: 5000 });
    const again = await screen.findByTestId('apply-dialog');
    expect(again).not.toBe(dialog);                                         // a new page, a new window, the same flow
    // the result arrives in the row: the PC, the usable line, the same grant line with Save
    const result = await within(again).findByTestId('far-result', {}, { timeout: 5000 });
    expect(within(result).getByTestId('far-result-pc').textContent).toBe('0.5026, 0.3272, 0.8458');
    expect(within(result).getByTestId('far-usable').textContent).toBe('scan_far now has a usable PC.');
    expect(within(result).getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_far: 0.5026, 0.3272, 0.8458. Kept until you close Orienta. The file on disk is not changed.');
    expect(within(result).getByText('Previous PC: 0.5019, 0.3301, 0.8420')).toBeTruthy();
    expect(within(result).getByRole('button', { name: 'Put it back' })).toBeTruthy();
    // once: re-renders and the poll of the result did not start a second search
    await new Promise((r) => setTimeout(r, 300));
    expect(optimizes()).toHaveLength(1);
    // Save from the row: the calibration is for the far file, and the row says so with its working distance
    fireEvent.click(within(result).getByRole('button', { name: 'Save as calibration…' }));
    const save = await screen.findByTestId('calibration-save-dialog');
    type(within(save).getByLabelText('Name'), 'REM far');
    fireEvent.click(within(save).getByRole('button', { name: 'Save' }));
    await within(save).findByTestId('save-done');
    // this window is already about applying: it does not open a second one for the far file's calibration
    expect(within(save).queryByRole('button', { name: 'Apply to other files…' })).toBeNull();
    expect(callsOf('create')).toEqual([['create', 'REM far', 'scan_far']]);
    fireEvent.click(within(save).getByRole('button', { name: 'Close' }));
    expect((await within(again).findByTestId('far-saved')).textContent).toBe('Saved as the calibration "REM far" for 15 mm.');
    expect(within(again).queryByRole('button', { name: 'Save as calibration…' })).toBeNull();
  }, 30000);

  it('a result that is not certain is not called a usable PC, but still has the grant line and Save', async () => {
    backend.result = ON_EDGE;
    const dialog = await openFarRow();
    fireEvent.click(within(await within(dialog).findByTestId('apply-other')).getByRole('button', { name: 'Find PC for it' }));
    const result = await screen.findByTestId('far-result', {}, { timeout: 8000 });
    expect(within(result).queryByTestId('far-usable')).toBeNull();
    expect(within(result).getByTestId('far-unsure').textContent).toBe(
      'scan_far has a PC now, but the search could not fix it clearly. Check the bands on the page behind this window before you use it.');
    expect(within(result).getByRole('button', { name: 'Save as calibration…' })).toBeTruthy();
  }, 30000);

  it('does not start Find PC while no phase is loaded for the file, and says why', async () => {
    backend.phaseLoaded = false;
    const dialog = await openFarRow();
    fireEvent.click(within(await within(dialog).findByTestId('apply-other')).getByRole('button', { name: 'Find PC for it' }));
    const again = await waitFor(() => {
      const d = screen.getByTestId('apply-dialog');
      expect(within(d).getByText('Find PC starts as soon as a phase is loaded for this file.')).toBeTruthy();
      return d;
    }, { timeout: 5000 });
    expect(within(again).getByTestId('far-progress').textContent).toContain('scan_far is open now. Find PC starts.');
    expect(optimizes()).toEqual([]);
  }, 30000);

  it('a file that cannot be opened is said in words and nothing is searched', async () => {
    backend.switchError = coded(404, 'unknownDataset');
    const dialog = await openFarRow();
    fireEvent.click(within(await within(dialog).findByTestId('apply-other')).getByRole('button', { name: 'Find PC for it' }));
    expect(await screen.findByText('The file could not be opened: This file is not loaded.')).toBeTruthy();
    expect(optimizes()).toEqual([]);
  }, 20000);

  const FAR_PATH = '/d/scan_far.h5oina';
  const handOpen = async (name, path) => {
    await act(async () => {
      backend.dataset = name; backend.filePath = path;
      await useDataStore.getState().syncFromBackend();
    });
    await waitFor(() => expect(useDataStore.getState().filePath).toBe(path));
    await screen.findAllByTitle(/^Pattern at row/);
  };
  const settle = () => new Promise((r) => setTimeout(r, 600));

  it('starts Find PC when the file comes back on a copy of its dataset: a file is told by its path, not by a dataset name', async () => {
    backend.result = { ...RESULT, mode: 'find' };
    backend.files.scan_far_copy = fileOf('scan_far_copy', [0.5, 0.5, 0.5]);
    backend.copyOnSwitch = 'scan_far_copy';                 // switching back restores the dataset that was active there last
    const dialog = await openFarRow();
    fireEvent.click(within(await within(dialog).findByTestId('apply-other')).getByRole('button', { name: 'Find PC for it' }));
    await waitFor(() => expect(optimizes()).toEqual([['optimize', 'scan_far_copy', 'find']]), { timeout: 5000 });
    const again = await screen.findByTestId('apply-dialog');
    const result = await within(again).findByTestId('far-result', {}, { timeout: 5000 });
    // the PC was written to the dataset the search ran on: the grant line names it, and "Put it back" goes there
    expect(within(result).getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_far_copy: 0.5026, 0.3272, 0.8458. Kept until you close Orienta. The file on disk is not changed.');
    fireEvent.click(within(result).getByRole('button', { name: 'Put it back' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Yes, put it back' }));
    await waitFor(() => expect(callsOf('putBack')).toEqual([['putBack', 'scan_far_copy']]));
    await settle();
    expect(optimizes()).toHaveLength(1);                    // once
  }, 30000);

  it('knows the far file by the path its switch answered with, when that is spelled otherwise than the row\'s', async () => {
    backend.result = { ...RESULT, mode: 'find' };
    backend.spelled = 'D:\\d\\scan_far.h5oina';
    const dialog = await openFarRow();
    fireEvent.click(within(await within(dialog).findByTestId('apply-other')).getByRole('button', { name: 'Find PC for it' }));
    await waitFor(() => expect(optimizes()).toEqual([['optimize', 'scan_far', 'find']]), { timeout: 5000 });
    expect(useDataStore.getState().filePath).toBe('D:\\d\\scan_far.h5oina');
    expect(useApplyFlow.getState().flow.far.path).toBe(FAR_PATH);
    expect(useApplyFlow.getState().flow.far.openedPath).toBe('D:\\d\\scan_far.h5oina');
  }, 30000);

  it('a switch that another load replaced clears the step, says the file was not opened, and opening it by hand later starts nothing', async () => {
    backend.superseded = true;
    const dialog = await openFarRow();
    fireEvent.click(within(await within(dialog).findByTestId('apply-other')).getByRole('button', { name: 'Find PC for it' }));
    expect((await within(dialog).findByTestId('far-open-failed')).textContent)
      .toBe('The file could not be opened: another file was opened at the same moment and took its place.');
    expect(within(dialog).queryByTestId('far-progress')).toBeNull();                       // no "is open now"
    expect(useApplyFlow.getState().flow.far.step).toBe('error');
    expect(within(dialog).getByRole('button', { name: 'Find PC for it' })).toBeTruthy();   // and it can be asked for again
    backend.superseded = false;
    await handOpen('scan_far', FAR_PATH);
    await settle();
    expect(optimizes()).toEqual([]);                                                       // no press, no search
  }, 30000);

  it('a switch that fails clears the step too: the file opened by hand later starts nothing, and the button is back', async () => {
    backend.switchError = coded(404, 'unknownDataset');
    const dialog = await openFarRow();
    fireEvent.click(within(await within(dialog).findByTestId('apply-other')).getByRole('button', { name: 'Find PC for it' }));
    expect((await within(dialog).findByTestId('far-open-failed')).textContent).toBe('The file could not be opened: This file is not loaded.');
    expect(within(dialog).getByRole('button', { name: 'Find PC for it' })).toBeTruthy();
    backend.switchError = null;
    await handOpen('scan_far', FAR_PATH);
    await settle();
    expect(optimizes()).toEqual([]);
  }, 30000);

  it('a window closed while the file is being opened leaves no step behind: nothing starts when the file opens', async () => {
    let release;
    backend.switchGate = new Promise((r) => { release = r; });
    const dialog = await openFarRow();
    fireEvent.click(within(await within(dialog).findByTestId('apply-other')).getByRole('button', { name: 'Find PC for it' }));
    await waitFor(() => expect(callsOf('switchFile')).toHaveLength(1));
    expect(useApplyFlow.getState().flow.far.step).toBe('starting');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));
    expect(useApplyFlow.getState().flow).toBeNull();
    await act(async () => { release(); });
    await waitFor(() => expect(useDataStore.getState().filePath).toBe(FAR_PATH));
    await screen.findAllByTitle(/^Pattern at row/);
    await settle();
    expect(optimizes()).toEqual([]);
    expect(screen.queryByTestId('apply-dialog')).toBeNull();
  }, 30000);

  it('the outcome of an older press does not touch the file chosen after it', async () => {
    backend.result = { ...RESULT, mode: 'find' };
    const FAR2 = cand('scan_far2', {
      match: false, pc: [0.5004, 0.333, 0.8399], settings: { file: { ...SETTINGS, working_distance_mm: 15 } },
      differences: [{ key: 'working_distance_mm', file: 15, calibration: 20 }],
    });
    backend.files.scan_far2 = fileOf('scan_far2', [0.5, 0.5, 0.5]);
    backend.loaded.push({ path: '/d/scan_far2.h5oina', name: 'scan_far2.h5oina' });
    let release;
    backend.switchGate = new Promise((r) => { release = r; });
    backend.switchErrors = [coded(404, 'unknownDataset'), null];         // the first press fails, the second one works
    const dialog = await openFarRow([FAR, FAR2]);
    const boxes = await within(dialog).findAllByTestId('apply-other');
    fireEvent.click(within(boxes[0]).getByRole('button', { name: 'Find PC for it' }));        // scan_far
    await waitFor(() => expect(callsOf('switchFile')).toHaveLength(1));
    fireEvent.click(within(boxes[1]).getByRole('button', { name: 'Find PC for it' }));        // then scan_far2
    await waitFor(() => expect(callsOf('switchFile')).toHaveLength(2));
    await act(async () => { release(); });
    // the failure of the first must not end the step of the second
    await waitFor(() => expect(optimizes()).toEqual([['optimize', 'scan_far2', 'find']]), { timeout: 5000 });
  }, 30000);

  it('an older press whose switch another load replaced does not clear the step of the file chosen after it', async () => {
    backend.result = { ...RESULT, mode: 'find' };
    const FAR2 = cand('scan_far2', {
      match: false, pc: [0.5004, 0.333, 0.8399], settings: { file: { ...SETTINGS, working_distance_mm: 15 } },
      differences: [{ key: 'working_distance_mm', file: 15, calibration: 20 }],
    });
    backend.files.scan_far2 = fileOf('scan_far2', [0.5, 0.5, 0.5]);
    backend.loaded.push({ path: '/d/scan_far2.h5oina', name: 'scan_far2.h5oina' });
    let releaseFirst; let releaseSecond;
    backend.gates = [new Promise((r) => { releaseFirst = r; }), new Promise((r) => { releaseSecond = r; })];
    backend.supersededQueue = [true, false];                              // the first press is replaced, the second one works
    const dialog = await openFarRow([FAR, FAR2]);
    const boxes = await within(dialog).findAllByTestId('apply-other');
    fireEvent.click(within(boxes[0]).getByRole('button', { name: 'Find PC for it' }));        // scan_far
    await waitFor(() => expect(callsOf('switchFile')).toHaveLength(1));
    fireEvent.click(within(boxes[1]).getByRole('button', { name: 'Find PC for it' }));        // then scan_far2
    await waitFor(() => expect(callsOf('switchFile')).toHaveLength(2));
    await act(async () => { releaseFirst(); });                           // the older press ends first, replaced
    await new Promise((r) => setTimeout(r, 300));
    expect(useApplyFlow.getState().flow.far).toMatchObject({ dataset: 'scan_far2', step: 'starting' });
    await act(async () => { releaseSecond(); });
    await waitFor(() => expect(optimizes()).toEqual([['optimize', 'scan_far2', 'find']]), { timeout: 5000 });
  }, 30000);
});
const optimizes = () => callsOf('optimize');

describe('The question after a search at a file on a saved calibration (screen 7)', () => {
  beforeEach(() => {
    backend.cands = [cand('scan_a'), cand('scan_b'), cand('scan_c'), FAR];
    backend.versionAnswer = {
      calibration: { id: 'sem', name: 'SEM 2026-10-14', version: 2, pc: FOUND, version_created: '2026-11-01T10:00:00+00:00', source_name: 'scan_a.h5oina' },
      applied: [
        { dataset: 'scan_b', status: 'applied', pc: FOUND, version: 2, previous: { pc: OLD, had_pc: true }, pc_map_action: 'none' },
        { dataset: 'scan_c', status: 'applied', pc: FOUND, version: 2, previous: { pc: OLD, had_pc: true }, pc_map_action: 'none' },
      ],
    };
  });

  it('comes first, with the change per axis and in pixels, the fit contrast before and after, the files and those not included', async () => {
    const res = await runFind(REFINE);
    const q = await screen.findByTestId('follower-question');
    expect(q.compareDocumentPosition(res) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();       // above the result
    expect(within(q).getByText('Your PC changed. Should scan_b and scan_c use it too?')).toBeTruthy();
    // hand: dx = 0.5026 - 0.5019 = +0.0007, dy = 0.3272 - 0.3301 = -0.0029, dz = 0.8458 - 0.8420 = +0.0038;
    // pixels: (0.0007 * 156, -0.0029 * 128, 0.0038 * 128) = (0.1092, -0.3712, 0.4864), length sqrt(0.38629) = 0.62 -> 0.6
    await waitFor(() => expect(within(q).getByTestId('question-change').textContent)
      .toBe('Change: x +0.0007, y -0.0029, z +0.0038 (about 0.6 pixels). Fit contrast before and after: 0.7, 0.8 (of 1, higher is better)'));
    expect(within(q).getByTestId('question-files').textContent).toBe('On version 1 of "SEM 2026-10-14", same settings: scan_b, scan_c');
    expect(within(q).getByTestId('question-not-included').textContent)
      .toBe('Not included: scan_far (working distance 15 mm, calibration 20 mm)');
    expect(within(q).getByRole('button', { name: 'Only this file' })).toBeTruthy();
    expect(text(q)).toContain('If you are unsure, choose this. Nothing else changes.');
    expect(text(q)).toContain('New version for these 2 files too');
    expect(text(q)).toContain('Saves the new PC as version 2 of the calibration. scan_b and scan_c move to it.');
    expect(text(q)).toContain('Leaves version 1 as it is and saves the new PC under a name you choose.');
    // Save as calibration… waits until the question is answered; nothing was passed on
    expect(within(res).queryByRole('button', { name: 'Save as calibration…' })).toBeNull();
    expect(callsOf('newVersion')).toEqual([]);
    expect(callsOf('apply')).toEqual([]);
    expect(callsOf('create')).toEqual([]);
  }, 20000);

  it('"Only this file" changes nothing in the backend, says so, and lets Save show', async () => {
    const res = await runFind(REFINE);
    const q = await screen.findByTestId('follower-question');
    fireEvent.click(within(q).getByRole('button', { name: 'Only this file' }));
    expect((await screen.findByTestId('question-answered')).textContent)
      .toBe('Only scan_a uses the new PC. scan_b and scan_c stay on version 1. The calibration is not changed.');
    expect(screen.queryByTestId('follower-question')).toBeNull();
    expect(within(res).getAllByRole('button', { name: 'Save as calibration…' }).length).toBeGreaterThan(0);
    expect(callsOf('newVersion')).toEqual([]);
    expect(callsOf('apply')).toEqual([]);
  }, 20000);

  it('"New version for these 2 files too" makes version 2 for the two followers only, then offers Apply to other files', async () => {
    await runFind(REFINE);
    const q = await screen.findByTestId('follower-question');
    await waitFor(() => expect(within(q).getByRole('button', { name: 'New version for these 2 files too' }).disabled).toBe(false));
    fireEvent.click(within(q).getByRole('button', { name: 'New version for these 2 files too' }));
    const done = await screen.findByTestId('question-version-done');
    // the version the question was about goes along: a follower that left it since is not written
    expect(callsOf('newVersion')).toEqual([['newVersion', 'sem', 'scan_a', ['scan_b', 'scan_c'], ['scan_b', 'scan_c'], 1]]);
    expect(done.textContent).toBe('Calibration "SEM 2026-10-14" is now version 2, updated 2026-11-01. It is saved in the calibration library.');
    expect(screen.getByTestId('question-not-included-after').textContent)
      .toBe('Not included: scan_far (working distance 15 mm, calibration 20 mm). Open it and press Find PC.');
    const answered = screen.getByTestId('question-answered');
    expect(within(answered).getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_b, scan_c: 0.5026, 0.3272, 0.8458. Kept until you close Orienta. The files on disk are not changed.');
    fireEvent.click(within(answered).getByRole('button', { name: 'Apply to other files…' }));
    expect(await screen.findByTestId('apply-dialog')).toBeTruthy();
  }, 20000);

  it('names the followers that hold a PC map before the click, with what a new version does to it; none for followers with one PC', async () => {
    await runFind(REFINE);
    const plain = await screen.findByTestId('follower-question');
    expect(within(plain).queryByTestId('question-map-grid')).toBeNull();
    expect(within(plain).queryByTestId('question-map-header')).toBeNull();
    cleanup();
    await runFind(REFINE_MAPS);
    const q = await screen.findByTestId('follower-question');
    expect(text(within(q).getByTestId('question-map-grid'))).toBe(
      'If you choose a new version, the PC map of scan_b (from a grid calibration) is replaced by the new PC. One PC then applies to the whole scan.');
    expect(text(within(q).getByTestId('question-map-header'))).toBe(
      'If you choose a new version, the PC map of scan_c is moved so that its mean is the new PC. Its shape is unchanged.');
    expect(callsOf('newVersion')).toEqual([]);                          // a line of text, not a click
  }, 20000);

  it('after "New version" says per file what became of each map', async () => {
    backend.versionAnswer.applied = [
      { dataset: 'scan_b', status: 'applied', pc: FOUND, version: 2, previous: { pc: OLD, had_pc: true }, pc_map_action: 'replaced' },
      { dataset: 'scan_c', status: 'applied', pc: FOUND, version: 2, previous: { pc: OLD, had_pc: true }, pc_map_action: 'shifted' },
    ];
    await runFind(REFINE_MAPS);
    const q = await screen.findByTestId('follower-question');
    await waitFor(() => expect(within(q).getByRole('button', { name: 'New version for these 2 files too' }).disabled).toBe(false));
    fireEvent.click(within(q).getByRole('button', { name: 'New version for these 2 files too' }));
    const notes = await screen.findAllByTestId('question-map-note');
    expect(notes.map((n) => [n.dataset.file, text(n)])).toEqual([
      ['scan_b', 'scan_b: The PC map of this file was replaced by this PC: one PC now applies to the whole scan.'],
      ['scan_c', 'scan_c: The PC map of this file was moved so that its mean is this PC. Its shape is unchanged.'],
    ]);
    expect(screen.queryByTestId('question-left')).toBeNull();
  }, 20000);

  it('a follower that has another PC since is said per file, is not counted as a failure, and is not in the grant line', async () => {
    backend.versionAnswer.applied = [
      { dataset: 'scan_b', status: 'applied', pc: FOUND, version: 2, previous: { pc: OLD, had_pc: true }, pc_map_action: 'none' },
      { dataset: 'scan_c', status: 'left' },
    ];
    await runFind(REFINE);
    const q = await screen.findByTestId('follower-question');
    await waitFor(() => expect(within(q).getByRole('button', { name: 'New version for these 2 files too' }).disabled).toBe(false));
    fireEvent.click(within(q).getByRole('button', { name: 'New version for these 2 files too' }));
    const left = await screen.findByTestId('question-left');
    expect(left.dataset.file).toBe('scan_c');
    expect(text(left)).toBe('scan_c was not moved to the new version: it has another PC now.');
    expect(screen.queryByText(/could not be moved to the new version/)).toBeNull();
    expect(within(screen.getByTestId('question-answered')).getByTestId('pc-grant-text').textContent)
      .toBe('PC in use for scan_b: 0.5026, 0.3272, 0.8458. Kept until you close Orienta. The file on disk is not changed.');
  }, 20000);

  it('"Save as a new calibration…" opens the save window and leaves the question open until it is saved', async () => {
    await runFind(REFINE);
    const q = await screen.findByTestId('follower-question');
    fireEvent.click(within(q).getByRole('button', { name: 'Save as a new calibration…' }));
    const dialog = await screen.findByTestId('calibration-save-dialog');
    expect(screen.getByTestId('follower-question')).toBeTruthy();
    expect(callsOf('create')).toEqual([]);
    type(within(dialog).getByLabelText('Name'), 'SEM new');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
    await within(dialog).findByTestId('save-done');
    expect(callsOf('create')).toEqual([['create', 'SEM new', 'scan_a']]);
    expect((await screen.findByTestId('question-answered')).textContent)
      .toBe('Version 1 of "SEM 2026-10-14" stays as it is. The new PC is saved as a calibration of its own.');
    expect(callsOf('newVersion')).toEqual([]);
  }, 20000);

  it.each([
    ['a search that left no calibration', { ...REFINE, calibration: { calibration: null, diverged: false, followers: [] } }],
    ['a file that did not leave it', { ...REFINE, calibration: { ...REFINE.calibration, diverged: false } }],
    ['a calibration nobody else is on', { ...REFINE, calibration: { ...REFINE.calibration, followers: [] } }],
  ])('shows nothing for %s', async (_name, result) => {
    await runFind(result);
    expect(screen.queryByTestId('follower-question')).toBeNull();
    expect(callsOf('get')).toEqual([]);
    expect(callsOf('candidates')).toEqual([]);
    expect(screen.getAllByRole('button', { name: 'Save as calibration…' }).length).toBeGreaterThan(0);
  }, 20000);
});

describe('Stored results of an older version', () => {
  const inUse = () => {
    backend.files.scan_a = { ...fileOf('scan_a', FOUND), source: 'calibration', kind: 'calibration', calibration: { id: 'sem', name: 'SEM 2026-10-14', version: 2 } };
  };

  it('lists only the results of older versions, with the way to Indexing', async () => {
    inUse();
    backend.older = [
      { result_id: 'r1', source_file: 'scan_a.h5oina', version: 1, latest: 2, state: 'older' },
      { result_id: 'r2', source_file: 'scan_b.h5oina', version: 2, latest: 2, state: 'current' },
    ];
    const onNavigate = vi.fn();
    render(<PCRefinement onNavigate={onNavigate} />);
    const box = await screen.findByTestId('older-results');
    expect(within(box).getByTestId('older-results-text').textContent).toBe(
      '1 stored result was computed with version 1 of this calibration. It was not changed, and it is marked with its version in the results list.');
    expect(text(box)).toContain('scan_a.h5oina (v1)');
    expect(text(box)).not.toContain('scan_b.h5oina');                    // the current one is not shown
    expect(callsOf('results')).toEqual([['results', 'sem']]);
    fireEvent.click(within(box).getByRole('button', { name: 'Go to Indexing' }));
    expect(onNavigate).toHaveBeenCalledWith('indexing');
  }, 15000);

  it('shows nothing when every stored result is current', async () => {
    inUse();
    backend.older = [{ result_id: 'r2', source_file: 'scan_b.h5oina', version: 2, latest: 2, state: 'current' }];
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    await waitFor(() => expect(callsOf('results')).toEqual([['results', 'sem']]));
    expect(screen.queryByTestId('older-results')).toBeNull();
  }, 15000);

  it('is not asked for while the PC in use is not a calibration', async () => {
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    await new Promise((r) => setTimeout(r, 50));
    expect(callsOf('results')).toEqual([]);
    expect(screen.queryByTestId('older-results')).toBeNull();
  }, 15000);
});

// ---------------------------------------------------------------------------------------------------------------------
// The window "Apply to other files", the question after a search and a name the backend does not know.
describe('The window "Apply to other files"', () => {
  const SAME = () => cand('scan_c', { on_this_version: true, ref: { id: 'rem-1', name: 'REM A', version: 1 }, pc: FOUND });
  const APPLIED_B = { dataset: 'scan_b', status: 'applied', pc: FOUND, version: 1, mismatch_accepted: false, pc_map_action: 'none',
    pc_map_actions: {}, previous: { pc: OLD, had_pc: true } };
  const APPLIED_NEW = { dataset: 'scan_new', status: 'applied', pc: FOUND, version: 1, mismatch_accepted: false, pc_map_action: 'none',
    pc_map_actions: {}, previous: { pc: null, had_pc: false } };
  const openDialog = async () => {
    const dialog = await openSave();
    type(within(dialog).getByLabelText('Name'), 'REM A');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
    fireEvent.click(await within(dialog).findByRole('button', { name: 'Apply to other files…' }));
    const apply = await screen.findByTestId('apply-dialog');
    await within(apply).findByTestId('apply-table');
    return apply;
  };
  // two files applied, the first with a PC before (scan_b) and the second without one (scan_new)
  const applyTwo = async () => {
    backend.cands = [cand('scan_b'), cand('scan_new', { has_pc: false, pc: null })];
    backend.applyAnswers = { scan_b: APPLIED_B, scan_new: APPLIED_NEW };
    backend.files.scan_b = { ...fileOf('scan_b'), before: { pc: OLD, had_pc: true } };
    const dialog = await openDialog();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 2 files' }));
    return within(dialog).findByTestId('apply-result');
  };

  it('1: a row marked "no change" cannot be ticked, by hand or by Select all, and is not sent', async () => {
    backend.cands = [cand('scan_b'), SAME()];
    const dialog = await openDialog();
    const box = within(dialog).getByRole('checkbox', { name: 'scan_c' });
    expect(box.disabled).toBe(true);
    expect(box.checked).toBe(false);
    fireEvent.click(box);
    expect(box.checked).toBe(false);
    expect(within(dialog).getByRole('button', { name: 'Apply to 1 file' })).toBeTruthy();       // scan_b only
    fireEvent.click(within(dialog).getByRole('checkbox', { name: 'Select all' }));              // off
    fireEvent.click(within(dialog).getByRole('checkbox', { name: 'Select all' }));              // on
    expect(box.checked).toBe(false);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 1 file' }));
    const result = await within(dialog).findByTestId('apply-result');
    expect(callsOf('apply')).toEqual([['apply', 'rem-1', ['scan_b'], [], 1]]);
    expect([...result.querySelectorAll('[data-testid="apply-line"]')].map((l) => l.textContent))
      .toEqual(['scan_b: applied.', 'scan_c: not applied (already on this version).']);
  }, 20000);

  it('1: when every file that fits is on this version, Select all is off and nothing can be ticked', async () => {
    backend.cands = [SAME()];
    const dialog = await openDialog();
    expect(within(dialog).getByRole('checkbox', { name: 'Select all' }).disabled).toBe(true);
    expect(within(dialog).getByRole('checkbox', { name: 'scan_c' }).disabled).toBe(true);
    expect(within(dialog).queryByRole('button', { name: /^Apply to \d+ file/ })).toBeNull();
  }, 20000);

  it('2: "Nothing is applied until you press Apply" is there before and gone after Done', async () => {
    backend.cands = [cand('scan_b')];
    const dialog = await openDialog();
    expect(within(dialog).getByTestId('apply-nothing-until').textContent)
      .toBe('Nothing is applied until you press Apply. The files on disk are not changed.');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 1 file' }));
    await within(dialog).findByTestId('apply-result');
    expect(within(dialog).getByTestId('apply-top').textContent).toBe('Done.');
    expect(within(dialog).queryByTestId('apply-nothing-until')).toBeNull();
    expect(text(dialog)).not.toContain('Nothing is applied until');
  }, 20000);

  it('2: "It is not ticked yet" is said only while the allowed file is not ticked, and not after Done', async () => {
    backend.cands = [cand('scan_b'), FAR];
    const dialog = await openDialog();
    const box = await within(dialog).findByTestId('apply-other');
    fireEvent.click(within(box).getByRole('button', { name: 'Allow this file' }));
    expect(within(box).getByTestId('far-allowed').textContent)
      .toBe('You allowed this file. It is not ticked yet. Then tick it and press Apply.');
    const tick = within(box).getByRole('checkbox', { name: 'scan_far' });
    fireEvent.click(tick);
    expect(within(box).getByTestId('far-allowed').textContent).toBe('You allowed this file.');
    fireEvent.click(tick);                                                  // unticked again: the sentence is true again
    expect(within(box).getByTestId('far-allowed').textContent).toContain('It is not ticked yet.');
    fireEvent.click(tick);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 2 files' }));
    await within(dialog).findByTestId('apply-result');
    expect(within(box).getByTestId('far-allowed').textContent).toBe('You allowed this file.');
    // an allowed file left unticked: after Done the sentence about ticking it is gone too
    cleanup();
    useApplyFlow.setState({ flow: null });
    backend.saved = [];
    backend.files = { scan_a: fileOf('scan_a'), scan_far: fileOf('scan_far', [0.5, 0.5, 0.5]), scan_b: fileOf('scan_b') };
    const again = await openDialog();
    const box2 = await within(again).findByTestId('apply-other');
    fireEvent.click(within(box2).getByRole('button', { name: 'Allow this file' }));
    fireEvent.click(within(again).getByRole('button', { name: 'Apply to 1 file' }));
    await within(again).findByTestId('apply-result');
    expect(within(box2).getByTestId('far-allowed').textContent).toBe('You allowed this file.');
  }, 30000);

  it('4: a row not applied says "other pattern shape" where the ratio differs and "other pattern size" where it does not', async () => {
    backend.cands = [
      cand('scan_b'),
      cand('scan_tall', { match: false, differences: [{ key: 'pattern_shape', file: [156, 128], calibration: [128, 156] }], path: '/d/scan_tall.h5oina' }),
      cand('scan_binned', { match: false, differences: [{ key: 'pattern_shape', file: [64, 78], calibration: [128, 156] }], path: '/d/scan_binned.h5oina' }),
    ];
    const dialog = await openDialog();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Apply to 1 file' }));
    const result = await within(dialog).findByTestId('apply-result');
    expect([...result.querySelectorAll('[data-testid="apply-line"]')].map((l) => l.textContent)).toEqual([
      'scan_b: applied.', 'scan_tall: not applied (other pattern shape).', 'scan_binned: not applied (other pattern size).',
    ]);
    // the rows above said the same words
    const rows = [...dialog.querySelectorAll('[data-testid="apply-difference"]')].map((d) => d.textContent);
    expect(rows[0]).toMatch(/^Other pattern shape:/);
    expect(rows[1]).toMatch(/^Other pattern size:/);
    for (const [lng, shape] of [['de', 'anderes Musterformat'], ['ja', 'パターンの形状が異なる'], ['zh', '图样形状不同']]) {
      expect(i18n.getFixedT(lng, 'pcrefinement')('applyDialog.why.differsShape')).toBe(shape);
    }
  }, 20000);

  describe('3: "Yes, put it back" for a file that is not open', () => {
    const lineOf = (result, file) => result.querySelector(`[data-testid="pc-undo"][data-file="${file}"]`);
    const pressPutBack = (result, file) => {
      fireEvent.click(within(lineOf(result, file)).getByRole('button', { name: file === 'scan_new' ? 'Remove this PC' : 'Put it back' }));
      fireEvent.click(screen.getByRole('button', { name: file === 'scan_new' ? 'Yes, remove this PC' : 'Yes, put it back' }));
    };

    it('a slow answer shows a running state, then the result in the row of that file only', async () => {
      let release;
      backend.putBackGate = new Promise((resolve) => { release = resolve; });
      const result = await applyTwo();
      pressPutBack(result, 'scan_b');
      expect(callsOf('putBack')).toEqual([['putBack', 'scan_b']]);
      expect(screen.getByRole('button', { name: 'Yes, put it back' }).disabled).toBe(true);
      expect(screen.getByTestId('pc-undo-working').textContent).toBe('Putting it back…');
      expect(screen.queryByTestId('pc-undo-done')).toBeNull();
      await act(async () => { release(); });
      const done = await within(result).findByTestId('pc-undo-done');
      expect(done.textContent).toBe('Previous PC put back for scan_b: 0.5019, 0.3301, 0.8420. Kept until you close Orienta. The file on disk is not changed.');
      expect(screen.queryByTestId('pc-undo-working')).toBeNull();
      // the other file's line is as it was: it still has its own step back
      expect(within(lineOf(result, 'scan_new')).getByRole('button', { name: 'Remove this PC' })).toBeTruthy();
      expect(callsOf('putBack')).toEqual([['putBack', 'scan_b']]);
    }, 30000);

    it('the answer for a file that had no PC says it has none again', async () => {
      const result = await applyTwo();
      backend.files.scan_new = { ...fileOf('scan_new'), before: { pc: null, had_pc: false } };
      pressPutBack(result, 'scan_new');
      const done = await within(result).findByTestId('pc-undo-done');
      expect(done.textContent).toBe('PC removed. scan_new has no PC again. Indexing is refused until there is a PC. The file on disk is not changed.');
    }, 30000);

    it('a refusal is said in words, the button comes back and the row is not marked done', async () => {
      backend.putBackError = coded(500, 'storeBusy', 'The store is busy.');
      const result = await applyTwo();
      pressPutBack(result, 'scan_b');
      const alert = await within(lineOf(result, 'scan_b')).findByRole('alert');
      expect(alert.textContent).toBe('The store is busy.');
      expect(within(lineOf(result, 'scan_b')).getByRole('button', { name: 'Put it back' }).disabled).toBe(false);
      expect(screen.queryByTestId('pc-undo-done')).toBeNull();
      expect(screen.queryByTestId('pc-undo-working')).toBeNull();
    }, 30000);

    it('"nothing to put back" says so in words', async () => {
      backend.putBackError = coded(409, 'nothingToPutBack', 'There is no earlier PC of this dataset to put back.');
      const result = await applyTwo();
      pressPutBack(result, 'scan_b');
      await waitFor(() => expect(within(lineOf(result, 'scan_b')).getByRole('status').textContent)
        .toBe('This PC can no longer be put back: a file was loaded since.'));
    }, 30000);

    it('the same line in the question after a search: the answer shows, the button is not stuck', async () => {
      backend.versionAnswer = {
        calibration: { id: 'sem', name: 'SEM 2026-10-14', version: 2, version_created: '2026-11-01T10:00:00+00:00', pc: FOUND },
        applied: [{ dataset: 'scan_b', status: 'applied', pc: FOUND, version: 2, previous: { pc: OLD, had_pc: true }, pc_map_action: 'none' }],
      };
      backend.files.scan_b = { ...fileOf('scan_b'), before: { pc: OLD, had_pc: true } };
      await runFind(REFINE_ONE);
      const q = await screen.findByTestId('follower-question');
      await waitFor(() => expect(within(q).getByRole('button', { name: 'New version for this file too' }).disabled).toBe(false));
      fireEvent.click(within(q).getByRole('button', { name: 'New version for this file too' }));
      const answered = await screen.findByTestId('question-answered');
      fireEvent.click(within(answered).getByRole('button', { name: 'Put it back' }));
      fireEvent.click(screen.getByRole('button', { name: 'Yes, put it back' }));
      const done = await within(answered).findByTestId('pc-undo-done');
      expect(done.textContent).toContain('Previous PC put back for scan_b: 0.5019, 0.3301, 0.8420.');
    }, 30000);
  });
});

// one follower: scan_b is still on version 1 of "SEM 2026-10-14"
const REFINE_ONE = {
  ...REFINE, calibration: { ...REFINE.calibration, followers: [{ dataset: 'scan_b', per_pixel: false, map_kind: null }] },
};

describe('The question after a search', () => {
  it('5: the confirmation of "Only this file" says "stays" for one file and "stay" for two', async () => {
    await runFind(REFINE_ONE);
    fireEvent.click(within(await screen.findByTestId('follower-question')).getByRole('button', { name: 'Only this file' }));
    expect((await screen.findByTestId('question-answered')).textContent)
      .toBe('Only scan_a uses the new PC. scan_b stays on version 1. The calibration is not changed.');
    cleanup();
    await runFind(REFINE);
    fireEvent.click(within(await screen.findByTestId('follower-question')).getByRole('button', { name: 'Only this file' }));
    expect((await screen.findByTestId('question-answered')).textContent)
      .toBe('Only scan_a uses the new PC. scan_b and scan_c stay on version 1. The calibration is not changed.');
  }, 40000);

  it('5: the singular sentence exists in all four languages', () => {
    const one = { count: 1, file: 'scan_a', files: 'scan_b', version: 1 };
    const two = { count: 2, file: 'scan_a', files: 'scan_b, scan_c', version: 1 };
    const t = (lng) => i18n.getFixedT(lng, 'pcrefinement');
    expect(t('de')('question.onlyDone', one)).toBe('Nur scan_a verwendet das neue PC. scan_b bleibt auf Version 1. Die Kalibrierung ist nicht verändert.');
    expect(t('de')('question.onlyDone', two)).toBe('Nur scan_a verwendet das neue PC. scan_b, scan_c bleiben auf Version 1. Die Kalibrierung ist nicht verändert.');
    for (const lng of ['ja', 'zh']) {
      expect(t(lng)('question.onlyDone', one)).toContain('scan_b');
      expect(t(lng)('question.onlyDone', one)).not.toContain('question.');
    }
  });

  it('6: a change under 0.05 pixels says "under 0.1 pixels", never "about 0.0"', async () => {
    // version 1 holds [0.5019, 0.3301, 0.8420]; z moves by +0.0003: 0.0003 * 128 rows = 0.0384 pixels (< 0.05)
    await runFind({ ...REFINE_ONE, mean_pc: [0.5019, 0.3301, 0.8423] });
    const q = await screen.findByTestId('follower-question');
    await waitFor(() => expect(within(q).getByTestId('question-change').textContent).toBe(
      'Change: x 0.0000, y 0.0000, z +0.0003 (under 0.1 pixels). Fit contrast before and after: 0.7, 0.8 (of 1, higher is better)'));
    expect(text(q)).not.toContain('about 0.0');
  }, 20000);

  it('6: a change of 0.0512 pixels (z +0.0004 * 128) is still "about 0.1 pixels"', async () => {
    await runFind({ ...REFINE_ONE, mean_pc: [0.5019, 0.3301, 0.8424] });
    const q = await screen.findByTestId('follower-question');
    await waitFor(() => expect(within(q).getByTestId('question-change').textContent)
      .toMatch(/^Change: x 0\.0000, y 0\.0000, z \+0\.0004 \(about 0\.1 pixels\)\./));
  }, 20000);

  it('6: the offer card says the same for a PC that is almost the one in use', async () => {
    const offer = (pc) => ({
      id: 'sem', name: 'SEM 2026-10-14', version: 2, pc, via_member: false, match: true, differences: [], unknown: [], unverified: false,
      checked: ALL6, settings: { file: SETTINGS, calibration: SETTINGS }, version_created: '2026-10-20T09:00:00+00:00',
      source_name: 'scan_x.h5oina', file_date: null, quality: null, replaces: { pc: OLD, kind: 'file' },
    });
    backend.suggestions = [offer([0.5019, 0.3301, 0.8423])];
    render(<PCRefinement />);
    expect((await screen.findByTestId('calibration-replaces')).textContent).toBe('Replaces the PC from the file. Change: under 0.1 pixels');
    cleanup();
    backend.suggestions = [offer([0.5019, 0.3301, 0.8424])];
    render(<PCRefinement />);
    expect((await screen.findByTestId('calibration-replaces')).textContent).toBe('Replaces the PC from the file. Change: about 0.1 pixels');
  }, 30000);

  it('6: the tiny sentence exists in all four languages', () => {
    const t = (lng) => i18n.getFixedT(lng, 'pcrefinement');
    expect(t('de')('calibration.replaces.changeTiny')).toBe('Änderung: unter 0.1 Pixel');
    expect(t('de')('question.changeTiny', { x: '0.0000', y: '0.0000', z: '+0.0003' })).toBe('Änderung: x 0.0000, y 0.0000, z +0.0003 (unter 0.1 Pixel).');
    for (const lng of ['ja', 'zh']) {
      expect(t(lng)('calibration.replaces.changeTiny')).toContain('0.1');
      expect(t(lng)('question.changeTiny', { x: 'X', y: 'Y', z: 'Z' })).toContain('0.1');
    }
  });
});

describe('A name the backend does not know', () => {
  const OFFER_NOW = {
    id: 'sem', name: 'SEM 2026-10-14', version: 2, pc: FOUND, via_member: false, match: true, differences: [], unknown: [], unverified: false,
    checked: ALL6, settings: { file: SETTINGS, calibration: SETTINGS }, version_created: '2026-10-20T09:00:00+00:00',
    source_name: 'scan_x.h5oina', file_date: null, quality: null, replaces: { pc: OLD, kind: 'file' },
  };

  it('the page ignores an answer that says unknown_dataset: no offer, no error', async () => {
    backend.suggestAnswer = { dataset: 'scan_a', target: 'scan_a', suggestions: [OFFER_NOW], current: null, unverified: false, unknown_dataset: true };
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    await waitFor(() => expect(backend.suggestCalls).toContain('scan_a'));
    await new Promise((r) => setTimeout(r, 200));
    expect(screen.queryByTestId('calibration-offer')).toBeNull();
    expect(screen.queryByRole('alert')).toBeNull();
  }, 20000);

  it('control: the same answer without the mark is an offer', async () => {
    backend.suggestAnswer = { dataset: 'scan_a', target: 'scan_a', suggestions: [OFFER_NOW], current: null, unverified: false };
    render(<PCRefinement />);
    expect(await screen.findByTestId('calibration-offer')).toBeTruthy();
  }, 20000);

  it('an empty answer with the mark does not ask a second time for "all"', async () => {
    backend.suggestAnswer = { dataset: 'scan_a', target: 'scan_a', suggestions: [], current: null, unverified: true, unknown_dataset: true };
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    await waitFor(() => expect(backend.suggestCalls).toContain('scan_a'));
    await new Promise((r) => setTimeout(r, 200));
    expect(backend.suggestCalls.filter((d) => d === 'scan_a')).toHaveLength(1);
  }, 20000);
});
