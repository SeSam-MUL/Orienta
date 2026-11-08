import { describe, it, expect, beforeEach } from 'vitest';
import useResultStore from './useResultStore';

// GET /api/analysis/status as the backend writes it for a loaded map (routes/analysis.py analysis_status)
const status = (extra = {}) => ({
  loaded: true, shape: [40, 60], has_grains: false, phases: ['Al (Fm-3m)', 'Si (Fd-3m)'],
  phase_options: [{ id: 1, name: 'Al' }, { id: 2, name: 'Si' }], step_size: 0.5,
  edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null, recon_params: null,
  dataset_token: 'a1', reconstruct: null,
  ...extra,
});
const withGrains = (extra = {}) => status({
  has_grains: true, edge_rule: 'mtex', ecd_basis: 'measured_pixels', stats_phase: 2,
  recon_params: { threshold: 6.5, min_pixels: 5, min_intercept: 2, alpha: 3.1, fill: false },
  ...extra,
});

describe('useResultStore.applyBackendStatus', () => {
  beforeEach(() => useResultStore.getState().clearResults());
  const apply = (d) => useResultStore.getState().applyBackendStatus(d);
  const state = () => useResultStore.getState();

  it('a map with grains that were made while the page was away: flag, settings, parameters, phases, one new version', () => {
    apply(status());                                  // the page saw the map without grains
    const before = state();
    apply(withGrains());                              // the reconstruction finished meanwhile
    expect(state()).toMatchObject({
      analysisLoaded: true, hasGrains: true, analysisShape: [40, 60],
      phaseOptions: [{ id: 1, name: 'Al' }, { id: 2, name: 'Si' }],
      grainsVersion: before.grainsVersion + 1, grainSizeEpoch: before.grainSizeEpoch + 1,
    });
    expect(state().grainStatsSettings).toMatchObject({
      edgeRule: 'mtex', ecdBasis: 'measured_pixels', statsPhase: 2,
      gbThreshold: '6.5', minPixels: '5', minIntercept: '2', alpha: '3.1', fill: false,
    });
  });

  it('hasGrains, the version and the settings arrive in ONE update (a grain layer is requested once)', () => {
    apply(status());
    const seen = [];
    const unsubscribe = useResultStore.subscribe((s) => seen.push({ g: s.hasGrains, v: s.grainsVersion, rule: s.grainStatsSettings.edgeRule }));
    apply(withGrains());
    unsubscribe();
    expect(seen).toEqual([{ g: true, v: 1, rule: 'mtex' }]);
  });

  it('nothing changed: no new version, the statistics on screen stay, a pending answer is not dropped', () => {
    apply(withGrains());
    useResultStore.getState().setGrainStats({ count: 98, meanEcd: 12 });
    const { grainsVersion, grainSizeEpoch } = state();
    apply(withGrains());
    expect(state().grainsVersion).toBe(grainsVersion);
    expect(state().grainSizeEpoch).toBe(grainSizeEpoch);
    expect(state().grainStats).toEqual({ count: 98, meanEcd: 12 });
  });

  it('a map loaded elsewhere has no grains: the flag goes, the old statistics go, answers on their way are dropped', () => {
    apply(withGrains());
    useResultStore.getState().setGrainStats({ count: 98, meanEcd: 12 });
    const before = state();
    apply(status({ shape: [40, 60], phase_options: [{ id: 5, name: 'Fe' }] }));
    expect(state()).toMatchObject({
      hasGrains: false, grainStats: null, phaseOptions: [{ id: 5, name: 'Fe' }],
      grainsVersion: before.grainsVersion + 1, grainSizeEpoch: before.grainSizeEpoch + 1,
    });
  });

  it('a map with grains replaced by another map with grains (other shape): maps are fetched again, statistics go', () => {
    apply(withGrains());
    useResultStore.getState().setGrainStats({ count: 98, meanEcd: 12 });
    const before = state();
    apply(withGrains({ shape: [10, 10] }));
    expect(state()).toMatchObject({
      hasGrains: true, grainStats: null, analysisShape: [10, 10],
      grainsVersion: before.grainsVersion + 1, grainSizeEpoch: before.grainSizeEpoch + 1,
    });
  });

  it('the first answer of a page that knew nothing is not a change of map', () => {
    apply(status());
    expect(state()).toMatchObject({ analysisLoaded: true, hasGrains: false, grainsVersion: 0, grainSizeEpoch: 0 });
  });

  it('a phase list that the answer does not carry leaves the known one', () => {
    apply(status());
    const { phase_options: _omitted, ...without } = status();
    apply(without);
    expect(state().phaseOptions).toHaveLength(2);
  });

  it('clearResults empties the phase list and the busy flag of the Grain tab', () => {
    apply(status());
    useResultStore.getState().setGrainReconBusy(true);
    useResultStore.getState().clearResults();
    expect(state()).toMatchObject({ phaseOptions: [], grainReconBusy: false });
  });

  it('setAnalysis (a load) takes the phase list of the answer', () => {
    useResultStore.getState().setAnalysis({ shape: [3, 3], has_grains: false, phase_options: [{ id: 1, name: 'Al' }] });
    expect(state().phaseOptions).toEqual([{ id: 1, name: 'Al' }]);
  });
});

describe('useResultStore.applyBackendStatus: a lost or swapped map, a running reconstruction', () => {
  beforeEach(() => useResultStore.getState().clearResults());
  const apply = (d) => useResultStore.getState().applyBackendStatus(d);
  const state = () => useResultStore.getState();

  it('"loaded: false" after a map was known: what is bound to the map goes, version and epoch move; the batch stays', () => {
    apply(withGrains());
    state().setGrainStats({ count: 98, meanEcd: 12 });
    state().setReconTaskId('t1');
    state().setBatchRun({ taskId: 'b1', running: true, msg: 'done', result: { processed: ['/d/a.h5'] } });
    state().setExportRun({ msg: 'Exported', warnings: ['rx: x'] });
    const before = state();
    apply({ loaded: false });
    expect(state()).toMatchObject({
      analysisLoaded: false, analysisShape: null, hasGrains: false, grainStats: null, phaseOptions: [],
      datasetToken: null, reconTaskId: null, grainReconBusy: false,
      grainsVersion: before.grainsVersion + 1, grainSizeEpoch: before.grainSizeEpoch + 1,
    });
    // a batch works on a folder, not on the map: it runs on and its result stays
    expect(state().batchRun).toMatchObject({ taskId: 'b1', running: true, msg: 'done', result: { processed: ['/d/a.h5'] } });
    // the export belongs to the map that is gone
    expect(state().exportRun).toMatchObject({ msg: null, warnings: [] });
    expect(state().grainStatsSettings.statsPhase).toBeNull();
  });

  it('"loaded: false" with nothing known is no event (a batch that runs without a map is kept)', () => {
    state().setBatchRun({ taskId: 'b1', running: true });
    apply({ loaded: false });
    expect(state()).toMatchObject({ grainsVersion: 0, grainSizeEpoch: 0 });
    expect(state().batchRun).toMatchObject({ taskId: 'b1', running: true });
  });

  it('another dataset_token with the same shape, phases and grain state is a new map', () => {
    apply(withGrains());
    state().setGrainStats({ count: 98, meanEcd: 12 });
    const before = state();
    apply(withGrains({ dataset_token: 'b2', edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null, recon_params: null }));
    expect(state()).toMatchObject({
      datasetToken: 'b2', grainStats: null,
      grainsVersion: before.grainsVersion + 1, grainSizeEpoch: before.grainSizeEpoch + 1,
    });
    expect(state().grainStatsSettings).toMatchObject({ statsPhase: null, edgeRule: 'matlab_script', ecdBasis: 'cell_area' });
  });

  it('a swap resets the settings first, then takes what the new map\'s grains were made with', () => {
    apply(status());
    state().setGrainStatsSettings({ edgeRule: 'keep_all', statsPhase: 1 });
    apply(withGrains({ dataset_token: 'b2' }));
    expect(state().grainStatsSettings).toMatchObject({ edgeRule: 'mtex', ecdBasis: 'measured_pixels', statsPhase: 2 });
  });

  it('the same token: the first token is adopted, nothing else happens', () => {
    apply(status());
    expect(state()).toMatchObject({ datasetToken: 'a1', grainsVersion: 0, grainSizeEpoch: 0 });
    state().setGrainStatsSettings({ statsPhase: 2 });
    apply(status());
    expect(state()).toMatchObject({ grainsVersion: 0 });
    expect(state().grainStatsSettings.statsPhase).toBe(2);
  });

  it('a load here forgets the token, so the next answer is adopted and not taken for a swap', () => {
    apply(status());
    state().setAnalysis({ shape: [40, 60], has_grains: false, phase_options: [{ id: 1, name: 'Al' }, { id: 2, name: 'Si' }] });
    state().setGrainStatsSettings({ statsPhase: 2 });
    apply(status({ dataset_token: 'b2' }));
    expect(state()).toMatchObject({ datasetToken: 'b2', grainsVersion: 0 });
    expect(state().grainStatsSettings.statsPhase).toBe(2);
  });

  it('other phases (ids or names) with no token: the pick of a phase is reset, the map counts as new', () => {
    apply(status());
    state().setGrainStatsSettings({ statsPhase: 2 });
    apply(status({ phase_options: [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Ferrite' }] }));      // same ids, other name
    expect(state().grainStatsSettings.statsPhase).toBeNull();
    expect(state().grainsVersion).toBe(1);
  });

  it('a running reconstruction is recorded, and busy; one the backend no longer lists is over', () => {
    apply(status({ reconstruct: { task_id: 't1', status: 'running', progress: 0.3 } }));
    expect(state()).toMatchObject({ reconTaskId: 't1', grainReconBusy: true });
    apply(status({ reconstruct: null }));
    expect(state()).toMatchObject({ reconTaskId: null, grainReconBusy: false });
  });

  it('"reconstruct: null" does not release a busy flag whose task this page does not know (it is just starting)', () => {
    apply(status());
    state().setGrainReconBusy(true);
    apply(status({ reconstruct: null }));
    expect(state().grainReconBusy).toBe(true);
  });

  it('an answer without the field (older backend) leaves the task alone', () => {
    apply(status({ reconstruct: { task_id: 't1', status: 'running', progress: 0 } }));
    const { reconstruct: _omitted, ...old } = status();
    apply(old);
    expect(state().reconTaskId).toBe('t1');
  });

  it('setReconTaskId moves the busy flag with it; setBatchRun and setExportRun merge a patch or a function', () => {
    const epoch = state().grainSizeEpoch;
    state().setReconTaskId('t1');
    expect(state()).toMatchObject({ reconTaskId: 't1', grainReconBusy: true, grainSizeEpoch: epoch + 1 });
    state().setReconTaskId(null);
    expect(state()).toMatchObject({ reconTaskId: null, grainReconBusy: false });
    state().setBatchRun({ log: 'a' });
    state().setBatchRun((prev) => ({ log: prev.log + 'b' }));
    expect(state().batchRun.log).toBe('ab');
    state().setExportRun({ path: 'x' });
    state().setExportRun({ running: true });
    expect(state().exportRun).toMatchObject({ path: 'x', running: true });
    state().clearResults();
    expect(state().batchRun.log).toBe('');
    expect(state().exportRun.path).toBeNull();
  });
});
