// The Batch tab's own statistics settings in the store.
import { describe, it, expect, beforeEach } from 'vitest';
import useResultStore from './useResultStore';

const state = () => useResultStore.getState();

beforeEach(() => { state().clearResults(); });

describe('batchStatsSettings', () => {
  it('is empty until the Batch tab is used, then starts as the Grain tab (a single phase becomes all phases)', () => {
    expect(state().batchStatsSettings).toBeNull();
    state().setGrainStatsSettings({ edgeRule: 'mtex', ecdBasis: 'measured_pixels', statsPhase: 1 });
    state().ensureBatchStatsSettings();
    expect(state().batchStatsSettings).toEqual({ edgeRule: 'mtex', ecdBasis: 'measured_pixels', statsPhase: null });
  });

  it('keeps "most measured" from the Grain tab', () => {
    state().setGrainStatsSettings({ statsPhase: 'most_measured' });
    state().ensureBatchStatsSettings();
    expect(state().batchStatsSettings.statsPhase).toBe('most_measured');
  });

  it('is copied once: later Grain-tab changes do not move it, and ensure does not overwrite a change', () => {
    state().ensureBatchStatsSettings();
    state().setBatchStatsSettings({ edgeRule: 'keep_all' });
    state().setGrainStatsSettings({ edgeRule: 'outer_row', ecdBasis: 'measured_pixels' });
    state().ensureBatchStatsSettings();
    expect(state().batchStatsSettings).toEqual({ edgeRule: 'keep_all', ecdBasis: 'cell_area', statsPhase: null });
    expect(state().grainStatsSettings.edgeRule).toBe('outer_row');
  });

  it('a first change starts from the Grain tab', () => {
    state().setGrainStatsSettings({ ecdBasis: 'measured_pixels' });
    state().setBatchStatsSettings({ statsPhase: 'most_measured' });
    expect(state().batchStatsSettings).toEqual({ edgeRule: 'matlab_script', ecdBasis: 'measured_pixels', statsPhase: 'most_measured' });
  });
});
