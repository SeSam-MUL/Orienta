import { describe, it, expect, beforeEach } from 'vitest';
import useResultStore from './useResultStore';

describe('useResultStore: grains and grain-size settings', () => {
  beforeEach(() => useResultStore.getState().clearResults());

  it('hasGrains and the version move in ONE update, so a grain layer is requested once', () => {
    const seen = [];
    const unsubscribe = useResultStore.subscribe((s) => seen.push({ hasGrains: s.hasGrains, v: s.grainsVersion }));
    useResultStore.getState().markGrainsReconstructed();
    unsubscribe();
    expect(seen).toEqual([{ hasGrains: true, v: 1 }]);
  });

  it('bumpGrainMaps raises the version without touching hasGrains', () => {
    useResultStore.getState().bumpGrainMaps();
    expect(useResultStore.getState()).toMatchObject({ hasGrains: false, grainsVersion: 1 });
  });

  it('the settings are one object; a patch changes only its field; reset and clear restore the defaults', () => {
    const { setGrainStatsSettings, resetGrainStatsSettings, clearResults } = useResultStore.getState();
    expect(useResultStore.getState().grainStatsSettings).toMatchObject({ edgeRule: 'matlab_script', ecdBasis: 'cell_area' });
    setGrainStatsSettings({ ecdBasis: 'measured_pixels' });
    expect(useResultStore.getState().grainStatsSettings).toMatchObject({ edgeRule: 'matlab_script', ecdBasis: 'measured_pixels' });
    setGrainStatsSettings({ edgeRule: 'mtex' });
    expect(useResultStore.getState().grainStatsSettings).toMatchObject({ edgeRule: 'mtex', ecdBasis: 'measured_pixels' });
    resetGrainStatsSettings();
    expect(useResultStore.getState().grainStatsSettings).toMatchObject({ edgeRule: 'matlab_script', ecdBasis: 'cell_area' });
    setGrainStatsSettings({ edgeRule: 'keep_all' });
    clearResults();
    expect(useResultStore.getState().grainStatsSettings.edgeRule).toBe('matlab_script');
  });

  it('the defaults object is never shared with the state (a patch cannot change the defaults)', () => {
    useResultStore.getState().setGrainStatsSettings({ edgeRule: 'keep_all' });
    useResultStore.getState().resetGrainStatsSettings();
    expect(useResultStore.getState().grainStatsSettings.edgeRule).toBe('matlab_script');
  });

  it('the settings object also holds the reconstruction parameters, with the Grain tab defaults', () => {
    expect(useResultStore.getState().grainStatsSettings).toEqual({
      gbThreshold: '4.0', minPixels: '3', minIntercept: '1.5', alpha: '3.1', fill: true,
      edgeRule: 'matlab_script', ecdBasis: 'cell_area', statsPhase: null,
    });
  });

  it('a new map resets the statistics settings but keeps the reconstruction parameters', () => {
    const { setGrainStatsSettings, resetGrainStatsSettings } = useResultStore.getState();
    setGrainStatsSettings({ gbThreshold: '6.5', fill: false, edgeRule: 'mtex' });
    resetGrainStatsSettings();
    expect(useResultStore.getState().grainStatsSettings).toMatchObject({
      gbThreshold: '6.5', fill: false, edgeRule: 'matlab_script',
    });
  });

  it('loading a map clears the statistics on screen and invalidates answers still on their way', () => {
    const { setGrainStats, resetGrainStatsSettings } = useResultStore.getState();
    setGrainStats({ count: 5, meanEcd: 1 });
    const before = useResultStore.getState().grainSizeEpoch;
    resetGrainStatsSettings();
    expect(useResultStore.getState().grainStats).toBeNull();
    expect(useResultStore.getState().grainSizeEpoch).toBe(before + 1);
  });

  it('a reconstruction that begins clears the statistics and raises the epoch', () => {
    const { setGrainStats, beginReconstruction } = useResultStore.getState();
    setGrainStats({ count: 5 });
    const before = useResultStore.getState().grainSizeEpoch;
    beginReconstruction();
    expect(useResultStore.getState().grainStats).toBeNull();
    expect(useResultStore.getState().grainSizeEpoch).toBe(before + 1);
  });
});
