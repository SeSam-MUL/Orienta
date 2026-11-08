// What the user typed on the Grain, Deformation and RX tabs and the backend has not been told yet is the current
// choice, and a status answer of the same map must not take it back (the page asks on every tab change). The store keeps
// a flag per group: `reconEdited` (the five reconstruction parameters), `deformationEdited` (KAM order, KAM threshold,
// HAGB limit) and `rxCriteriaEdited` (the four criteria and the RX phase). A map that is replaced puts everything back.
import { describe, it, expect, beforeEach } from 'vitest';
import useResultStore from './useResultStore';

const state = () => useResultStore.getState();
const PHASES = [{ id: 1, name: 'Aluminium' }, { id: 2, name: 'Si' }];

/** GET /status as routes/analysis.py writes it, for a map with grains. */
const status = (extra = {}) => ({
  loaded: true, shape: [40, 60], has_grains: true, phase_options: PHASES, dataset_token: 'a1',
  edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null,
  recon_params: { threshold: 4, min_pixels: 3, min_intercept: null, alpha: 3.1, fill: true },
  ...extra,
});

beforeEach(() => {
  state().clearResults();
  state().applyBackendStatus(status());
});

describe('reconstruction parameters typed and not used yet', () => {
  it('a status answer of the same map does not take them back', () => {
    state().setGrainStatsSettings({ gbThreshold: '6', alpha: '1' });
    expect(state().reconEdited).toBe(true);
    state().applyBackendStatus(status());                                   // the page asks on every tab change
    expect(state().grainStatsSettings).toMatchObject({ gbThreshold: '6', alpha: '1' });
  });

  it('without anything typed the backend\'s values of the last reconstruction are adopted', () => {
    state().applyBackendStatus(status({ recon_params: { threshold: 5, min_pixels: 7, min_intercept: null, alpha: 2, fill: false } }));
    expect(state().grainStatsSettings).toMatchObject({ gbThreshold: '5', minPixels: '7', alpha: '2', fill: false });
    expect(state().reconEdited).toBe(false);
  });

  it('a change of the statistics settings (edge rule, basis, phase) is not a typed parameter', () => {
    state().setGrainStatsSettings({ edgeRule: 'keep_all' });
    expect(state().reconEdited).toBe(false);
  });

  it('the end of a reconstruction, the chain\'s grains and a swapped map clear the flag', () => {
    state().setGrainStatsSettings({ gbThreshold: '6' });
    state().finishReconstruction({ stats: { count: 3 }, text: 'ok' });
    expect(state().reconEdited).toBe(false);
    state().setGrainStatsSettings({ gbThreshold: '7' });
    state().markGrainsReconstructed();
    expect(state().reconEdited).toBe(false);
    state().setGrainStatsSettings({ gbThreshold: '8' });
    state().applyBackendStatus(status({ dataset_token: 'b2', recon_params: null, has_grains: false }));
    expect(state().reconEdited).toBe(false);
  });
});

describe('Deformation parameters', () => {
  it('start at the values of the script', () => {
    expect(state().deformationParams).toEqual({ kamOrder: '2', kamThreshold: '7.0', hagbMin: '15.0' });
    expect(state().deformationEdited).toBe(false);
  });

  it('a status answer is adopted when nothing is pending, and a typed value is not taken back', () => {
    state().applyBackendStatus(status({ deformation_params: { kam_order: 3, kam_threshold_deg: 5, hagb_min_deg: 12 } }));
    expect(state().deformationParams).toEqual({ kamOrder: '3', kamThreshold: '5', hagbMin: '12' });
    state().setDeformationParams({ kamOrder: '4' });
    expect(state().deformationEdited).toBe(true);
    state().applyBackendStatus(status({ deformation_params: { kam_order: 3, kam_threshold_deg: 5, hagb_min_deg: 12 } }));
    expect(state().deformationParams.kamOrder).toBe('4');
  });

  it('an answer of the run clears the pending flag and keeps a typed "7" for 7.0', () => {
    state().setDeformationParams({ kamThreshold: '7' });
    state().adoptDeformationUsed({ kam_order: 2, kam_threshold_deg: 7, hagb_min_deg: 15 });
    expect(state().deformationEdited).toBe(false);
    expect(state().deformationParams).toEqual({ kamOrder: '2', kamThreshold: '7', hagbMin: '15.0' });
  });

  it('a swapped map puts them back at the defaults', () => {
    state().setDeformationParams({ kamOrder: '5' });
    state().applyBackendStatus(status({ dataset_token: 'b2' }));
    expect(state().deformationParams.kamOrder).toBe('2');
    expect(state().deformationEdited).toBe(false);
  });
});

describe('the RX phase', () => {
  it('is "most_measured" until the backend says otherwise; a pick is pending with the criteria', () => {
    expect(state().rxPhase).toBe('most_measured');
    state().applyBackendStatus(status({ rx_phase: 2 }));
    expect(state().rxPhase).toBe(2);
    state().setRxPhase(1);
    expect(state().rxCriteriaEdited).toBe(true);
    state().applyBackendStatus(status({ rx_phase: 2 }));
    expect(state().rxPhase).toBe(1);                         // the pick has not been sent: it stays
  });

  it('the answer of Classify makes the pick no longer pending, and the pick stays what it was ("most_measured" is not its id)', () => {
    state().setRxPhase('most_measured');
    state().adoptRxCriteriaUsed({ rx_gos_threshold: 1.25, rx_phase: { id: 1, name: 'Aluminium' } });
    expect(state().rxCriteriaEdited).toBe(false);
    expect(state().rxPhase).toBe('most_measured');
  });

  it('a phase id the map does not have is not taken from a status answer; a swapped map goes back to the default', () => {
    state().applyBackendStatus(status({ rx_phase: 9 }));
    expect(state().rxPhase).toBe('most_measured');
    state().applyBackendStatus(status({ rx_phase: 2 }));
    state().applyBackendStatus(status({ dataset_token: 'b2', rx_phase: undefined }));
    expect(state().rxPhase).toBe('most_measured');
  });
});

describe('mapEpoch: the results the tabs keep belong to one map', () => {
  it('moves when the map is replaced or lost, and not for a status answer of the same map', () => {
    const e0 = state().mapEpoch;
    state().applyBackendStatus(status());
    expect(state().mapEpoch).toBe(e0);
    state().applyBackendStatus(status({ dataset_token: 'b2' }));
    expect(state().mapEpoch).toBe(e0 + 1);
    state().resetGrainStatsSettings();
    expect(state().mapEpoch).toBe(e0 + 2);
    state().applyBackendStatus({ loaded: false });
    expect(state().mapEpoch).toBe(e0 + 3);
  });
});
