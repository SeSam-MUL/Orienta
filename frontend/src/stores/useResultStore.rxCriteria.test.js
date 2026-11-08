// The RX criteria in the store: typed values, what the backend's answers do to them, and rxCriteria.js.
import { describe, it, expect, beforeEach } from 'vitest';
import useResultStore from './useResultStore';
import {
  DEFAULT_RX_CRITERIA, rxCriteriaArgs, rxCriteriaFromBackend, RX_RANGES, RX_FIELDS, RX_WIRE,
} from '../components/Analysis/rxCriteria';

const state = () => useResultStore.getState();
const STATUS = (extra = {}) => ({
  loaded: true, shape: [4, 5], has_grains: true, phase_options: [{ id: 1, name: 'Al' }], dataset_token: 'a',
  edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null, recon_params: null, reconstruct: null,
  rx_gos_threshold: 1.25, rx_gbc_fraction: 0.7, rx_gkam_threshold: 0.55, rx_min_radius_um: 0.4,
  ...extra,
});

beforeEach(() => {
  state().clearResults();
  state().applyBackendStatus(STATUS());          // a loaded map of token "a"
});

describe('rxCriteriaArgs', () => {
  it('turns the typed strings into numbers', () => {
    expect(rxCriteriaArgs(DEFAULT_RX_CRITERIA)).toEqual({ gosThreshold: 1.25, gbcFraction: 0.7, gkamThreshold: 0.55, minRadius: 0.4 });
  });
  it('is null for an empty field, text, or a number outside the backend range (inclusive ends)', () => {
    for (const field of RX_FIELDS) {
      const [low, high] = RX_RANGES[field];
      expect(rxCriteriaArgs({ ...DEFAULT_RX_CRITERIA, [field]: '' }), `${field} empty`).toBeNull();
      expect(rxCriteriaArgs({ ...DEFAULT_RX_CRITERIA, [field]: 'abc' }), `${field} text`).toBeNull();
      expect(rxCriteriaArgs({ ...DEFAULT_RX_CRITERIA, [field]: String(low - 0.01) }), `${field} below`).toBeNull();
      expect(rxCriteriaArgs({ ...DEFAULT_RX_CRITERIA, [field]: String(high + 0.01) }), `${field} above`).toBeNull();
      expect(rxCriteriaArgs({ ...DEFAULT_RX_CRITERIA, [field]: String(low) })?.[field], `${field} low end`).toBe(low);
      expect(rxCriteriaArgs({ ...DEFAULT_RX_CRITERIA, [field]: String(high) })?.[field], `${field} high end`).toBe(high);
    }
  });
});

describe('rxCriteriaFromBackend', () => {
  it('keeps the text that shows the same number, and the same object when nothing changes', () => {
    const typed = { ...DEFAULT_RX_CRITERIA, gbcFraction: '0.70' };
    expect(rxCriteriaFromBackend(typed, STATUS())).toBe(typed);
  });
  it('takes the numbers that differ, and ignores an answer without them', () => {
    expect(rxCriteriaFromBackend(DEFAULT_RX_CRITERIA, { rx_gos_threshold: 2, rx_min_radius_um: 0 }))
      .toEqual({ ...DEFAULT_RX_CRITERIA, gosThreshold: '2', minRadius: '0' });
    expect(rxCriteriaFromBackend(DEFAULT_RX_CRITERIA, { success: true })).toBe(DEFAULT_RX_CRITERIA);
    expect(rxCriteriaFromBackend(DEFAULT_RX_CRITERIA, null)).toBe(DEFAULT_RX_CRITERIA);
  });
  it('reads the four wire names', () => {
    expect(Object.values(RX_WIRE)).toEqual(['rx_gos_threshold', 'rx_gbc_fraction', 'rx_gkam_threshold', 'rx_min_radius_um']);
  });
});

describe('the store and the status of the backend', () => {
  it('typing marks the criteria as not yet sent', () => {
    expect(state().rxCriteriaEdited).toBe(false);
    state().setRxCriteria({ gosThreshold: '3' });
    expect(state().rxCriteria).toEqual({ ...DEFAULT_RX_CRITERIA, gosThreshold: '3' });
    expect(state().rxCriteriaEdited).toBe(true);
  });

  it('a status answer takes the values the backend holds when nothing was typed (a reloaded window)', () => {
    state().applyBackendStatus(STATUS({ rx_gos_threshold: 2, rx_min_radius_um: 1 }));
    expect(state().rxCriteria).toEqual({ ...DEFAULT_RX_CRITERIA, gosThreshold: '2', minRadius: '1' });
    expect(state().rxCriteriaEdited).toBe(false);
  });

  it('a status answer never takes back what was typed and not yet sent (the page asks on every tab change)', () => {
    state().setRxCriteria({ gosThreshold: '3' });
    state().applyBackendStatus(STATUS({ rx_gos_threshold: 1.25 }));
    expect(state().rxCriteria.gosThreshold).toBe('3');
    expect(state().rxCriteriaEdited).toBe(true);
  });

  it('what the answer of Classify says becomes the stored value and ends the pending state', () => {
    state().setRxCriteria({ gosThreshold: '3' });
    state().adoptRxCriteriaUsed({ rx_gos_threshold: 3, rx_gbc_fraction: 0.7, rx_gkam_threshold: 0.55, rx_min_radius_um: 0.4 });
    expect(state().rxCriteria).toEqual({ ...DEFAULT_RX_CRITERIA, gosThreshold: '3' });
    expect(state().rxCriteriaEdited).toBe(false);
    state().applyBackendStatus(STATUS({ rx_gos_threshold: 5 }));       // now the backend is the truth again
    expect(state().rxCriteria.gosThreshold).toBe('5');
  });

  it('a swapped map puts the criteria back at the defaults, typed or not (the backend resets them on a load)', () => {
    state().setRxCriteria({ gosThreshold: '3' });
    state().applyBackendStatus(STATUS({ dataset_token: 'b' }));
    expect(state().rxCriteria).toEqual(DEFAULT_RX_CRITERIA);
    expect(state().rxCriteriaEdited).toBe(false);
  });

  it('an unchanged status answer is no store update', () => {
    const before = state().rxCriteria;
    state().applyBackendStatus(STATUS());
    expect(state().rxCriteria).toBe(before);
  });
});
