// @vitest-environment jsdom
/**
 * Re-classifying discards hand-given region NAMES and keeps hand-painted
 * PIXELS. The asymmetry is deliberate on the backend (`assign_region_phase`
 * does not lock, `assign_mask` does) and three users independently called it
 * the worst trap on this page — because it lived only in a tooltip.
 *
 * The guard lives in `usePhaseMap`, not in the controls, and that placement is
 * the point: three different buttons reach a re-classification (the rail, the
 * Apply in the region-definition editor, the Apply in the rules editor) and
 * they all call `handleAutoClassify`. A guard on only the first is not a
 * guard.
 *
 * The count is the part that makes it actionable — "6 names" is a decision,
 * "some names" is a shrug — so what counts is pinned here too: a region the
 * CLASSIFIER named is not work the user is about to lose.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));
vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: '/tmp/x.h5oina' }),
}));

const MAP = {
  loaded: true,
  n_rows: 4,
  n_cols: 4,
  summary: [],
  all_phases: [],
  regions: [
    // Named by the classifier — not work the user did.
    { region_id: 0, phase_index: 0, n_pixels: 10 },
    { region_id: 1, phase_index: -1, n_pixels: 4 },
    { region_id: 2, phase_index: -1, n_pixels: 2 },
  ],
};
const NAMED = {
  ...MAP,
  regions: MAP.regions.map((r) => (r.region_id === 2 ? { ...r, phase_index: 1 } : r)),
};

const autoClassify = vi.fn();
const assignRegionPhase = vi.fn();
vi.mock('../../services/api', () => ({
  edsApi: {
    autoClassify: (...a) => autoClassify(...a),
    assignRegionPhase: (...a) => assignRegionPhase(...a),
    getPhaseMap: vi.fn(() => Promise.resolve({ data: MAP })),
    clearPhaseMap: vi.fn(() => Promise.resolve({ data: {} })),
    setPhaseColors: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    cifPhases: vi.fn(() => Promise.resolve({ data: { phases: [] } })),
  },
}));

import { usePhaseMap, countHandNamedRegions } from './PhaseMapPanel';

const flush = () => act(async () => { await Promise.resolve(); await Promise.resolve(); });

beforeEach(() => {
  vi.clearAllMocks();
  autoClassify.mockResolvedValue({ data: MAP });
  assignRegionPhase.mockResolvedValue({ data: NAMED });
});

describe('countHandNamedRegions', () => {
  it('counts only regions the USER named and that still carry a phase', () => {
    // Region 0 has a phase from the classifier and is not in the set.
    expect(countHandNamedRegions(NAMED.regions, new Set([2]))).toBe(1);
    expect(countHandNamedRegions(NAMED.regions, new Set())).toBe(0);
    // A region that was named and then cleared back to unclassified is not
    // a name about to be lost.
    expect(countHandNamedRegions(MAP.regions, new Set([2]))).toBe(0);
    // A region that no longer exists cannot be lost either.
    expect(countHandNamedRegions(NAMED.regions, new Set([99]))).toBe(0);
  });

  it('is cheap and safe on nothing', () => {
    expect(countHandNamedRegions(undefined, undefined)).toBe(0);
    expect(countHandNamedRegions(null, new Set([1]))).toBe(0);
  });
});

describe('the re-classify guard', () => {
  it('does not ask when there is nothing to lose', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    await act(async () => { await result.current.handleAutoClassify(); });
    expect(autoClassify).toHaveBeenCalledTimes(1);
    expect(result.current.pendingReclassify).toBe(false);
  });

  it('asks first once a region has been named by hand, and does NOT run', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    await act(async () => { await result.current.handleAssignRegionPhase(2, 1); });
    expect(result.current.handNamedCount).toBe(1);

    await act(async () => { await result.current.handleAutoClassify(); });
    expect(result.current.pendingReclassify).toBe(true);
    expect(autoClassify).not.toHaveBeenCalled();
  });

  it('confirming runs it; cancelling leaves the map alone', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    await act(async () => { await result.current.handleAssignRegionPhase(2, 1); });
    await act(async () => { await result.current.handleAutoClassify(); });

    await act(async () => { result.current.cancelReclassify(); });
    expect(result.current.pendingReclassify).toBe(false);
    expect(autoClassify).not.toHaveBeenCalled();

    await act(async () => { await result.current.handleAutoClassify(); });
    await act(async () => { await result.current.confirmReclassify(); });
    expect(autoClassify).toHaveBeenCalledTimes(1);
    expect(result.current.pendingReclassify).toBe(false);
    // The regions were rebuilt, so the recorded names describe groups that
    // no longer exist — the next run must not ask about them again.
    expect(result.current.handNamedCount).toBe(0);
  });

  it('clearing a region back to unclassified is not a name worth guarding', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    await act(async () => { await result.current.handleAssignRegionPhase(2, 1); });
    assignRegionPhase.mockResolvedValue({ data: MAP });
    await act(async () => { await result.current.handleAssignRegionPhase(2, -1); });
    expect(result.current.handNamedCount).toBe(0);

    await act(async () => { await result.current.handleAutoClassify(); });
    expect(autoClassify).toHaveBeenCalledTimes(1);
  });

  it('a failed assign records nothing', async () => {
    const { result } = renderHook(() => usePhaseMap({}));
    await flush();
    assignRegionPhase.mockRejectedValue({ response: { data: { detail: 'nope' } } });
    await act(async () => { await result.current.handleAssignRegionPhase(2, 1); });
    expect(result.current.handNamedCount).toBe(0);
  });
});
