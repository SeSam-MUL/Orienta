// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePhaseMapRegionStats } from './usePhaseMapRegionStats';

vi.mock('../../../services/api', () => ({
  phaseMapApi: { regionStats: vi.fn() },
}));
import { phaseMapApi } from '../../../services/api';

beforeEach(() => { vi.clearAllMocks(); });

describe('usePhaseMapRegionStats', () => {
  it('fetches stats and stores data', async () => {
    phaseMapApi.regionStats.mockResolvedValue({ data: { n_pixels: 100, phases: [], scalars: {} } });
    const { result } = renderHook(() => usePhaseMapRegionStats());
    await act(async () => { await result.current.fetchStats({ row: 0, col: 0 }, { row: 10, col: 10 }); });
    expect(result.current.data).toEqual({ n_pixels: 100, phases: [], scalars: {} });
    expect(phaseMapApi.regionStats).toHaveBeenCalledWith({ row_start: 0, col_start: 0, row_end: 10, col_end: 10 });
  });

  it('captures errors', async () => {
    phaseMapApi.regionStats.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => usePhaseMapRegionStats());
    await act(async () => { await result.current.fetchStats({ row: 0, col: 0 }, { row: 1, col: 1 }); });
    expect(result.current.error).toMatch(/boom/);
    expect(result.current.data).toBeNull();
  });

  it('remembers region after error for visual feedback', async () => {
    phaseMapApi.regionStats.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => usePhaseMapRegionStats());
    await act(async () => { await result.current.fetchStats({ row: 5, col: 5 }, { row: 9, col: 9 }); });
    expect(result.current.region).toEqual({ start: { row: 5, col: 5 }, end: { row: 9, col: 9 } });
  });

  it('clear() resets data + region', async () => {
    phaseMapApi.regionStats.mockResolvedValue({ data: { n_pixels: 1, phases: [], scalars: {} } });
    const { result } = renderHook(() => usePhaseMapRegionStats());
    await act(async () => { await result.current.fetchStats({ row: 0, col: 0 }, { row: 1, col: 1 }); });
    act(() => result.current.clear());
    expect(result.current.data).toBeNull();
    expect(result.current.region).toBeNull();
  });
});
