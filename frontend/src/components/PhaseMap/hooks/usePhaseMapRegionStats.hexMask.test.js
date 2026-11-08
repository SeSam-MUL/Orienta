// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePhaseMapRegionStats } from './usePhaseMapRegionStats';

vi.mock('../../../services/api', () => ({
  phaseMapApi: { regionStats: vi.fn() },
}));
import { phaseMapApi } from '../../../services/api';

beforeEach(() => { vi.clearAllMocks(); });

const BOX = { row0: 1, col0: 0, rows: 2, cols: 2 };

/** An answer the test releases itself. */
function held(data) {
  let release;
  const promise = new Promise((resolve) => { release = () => resolve({ data }); });
  return { promise, release };
}

describe('usePhaseMapRegionStats.fetchMasked', () => {
  it('asks for the box with its mask, as booleans', async () => {
    phaseMapApi.regionStats.mockResolvedValue({ data: { n_pixels: 3, phases: [], scalars: {} } });
    const { result } = renderHook(() => usePhaseMapRegionStats());
    await act(async () => { await result.current.fetchMasked({ bbox: BOX, mask: Uint8Array.from([1, 1, 0, 1]) }); });
    expect(phaseMapApi.regionStats).toHaveBeenCalledWith({
      row_start: 1, col_start: 0, row_end: 2, col_end: 1, mask: [true, true, false, true],
    });
    expect(result.current.data).toEqual({ n_pixels: 3, phases: [], scalars: {} });
  });

  it('sends no mask when every cell of the box is selected', async () => {
    phaseMapApi.regionStats.mockResolvedValue({ data: { n_pixels: 4, phases: [], scalars: {} } });
    const { result } = renderHook(() => usePhaseMapRegionStats());
    await act(async () => { await result.current.fetchMasked({ bbox: BOX, mask: null }); });
    expect(phaseMapApi.regionStats).toHaveBeenCalledWith({ row_start: 1, col_start: 0, row_end: 2, col_end: 1 });
  });

  it('remembers the region after an error, like fetchStats does', async () => {
    phaseMapApi.regionStats.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => usePhaseMapRegionStats());
    await act(async () => { await result.current.fetchMasked({ bbox: BOX, mask: null }); });
    expect(result.current.error).toMatch(/boom/);
    expect(result.current.region).toEqual({ start: { row: 1, col: 0 }, end: { row: 2, col: 1 } });
  });

  it('is the old request, with the two corners as given, from fetchStats', async () => {
    phaseMapApi.regionStats.mockResolvedValue({ data: { n_pixels: 6, phases: [], scalars: {} } });
    const { result } = renderHook(() => usePhaseMapRegionStats());
    await act(async () => { await result.current.fetchStats({ row: 3, col: 4 }, { row: 5, col: 6 }); });
    expect(phaseMapApi.regionStats).toHaveBeenCalledWith({ row_start: 3, col_start: 4, row_end: 5, col_end: 6 });
  });
});

describe('an answer that does not belong to the latest request', () => {
  it('is dropped when a newer request was made, and the newer one is kept', async () => {
    const first = held({ n_pixels: 1 });
    const second = held({ n_pixels: 2 });
    phaseMapApi.regionStats.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const { result } = renderHook(() => usePhaseMapRegionStats());
    let p1;
    let p2;
    act(() => { p1 = result.current.fetchMasked({ bbox: BOX, mask: null }); });
    act(() => { p2 = result.current.fetchMasked({ bbox: { row0: 0, col0: 0, rows: 1, cols: 1 }, mask: null }); });
    await act(async () => { second.release(); await p2; });
    expect(result.current.data).toEqual({ n_pixels: 2 });
    expect(result.current.loading).toBe(false);
    await act(async () => { first.release(); await p1; });       // the slower, older answer lands last
    expect(result.current.data).toEqual({ n_pixels: 2 });
    expect(result.current.loading).toBe(false);
  });

  it('does not end the loading of the newer request when the older one lands', async () => {
    const first = held({ n_pixels: 1 });
    const second = held({ n_pixels: 2 });
    phaseMapApi.regionStats.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const { result } = renderHook(() => usePhaseMapRegionStats());
    let p1;
    act(() => { p1 = result.current.fetchMasked({ bbox: BOX, mask: null }); });
    act(() => { result.current.fetchMasked({ bbox: BOX, mask: null }); });
    await act(async () => { first.release(); await p1; });
    expect(result.current.loading).toBe(true);                    // the second is still on its way
    expect(result.current.data).toBeNull();
  });

  it('is dropped after clear(), and clear() ends the loading', async () => {
    const pending = held({ n_pixels: 9 });
    phaseMapApi.regionStats.mockReturnValueOnce(pending.promise);
    const { result } = renderHook(() => usePhaseMapRegionStats());
    let p;
    act(() => { p = result.current.fetchMasked({ bbox: BOX, mask: null }); });
    expect(result.current.loading).toBe(true);
    act(() => { result.current.clear(); });
    expect(result.current.loading).toBe(false);
    await act(async () => { pending.release(); await p; });
    expect(result.current.data).toBeNull();
    expect(result.current.region).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it('does not show the error of a request that was replaced', async () => {
    let fail;
    const old = new Promise((_, reject) => { fail = () => reject(new Error('late failure')); });
    phaseMapApi.regionStats.mockReturnValueOnce(old).mockResolvedValueOnce({ data: { n_pixels: 5 } });
    const { result } = renderHook(() => usePhaseMapRegionStats());
    let p1;
    act(() => { p1 = result.current.fetchMasked({ bbox: BOX, mask: null }); });
    await act(async () => { await result.current.fetchMasked({ bbox: BOX, mask: null }); });
    await act(async () => { fail(); await p1; });
    expect(result.current.error).toBeNull();
    expect(result.current.data).toEqual({ n_pixels: 5 });
  });
});
