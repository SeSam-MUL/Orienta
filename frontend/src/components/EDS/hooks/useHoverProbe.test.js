// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useHoverProbe } from './useHoverProbe';

vi.mock('../../../services/api', () => ({
  edsApi: { probe: vi.fn() },
}));
import { edsApi } from '../../../services/api';

beforeEach(() => { vi.clearAllMocks(); });

describe('useHoverProbe', () => {
  it('debounces and returns probe data once the timer fires', async () => {
    edsApi.probe.mockResolvedValue({ data: { bc: 178, elements: { Al: { at_pct: 65 } }, phase: null } });
    const { result } = renderHook(() => useHoverProbe({ displayMode: 'at_pct', debounceMs: 1 }));
    act(() => { result.current.requestProbe(5, 7); });
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    expect(edsApi.probe).toHaveBeenCalledTimes(1);
    expect(edsApi.probe).toHaveBeenCalledWith(5, 7, 'at_pct');
    expect(result.current.probe).toEqual(expect.objectContaining({ bc: 178 }));
    expect(result.current.error).toBeNull();
  });

  it('coalesces rapid requestProbe calls into a single fetch', async () => {
    edsApi.probe.mockResolvedValue({ data: { bc: 1, elements: {}, phase: null } });
    const { result } = renderHook(() => useHoverProbe({ displayMode: 'at_pct', debounceMs: 30 }));
    act(() => {
      result.current.requestProbe(1, 1);
      result.current.requestProbe(2, 2);
      result.current.requestProbe(3, 3);
    });
    await act(async () => { await new Promise(r => setTimeout(r, 80)); });
    expect(edsApi.probe).toHaveBeenCalledTimes(1);
    // The last requested pixel wins.
    expect(edsApi.probe).toHaveBeenCalledWith(3, 3, 'at_pct');
  });

  it('serves the cache on a repeat (row, col, displayMode) without refetching', async () => {
    edsApi.probe.mockResolvedValue({ data: { bc: 42, elements: {}, phase: null } });
    const { result } = renderHook(() => useHoverProbe({ displayMode: 'at_pct', debounceMs: 1 }));
    act(() => { result.current.requestProbe(5, 5); });
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    expect(edsApi.probe).toHaveBeenCalledTimes(1);
    act(() => { result.current.requestProbe(5, 5); });
    // Cache hit -> no new fetch, and the existing probe stays.
    expect(edsApi.probe).toHaveBeenCalledTimes(1);
    expect(result.current.probe).toEqual(expect.objectContaining({ bc: 42 }));
  });

  it('refetches when displayMode changes for the same pixel', async () => {
    edsApi.probe.mockResolvedValueOnce({ data: { bc: 1, elements: {}, phase: null } })
                .mockResolvedValueOnce({ data: { bc: 2, elements: {}, phase: null } });
    const { result, rerender } = renderHook(
      ({ mode }) => useHoverProbe({ displayMode: mode, debounceMs: 1 }),
      { initialProps: { mode: 'at_pct' } },
    );
    act(() => { result.current.requestProbe(5, 5); });
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    rerender({ mode: 'wt_pct' });
    act(() => { result.current.requestProbe(5, 5); });
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    expect(edsApi.probe).toHaveBeenCalledTimes(2);
    expect(edsApi.probe).toHaveBeenLastCalledWith(5, 5, 'wt_pct');
  });

  it('exposes a fetch error via result.error', async () => {
    edsApi.probe.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => useHoverProbe({ displayMode: 'at_pct', debounceMs: 1 }));
    act(() => { result.current.requestProbe(5, 5); });
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    expect(result.current.probe).toBeNull();
    expect(result.current.error).toMatch(/boom/);
  });

  it('clear() resets the probe state', async () => {
    edsApi.probe.mockResolvedValue({ data: { bc: 7, elements: {}, phase: null } });
    const { result } = renderHook(() => useHoverProbe({ displayMode: 'at_pct', debounceMs: 1 }));
    act(() => { result.current.requestProbe(1, 1); });
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    expect(result.current.probe).not.toBeNull();
    act(() => { result.current.clear(); });
    expect(result.current.probe).toBeNull();
  });

  it('evicts the oldest entry when the cache exceeds CACHE_MAX (32)', async () => {
    edsApi.probe.mockImplementation((r, c) => Promise.resolve({ data: { bc: r * 100 + c, elements: {}, phase: null } }));
    const { result } = renderHook(() => useHoverProbe({ displayMode: 'at_pct', debounceMs: 1 }));
    // Fill 33 distinct pixels; first one should be evicted.
    for (let i = 0; i < 33; i++) {
      act(() => { result.current.requestProbe(i, i); });
      await act(async () => { await new Promise(r => setTimeout(r, 5)); });
    }
    expect(edsApi.probe).toHaveBeenCalledTimes(33);
    // Re-request the very first pixel — cache evicted, must refetch.
    act(() => { result.current.requestProbe(0, 0); });
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    expect(edsApi.probe).toHaveBeenCalledTimes(34);
  });
});
