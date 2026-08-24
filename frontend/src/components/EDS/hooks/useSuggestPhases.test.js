// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSuggestPhases } from './useSuggestPhases';

vi.mock('../../../services/api', () => ({
  edsApi: { suggestPhases: vi.fn() },
}));
import { edsApi } from '../../../services/api';

/** Backend-shaped response so the test exercises the real unwrapping. */
const reply = (name, row, col) => ({
  data: {
    suggestions: [{ name, cif_filename: name, formula: 'X', score: 0.8 }],
    atomic_pct: { Al: 90 },
    library_source: 'cif',
    library_size: 30,
    _pixel: `${row},${col}`,
  },
});

beforeEach(() => { vi.clearAllMocks(); });

describe('useSuggestPhases', () => {
  it('run() fetches for the given pixel and records which pixel it belongs to', async () => {
    edsApi.suggestPhases.mockResolvedValue(reply('Al.cif', 10, 10));
    const { result } = renderHook(() => useSuggestPhases());

    await act(async () => { await result.current.run(10, 10); });

    expect(edsApi.suggestPhases).toHaveBeenCalledWith(10, 10);
    expect(result.current.suggestions[0].name).toBe('Al.cif');
    expect(result.current.pixel).toEqual({ row: 10, col: 10 });
    expect(result.current.librarySource).toBe('cif');
    expect(result.current.librarySize).toBe(30);
  });

  // THE REPORTED BUG: clicking another pixel left the panel showing the first
  // pixel's phases. Once results are on screen they must follow the cursor.
  it('followPixel() refreshes the result when another pixel is clicked', async () => {
    edsApi.suggestPhases.mockResolvedValueOnce(reply('Al.cif', 10, 10));
    const { result } = renderHook(() => useSuggestPhases());
    await act(async () => { await result.current.run(10, 10); });

    edsApi.suggestPhases.mockResolvedValueOnce(reply('sd_0302719.cif', 45, 60));
    await act(async () => { await result.current.followPixel(45, 60); });

    expect(edsApi.suggestPhases).toHaveBeenLastCalledWith(45, 60);
    expect(result.current.suggestions[0].name).toBe('sd_0302719.cif');
    expect(result.current.pixel).toEqual({ row: 45, col: 60 });
  });

  it('followPixel() stays silent until the user has asked for suggestions once', async () => {
    const { result } = renderHook(() => useSuggestPhases());
    await act(async () => { await result.current.followPixel(1, 2); });
    expect(edsApi.suggestPhases).not.toHaveBeenCalled();
    expect(result.current.suggestions).toBeNull();
  });

  it('clear() stops the panel from following the cursor again', async () => {
    edsApi.suggestPhases.mockResolvedValue(reply('Al.cif', 10, 10));
    const { result } = renderHook(() => useSuggestPhases());
    await act(async () => { await result.current.run(10, 10); });
    act(() => { result.current.clear(); });

    expect(result.current.suggestions).toBeNull();
    expect(result.current.pixel).toBeNull();

    vi.clearAllMocks();
    await act(async () => { await result.current.followPixel(5, 5); });
    expect(edsApi.suggestPhases).not.toHaveBeenCalled();
  });

  it('a slow reply for an old pixel never overwrites a newer one', async () => {
    edsApi.suggestPhases.mockResolvedValueOnce(reply('Al.cif', 10, 10));
    const { result } = renderHook(() => useSuggestPhases());
    await act(async () => { await result.current.run(10, 10); });

    let releaseSlow;
    const slow = new Promise((res) => { releaseSlow = () => res(reply('STALE.cif', 20, 20)); });
    edsApi.suggestPhases.mockReturnValueOnce(slow);                     // pixel B (slow)
    edsApi.suggestPhases.mockResolvedValueOnce(reply('FRESH.cif', 30, 30)); // pixel C (fast)

    await act(async () => {
      const pB = result.current.followPixel(20, 20);
      const pC = result.current.followPixel(30, 30);
      releaseSlow();
      await Promise.all([pB, pC]);
    });

    expect(result.current.suggestions[0].name).toBe('FRESH.cif');
    expect(result.current.pixel).toEqual({ row: 30, col: 30 });
  });

  it('surfaces backend errors and drops the stale list', async () => {
    edsApi.suggestPhases.mockResolvedValueOnce(reply('Al.cif', 10, 10));
    const { result } = renderHook(() => useSuggestPhases());
    await act(async () => { await result.current.run(10, 10); });

    edsApi.suggestPhases.mockRejectedValueOnce({ response: { data: { detail: 'pixel outside scan' } } });
    await act(async () => { await result.current.followPixel(999, 999); });

    expect(result.current.error).toBe('pixel outside scan');
    expect(result.current.suggestions).toBeNull();
    expect(result.current.loading).toBe(false);
  });
});
