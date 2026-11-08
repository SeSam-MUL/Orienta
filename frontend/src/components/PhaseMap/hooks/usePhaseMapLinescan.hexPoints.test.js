// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePhaseMapLinescan } from './usePhaseMapLinescan';

vi.mock('../../../services/api', () => ({
  phaseMapApi: { linescan: vi.fn() },
}));
import { phaseMapApi } from '../../../services/api';

beforeEach(() => { vi.clearAllMocks(); });

const CELLS = [[0, 0], [1, 0], [2, 0]];
const LAYERS = ['ci'];            // one array for the whole file: the hook takes a new array as a new set of layers

/** An answer the test releases itself. */
function held(data) {
  let release;
  const promise = new Promise((resolve) => { release = () => resolve({ data }); });
  return { promise, release };
}

describe('usePhaseMapLinescan.fetchPoints', () => {
  it('asks for the profile at exactly those scan points', async () => {
    phaseMapApi.linescan.mockResolvedValue({ data: { samples: 3, series: { ci: [1, 2, 3] } } });
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: LAYERS }));
    await act(async () => { await result.current.fetchPoints(CELLS); });
    expect(phaseMapApi.linescan).toHaveBeenCalledWith({ points: CELLS, layers: ['ci'] });
    expect(result.current.data).toEqual({ samples: 3, series: { ci: [1, 2, 3] } });
  });

  it('remembers the line it was given', async () => {
    phaseMapApi.linescan.mockResolvedValue({ data: { samples: 3, series: {} } });
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: LAYERS }));
    await act(async () => { await result.current.fetchPoints(CELLS); });
    expect(result.current.line).toEqual({ points: CELLS });
  });

  it('asks nothing when no layer is visible', async () => {
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: [] }));
    await act(async () => { await result.current.fetchPoints(CELLS); });
    expect(phaseMapApi.linescan).not.toHaveBeenCalled();
    expect(result.current.data).toBeNull();
  });

  it('captures an error', async () => {
    phaseMapApi.linescan.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: LAYERS }));
    await act(async () => { await result.current.fetchPoints(CELLS); });
    expect(result.current.error).toMatch(/boom/);
    expect(result.current.data).toBeNull();
  });

  it('keeps fetchProfile as it was: two corners and the sample count', async () => {
    phaseMapApi.linescan.mockResolvedValue({ data: { samples: 128, series: {} } });
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: LAYERS }));
    await act(async () => { await result.current.fetchProfile({ row: 1, col: 2 }, { row: 3, col: 4 }); });
    expect(phaseMapApi.linescan).toHaveBeenCalledWith({
      start_row: 1, start_col: 2, end_row: 3, end_col: 4, n_samples: 128, layers: ['ci'],
    });
  });
});

describe('an answer that does not belong to the latest request', () => {
  it('is dropped when a newer request was made, and the newer one is kept', async () => {
    const first = held({ samples: 3, series: { ci: [1, 1, 1] } });
    const second = held({ samples: 3, series: { ci: [2, 2, 2] } });
    phaseMapApi.linescan.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: LAYERS }));
    let p1;
    let p2;
    act(() => { p1 = result.current.fetchPoints(CELLS); });
    act(() => { p2 = result.current.fetchPoints([[0, 1], [0, 2]]); });
    await act(async () => { second.release(); await p2; });
    expect(result.current.data.series.ci).toEqual([2, 2, 2]);
    await act(async () => { first.release(); await p1; });
    expect(result.current.data.series.ci).toEqual([2, 2, 2]);
    expect(result.current.loading).toBe(false);
  });

  it('is dropped after clear(), and clear() ends the loading', async () => {
    const pending = held({ samples: 3, series: { ci: [9, 9, 9] } });
    phaseMapApi.linescan.mockReturnValueOnce(pending.promise);
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: LAYERS }));
    let p;
    act(() => { p = result.current.fetchPoints(CELLS); });
    expect(result.current.loading).toBe(true);
    act(() => { result.current.clear(); });
    expect(result.current.loading).toBe(false);
    await act(async () => { pending.release(); await p; });
    expect(result.current.data).toBeNull();
    expect(result.current.line).toBeNull();
  });

  it('is dropped when the next request finds no visible layer, which does not leave the loading on', async () => {
    const pending = held({ samples: 3, series: { ci: [9, 9, 9] } });
    phaseMapApi.linescan.mockReturnValueOnce(pending.promise);
    const { result, rerender } = renderHook(({ ids }) => usePhaseMapLinescan({ layerIds: ids }), {
      initialProps: { ids: LAYERS } });
    let p;
    act(() => { p = result.current.fetchPoints(CELLS); });
    rerender({ ids: [] });
    await act(async () => { await result.current.fetchPoints(CELLS); });
    expect(result.current.loading).toBe(false);
    await act(async () => { pending.release(); await p; });
    expect(result.current.data).toBeNull();
  });
});
