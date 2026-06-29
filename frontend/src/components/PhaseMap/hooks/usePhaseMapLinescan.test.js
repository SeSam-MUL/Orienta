// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePhaseMapLinescan } from './usePhaseMapLinescan';

vi.mock('../../../services/api', () => ({
  phaseMapApi: { linescan: vi.fn() },
}));
import { phaseMapApi } from '../../../services/api';

beforeEach(() => { vi.clearAllMocks(); });

describe('usePhaseMapLinescan', () => {
  it('fetches profile and stores data', async () => {
    phaseMapApi.linescan.mockResolvedValue({ data: { samples: 4, series: { bc: [1, 2, 3, 4] } } });
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: ['bc'] }));
    await act(async () => { await result.current.fetchProfile({ row: 0, col: 0 }, { row: 3, col: 3 }); });
    expect(result.current.data).toEqual({ samples: 4, series: { bc: [1, 2, 3, 4] } });
    expect(result.current.line).toEqual({ start: { row: 0, col: 0 }, end: { row: 3, col: 3 } });
  });

  it('passes layers + nSamples to the API', async () => {
    phaseMapApi.linescan.mockResolvedValue({ data: { samples: 2, series: {} } });
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: ['phase', 'bc'], nSamples: 64 }));
    await act(async () => { await result.current.fetchProfile({ row: 0, col: 0 }, { row: 1, col: 1 }); });
    expect(phaseMapApi.linescan).toHaveBeenCalledWith({
      start_row: 0, start_col: 0, end_row: 1, end_col: 1,
      n_samples: 64, layers: ['phase', 'bc'],
    });
  });

  it('does not fetch when layerIds is empty', async () => {
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: [] }));
    await act(async () => { await result.current.fetchProfile({ row: 0, col: 0 }, { row: 3, col: 3 }); });
    expect(phaseMapApi.linescan).not.toHaveBeenCalled();
    expect(result.current.data).toBeNull();
    // But the line is still remembered for visual feedback.
    expect(result.current.line).not.toBeNull();
  });

  it('captures errors into result.error', async () => {
    phaseMapApi.linescan.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: ['bc'] }));
    await act(async () => { await result.current.fetchProfile({ row: 0, col: 0 }, { row: 3, col: 3 }); });
    expect(result.current.error).toMatch(/boom/);
    expect(result.current.data).toBeNull();
  });

  it('clear() resets line, data, error', async () => {
    phaseMapApi.linescan.mockResolvedValue({ data: { samples: 2, series: { bc: [1, 2] } } });
    const { result } = renderHook(() => usePhaseMapLinescan({ layerIds: ['bc'] }));
    await act(async () => { await result.current.fetchProfile({ row: 0, col: 0 }, { row: 1, col: 1 }); });
    expect(result.current.data).not.toBeNull();
    act(() => { result.current.clear(); });
    expect(result.current.data).toBeNull();
    expect(result.current.line).toBeNull();
    expect(result.current.error).toBeNull();
  });
});
