// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePhaseMapProbe } from './usePhaseMapProbe';

vi.mock('../../../services/api', () => ({
  phaseMapApi: { probe: vi.fn() },
}));
import { phaseMapApi } from '../../../services/api';

beforeEach(() => { vi.clearAllMocks(); });

describe('usePhaseMapProbe', () => {
  it('debounces and returns probe data', async () => {
    phaseMapApi.probe.mockResolvedValue({ data: { phase: { id: 1, name: 'Al-rich' }, scalars: { ci: 0.78 } } });
    const { result } = renderHook(() => usePhaseMapProbe({ debounceMs: 1 }));
    act(() => result.current.requestProbe(5, 7));
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    expect(phaseMapApi.probe).toHaveBeenCalledTimes(1);
    expect(phaseMapApi.probe).toHaveBeenCalledWith(5, 7);
    expect(result.current.probe).toEqual(expect.objectContaining({ phase: { id: 1, name: 'Al-rich' } }));
  });

  it('coalesces rapid requests', async () => {
    phaseMapApi.probe.mockResolvedValue({ data: { phase: null, scalars: {} } });
    const { result } = renderHook(() => usePhaseMapProbe({ debounceMs: 30 }));
    act(() => {
      result.current.requestProbe(1, 1);
      result.current.requestProbe(2, 2);
      result.current.requestProbe(3, 3);
    });
    await act(async () => { await new Promise(r => setTimeout(r, 80)); });
    expect(phaseMapApi.probe).toHaveBeenCalledTimes(1);
    expect(phaseMapApi.probe).toHaveBeenCalledWith(3, 3);
  });

  it('serves cache on repeat', async () => {
    phaseMapApi.probe.mockResolvedValue({ data: { phase: null, scalars: { ci: 0.5 } } });
    const { result } = renderHook(() => usePhaseMapProbe({ debounceMs: 1 }));
    act(() => result.current.requestProbe(5, 5));
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    expect(phaseMapApi.probe).toHaveBeenCalledTimes(1);
    act(() => result.current.requestProbe(5, 5));
    expect(phaseMapApi.probe).toHaveBeenCalledTimes(1);
    expect(result.current.probe?.scalars?.ci).toBe(0.5);
  });

  it('captures errors', async () => {
    phaseMapApi.probe.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => usePhaseMapProbe({ debounceMs: 1 }));
    act(() => result.current.requestProbe(0, 0));
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    expect(result.current.error).toMatch(/boom/);
  });

  it('clear() resets', async () => {
    phaseMapApi.probe.mockResolvedValue({ data: { phase: null, scalars: {} } });
    const { result } = renderHook(() => usePhaseMapProbe({ debounceMs: 1 }));
    act(() => result.current.requestProbe(0, 0));
    await act(async () => { await new Promise(r => setTimeout(r, 25)); });
    act(() => result.current.clear());
    expect(result.current.probe).toBeNull();
  });
});
