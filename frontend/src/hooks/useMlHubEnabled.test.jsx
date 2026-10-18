// @vitest-environment jsdom
//
// The ML Hub is internal. /api/ml/status always answers 200 and says
// {enabled: true|false} (true only if the backend was started with
// ORIENTA_EXPERIMENTAL_ML=1); this hook turns that into
// 'enabled' / 'disabled' / 'unknown'. The case that matters is the third: a
// request that did not get an answer is NOT "disabled" -- a one-shot fetch at
// mount that took a failure as the answer would hide the entry for good.
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { renderHook, cleanup, act } from '@testing-library/react';

const mockStatus = vi.fn();
vi.mock('../services/api', () => ({
  mlApi: { status: (...a) => mockStatus(...a) },
}));

import useMlHubEnabled, { useMlHubState, __resetMlHubEnabledForTests } from './useMlHubEnabled';

const notFound = () => Object.assign(new Error('404'), { response: { status: 404 } });
const noResponse = () => new Error('Network Error');
const ON = { status: 200, data: { enabled: true } };
const OFF = { status: 200, data: { enabled: false } };

beforeEach(() => { __resetMlHubEnabledForTests(); });
afterEach(() => {
  cleanup();
  __resetMlHubEnabledForTests();
  mockStatus.mockReset();
  vi.useRealTimers();
});

describe('useMlHubEnabled', () => {
  it('is false while the first request is out, true on a 200', async () => {
    let resolve;
    mockStatus.mockImplementationOnce(() => new Promise((r) => { resolve = r; }));
    const { result } = renderHook(() => useMlHubEnabled());
    expect(result.current).toBe(false);
    await act(async () => { resolve(ON); });
    expect(result.current).toBe(true);
  });

  it('a 200 with enabled:false is final: hidden, and nothing is asked again', async () => {
    vi.useFakeTimers();
    mockStatus.mockResolvedValue(OFF);
    const { result } = renderHook(() => useMlHubState());
    await vi.advanceTimersByTimeAsync(0);
    expect(result.current).toBe('disabled');
    await vi.advanceTimersByTimeAsync(180000);
    expect(mockStatus).toHaveBeenCalledTimes(1);
  });

  it('a 200 without enabled === true is not enabled (missing, "true", 1)', async () => {
    for (const data of [{}, { enabled: 'true' }, { enabled: 1 }, null]) {
      __resetMlHubEnabledForTests();
      mockStatus.mockReset();
      mockStatus.mockResolvedValue({ status: 200, data });
      const { result, unmount } = renderHook(() => useMlHubState());
      await act(async () => {});
      expect(result.current).toBe('disabled');
      unmount();
    }
  });

  it('a 404 (older backend) is also final: hidden, and nothing is asked again', async () => {
    vi.useFakeTimers();
    mockStatus.mockRejectedValue(notFound());
    const { result } = renderHook(() => useMlHubState());
    await vi.advanceTimersByTimeAsync(0);
    expect(result.current).toBe('disabled');
    await vi.advanceTimersByTimeAsync(180000);
    expect(mockStatus).toHaveBeenCalledTimes(1);
  });

  it('no response is "unknown", not "disabled": it retries and then shows', async () => {
    vi.useFakeTimers();
    mockStatus.mockRejectedValueOnce(noResponse());
    mockStatus.mockRejectedValueOnce(noResponse());
    mockStatus.mockResolvedValue(ON);
    const { result } = renderHook(() => useMlHubState());
    await vi.advanceTimersByTimeAsync(0);
    expect(result.current).toBe('unknown');
    expect(mockStatus).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1000); // first retry after 1 s
    expect(mockStatus).toHaveBeenCalledTimes(2);
    expect(result.current).toBe('unknown');
    await vi.advanceTimersByTimeAsync(2000); // doubled
    expect(mockStatus).toHaveBeenCalledTimes(3);
    expect(result.current).toBe('enabled');
  });

  it('a 5xx is also unknown, not an answer', async () => {
    vi.useFakeTimers();
    mockStatus.mockRejectedValueOnce(Object.assign(new Error('500'), { response: { status: 500 } }));
    mockStatus.mockResolvedValue(ON);
    const { result } = renderHook(() => useMlHubState());
    await vi.advanceTimersByTimeAsync(0);
    expect(result.current).toBe('unknown');
    await vi.advanceTimersByTimeAsync(1000);
    expect(result.current).toBe('enabled');
  });

  it('two subscribers share one request loop', async () => {
    mockStatus.mockResolvedValue(ON);
    const a = renderHook(() => useMlHubEnabled());
    const b = renderHook(() => useMlHubEnabled());
    await act(async () => {});
    expect(a.result.current).toBe(true);
    expect(b.result.current).toBe(true);
    expect(mockStatus).toHaveBeenCalledTimes(1);
  });

  it('stops retrying when the last subscriber is gone', async () => {
    vi.useFakeTimers();
    mockStatus.mockRejectedValue(noResponse());
    const { unmount } = renderHook(() => useMlHubState());
    await vi.advanceTimersByTimeAsync(1000);
    expect(mockStatus).toHaveBeenCalledTimes(2);
    unmount();
    await vi.advanceTimersByTimeAsync(120000);
    expect(mockStatus).toHaveBeenCalledTimes(2);
  });
});
