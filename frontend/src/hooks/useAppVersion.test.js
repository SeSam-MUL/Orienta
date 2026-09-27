// @vitest-environment jsdom
//
// The hook's own tests, for the half its two callers cannot reach: what
// happens after the component goes away. A review measured the gap — with the
// cleanup deleted, 29 component tests stayed green while 7 requests fired in
// the 120 s after unmount. ProblemReportDialog unmounts every time it is
// closed, so on a down backend that is one orphaned retry chain per open.
import { describe, it, expect, afterEach, vi } from 'vitest';
import { renderHook, cleanup, act } from '@testing-library/react';

const mockGetAppVersion = vi.fn();
vi.mock('../services/api', () => ({
  getAppVersion: (...a) => mockGetAppVersion(...a),
}));

afterEach(() => {
  cleanup();
  mockGetAppVersion.mockReset();
  vi.useRealTimers();
});

import useAppVersion from './useAppVersion';

describe('useAppVersion', () => {
  it('stops retrying once the component is gone', async () => {
    vi.useFakeTimers();
    mockGetAppVersion.mockRejectedValue(new Error('backend down'));
    const { unmount } = renderHook(() => useAppVersion());

    await vi.advanceTimersByTimeAsync(1000);
    expect(mockGetAppVersion).toHaveBeenCalledTimes(2);

    unmount();
    // Two minutes on the clock. The schedule would have fired at 2 s, 4 s,
    // 8 s, 16 s, 32 s and 64 s from here.
    await vi.advanceTimersByTimeAsync(120000);
    expect(mockGetAppVersion).toHaveBeenCalledTimes(2);
  });

  it('schedules nothing when the in-flight request fails after unmount', async () => {
    vi.useFakeTimers();
    let reject;
    mockGetAppVersion.mockImplementationOnce(
      () => new Promise((_, rej) => { reject = rej; }),
    );
    const { unmount } = renderHook(() => useAppVersion());
    expect(mockGetAppVersion).toHaveBeenCalledTimes(1);

    unmount();
    // The request the hook left behind now fails. Without the `cancelled`
    // guard the catch would happily book the next attempt.
    await act(async () => { reject(new Error('socket closed')); });
    await vi.advanceTimersByTimeAsync(120000);
    expect(mockGetAppVersion).toHaveBeenCalledTimes(1);
  });

  it('reports the answer, and stops asking, even when it says unknown', async () => {
    mockGetAppVersion.mockResolvedValue({ app: 'Orienta', version: 'unknown' });
    const { result } = renderHook(() => useAppVersion());
    await vi.waitFor(() => expect(result.current.info).not.toBeNull());
    expect(result.current.info.version).toBe('unknown');
    // "Unknown" IS an answer — the caller must render it, not keep waiting.
    expect(result.current.unreachable).toBe(false);
    expect(mockGetAppVersion).toHaveBeenCalledTimes(1);
  });

  it('is not "unreachable" while the first request is merely outstanding', () => {
    // The state the About line renders as "checking…". Distinct from the one
    // below, and the whole reason the hook returns two fields instead of one.
    mockGetAppVersion.mockImplementationOnce(() => new Promise(() => {}));
    const { result } = renderHook(() => useAppVersion());
    expect(result.current.info).toBeNull();
    expect(result.current.unreachable).toBe(false);
  });

  it('turns unreachable once an attempt has actually failed', async () => {
    vi.useFakeTimers();
    mockGetAppVersion.mockRejectedValue(new Error('ECONNREFUSED'));
    const { result } = renderHook(() => useAppVersion());
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(result.current.info).toBeNull();
    expect(result.current.unreachable).toBe(true);
  });

  it('starts a fresh mount at the short delay, not where the last one stopped', async () => {
    vi.useFakeTimers();
    mockGetAppVersion.mockRejectedValue(new Error('backend down'));
    const first = renderHook(() => useAppVersion());
    await vi.advanceTimersByTimeAsync(7000);          // 1 s + 2 s + 4 s
    expect(mockGetAppVersion).toHaveBeenCalledTimes(4);
    first.unmount();

    mockGetAppVersion.mockClear();
    renderHook(() => useAppVersion());
    expect(mockGetAppVersion).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(mockGetAppVersion).toHaveBeenCalledTimes(2);
  });
});
