// @vitest-environment jsdom
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

const mockCheck = vi.fn();
vi.mock('../services/api', () => ({
  checkForUpdate: (...a) => mockCheck(...a),
}));

import useUpdateCheck, { isCheckEnabled, setCheckEnabled } from './useUpdateCheck';

const AVAILABLE = { available: true, latest: 'v0.2.0', current: 'v0.1.0', install_kind: 'git' };

beforeEach(() => {
  localStorage.clear();
  mockCheck.mockReset();
  mockCheck.mockResolvedValue(AVAILABLE);
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.useRealTimers();
});

describe('useUpdateCheck', () => {
  it('checks shortly after start, not immediately', async () => {
    renderHook(() => useUpdateCheck());
    expect(mockCheck).not.toHaveBeenCalled();
    await act(async () => { vi.advanceTimersByTime(5000); });
    expect(mockCheck).toHaveBeenCalledTimes(1);
  });

  it('surfaces an available update', async () => {
    const { result } = renderHook(() => useUpdateCheck());
    await act(async () => { vi.advanceTimersByTime(5000); });
    await waitFor(() => expect(result.current.updateInfo).toMatchObject({ latest: 'v0.2.0' }));
  });

  it('stays silent when nothing is available', async () => {
    mockCheck.mockResolvedValue({ available: false, reason: 'up_to_date' });
    const { result } = renderHook(() => useUpdateCheck());
    await act(async () => { vi.advanceTimersByTime(5000); });
    expect(result.current.updateInfo).toBeNull();
  });

  it('stays silent when the check itself fails — offline is normal', async () => {
    mockCheck.mockRejectedValue(new Error('network down'));
    const { result } = renderHook(() => useUpdateCheck());
    await act(async () => { vi.advanceTimersByTime(5000); });
    expect(result.current.updateInfo).toBeNull();
  });

  it('does not offer a version the user skipped', async () => {
    localStorage.setItem('orienta.update.skippedVersion', 'v0.2.0');
    const { result } = renderHook(() => useUpdateCheck());
    await act(async () => { vi.advanceTimersByTime(5000); });
    expect(result.current.updateInfo).toBeNull();
  });

  it('still shows a skipped version on an explicit manual check', async () => {
    localStorage.setItem('orienta.update.skippedVersion', 'v0.2.0');
    const { result } = renderHook(() => useUpdateCheck());
    await act(async () => { await result.current.checkNow(); });
    await waitFor(() => expect(result.current.updateInfo).toMatchObject({ latest: 'v0.2.0' }));
  });

  it('skip remembers the version and closes the dialog', async () => {
    const { result } = renderHook(() => useUpdateCheck());
    await act(async () => { vi.advanceTimersByTime(5000); });
    await waitFor(() => expect(result.current.updateInfo).toBeTruthy());
    act(() => { result.current.skip(); });
    expect(result.current.updateInfo).toBeNull();
    expect(localStorage.getItem('orienta.update.skippedVersion')).toBe('v0.2.0');
  });

  it('later closes without remembering', async () => {
    const { result } = renderHook(() => useUpdateCheck());
    await act(async () => { vi.advanceTimersByTime(5000); });
    await waitFor(() => expect(result.current.updateInfo).toBeTruthy());
    act(() => { result.current.close(); });
    expect(result.current.updateInfo).toBeNull();
    expect(localStorage.getItem('orienta.update.skippedVersion')).toBeNull();
  });

  it('does not check at all when the user switched it off', async () => {
    setCheckEnabled(false);
    renderHook(() => useUpdateCheck());
    await act(async () => { vi.advanceTimersByTime(10000); });
    expect(mockCheck).not.toHaveBeenCalled();
  });

  it('the switch defaults to on and round-trips', () => {
    expect(isCheckEnabled()).toBe(true);
    setCheckEnabled(false);
    expect(isCheckEnabled()).toBe(false);
    setCheckEnabled(true);
    expect(isCheckEnabled()).toBe(true);
  });
});
