// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';

const version = vi.fn();
vi.mock('../../services/api', () => ({ stateApi: { version: () => version() } }));

import { useStateVersionPoll } from './useStateVersionPoll';

beforeEach(() => { vi.useFakeTimers(); version.mockReset(); });
afterEach(() => vi.useRealTimers());

describe('useStateVersionPoll', () => {
  it('fires onChange when version increments', async () => {
    version
      .mockResolvedValueOnce({ data: { version: 1, active_result_id: 'a' } })
      .mockResolvedValueOnce({ data: { version: 1, active_result_id: 'a' } })
      .mockResolvedValueOnce({ data: { version: 2, active_result_id: 'a' } });
    const onChange = vi.fn();
    renderHook(() => useStateVersionPoll(onChange, { intervalMs: 100 }));
    await vi.advanceTimersByTimeAsync(0);     // mount poll
    await vi.advanceTimersByTimeAsync(100);   // same version → no extra call
    await vi.advanceTimersByTimeAsync(100);   // version 2 → fire
    expect(onChange).toHaveBeenCalledTimes(2); // once on mount, once on change
  });
});
