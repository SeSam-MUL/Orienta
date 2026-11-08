// @vitest-environment jsdom
/**
 * dbApi.sphere hands its AbortSignal to axios. The viewer aborts the request it
 * leaves, and the server drops a queued reconstruction whose client is gone --
 * a signal that never reaches axios would cancel nothing.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mockGet = vi.fn();
const mockAxiosInstance = {
  get: mockGet,
  post: vi.fn(),
  interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
};
vi.mock('axios', () => ({ default: { create: vi.fn(() => mockAxiosInstance) } }));

const { dbApi } = await import('../api');

describe('dbApi.sphere', () => {
  beforeEach(() => { mockGet.mockReset(); mockGet.mockResolvedValue({ data: {} }); });

  it('passes the signal and the resolution params to axios', async () => {
    const controller = new AbortController();
    await dbApi.sphere('a b.sht', { max_bandwidth: 256, target_size: 192 }, { signal: controller.signal });
    const [url, config] = mockGet.mock.calls[0];
    expect(url).toBe('/api/database/sphere/a%20b.sht');
    expect(config.signal).toBe(controller.signal);
    expect(config.params).toEqual({ max_bandwidth: 256, target_size: 192 });
  });

  it('still works without a config (older callers)', async () => {
    await dbApi.sphere('x.sht');
    const [, config] = mockGet.mock.calls[0];
    expect(config.params).toEqual({});
    expect(config.signal).toBeUndefined();
  });
});
