/**
 * The BG Static button must not name a reference pattern.
 *
 * The backend subtracts the scan average when the request carries no
 * static_bg_row / static_bg_col, and one single pattern when it does. A default
 * of (0, 0) on the client turned every click on "BG Static" into "subtract the
 * pattern at the scan origin from the whole map". This asserts the real axios
 * body.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('axios', () => {
  const post = vi.fn(() => Promise.resolve({ data: {} }));
  const inst = {
    post, get: vi.fn(), put: vi.fn(), delete: vi.fn(),
    interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
    defaults: { headers: { common: {} } },
  };
  return { default: { create: () => inst, ...inst } };
});

const load = async () => {
  const mod = await import('./api');
  const axios = (await import('axios')).default;
  return { ebsdApi: mod.ebsdApi ?? mod.default?.ebsdApi, post: axios.post };
};

describe('ebsdApi.backgroundRemoval wire format', () => {
  beforeEach(() => vi.resetModules());

  it('static without a reference sends only the method', async () => {
    const { ebsdApi, post } = await load();
    post.mockClear();
    await ebsdApi.backgroundRemoval('static');
    const [url, body] = post.mock.calls.at(-1);
    expect(url).toBe('/api/ebsd/background-removal');
    expect(body).toEqual({ method: 'static' });
  });

  it('dynamic sends only the method', async () => {
    const { ebsdApi, post } = await load();
    post.mockClear();
    await ebsdApi.backgroundRemoval('dynamic');
    expect(post.mock.calls.at(-1)[1]).toEqual({ method: 'dynamic' });
  });

  it('static with an explicit reference pattern sends both coordinates', async () => {
    const { ebsdApi, post } = await load();
    post.mockClear();
    await ebsdApi.backgroundRemoval('static', 3, 5);
    expect(post.mock.calls.at(-1)[1]).toEqual({
      method: 'static', static_bg_row: 3, static_bg_col: 5,
    });
  });
});
