/**
 * The version a follower question was about has to reach the backend: it is what keeps a file that got another PC
 * since the question from being overwritten. The page tests use a fake of this module, so this asserts the WIRE:
 * the real request body, with the field name the route reads (`from_version`).
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
  return { api: mod.pcCalibrationsApi, post: axios.post };
};

describe('pcCalibrationsApi.newVersion', () => {
  beforeEach(() => vi.resetModules());

  it('sends the version the files were on as from_version, with the files and the accepted ones', async () => {
    const { api, post } = await load();
    post.mockClear();
    await api.newVersion('sem', 'scan_a', ['scan_b', 'scan_c'], ['scan_b'], 3);
    const [url, body] = post.mock.calls.at(-1);
    expect(url).toBe('/api/pc-calibrations/sem/versions');
    expect(body).toEqual({ dataset: 'scan_a', apply_to: ['scan_b', 'scan_c'], accept_mismatch: ['scan_b'], from_version: 3 });
  });

  it('sends no version to compare when none is given', async () => {
    const { api, post } = await load();
    post.mockClear();
    await api.newVersion('sem', 'scan_a');
    expect(post.mock.calls.at(-1)[1]).toEqual({ dataset: 'scan_a', apply_to: [], accept_mismatch: [], from_version: null });
  });
});
