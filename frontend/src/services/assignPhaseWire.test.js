/**
 * The seed orientation has to actually reach the backend.
 *
 * This is the defect class this codebase keeps producing: the panel holds a
 * good orientation, the route knows what to do with one, and nothing carries
 * it between them. A unit test of either half passes while the feature does
 * nothing. So this asserts the WIRE — the real axios body, with the field name
 * the backend reads (`seed_quat`).
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
  return { indexApi: mod.indexApi ?? mod.default?.indexApi ?? mod, post: axios.post };
};

describe('assignPhaseToGrain sends the seed orientation', () => {
  beforeEach(() => vi.resetModules());

  it('puts the quaternion in the body as seed_quat', async () => {
    const { indexApi, post } = await load();
    post.mockClear();
    await indexApi.assignPhaseToGrain({
      row: 28, col: 26, targetPhaseId: 2, seedQuat: [0.1, 0.2, 0.3, 0.4],
    });
    const [url, body] = post.mock.calls.at(-1);
    expect(url).toBe('/api/indexing/pattern-match/assign-phase');
    expect(body.seed_quat).toEqual([0.1, 0.2, 0.3, 0.4]);
    expect(body.target_phase_id).toBe(2);
    expect(body.row).toBe(28);
    expect(body.col).toBe(26);
  });

  it('sends null when there is no candidate orientation', async () => {
    // Explicit null, not a missing key: the backend treats null as "use the
    // old Hough-only behaviour", which is what we want when we have nothing.
    const { indexApi, post } = await load();
    post.mockClear();
    await indexApi.assignPhaseToGrain({ row: 0, col: 0, targetPhaseId: 1 });
    expect(post.mock.calls.at(-1)[1].seed_quat).toBeNull();
  });
});
