/**
 * The ML Hub requests carry what the user chose: the learning rate typed in the
 * Training tab goes into the /api/ml/train body, and a clicked run goes to
 * /api/ml/models/load. Asserts the real axios bodies.
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
  return { mlApi: mod.mlApi, post: axios.post };
};

describe('mlApi wire format', () => {
  beforeEach(() => vi.resetModules());

  it('train sends the learning rate', async () => {
    const { mlApi, post } = await load();
    post.mockClear();
    await mlApi.train({ epochs: 3, batchSize: 8, learningRate: 0.0042, ciThreshold: 0.5 });
    const [url, body] = post.mock.calls.at(-1);
    expect(url).toBe('/api/ml/train');
    expect(body).toEqual({
      training_data_path: '__last_indexing__', epochs: 3, batch_size: 8,
      learning_rate: 0.0042, ci_threshold: 0.5, include_existing_store: false,
    });
  });

  it('train falls back to the backend default learning rate value', async () => {
    const { mlApi, post } = await load();
    post.mockClear();
    await mlApi.train();
    expect(post.mock.calls.at(-1)[1].learning_rate).toBe(0.001);
  });

  it('train asks for this request\'s samples only unless told to include the store', async () => {
    const { mlApi, post } = await load();
    post.mockClear();
    await mlApi.train();
    expect(post.mock.calls.at(-1)[1].include_existing_store).toBe(false);
    await mlApi.train({ includeExistingStore: true });
    expect(post.mock.calls.at(-1)[1].include_existing_store).toBe(true);
  });

  it('loadModel posts the run id', async () => {
    const { mlApi, post } = await load();
    post.mockClear();
    await mlApi.loadModel('20261014T120000Z');
    expect(post.mock.calls.at(-1)).toEqual(['/api/ml/models/load', { run_id: '20261014T120000Z' }]);
  });
});
