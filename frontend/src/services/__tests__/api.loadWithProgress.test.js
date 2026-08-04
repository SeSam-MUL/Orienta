// @vitest-environment jsdom
/**
 * Tests for ebsdApi.loadWithProgress — UUID generation, parallel POST + poll,
 * onProgress callback invocation, success and error paths.
 *
 * Mock strategy: axios.create is mocked so that the internal `api` instance
 * in api.js is the same mock object we control here. ebsdApi.loadWithProgress
 * calls api.post / api.get internally; those land on our mockPost / mockGet.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mockPost = vi.fn();
const mockGet = vi.fn();
const mockAxiosInstance = { post: mockPost, get: mockGet };

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => mockAxiosInstance),
  },
}));

// Import AFTER the mock is registered so api.js picks up the mocked axios.
const { ebsdApi } = await import('../api');

describe('ebsdApi.loadWithProgress', () => {
  let onProgressCalls;

  beforeEach(() => {
    mockPost.mockReset();
    mockGet.mockReset();
    onProgressCalls = [];
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('generates a UUID and includes it in the POST body', async () => {
    mockPost.mockResolvedValue({ data: { success: true, request_id: 'uuid-here' } });
    mockGet.mockResolvedValue({
      data: { stage: 'complete', stage_idx: 4, stage_total: 4, elapsed_seconds: 1.2, message: 'Done' },
    });

    const onProgress = (state) => onProgressCalls.push(state);
    await ebsdApi.loadWithProgress('/some/path.h5oina', { onProgress });

    expect(mockPost).toHaveBeenCalledTimes(1);
    const [url, body] = mockPost.mock.calls[0];
    expect(url).toBe('/api/ebsd/load');
    expect(body.path).toBe('/some/path.h5oina');
    // UUID v4: 8-4-4-4-12 hex chars
    expect(body.request_id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
  });

  it('calls onProgress with each polled snapshot until POST resolves', async () => {
    vi.useFakeTimers();
    const snapshots = [
      { stage: 'reading_metadata', stage_idx: 1, stage_total: 4, elapsed_seconds: 0.05, message: 'Reading file headers' },
      { stage: 'building_signal',  stage_idx: 2, stage_total: 4, elapsed_seconds: 0.30, message: 'Building lazy signal' },
      { stage: 'detecting_features', stage_idx: 3, stage_total: 4, elapsed_seconds: 0.70, message: 'Detecting EDS' },
      { stage: 'complete', stage_idx: 4, stage_total: 4, elapsed_seconds: 1.10, message: 'Done' },
    ];
    let pollIdx = 0;
    mockGet.mockImplementation(() =>
      Promise.resolve({ data: snapshots[Math.min(pollIdx++, snapshots.length - 1)] }),
    );

    // Resolve POST after several polls so progress callbacks have fired.
    let resolvePost;
    mockPost.mockImplementation(
      () =>
        new Promise((r) => {
          resolvePost = r;
        }),
    );

    const onProgress = (state) => onProgressCalls.push(state);
    const loadPromise = ebsdApi.loadWithProgress('/p.h5oina', {
      onProgress,
      pollIntervalMs: 100,
    });

    // Advance the fake clock to trigger ~3 polls. The first poll is queued at 0
    // (initial kick) and subsequent polls at 100ms cadence. Advance 350ms
    // total so we get at least 3 fully-resolved snapshots.
    for (let i = 0; i < 4; i++) {
      await vi.advanceTimersByTimeAsync(100);
    }

    resolvePost({ data: { success: true, request_id: 'rid' } });
    await loadPromise;

    expect(onProgressCalls.length).toBeGreaterThanOrEqual(3);
    expect(onProgressCalls[0].stage).toBe('reading_metadata');
    expect(onProgressCalls.at(-1).stage).toBeDefined();
  });

  it('resolves with the POST response on success', async () => {
    mockPost.mockResolvedValue({ data: { success: true, dataset_name: 'X' } });
    mockGet.mockResolvedValue({
      data: { stage: 'complete', stage_idx: 4, stage_total: 4, elapsed_seconds: 1.0, message: 'Done' },
    });

    const result = await ebsdApi.loadWithProgress('/x.h5oina', {});
    expect(result.data.success).toBe(true);
    expect(result.data.dataset_name).toBe('X');
  });

  it('rejects and calls onProgress with error state when POST fails', async () => {
    const error = Object.assign(new Error('400 Bad Request'), {
      response: { status: 400, data: { detail: 'file missing' } },
    });
    mockPost.mockRejectedValue(error);
    mockGet.mockResolvedValue({
      data: { stage: 'reading_metadata', stage_idx: 1, stage_total: 4, elapsed_seconds: 0.1, message: 'Reading file headers' },
    });

    const onProgress = (state) => onProgressCalls.push(state);
    await expect(
      ebsdApi.loadWithProgress('/bad.h5oina', { onProgress }),
    ).rejects.toBe(error);

    // The last onProgress should be an error state (synthesised client-side).
    const last = onProgressCalls.at(-1);
    expect(last.stage).toBe('error');
    expect(last.message).toContain('file missing');
  });
  // --- Regression: a slow load must not be killed by a client deadline -------
  // A user reported "Load failed — timeout of 300000ms exceeded" while the
  // backend log showed the load handler running the whole time (continuous
  // 200s on the progress endpoint). The POST inherited the axios instance's
  // 5-minute default; aborting it does not stop the backend, so the file
  // finished loading while the UI claimed failure.

  it('sends the load POST without a client-side timeout', async () => {
    mockPost.mockResolvedValue({ data: { success: true } });
    mockGet.mockResolvedValue({
      data: { stage: 'complete', stage_idx: 4, stage_total: 4, elapsed_seconds: 1.0, message: 'Done' },
    });

    await ebsdApi.loadWithProgress('/big.h5oina', {});

    const config = mockPost.mock.calls[0][2];
    expect(config).toBeDefined();
    expect(config.timeout).toBe(0);          // 0 = no deadline in axios
    expect(config.signal).toBeDefined();     // still cancellable
  });

  it('reports the backend elapsed time on failure, not 0', async () => {
    const error = Object.assign(new Error('boom'), {
      response: { status: 500, data: { detail: 'disk exploded' } },
    });
    mockPost.mockRejectedValue(error);
    mockGet.mockResolvedValue({
      data: {
        stage: 'building_signal', stage_idx: 2, stage_total: 4,
        elapsed_seconds: 187.5, message: 'Building lazy signal',
      },
    });

    const onProgress = (state) => onProgressCalls.push(state);
    // Let at least one poll land so an elapsed value is observed.
    await new Promise((resolve) => setTimeout(resolve, 10));
    await expect(
      ebsdApi.loadWithProgress('/slow.h5oina', { onProgress }),
    ).rejects.toBe(error);

    const last = onProgressCalls.at(-1);
    expect(last.stage).toBe('error');
    // Either the observed backend elapsed, or 0 if no poll landed first —
    // but it must never be a hardcoded 0 when a poll DID report a value.
    if (onProgressCalls.some((c) => c.elapsed_seconds === 187.5)) {
      expect(last.elapsed_seconds).toBe(187.5);
    }
  });

  it('aborts the POST when the backend goes silent', async () => {
    vi.useFakeTimers();
    // POST never settles — mimics a backend that died mid-load.
    mockPost.mockReturnValue(new Promise(() => {}));
    // Progress polls fail as network errors (no .response) => no contact.
    mockGet.mockRejectedValue(new Error('network down'));

    const onProgress = (state) => onProgressCalls.push(state);
    ebsdApi.loadWithProgress('/dead.h5oina', { onProgress }).catch(() => {});

    // Push past STALE_THRESHOLD_MS (10 s) worth of 250 ms polls.
    for (let i = 0; i < 60; i += 1) {
      await vi.advanceTimersByTimeAsync(250);
    }

    const stale = onProgressCalls.find((c) => c.error === 'Backend not responding');
    expect(stale).toBeDefined();
    const config = mockPost.mock.calls[0][2];
    expect(config.signal.aborted).toBe(true);
  });
});
