// @vitest-environment jsdom
//
// Mounts the real Grain Analysis panel with only the HTTP layer mocked, so the
// polling branch of handleReconstruct is exercised end to end: nested task
// payload -> statistics in the DOM, failed/vanished task -> error text and the
// spinner stops. (The pure helpers are covered in grainStats.test.js.)
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  analysisApi: {
    reconstructGrains: vi.fn(),
    getTaskStatus: vi.fn(),
    getMap: vi.fn(() => new Promise(() => {})),
    grainSize: vi.fn(),
  },
  indexApi: {},
  edsApi: {},
}));

import { analysisApi } from '../../../services/api';
import useResultStore from '../../../stores/useResultStore';
import { GrainTabInPage } from './grainTabHarness';

const COMPLETED = {
  status: 'completed',
  progress: 1.0,
  grain_count: 98,
  result: {
    n_grains: 98, grain_count: 98, alpha: 3.1, fill: true, n_filled: 4,
    mean_ecd: 12.34, median_ecd: 11.0, std_ecd: 3.21, mean_aspect_ratio: 1.678,
  },
};

function mountAndStart() {
  render(<GrainTabInPage />);
  fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
}

const POLL_TIMEOUT = { timeout: 5000 };

describe('GrainTab polling a reconstruction task', () => {
  beforeEach(() => {
    analysisApi.reconstructGrains.mockResolvedValue({ data: { task_id: 't1' } });
  });
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    useResultStore.getState().clearResults();
  });

  it('shows ECD and aspect ratio from the nested result, not only the count', async () => {
    analysisApi.getTaskStatus.mockResolvedValue({ data: COMPLETED });
    mountAndStart();
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.getByText(/Mean AR:\s+1\.678/)).toBeTruthy();
    expect(screen.getByText(/Total grains:\s+98/)).toBeTruthy();
    expect(screen.getByText(/Reconstructed 98 grains/)).toBeTruthy();
  });

  it('an empty min intercept does not stop the reconstruction: the request carries none', async () => {
    useResultStore.getState().setGrainStatsSettings({ minIntercept: '' });
    analysisApi.getTaskStatus.mockResolvedValue({ data: COMPLETED });
    mountAndStart();
    await waitFor(() => expect(analysisApi.reconstructGrains).toHaveBeenCalledTimes(1));
    const args = analysisApi.reconstructGrains.mock.calls[0];
    expect(args[2]).toBeUndefined();                                  // minIntercept
    expect(args.slice(0, 2)).toEqual([4, 3]);
    await waitFor(() => expect(screen.getByText(/Reconstructed 98 grains/)).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.queryByText('Reconstruction failed')).toBeNull();
  });

  it('a failed task shows the backend error and stops the spinner', async () => {
    analysisApi.getTaskStatus.mockResolvedValue({ data: { status: 'failed', progress: 1.0, error: 'no grains today' } });
    mountAndStart();
    await waitFor(() => expect(screen.getByText('no grains today')).toBeTruthy(), POLL_TIMEOUT);
    const btn = screen.getByRole('button', { name: /reconstruct grains/i });
    expect(btn.disabled).toBe(false);
    expect(screen.queryByText(/Total grains/)).toBeNull();
  });

  it('a vanished task (404) stops polling with the generic failure text', async () => {
    analysisApi.getTaskStatus.mockRejectedValue({ response: { status: 404 } });
    mountAndStart();
    await waitFor(() => expect(screen.getByText('Reconstruction failed')).toBeTruthy(), POLL_TIMEOUT);
    expect(screen.getByRole('button', { name: /reconstruct grains/i }).disabled).toBe(false);
    const calls = analysisApi.getTaskStatus.mock.calls.length;
    await new Promise((r) => setTimeout(r, 1800));
    expect(analysisApi.getTaskStatus.mock.calls.length).toBe(calls);
  });

  it('a network blip keeps polling and the next answer completes the task', async () => {
    analysisApi.getTaskStatus
      .mockRejectedValueOnce(new Error('Network Error'))
      .mockResolvedValue({ data: COMPLETED });
    mountAndStart();
    await waitFor(() => expect(screen.getByText(/Mean ECD:\s+12\.34/)).toBeTruthy(), { timeout: 6000 });
    expect(analysisApi.getTaskStatus.mock.calls.length).toBeGreaterThanOrEqual(2);
  }, 10000);
});

describe('GrainTab input validation', () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    useResultStore.getState().clearResults();
  });

  it('an empty alpha field sends nothing, shows the generic failure and does not crash', async () => {
    render(<GrainTabInPage />);
    const alpha = screen.getByTitle(/Not-indexed areas narrower than about/);
    expect(alpha.max).toBe('20');
    fireEvent.change(alpha, { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(screen.getByText('Reconstruction failed')).toBeTruthy());
    expect(analysisApi.reconstructGrains).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /reconstruct grains/i }).disabled).toBe(false);
  });

  it('a 422 with a list detail is shown as readable text', async () => {
    analysisApi.reconstructGrains.mockRejectedValue({
      message: 'Request failed with status code 422',
      response: {
        status: 422,
        data: { detail: [{ type: 'less_than_equal', loc: ['body', 'alpha'], msg: 'Input should be <= 20' }] },
      },
    });
    render(<GrainTabInPage />);
    fireEvent.click(screen.getByRole('button', { name: /reconstruct grains/i }));
    await waitFor(() => expect(screen.getByText('alpha: Input should be <= 20')).toBeTruthy());
    expect(screen.getByRole('button', { name: /reconstruct grains/i }).disabled).toBe(false);
  });
});
