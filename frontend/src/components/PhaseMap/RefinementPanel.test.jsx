// @vitest-environment jsdom
/**
 * The refinement panel's summary line and its reading of a finished job.
 *
 * Two things went wrong together in the running app: the summary route answered
 * 500 for a refinement without forward diagnostics, and the panel (a) printed
 * the NCC improvement from a key the summary never has ("median": the summary
 * has mean/std/min/max/p05/p95), and (b) took the failed summary request after
 * a COMPLETED job for a failed refinement.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';

const progress = vi.fn();
const summary = vi.fn();
vi.mock('../../services/api', () => ({
  refinementApi: {
    progress: (...a) => progress(...a),
    summary: (...a) => summary(...a),
    compute: vi.fn(), resmooth: vi.fn(), cancel: vi.fn(),
  },
}));

import useDataStore from '../../stores/useDataStore';
import RefinementPanel from './RefinementPanel';

const RID = 'spherical_1';

function computedWith(nccImprovement) {
  return {
    lambda: 0.01,
    computed_at: '2026-10-23T10:00:00',
    refined_result_id: 'refinement_2',
    stage_timings: { stage_1: 30, stage_2: 5, stage_3: 60 },
    summary: {
      n_pixels_refined: 400, n_converged: 398, n_failed: 2,
      ncc_improvement: nccImprovement,
    },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  useDataStore.setState({ refinementComputed: {}, refinementJob: null, refinementLambdaPending: 0.01 });
});
afterEach(() => {
  cleanup();
  useDataStore.setState({ refinementComputed: {}, refinementJob: null, refinementLambdaPending: 0.01 });
});

describe('the NCC improvement line', () => {
  it('shows the mean of the improvement, signed and formatted', () => {
    useDataStore.setState({ refinementComputed: {
      [RID]: computedWith({ mean: 0.1234, std: 0.1, min: 0, max: 0.5, p05: 0, p95: 0.4 }),
    } });
    render(<RefinementPanel resultId={RID} />);
    expect(screen.getByText('NCC improvement (mean): +0.12')).toBeTruthy();
  });

  it('keeps the minus sign of a loss', () => {
    useDataStore.setState({ refinementComputed: {
      [RID]: computedWith({ mean: -0.05 }),
    } });
    render(<RefinementPanel resultId={RID} />);
    expect(screen.getByText('NCC improvement (mean): -0.05')).toBeTruthy();
  });

  it('does not print a minus sign for a mean that rounds to zero', () => {
    useDataStore.setState({ refinementComputed: {
      [RID]: computedWith({ mean: -0.0004 }),
    } });
    render(<RefinementPanel resultId={RID} />);
    expect(screen.getByText('NCC improvement (mean): +0.00')).toBeTruthy();
  });

  it('is not shown at all when there is no improvement to report (null)', () => {
    // No forward diagnostics on the original: nothing to improve on.
    useDataStore.setState({ refinementComputed: { [RID]: computedWith(null) } });
    const { container } = render(<RefinementPanel resultId={RID} />);
    expect(screen.queryByText(/NCC/)).toBeNull();
    expect(container.textContent).not.toMatch(/undefined|NaN/);
    // the rest of the summary is still there
    expect(container.textContent).toContain('400 pixel');
  });

  it('is not shown for a mean that is not a number (null from the route)', () => {
    useDataStore.setState({ refinementComputed: {
      [RID]: computedWith({ mean: null, std: null }),
    } });
    const { container } = render(<RefinementPanel resultId={RID} />);
    expect(screen.queryByText(/NCC/)).toBeNull();
    expect(container.textContent).not.toMatch(/undefined|NaN|null/);
  });
});

describe('a refined result the backend lists but this session never computed', () => {
  it('offers Re-Refine and Open Refined without any store entry', () => {
    summary.mockRejectedValue(new Error('404'));
    const onSwitchToRefined = vi.fn();
    render(<RefinementPanel resultId={RID} refinedResultId="refinement_2"
      onSwitchToRefined={onSwitchToRefined} />);

    expect(screen.getByRole('button', { name: 'Re-Refine' }).disabled).toBe(false);
    expect(screen.queryByRole('button', { name: 'Refine R + PC' })).toBeNull();
    screen.getByRole('button', { name: 'Open Refined' }).click();
    expect(onSwitchToRefined).toHaveBeenCalled();
  });

  it('reads the stored summary once to show it (lambda, timings, pixel counts)', async () => {
    summary.mockResolvedValue({
      smoothness_lambda: 0.02, computed_at: '2026-10-23T09:00:00',
      refined_result_id: 'refinement_2',
      stage_timings: { stage_1: 30, stage_2: 5, stage_3: 60 },
      summary: { n_pixels_refined: 400, n_converged: 398, n_failed: 2, ncc_improvement: null },
    });
    const { container } = render(<RefinementPanel resultId={RID} refinedResultId="refinement_2" />);

    await waitFor(() => expect(container.textContent).toContain('400 pixel'));
    expect(summary).toHaveBeenCalledTimes(1);
    expect(summary).toHaveBeenCalledWith(RID);
  });

  it('takes the stored lambda after a reload, so Open Refined stays offered', async () => {
    // The slider starts at its default (0.01); the stored refinement used 0.02.
    // With the two apart the panel offers only Re-Smooth, and the way to the
    // refined result is gone before the user has touched anything.
    summary.mockResolvedValue({
      smoothness_lambda: 0.02, computed_at: '2026-10-23T09:00:00',
      refined_result_id: 'refinement_2',
      stage_timings: { stage_1: 1, stage_2: 1, stage_3: 1 },
      summary: { n_pixels_refined: 400, ncc_improvement: null },
    });
    render(<RefinementPanel resultId={RID} refinedResultId="refinement_2" />);

    await waitFor(() =>
      expect(useDataStore.getState().refinementLambdaPending).toBeCloseTo(0.02, 6));
    expect(screen.getByRole('button', { name: 'Open Refined' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Re-Smooth/ })).toBeNull();
  });

  it('takes the stored lambda of each original when the shown original changes (A, B, A)', async () => {
    const stored = { A: 0.02, B: 0.05 };
    summary.mockImplementation((id) => Promise.resolve({
      smoothness_lambda: stored[id], computed_at: 'then', refined_result_id: `refined_${id}`,
      stage_timings: { stage_1: 1, stage_2: 1, stage_3: 1 },
      summary: { n_pixels_refined: 400, ncc_improvement: null },
    }));
    const { rerender } = render(<RefinementPanel resultId="A" refinedResultId="refined_A" />);
    await waitFor(() =>
      expect(useDataStore.getState().refinementLambdaPending).toBeCloseTo(0.02, 6));

    rerender(<RefinementPanel resultId="B" refinedResultId="refined_B" />);
    await waitFor(() =>
      expect(useDataStore.getState().refinementLambdaPending).toBeCloseTo(0.05, 6));

    // back to A: its summary is already in the store, its lambda still applies
    rerender(<RefinementPanel resultId="A" refinedResultId="refined_A" />);
    await waitFor(() =>
      expect(useDataStore.getState().refinementLambdaPending).toBeCloseTo(0.02, 6));
    expect(screen.getByRole('button', { name: 'Open Refined' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Re-Smooth/ })).toBeNull();
  });

  it('without a refined result it offers the plain Refine button', () => {
    render(<RefinementPanel resultId={RID} refinedResultId={null} />);
    expect(screen.getByRole('button', { name: 'Refine R + PC' }).disabled).toBe(false);
    expect(summary).not.toHaveBeenCalled();
  });
});

describe('the note about an unreadable summary', () => {
  it('goes away when the panel is shown for another result', async () => {
    progress.mockResolvedValue({ state: 'completed', progress: 1 });
    summary.mockRejectedValue(new Error('Internal Server Error'));
    useDataStore.setState({ refinementJob: {
      result_id: RID, job_id: 'job1', state: 'running', progress: 0.9,
      current_stage: 'stage_3', outer_iter: 1,
    } });
    const { rerender } = render(<RefinementPanel resultId={RID} />);
    await screen.findByText(/The refinement finished/, {}, { timeout: 3000 });

    rerender(<RefinementPanel resultId="another_result" />);

    expect(screen.queryByText(/The refinement finished/)).toBeNull();
  });
});

describe('a job that has completed', () => {
  function startedJob() {
    useDataStore.setState({ refinementJob: {
      result_id: RID, job_id: 'job1', state: 'running', progress: 0.9,
      current_stage: 'stage_3', outer_iter: 1,
    } });
  }

  it('is recorded with its summary when the summary can be read', async () => {
    progress.mockResolvedValue({ state: 'completed', progress: 1 });
    summary.mockResolvedValue({
      smoothness_lambda: 0.01, computed_at: 'now', refined_result_id: 'refinement_2',
      stage_timings: { stage_1: 1, stage_2: 1, stage_3: 1 },
      summary: { n_pixels_refined: 400, ncc_improvement: null },
    });
    startedJob();
    render(<RefinementPanel resultId={RID} />);

    await waitFor(() =>
      expect(useDataStore.getState().refinementComputed[RID]?.refined_result_id)
        .toBe('refinement_2'), { timeout: 3000 });
    expect(useDataStore.getState().refinementJob).toBeNull();
  });

  it('is not reported as a failed refinement when only the summary request fails', async () => {
    progress.mockResolvedValue({ state: 'completed', progress: 1 });
    summary.mockRejectedValue(new Error('Internal Server Error'));
    startedJob();
    const { container } = render(<RefinementPanel resultId={RID} />);

    await waitFor(() => expect(useDataStore.getState().refinementJob).toBeNull(),
      { timeout: 3000 });
    // The job's own state is the truth: it finished. Nothing says it failed,
    // and the server's error text is not shown as the refinement's error.
    expect(container.textContent).not.toMatch(/Internal Server Error/);
    expect(screen.getByText(/The refinement finished/)).toBeTruthy();
    // ... and it is not recorded as computed, since its summary was not read
    expect(useDataStore.getState().refinementComputed[RID]).toBeUndefined();
  });

  it('still leaves the way to the refined result when only the summary request fails', async () => {
    // The page knows from the backend's result list that a refined result
    // exists; the panel's "Open Refined" does not depend on the summary.
    progress.mockResolvedValue({ state: 'completed', progress: 1 });
    summary.mockRejectedValue(new Error('Internal Server Error'));
    startedJob();
    const onSwitchToRefined = vi.fn();
    render(<RefinementPanel resultId={RID} refinedResultId="refinement_2"
      onSwitchToRefined={onSwitchToRefined} />);

    const open = await screen.findByRole('button', { name: 'Open Refined' }, { timeout: 3000 });
    expect(open.disabled).toBe(false);
    screen.getByRole('button', { name: 'Open Refined' }).click();
    expect(onSwitchToRefined).toHaveBeenCalled();
    // the note promises nothing that is not on screen
    expect(screen.getByText(/The refinement finished/).textContent)
      .not.toMatch(/Refined view|run the refinement again/);
  });

  it('is still reported as failed when the job itself failed', async () => {
    progress.mockResolvedValue({ state: 'failed', error: 'solver exploded' });
    startedJob();
    render(<RefinementPanel resultId={RID} />);

    expect(await screen.findByText('solver exploded', {}, { timeout: 3000 })).toBeTruthy();
    expect(screen.queryByText(/The refinement finished/)).toBeNull();
  });
});
