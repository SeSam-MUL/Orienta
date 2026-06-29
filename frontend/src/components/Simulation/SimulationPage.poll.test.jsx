// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { isActiveJobStatus } from './SimulationPage';

// Regression guard for the "Queue froze at 1 done · N pending" bug (2026-06-23).
// The batch poller / counters must treat a job as ACTIVE while it is 'pending'
// (the status the backend gives queued batch jobs) — not only 'running'/'queued'.
// Dropping 'pending' made polling stop the moment the snapshot held only
// completed + pending jobs, so a fully-completing backend batch looked dead-ended.
describe('isActiveJobStatus', () => {
  it('treats pending/running/queued as active (keep polling)', () => {
    expect(isActiveJobStatus('pending')).toBe(true);   // <-- the bug: must be true
    expect(isActiveJobStatus('running')).toBe(true);
    expect(isActiveJobStatus('queued')).toBe(true);
  });

  it('treats terminal states as inactive (stop polling)', () => {
    for (const s of ['completed', 'failed', 'cancelled', 'stopped', '', undefined, null]) {
      expect(isActiveJobStatus(s)).toBe(false);
    }
  });

  it('a snapshot of [completed, pending, pending] still has active work', () => {
    const jobs = [{ status: 'completed' }, { status: 'pending' }, { status: 'pending' }];
    expect(jobs.some(j => isActiveJobStatus(j.status))).toBe(true);
    expect(jobs.filter(j => isActiveJobStatus(j.status)).length).toBe(2);
  });
});
