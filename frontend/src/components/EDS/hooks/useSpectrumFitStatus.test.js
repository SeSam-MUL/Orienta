// @vitest-environment jsdom
/**
 * The spectrum fit's state, as the EDS page and the indexing page see it.
 *
 * A file with a spectrum cube serves no EDS numbers until its spectra are
 * evaluated (2-6 s on the test maps, 38.5 s on a 29 GB file). The
 * pages ask `GET /api/eds/spectrum-fit/status` and poll it every 500 ms while
 * the answer is "running" -- and must STOP once it is not: a poll that runs on
 * after "ready" is two requests a second for as long as the app is open.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

const spectrumFitStatus = vi.fn();
vi.mock('../../../services/api', () => ({
  edsApi: { spectrumFitStatus: (...a) => spectrumFitStatus(...a) },
}));

import { useSpectrumFitStatus, FIT_POLL_MS, FIT_RETRY_MS } from './useSpectrumFitStatus';

const answer = (state, extra = {}) => ({
  data: { state, progress: 0, stage: null, error: null, elapsed_s: 0.1,
          fit: null, ...extra },
});

/** Let the pending status promise settle. */
const settle = () => act(async () => { await Promise.resolve(); await Promise.resolve(); });
/** Advance the poll clock one step and let the answer land. */
const step = async () => {
  await act(async () => { vi.advanceTimersByTime(FIT_POLL_MS); });
  await settle();
};

beforeEach(() => {
  vi.useFakeTimers();
  spectrumFitStatus.mockReset();
});
afterEach(() => { vi.useRealTimers(); });

describe('useSpectrumFitStatus', () => {
  it('polls every 500 ms while the fit runs and stops on ready', async () => {
    spectrumFitStatus
      .mockResolvedValueOnce(answer('running', { progress: 0.1 }))
      .mockResolvedValueOnce(answer('running', { progress: 0.6 }))
      .mockResolvedValueOnce(answer('ready', { progress: 1,
        fit: { model_version: 'eds-spectrum-fit/1', detected_not_exported: [] } }));
    const { result } = renderHook(() => useSpectrumFitStatus('/a.h5oina'));
    expect(FIT_POLL_MS).toBe(500);
    await settle();
    expect(result.current.state).toBe('running');
    expect(result.current.progress).toBeCloseTo(0.1);
    await step();
    expect(result.current.progress).toBeCloseTo(0.6);
    await step();
    expect(result.current.state).toBe('ready');
    expect(result.current.fit.model_version).toBe('eds-spectrum-fit/1');
    expect(spectrumFitStatus).toHaveBeenCalledTimes(3);
    // Long after: not one more request.
    for (let i = 0; i < 10; i += 1) await step();
    expect(spectrumFitStatus).toHaveBeenCalledTimes(3);
  });

  it('stops on failed and on none as well', async () => {
    for (const final of ['failed', 'none']) {
      spectrumFitStatus.mockReset();
      spectrumFitStatus
        .mockResolvedValueOnce(answer('running'))
        .mockResolvedValueOnce(answer(final, { error: final === 'failed' ? 'boom' : null }));
      const { result, unmount } = renderHook(() => useSpectrumFitStatus('/b.h5oina'));
      await settle();
      await step();
      expect(result.current.state).toBe(final);
      if (final === 'failed') expect(result.current.error).toBe('boom');
      for (let i = 0; i < 5; i += 1) await step();
      expect(spectrumFitStatus).toHaveBeenCalledTimes(2);
      unmount();
    }
  });

  it('asks nothing without a file, and is unknown until the first answer', async () => {
    const { result, rerender } = renderHook(
      ({ p }) => useSpectrumFitStatus(p), { initialProps: { p: null } });
    await settle();
    expect(spectrumFitStatus).not.toHaveBeenCalled();
    expect(result.current.state).toBeNull();

    let release;
    spectrumFitStatus.mockReturnValueOnce(new Promise((r) => { release = r; }));
    rerender({ p: '/c.h5oina' });
    // In flight: unknown, which the page must treat as "not ready yet".
    expect(result.current.state).toBeNull();
    await act(async () => { release(answer('ready')); });
    await settle();
    expect(result.current.state).toBe('ready');
  });

  it('a new file starts from unknown, not from the old file\'s answer', async () => {
    spectrumFitStatus.mockResolvedValueOnce(answer('ready'));
    const { result, rerender } = renderHook(
      ({ p }) => useSpectrumFitStatus(p), { initialProps: { p: '/one.h5oina' } });
    await settle();
    expect(result.current.state).toBe('ready');
    spectrumFitStatus.mockReturnValueOnce(new Promise(() => {}));
    rerender({ p: '/two.h5oina' });
    expect(result.current.state).toBeNull();
  });

  it('does not poll while inactive, and asks again when it becomes active', async () => {
    spectrumFitStatus.mockResolvedValue(answer('running'));
    const { rerender } = renderHook(
      ({ a }) => useSpectrumFitStatus('/d.h5oina', { active: a }),
      { initialProps: { a: false } });
    await settle();
    await step();
    expect(spectrumFitStatus).not.toHaveBeenCalled();
    rerender({ a: true });
    await settle();
    expect(spectrumFitStatus).toHaveBeenCalledTimes(1);
    await step();
    expect(spectrumFitStatus).toHaveBeenCalledTimes(2);
    // Hidden again: the poll stops with it.
    rerender({ a: false });
    for (let i = 0; i < 5; i += 1) await step();
    expect(spectrumFitStatus).toHaveBeenCalledTimes(2);
  });

  /** No answer at all: a timeout or a dropped connection (axios sets no `response`). */
  const noAnswer = () => Object.assign(new Error('timeout of 4000ms exceeded'),
    { code: 'ECONNABORTED' });

  it('an unanswered first request is never read as "no spectra": it keeps asking', async () => {
    // Read as "none", the page would mount its tabs on a file whose fit may be
    // running: element list and default layers load EMPTY on the 503s and
    // nothing recovers -- the failure the gate exists to prevent.
    spectrumFitStatus
      .mockRejectedValueOnce(noAnswer())
      .mockRejectedValueOnce(noAnswer())
      .mockResolvedValueOnce(answer('running', { progress: 0.2 }));
    const { result } = renderHook(() => useSpectrumFitStatus('/e.h5oina'));
    await settle();
    expect(result.current.state).toBeNull();
    expect(result.current.unreachable).toBe(true);
    await act(async () => { vi.advanceTimersByTime(FIT_RETRY_MS); });
    await settle();
    expect(spectrumFitStatus).toHaveBeenCalledTimes(2);
    expect(result.current.state).toBeNull();
    await act(async () => { vi.advanceTimersByTime(FIT_RETRY_MS); });
    await settle();
    expect(spectrumFitStatus).toHaveBeenCalledTimes(3);
    expect(result.current.state).toBe('running');
    expect(result.current.unreachable).toBe(false);
    expect(FIT_RETRY_MS).toBe(1000);
  });

  it('a 5xx is not an answer either', async () => {
    spectrumFitStatus
      .mockRejectedValueOnce({ message: 'Request failed with status code 502',
        response: { status: 502, data: {} } })
      .mockResolvedValueOnce(answer('ready'));
    const { result } = renderHook(() => useSpectrumFitStatus('/g.h5oina'));
    await settle();
    expect(result.current.state).toBeNull();
    expect(result.current.unreachable).toBe(true);
    await act(async () => { vi.advanceTimersByTime(FIT_RETRY_MS); });
    await settle();
    expect(result.current.state).toBe('ready');
  });

  it('a 4xx IS an answer: the backend has no open file, so there is no fit to wait for', async () => {
    // `spectrum_fit_status` answers 400 "No file open"; asking again every
    // second would not change that, and nothing is running behind it.
    spectrumFitStatus.mockRejectedValueOnce({ message: 'Request failed with status code 400',
      response: { status: 400, data: { detail: 'No file open' } } });
    const { result } = renderHook(() => useSpectrumFitStatus('/h.h5oina'));
    await settle();
    expect(result.current.state).toBe('none');
    expect(result.current.unreachable).toBe(false);
    for (let i = 0; i < 5; i += 1) await step();
    expect(spectrumFitStatus).toHaveBeenCalledTimes(1);
  });

  it('a running fit whose status stops answering stays running, says so, and recovers', async () => {
    spectrumFitStatus
      .mockResolvedValueOnce(answer('running', { progress: 0.3 }))
      .mockRejectedValueOnce(noAnswer())
      .mockResolvedValueOnce(answer('ready'));
    const { result } = renderHook(() => useSpectrumFitStatus('/f.h5oina'));
    await settle();
    expect(result.current.state).toBe('running');
    expect(result.current.unreachable).toBe(false);
    await step();
    expect(result.current.state).toBe('running');
    expect(result.current.progress).toBeCloseTo(0.3);
    expect(result.current.unreachable).toBe(true);
    await act(async () => { vi.advanceTimersByTime(FIT_RETRY_MS); });
    await settle();
    expect(result.current.state).toBe('ready');
    expect(result.current.unreachable).toBe(false);
  });
});
