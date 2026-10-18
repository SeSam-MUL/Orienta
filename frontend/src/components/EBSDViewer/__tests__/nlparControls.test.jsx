// @vitest-environment jsdom
/**
 * NLPAR controls, the busy-overlay Cancel button, and the hook behind them.
 *
 *   1. The button runs the op and is disabled while another op is busy.
 *   2. The radius field only ever holds 1-3.
 *   3. Cancel is only drawn while NLPAR runs, names the stage, and cannot be
 *      pressed twice.
 *   4. The step cancels with the request id of ITS run, and a cancelled run
 *      tells the caller to stop (the Recommended Pipeline must not go on).
 *   5. A failing run resets the running flag, so the overlay cannot stay stuck.
 */
import '@testing-library/jest-dom/vitest';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react';

vi.mock('../../../services/api', () => ({
  ebsdApi: { nlpar: vi.fn(), nlparCancel: vi.fn(), nlparAdvice: vi.fn() },
}));

import { ebsdApi } from '../../../services/api';
import { NlparControls, NlparOverlayExtras } from '../NlparControls';
import { useNlpar, clampRadius, nlparStepFor } from '../useNlpar';

beforeEach(() => { vi.clearAllMocks(); });
afterEach(() => cleanup());

const fakeNlpar = (over = {}) => ({
  radius: 3, setRadius: vi.fn(), denoiseFirst: false, setDenoiseFirst: vi.fn(),
  running: false, cancelling: false, cancel: vi.fn(),
  advice: { recommendation: 'recommended', applied: false, lam: 0.93, sigma: 19.2, threshold: 1.7 },
  applied: false, canRun: true, ...over,
});

describe('NlparControls', () => {
  it('runs NLPAR when the button is pressed', () => {
    const onRun = vi.fn();
    render(<NlparControls nlpar={fakeNlpar()} disabled={false} onRun={onRun} />);
    fireEvent.click(screen.getByRole('button', { name: 'NLPAR denoise' }));
    expect(onRun).toHaveBeenCalledTimes(1);
  });

  it('is disabled while another operation is busy or nothing is loaded', () => {
    const onRun = vi.fn();
    render(<NlparControls nlpar={fakeNlpar()} disabled onRun={onRun} />);
    const btn = screen.getByRole('button', { name: 'NLPAR denoise' });
    expect(btn).toBeDisabled();
    fireEvent.click(btn);
    expect(onRun).not.toHaveBeenCalled();
  });

  it('hands the typed radius to the state setter', () => {
    const nlpar = fakeNlpar();
    render(<NlparControls nlpar={nlpar} disabled={false} onRun={() => {}} />);
    fireEvent.change(screen.getByLabelText('NLPAR radius:'), { target: { value: '2' } });
    expect(nlpar.setRadius).toHaveBeenCalledWith('2');
  });

  it('offers denoising first in the Recommended Pipeline, off by default', () => {
    const nlpar = fakeNlpar();
    render(<NlparControls nlpar={nlpar} disabled={false} onRun={() => {}} />);
    const box = screen.getByRole('checkbox');
    expect(box).not.toBeChecked();
    fireEvent.click(box);
    expect(nlpar.setDenoiseFirst).toHaveBeenCalledWith(true);
  });
});

describe('NlparControls advice and the applied state', () => {
  it('shows the measured numbers for a recommended scan', () => {
    render(<NlparControls nlpar={fakeNlpar()} disabled={false} onRun={() => {}} />);
    const box = screen.getByTestId('nlpar-advice');
    expect(box).toHaveAttribute('data-advice', 'recommended');
    expect(box.textContent).toContain('Recommended');
    expect(box.textContent).toContain('0.93');
    expect(box.textContent).toContain('19.2');
    expect(box.textContent).toContain('1.7');
  });

  it('says "not recommended" with the numbers, and the button still works', () => {
    const onRun = vi.fn();
    const nlpar = fakeNlpar({ advice: { recommendation: 'not_recommended', applied: false, lam: 2.46, sigma: 28.3, threshold: 1.7 } });
    render(<NlparControls nlpar={nlpar} disabled={false} onRun={onRun} />);
    const box = screen.getByTestId('nlpar-advice');
    expect(box).toHaveAttribute('data-advice', 'notRecommended');
    expect(box.textContent).toContain('Not recommended');
    expect(box.textContent).toContain('2.46');
    fireEvent.click(screen.getByRole('button', { name: 'NLPAR denoise' }));
    expect(onRun).toHaveBeenCalledTimes(1);          // advice, not a lock
  });

  it('an applied dataset shows the state and cannot be run again', () => {
    const onRun = vi.fn();
    const nlpar = fakeNlpar({ advice: { recommendation: 'applied', applied: true, applied_radius: 3 }, applied: true, canRun: false });
    render(<NlparControls nlpar={nlpar} disabled={false} onRun={onRun} />);
    expect(screen.getByTestId('nlpar-advice').textContent).toContain('Already applied');
    expect(screen.getByTestId('nlpar-advice').textContent).toContain('radius 3');
    const btn = screen.getByRole('button', { name: 'NLPAR denoise' });
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute('title', expect.stringContaining('already been applied'));
    fireEvent.click(btn);
    expect(onRun).not.toHaveBeenCalled();
  });

  it('the advice says where it comes from, in a tooltip: an estimate, a few test scans, raw patterns', () => {
    render(<NlparControls nlpar={fakeNlpar()} disabled={false} onRun={() => {}} />);
    const title = screen.getByTestId('nlpar-advice').getAttribute('title');
    expect(title).toMatch(/estimate/i);
    expect(title).toMatch(/few measured test scans/);
    expect(title).toMatch(/raw patterns/);
    expect(title).toMatch(/background removal or CLAHE/);
  });

  it('the advice does not promise: "may help" / "may then blur", not "is likely to improve" / "tends to"', () => {
    const { unmount } = render(<NlparControls nlpar={fakeNlpar()} disabled={false} onRun={() => {}} />);
    expect(screen.getByTestId('nlpar-advice').textContent).toContain('may help');
    expect(screen.getByTestId('nlpar-advice').textContent).not.toMatch(/likely to improve/);
    unmount();
    const nlpar = fakeNlpar({ advice: { recommendation: 'not_recommended', applied: false, lam: 2.46, sigma: 28.3, threshold: 1.7 } });
    render(<NlparControls nlpar={nlpar} disabled={false} onRun={() => {}} />);
    expect(screen.getByTestId('nlpar-advice').textContent).toContain('may then blur');
    expect(screen.getByTestId('nlpar-advice').textContent).not.toMatch(/tends to/);
  });

  it('"denoise first" cannot be ticked while NLPAR cannot run on this dataset', () => {
    render(<NlparControls nlpar={fakeNlpar({ canRun: false, applied: true })} disabled={false} onRun={() => {}} />);
    expect(screen.getByRole('checkbox')).toBeDisabled();
  });

  it('while the advice is being worked out the button waits too', () => {
    render(<NlparControls nlpar={fakeNlpar({ advice: null, canRun: false })} disabled={false} onRun={() => {}} />);
    expect(screen.getByTestId('nlpar-advice')).toHaveAttribute('data-advice', 'checking');
    expect(screen.getByRole('button', { name: 'NLPAR denoise' })).toBeDisabled();
  });

  it.each([['too_small', /too small/], ['hex', /hexagonal/], ['uniform', /too uniform/], ['error', /No advice for this scan/]])(
    'unavailable (%s) is explained', (reason, pattern) => {
      const nlpar = fakeNlpar({ advice: { recommendation: 'unavailable', applied: false, reason } });
      render(<NlparControls nlpar={nlpar} disabled={false} onRun={() => {}} />);
      expect(screen.getByTestId('nlpar-advice').textContent).toMatch(pattern);
    });
});

describe('clampRadius', () => {
  it.each([[0, 1], [1, 1], [2, 2], [3, 3], [4, 3], [-5, 1], ['2', 2], [2.6, 3], ['abc', 3], ['', 1], [null, 1]])(
    '%j -> %j', (input, want) => {
      // '' and null are numbers (0) to Number(); the field would clamp them to 1
      expect(clampRadius(input)).toBe(want);
    });
});

describe('NlparOverlayExtras', () => {
  it('draws nothing when NLPAR is not running (BG removal etc. keep their overlay)', () => {
    const { container } = render(<NlparOverlayExtras nlpar={fakeNlpar({ running: false })} phase="sigma" />);
    expect(container).toBeEmptyDOMElement();
  });

  it('names the stage and cancels', () => {
    const nlpar = fakeNlpar({ running: true });
    render(<NlparOverlayExtras nlpar={nlpar} phase="sigma" />);
    expect(screen.getByText(/noise level of every pattern/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(nlpar.cancel).toHaveBeenCalledTimes(1);
  });

  it('cannot be pressed again while cancelling', () => {
    render(<NlparOverlayExtras nlpar={fakeNlpar({ running: true, cancelling: true })} phase="filter" />);
    expect(screen.getByRole('button', { name: 'Cancelling...' })).toBeDisabled();
  });

  it('falls back to the first stage for a phase it does not know', () => {
    render(<NlparOverlayExtras nlpar={fakeNlpar({ running: true })} phase="scan_average" />);
    expect(screen.getByText(/numerical kernels/)).toBeInTheDocument();
  });
});

describe('useNlpar advice', () => {
  const answer = (over = {}) => ({ data: { recommendation: 'recommended', applied: false, lam: 0.9, sigma: 10, threshold: 1.7, ...over } });

  it('is null (checking) until the answer is in, and then holds it', async () => {
    let resolve;
    ebsdApi.nlparAdvice.mockReturnValue(new Promise((r) => { resolve = r; }));
    const { result } = renderHook(() => useNlpar({ ebsdLoaded: true, activeDataset: 'a', busy: false }));
    expect(result.current.advice).toBeNull();
    expect(result.current.canRun).toBe(false);
    await act(async () => { resolve(answer()); });
    expect(result.current.advice.recommendation).toBe('recommended');
    expect(result.current.canRun).toBe(true);
  });

  it('asks nothing without a loaded file and nothing while an operation runs', async () => {
    ebsdApi.nlparAdvice.mockResolvedValue(answer());
    const { rerender } = renderHook((p) => useNlpar(p), { initialProps: { ebsdLoaded: false, activeDataset: null, busy: false } });
    expect(ebsdApi.nlparAdvice).not.toHaveBeenCalled();
    await act(async () => { rerender({ ebsdLoaded: true, activeDataset: 'a', busy: true }); });
    expect(ebsdApi.nlparAdvice).not.toHaveBeenCalled();
    await act(async () => { rerender({ ebsdLoaded: true, activeDataset: 'a', busy: false }); });
    expect(ebsdApi.nlparAdvice).toHaveBeenCalledTimes(1);
  });

  it('asks again for another dataset and drops the old answer at once', async () => {
    ebsdApi.nlparAdvice.mockResolvedValueOnce(answer({ recommendation: 'applied', applied: true, applied_radius: 3 }));
    const { result, rerender } = renderHook((p) => useNlpar(p), { initialProps: { ebsdLoaded: true, activeDataset: 'a_nlpar', busy: false } });
    await act(async () => {});
    expect(result.current.applied).toBe(true);
    expect(result.current.canRun).toBe(false);

    let resolve;
    ebsdApi.nlparAdvice.mockReturnValue(new Promise((r) => { resolve = r; }));
    await act(async () => { rerender({ ebsdLoaded: true, activeDataset: 'a', busy: false }); });
    expect(result.current.applied).toBe(false);       // not the other dataset's state
    expect(result.current.canRun).toBe(false);        // and no click until the answer is in
    await act(async () => { resolve(answer()); });
    expect(result.current.canRun).toBe(true);
  });

  it('a late answer for a dataset that is no longer active is ignored', async () => {
    let resolveOld;
    ebsdApi.nlparAdvice.mockReturnValueOnce(new Promise((r) => { resolveOld = r; }));
    const { result, rerender } = renderHook((p) => useNlpar(p), { initialProps: { ebsdLoaded: true, activeDataset: 'a', busy: false } });
    ebsdApi.nlparAdvice.mockResolvedValueOnce(answer({ recommendation: 'not_recommended', lam: 3 }));
    await act(async () => { rerender({ ebsdLoaded: true, activeDataset: 'b', busy: false }); });
    expect(result.current.advice.recommendation).toBe('not_recommended');
    await act(async () => { resolveOld(answer({ recommendation: 'applied', applied: true })); });
    expect(result.current.advice.recommendation).toBe('not_recommended');
  });

  it.each(['too_small', 'hex', 'uniform'])('a scan the backend would refuse (%s) cannot be run: no copy, no 400', async (reason) => {
    ebsdApi.nlparAdvice.mockResolvedValue(answer({ recommendation: 'unavailable', reason }));
    const { result } = renderHook(() => useNlpar({ ebsdLoaded: true, activeDataset: 'a', busy: false }));
    await act(async () => {});
    expect(result.current.canRun).toBe(false);
    expect(result.current.applied).toBe(false);
  });

  it('a failing request is "unavailable", never a lock', async () => {
    ebsdApi.nlparAdvice.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => useNlpar({ ebsdLoaded: true, activeDataset: 'a', busy: false }));
    await act(async () => {});
    expect(result.current.advice.recommendation).toBe('unavailable');
    expect(result.current.canRun).toBe(true);
  });

  it('after a successful run the dataset is applied at once, before any new fetch', async () => {
    ebsdApi.nlparAdvice.mockResolvedValue(answer());
    ebsdApi.nlpar.mockResolvedValue({ data: { success: true, lam: 0.9, sigma_median: 10, radius: 2 } });
    const { result } = renderHook(() => useNlpar({ ebsdLoaded: true, activeDataset: 'a', busy: false }));
    await act(async () => {});
    expect(result.current.canRun).toBe(true);
    await act(async () => {
      await result.current.runStep({ timed: (_n, fn) => fn(), withProgress: (call) => call('r'), log: vi.fn() });
    });
    expect(result.current.applied).toBe(true);
    expect(result.current.canRun).toBe(false);
    expect(result.current.advice.applied_radius).toBe(2);
  });
});

describe('nlparStepFor: decided before the auto-copy', () => {
  it.each([
    // op, denoiseFirst, canRun, expected
    ['nlpar', false, true, 'run'],
    ['nlpar', false, false, 'refuse'],
    ['pipeline', true, true, 'run'],
    ['pipeline', true, false, 'skip'],        // the stray-copy case: the pipeline must not run NLPAR into a 400
    ['pipeline', false, true, null],
    ['pipeline', false, false, null],
    ['bg_dyn', true, true, null],
    ['clahe', true, false, null],
  ])('%s, denoiseFirst %s, canRun %s -> %s', (op, denoiseFirst, canRun, want) => {
    expect(nlparStepFor(op, { denoiseFirst, canRun })).toBe(want);
  });
});

describe('useNlpar', () => {
  const deferred = () => {
    let resolve; let reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
  };
  // the viewer's withProgress: makes a request id and hands it to the call
  const withProgress = (rid) => (call) => call(rid);

  it('sends the radius the user chose', async () => {
    ebsdApi.nlpar.mockResolvedValue({ data: { success: true, lam: 0.8, sigma_median: 12.3, radius: 2 } });
    const { result } = renderHook(() => useNlpar());
    act(() => result.current.setRadius(2));
    const log = vi.fn();
    let ok;
    await act(async () => {
      ok = await result.current.runStep({ timed: (_n, fn) => fn(), withProgress: withProgress('r1'), log });
    });
    expect(ok).toBe(true);
    expect(ebsdApi.nlpar).toHaveBeenCalledWith(2, 'r1');
    expect(log).toHaveBeenCalledWith(expect.stringContaining('0.80'));
  });

  it('cancels with the id of the run in flight, and not after it ended', async () => {
    const d = deferred();
    ebsdApi.nlpar.mockReturnValue(d.promise);
    ebsdApi.nlparCancel.mockResolvedValue({ data: { cancelled: true } });
    const { result } = renderHook(() => useNlpar());
    let step;
    act(() => {
      step = result.current.runStep({ timed: (_n, fn) => fn(), withProgress: withProgress('r-live'), log: vi.fn() });
    });
    expect(result.current.running).toBe(true);
    await act(async () => { await result.current.cancel(); });
    expect(ebsdApi.nlparCancel).toHaveBeenCalledWith('r-live');
    expect(result.current.cancelling).toBe(true);

    await act(async () => { d.resolve({ data: { success: false, cancelled: true } }); await step; });
    expect(result.current.running).toBe(false);
    expect(result.current.cancelling).toBe(false);

    ebsdApi.nlparCancel.mockClear();
    await act(async () => { await result.current.cancel(); });
    expect(ebsdApi.nlparCancel).not.toHaveBeenCalled();
  });

  it('a cancelled run says so and tells the caller to stop', async () => {
    ebsdApi.nlpar.mockResolvedValue({ data: { success: false, cancelled: true } });
    const { result } = renderHook(() => useNlpar());
    const log = vi.fn();
    let ok;
    await act(async () => {
      ok = await result.current.runStep({ timed: (_n, fn) => fn(), withProgress: withProgress('r2'), log });
    });
    expect(ok).toBe(false);
    expect(log).toHaveBeenCalledWith(expect.stringContaining('cancelled'));
  });

  it('a failing run resets the running flag and passes the error on', async () => {
    ebsdApi.nlpar.mockRejectedValue(new Error('409'));
    const { result } = renderHook(() => useNlpar());
    await act(async () => {
      await expect(result.current.runStep({
        timed: (_n, fn) => fn(), withProgress: withProgress('r3'), log: vi.fn(),
      })).rejects.toThrow('409');
    });
    expect(result.current.running).toBe(false);
  });
});
