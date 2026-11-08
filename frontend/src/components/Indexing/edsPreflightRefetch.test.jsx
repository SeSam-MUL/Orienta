// @vitest-environment jsdom
/**
 * The EDS pre-flight asks again when the spectrum fit is done.
 *
 * While the spectra of a file are being evaluated the pre-flight answers
 * "blocked" (`eds_present`, with the fit's own message) -- correctly: there
 * are no EDS numbers yet. It used to be asked only when its INPUTS changed
 * (dataset, phase list, the EDS switch), so after the fit finished the Start
 * button stayed blocked on a verdict about a state that no longer existed,
 * until the user happened to touch a phase.
 *
 * `useEdsPreflight` is exported from IndexingPage.jsx for this test, the
 * same way `useAllowedPaths` is: the page itself is never rendered in tests.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

const edsPreflight = vi.fn();
vi.mock('../../services/api', () => ({
  indexApi: { edsPreflight: (...a) => edsPreflight(...a) },
}));

const { useEdsPreflight } = await import('./IndexingPage');

const BLOCKED = { data: { can_index: false, eds_usable: false, phases: [],
  blocking: ['eds_present'], checks: { eds_present: {
    severity: 'block', ok: false, code: 'edsUnavailable', params: {},
    detail: 'The spectra of this map are still being evaluated (40 %).' } } } };
const READY = { data: { can_index: true, eds_usable: true, phases: [],
  blocking: [], checks: {} } };

/** Past the 300 ms debounce, and let the answer land. */
const debounce = async () => {
  await act(async () => { vi.advanceTimersByTime(350); });
  await act(async () => { await Promise.resolve(); await Promise.resolve(); });
};

beforeEach(() => {
  vi.useFakeTimers();
  edsPreflight.mockReset();
});
afterEach(() => { vi.useRealTimers(); });

const props = (fitState) => ({
  enabled: true, dataset: 'Scan1', phaseKey: '/lib/Al.cif\n/lib/Si.cif', fitState,
});

describe('useEdsPreflight', () => {
  it('asks again when the spectrum fit turns ready, and Start is no longer blocked', async () => {
    edsPreflight.mockResolvedValueOnce(BLOCKED).mockResolvedValueOnce(READY);
    const { result, rerender } = renderHook((p) => useEdsPreflight(p),
      { initialProps: props('running') });
    await debounce();
    expect(edsPreflight).toHaveBeenCalledTimes(1);
    expect(edsPreflight).toHaveBeenCalledWith(['/lib/Al.cif', '/lib/Si.cif']);
    expect(result.current.result.can_index).toBe(false);

    rerender(props('ready'));
    await debounce();
    expect(edsPreflight).toHaveBeenCalledTimes(2);
    expect(result.current.result.can_index).toBe(true);
  });

  it('does not ask again while the fit keeps running', async () => {
    edsPreflight.mockResolvedValue(BLOCKED);
    const { rerender } = renderHook((p) => useEdsPreflight(p),
      { initialProps: props('running') });
    await debounce();
    rerender(props('running'));
    await debounce();
    expect(edsPreflight).toHaveBeenCalledTimes(1);
  });

  it('asks nothing while disabled, and clears what it had', async () => {
    edsPreflight.mockResolvedValue(READY);
    const { result, rerender } = renderHook((p) => useEdsPreflight(p),
      { initialProps: props('ready') });
    await debounce();
    expect(result.current.result).not.toBeNull();
    rerender({ ...props('ready'), enabled: false });
    await debounce();
    expect(edsPreflight).toHaveBeenCalledTimes(1);
    expect(result.current.result).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it('a failed check degrades to an error string and never blocks', async () => {
    edsPreflight.mockRejectedValue({ message: 'Network Error' });
    const { result } = renderHook((p) => useEdsPreflight(p),
      { initialProps: props('none') });
    await debounce();
    expect(result.current.result).toBeNull();
    expect(result.current.error).toBe('Network Error');
  });
});

describe('IndexingPage wires the fit state into the pre-flight', () => {
  it('passes the spectrum fit status to useEdsPreflight', async () => {
    // The hook tests above would stay green with the wiring deleted; this
    // reads the page.
    const fs = await import('node:fs');
    const path = await import('node:path');
    const src = fs.readFileSync(
      path.resolve(process.cwd(), 'src/components/Indexing/IndexingPage.jsx'), 'utf8');
    expect(src).toMatch(/useSpectrumFitStatus\(/);
    // The call, not the definition (`function useEdsPreflight({ ... })`).
    const call = src.match(/=\s*useEdsPreflight\(\{([\s\S]*?)\}\)/);
    expect(call, 'useEdsPreflight is not called in the page').toBeTruthy();
    expect(call[1]).toMatch(/fitState:\s*\w+\.state/);
  });
});
