// @vitest-environment jsdom
//
// The note under the Indexing page follows the ACTIVE result: a run with a note shows
// it, a run without one removes it, and a switch or an import of another result
// (which only changes the backend's state version) shows that result's notes.
// The hook runs with the real banner and the real locale files; only the two
// requests are replaced.
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, act } from '@testing-library/react';
import i18n from '../../i18n';

const version = vi.fn();
const lastResult = vi.fn();
vi.mock('../../services/api', () => ({
  stateApi: { version: () => version() },
  indexApi: { getLastResult: () => lastResult() },
}));

import ResultCaveats from './ResultCaveats';
import useActiveResultCaveats from './useActiveResultCaveats';

const NOTE = { code: 'houghTrigonalSetting', phases: ['Cr2N'], text: 'backend text' };
const withNote = { available: true, caveats: [NOTE] };
const withoutNote = { available: true };
const snap = (v, id) => ({ data: { version: v, active_result_id: id } });

let api; // what the page gets back from the hook
function Page({ enabled = true }) {
  api = useActiveResultCaveats({ enabled, intervalMs: 100 });
  return <ResultCaveats caveats={api.caveats} />;
}
const banner = (c) => c.querySelector('[role="status"]');
const tick = (ms = 100) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

beforeEach(async () => {
  await i18n.changeLanguage('en');
  vi.useFakeTimers();
  version.mockReset();
  lastResult.mockReset();
});
afterEach(() => { cleanup(); vi.useRealTimers(); });

describe('the banner follows the active result', () => {
  it('shows the note of the result that is active when the page opens', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    lastResult.mockResolvedValue({ data: withNote });
    const { container } = render(<Page />);
    await tick(0);
    expect(banner(container).textContent).toContain('Phase Cr2N');
    expect(lastResult).toHaveBeenCalledTimes(1);
  });

  it('asks again only when the state version or the active id changes', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    lastResult.mockResolvedValue({ data: withNote });
    render(<Page />);
    await tick(0);
    await tick(); await tick(); await tick();
    expect(lastResult).toHaveBeenCalledTimes(1);
    version.mockResolvedValue(snap(2, 'r1'));
    await tick();
    expect(lastResult).toHaveBeenCalledTimes(2);
  });

  it('a switch to a result without a note removes the banner', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    lastResult.mockResolvedValue({ data: withNote });
    const { container } = render(<Page />);
    await tick(0);
    expect(banner(container)).not.toBeNull();
    version.mockResolvedValue(snap(2, 'r2'));
    lastResult.mockResolvedValue({ data: withoutNote });
    await tick();
    expect(banner(container)).toBeNull();
  });

  it('a switch or an import to a result with a note shows it', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    lastResult.mockResolvedValue({ data: withoutNote });
    const { container } = render(<Page />);
    await tick(0);
    expect(banner(container)).toBeNull();
    version.mockResolvedValue(snap(2, 'imported'));
    lastResult.mockResolvedValue({ data: withNote });
    await tick();
    expect(banner(container).textContent).toContain('Phase Cr2N');
  });

  it('no active result at all leaves no banner', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    lastResult.mockResolvedValue({ data: withNote });
    const { container } = render(<Page />);
    await tick(0);
    version.mockResolvedValue(snap(2, null));
    lastResult.mockResolvedValue({ data: { available: false } });
    await tick();
    expect(banner(container)).toBeNull();
  });

  it('a run: clear at the start, the answer after it decides', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    lastResult.mockResolvedValue({ data: withoutNote });
    const { container } = render(<Page />);
    await tick(0);
    // run with a note
    act(() => api.clear());
    expect(banner(container)).toBeNull();
    act(() => api.apply(withNote));
    expect(banner(container).textContent).toContain('Phase Cr2N');
    // next run without one
    act(() => api.clear());
    act(() => api.apply(withoutNote));
    expect(banner(container)).toBeNull();
  });

  it('an answer that is still on its way does not undo a clear', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    let release;
    lastResult.mockReturnValue(new Promise((resolve) => { release = resolve; }));
    const { container } = render(<Page />);
    await tick(0);
    act(() => api.clear()); // a run starts
    await act(async () => { release({ data: withNote }); });
    expect(banner(container)).toBeNull();
  });

  it('a slow /result/last answer over several ticks arrives and shows the banner', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    // every request takes 350 ms, i.e. more than three ticks of 100 ms
    lastResult.mockImplementation(
      () => new Promise((resolve) => { setTimeout(() => resolve({ data: withNote }), 350); }),
    );
    const { container } = render(<Page />);
    await tick(0);
    await tick(); await tick(); await tick();
    expect(banner(container)).toBeNull();
    expect(lastResult).toHaveBeenCalledTimes(1); // no second request while the first runs
    await tick(); // 400 ms: the answer is in
    expect(banner(container).textContent).toContain('Phase Cr2N');
    await tick(); await tick();
    expect(lastResult).toHaveBeenCalledTimes(1); // same key: nothing more to ask
  });

  it('a failing /result/last leaves the page usable and the banner empty, then recovers', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    lastResult.mockResolvedValue({ data: withNote });
    const { container } = render(<Page />);
    await tick(0);
    expect(banner(container)).not.toBeNull();
    version.mockResolvedValue(snap(2, 'r2'));
    lastResult.mockRejectedValue(new Error('500'));
    await tick();
    expect(banner(container)).toBeNull();
    lastResult.mockResolvedValue({ data: withNote });
    await tick(); // same version, but the last answer was not got: ask again
    expect(banner(container).textContent).toContain('Phase Cr2N');
  });

  it('a failing state-version request changes nothing', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    lastResult.mockResolvedValue({ data: withNote });
    const { container } = render(<Page />);
    await tick(0);
    version.mockRejectedValue(new Error('down'));
    await tick(); await tick();
    expect(banner(container)).not.toBeNull();
  });

  it('does not ask while disabled, and asks at once when it is enabled again', async () => {
    version.mockResolvedValue(snap(1, 'r1'));
    lastResult.mockResolvedValue({ data: withNote });
    const { container, rerender } = render(<Page enabled={false} />);
    await tick(0); await tick();
    expect(version).not.toHaveBeenCalled();
    rerender(<Page enabled />);
    await tick(0);
    expect(banner(container).textContent).toContain('Phase Cr2N');
  });
});
