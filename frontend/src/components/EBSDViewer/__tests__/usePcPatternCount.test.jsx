/**
 * The number on "Open PC Refinement" follows the backend's list.
 *
 * Regression: removing a pattern on the PC Refinement page left the
 * old number on the viewer's button, because only Add ever set it. These cases
 * pin the repair: the number is asked for again whenever the viewer page is
 * shown, and an answer to an older question cannot overwrite a newer number.
 */
// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';

// A plain function, not a vi.fn: a spy follows every promise it returns, and a
// promise it follows that rejects is reported as unhandled even though the hook
// catches its own copy.
let answer = () => Promise.resolve({ data: { patterns: [] } });
let asked = 0;
vi.mock('../../../services/api', () => ({
  pcApi: { status: () => { asked += 1; return answer(); } },
}));

const { usePcPatternCount } = await import('../usePcPatternCount');

const reply = (n) => ({ data: { patterns: Array.from({ length: n }, (_, i) => ({ index: i })) } });
const answerWith = (n) => { answer = () => Promise.resolve(reply(n)); };
const showPage = (page) => act(() => {
  window.dispatchEvent(new CustomEvent('page-changed', { detail: { page } }));
});

beforeEach(() => { asked = 0; answerWith(0); });
// No global auto-cleanup in this config: a hook left mounted would keep
// listening and answer the next case's page events.
afterEach(() => cleanup());

describe('usePcPatternCount', () => {
  it('starts from the backend list', async () => {
    answerWith(3);
    const { result } = renderHook(() => usePcPatternCount('ebsdviewer'));
    await waitFor(() => expect(result.current[0]).toBe(3));
  });

  it('follows a removal made on the PC Refinement page when the viewer is shown again', async () => {
    answerWith(3);
    const { result } = renderHook(() => usePcPatternCount('ebsdviewer'));
    await waitFor(() => expect(result.current[0]).toBe(3));

    answerWith(2);                        // one removed over there
    showPage('pcrefinement');
    expect(asked).toBe(1);                // other pages do not ask
    showPage('ebsdviewer');
    await waitFor(() => expect(result.current[0]).toBe(2));
  });

  it('shows Add at once and drops a status answer that was asked before it', async () => {
    let late;
    answer = () => new Promise((res) => { late = res; });
    const { result } = renderHook(() => usePcPatternCount('ebsdviewer'));

    act(() => result.current[1](4));      // Add replied: 4
    expect(result.current[0]).toBe(4);
    await act(async () => { late(reply(3)); });   // the older question answers late
    expect(result.current[0]).toBe(4);
  });

  it('keeps the number when the backend does not answer', async () => {
    answerWith(2);
    const { result } = renderHook(() => usePcPatternCount('ebsdviewer'));
    await waitFor(() => expect(result.current[0]).toBe(2));

    answer = () => Promise.reject(new Error('backend down'));
    showPage('ebsdviewer');
    await act(async () => {});
    expect(asked).toBe(2);
    expect(result.current[0]).toBe(2);
  });

  it('asks again when the active dataset changes while the page stays on screen', async () => {
    answerWith(3);
    const { result, rerender } = renderHook(({ ds }) => usePcPatternCount('ebsdviewer', ds),
      { initialProps: { ds: 'scan' } });
    await waitFor(() => expect(result.current[0]).toBe(3));
    answerWith(0);                         // the processed copy has its own, empty list
    rerender({ ds: 'scan_bg_dyn' });
    await waitFor(() => expect(result.current[0]).toBe(0));
  });

  it('stops listening when the page goes away', () => {
    const { unmount } = renderHook(() => usePcPatternCount('ebsdviewer'));
    unmount();
    showPage('ebsdviewer');
    expect(asked).toBe(1);
  });
});
