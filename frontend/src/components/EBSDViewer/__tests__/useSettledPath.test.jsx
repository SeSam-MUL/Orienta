/**
 * The settle behind the lasso mask.
 *
 * The case that matters is the SLOW drag. A trailing-edge debounce on the
 * points passes "a fast drag does not rebuild" and still stutters through every
 * pause of a careful trace — and each stutter costs more than the last, since
 * the mask is rebuilt from the whole path. So these cases pin the property that
 * separates the two shapes: a hand crossing a new pixel MORE SLOWLY than the
 * interval must still not settle while the drag is live.
 *
 * The sequence used throughout is the viewer's real one: mouse-down publishes a
 * one-point path and turns the drag on together, and every later point arrives
 * with the drag already live.
 */
// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useSettledPath } from '../useSettledPath';

const MS = 1000;
const START = [{ r: 0, c: 0 }];

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

/** Mount idle, then begin a drag at START — what mouse-down does. */
function beginDrag() {
  const hook = renderHook(
    ({ v, d }) => useSettledPath(v, d, MS),
    { initialProps: { v: [], d: false } },
  );
  hook.rerender({ v: START, d: true });
  return hook;
}

describe('useSettledPath', () => {
  it('starts settled on whatever it is given', () => {
    const path = [{ r: 1, c: 1 }];
    const { result } = renderHook(() => useSettledPath(path, false, MS));
    expect(result.current).toBe(path);
  });

  it('freezes where this drag began, not on the previous shape', () => {
    const before = [{ r: 9, c: 9 }, { r: 9, c: 12 }];
    const { result, rerender } = renderHook(
      ({ v, d }) => useSettledPath(v, d, MS),
      { initialProps: { v: before, d: false } },
    );
    expect(result.current).toBe(before);

    rerender({ v: START, d: true });
    // Not `before`: that shape is no longer drawn, and reporting it against the
    // new drag's box would describe a selection nobody can see.
    expect(result.current).toEqual(START);

    rerender({ v: [...START, { r: 0, c: 1 }, { r: 1, c: 1 }], d: true });
    expect(result.current).toEqual(START);
  });

  it('never settles again when the drag is shorter than the backstop', () => {
    const { result, rerender } = renderHook(
      ({ v, d }) => useSettledPath(v, d, MS),
      { initialProps: { v: [], d: false } },
    );
    const path = [...START];
    rerender({ v: [...path], d: true });
    for (let i = 1; i < 5; i += 1) {
      path.push({ r: i, c: i });
      rerender({ v: [...path], d: true });
      act(() => { vi.advanceTimersByTime(MS / 10); });
      expect(result.current).toEqual(START);
    }
  });

  it('a slow trace rebuilds at most once, however long it lasts', () => {
    // THE case this hook exists for. A careful hand crosses a new pixel more
    // slowly than the backstop, five times over. A debounce keyed on the gap
    // between points would settle on EVERY one of them — five full rebuilds,
    // each dearer than the last. Keyed on the drag, the backstop fires once and
    // is never rearmed.
    const { result, rerender } = renderHook(
      ({ v, d }) => useSettledPath(v, d, MS),
      { initialProps: { v: [], d: false } },
    );
    const path = [...START];
    rerender({ v: [...path], d: true });
    const seen = [result.current];
    for (let i = 1; i < 6; i += 1) {
      path.push({ r: i, c: i });
      rerender({ v: [...path], d: true });
      act(() => { vi.advanceTimersByTime(MS * 1.5); });
      if (seen[seen.length - 1] !== result.current) seen.push(result.current);
    }
    // The value at drag start, plus ONE backstop firing — not one per point.
    expect(seen.length).toBe(2);
    // And it stopped tracking the path: the points added after the backstop
    // fired are not in the settled value.
    expect(result.current.length).toBeLessThan(path.length);
  });

  it('a new point does NOT postpone the backstop', () => {
    // The opposite of a debounce, and the whole reason a slow trace is safe:
    // the timer belongs to the drag, so drawing on can neither push it out of
    // reach nor re-arm it.
    const { result, rerender } = renderHook(
      ({ v, d }) => useSettledPath(v, d, MS),
      { initialProps: { v: [], d: false } },
    );
    rerender({ v: START, d: true });
    act(() => { vi.advanceTimersByTime(MS - 1); });
    const grown = [...START, { r: 1, c: 1 }];
    rerender({ v: grown, d: true });
    act(() => { vi.advanceTimersByTime(2); });    // MS since the drag STARTED
    expect(result.current).toEqual(grown);        // fired on schedule

    const after = [...grown, { r: 2, c: 2 }];     // ...and never again
    rerender({ v: after, d: true });
    act(() => { vi.advanceTimersByTime(MS * 10); });
    expect(result.current).toEqual(grown);
  });

  it('settles the instant the drag ends — no timer to race', () => {
    const { result, rerender } = beginDrag();
    const drawn = [...START, { r: 2, c: 2 }];
    rerender({ v: drawn, d: true });
    expect(result.current).toEqual(START);
    rerender({ v: drawn, d: false });   // mouse-up, no clock advance at all
    expect(result.current).toBe(drawn);
  });

  it('settles an abandoned drag after the backstop', () => {
    // The release happened somewhere we never heard about, so `drawing` is
    // stuck true. The path stops growing; the backstop finishes the job.
    const { result, rerender } = beginDrag();
    const drawn = [...START, { r: 3, c: 3 }];
    rerender({ v: drawn, d: true });
    act(() => { vi.advanceTimersByTime(MS - 1); });
    expect(result.current).toEqual(START);
    act(() => { vi.advanceTimersByTime(2); });
    expect(result.current).toBe(drawn);
  });

  it('follows the value freely while no drag is in progress', () => {
    // Clearing the selection must take effect at once, not after a backstop.
    const { result, rerender } = renderHook(
      ({ v, d }) => useSettledPath(v, d, MS),
      { initialProps: { v: [{ r: 4, c: 4 }], d: false } },
    );
    rerender({ v: [], d: false });
    expect(result.current).toEqual([]);
  });
});
