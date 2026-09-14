// @vitest-environment jsdom
/**
 * A shift-drag must average the rectangle the user just drew.
 *
 * The page wrote the four Region Average fields and then fired the compute on
 * a `setTimeout(..., 0)` "so the state updates have a chance to land". They do
 * — but a timeout defers WHEN a function runs, not WHICH function: the one it
 * held was bound in the render before the drag, so it read the pre-drag
 * values. The first drag of a session averaged the 0,0,0,0 default, every
 * later one averaged the previous rectangle, while the fields and the dashed
 * outline showed the new one.
 *
 * The drag is driven through the real `useRectangleDrag`, so what arrives here
 * is what the pointer produces, not a hand-written rectangle.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';

const regionQuantify = vi.fn(() => Promise.resolve({ data: { data: {} } }));
vi.mock('../../../services/api', () => ({
  edsApi: { regionQuantify: (...a) => regionQuantify(...a) },
}));

import { useRegionAverage } from './useRegionAverage';
import { useRectangleDrag } from './useRectangleDrag';

const t = (k) => k;

// A pointer event on a 100x100 host showing a 100x100 grid 1:1.
function ptr(clientX, clientY, shiftKey = false) {
  return {
    shiftKey, clientX, clientY,
    currentTarget: {
      getBoundingClientRect: () => ({
        left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100,
      }),
    },
  };
}

/** The page's wiring: a drag hook whose onRegion is the region average's. */
function useDragIntoRegionAverage(displayMode = 'at_pct') {
  const region = useRegionAverage(displayMode, t);
  const drag = useRectangleDrag({ shape: [100, 100], onRegion: region.onRegionSelected });
  return { region, drag };
}

function dragRect(result, x0, y0, x1, y1) {
  act(() => {
    result.current.drag.onPointerDown(ptr(x0, y0, true));
    result.current.drag.onPointerMove(ptr(x1, y1));
    result.current.drag.onPointerUp(ptr(x1, y1));
  });
}

beforeEach(() => { regionQuantify.mockClear(); });

describe('shift-drag → Region Average', () => {
  it('averages the rectangle just dragged, twice in a row', async () => {
    const { result } = renderHook(() => useDragIntoRegionAverage());

    dragRect(result, 10, 10, 40, 50);
    // Even a deferred call would have run by now.
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });

    expect(regionQuantify).toHaveBeenCalledTimes(1);
    // (rowStart, rowEnd, colStart, colEnd, mode)
    expect(regionQuantify.mock.calls[0]).toEqual([10, 50, 10, 40, 'at_pct']);

    // The second drag is where the old code re-sent the first one.
    regionQuantify.mockClear();
    dragRect(result, 60, 70, 80, 90);
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });

    expect(regionQuantify).toHaveBeenCalledTimes(1);
    expect(regionQuantify.mock.calls[0]).toEqual([70, 90, 60, 80, 'at_pct']);
  });

  it('shows the dragged rectangle in the fields as well', () => {
    const { result } = renderHook(() => useDragIntoRegionAverage());
    dragRect(result, 10, 10, 40, 50);
    expect({
      rowStart: result.current.region.rowStart, rowEnd: result.current.region.rowEnd,
      colStart: result.current.region.colStart, colEnd: result.current.region.colEnd,
    }).toEqual({ rowStart: 10, rowEnd: 50, colStart: 10, colEnd: 40 });
  });

  it('"Compute" without an argument still averages what the fields hold', async () => {
    const { result } = renderHook(() => useRegionAverage('counts', t));
    act(() => {
      result.current.setRowStart(3); result.current.setRowEnd(9);
      result.current.setColStart(1); result.current.setColEnd(4);
    });
    await act(async () => { await result.current.quantify(); });
    expect(regionQuantify.mock.calls[0]).toEqual([3, 9, 1, 4, 'counts']);
  });

  it('orders the typed fields, so the panel shows what was measured', async () => {
    const { result } = renderHook(() => useRegionAverage('counts', t));
    act(() => {
      result.current.setRowStart(40); result.current.setRowEnd(10);
      result.current.setColStart(90); result.current.setColEnd(20);
    });
    await act(async () => { await result.current.quantify(); });
    expect(regionQuantify.mock.calls[0]).toEqual([10, 40, 20, 90, 'counts']);
    expect({
      rowStart: result.current.rowStart, rowEnd: result.current.rowEnd,
      colStart: result.current.colStart, colEnd: result.current.colEnd,
    }).toEqual({ rowStart: 10, rowEnd: 40, colStart: 20, colEnd: 90 });
  });

  it('leaves an unreadable field alone for the backend to reject', async () => {
    const { result } = renderHook(() => useRegionAverage('counts', t));
    act(() => { result.current.setRowStart(''); result.current.setRowEnd(40); });
    await act(async () => { await result.current.quantify(); });
    expect(regionQuantify.mock.calls[0]).toEqual(['', 40, 0, 0, 'counts']);
    expect(result.current.rowStart).toBe('');
  });

  it('a click handler passing its event is not mistaken for a rectangle', async () => {
    const { result } = renderHook(() => useRegionAverage('counts', t));
    act(() => { result.current.setRowStart(2); result.current.setRowEnd(5); });
    // onClick={quantify} hands over a DOM event, not a rect.
    await act(async () => { await result.current.quantify({ type: 'click', target: {} }); });
    expect(regionQuantify.mock.calls[0]).toEqual([2, 5, 0, 0, 'counts']);
  });

  it('reports a failure instead of leaving stale numbers on screen', async () => {
    regionQuantify.mockImplementationOnce(() =>
      Promise.reject({ response: { data: { detail: 'No file open' } } }));
    const { result } = renderHook(() => useDragIntoRegionAverage());
    dragRect(result, 10, 10, 40, 50);
    await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
    expect(result.current.region.error).toBe('No file open');
    expect(result.current.region.data).toBeNull();
    expect(result.current.region.loading).toBe(false);
  });
});
