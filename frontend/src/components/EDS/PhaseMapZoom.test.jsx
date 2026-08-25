// @vitest-environment jsdom
/**
 * The pointer maths under zoom.
 *
 * The failure this guards against is subtle and would look almost right:
 * `getBoundingClientRect()` already includes a CSS transform, so measuring the
 * ZOOMED element counts the zoom twice. The centre of the map stays correct
 * either way — which is exactly why a "click the centre, it still works" check
 * proves nothing. Every assertion here is therefore off-centre.
 */
import { describe, it, expect } from 'vitest';

import { pixelFromClickAt } from './PhaseMapPanel';
import { IDENTITY_VIEW } from './zoomView';

/** A square host, so the map fills it exactly and there is no letterbox. */
const HOST = {
  left: 0, top: 0, width: 120, height: 90,
  right: 120, bottom: 90, x: 0, y: 0,
};
// Half-pixel offsets on purpose: a click exactly on a pixel BOUNDARY is a
// knife edge. With a host whose aspect ratio matches the grid's, `120/(120/90)`
// evaluates to 90.00000000000001 in floating point, so a boundary click floors
// one row either way depending on the last bit. Real clicks land inside a
// pixel; the tests should too.
const at = (x, y) => ({ clientX: x, clientY: y });

describe('pixelFromClickAt', () => {
  it('maps a click to the pixel under it at 1x', () => {
    // 120x90 host, 120x90 grid -> one screen px per scan px.
    expect(pixelFromClickAt(at(30.5, 63.5), HOST, IDENTITY_VIEW, 90, 120))
      .toEqual({ row: 63, col: 30 });
  });

  it('keeps the corners addressable at 1x', () => {
    expect(pixelFromClickAt(at(0.5, 0.5), HOST, IDENTITY_VIEW, 90, 120))
      .toEqual({ row: 0, col: 0 });
    expect(pixelFromClickAt(at(119.5, 89.5), HOST, IDENTITY_VIEW, 90, 120))
      .toEqual({ row: 89, col: 119 });
  });

  it('returns null outside the map', () => {
    expect(pixelFromClickAt(at(-5, 10), HOST, IDENTITY_VIEW, 90, 120)).toBeNull();
    expect(pixelFromClickAt(at(10, 200), HOST, IDENTITY_VIEW, 90, 120)).toBeNull();
  });

  it('halves the addressed span when zoomed 2x — the off-centre test', () => {
    // Centred 2x: the visible window is the middle half of the map, columns
    // 30..90 and rows 22.5..67.5. The LEFT EDGE of the screen therefore shows
    // column 30, not column 0. A double-counted transform would report
    // something near 45 here (half again), so this pins the bug directly.
    const view = { scale: 2, cx: 0.5, cy: 0.5 };
    expect(pixelFromClickAt(at(0.5, 45), HOST, view, 90, 120).col).toBe(30);
    expect(pixelFromClickAt(at(119.5, 45), HOST, view, 90, 120).col).toBe(89);
  });

  it('leaves the centre invariant under a centred zoom', () => {
    // True, but weak on its own — kept as documentation of why the test above
    // exists.
    const one = pixelFromClickAt(at(60, 45), HOST, IDENTITY_VIEW, 90, 120);
    const two = pixelFromClickAt(at(60, 45), HOST, { scale: 2, cx: 0.5, cy: 0.5 }, 90, 120);
    expect(two).toEqual(one);
  });

  it('follows a panned view', () => {
    // Zoomed 2x and panned to the top-left quadrant: the screen's left edge
    // now shows column 0 again, and its right edge column 60.
    const view = { scale: 2, cx: 0.25, cy: 0.25 };
    expect(pixelFromClickAt(at(0.5, 45), HOST, view, 90, 120).col).toBe(0);
    expect(pixelFromClickAt(at(119.5, 45), HOST, view, 90, 120).col).toBe(59);
  });

  it('survives a missing rect or shape instead of throwing', () => {
    expect(pixelFromClickAt(at(1, 1), null, IDENTITY_VIEW, 90, 120)).toBeNull();
    expect(pixelFromClickAt(at(1, 1), { ...HOST, width: 0 }, IDENTITY_VIEW, 90, 120))
      .toBeNull();
    expect(pixelFromClickAt(at(1, 1), HOST, IDENTITY_VIEW, 0, 0)).toBeNull();
  });

  it('treats a missing view as no zoom', () => {
    expect(pixelFromClickAt(at(30.5, 63.5), HOST, undefined, 90, 120))
      .toEqual({ row: 63, col: 30 });
  });
});
