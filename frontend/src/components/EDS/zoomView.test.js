import { describe, it, expect } from 'vitest';
import {
  IDENTITY_VIEW, MIN_SCALE, MAX_SCALE, WHEEL_STEP,
  clampView, zoomedRect, viewToTransform, zoomAt, panBy, wheelFactor, isZoomed,
} from './zoomView';
import { pointerToRowCol, rowColToContainerPx } from './mapCoords';

const rect = (w, h, left = 0, top = 0) => ({ left, top, width: w, height: h });

describe('zoomView — clamping', () => {
  it('pins the centre at 1× and keeps scale inside [1, 16]', () => {
    expect(clampView({ scale: 1, cx: 0.9, cy: 0.1 })).toEqual({ scale: 1, cx: 0.5, cy: 0.5 });
    expect(clampView({ scale: 0.2, cx: 0.5, cy: 0.5 }).scale).toBe(MIN_SCALE);
    expect(clampView({ scale: 999, cx: 0.5, cy: 0.5 }).scale).toBe(MAX_SCALE);
  });

  it('keeps the content covering the box (no empty margin) when zoomed', () => {
    const v = clampView({ scale: 2, cx: 0, cy: 1 });
    expect(v.cx).toBeCloseTo(0.25, 6);   // 0.5 / 2
    expect(v.cy).toBeCloseTo(0.75, 6);   // 1 - 0.5 / 2
  });

  it('survives garbage input instead of producing NaN', () => {
    expect(clampView({ scale: NaN, cx: undefined, cy: null })).toEqual(IDENTITY_VIEW);
    expect(clampView(null)).toEqual(IDENTITY_VIEW);
  });

  it('isZoomed ignores float noise around 1×', () => {
    expect(isZoomed({ scale: 1.0005, cx: 0.5, cy: 0.5 })).toBe(false);
    expect(isZoomed({ scale: 1.25, cx: 0.5, cy: 0.5 })).toBe(true);
  });
});

describe('zoomView — rect and CSS transform agree', () => {
  it('is the identity rect at 1×', () => {
    const r = zoomedRect(rect(200, 100, 10, 20), IDENTITY_VIEW);
    expect(r).toMatchObject({ left: 10, top: 20, width: 200, height: 100 });
    expect(viewToTransform(IDENTITY_VIEW)).toBe('none');
  });

  it('places the view centre on the box centre', () => {
    const host = rect(200, 200);
    const view = { scale: 2, cx: 0.25, cy: 0.75 };
    const zr = zoomedRect(host, view);
    // The content fraction cx must land exactly on the box centre.
    expect(zr.left + view.cx * zr.width).toBeCloseTo(host.width / 2, 6);
    expect(zr.top + view.cy * zr.height).toBeCloseTo(host.height / 2, 6);
  });

  it('emits a CSS transform matching the same geometry', () => {
    // scale(s) translate(t%) with t = (0.5 - c) * 100 — the derivation
    // zoomedRect assumes. If these two ever drift, the picture and the pointer
    // mapping disagree, which is the bug class this test exists for.
    const view = { scale: 2, cx: 0.25, cy: 0.75 };
    expect(viewToTransform(view)).toBe('scale(2) translate(25%, -25%)');
  });
});

describe('zoomView — zoom keeps the pointer anchored', () => {
  it('holds the content under the cursor in place', () => {
    const host = rect(200, 200);
    const shape = [100, 100];
    const start = IDENTITY_VIEW;
    // Pointer at 25% / 25% of the box → pixel (25, 25) before zooming.
    const ev = { clientX: 50, clientY: 50 };
    const before = pointerToRowCol(ev, zoomedRect(host, start), shape);
    const zoomed = zoomAt(start, 2, 0.25, 0.25);
    const after = pointerToRowCol(ev, zoomedRect(host, zoomed), shape);
    expect(zoomed.scale).toBe(2);
    expect(after).toEqual(before);
  });

  it('never zooms past the limits', () => {
    let v = IDENTITY_VIEW;
    for (let i = 0; i < 40; i += 1) v = zoomAt(v, WHEEL_STEP, 0.5, 0.5);
    expect(v.scale).toBe(MAX_SCALE);
    for (let i = 0; i < 80; i += 1) v = zoomAt(v, 1 / WHEEL_STEP, 0.5, 0.5);
    expect(v).toEqual(IDENTITY_VIEW);
  });

  it('maps wheel direction to zoom direction', () => {
    expect(wheelFactor(-100)).toBeGreaterThan(1);
    expect(wheelFactor(100)).toBeLessThan(1);
  });
});

describe('zoomView — panning', () => {
  it('moves the content with the drag and clamps at the edge', () => {
    const v = panBy({ scale: 2, cx: 0.5, cy: 0.5 }, 0.1, 0);   // drag right
    expect(v.cx).toBeCloseTo(0.45, 6);
    const edge = panBy({ scale: 2, cx: 0.5, cy: 0.5 }, 10, 10);
    expect(edge.cx).toBeCloseTo(0.25, 6);                      // clamped, not lost
    expect(edge.cy).toBeCloseTo(0.25, 6);
  });

  it('is a no-op at 1× (nothing to pan)', () => {
    expect(panBy(IDENTITY_VIEW, 0.3, -0.2)).toEqual(IDENTITY_VIEW);
  });
});

describe('zoomView — syncing across differently sized boxes', () => {
  it('shows the same map pixel at the box centre for a small and a large box', () => {
    // The tiles and the composite overlay have different CSS sizes; a shared
    // view must still point at the same place on the sample.
    const view = { scale: 4, cx: 0.3, cy: 0.7 };
    const shape = [128, 156];
    const small = rect(240, 197);
    const large = rect(600, 492, 33, 77);
    const centreOf = (r) => pointerToRowCol(
      { clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 },
      zoomedRect(r, view),
      shape,
    );
    expect(centreOf(small)).toEqual(centreOf(large));
  });

  it('round-trips a pixel through pointer mapping and back to CSS px', () => {
    const host = rect(300, 300);
    const view = { scale: 3, cx: 0.4, cy: 0.6 };
    const shape = [100, 100];
    const zr = zoomedRect(host, view);
    const p = rowColToContainerPx(40, 60, zr, shape);
    const back = pointerToRowCol(
      { clientX: p.x + (zr.left - host.left), clientY: p.y + (zr.top - host.top) },
      zr, shape,
    );
    expect(back).toEqual({ row: 40, col: 60 });
  });
});
