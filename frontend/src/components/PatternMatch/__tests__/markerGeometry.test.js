// @vitest-environment node
import { describe, it, expect } from 'vitest';
import { contentBox, pointerToNorm, normToBoxPct, hitTestMarker, renumber } from '../markerGeometry';

describe('contentBox (objectFit: contain)', () => {
  it('letterboxes a wide image into a square box (pillarbox top/bottom)', () => {
    // 200x100 image into 100x100 box -> fills width, 50 tall, centered vertically
    const b = contentBox({ naturalW: 200, naturalH: 100, boxW: 100, boxH: 100 });
    expect(b.w).toBeCloseTo(100); expect(b.h).toBeCloseTo(50);
    expect(b.x).toBeCloseTo(0);   expect(b.y).toBeCloseTo(25);
  });
  it('letterboxes a tall image (bars left/right)', () => {
    const b = contentBox({ naturalW: 100, naturalH: 200, boxW: 100, boxH: 100 });
    expect(b.w).toBeCloseTo(50); expect(b.h).toBeCloseTo(100);
    expect(b.x).toBeCloseTo(25); expect(b.y).toBeCloseTo(0);
  });
});

describe('pointerToNorm', () => {
  const rect = { left: 0, top: 0, width: 100, height: 100 };
  const natural = { w: 100, h: 100 };
  it('maps center to (0.5,0.5)', () => {
    expect(pointerToNorm({ clientX: 50, clientY: 50 }, rect, natural)).toEqual({ x: 0.5, y: 0.5 });
  });
  it('returns null outside the content box', () => {
    const tall = { w: 50, h: 100 }; // content is centered 25px..75px in x
    expect(pointerToNorm({ clientX: 5, clientY: 50 }, rect, tall)).toBeNull();
  });
});

describe('hitTestMarker', () => {
  it('returns the id of a marker within radius', () => {
    const ms = [{ id: 'a', x: 0.5, y: 0.5 }, { id: 'b', x: 0.1, y: 0.1 }];
    expect(hitTestMarker(ms, { x: 0.51, y: 0.5 }, 0.03)).toBe('a');
    expect(hitTestMarker(ms, { x: 0.8, y: 0.8 }, 0.03)).toBeNull();
  });
});

describe('renumber', () => {
  it('reassigns n contiguously', () => {
    const out = renumber([{ id: 'x', n: 5 }, { id: 'y', n: 9 }]);
    expect(out.map(m => m.n)).toEqual([1, 2]);
  });
});
