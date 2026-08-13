import { describe, it, expect } from 'vitest';
import {
  defaultBands, applyBandEdit, bandsArePartition, FLOOR_DEG,
} from './grainBoundaryBands';

const asRanges = (bands) => bands.map((b) => [b.min, b.max]);

describe('grain-boundary bands', () => {
  it('starts as sub < 5°, low-angle 5–15°, high-angle ≥ 15°', () => {
    expect(asRanges(defaultBands())).toEqual([[FLOOR_DEG, 5], [5, 15], [15, null]]);
    expect(bandsArePartition(defaultBands())).toBe(true);
  });

  it('moves the high-angle threshold when the low-angle class is widened', () => {
    // The reported wish: push the low-angle class past the high-angle one and
    // the other should follow rather than leaving a contradiction on screen.
    const out = applyBandEdit(defaultBands(), 1, 'max', 20);
    expect(asRanges(out)).toEqual([[FLOOR_DEG, 5], [5, 20], [20, null]]);
    expect(bandsArePartition(out)).toBe(true);
  });

  it('moves the low-angle end when the high-angle threshold is lowered', () => {
    const out = applyBandEdit(defaultBands(), 2, 'min', 10);
    expect(asRanges(out)).toEqual([[FLOOR_DEG, 5], [5, 10], [10, null]]);
    expect(bandsArePartition(out)).toBe(true);
  });

  it('pushes the class above out of the way instead of refusing the edit', () => {
    // Sub-boundaries up to 18° leaves no room for a 5–15° class; the low-angle
    // class is carried along rather than the edit being dropped.
    const out = applyBandEdit(defaultBands(), 0, 'max', 18);
    expect(out[0].max).toBe(18);
    expect(bandsArePartition(out)).toBe(true);
    expect(out[1].min).toBe(18);
    expect(out[2].min).toBeGreaterThanOrEqual(18);
  });

  it('drags the classes below down when a cut point is pulled under them', () => {
    const out = applyBandEdit(defaultBands(), 2, 'min', 2);
    expect(bandsArePartition(out)).toBe(true);
    expect(out[2].min).toBe(2);
    expect(out[1].max).toBe(2);
    expect(out[0].max).toBeLessThan(2);
  });

  it('keeps the open-ended class open', () => {
    const out = applyBandEdit(defaultBands(), 2, 'max', 40);
    expect(out).toEqual(defaultBands());     // refused, nothing moved
  });

  it('holds angles inside the physical range', () => {
    const low = applyBandEdit(defaultBands(), 1, 'max', -7);
    expect(low[1].max).toBeGreaterThanOrEqual(FLOOR_DEG);
    expect(bandsArePartition(low)).toBe(true);
    expect(applyBandEdit(defaultBands(), 1, 'max', 999)[1].max).toBeLessThanOrEqual(62.8);
  });

  it('passes colour and width through, and bounds the width', () => {
    expect(applyBandEdit(defaultBands(), 0, 'color', '#ff0000')[0].color).toBe('#ff0000');
    expect(applyBandEdit(defaultBands(), 0, 'width', 99)[0].width).toBe(8);
    expect(applyBandEdit(defaultBands(), 0, 'width', 0)[0].width).toBe(1);
    expect(applyBandEdit(defaultBands(), 0, 'on', false)[0].on).toBe(false);
  });

  it('ignores nonsense instead of corrupting the list', () => {
    expect(applyBandEdit(defaultBands(), 1, 'max', 'abc')).toEqual(defaultBands());
    expect(applyBandEdit(defaultBands(), 9, 'max', 10)).toEqual(defaultBands());
  });
});
