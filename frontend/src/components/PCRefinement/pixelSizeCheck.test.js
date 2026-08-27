import { describe, it, expect } from 'vitest';
import {
  checkPixelSize, plausibleWidthMm, TYPICAL_MIN_UM, TYPICAL_MAX_UM,
} from './pixelSizeCheck';

describe('checkPixelSize', () => {
  it('flags the reported case: 0.1 mm over 78 px', () => {
    const umPerPx = (0.1 * 1000) / 78;      // 1.28
    const r = checkPixelSize(umPerPx);
    expect(r.level).toBe('warn');
    expect(r.umPerPx).toBeCloseTo(1.28, 2);
  });

  it('accepts the range real detectors live in', () => {
    for (const um of [20, 35, 50, 59.2, 70, 100, 150]) {
      expect(checkPixelSize(um).level).toBe('ok');
    }
  });

  it('flags both directions of implausible', () => {
    expect(checkPixelSize(TYPICAL_MIN_UM - 0.1).level).toBe('warn');
    expect(checkPixelSize(TYPICAL_MAX_UM + 0.1).level).toBe('warn');
  });

  it('says nothing when the field is empty rather than crying wolf', () => {
    for (const v of [null, undefined, 0, NaN, '', -5]) {
      expect(checkPixelSize(v).level).toBe('unset');
    }
  });

  it('accepts a numeric string', () => {
    expect(checkPixelSize('59.2').level).toBe('ok');
  });
});

describe('plausibleWidthMm', () => {
  it('suggests a width that lands in the typical range', () => {
    const w = plausibleWidthMm(78);
    expect(w).toBeCloseTo(4.68, 2);
    expect(checkPixelSize((w * 1000) / 78).level).toBe('ok');
  });

  it('works for a full-size detector', () => {
    const w = plausibleWidthMm(480);
    expect(checkPixelSize((w * 1000) / 480).level).toBe('ok');
  });

  it('returns null without a pattern width', () => {
    expect(plausibleWidthMm(0)).toBeNull();
    expect(plausibleWidthMm(null)).toBeNull();
  });
});
