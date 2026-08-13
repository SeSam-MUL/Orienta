import { describe, it, expect } from 'vitest';
import {
  niceLength, niceLengthCapped, formatMicrons, scalebarGeometry,
  umToScreenPx, visibleWidthUm,
} from './scalebarGeometry';

describe('niceLength', () => {
  it('snaps to the 1/2/5 ladder', () => {
    expect(niceLength(1.2)).toBe(1);
    expect(niceLength(2.9)).toBe(2);
    expect(niceLength(5.8)).toBe(5);
    expect(niceLength(9)).toBe(10);
    expect(niceLength(0.23)).toBeCloseTo(0.2, 10);
    expect(niceLength(340)).toBe(200);
  });

  it('rejects nonsense', () => {
    expect(niceLength(0)).toBeNull();
    expect(niceLength(-3)).toBeNull();
    expect(niceLength(NaN)).toBeNull();
  });
});

describe('niceLengthCapped', () => {
  it('steps down the ladder until the bar fits what is on screen', () => {
    // 9 rounds up to 10, which would overflow a 9.5 µm wide view.
    expect(niceLengthCapped(9, 9.5)).toBe(5);
    expect(niceLengthCapped(9, 20)).toBe(10);
  });

  it('crosses a decade boundary when it has to', () => {
    expect(niceLengthCapped(9, 0.9)).toBeCloseTo(0.5, 10);
  });

  it('is a no-op without a usable cap', () => {
    expect(niceLengthCapped(9, 0)).toBe(10);
    expect(niceLengthCapped(9, NaN)).toBe(10);
  });
});

describe('formatMicrons', () => {
  it('picks the readable unit', () => {
    expect(formatMicrons(5)).toBe('5 µm');
    expect(formatMicrons(2.5)).toBe('2.5 µm');
    expect(formatMicrons(0.2)).toBe('200 nm');
    expect(formatMicrons(2000)).toBe('2.0 mm');
  });
});

describe('umToScreenPx', () => {
  // Same real case: 145x174 map at 0.2 µm/px, drawn 1019.2 CSS px wide,
  // auto-zoomed to a 96x128 indexed box.
  const REAL = {
    nativeSize: { w: 145, h: 174 },
    contentBbox: { x: 25, y: 23, w: 96, h: 128 },
    stepX: 0.2,
    fitWidth: 1019.2,
  };

  it('converts a physical length to what it measures on screen', () => {
    // 5 µm = 25 map px; the 96-wide box is drawn across 131 buffer px, and one
    // buffer px is 1019.2/145 CSS px.
    expect(umToScreenPx({ um: 5, ...REAL }))
      .toBeCloseTo(25 * (131 / 96) * (1019.2 / 145), 3);
  });

  it('is linear in the length — the property the annotation bar relies on', () => {
    const one = umToScreenPx({ um: 1, ...REAL });
    expect(umToScreenPx({ um: 7, ...REAL })).toBeCloseTo(7 * one, 6);
  });

  it('agrees with the automatic scalebar for the length that one picked', () => {
    // Two implementations, one answer: the annotation bar and the layer-stack
    // bar must not disagree about how long a µm is.
    const auto = scalebarGeometry({ ...REAL, lengthFrac: 0.2 });
    expect(umToScreenPx({ um: auto.lengthUm, ...REAL })).toBeCloseTo(auto.barPx, 6);
  });

  it('follows the user zoom', () => {
    const plain = umToScreenPx({ um: 5, ...REAL });
    expect(umToScreenPx({ um: 5, ...REAL, zoomScale: 3 })).toBeCloseTo(3 * plain, 6);
  });

  it('without an auto-zoom box it is the plain share of the map width', () => {
    // 145 px * 0.2 µm = 29 µm across.
    expect(umToScreenPx({ um: 5, ...REAL, contentBbox: null }))
      .toBeCloseTo((5 / 29) * 1019.2, 3);
  });

  it('returns null rather than a guess when the geometry is unknown', () => {
    expect(umToScreenPx({ um: 5, ...REAL, stepX: 0 })).toBeNull();
    expect(umToScreenPx({ um: 5, ...REAL, fitWidth: 0 })).toBeNull();
    expect(umToScreenPx({ um: 5, ...REAL, nativeSize: null })).toBeNull();
    expect(umToScreenPx({ um: 0, ...REAL })).toBeNull();
  });
});

describe('visibleWidthUm', () => {
  it('counts only what is on screen', () => {
    const base = { nativeSize: { w: 145, h: 174 }, stepX: 0.2 };
    expect(visibleWidthUm({ ...base, contentBbox: null })).toBeCloseTo(29, 6);
    expect(visibleWidthUm({ ...base, contentBbox: { w: 96, h: 128 } })).toBeCloseTo(19.2, 6);
    expect(visibleWidthUm({ ...base, contentBbox: { w: 96, h: 128 }, zoomScale: 2 })).toBeCloseTo(9.6, 6);
  });
});

describe('scalebarGeometry', () => {
  // The real case that exposed the bug: a 145x174 map at 0.2 µm/px, drawn
  // 1019.2 CSS px wide, auto-zoomed to a 96x128 indexed box.
  const REAL = {
    nativeSize: { w: 145, h: 174 },
    contentBbox: { x: 25, y: 23, w: 96, h: 128 },
    stepX: 0.2,
    lengthFrac: 0.2,
    fitWidth: 1019.2,
  };

  it('measures the bar in pixels of the map, not of the label', () => {
    const g = scalebarGeometry(REAL);
    expect(g.label).toBe('5 µm');
    // 5 µm = 25 map px; the 96-wide box is drawn across 131 buffer px
    // (contain-fit of 0.75 aspect into 145x174), and one buffer px is
    // 1019.2/145 = 7.029 CSS px.
    expect(g.barPx).toBeCloseTo(25 * (131 / 96) * (1019.2 / 145), 1);
    // Regression guard: the old code produced 5.1 px here.
    expect(g.barPx).toBeGreaterThan(200);
  });

  it('accounts for the auto-zoom magnification', () => {
    const zoomed = scalebarGeometry(REAL);
    const whole = scalebarGeometry({ ...REAL, contentBbox: null });
    // Same physical length, but the zoomed view shows it larger.
    expect(zoomed.lengthUm).toBe(whole.lengthUm);
    expect(zoomed.barPx).toBeGreaterThan(whole.barPx);
    expect(zoomed.barPx / whole.barPx).toBeCloseTo(131 / 96, 2);
  });

  it('without a bbox, the bar is its physical share of the map width', () => {
    const g = scalebarGeometry({ ...REAL, contentBbox: null });
    // 145 px * 0.2 µm = 29 µm across; a 5 µm bar is 5/29 of the width.
    expect(g.barPx).toBeCloseTo((5 / 29) * 1019.2, 1);
  });

  it('never draws a bar wider than the visible map', () => {
    const g = scalebarGeometry({ ...REAL, lengthFrac: 1 });
    const visibleUm = 96 * 0.2;
    expect(g.lengthUm).toBeLessThanOrEqual(visibleUm);
    const dstW = 131;
    const visiblePx = dstW * (1019.2 / 145);
    expect(g.barPx).toBeLessThanOrEqual(visiblePx + 0.5);
  });

  it('scales with the on-screen size, so a resize keeps it honest', () => {
    const small = scalebarGeometry({ ...REAL, fitWidth: 500 });
    const big = scalebarGeometry({ ...REAL, fitWidth: 1000 });
    expect(big.lengthUm).toBe(small.lengthUm);
    expect(big.barPx / small.barPx).toBeCloseTo(2, 6);
  });

  it('stays physically honest when the user zooms in', () => {
    const at1 = scalebarGeometry(REAL);
    const at4 = scalebarGeometry({ ...REAL, zoomScale: 4 });
    // A µm covers 4x more screen pixels, so the same physical length would be
    // drawn 4x longer. The bar is allowed to pick a shorter round number
    // instead — what must hold is that pixels-per-µm quadrupled.
    const perUm1 = at1.barPx / at1.lengthUm;
    const perUm4 = at4.barPx / at4.lengthUm;
    expect(perUm4 / perUm1).toBeCloseTo(4, 6);
  });

  it('shortens the label rather than running off the zoomed view', () => {
    const at8 = scalebarGeometry({ ...REAL, zoomScale: 8 });
    // Only 96/8 = 12 map px = 2.4 µm are visible.
    expect(at8.lengthUm).toBeLessThanOrEqual(2.4);
    const visiblePx = 131 * (1019.2 / 145);   // the map box on screen
    expect(at8.barPx).toBeLessThanOrEqual(visiblePx + 0.5);
  });

  it('treats a missing or nonsense zoom as no zoom', () => {
    const base = scalebarGeometry(REAL);
    for (const z of [undefined, 0, -2, NaN]) {
      expect(scalebarGeometry({ ...REAL, zoomScale: z }).barPx).toBeCloseTo(base.barPx, 10);
    }
  });

  it('returns null rather than guessing when inputs are missing', () => {
    expect(scalebarGeometry({ ...REAL, stepX: 0 })).toBeNull();
    expect(scalebarGeometry({ ...REAL, fitWidth: 0 })).toBeNull();
    expect(scalebarGeometry({ ...REAL, nativeSize: null })).toBeNull();
  });

  it('falls back to a sane fraction when none is given', () => {
    const g = scalebarGeometry({ ...REAL, lengthFrac: undefined });
    expect(g).not.toBeNull();
    expect(g.lengthUm).toBeGreaterThan(0);
  });
});
