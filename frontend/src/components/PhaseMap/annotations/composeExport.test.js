// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { drawAnnotationsOnto } from './composeExport';

// A canvas context that records the calls instead of painting. jsdom has no
// 2D context, and what matters here is what each drawer ASKS for.
function recordingCtx() {
  const calls = [];
  const rec = (name) => (...args) => calls.push([name, ...args]);
  const ctx = {
    calls,
    save: rec('save'), restore: rec('restore'),
    translate: rec('translate'), rotate: rec('rotate'), scale: rec('scale'),
    fillRect: rec('fillRect'), strokeRect: rec('strokeRect'),
    beginPath: rec('beginPath'), moveTo: rec('moveTo'), lineTo: rec('lineTo'),
    closePath: rec('closePath'), fill: rec('fill'), stroke: rec('stroke'),
    fillText: rec('fillText'), drawImage: rec('drawImage'),
    measureText: () => ({ width: 10 }),
    clip: rec('clip'), rect: rec('rect'), arc: rec('arc'),
    quadraticCurveTo: rec('quadraticCurveTo'), bezierCurveTo: rec('bezierCurveTo'),
    setLineDash: rec('setLineDash'), arcTo: rec('arcTo'), ellipse: rec('ellipse'),
    createLinearGradient: () => ({ addColorStop: () => {} }),
  };
  // `font` is a plain property; capture every assignment so we can check sizes.
  const fonts = [];
  Object.defineProperty(ctx, 'font', {
    get: () => fonts[fonts.length - 1] || '',
    set: (v) => { fonts.push(v); },
  });
  ctx.fonts = fonts;
  return ctx;
}

const ALL_TYPES = [
  { id: 'l', type: 'legend',   x: 0.02, y: 0.02, w: 0.3, h: 0.3, props: {} },
  { id: 's', type: 'scalebar', x: 0.6,  y: 0.85, w: 0.3, h: 0.08, props: { lengthUm: 5 } },
  { id: 't', type: 'title',    x: 0.02, y: 0.9,  w: 0.5, h: 0.08, props: { text: 'Phase Map' } },
  { id: 'a', type: 'arrow',    x: 0.85, y: 0.05, w: 0.1, h: 0.12, props: { label: 'ND' } },
];

const OPTS = {
  width: 800, height: 600,
  phaseStats: { phases: [{ phase_id: 0, name: 'Al', color_hex: '#ff0000', area_pct: 42 }] },
  scale: 2, stepX: 0.2, scanCols: 145,
};

const SCALE_LEGENDS = [
  { id: 'bc', label: 'Band Contrast', scale: { min: 12, max: 214, unit: 'a.u.', stops: ['#000', '#fff'] } },
];

describe('scale bodies', () => {
  const fakeImg = { width: 200, height: 260 };

  it('draws a value scale from the LIVE layer, not from the annotation', () => {
    const ctx = recordingCtx();
    const annot = { id: 'v', type: 'valuescale', x: 0.7, y: 0.1, w: 0.2, h: 0.4, props: { layerId: 'bc' } };
    drawAnnotationsOnto(ctx, [annot], { ...OPTS, scaleLegends: SCALE_LEGENDS });
    const texts = ctx.calls.filter((c) => c[0] === 'fillText').map((c) => c[1]);
    // Caption, the two ends, the middle and the unit — the numbers come from
    // the layer's own range.
    // '12.0' not '12': the same formatter the on-screen legend uses.
    expect(texts).toEqual(expect.arrayContaining(['Band Contrast', '214', '113', '12.0', 'a.u.']));
  });

  it('draws nothing when the layer it belongs to is gone', () => {
    // A stale bar would keep claiming a range nobody can check.
    const ctx = recordingCtx();
    const annot = { id: 'v', type: 'valuescale', x: 0.7, y: 0.1, w: 0.2, h: 0.4, props: { layerId: 'vanished' } };
    drawAnnotationsOnto(ctx, [annot], { ...OPTS, scaleLegends: SCALE_LEGENDS });
    expect(ctx.calls.some((c) => c[0] === 'fillText')).toBe(false);
  });

  it('contain-fits the colour key inside its box', () => {
    const ctx = recordingCtx();
    const annot = { id: 'k', type: 'colorkey', x: 0.1, y: 0.1, w: 0.25, h: 0.25, props: {} };
    drawAnnotationsOnto(ctx, [annot], { ...OPTS, ipfKeyImg: fakeImg });
    const draw = ctx.calls.find((c) => c[0] === 'drawImage');
    expect(draw).toBeTruthy();
    const [, , , , dw, dh] = draw;
    // 200x260 into 200x150 minus padding: height is the binding side, and the
    // aspect ratio survives.
    expect(dh).toBeLessThanOrEqual(0.25 * OPTS.height);
    expect(dw / dh).toBeCloseTo(200 / 260, 5);
  });

  it('draws no key at all when none was loaded', () => {
    const ctx = recordingCtx();
    const annot = { id: 'k', type: 'colorkey', x: 0.1, y: 0.1, w: 0.25, h: 0.25, props: {} };
    drawAnnotationsOnto(ctx, [annot], OPTS);
    expect(ctx.calls.some((c) => c[0] === 'drawImage')).toBe(false);
  });

  it('puts a body dragged off the map outside the map box', () => {
    // Negative coordinates are how a body sits in the dialog's border.
    const ctx = recordingCtx();
    const annot = { id: 'k', type: 'colorkey', x: -0.3, y: 0.1, w: 0.25, h: 0.25, props: {} };
    drawAnnotationsOnto(ctx, [annot], { ...OPTS, ipfKeyImg: fakeImg, offsetX: 500 });
    const translate = ctx.calls.find((c) => c[0] === 'translate');
    expect(translate[1]).toBe(500 + -0.3 * OPTS.width);
  });
});

describe('drawAnnotationsOnto', () => {
  it('draws every annotation type without blowing up', () => {
    // Each drawer takes its own parameter list. One of them once read an
    // options object it had never been handed, and the export died with
    // "opts is not defined" — only at save time, with the picture already on
    // screen. Exercising all four types here catches that at build time.
    const ctx = recordingCtx();
    expect(() => drawAnnotationsOnto(ctx, ALL_TYPES, OPTS)).not.toThrow();
    expect(ctx.calls.length).toBeGreaterThan(0);
  });

  it('scales the lettering with textScale', () => {
    const sizes = (scale) => {
      const ctx = recordingCtx();
      drawAnnotationsOnto(ctx, ALL_TYPES, { ...OPTS, textScale: scale });
      return ctx.fonts.map((f) => parseFloat(String(f).match(/(\d+(?:\.\d+)?)px/)?.[1] ?? '0'));
    };
    const one = sizes(1);
    const four = sizes(4);
    expect(one.length).toBeGreaterThan(0);
    expect(four).toEqual(one.map((s) => s * 4));
  });

  it('leaves the default untouched — 1x means the old rendering', () => {
    const ctxA = recordingCtx();
    const ctxB = recordingCtx();
    drawAnnotationsOnto(ctxA, ALL_TYPES, OPTS);
    drawAnnotationsOnto(ctxB, ALL_TYPES, { ...OPTS, textScale: 1 });
    expect(ctxB.fonts).toEqual(ctxA.fonts);
  });
});
