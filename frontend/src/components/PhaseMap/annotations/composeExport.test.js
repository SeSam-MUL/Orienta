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
