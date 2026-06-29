// @vitest-environment node
import { describe, it, expect } from 'vitest';
import { paintFigure } from '../paintFigure';
import { defaultModel } from '../figureModel';

// Recording fake 2D context — captures every draw op so we can assert that the
// painter emits ONLY model-derived ops (proof: no handles/crosshair can leak).
function fakeCtx() {
  const ops = [];
  const rec = (name) => (...args) => ops.push({ name, args });
  return {
    ops,
    set fillStyle(v) { ops.push({ name: 'set:fillStyle', args: [v] }); },
    set strokeStyle(v) { ops.push({ name: 'set:strokeStyle', args: [v] }); },
    set lineWidth(v) {}, set font(v) {}, set textAlign(v) {}, set textBaseline(v) {},
    set imageSmoothingEnabled(v) { ops.push({ name: 'set:smoothing', args: [v] }); },
    set globalAlpha(v) {}, set shadowColor(v) {}, set shadowBlur(v) {},
    fillRect: rec('fillRect'), strokeRect: rec('strokeRect'), drawImage: rec('drawImage'),
    fillText: rec('fillText'), beginPath: rec('beginPath'), arc: rec('arc'),
    fill: rec('fill'), stroke: rec('stroke'), moveTo: rec('moveTo'), lineTo: rec('lineTo'),
    save: rec('save'), restore: rec('restore'), translate: rec('translate'),
    createLinearGradient: () => ({ addColorStop() {} }),
    measureText: () => ({ width: 10 }),
  };
}

describe('paintFigure', () => {
  const sources = { experimental: { _img: 'e' }, simulated: { _img: 's' }, ncc: { _img: 'n' }, heatmap: { _img: 'h' }, markers: [], stepUm: 0.1 };

  it('draws exactly one image per panel and a background', () => {
    const model = defaultModel(['experimental', 'simulated']);
    const ctx = fakeCtx();
    paintFigure(ctx, model, sources, { W: 1200, H: 600 });
    const draws = ctx.ops.filter(o => o.name === 'drawImage');
    expect(draws).toHaveLength(2);
    expect(ctx.ops.some(o => o.name === 'fillRect')).toBe(true); // bg
  });

  it('honors target resolution (panel rects within W×H)', () => {
    const model = defaultModel(['experimental']);
    const ctx = fakeCtx();
    paintFigure(ctx, model, sources, { W: 800, H: 400 });
    const d = ctx.ops.find(o => o.name === 'drawImage');
    const [, x, y, w, h] = d.args;
    expect(x).toBeGreaterThanOrEqual(0); expect(x + w).toBeLessThanOrEqual(800.5);
    expect(y).toBeGreaterThanOrEqual(0); expect(y + h).toBeLessThanOrEqual(400.5);
  });

  it('transparent background skips the full-canvas bg fillRect', () => {
    const model = defaultModel(['experimental']);
    model.canvas.bg = 'transparent';
    const ctx = fakeCtx();
    paintFigure(ctx, model, sources, { W: 800, H: 400 });
    const bg = ctx.ops.find(o => o.name === 'fillRect' && o.args[0] === 0 && o.args[1] === 0 && o.args[2] === 800 && o.args[3] === 400);
    expect(bg).toBeUndefined();
  });

  it('draws a border strokeRect only when border.on', () => {
    let model = defaultModel(['experimental']);
    const ctx0 = fakeCtx();
    paintFigure(ctx0, model, sources, { W: 800, H: 400 });
    expect(ctx0.ops.some(o => o.name === 'strokeRect')).toBe(false);
    model.elements[0].border.on = true;
    const ctx1 = fakeCtx();
    paintFigure(ctx1, model, sources, { W: 800, H: 400 });
    expect(ctx1.ops.some(o => o.name === 'strokeRect')).toBe(true);
  });

  it('draws a numbered marker (arc + fillText) on a panel when includeMarkers', () => {
    const model = defaultModel(['experimental']);
    const withMarker = { ...sources, markers: [{ id: 'm1', n: 1, x: 0.5, y: 0.5 }] };
    const ctx = fakeCtx();
    paintFigure(ctx, model, withMarker, { W: 800, H: 400 });
    expect(ctx.ops.some(o => o.name === 'arc')).toBe(true);
    expect(ctx.ops.some(o => o.name === 'fillText' && o.args[0] === '1')).toBe(true);
  });

  it('contains a square image inside a non-square panel box (aspect preserved, centered)', () => {
    const model = defaultModel(['experimental']);
    // Force a deliberately 2:1 panel box covering the whole canvas to prove the
    // contain-fit (a square source must NOT be stretched into the box).
    model.elements[0].x = 0; model.elements[0].y = 0; model.elements[0].w = 1; model.elements[0].h = 1;
    const sq = { naturalWidth: 100, naturalHeight: 100 }; // square (circular pattern)
    const ctx = fakeCtx();
    paintFigure(ctx, model, { experimental: sq, markers: [] }, { W: 800, H: 400 });
    const [, x, y, w, h] = ctx.ops.find(o => o.name === 'drawImage').args;
    expect(w).toBeCloseTo(h, 5);          // drawn square — aspect preserved, no ellipse
    expect(h).toBeCloseTo(400, 5);        // fit by the limiting (height) axis
    expect(x).toBeCloseTo(200, 5);        // centered horizontally in the 800-wide box
    expect(y).toBeCloseTo(0, 5);
  });
});
