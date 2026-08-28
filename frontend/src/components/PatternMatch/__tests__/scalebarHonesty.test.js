// @vitest-environment node
/**
 * The scale bar of the Pattern-Match figure has to measure the heatmap.
 *
 * What it did instead, until 2026-08-27:
 *
 *   * `stepUm` was `null` at all three call sites (IndexingPage,
 *     SinglePixelPhaseTestDialog, PhaseMapPage), so the micrometre branch could
 *     never fire and the button always produced the fallback;
 *   * that fallback wrote `labelText: '100 px'` beside `fracOfPanel: 0.3`, two
 *     numbers with no relation to each other, and the painter drew
 *     `fracOfPanel x <the scale bar element's OWN box>`;
 *   * so dragging the bar's resize handle changed its length and left the label
 *     alone — in micrometre mode too.
 *
 * The rule now: the bar's length is physics, read off the heatmap panel. Its
 * box is a frame the user drags for position. These tests measure the drawn
 * rectangle and check it against the label.
 */
import { describe, it, expect } from 'vitest';
import { paintFigure } from '../paintFigure';
import {
  addElement, defaultModel, heatmapScale, heatmapUmPerOutputPx, niceCount, niceScaleLength,
} from '../figureModel';

function fakeCtx() {
  const ops = [];
  const rec = (name) => (...args) => ops.push({ name, args });
  return {
    ops,
    set fillStyle(v) { ops.push({ name: 'set:fillStyle', args: [v] }); },
    set strokeStyle(v) {}, set lineWidth(v) {}, set font(v) {},
    set textAlign(v) {}, set textBaseline(v) {}, set imageSmoothingEnabled(v) {},
    set globalAlpha(v) {}, set shadowColor(v) {}, set shadowBlur(v) {},
    fillRect: rec('fillRect'), strokeRect: rec('strokeRect'), drawImage: rec('drawImage'),
    fillText: rec('fillText'), beginPath: rec('beginPath'), arc: rec('arc'),
    fill: rec('fill'), stroke: rec('stroke'), moveTo: rec('moveTo'), lineTo: rec('lineTo'),
    save: rec('save'), restore: rec('restore'), translate: rec('translate'),
    createLinearGradient: () => ({ addColorStop() {} }),
    measureText: () => ({ width: 10 }),
  };
}

// A real NCC heatmap: 120 scan columns x 90 rows, upscaled 3x by the backend
// for click precision, at a 0.5 um scan step.
const MAP_COLS = 120;
const STEP_UM = 0.5;
const heatmapImg = { naturalWidth: 360, naturalHeight: 270 };
const SOURCES = {
  experimental: { naturalWidth: 156, naturalHeight: 128 },
  heatmap: heatmapImg,
  markers: [],
  stepUm: STEP_UM,
  mapCols: MAP_COLS,
};
const SIZE = { W: 1200, H: 600 };

/** The bar rectangle the painter drew: the first fillRect after the panels. */
function barRect(ctx) {
  const rects = ctx.ops.filter((o) => o.name === 'fillRect');
  // [0] is the canvas background; the bar is the last thin one.
  return rects.slice(1).map((o) => o.args).find((a) => a[3] > 0 && a[3] < a[2]);
}
function labels(ctx) {
  return ctx.ops.filter((o) => o.name === 'fillText').map((o) => String(o.args[0]));
}

/** A figure with the heatmap panel plus a scale bar of `lengthUm`. */
function figureWithBar(lengthUm, boxW = 0.3) {
  let m = defaultModel(['heatmap']);
  const panel = m.elements.find((e) => e.type === 'panel' && e.source === 'heatmap');
  expect(panel).toBeTruthy();
  m = addElement(m, {
    type: 'scalebar', mode: 'um', unit: 'µm', lengthValue: lengthUm,
    x: panel.x, y: panel.y + panel.h + 0.02, w: boxW, h: 0.06,
    color: '#000000', fontSize: 12,
  });
  return m;
}

describe('heatmapUmPerOutputPx', () => {
  it('measures the panel as DRAWN, not as boxed', () => {
    // The painter fits the image inside the panel box (object-fit: contain), so
    // a wide box holding a 4:3 heatmap draws it narrower than the box. Sizing
    // the bar off the box width would stretch it by exactly that mismatch.
    const m = defaultModel(['heatmap']);
    const panel = m.elements.find((e) => e.source === 'heatmap');
    const upp = heatmapUmPerOutputPx(m, SOURCES, SIZE);
    const boxW = panel.w * SIZE.W;
    const ar = heatmapImg.naturalWidth / heatmapImg.naturalHeight;
    const drawnW = (ar > boxW / (panel.h * SIZE.H)) ? boxW : (panel.h * SIZE.H) * ar;
    expect(upp).toBeCloseTo((STEP_UM * MAP_COLS) / drawnW, 9);
  });

  it('shrinks with the image when the panel box is TALLER than the heatmap', () => {
    // The discriminating case: a 4:3 heatmap in a 1:2 box is fitted by HEIGHT,
    // so it is drawn far narrower than the box. Sizing a bar off the box width
    // would stretch it by exactly that mismatch — here 2.67x.
    let m = defaultModel(['heatmap']);
    const panel = m.elements.find((e) => e.source === 'heatmap');
    m = { ...m, elements: m.elements.map((e) => (
      e.id === panel.id ? { ...e, x: 0.05, y: 0.05, w: 0.8, h: 0.4 } : e)) };
    const boxW = 0.8 * SIZE.W;                       // 960
    const boxH = 0.4 * SIZE.H;                       // 240
    const drawnW = boxH * (heatmapImg.naturalWidth / heatmapImg.naturalHeight); // 320
    expect(drawnW).toBeLessThan(boxW);
    expect(heatmapUmPerOutputPx(m, SOURCES, SIZE))
      .toBeCloseTo((STEP_UM * MAP_COLS) / drawnW, 9);
  });

  it('is null when there is nothing to measure with', () => {
    const m = defaultModel(['heatmap']);
    expect(heatmapUmPerOutputPx(m, { ...SOURCES, stepUm: null }, SIZE)).toBeNull();
    expect(heatmapUmPerOutputPx(m, { ...SOURCES, mapCols: null }, SIZE)).toBeNull();
    expect(heatmapUmPerOutputPx(defaultModel(['experimental']), SOURCES, SIZE)).toBeNull();
  });
});

describe('the drawn bar matches its label', () => {
  it('a 10 µm bar is 10 µm long on the figure', () => {
    const ctx = fakeCtx();
    paintFigure(ctx, figureWithBar(10), SOURCES, SIZE);
    const rect = barRect(ctx);
    expect(rect).toBeTruthy();
    const upp = heatmapUmPerOutputPx(figureWithBar(10), SOURCES, SIZE);
    expect(rect[2] * upp).toBeCloseTo(10, 6);
    expect(labels(ctx).some((s) => s.includes('10') && s.includes('µm'))).toBe(true);
  });

  it('resizing the bar\'s frame does not change how long it is', () => {
    // The old painter multiplied `fracOfPanel` by the element's own width, so
    // this doubled the bar while the label stayed at 10 µm.
    const a = fakeCtx(); paintFigure(a, figureWithBar(10, 0.2), SOURCES, SIZE);
    const b = fakeCtx(); paintFigure(b, figureWithBar(10, 0.6), SOURCES, SIZE);
    expect(barRect(a)[2]).toBeCloseTo(barRect(b)[2], 6);
  });

  it('a longer length draws a proportionally longer bar', () => {
    const a = fakeCtx(); paintFigure(a, figureWithBar(5), SOURCES, SIZE);
    const b = fakeCtx(); paintFigure(b, figureWithBar(20), SOURCES, SIZE);
    expect(barRect(b)[2] / barRect(a)[2]).toBeCloseTo(4, 6);
  });

  it('draws NOTHING when the figure carries no scale', () => {
    // Better an absent bar than one whose length means nothing. This is the
    // state every caller was in until stepUm got wired up.
    const ctx = fakeCtx();
    paintFigure(ctx, figureWithBar(10), { ...SOURCES, stepUm: null }, SIZE);
    expect(barRect(ctx)).toBeUndefined();
    expect(labels(ctx).some((s) => s.includes('µm'))).toBe(false);
  });
});

describe('a map-pixel bar, for files with no step size', () => {
  it('measures scan pixels and needs no micrometres', () => {
    // 30 scan px of a 120-column map drawn across the panel: three tenths less
    // than a quarter of it, whatever the step size is or is not.
    let m = defaultModel(['heatmap']);
    const panel = m.elements.find((e) => e.source === 'heatmap');
    m = addElement(m, {
      type: 'scalebar', mode: 'px', unit: 'px', lengthValue: 30,
      x: panel.x, y: panel.y + panel.h + 0.02, w: 0.3, h: 0.06,
      color: '#000000', fontSize: 12,
    });
    const noStep = { ...SOURCES, stepUm: null };
    const ctx = fakeCtx();
    paintFigure(ctx, m, noStep, SIZE);
    const rect = barRect(ctx);
    expect(rect).toBeTruthy();
    const { outPxPerScanPx } = heatmapScale(m, noStep, SIZE);
    expect(rect[2]).toBeCloseTo(30 * outPxPerScanPx, 6);
    expect(labels(ctx)).toContain('30 px');
  });
});

describe('niceScaleLength', () => {
  it('picks a 1/2/5 x 10^n length near a third of the map', () => {
    // 120 columns x 0.5 µm = 60 µm across; a third is 20.
    expect(niceScaleLength(STEP_UM, 1.0, MAP_COLS).valueUm).toBe(20);
    expect(niceScaleLength(0.05, 1.0, 100).valueUm).toBe(1);
  });
  it('refuses a nonsensical scale instead of returning 0-length', () => {
    expect(niceScaleLength(0, 1.0, MAP_COLS).valueUm).toBe(0);
    expect(niceScaleLength(STEP_UM, 1.0, 0).valueUm).toBe(0);
  });
  it('niceCount rounds a pixel count down the same 1/2/5 ladder', () => {
    expect(niceCount(40)).toBe(20);
    expect(niceCount(120 / 3)).toBe(20);
    expect(niceCount(7)).toBe(5);
    expect(niceCount(0)).toBe(0);
    expect(niceCount(-3)).toBe(0);
  });
});
