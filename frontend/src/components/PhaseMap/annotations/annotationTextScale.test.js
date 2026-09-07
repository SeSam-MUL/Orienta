// What the user arranged in the preview must be what lands in the file.
//
// The preview draws annotation text at `props.fontSize` in the preview's own
// CSS pixels (AnnotationLayer). The burn-in multiplies that same number by a
// `textScale`. So the ONLY value that keeps a figure looking like its preview
// is "how many output pixels per preview pixel".
//
// Shipped bug (2026-09-03, user-reported "die Größe der Maßstäbe passt nicht"):
// the caller passed `geom.sx`, which is output pixels per SOURCE DATA pixel.
// On an 8x export of a 136x39 scan that made every caption ~4x too large: the
// scale bar's plate came out 245 px tall in a 312 px image and its "2 µm"
// label was cut off by the bottom edge.
//
// These tests measure the drawn geometry through the real drawer, using a
// recording 2D context (jsdom has no canvas).
import { describe, it, expect } from 'vitest';
import { drawAnnotationsOnto } from './composeExport';

/** Minimal recording stand-in for a CanvasRenderingContext2D. */
function recordingCtx() {
  const calls = [];
  let tx = 0, ty = 0;
  const ctx = {
    save() {}, restore() { tx = 0; ty = 0; },
    translate(x, y) { tx += x; ty += y; },
    rotate() {},
    fillRect(x, y, w, h) { calls.push({ op: 'fillRect', x: tx + x, y: ty + y, w, h }); },
    strokeRect() {},
    fillText(text, x, y) { calls.push({ op: 'fillText', text, x: tx + x, y: ty + y }); },
    measureText: (t) => ({ width: String(t).length * 6 }),
    createLinearGradient: () => ({ addColorStop() {} }),
    drawImage() {},
    beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, closePath() {}, arc() {},
    set font(_v) {}, get font() { return ''; },
  };
  return { ctx, calls };
}

const SCALEBAR = {
  type: 'scalebar', x: 0.05, y: 0.75, w: 0.3, h: 0.1, rotation: 0,
  // No plate colour: the only fillRect is then the bar itself.
  props: { lengthUm: 2, fontSize: 12, barColor: '#ffffff', textColor: '#ffffff' },
};

// The user's real figure: a 136 x 39 scan exported at 8x.
const SCAN_COLS = 136;
const STEP_X = 0.1;
const PREVIEW = { w: 600, h: 172 };
const FILE = { w: 1088, h: 312 };

/** Draw one annotation and return the bar rect + label, as fractions. */
function drawFractions(size, textScale) {
  const { ctx, calls } = recordingCtx();
  drawAnnotationsOnto(ctx, [SCALEBAR], {
    width: size.w, height: size.h, stepX: STEP_X, scanCols: SCAN_COLS, textScale,
  });
  const bar = calls.find((c) => c.op === 'fillRect');
  const label = calls.find((c) => c.op === 'fillText');
  expect(bar, 'no bar was drawn').toBeTruthy();
  expect(label, 'no label was drawn').toBeTruthy();
  return {
    barW: bar.w / size.w,
    barY: bar.y / size.h,
    barH: bar.h / size.h,
    labelY: label.y / size.h,
  };
}

describe('annotation size in the file matches the preview', () => {
  it('the bar keeps its physical length either way (that part was never wrong)', () => {
    // 2 µm of a 136-column scan at 0.1 µm/px = 14.7% of the picture width.
    for (const [size, ts] of [[PREVIEW, 1], [FILE, FILE.w / PREVIEW.w]]) {
      expect(drawFractions(size, ts).barW).toBeCloseTo((2 / STEP_X) / SCAN_COLS, 3);
    }
  });

  it('preview-relative textScale reproduces the preview', () => {
    // Tolerance 0.005 of the picture height, not 0.0005: the layout carries a
    // couple of unscaled constants (`+2` between bar and label, `-4` in the
    // vertical centring), which leaves a residual of ~0.26% here. That is a
    // rounding artefact of the formula, not the defect — the defect below is
    // a hundred times larger.
    const preview = drawFractions(PREVIEW, 1);
    const file = drawFractions(FILE, FILE.w / PREVIEW.w);
    expect(file.barY).toBeCloseTo(preview.barY, 2);
    expect(file.barH).toBeCloseTo(preview.barH, 2);
    expect(file.labelY).toBeCloseTo(preview.labelY, 2);
  });

  it('data-pixel textScale (the shipped bug) blows the annotation up', () => {
    // geom.sx for this figure is 8 — output px per SOURCE px.
    const preview = drawFractions(PREVIEW, 1);
    const wrong = drawFractions(FILE, 8);
    // The label is pushed far down the picture and off the bottom edge.
    expect(wrong.labelY).toBeGreaterThan(preview.labelY + 0.25);
    expect(wrong.labelY).toBeGreaterThan(1.0);
  });
});
