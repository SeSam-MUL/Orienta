// @vitest-environment jsdom
/**
 * Export with the tilt correction: the written picture has the sample's true
 * aspect ratio, and its scale bar stays true.
 *
 * AB6-like case: SE image 1024 × 768 at 0.0621 µm/px, foreshortened in X by
 * k = 2.92. On the sample the image is 1024·0.0621·2.92 = 185.7 µm wide and
 * 768·0.0621 = 47.7 µm high. Stretched to 2990 × 768 px, one pixel is again
 * 0.0621 µm in both directions.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { buildSingleCanvas, buildMontageCanvas, stretchBuilt } from './edsExportSources';

let realGetContext;
const drawn = [];
beforeAll(() => {
  realGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function fake(type) {
    if (type !== '2d') return null;
    const ctx = {
      canvas: this, globalAlpha: 1, globalCompositeOperation: 'source-over',
      fillStyle: '#000', font: '', textAlign: 'left', textBaseline: 'alphabetic',
      imageSmoothingEnabled: true, imageSmoothingQuality: 'low',
      clearRect() {}, fillRect() {}, fillText() {}, save() {}, restore() {}, putImageData() {},
      drawImage(...args) { drawn.push({ smoothing: ctx.imageSmoothingEnabled, args }); },
      getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
    };
    return ctx;
  };
});
afterAll(() => { HTMLCanvasElement.prototype.getContext = realGetContext; });

const SE = { width: 1024, height: 768 };
const layer = { id: 'electron-SE', kind: 'electron', visible: true };

describe('export with the true aspect ratio', () => {
  it('stretches X by k and keeps the micrometres per pixel', () => {
    const built = stretchBuilt(buildSingleCanvas(layer, SE, 0.0621), { sx: 2.92, sy: 1 }, { smooth: true });
    expect(built.canvas.width).toBe(2990);
    expect(built.canvas.height).toBe(768);
    expect(built.umPerPx).toBe(0.0621);
    // true widths on the sample: 2990 px · 0.0621 ≈ 1024 · 0.0621 · 2.92
    expect(built.canvas.width * built.umPerPx).toBeCloseTo(1024 * 0.0621 * 2.92, 0);
    expect(built.stretched).toEqual({ sx: 2.92, sy: 1 });
  });

  it('smooths an electron image but keeps a map pixel-exact', () => {
    drawn.length = 0;
    stretchBuilt(buildSingleCanvas(layer, SE, 0.0621), { sx: 2.92, sy: 1 }, { smooth: true });
    expect(drawn.at(-1).smoothing).toBe(true);
    stretchBuilt(buildSingleCanvas({ ...layer, kind: 'eds-element' }, { width: 12, height: 9 }, 0.658),
      { sx: 2.92, sy: 1 });
    expect(drawn.at(-1).smoothing).toBe(false);
  });

  it('returns the picture untouched without a stretch', () => {
    const built = buildSingleCanvas(layer, SE, 0.0621);
    expect(stretchBuilt(built, { sx: 1, sy: 1 })).toBe(built);
    expect(stretchBuilt(built, null)).toBe(built);
  });

  it('gives the montage cells the true aspect ratio', () => {
    const layers = [{ id: 'a', kind: 'eds-element', visible: true }];
    const bitmaps = new Map([['a', { width: 12, height: 9 }]]);
    const plain = buildMontageCanvas({ layers, bitmaps, shape: [9, 12], cellWidth: 320, gap: 0 });
    const wide = buildMontageCanvas({
      layers, bitmaps, shape: [9, 12], cellWidth: 320, gap: 0, stretch: { sx: 2.92, sy: 1 },
    });
    expect(wide.canvas.width).toBe(plain.canvas.width);
    expect(wide.canvas.height).toBeLessThan(plain.canvas.height);
    expect(wide.umPerPx).toBeNull();
  });
});
