// @vitest-environment jsdom
/**
 * The scale bar of an exported figure, end to end.
 *
 * The rule these tests exist to enforce is one sentence:
 *
 *     `unitsPerPixel` is the micrometres per pixel OF THE EXPORTED BITMAP.
 *
 * It is NOT "the pixel size of the area the layer came from". Those two agree
 * only while the bitmap happens to be that area's native raster, and three of
 * the export paths resample:
 *
 *   * the composite draws every layer into the SCAN grid, whatever raster it
 *     came from — measured 2026-08-27 on SampleB, the SE image is 1024 px wide
 *     at 0.058973 um/px and gets stretched into 120 scan columns at 0.5 um/px,
 *     a factor of 8.4785;
 *   * the montage stretches each panel into a fixed 320 px cell and puts gaps
 *     between them, so no single number is true for the sheet;
 *   * the phase-map export draws at 2x.
 *
 * Every case here is checked the same way: take the bar the dialog would draw,
 * work out how long it REALLY is on the specimen, and compare with its label.
 * A ratio of 1 is a correct scale bar. Nothing else is.
 */
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { buildSingleCanvas, buildCompositeCanvas, buildMontageCanvas } from './edsExportSources';
import { pixelSizeForLayer } from './layerPixelSize';
import { niceScalebar } from '../common/imageExport';

// Read out of Test_data/EBSD_SampleB_...h5oina on 2026-08-27:
//   EBSD / EDS      120 x 90    X Step 0.5        um   (bbox 60    x 45)
//   Electron Image  1024 x 768  X Step 0.05897275 um   (bbox 60.39 x 45.29)
const PIXEL_SIZES = {
  ebsd: { x: 0.5, y: 0.5, units: 'um', source: 'step' },
  eds: { x: 0.5, y: 0.5, units: 'um', source: 'step' },
  electron_image: { x: 0.05897275358438492, y: 0.05897275358438492, units: 'um', source: 'step' },
};
const SCAN = [90, 120];          // [rows, cols]
const SCAN_UM_PER_PX = 0.5;
const SE_UM_PER_PX = 0.05897275358438492;
const SE_SIZE = { w: 1024, h: 768 };

// --- jsdom has no 2d context; the builders only need the drawing calls to be
// swallowed and the canvas to keep the width/height they set on it.
let realGetContext;
beforeAll(() => {
  realGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function fake(type) {
    if (type !== '2d') return null;
    return {
      canvas: this,
      globalAlpha: 1, globalCompositeOperation: 'source-over',
      fillStyle: '#000', strokeStyle: '#000', font: '', textAlign: 'left',
      textBaseline: 'alphabetic', imageSmoothingEnabled: true, lineWidth: 1,
      clearRect() {}, fillRect() {}, strokeRect() {}, drawImage() {},
      fillText() {}, save() {}, restore() {}, translate() {}, rotate() {},
      beginPath() {}, arc() {}, stroke() {}, putImageData() {},
      getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
      createLinearGradient: () => ({ addColorStop() {} }),
      measureText: () => ({ width: 10 }),
    };
  };
  HTMLCanvasElement.prototype.toDataURL = () => 'data:image/png;base64,';
});
afterAll(() => { HTMLCanvasElement.prototype.getContext = realGetContext; });

/** A stand-in for an ImageBitmap: the builders only read width/height. */
const bmp = (w, h) => ({ width: w, height: h });

/**
 * How long the drawn bar really is, divided by what its label claims.
 * 1.0 = honest.
 */
function labelHonesty({ canvas, umPerPx }, trueUmPerPx) {
  const bar = niceScalebar(umPerPx, canvas.width, 0.25);
  if (!bar) return null;
  return (bar.lengthPx * trueUmPerPx) / bar.lengthUnits;
}

describe('a single map exports at its own raster', () => {
  it('the SE image keeps the SE pixel size', () => {
    const layer = { id: 'electron-SE/Elektronenbild 1', kind: 'electron', visible: true };
    const built = buildSingleCanvas(
      layer, bmp(SE_SIZE.w, SE_SIZE.h),
      pixelSizeForLayer(layer.id, PIXEL_SIZES)?.x,
    );
    expect(built.canvas.width).toBe(SE_SIZE.w);
    expect(built.umPerPx).toBeCloseTo(SE_UM_PER_PX, 9);
    expect(labelHonesty(built, SE_UM_PER_PX)).toBeCloseTo(1, 6);
  });

  it('an element map keeps the scan step', () => {
    const layer = { id: 'eds-Al Kα1', kind: 'eds-element', visible: true };
    const built = buildSingleCanvas(
      layer, bmp(SCAN[1], SCAN[0]),
      pixelSizeForLayer(layer.id, PIXEL_SIZES)?.x,
    );
    expect(built.umPerPx).toBeCloseTo(SCAN_UM_PER_PX, 9);
    expect(labelHonesty(built, SCAN_UM_PER_PX)).toBeCloseTo(1, 6);
  });

  it('a mask follows the raster of the layer it masks, not its own id', () => {
    // `mask-<sourceId>-<stamp>`: the id names the mask, but the pixels drawn
    // are the SOURCE bitmap's. Reading the area off the mask id put the scan
    // step on a 1024 px SE image — 8.48x wrong.
    const src = 'electron-SE/Elektronenbild 1';
    const layer = {
      id: `mask-${src}-1756300000000`, kind: 'mask', isMaskFor: src,
      visible: true, threshold: { min: 20, max: 200 },
    };
    const size = pixelSizeForLayer(layer.id, PIXEL_SIZES);
    expect(size?.x).toBeCloseTo(SE_UM_PER_PX, 9);
    const built = buildSingleCanvas(layer, bmp(SE_SIZE.w, SE_SIZE.h), size?.x);
    expect(labelHonesty(built, SE_UM_PER_PX)).toBeCloseTo(1, 6);
  });
});

describe('the composite is drawn in the scan grid, whatever is stacked on it', () => {
  const layers = [
    { id: 'electron-SE/Elektronenbild 1', kind: 'electron', visible: true, opacity: 0.6, blend: 'normal' },
    { id: 'eds-Al Kα1', kind: 'eds-element', visible: true, opacity: 0.7, blend: 'screen' },
  ];
  const bitmaps = new Map([
    ['electron-SE/Elektronenbild 1', bmp(SE_SIZE.w, SE_SIZE.h)],
    ['eds-Al Kα1', bmp(SCAN[1], SCAN[0])],
  ]);

  it('reports the SCAN step even though the bottom layer is the SE image', () => {
    const built = buildCompositeCanvas({
      layers, bitmaps, shape: SCAN, umPerPx: pixelSizeForLayer('phase', PIXEL_SIZES)?.x,
    });
    expect(built.canvas.width).toBe(SCAN[1]);
    expect(built.umPerPx).toBeCloseTo(SCAN_UM_PER_PX, 9);
    expect(labelHonesty(built, SCAN_UM_PER_PX)).toBeCloseTo(1, 6);
  });

  it('the old rule — bottom layer decides — was wrong by exactly the area ratio', () => {
    // Kept as a witness so nobody reintroduces it: this is what the page did
    // until 2026-08-27.
    const wrong = buildCompositeCanvas({
      layers, bitmaps, shape: SCAN, umPerPx: pixelSizeForLayer(layers[0].id, PIXEL_SIZES)?.x,
    });
    expect(labelHonesty(wrong, SCAN_UM_PER_PX)).toBeCloseTo(SCAN_UM_PER_PX / SE_UM_PER_PX, 3);
  });
});

describe('a montage has no single scale', () => {
  it('reports none rather than a number that is true for no panel', () => {
    const layers = [
      { id: 'eds-Al Kα1', kind: 'eds-element', visible: true },
      { id: 'electron-SE/Elektronenbild 1', kind: 'electron', visible: true },
    ];
    const bitmaps = new Map([
      ['eds-Al Kα1', bmp(SCAN[1], SCAN[0])],
      ['electron-SE/Elektronenbild 1', bmp(SE_SIZE.w, SE_SIZE.h)],
    ]);
    const built = buildMontageCanvas({ layers, bitmaps, shape: SCAN, labelFor: (l) => l.id });
    expect(built.umPerPx).toBeNull();
    expect(niceScalebar(built.umPerPx, built.canvas.width, 0.25)).toBeNull();
  });
});
