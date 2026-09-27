// @vitest-environment jsdom
/**
 * A thresholded layer is cut out on its own canvas before it joins the map.
 *
 * The toolbar's old exporter honoured layer thresholds; the dialog path
 * (`composeMapCanvas`) ignored them, so routing the button through the dialog
 * would have silently dropped a threshold the user had set on a BC or CI
 * layer. The cut happens on a separate canvas so the alpha holes fall on that
 * layer only, not through the background and every layer below.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { composeMapCanvas } from './composeExport';

const calls = [];
let realGetContext;
beforeAll(() => {
  realGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function fake(type) {
    if (type !== '2d') return null;
    const canvas = this;
    return {
      canvas,
      globalAlpha: 1, globalCompositeOperation: 'source-over',
      fillStyle: '#000', imageSmoothingEnabled: true,
      clearRect() {}, fillRect() {},
      drawImage(src, ...rest) { calls.push(['drawImage', canvas, src, rest.length]); },
      putImageData() { calls.push(['putImageData', canvas]); },
      getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
    };
  };
});
afterAll(() => { HTMLCanvasElement.prototype.getContext = realGetContext; });

const bmp = (w, h) => ({ width: w, height: h });

describe('composeMapCanvas and layer thresholds', () => {
  it('cuts a thresholded scalar layer on its own canvas, then draws that canvas', async () => {
    calls.length = 0;
    const bitmaps = new Map([['bc', bmp(21, 21)]]);
    const layers = [{ id: 'bc', kind: 'bc', visible: true, opacity: 1, threshold: { min: 40, max: 200 } }];
    const out = await composeMapCanvas({ layers, bitmaps, shape: { rows: 21, cols: 21 }, scale: 2 });
    expect(out.canvas.width).toBe(42);
    const puts = calls.filter((c) => c[0] === 'putImageData');
    expect(puts).toHaveLength(1);                       // the threshold was applied once
    expect(puts[0][1]).not.toBe(out.canvas);            // ...on a canvas that is not the map
    const draws = calls.filter((c) => c[0] === 'drawImage' && c[1] === out.canvas);
    expect(draws).toHaveLength(1);
    expect(draws[0][2]).toBeInstanceOf(HTMLCanvasElement); // the cut, not the raw bitmap
  });

  it('draws an unthresholded layer straight from its bitmap', async () => {
    calls.length = 0;
    const bitmaps = new Map([['phase', bmp(21, 21)]]);
    const layers = [{ id: 'phase', kind: 'phase', visible: true, opacity: 1 }];
    const out = await composeMapCanvas({ layers, bitmaps, shape: { rows: 21, cols: 21 }, scale: 2 });
    expect(calls.some((c) => c[0] === 'putImageData')).toBe(false);
    const draws = calls.filter((c) => c[0] === 'drawImage' && c[1] === out.canvas);
    expect(draws).toHaveLength(1);
    expect(draws[0][2]).toBe(bitmaps.get('phase'));
  });

  it('draws a mask layer from the layer it masks', async () => {
    calls.length = 0;
    const bitmaps = new Map([['bc', bmp(21, 21)]]);
    const layers = [{ id: 'mask-bc', kind: 'mask', isMaskFor: 'bc', visible: true, opacity: 1,
                      blend: 'multiply', threshold: { min: 40, max: 200 } }];
    const out = await composeMapCanvas({ layers, bitmaps, shape: { rows: 21, cols: 21 }, scale: 2 });
    // buildMaskCanvas reads the SOURCE bitmap and writes a B/W canvas, which
    // is then drawn into the cut canvas and the cut into the map.
    expect(calls.filter((c) => c[0] === 'putImageData')).toHaveLength(1);
    const draws = calls.filter((c) => c[0] === 'drawImage' && c[1] === out.canvas);
    expect(draws).toHaveLength(1);
  });
});
