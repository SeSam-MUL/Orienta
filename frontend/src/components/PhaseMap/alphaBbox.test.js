/**
 * A handful of stray pixels must not decide how large the map is drawn.
 *
 * BUG-PM-03: a small region run (608 px inside a ~168x150 grid) came out
 * "almost black" — the indexed area a thin bright strip at the top of an
 * otherwise empty canvas. The auto-zoom is supposed to blow that region up to
 * fill the frame; it measured the box with min/max over every pixel carrying
 * alpha, so three speckles in the far corners spanned the box across the whole
 * grid, the box then counted as "full coverage" and the zoom abstained.
 *
 * The frame follows the BODY of the content. An isolated fringe holding a few
 * percent of the pixels is a finding, not a region.
 */
import { describe, it, expect } from 'vitest';
import { bboxFromAlpha } from './alphaBbox';

// RGBA buffer with a given set of opaque pixels.
function canvas(w, h, pixels) {
  const d = new Uint8ClampedArray(w * h * 4);
  for (const [x, y] of pixels) d[(y * w + x) * 4 + 3] = 255;
  return d;
}

function block(x0, y0, bw, bh) {
  const out = [];
  for (let y = y0; y < y0 + bh; y++) for (let x = x0; x < x0 + bw; x++) out.push([x, y]);
  return out;
}

describe('bboxFromAlpha — the frame follows the body', () => {
  it('frames the 5x5 block, not the three speckles far away', () => {
    const px = [...block(10, 10, 5, 5), [2, 2], [40, 30], [5, 45]];
    expect(bboxFromAlpha(canvas(50, 50, px), 50, 50))
      .toEqual({ x: 10, y: 10, w: 5, h: 5 });
  });

  it('a clean region without speckles is unchanged', () => {
    const px = block(10, 10, 5, 5);
    expect(bboxFromAlpha(canvas(50, 50, px), 50, 50))
      .toEqual({ x: 10, y: 10, w: 5, h: 5 });
  });

  it('speckles on one axis only are trimmed on that axis', () => {
    // strays share the block's rows, so only the columns need trimming
    const px = [...block(10, 10, 5, 5), [2, 11], [45, 12]];
    expect(bboxFromAlpha(canvas(50, 50, px), 50, 50))
      .toEqual({ x: 10, y: 10, w: 5, h: 5 });
  });

  it('keeps a second region that carries real weight', () => {
    // two blocks of equal size: both are the result, the frame spans both
    const px = [...block(5, 5, 5, 5), ...block(30, 30, 5, 5)];
    expect(bboxFromAlpha(canvas(50, 50, px), 50, 50))
      .toEqual({ x: 5, y: 5, w: 30, h: 30 });
  });

  it('keeps a much smaller second region — 900 px next to 100 px', () => {
    // The case that made the first rule wrong: equal blocks are the easy
    // half. 100 px is a tenth of the content and an eleventh of the frame,
    // and the first rule dropped it — which means not drawn at all, because
    // the caller hard-crops to this box and zoom sits after the crop.
    const px = [...block(10, 10, 30, 30), ...block(140, 140, 10, 10)];
    expect(bboxFromAlpha(canvas(200, 200, px), 200, 200))
      .toEqual({ x: 10, y: 10, w: 140, h: 140 });
  });

  it('keeps a 30-px region beside a 900-px one, and still drops a 3-px speckle', () => {
    // Where the line sits now: a couple of dozen pixels, not a share of the
    // layer. Same geometry, only the size of the far cluster changes.
    const big = block(10, 10, 30, 30);
    const keep = bboxFromAlpha(canvas(200, 200, [...big, ...block(140, 140, 6, 5)]), 200, 200);
    expect(keep, '30 px is a region').toEqual({ x: 10, y: 10, w: 136, h: 135 });

    const drop = bboxFromAlpha(canvas(200, 200, [...big, ...block(140, 140, 3, 1)]), 200, 200);
    expect(drop, '3 px is a speckle').toEqual({ x: 10, y: 10, w: 30, h: 30 });
  });

  it('still abstains for a full-width layer with a gap in it', () => {
    // 5 indexed rows at the top, nothing until row 100, then rows 100-199.
    // The box spans the whole canvas, so the answer is null — "this layer
    // does not constrain the frame", which is what roiFrame.js relies on to
    // let a full-coverage layer stand aside. Cropping to rows 100-199 would
    // both break that and throw away 1000 real pixels.
    const px = [...block(0, 0, 200, 5), ...block(0, 100, 200, 100)];
    expect(bboxFromAlpha(canvas(200, 200, px), 200, 200)).toBeNull();
  });

  it('does not nibble at a round region — no gap, nothing to drop', () => {
    // a filled disc r=10 at (25,25): the tip rows are thin but contiguous,
    // and cropping them would cut off real indexed pixels
    const px = [];
    for (let y = 0; y < 50; y++) {
      for (let x = 0; x < 50; x++) {
        if ((x - 25) ** 2 + (y - 25) ** 2 <= 100) px.push([x, y]);
      }
    }
    expect(bboxFromAlpha(canvas(50, 50, px), 50, 50))
      .toEqual({ x: 15, y: 15, w: 21, h: 21 });
  });

  it('still abstains for a layer that covers the whole canvas', () => {
    expect(bboxFromAlpha(canvas(8, 8, block(0, 0, 8, 8)), 8, 8)).toBeNull();
  });

  it('answers null for an empty layer', () => {
    expect(bboxFromAlpha(canvas(8, 8, []), 8, 8)).toBeNull();
  });

  it('survives a degenerate canvas', () => {
    expect(bboxFromAlpha(new Uint8ClampedArray(0), 0, 0)).toBeNull();
  });
});
