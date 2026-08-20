import { describe, it, expect } from 'vitest';
import {
  boundsOf, rectMask, countSelected, estimateBytes, formatBytes, bytesPerSample,
} from './navSelection';

describe('boundsOf', () => {
  it('spans the drawn points inclusively', () => {
    expect(boundsOf([{ r: 4, c: 9 }, { r: 7, c: 2 }]))
      .toEqual({ row0: 4, col0: 2, rows: 4, cols: 8 });
  });

  it('is a single pixel for a single point', () => {
    expect(boundsOf([{ r: 3, c: 3 }]))
      .toEqual({ row0: 3, col0: 3, rows: 1, cols: 1 });
  });

  it('returns null for nothing drawn', () => {
    expect(boundsOf([])).toBeNull();
    expect(boundsOf(null)).toBeNull();
  });
});

describe('rectMask', () => {
  it('is null — a rectangle selects its whole box', () => {
    expect(rectMask({ row0: 0, col0: 0, rows: 3, cols: 4 })).toBeNull();
  });
});

describe('countSelected', () => {
  it('counts the whole box when there is no mask', () => {
    expect(countSelected(null, { row0: 0, col0: 0, rows: 3, cols: 4 })).toBe(12);
  });

  it('counts the true entries of a mask', () => {
    const mask = new Uint8Array(12);
    mask[0] = 1; mask[5] = 1;
    expect(countSelected(mask, { row0: 0, col0: 0, rows: 3, cols: 4 })).toBe(2);
  });
});

describe('estimateBytes', () => {
  it('is pixels times pattern size times depth', () => {
    // 8x10 pixels of 128x156 uint8
    expect(estimateBytes(8, 10, [128, 156], 1)).toBe(8 * 10 * 128 * 156);
  });

  it('scales with the sample depth', () => {
    expect(estimateBytes(2, 2, [4, 4], 2)).toBe(128);
  });

  it('is 0 when the pattern shape is unknown', () => {
    expect(estimateBytes(8, 10, null, 1)).toBe(0);
  });
});

describe('formatBytes', () => {
  it('reads as a person would say it', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(256 * 1024 * 1024)).toBe('256.0 MB');
    expect(formatBytes(3 * 1024 ** 3)).toBe('3.0 GB');
  });
});

describe('bytesPerSample', () => {
  it('reads the width out of the numpy dtype name', () => {
    expect(bytesPerSample('uint8')).toBe(1);
    expect(bytesPerSample('uint16')).toBe(2);
    expect(bytesPerSample('int16')).toBe(2);
    expect(bytesPerSample('float32')).toBe(4);
    expect(bytesPerSample('float64')).toBe(8);
  });

  it('falls back to 1 when the dtype is unknown — the pre-dtype assumption', () => {
    expect(bytesPerSample(null)).toBe(1);
    expect(bytesPerSample(undefined)).toBe(1);
    expect(bytesPerSample('')).toBe(1);
    expect(bytesPerSample('bool')).toBe(1);
    expect(bytesPerSample('uint4')).toBe(1);   // sub-byte: never less than 1
  });
});

import { ellipseMask, lassoMask } from './navSelection';

const at = (mask, bbox, r, c) => mask[(r - bbox.row0) * bbox.cols + (c - bbox.col0)];

describe('ellipseMask', () => {
  it('fills the inscribed ellipse of the box', () => {
    const bbox = { row0: 0, col0: 0, rows: 5, cols: 5 };
    const mask = ellipseMask(bbox);
    expect(at(mask, bbox, 2, 2)).toBe(1);   // centre
    expect(at(mask, bbox, 2, 0)).toBe(1);   // on the horizontal axis
    expect(at(mask, bbox, 0, 2)).toBe(1);   // on the vertical axis
    expect(at(mask, bbox, 0, 0)).toBe(0);   // corner is outside
    expect(at(mask, bbox, 4, 4)).toBe(0);
  });

  it('handles a one-pixel box', () => {
    const mask = ellipseMask({ row0: 0, col0: 0, rows: 1, cols: 1 });
    expect(mask.length).toBe(1);
    expect(mask[0]).toBe(1);
  });

  it('is wide for a wide box — the radii are not transposed', () => {
    // 3 rows x 9 cols. The middle row must reach both ends; the corners must
    // not. Swapping rx and ry would fill a tall sliver down the middle
    // instead, and the square boxes above could never tell the difference.
    const bbox = { row0: 0, col0: 0, rows: 3, cols: 9 };
    const mask = ellipseMask(bbox);
    expect(at(mask, bbox, 1, 0)).toBe(1);
    expect(at(mask, bbox, 1, 8)).toBe(1);
    expect(at(mask, bbox, 0, 0)).toBe(0);
    expect(at(mask, bbox, 2, 8)).toBe(0);
  });
});

describe('lassoMask', () => {
  it('fills a drawn triangle and leaves the rest out', () => {
    // Triangle with corners (0,0), (0,4), (4,0) in grid coordinates.
    const pts = [{ r: 0, c: 0 }, { r: 0, c: 4 }, { r: 4, c: 0 }];
    const bbox = boundsOf(pts);
    const mask = lassoMask(pts, bbox);
    expect(at(mask, bbox, 0, 0)).toBe(1);
    expect(at(mask, bbox, 1, 1)).toBe(1);
    expect(at(mask, bbox, 3, 3)).toBe(0);   // beyond the hypotenuse
    expect(at(mask, bbox, 4, 4)).toBe(0);
  });

  it('handles a concave shape — the notch stays out', () => {
    // A "C": a 5x5 box with the middle-right bitten out.
    const pts = [
      { r: 0, c: 0 }, { r: 0, c: 4 }, { r: 1, c: 4 }, { r: 1, c: 1 },
      { r: 3, c: 1 }, { r: 3, c: 4 }, { r: 4, c: 4 }, { r: 4, c: 0 },
    ];
    const bbox = boundsOf(pts);
    const mask = lassoMask(pts, bbox);
    expect(at(mask, bbox, 0, 2)).toBe(1);   // top bar
    expect(at(mask, bbox, 2, 0)).toBe(1);   // spine
    expect(at(mask, bbox, 2, 3)).toBe(0);   // the notch
  });

  it('closes an open path automatically', () => {
    const open = [{ r: 0, c: 0 }, { r: 0, c: 3 }, { r: 3, c: 3 }, { r: 3, c: 0 }];
    const bbox = boundsOf(open);
    expect(at(lassoMask(open, bbox), bbox, 1, 1)).toBe(1);
  });

  it('selects nothing for fewer than three points', () => {
    const bbox = { row0: 0, col0: 0, rows: 2, cols: 2 };
    expect(Array.from(lassoMask([{ r: 0, c: 0 }], bbox))).toEqual([0, 0, 0, 0]);
  });

  it('fills an asymmetric shape at a non-zero origin', () => {
    // Right triangle (10,20)-(10,27)-(13,27): 4 rows x 8 cols, drawn far from
    // the origin. The hypotenuse runs from (10,20) to (13,27), so at row 11 it
    // sits at column 20 + 7/3 = 22.33 and at row 12 at 20 + 14/3 = 24.67 —
    // the filled part of each row starts just right of that. An off-by-one in
    // the origin, or a transposed axis, breaks this shape immediately.
    const pts = [{ r: 10, c: 20 }, { r: 10, c: 27 }, { r: 13, c: 27 }];
    const bbox = boundsOf(pts);
    expect(bbox).toEqual({ row0: 10, col0: 20, rows: 4, cols: 8 });
    const mask = lassoMask(pts, bbox);
    expect(at(mask, bbox, 10, 20)).toBe(1);   // the wide top edge
    expect(at(mask, bbox, 10, 26)).toBe(1);
    expect(at(mask, bbox, 11, 20)).toBe(0);   // left of the hypotenuse
    expect(at(mask, bbox, 11, 23)).toBe(1);   // right of it
    expect(at(mask, bbox, 12, 24)).toBe(0);
    expect(at(mask, bbox, 12, 25)).toBe(1);   // narrowing towards the apex
  });
});
