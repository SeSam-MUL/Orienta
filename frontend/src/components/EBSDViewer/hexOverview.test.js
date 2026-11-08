import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { makeHexMap } from '../../services/hexMap';
import { parseScanGeometry } from '../../services/scanGeometry';
import {
  overviewPointer, crosshairPercent, selectionCrop, roiLabelBox, overviewMismatch, rasterKey, overviewScale,
} from './hexOverview';
import { countSelected } from './navSelection';

// Written by tests/hex_raster_oracle.py from the lattice alone; the expectations below come from it or are
// worked by hand, never from hexGeometry.js.
const FIXTURE = JSON.parse(
  readFileSync(new URL('../../../../tests/fixtures/hex_raster_cases.json', import.meta.url), 'utf8'));
const CASE = FIXTURE.geometries.find((c) => c.name === 'a_3x3_right');       // a 12 x 14 raster
const hexMap = makeHexMap(parseScanGeometry(CASE.geometry));
const [H, W] = hexMap.shape;

describe('overviewPointer', () => {
  it('names the measurement point under every sampled pixel of the fixture', () => {
    for (const p of CASE.pixels) {
      const got = overviewPointer(hexMap, (p.j + 0.5) / W, (p.i + 0.5) / H);
      expect(got.sr).toBe(p.i);
      expect(got.sc).toBe(p.j);
      expect(got.r === null ? null : [got.r, got.c]).toEqual(p.cell);
    }
  });

  it('keeps the pixel but names no point in a corner no hexagon covers', () => {
    const corner = CASE.pixels.find((p) => p.cell === null);
    const got = overviewPointer(hexMap, (corner.j + 0.5) / W, (corner.i + 0.5) / H);
    expect(got).toMatchObject({ r: null, c: null, sr: corner.i, sc: corner.j });
  });

  it('clamps the far edge into the last pixel and refuses outside the picture', () => {
    expect(overviewPointer(hexMap, 1, 1)).toMatchObject({ sr: H - 1, sc: W - 1 });
    for (const [rx, ry] of [[1.01, 0.5], [0.5, -0.01], [NaN, 0.5]]) {
      expect(overviewPointer(hexMap, rx, ry)).toBeNull();
    }
  });
});

describe('crosshairPercent', () => {
  it('puts the crosshair on the exact centre of the cell, by hand', () => {
    // cell (0, 0) is at (0, 0) um; the raster is 14 x 12 pixels of 0.5 um from (-1, -1.155) um:
    // 1 um of 7 across, 1.155 um of 6 down
    const p = crosshairPercent(hexMap, 0, 0);
    expect(p.x).toBeCloseTo((1 / 7) * 100, 9);
    expect(p.y).toBeCloseTo((Math.sqrt(3) / 9) * 100, 9);
  });
});

describe('selectionCrop', () => {
  const rectPoints = [{ r: 3, c: 3 }, { r: 9, c: 9 }];

  it('is the crop window of the cells whose centres the drag contains, by hand', () => {
    // x in [0.5, 4.0), y in [0.345, 3.845): cells (1,0), (1,1), (2,1)
    const crop = selectionCrop(hexMap, 'rect', rectPoints);
    expect(crop.bbox).toEqual({ row0: 1, col0: 0, rows: 2, cols: 2 });
    expect(Array.from(crop.mask)).toEqual([1, 1, 0, 1]);
  });

  it('reads the pixels as (row, column) = (i, j), which a symmetric drag cannot show', () => {
    // pixel rows 3..9 and columns 2..12: x in [0, 5.5), y in [0.345, 3.845) -> rows 1 and 2, every cell of them
    const crop = selectionCrop(hexMap, 'rect', [{ r: 3, c: 2 }, { r: 9, c: 12 }]);
    expect(crop.bbox).toEqual({ row0: 1, col0: 0, rows: 2, cols: 3 });
    expect(crop.mask).toBeNull();
  });

  it('hands the tool on: an ellipse drawn over the same pixels drops the corners', () => {
    // pixels 2..10 x 2..12: x in [0, 5.5), y in [-0.155, 4.345); the ellipse centre (2.75, 2.095), semi-axes
    // (2.75, 2.25). Cell centres: row 0 at x 0, 2, 4 -> only x = 2 is inside (1.87, 0.94, 1.07 in ellipse units);
    // row 1 at x 1, 3, 5 -> all three; row 2 at x 0, 2, 4 -> x = 2 and 4.
    const points = [{ r: 2, c: 2 }, { r: 10, c: 12 }];
    expect(selectionCrop(hexMap, 'rect', points).mask).toBeNull();
    const crop = selectionCrop(hexMap, 'ellipse', points);
    expect(crop.bbox).toEqual({ row0: 0, col0: 0, rows: 3, cols: 3 });
    expect(Array.from(crop.mask)).toEqual([0, 1, 0, 1, 1, 1, 0, 1, 1]);
  });

  it('reads no tool as the Shift+drag rectangle', () => {
    expect(selectionCrop(hexMap, null, rectPoints)).toEqual(selectionCrop(hexMap, 'rect', rectPoints));
  });

  it('is read by the crop panel\'s own counter as the number of cells chosen, by hand', () => {
    // the rectangle above chooses 3 cells, the ellipse above 6 (see the two hand calculations)
    expect(countSelected(selectionCrop(hexMap, 'rect', rectPoints).mask, selectionCrop(hexMap, 'rect', rectPoints).bbox)).toBe(3);
    const ell = selectionCrop(hexMap, 'ellipse', [{ r: 2, c: 2 }, { r: 10, c: 12 }]);
    expect(countSelected(ell.mask, ell.bbox)).toBe(6);
    // a box of which every cell is chosen has no mask and counts rows * cols
    const all = selectionCrop(hexMap, 'rect', [{ r: 3, c: 2 }, { r: 9, c: 12 }]);
    expect(countSelected(all.mask, all.bbox)).toBe(2 * 3);
  });

  it('answers null for no points and for a selection in a corner', () => {
    expect(selectionCrop(hexMap, 'rect', [])).toBeNull();
    expect(selectionCrop(hexMap, 'rect', [{ r: 0, c: 0 }, { r: 0, c: 0 }])).toBeNull();
  });
});

describe('roiLabelBox', () => {
  it('prints the cells of the crop window on a hexagonal scan, the drag itself otherwise', () => {
    const crop = selectionCrop(hexMap, 'rect', [{ r: 3, c: 3 }, { r: 9, c: 9 }]);
    expect(roiLabelBox(crop, { startRow: 3, startCol: 3, endRow: 9, endCol: 9 }))
      .toEqual({ startRow: 1, startCol: 0, endRow: 2, endCol: 1 });
    const roi = { startRow: 3, startCol: 3, endRow: 9, endCol: 9 };
    expect(roiLabelBox(null, roi)).toBe(roi);
  });
});

describe('rasterKey', () => {
  const g = { kind: 'hex', n_rows: 3, n_cols: 3, step_x: 2, step_y: 1.5, odd_row_shift: 0.5, shift_confirmed: true };

  it('is null where there is no raster: a square scan, nothing, a stray value', () => {
    expect(rasterKey({ kind: 'square', n_rows: 3, n_cols: 3 })).toBeNull();
    expect(rasterKey(null)).toBeNull();
    expect(rasterKey(undefined)).toBeNull();
    expect(rasterKey('hex')).toBeNull();
  });

  it('is the same for the same raster, whatever else the stamp carries', () => {
    expect(rasterKey({ ...g, shift_confirmed: false, grid_shape: [3, 3] })).toBe(rasterKey(g));
  });

  it('differs for each field that decides the raster, the sign of the offset included', () => {
    const k = rasterKey(g);
    for (const change of [{ n_rows: 4 }, { n_cols: 4 }, { step_x: 3 }, { step_y: 1 }, { odd_row_shift: -0.5 }]) {
      expect(rasterKey({ ...g, ...change }), JSON.stringify(change)).not.toBe(k);
    }
  });
});

describe('overviewMismatch', () => {
  // What the backend stamps for the fixture's scan: the oracle's own `spec`, in the backend's field names.
  const STAMP = { kind: 'hex_raster', factor: CASE.spec.factor, shape: [CASE.spec.height, CASE.spec.width],
    step_um: CASE.spec.step_um, x0: CASE.spec.x0, y0: CASE.spec.y0 };

  it('is false when the picture is the raster the page expects', () => {
    expect(overviewMismatch(hexMap, STAMP)).toBe(false);
  });

  it('is true when the answer has no raster, or another one, for a hexagonal scan', () => {
    expect(overviewMismatch(hexMap, null)).toBe(true);
    expect(overviewMismatch(hexMap, undefined)).toBe(true);
    expect(overviewMismatch(hexMap, { ...STAMP, shape: [H, W + 1] })).toBe(true);
  });

  it('is true for a raster of the same shape that starts a half step to the side', () => {
    // the scan's step is 2 um, so its half step is 1 um: x0 -1 becomes -2
    expect(overviewMismatch(hexMap, { ...STAMP, x0: -2 })).toBe(true);
  });

  it('is true when a square page is handed a raster, and false when it is not', () => {
    expect(overviewMismatch(null, STAMP)).toBe(true);
    expect(overviewMismatch(null, undefined)).toBe(false);
    expect(overviewMismatch(null, null)).toBe(false);
  });
});

describe('overviewScale', () => {
  it('is micrometres on a hexagonal scan, whatever unit the file reported its step in', () => {
    // a raster pixel of this scan is 0.5 um (the fixture's spec: 14 x 12 pixels of 0.5 um)
    expect(overviewScale(hexMap, { x: 2, y: 2, units: 'mm' })).toEqual({ unitsPerPixel: 0.5, unitLabel: 'µm' });
    expect(overviewScale(hexMap, { x: 2, y: 2, units: 'nm' })).toEqual({ unitsPerPixel: 0.5, unitLabel: 'µm' });
    expect(overviewScale(hexMap, null)).toEqual({ unitsPerPixel: 0.5, unitLabel: 'µm' });
  });

  it('keeps the file\'s own step and unit on a square scan, and defaults to um without one', () => {
    expect(overviewScale(null, { x: 2.5, y: 2.5, units: 'nm' })).toEqual({ unitsPerPixel: 2.5, unitLabel: 'nm' });
    expect(overviewScale(null, { x: 2.5, y: 2.5 })).toEqual({ unitsPerPixel: 2.5, unitLabel: 'µm' });
    expect(overviewScale(null, null)).toEqual({ unitsPerPixel: null, unitLabel: 'µm' });
  });
});
