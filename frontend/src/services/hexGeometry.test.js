import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseScanGeometry } from './scanGeometry';
import {
  rasterSpec, chooseFactor, cellAtPixel, pixelOfCell, pixelCentreUm, pixelFromFraction,
  fractionOfPixel, nearestCell, pointUm, EDGE_TOLERANCE,
} from './hexGeometry';

// Written by tests/hex_raster_oracle.py from the lattice alone (a plain argmin over every point);
// tests/test_hex_raster.py holds the Python implementation to the same numbers.
const FIXTURE = JSON.parse(
  readFileSync(new URL('../../../tests/fixtures/hex_raster_cases.json', import.meta.url), 'utf8'),
);
const geometryOf = (raw) => parseScanGeometry({ ...raw, steps_known: true });
const A = geometryOf({ kind: 'hex', n_rows: 3, n_cols: 3, step_x: 2.0, step_y: Math.sqrt(3), odd_row_shift: 0.5 });
const CASES = FIXTURE.geometries.map((c) => [c.name, c]);
// The factor of a case is the one the fixture records, not the one the size of the scan would choose.
const specOf = (c) => rasterSpec(geometryOf(c.geometry), c.spec.factor);
const asPair = (cell) => (cell === null ? null : [cell.row, cell.col]);

describe('rasterSpec', () => {
  it('is 14 x 12 pixels of 0.5 um for a 3 x 3 scan with a = 2, by hand', () => {
    // x: -1 .. 6 = 7 um = 14 px; y: -2/3 sqrt(3) .. 2 sqrt(3) + 2/3 sqrt(3) = 5.77 um = 11.55 -> 12 px
    const spec = rasterSpec(A);
    expect([spec.factor, spec.width, spec.height]).toEqual([4, 14, 12]);
    expect(spec.shape).toEqual([12, 14]);
    expect(spec.stepUm).toBe(0.5);
    expect(spec.x0).toBe(-1);
    expect(spec.y0).toBeCloseTo((-2 * Math.sqrt(3)) / 3, 12);
  });

  it('answers null for a square scan and for no geometry', () => {
    expect(rasterSpec(parseScanGeometry({ kind: 'square', n_rows: 3, n_cols: 3, step_x: 1, step_y: 1 }))).toBeNull();
    expect(rasterSpec(null)).toBeNull();
  });

  it('answers null for a hexagonal scan whose steps are unknown', () => {
    expect(rasterSpec(parseScanGeometry({ kind: 'hex', n_rows: 3, n_cols: 3, odd_row_shift: 0.5 }))).toBeNull();
  });

  it('does not add a pixel for the rounding noise of the extent, by hand', () => {
    // a = 0.3 um, 3 columns, two rows, offset +0.5: the width is (3 + 0.5) a = 1.05 um = 14 pixels of
    // a / 4. In doubles 1.05 / 0.075 is 14.000000000000002, which would be 15 without the tolerance.
    const g = geometryOf({ kind: 'hex', n_rows: 2, n_cols: 3, step_x: 0.3, step_y: 0.3 * (Math.sqrt(3) / 2),
      odd_row_shift: 0.5 });
    expect(rasterSpec(g).width).toBe(14);
  });

  it('honours a factor that is asked for', () => {
    const spec = rasterSpec(A, 1);
    expect([spec.factor, spec.stepUm, spec.width, spec.height]).toEqual([1, 2, 4, 3]);
  });

  it.each(CASES)('matches the fixture: %s', (_n, c) => {
    const spec = specOf(c);
    expect([spec.factor, spec.height, spec.width]).toEqual([c.spec.factor, c.spec.height, c.spec.width]);
    expect(spec.shape).toEqual([c.spec.height, c.spec.width]);
    expect(spec.stepUm).toBe(c.spec.step_um);
    expect(spec.x0).toBeCloseTo(c.spec.x0, 12);
    expect(spec.y0).toBeCloseTo(c.spec.y0, 12);
  });

  it('chooses factor 4 for a small scan when none is asked for', () => {
    // the fixture asks for factor 1 on its tie case; left alone, a 3 x 3 scan gets 4
    expect(rasterSpec(G).factor).toBe(4);
  });

  it('falls to the factor chooseFactor gives for a large scan when none is asked for', () => {
    // 1200 x 1000 at a = 0.5 um: factor 4 would need about 17 million pixels, factor 2 about 4.3 million
    const big = geometryOf({ kind: 'hex', n_rows: 1200, n_cols: 1000, step_x: 0.5,
      step_y: 0.5 * (Math.sqrt(3) / 2), odd_row_shift: 0.5 });
    const spec = rasterSpec(big);
    expect(spec.factor).toBe(2);
    expect(spec.height * spec.width).toBeLessThanOrEqual(16_000_000);
  });
});

describe('chooseFactor', () => {
  it.each(FIXTURE.factors.map((f) => [`${f.n_rows} x ${f.n_cols}`, f]))('chooses the same factor as the oracle: %s', (_n, f) => {
    const g = geometryOf({ kind: 'hex', n_rows: f.n_rows, n_cols: f.n_cols, step_x: f.step_x,
      step_y: f.step_x * (Math.sqrt(3) / 2), odd_row_shift: 0.5 });
    expect(chooseFactor(g)).toBe(f.factor);
  });
});

describe('cellAtPixel and pixelOfCell', () => {
  it('finds the cell of a pixel inside a hexagon, by hand', () => {
    // pixel (2, 2): centre (0.25, 0.095) um, 0.27 um from point (0, 0)
    expect(cellAtPixel(A, rasterSpec(A), 2, 2)).toEqual({ row: 0, col: 0 });
  });

  it('answers null where no hexagon covers the pixel, by hand', () => {
    // pixel (0, 0): centre (-0.75, -0.905). dx = 0.75 <= a/2 = 1, but the hexagon is only
    // rho * (1 - 0.75/2) = 0.722 um high there and dy = 0.905.
    expect(cellAtPixel(A, rasterSpec(A), 0, 0)).toBeNull();
  });

  it.each(CASES)('agrees with every sampled pixel of the fixture: %s', (_n, c) => {
    const g = geometryOf(c.geometry);
    const spec = specOf(c);
    for (const p of c.pixels) {
      expect(asPair(cellAtPixel(g, spec, p.i, p.j)), `pixel (${p.i}, ${p.j})`).toEqual(p.cell);
    }
  });

  it.each(CASES)('places the marker of every cell on the pixel the fixture names: %s', (_n, c) => {
    const g = geometryOf(c.geometry);
    const spec = specOf(c);
    for (const cell of c.cells) {
      const { i, j } = pixelOfCell(g, spec, cell.row, cell.col);
      expect([i, j], `cell (${cell.row}, ${cell.col})`).toEqual(cell.pixel);
    }
  });

  // Holds for every case of the fixture. In general it holds at factor 4 only: at factor 2 and 1 the pixel
  // that holds a cell's centre can lie in a neighbouring cell or in none (hex_raster.pixel_of_cell says so);
  // whoever needs the cell of a pixel asks cellAtPixel.
  it.each(CASES)('gives every cell the pixel that holds its centre, and the cell owns it: %s', (_n, c) => {
    const g = geometryOf(c.geometry);
    const spec = specOf(c);
    for (const cell of c.cells) {
      const { i, j } = pixelOfCell(g, spec, cell.row, cell.col);
      expect(asPair(cellAtPixel(g, spec, i, j)), `cell (${cell.row}, ${cell.col})`).toEqual([cell.row, cell.col]);
    }
  });
});

// The same scan as A with the odd rows offset the other way (what a crop starting on an odd row carries).
const G = geometryOf({ kind: 'hex', n_rows: 3, n_cols: 3, step_x: 2.0, step_y: Math.sqrt(3), odd_row_shift: -0.5 });

describe('ties and borders at factor 1', () => {
  // a = 2 um (exact in binary), odd rows at -1 um, factor 1: pixels of 2 um, and every pixel centre meets
  // the lattice exactly. Rows are b = sqrt(3) um apart, the hexagon is rho = 2b/3 = 1.155 um tall at dx = 0.
  // x: -2 .. 5 um -> 4 pixels, centres -1, 1, 3, 5. y: -1.155 .. 4.619 um -> 3 pixels, centres
  // -0.155, 1.845, 3.845. Row 0 points sit at x = 0, 2, 4; row 1 at x = -1, 1, 3; row 2 at 0, 2, 4.
  it('gives a tie to the lower column and the border to the cell, by hand', () => {
    const spec = rasterSpec(G, 1);
    expect([spec.height, spec.width, spec.stepUm, spec.x0]).toEqual([3, 4, 2, -2]);
    const want = [
      // centre x = -1: on the border dx = a/2 of the first column, which keeps it (row 1: on the point).
      // centre x = 1 and 3 in rows 0 and 2: halfway between two points, a tie, to the lower column.
      // row 1, x = 5: the nearest point is (2, 2), whose hexagon is 1.619 um away in y: no cell.
      [[0, 0], [0, 0], [0, 1], [0, 2]],
      [[1, 0], [1, 1], [1, 2], null],
      [[2, 0], [2, 0], [2, 1], [2, 2]],
    ];
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 4; j += 1) {
        expect(asPair(cellAtPixel(G, spec, i, j)), `pixel (${i}, ${j})`).toEqual(want[i][j]);
      }
    }
  });

  it('gives the cell a pixel centre on the top or bottom tip of its hexagon, by hand', () => {
    // One row, a = 2, b = 3, so rho = 2b/3 = 2. Raster written out by hand: pixels of 2 um from x0 = -1,
    // y0 = 1. Pixel (0, 0) has its centre at x = -1 + 0.5 * 2 = 0, y = 1 + 0.5 * 2 = 2: directly above the
    // point (0, 0) (dx = 0) at dy = 2 = rho. The allowed height there is rho * (1 - 0) = 2, so dy <= 2 holds.
    const one = geometryOf({ kind: 'hex', n_rows: 1, n_cols: 2, step_x: 2, step_y: 3, odd_row_shift: 0.5 });
    const spec = { x0: -1, y0: 1, stepUm: 2, rho: 2 };
    expect(pixelCentreUm(spec, 0, 0)).toEqual({ x: 0, y: 2 });
    expect(cellAtPixel(one, spec, 0, 0)).toEqual({ row: 0, col: 0 });
  });
});

describe('a step with rounding noise at factor 1', () => {
  // A pixel centre on the border between two cells used to fall into none when the step is not exact in binary
  // (1.7, say): the nearest point was the one whose border the noise had pushed 1e-16 away. The fixture holds,
  // written by the oracle from the union of the hexagons, which pixels belong to the scan; the cell chosen on a
  // border is not part of it (either of the two is right), the existence of one is.
  it.each(FIXTURE.coverage.map((c) => [c.name, c]))('leaves no pixel of the scan without a cell and covers no other: %s', (_n, c) => {
    const g = geometryOf(c.geometry);
    const spec = specOf(c);
    expect([spec.height, spec.width]).toEqual([c.covered.length, c.covered[0].length]);
    const got = [];
    for (let i = 0; i < spec.height; i += 1) {
      let line = '';
      for (let j = 0; j < spec.width; j += 1) line += cellAtPixel(g, spec, i, j) === null ? '.' : '#';
      got.push(line);
    }
    expect(got).toEqual(c.covered);
  });

  it('gives the pixel between cells (1, 1) and (1, 2) of a 6 x 6 scan of step 1.7 to one of them, by hand', () => {
    // x0 = -0.85 um, so pixel column j has its centre at 1.7 * j; pixel (1, 2) is at x = 3.4 um, halfway
    // between the points 2.55 and 4.25 um of the odd row 1, and 0.097 um from that row in y.
    const g = geometryOf({ kind: 'hex', n_rows: 6, n_cols: 6, step_x: 1.7, step_y: (1.7 * Math.sqrt(3)) / 2, odd_row_shift: 0.5 });
    const cell = cellAtPixel(g, rasterSpec(g, 1), 1, 2);
    expect(cell).not.toBeNull();
    expect(cell.row).toBe(1);
    expect([1, 2]).toContain(cell.col);
  });
});

describe('the border of a hexagon', () => {
  // One row of two points, a = 2 um at x = 0 and 2, rows b = sqrt(3) um apart, rho = 2b/3 = 2/sqrt(3) um. The
  // vertical border of cell (0, 0) is at x = -1, the slanted one at dx = 0.5 at dy = rho * (1 - 0.5 / 2) =
  // sqrt(3)/2 um. A raster of one pixel of 1 um whose centre is (x, y).
  const one = geometryOf({ kind: 'hex', n_rows: 1, n_cols: 2, step_x: 2, step_y: Math.sqrt(3), odd_row_shift: 0.5 });
  const R = Math.sqrt(3) / 2;
  it.each([
    [-1 - 1e-13, 0, true], [-1 - 1e-6, 0, false],
    [0.5, R + 1e-13, true], [0.5, R + 1e-6, false], [0.5, R - 1e-6, true],
  ])('holds a hair beyond it and not a millionth of a micrometre: (%s, %s)', (x, y, inside) => {
    const spec = { x0: x - 0.5, y0: y - 0.5, stepUm: 1, rho: (2 * Math.sqrt(3)) / 3 };
    expect(cellAtPixel(one, spec, 0, 0)).toEqual(inside ? { row: 0, col: 0 } : null);
  });
});

describe('the tolerance of the border', () => {
  it('is the number the fixture carries, the one hex_raster.py is held to as well', () => {
    expect(EDGE_TOLERANCE).toBe(FIXTURE.edge_tolerance);
  });

  // The offsets are fractions of the step, so one fixture row means the same at every step. A tolerance in
  // micrometres instead of in steps fails at a step of 0.0017 um (too wide: a pixel beyond the border is taken
  // in) and at 1700 um (too narrow: offsets of a few steps' millionths are refused).
  it.each(FIXTURE.borders.map((c) => [c.name, c]))('holds a pixel centre at the fixture offset from the border: %s', (_n, c) => {
    const g = geometryOf(c.geometry);
    const a = c.geometry.step_x;
    const spec = { x0: c.x - a / 2, y0: c.y - a / 2, stepUm: a, rho: (2 * g.stepY) / 3 };
    expect(cellAtPixel(g, spec, 0, 0)).toEqual(c.inside ? { row: 0, col: 0 } : null);
  });
});

describe('small helpers', () => {
  it('places an odd row half a step to the right, by hand', () => {
    expect(pointUm(A, 1, 2)).toEqual({ x: 5, y: Math.sqrt(3) });
    expect(pointUm(A, 2, 2)).toEqual({ x: 4, y: 2 * Math.sqrt(3) });
  });

  it('places an odd row half a step to the left when the offset is reversed, by hand', () => {
    expect(pointUm(G, 1, 2)).toEqual({ x: 3, y: Math.sqrt(3) });
  });

  it('breaks a tie between two columns to the lower one', () => {
    const one = geometryOf({ kind: 'hex', n_rows: 1, n_cols: 2, step_x: 2, step_y: Math.sqrt(3), odd_row_shift: 0.5 });
    expect(nearestCell(one, 1, 0)).toMatchObject({ row: 0, col: 0 });
  });

  it('breaks a tie between two rows to the lower one, by hand', () => {
    // A: a = 2, b = sqrt(3), odd rows at x = 1, 3, 5. Query (0.5, sqrt(3)/2): from point (0, 0) of row 0 it is
    // (0.5, sqrt(3)/2) away, from point (1, 0) of row 1 it is (-0.5, -sqrt(3)/2). Both squared distances are
    // 0.25 + 0.75 = 1, to the last bit (the offsets differ only in sign), so the lower row wins.
    expect(nearestCell(A, 0.5, Math.sqrt(3) / 2)).toMatchObject({ row: 0, col: 0 });
  });

  it('gives the physical centre of a pixel', () => {
    const spec = rasterSpec(A);
    expect(pixelCentreUm(spec, 2, 2)).toEqual({ x: -1 + 2.5 * 0.5, y: spec.y0 + 2.5 * 0.5 });
  });

  it('maps a fraction of the displayed image to a pixel and back, and refuses outside', () => {
    const spec = rasterSpec(A);                       // 12 rows, 14 columns
    expect(pixelFromFraction(spec, 0.5, 0.5)).toEqual({ i: 6, j: 7 });
    expect(pixelFromFraction(spec, 0, 0)).toEqual({ i: 0, j: 0 });
    expect(pixelFromFraction(spec, 0.999999, 0.999999)).toEqual({ i: 11, j: 13 });
    for (const [fx, fy] of [[1, 0.5], [0.5, 1], [-0.01, 0.5], [0.5, -0.01], [NaN, 0.5]]) {
      expect(pixelFromFraction(spec, fx, fy)).toBeNull();
    }
    expect(fractionOfPixel(spec, 6, 7)).toEqual({ fx: 7.5 / 14, fy: 6.5 / 12 });
  });
});
