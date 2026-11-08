import { describe, it, expect } from 'vitest';
import { parseScanGeometry } from './scanGeometry';
import { makeHexMap } from './hexMap';

const hex = parseScanGeometry({ kind: 'hex', n_rows: 3, n_cols: 3, step_x: 2, step_y: Math.sqrt(3), odd_row_shift: 0.5 });

describe('makeHexMap', () => {
  it('is null for a square scan, a missing geometry and nothing', () => {
    expect(makeHexMap(parseScanGeometry({ kind: 'square', n_rows: 3, n_cols: 3, step_x: 1, step_y: 1 }))).toBeNull();
    expect(makeHexMap(null)).toBeNull();
    expect(makeHexMap(undefined)).toBeNull();
  });

  it('holds the raster and answers in both directions', () => {
    const m = makeHexMap(hex);
    expect(m.shape).toEqual([12, 14]);
    expect(m.umPerPx).toBe(0.5);
    expect(m.cellAt(2, 2)).toEqual({ row: 0, col: 0 });
    expect(m.cellAt(0, 0)).toBeNull();
    expect(m.pixelOf(0, 0)).toEqual({ i: 2, j: 2 });
    expect(m.cellAtFraction(2.5 / 14, 2.5 / 12)).toEqual({ row: 0, col: 0 });
    expect(m.cellAtFraction(1, 0.5)).toBeNull();
    // cell (0, 0) lies at (0, 0) um; the raster starts at (-1, -2 sqrt(3) / 3) um and is 14 x 12 pixels of 0.5 um
    const f = m.fractionOf(0, 0);
    expect(f.fx).toBeCloseTo(1 / 7, 12);
    expect(f.fy).toBeCloseTo(Math.sqrt(3) / 9, 12);
    // cell (1, 1) lies at (3, sqrt 3) um
    const g = m.fractionOf(1, 1);
    expect(g.fx).toBeCloseTo(4 / 7, 12);
    expect(g.fy).toBeCloseTo((5 * Math.sqrt(3)) / 18, 12);
  });

  describe('fractionOf is the exact centre of the cell, at every factor', () => {
    // a = 1 um, b = sqrt(3) / 2: the cell grid size decides the factor (4 up to 16 M pixels, then 2, then 1)
    const square = (n) => makeHexMap(parseScanGeometry(
      { kind: 'hex', n_rows: n, n_cols: n, step_x: 1, step_y: Math.sqrt(3) / 2, odd_row_shift: 0.5 }));

    it('factor 4: the centre of cell (0, 0) is a whole pixel from the origin', () => {
      const m = square(3);
      expect(m.spec.factor).toBe(4);
      // x0 = -0.5 and pixels of 0.25 um: the centre of cell (0, 0) lies 2 pixels in
      expect(m.fractionOf(0, 0).fx * m.shape[1]).toBeCloseTo(2, 9);
    });

    it('factor 2: a centre that is a whole pixel in, not the middle of one', () => {
      const m = square(1300);
      expect(m.spec.factor).toBe(2);
      // x0 = -0.5, pixels of 0.5 um, 2601 of them: even rows put cell c at 2c + 1 pixels, odd rows at 2c + 2
      expect(m.shape[1]).toBe(2601);
      expect(m.fractionOf(0, 0).fx).toBeCloseTo(1 / 2601, 12);
      expect(m.fractionOf(0, 7).fx).toBeCloseTo(15 / 2601, 12);
      expect(m.fractionOf(1, 7).fx).toBeCloseTo(16 / 2601, 12);
      // rows: (r * sqrt(3) / 2 + sqrt(3) / 3) um from the top, in pixels of 0.5 um
      expect(m.fractionOf(3, 0).fy * m.shape[0]).toBeCloseTo(2 * (3 * Math.sqrt(3) / 2 + Math.sqrt(3) / 3), 9);
    });

    it('factor 1: the centre of an even-row cell is the middle of a pixel, of an odd-row cell its corner', () => {
      const m = square(2200);
      expect(m.spec.factor).toBe(1);
      // x0 = -0.5, pixels of 1 um: cell c of an even row is at c + 0.5 pixels, of an odd row at c + 1
      expect(m.fractionOf(0, 5).fx * m.shape[1]).toBeCloseTo(5.5, 9);
      expect(m.fractionOf(1, 5).fx * m.shape[1]).toBeCloseTo(6, 9);
    });

    const sample = (n) => {
      const rows = [0, 1, 2, 3, 4, 5, 6, 7, Math.floor(n / 2), Math.floor(n / 2) + 1, n - 2, n - 1];
      const cols = [0, 1, 2, 3, 4, 5, Math.floor(n / 3), Math.floor(n / 2), n - 3, n - 2, n - 1];
      return rows.flatMap((row) => cols.map((col) => [row, col])).filter(([row, col]) => row < n && col < n);
    };

    it.each([[3, 4], [1300, 2]])('cellAtFraction(fractionOf(cell)) is the cell, grid of %i (factor %i)', (n, factor) => {
      const m = square(n);
      expect(m.spec.factor).toBe(factor);
      for (const [row, col] of sample(n)) {
        const f = m.fractionOf(row, col);
        expect(m.cellAtFraction(f.fx, f.fy), `cell ${row},${col}`).toEqual({ row, col });
      }
    });

    it('factor 1: the pixel that holds a centre may show a neighbour (a pixel is wider than the hexagon is tall)', () => {
      // Rows are sqrt(3) / 2 = 0.87 um apart and pixels 1 um, so some pixels hold the centre of one cell while the
      // nearest centre to the middle of that pixel is a neighbour's: cell (3, 0) at (0.5, 2.598) um lies in pixel
      // (3, 1) whose middle (1.0, 2.923) um is nearer to cell (4, 1) at (1.0, 3.464) um. Also true before fractionOf
      // gave the exact centre.
      const m = square(2200);
      expect(m.spec.factor).toBe(1);
      const f = m.fractionOf(3, 0);
      expect(m.cellAtFraction(f.fx, f.fy)).toEqual({ row: 4, col: 1 });
      // and never further than one row and one column away (at the rim the pixel can lie in no hexagon at all)
      for (const [row, col] of sample(2200)) {
        const g = m.fractionOf(row, col);
        const got = m.cellAtFraction(g.fx, g.fy);
        if (!got) continue;
        expect(Math.abs(got.row - row), `cell ${row},${col}`).toBeLessThanOrEqual(1);
        expect(Math.abs(got.col - col), `cell ${row},${col}`).toBeLessThanOrEqual(1);
      }
    });
  });

  // What the backend stamps for this scan (a = 2, b = sqrt 3, odd rows +0.5 of a step): 12 x 14 pixels of
  // 0.5 um from x0 = -(a / 2) = -1 and y0 = -(2 b / 3) = -1.1547 um.
  const STAMP = { kind: 'hex_raster', factor: 4, shape: [12, 14], step_um: 0.5, x0: -1, y0: -1.1547005383792515 };

  it('compares itself with what the backend says it drew', () => {
    const m = makeHexMap(hex);
    expect(m.matches(STAMP)).toBe(true);
    expect(m.matches({ ...STAMP, shape: [12, 15] })).toBe(false);
    expect(m.matches({ ...STAMP, factor: 2 })).toBe(false);
    expect(m.matches({ ...STAMP, step_um: 1 })).toBe(false);
    expect(m.matches({ ...STAMP, y0: -1 })).toBe(false);
    expect(m.matches(null)).toBe(false);
  });

  it('tells a raster that starts a half step to the side from this one, though its shape is the same', () => {
    // The same window cut from the other parity of rows: odd rows -0.5 of a step instead of +0.5. The extent in x
    // runs from min(0, -1) - 1 = -2 to 2 * 2 + 0 + 1 = 5, so 14 pixels again, now from x0 = -2.
    const m = makeHexMap(hex);
    expect(m.matches({ ...STAMP, x0: -2 })).toBe(false);
    const opposite = makeHexMap(parseScanGeometry(
      { kind: 'hex', n_rows: 3, n_cols: 3, step_x: 2, step_y: Math.sqrt(3), odd_row_shift: -0.5 }));
    expect(opposite.shape).toEqual([12, 14]);
    expect(opposite.matches({ ...STAMP, x0: -2 })).toBe(true);
    expect(opposite.matches(STAMP)).toBe(false);
  });

  it('does not take an answer that leaves the origin out for a match', () => {
    const m = makeHexMap(hex);
    expect(m.matches({ shape: [12, 14], factor: 4 })).toBe(false);
    expect(m.matches({ ...STAMP, x0: undefined })).toBe(false);
    expect(m.matches({ ...STAMP, x0: '-1' })).toBe(false);
  });

  it('hands selections, lines and boxes to the same rules, by hand', () => {
    // 3 x 3 cells, raster 12 x 14 pixels of 0.5 um from (-1, -1.155) um; centres: row 0 at x = 0, 2, 4.
    const m = makeHexMap(hex);
    const all = m.selection('rect', [{ i: 0, j: 0 }, { i: 11, j: 13 }]);
    expect(all.bbox).toEqual({ row0: 0, col0: 0, rows: 3, cols: 3 });
    expect(all.mask).toBeNull();
    expect(all.nSelected).toBe(9);
    // pixel centres (0.25, 0.095) and (4.25, 0.095), three samples at x = 0.25, 2.25, 4.25 on row 0
    expect(m.line({ i: 2, j: 2 }, { i: 2, j: 10 }, 3)).toEqual([[0, 0], [0, 1], [0, 2]]);
    // x from -1 to 5 + 1 (row 1 reaches x = 5), y from -1.155 to 2 * 1.732 + 1.155 = 4.619
    expect(m.boxToPixels({ row0: 0, col0: 0, rows: 3, cols: 3 })).toEqual({ x: 0, y: 0, w: 14, h: 12 });
  });

  it('a line without a sample count still answers its two end cells', () => {
    const m = makeHexMap(hex);
    expect(m.line({ i: 2, j: 2 }, { i: 2, j: 10 })).toEqual([[0, 0], [0, 2]]);
  });

  it('is frozen: a page cannot edit the raster it shares', () => {
    expect(Object.isFrozen(makeHexMap(hex))).toBe(true);
  });

  it('is null for a hexagonal scan whose steps are unknown', () => {
    expect(makeHexMap({ ...hex, stepX: null, stepY: null })).toBeNull();
  });
});
