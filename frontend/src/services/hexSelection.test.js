import { describe, it, expect } from 'vitest';
import { parseScanGeometry } from './scanGeometry';
import { rasterSpec, cellAtPixel } from './hexGeometry';
import { selectionFromPixels, lineCells, cellBoxToPixelBox } from './hexSelection';

// 4 x 4 cells, a = 2 um, odd rows +1 um, rows sqrt(3) um apart. Raster: 18 x 16 pixels of 0.5 um,
// x0 = -1, y0 = -2/3 sqrt(3) = -1.155. Cell centres: row 0 at x = 0, 2, 4, 6 (y = 0); row 1 at x = 1, 3, 5, 7
// (y = 1.732); row 2 at x = 0, 2, 4, 6 (y = 3.464); row 3 at x = 1, 3, 5, 7 (y = 5.196).
const g = parseScanGeometry({ kind: 'hex', n_rows: 4, n_cols: 4, step_x: 2, step_y: Math.sqrt(3), odd_row_shift: 0.5 });
const spec = rasterSpec(g);
const px = (i, j) => ({ i, j });
const asList = (r) => r && { bbox: r.bbox, mask: r.mask && Array.from(r.mask), n: r.nSelected };

describe('selectionFromPixels', () => {
  it('selects the cells whose centres lie in a rectangle, by hand', () => {
    // pixels (3,3)..(9,9): x in [0.5, 4.0), y in [0.345, 3.845). Row 1 (y 1.73): x = 1, 3. Row 2 (y 3.46): x = 2
    // only (0 < 0.5, 4 is not < 4.0). Row 0 (y 0) is above 0.345, row 3 below 3.845.
    expect(asList(selectionFromPixels(g, spec, 'rect', [px(3, 3), px(9, 9)])))
      .toEqual({ bbox: { row0: 1, col0: 0, rows: 2, cols: 2 }, mask: [1, 1, 0, 1], n: 3 });
  });

  it('does not care which corner the drag started in', () => {
    const a = selectionFromPixels(g, spec, 'rect', [px(3, 3), px(9, 9)]);
    const b = selectionFromPixels(g, spec, 'rect', [px(9, 9), px(3, 3)]);
    expect(asList(b)).toEqual(asList(a));
  });

  it('returns no mask when every cell of the box is selected', () => {
    const all = selectionFromPixels(g, spec, 'rect', [px(0, 0), px(100, 100)]);
    expect(all.bbox).toEqual({ row0: 0, col0: 0, rows: 4, cols: 4 });
    expect(all.mask).toBeNull();
    expect(all.nSelected).toBe(16);
  });

  it('a rectangle smaller than a cell selects the cell under it', () => {
    // pixel (5,5): centre (1.75, 1.595) um, inside the hexagon of cell (1, 0) at (1, 1.732); no centre is in the box
    expect(asList(selectionFromPixels(g, spec, 'rect', [px(5, 5), px(5, 5)])))
      .toEqual({ bbox: { row0: 1, col0: 0, rows: 1, cols: 1 }, mask: null, n: 1 });
  });

  it('a rectangle wholly outside every hexagon selects nothing', () => {
    expect(selectionFromPixels(g, spec, 'rect', [px(0, 0), px(0, 0)])).toBeNull();
  });

  it('selects the cells inside the inscribed ellipse', () => {
    // pixels (0,0)..(11,13): box x [-1, 6), y [-1.155, 4.845): centre (2.5, 1.845), radii (3.5, 3.0).
    // The nine cells of rows 0-2, columns 0-2 all lie inside (the farthest, (0,0), at 0.89 of the radius).
    const r = selectionFromPixels(g, spec, 'ellipse', [px(0, 0), px(11, 13)]);
    expect(r.nSelected).toBe(9);
    expect(r.bbox).toEqual({ row0: 0, col0: 0, rows: 3, cols: 3 });
  });

  it('leaves out a corner cell the ellipse does not reach, where the rectangle would take it', () => {
    // pixels (0,0)..(15,17): box x [-1, 8), y [-1.155, 6.845): centre (3.5, 2.845), radii (4.5, 4.0).
    // Cell (0,0) at (0, 0): (3.5/4.5)^2 + (2.845/4.0)^2 = 0.605 + 0.506 = 1.11 > 1. The next farthest,
    // (3,3) at (7, 5.196): 0.605 + (2.351/4.0)^2 = 0.605 + 0.345 = 0.95 <= 1. All other cells are nearer.
    const r = selectionFromPixels(g, spec, 'ellipse', [px(0, 0), px(15, 17)]);
    expect(r.bbox).toEqual({ row0: 0, col0: 0, rows: 4, cols: 4 });
    expect(r.nSelected).toBe(15);
    expect(r.mask[0]).toBe(0);
    expect(Array.from(r.mask).slice(1).every((v) => v === 1)).toBe(true);
  });

  it('selects the cells inside a lasso, plus the cell under each of its points', () => {
    // rectangle of pixel centres (0.75..4.75) x (0.595..3.595): centres inside are (1,0) (1,1) (2,1) (2,2);
    // the four corner points add (0,0) (0,2) (2,2) (2,0).
    const r = selectionFromPixels(g, spec, 'lasso', [px(3, 3), px(3, 11), px(9, 11), px(9, 3)]);
    expect(asList(r)).toEqual({ bbox: { row0: 0, col0: 0, rows: 3, cols: 3 }, mask: [1, 0, 1, 1, 1, 0, 1, 1, 1], n: 7 });
  });

  it('a lasso of one point selects its cell, and none outside the scan', () => {
    expect(asList(selectionFromPixels(g, spec, 'lasso', [px(5, 5)])))
      .toEqual({ bbox: { row0: 1, col0: 0, rows: 1, cols: 1 }, mask: null, n: 1 });
    expect(selectionFromPixels(g, spec, 'lasso', [px(0, 0)])).toBeNull();
  });

  it('answers null for no points', () => {
    expect(selectionFromPixels(g, spec, 'rect', [])).toBeNull();
    expect(selectionFromPixels(g, spec, 'lasso', null)).toBeNull();
  });

  it('keeps the mask the length of its box', () => {
    const r = selectionFromPixels(g, spec, 'lasso', [px(3, 3), px(3, 11), px(9, 11), px(9, 3)]);
    expect(r.mask.length).toBe(r.bbox.rows * r.bbox.cols);
  });
});

// ---------------------------------------------------------------------------------------------------------
// A reference written from the grid positions, not from the module. Scans here have stepY = 0.75 * stepX, so
// rho = stepX / 2 and every cell centre has a closed form in raster pixels (u along x, v along y):
//   offset +0.5: x0 = -a/2, u = f (c + 0.5 + [odd] 0.5)      offset -0.5: x0 = -a, u = f (c + 1 - [odd] 0.5)
//   v = f (0.75 r + 0.5)
// in both. U = 4u and V = 4v are integers, so a centre on a pixel edge is an integer comparison, whatever
// rounding the module's own floating-point positions carry.
// ---------------------------------------------------------------------------------------------------------
const hexScan = (a, o, f, nRows = 6, nCols = 6) => {
  const grid = parseScanGeometry({ kind: 'hex', n_rows: nRows, n_cols: nCols, step_x: a, step_y: 0.75 * a, odd_row_shift: o });
  return { g: grid, spec: rasterSpec(grid, f), a, o, f, nRows, nCols };
};
const centre4 = (s, r, c) => {
  const odd = r % 2;
  return { U: s.f * (s.o > 0 ? 4 * c + 2 + 2 * odd : 4 * c + 4 - 2 * odd), V: s.f * (3 * r + 2) };
};
const eachCell = (s, fn) => {
  const out = [];
  for (let r = 0; r < s.nRows; r += 1) for (let c = 0; c < s.nCols; c += 1) if (fn(centre4(s, r, c), r, c)) out.push([r, c]);
  return out;
};
const pack = (cells) => {
  if (cells.length === 0) return null;
  const r0 = Math.min(...cells.map((x) => x[0]));
  const c0 = Math.min(...cells.map((x) => x[1]));
  const rows = Math.max(...cells.map((x) => x[0])) - r0 + 1;
  const cols = Math.max(...cells.map((x) => x[1])) - c0 + 1;
  const mask = new Array(rows * cols).fill(0);
  for (const [r, c] of cells) mask[(r - r0) * cols + (c - c0)] = 1;
  return { bbox: { row0: r0, col0: c0, rows, cols }, mask: cells.length === rows * cols ? null : mask, n: cells.length };
};
let seed = 20261101;
const rand = (n) => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return Math.floor((seed / 2147483648) * n);
};
const STEPS = [2, 0.37, 1.37, 0.0137, 0.1];

describe('selectionFromPixels against a reference from the grid positions', () => {
  for (const f of [1, 2, 4]) {
    for (const o of [0.5, -0.5]) {
      it(`rectangles and ellipses, factor ${f}, offset ${o}, five steps`, () => {
        for (const a of STEPS) {
          const s = hexScan(a, o, f);
          for (let t = 0; t < 120; t += 1) {
            const ia = rand(s.spec.height);
            const ib = rand(s.spec.height);
            const ja = rand(s.spec.width);
            const jb = rand(s.spec.width);
            const [i0, i1, j0, j1] = [Math.min(ia, ib), Math.max(ia, ib), Math.min(ja, jb), Math.max(ja, jb)];
            const inBox = ({ U, V }) => U >= 4 * j0 && U < 4 * (j1 + 1) && V >= 4 * i0 && V < 4 * (i1 + 1);
            // ellipse inscribed in the box, all in integers: ((U - CU) / RU)^2 + ((V - CV) / RV)^2 <= 1
            const CU = 2 * (j0 + j1 + 1);
            const RU = 2 * (j1 + 1 - j0);
            const CV = 2 * (i0 + i1 + 1);
            const RV = 2 * (i1 + 1 - i0);
            // the ellipse is closed, the box half-open: a centre on the ellipse counts, one on the far box edge does not
            const inEllipse = (c) => inBox(c) && (c.U - CU) ** 2 * RV ** 2 + (c.V - CV) ** 2 * RU ** 2 <= RU ** 2 * RV ** 2;
            const pts = t % 2 ? [px(i0, j0), px(i1, j1)] : [px(i1, j0), px(i0, j1)];
            for (const [tool, want] of [['rect', pack(eachCell(s, inBox))], ['ellipse', pack(eachCell(s, inEllipse))]]) {
              const got = asList(selectionFromPixels(s.g, s.spec, tool, pts));
              const label = `${tool} a=${a} box i ${i0}..${i1} j ${j0}..${j1}`;
              if (want) expect(got, label).toEqual(want);
              else expect(got === null || got.n === 1, label).toBe(true);
            }
          }
        }
      });
    }
  }

  it('a rectangle two cells wide holds two cells in every row, whatever the rounding of the step', () => {
    // 8 pixels at factor 4 are two steps; a half-open interval of two steps holds exactly two centres of a row
    for (const o of [0.5, -0.5]) {
      for (const a of STEPS) {
        const s = hexScan(a, o, 4, 6, 8);
        for (let j0 = 4; j0 <= 20; j0 += 1) {
          const r = selectionFromPixels(s.g, s.spec, 'rect', [px(0, j0), px(s.spec.height - 1, j0 + 7)]);
          expect(r.bbox.rows, `a=${a} o=${o} j0=${j0}`).toBe(6);
          for (let row = 0; row < 6; row += 1) {
            let n = 0;
            for (let c = 0; c < r.bbox.cols; c += 1) n += r.mask ? r.mask[row * r.bbox.cols + c] : 1;
            expect(n, `a=${a} o=${o} j0=${j0} row ${row}`).toBe(2);
          }
        }
      }
    }
  });

  it('lassos, factor 4, both offsets: even-odd cells plus the cell under each point', () => {
    let checked = 0;
    for (const o of [0.5, -0.5]) {
      for (const a of [2, 0.37, 1.37]) {
        const s = hexScan(a, o, 4);
        for (let t = 0; t < 150; t += 1) {
          const pts = [];
          for (let k = 0, n = 3 + rand(7); k < n; k += 1) pts.push(px(rand(s.spec.height), rand(s.spec.width)));
          // A ray to the right from the centre, in integers (everything doubled: a vertex is (2j + 1, 2i + 1), a
          // centre (U / 2, V / 2)). An edge through a centre makes that cell ambiguous; such a polygon is skipped.
          let tie = false;
          const inside = ({ U, V }) => {
            let odd = false;
            for (let k = 0; k < pts.length; k += 1) {
              const p = { x: 2 * pts[k].j + 1, y: 2 * pts[k].i + 1 };
              const q = { x: 2 * pts[(k + 1) % pts.length].j + 1, y: 2 * pts[(k + 1) % pts.length].i + 1 };
              if ((p.y > V / 2) !== (q.y > V / 2)) {
                // sign of (x of the edge at V / 2) - U / 2, times |dy|
                const num = (p.x - U / 2) * (q.y - p.y) + (V / 2 - p.y) * (q.x - p.x);
                if (num === 0) tie = true;
                if (num * Math.sign(q.y - p.y) > 0) odd = !odd;
              }
            }
            return odd;
          };
          const under = new Set(pts.map((p) => cellAtPixel(s.g, s.spec, p.i, p.j)).filter(Boolean).map((c) => `${c.row},${c.col}`));
          const want = pack(eachCell(s, (ctr, r, c) => inside(ctr) || under.has(`${r},${c}`)));
          if (tie) continue;
          checked += 1;
          expect(asList(selectionFromPixels(s.g, s.spec, 'lasso', pts)), `a=${a} o=${o} ${JSON.stringify(pts)}`).toEqual(want);
        }
      }
    }
    expect(checked).toBeGreaterThan(600);
  });

  it('a self-intersecting lasso selects the two triangles of a bow tie and the cells under its corners, by hand', () => {
    // 4 x 4 cells of 2 um at factor 4, offset +0.5: centres at pixel (u, v) = row 0: u 2 6 10 14, v 2; row 1: u 4 8 12 16,
    // v 5; row 2: v 8 like row 0; row 3: v 11 like row 1. Bow tie through the pixel centres (3.5, 3.5), (11.5, 12.5),
    // (11.5, 3.5), (3.5, 12.5): its diagonals cross at (7.5, 8); the left triangle has the edge u = 3.5, the right
    // one u = 11.5. A diagonal moves 1.125 v per u. Left triangle: at u = 4 v in (4.06, 11.94) holds (1,0) v=5 and
    // (3,0) v=11; at u = 6, v in (6.31, 9.69) holds (2,1) v=8. Right triangle: at u = 10, v in (5.19, 9.81) holds (2,2).
    // Not inside: (1,1) u=8 (the pinch there is only v 7.44..8.56), (3,1), row 0 (v = 2 is above both). The corner
    // pixels lie in the hexagons of (1,0), (3,2), (1,2), (3,0), which adds (3,2) and (1,2).
    const s = hexScan(2, 0.5, 4, 4, 4);
    const r = selectionFromPixels(s.g, s.spec, 'lasso', [px(3, 3), px(12, 11), px(3, 11), px(12, 3)]);
    expect(asList(r)).toEqual({ bbox: { row0: 1, col0: 0, rows: 3, cols: 3 }, mask: [1, 0, 1, 0, 1, 1, 1, 0, 1], n: 6 });
  });

  it('one dragged rectangle on 1000 x 1200 cells does not overflow the stack (a million cells)', () => {
    const s = hexScan(1, 0.5, 4, 1000, 1200);
    const r = selectionFromPixels(s.g, s.spec, 'rect', [px(0, 0), px(s.spec.height - 1, s.spec.width - 1)]);
    expect(r.nSelected).toBe(1_200_000);
    expect(r.bbox).toEqual({ row0: 0, col0: 0, rows: 1000, cols: 1200 });
    expect(r.mask).toBeNull();
  });

  it('an ellipse and a lasso over 1000 x 1200 cells do not overflow the stack either', () => {
    const s = hexScan(1, 0.5, 4, 1000, 1200);
    const h = s.spec.height - 1;
    const w = s.spec.width - 1;
    const e = selectionFromPixels(s.g, s.spec, 'ellipse', [px(0, 0), px(h, w)]);
    // the inscribed ellipse covers pi/4 = 0.785 of the box
    expect(e.nSelected).toBeGreaterThan(0.77 * 1_200_000);
    expect(e.nSelected).toBeLessThan(0.80 * 1_200_000);
    expect(e.mask.length).toBe(e.bbox.rows * e.bbox.cols);
    // the pixel centres of the four corners enclose every cell centre (u from 2 to 4800, v from 2 to 2999 of 4802 x 3002)
    const l = selectionFromPixels(s.g, s.spec, 'lasso', [px(0, 0), px(0, w), px(h, w), px(h, 0)]);
    expect(l.nSelected).toBe(1_200_000);
    expect(l.mask).toBeNull();
  });
});

describe('selectionFromPixels, a selection without a cell centre', () => {
  it('a one-pixel-wide strip between two columns of centres selects the cell nearest its middle', () => {
    // pixel column 3 is x in [0.5, 1.0): the centres of the 4 x 4 grid are at pixel u = 2, 4, 6, ... (even), so
    // none lies in [3, 4). The first point (0, 3) is a transparent corner (centre (0.75, -0.905) um, beyond the
    // hexagon edge at |dy| <= 1.155 (1 - 0.75/2) = 0.72); the middle pixel (7, 3) is at (0.75, 2.595) um and
    // lies in the hexagon of cell (1, 0) at (1, 1.732): dx 0.25, dy 0.863 <= 1.155 (1 - 0.125) = 1.01.
    const want = { bbox: { row0: 1, col0: 0, rows: 1, cols: 1 }, mask: null, n: 1 };
    expect(asList(selectionFromPixels(g, spec, 'rect', [px(0, 3), px(15, 3)]))).toEqual(want);
    expect(asList(selectionFromPixels(g, spec, 'rect', [px(15, 3), px(0, 3)]))).toEqual(want);
    expect(asList(selectionFromPixels(g, spec, 'ellipse', [px(0, 3), px(15, 3)]))).toEqual(want);
  });

  it('takes the nearest pixel that shows a cell when the middle pixel shows none', () => {
    // pixels (0,0) and (0,1) (x -1..0, y -1.155..-0.655 um) hold no centre. The middle pixel is (0, 0), a
    // transparent corner; pixel (0, 1) is at (-0.25, -0.905), inside the hexagon of cell (0, 0)
    // (dx 0.25, dy 0.905 <= 1.155 (1 - 0.125) = 1.01).
    expect(asList(selectionFromPixels(g, spec, 'rect', [px(0, 0), px(0, 1)])))
      .toEqual({ bbox: { row0: 0, col0: 0, rows: 1, cols: 1 }, mask: null, n: 1 });
  });

  it('an ellipse wholly in a transparent corner selects nothing', () => {
    expect(selectionFromPixels(g, spec, 'ellipse', [px(0, 0), px(0, 0)])).toBeNull();
  });
});

describe('lineCells', () => {
  it('answers two samples when the count is not a number', () => {
    // endpoints of the vertical line: cells (0, 0) and (2, 0), as in the three-sample line above
    for (const n of [NaN, undefined, Infinity, null, 'x']) {
      expect(lineCells(g, spec, px(3, 3), px(9, 3), n), String(n)).toEqual([[0, 0], [2, 0]]);
    }
  });

  it('takes the nearest scan point at equally spaced positions, by hand', () => {
    // horizontal line at y = 0.595 from x = 0.75 to 4.75, five samples at x = 0.75, 1.75, 2.75, 3.75, 4.75.
    // Row 0 (y = 0, x = 0, 2, 4, 6) is the nearest row; its nearest columns are 0, 1 (x=2), 1, 2 (x=4), 2.
    expect(lineCells(g, spec, px(3, 3), px(3, 11), 5)).toEqual([[0, 0], [0, 1], [0, 1], [0, 2], [0, 2]]);
  });

  it('follows a vertical line down the rows', () => {
    expect(lineCells(g, spec, px(3, 3), px(9, 3), 3)).toEqual([[0, 0], [1, 0], [2, 0]]);
  });

  it('always answers at least two samples', () => {
    expect(lineCells(g, spec, px(3, 3), px(9, 3), 0)).toHaveLength(2);
  });
});

describe('cellBoxToPixelBox', () => {
  it('covers the hexagons of an index box, by hand', () => {
    // box rows 1-2, cols 0-1: cell edges x from -1 (cell (2,0) at x = 0) to 4 (cell (1,1) at x = 3); y from
    // 1*1.732 - 1.155 = 0.577 to 2*1.732 + 1.155 = 4.619. In pixels of 0.5 um from (-1, -1.155): x 0..10,
    // y floor(3.46) = 3 .. ceil(11.55) = 12.
    expect(cellBoxToPixelBox(g, spec, { row0: 1, col0: 0, rows: 2, cols: 2 })).toEqual({ x: 0, y: 3, w: 10, h: 9 });
  });

  it('covers the whole raster for the whole scan', () => {
    expect(cellBoxToPixelBox(g, spec, { row0: 0, col0: 0, rows: 4, cols: 4 })).toEqual({ x: 0, y: 0, w: 18, h: 16 });
  });
});
