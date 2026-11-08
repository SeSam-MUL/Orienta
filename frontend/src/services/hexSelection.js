// frontend/src/services/hexSelection.js
import { cellAtPixel, nearestCell, pixelCentreUm, pointUm } from './hexGeometry';

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const EPS = 1e-9;

const rowShift = (g, r) => (r % 2 === 1 ? g.oddRowShift * g.stepX : 0);

/**
 * The smallest column c of row-offset `shift` whose centre c * stepX + shift is >= xv. The division is
 * clamped first (so a far-away xv cannot give a huge index) and corrected by one column, which is all the
 * rounding of the division can cost; the comparison itself then is the one made on the centre.
 */
function firstColAtLeast(g, xv, shift) {
  let c = clamp(Math.ceil((xv - shift) / g.stepX), -1, g.nCols + 1);
  if (c * g.stepX + shift < xv) c += 1;
  else if ((c - 1) * g.stepX + shift >= xv) c -= 1;
  return c;
}

/**
 * A window of the index grid [r0, r1) x [c0, c1) holding a 0/1 mask that the shapes fill row by row, with the
 * bounding box of what was set. Nothing is held per selected cell.
 */
function makeWindow(g, ya, yb, xa, xb, extra) {
  const s = g.oddRowShift * g.stepX;
  let r0 = Math.max(0, Math.floor(ya / g.stepY) - 1);
  let r1 = Math.min(g.nRows, Math.ceil(yb / g.stepY) + 2);
  let c0 = Math.max(0, Math.floor((xa - Math.max(0, s)) / g.stepX) - 1);
  let c1 = Math.min(g.nCols, Math.ceil((xb - Math.min(0, s)) / g.stepX) + 2);
  for (const [r, c] of extra) {
    r0 = Math.min(r0, r);
    r1 = Math.max(r1, r + 1);
    c0 = Math.min(c0, c);
    c1 = Math.max(c1, c + 1);
  }
  const nr = Math.max(0, r1 - r0);
  const nc = Math.max(0, c1 - c0);
  return {
    r0, r1: r0 + nr, c0, c1: c0 + nc, nc, data: new Uint8Array(nr * nc),
    minR: Infinity, maxR: -Infinity, minC: Infinity, maxC: -Infinity,
  };
}

/** Mark the cells [ca, cb) of row r (clipped to the window). */
function setSpan(w, r, ca, cb) {
  const lo = Math.max(ca, w.c0);
  const hi = Math.min(cb, w.c1);
  if (r < w.r0 || r >= w.r1 || lo >= hi) return;
  const base = (r - w.r0) * w.nc - w.c0;
  w.data.fill(1, base + lo, base + hi);
  if (r < w.minR) w.minR = r;
  if (r > w.maxR) w.maxR = r;
  if (lo < w.minC) w.minC = lo;
  if (hi - 1 > w.maxC) w.maxC = hi - 1;
}

/** The result for a filled window: crop it to the bounding box of what was set, count, drop a full mask. */
function resultOfWindow(w) {
  if (w.minR > w.maxR) return null;
  const rows = w.maxR - w.minR + 1;
  const cols = w.maxC - w.minC + 1;
  let mask = w.data;
  if (rows !== w.r1 - w.r0 || cols !== w.nc) {
    mask = new Uint8Array(rows * cols);
    for (let r = 0; r < rows; r += 1) {
      const from = (w.minR - w.r0 + r) * w.nc + (w.minC - w.c0);
      mask.set(w.data.subarray(from, from + cols), r * cols);
    }
  }
  let n = 0;
  for (let k = 0; k < mask.length; k += 1) n += mask[k];
  return {
    bbox: { row0: w.minR, col0: w.minC, rows, cols },
    mask: n === rows * cols ? null : mask,
    nSelected: n,
  };
}

/**
 * Even-odd fill of a polygon (vertices in micrometres), one scanline per row of the window. For the row's y
 * the edges that straddle it (the rule `(yi > y) !== (yj > y)`) give the x of their crossing; sorted, the
 * cells with x in [x1, x2), [x3, x4), ... are inside, which is a ray cast to the right (a cell is inside when
 * an odd number of crossings lie beyond it) done once per row instead of once per cell.
 */
function fillPolygon(g, w, poly) {
  const xs = [];
  const n = poly.length;
  for (let r = w.r0; r < w.r1; r += 1) {
    const y = r * g.stepY;
    xs.length = 0;
    for (let a = 0, b = n - 1; a < n; b = a, a += 1) {
      const yi = poly[a].y;
      const yj = poly[b].y;
      if ((yi > y) !== (yj > y)) xs.push(poly[a].x + ((y - yi) / (yj - yi)) * (poly[b].x - poly[a].x));
    }
    if (xs.length < 2) continue;
    xs.sort((p, q) => p - q);
    const shift = rowShift(g, r);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      setSpan(w, r, firstColAtLeast(g, xs[k], shift), firstColAtLeast(g, xs[k + 1], shift));
    }
  }
}

/**
 * The cell shown by the pixel of a box (inclusive pixel indices) that lies nearest to the middle of the box,
 * or null when no pixel of the box shows a cell. The middle pixel is tried first; a box that holds no cell
 * centre is thin in one direction, so scanning it is short.
 */
function cellNearBoxMiddle(g, spec, i0, i1, j0, j1) {
  const mid = cellAtPixel(g, spec, Math.floor((i0 + i1) / 2), Math.floor((j0 + j1) / 2));
  if (mid) return mid;
  const mi = (i0 + i1) / 2;
  const mj = (j0 + j1) / 2;
  let best = null;
  let bestD = Infinity;
  for (let i = Math.max(i0, 0); i <= Math.min(i1, spec.height - 1); i += 1) {
    for (let j = Math.max(j0, 0); j <= Math.min(j1, spec.width - 1); j += 1) {
      const d = (i - mi) ** 2 + (j - mj) ** 2;
      if (d >= bestD) continue;
      const c = cellAtPixel(g, spec, i, j);
      if (c) {
        best = c;
        bestD = d;
      }
    }
  }
  return best;
}

/**
 * A drawn selection on the raster, as a crop window on the index grid.
 *
 * A physical rectangle is not an index rectangle on a hexagonal grid (every second row is shifted), so a
 * selection is the set of cells whose CENTRES lie in the drawn shape, returned as the index bounding box of
 * that set plus a mask over the box. `mask` is a Uint8Array of rows*cols, row-major over the box, 1 = selected,
 * or null when every cell of the box is selected (the contract of navSelection.js and of POST /api/ebsd/crop).
 *
 *  - 'rect'    points = [start, end] as dragged; the box spans both pixels inclusively.
 *  - 'ellipse' the ellipse inscribed in that same box.
 *  - 'lasso'   the polygon through the pixel centres of `points` (even-odd), plus the cell under each point.
 *
 * A rectangle or ellipse too small to contain a centre selects the cell shown by the pixel nearest to the middle
 * of its box; one whose pixels all lie outside every hexagon (a transparent corner) selects nothing and
 * returns null. The box is half open: a centre exactly on its low edge is inside, one on its high edge outside.
 *
 * @param {object} g  parsed scan geometry (hexagonal)
 * @param {object} spec  rasterSpec(g)
 * @param {'rect'|'ellipse'|'lasso'} tool
 * @param {{i:number,j:number}[]} points  raster pixels
 * @returns {{bbox:{row0:number,col0:number,rows:number,cols:number}, mask:Uint8Array|null, nSelected:number}|null}
 */
export function selectionFromPixels(g, spec, tool, points) {
  if (!points || points.length === 0) return null;
  if (tool === 'lasso') {
    const poly = points.map((p) => pixelCentreUm(spec, p.i, p.j));
    let xa = Infinity;
    let xb = -Infinity;
    let ya = Infinity;
    let yb = -Infinity;
    for (const p of poly) {
      if (p.x < xa) xa = p.x;
      if (p.x > xb) xb = p.x;
      if (p.y < ya) ya = p.y;
      if (p.y > yb) yb = p.y;
    }
    const under = [];
    for (const p of points) {
      const c = cellAtPixel(g, spec, p.i, p.j);
      if (c) under.push([c.row, c.col]);
    }
    const w = makeWindow(g, ya, yb, xa, xb, under);
    fillPolygon(g, w, poly);
    for (const [r, c] of under) setSpan(w, r, c, c + 1);
    return resultOfWindow(w);
  }

  const first = points[0];
  const last = points[points.length - 1];
  const i0 = Math.min(first.i, last.i);
  const i1 = Math.max(first.i, last.i);
  const j0 = Math.min(first.j, last.j);
  const j1 = Math.max(first.j, last.j);
  const xa = spec.x0 + j0 * spec.stepUm;
  const xb = spec.x0 + (j1 + 1) * spec.stepUm;
  const ya = spec.y0 + i0 * spec.stepUm;
  const yb = spec.y0 + (i1 + 1) * spec.stepUm;
  // The box is [xa, xb) x [ya, yb) on the centres. The pixel edges are computed in floating point and the
  // centres lie exactly on them at factor 4, so both edges of each interval give way by the same EPS-sized
  // amount: a centre on an edge counts as on the edge.
  const tolX = EPS * g.stepX;
  const rA = Math.max(0, Math.ceil(ya / g.stepY - EPS));
  const rB = Math.min(g.nRows, Math.ceil(yb / g.stepY - EPS));
  const ellipse = tool === 'ellipse';
  const cx = (xa + xb) / 2;
  const cy = (ya + yb) / 2;
  const rx = (xb - xa) / 2;
  const ry = (yb - ya) / 2;
  // closed, with the same give as the box edges: a centre on the ellipse is on it, however the squares round
  const inEllipse = (x, y) => ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1 + EPS;
  const w = makeWindow(g, ya, yb, xa, xb, []);
  for (let r = rA; r < rB; r += 1) {
    const shift = rowShift(g, r);
    let ca = firstColAtLeast(g, xa - tolX, shift);
    let cb = firstColAtLeast(g, xb - tolX, shift);
    if (ellipse) {
      // the row's chord of the ellipse, widened by a column each way, then trimmed with the exact test
      const y = r * g.stepY;
      const q = 1 - ((y - cy) / ry) ** 2;
      if (q < -EPS) continue;
      const half = rx * Math.sqrt(Math.max(q, 0));
      ca = Math.max(ca, firstColAtLeast(g, cx - half, shift) - 1);
      cb = Math.min(cb, firstColAtLeast(g, cx + half, shift) + 1);
      ca = Math.max(ca, 0);
      cb = Math.min(cb, g.nCols);
      while (ca < cb && !inEllipse(ca * g.stepX + shift, y)) ca += 1;
      while (cb > ca && !inEllipse((cb - 1) * g.stepX + shift, y)) cb -= 1;
    }
    setSpan(w, r, ca, cb);
  }
  if (w.minR > w.maxR) {
    // No cell centre in the box. A selection is never empty while a pixel of it shows a cell.
    const c = cellNearBoxMiddle(g, spec, i0, i1, j0, j1);
    if (!c) return null;
    return { bbox: { row0: c.row, col0: c.col, rows: 1, cols: 1 }, mask: null, nSelected: 1 };
  }
  return resultOfWindow(w);
}

/**
 * The cells under a straight line between two raster pixels, sampled at `n` equally spaced positions
 * (n >= 2) along the physical segment, each taken as the nearest scan point. A line crossing a corner that no
 * hexagon covers still answers a cell (the nearest), so a profile never has a hole in it.
 * @returns {Array<[number, number]>}  [row, col] per sample
 */
export function lineCells(g, spec, a, b, n) {
  const p = pixelCentreUm(spec, a.i, a.j);
  const q = pixelCentreUm(spec, b.i, b.j);
  // a profile needs two samples at least, whatever n is (NaN, undefined and Infinity fall back to two)
  const count = Number.isFinite(n) ? Math.max(2, Math.floor(n)) : 2;
  const out = [];
  for (let k = 0; k < count; k += 1) {
    const t = k / (count - 1);
    const { row, col } = nearestCell(g, p.x + t * (q.x - p.x), p.y + t * (q.y - p.y));
    out.push([row, col]);
  }
  return out;
}

/**
 * The raster rectangle {x, y, w, h} (pixels) that covers the hexagons of an index box
 * {row0, col0, rows, cols}: the frame of a region run, or of an indexed region.
 */
export function cellBoxToPixelBox(g, spec, box) {
  const { row0, col0, rows, cols } = box;
  let xMin = Infinity;
  let xMax = -Infinity;
  for (let r = row0; r < row0 + rows; r += 1) {
    xMin = Math.min(xMin, pointUm(g, r, col0).x);
    xMax = Math.max(xMax, pointUm(g, r, col0 + cols - 1).x);
  }
  const yMin = row0 * g.stepY - spec.rho;
  const yMax = (row0 + rows - 1) * g.stepY + spec.rho;
  const j0 = clamp(Math.floor((xMin - g.stepX / 2 - spec.x0) / spec.stepUm + EPS), 0, spec.width);
  const j1 = clamp(Math.ceil((xMax + g.stepX / 2 - spec.x0) / spec.stepUm - EPS), 0, spec.width);
  const i0 = clamp(Math.floor((yMin - spec.y0) / spec.stepUm + EPS), 0, spec.height);
  const i1 = clamp(Math.ceil((yMax - spec.y0) / spec.stepUm - EPS), 0, spec.height);
  return { x: j0, y: i0, w: j1 - j0, h: i1 - i0 };
}
