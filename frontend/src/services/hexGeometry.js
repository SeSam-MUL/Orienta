// frontend/src/services/hexGeometry.js
/**
 * The display raster of a hexagonal scan and the mapping between its pixels and the stored index grid.
 *
 * The backend draws every map layer of a hexagonal scan on a square raster (backend/api/services/
 * hex_raster.py): pixels of stepX / factor, each taking the value of the nearest scan point and drawn only
 * inside that point's hexagon. This module is the same rule in JavaScript, so that a pixel under the pointer
 * can be turned into the (row, col) of the measurement it shows, and a (row, col) into the pixel to mark.
 * It follows hex_raster.py operation for operation (same expressions in the same order, so IEEE doubles
 * agree with numpy's); tests/fixtures/hex_raster_cases.json, written by an independent oracle, holds the
 * two together. The raster is display only: nothing here (or on the backend) treats a pixel as data.
 *
 * Notation: (row, col) is the stored index grid, (i, j) a raster pixel (row, column), (x, y) micrometres on
 * the sample with x along the rows. a = stepX, b = stepY, rho = 2b/3 the circumradius of a hexagon.
 * `geometry` is the object parseScanGeometry() returns.
 *
 * Ties. A position exactly between two scan points belongs to the lower row and, within a row, to the lower
 * column. A pixel centre on the border of a hexagon belongs to the hexagon (the comparisons are <=). Both
 * matter at factor 1 with a step that is exact in binary, where pixel centres fall on such positions.
 */

export const FACTORS = [4, 2, 1];
export const MAX_RASTER_PIXELS = 16_000_000;
/**
 * Part of a step a pixel centre may lie beyond the border of a hexagon and still belong to it: the border is
 * only known to the last bits of the positions, and without this a pixel on the border between two cells can
 * fall into none at factor 1. The same number as EDGE_TOLERANCE in hex_raster.py.
 */
export const EDGE_TOLERANCE = 1e-9;

const isHex = (g) => !!g && g.kind === 'hex';
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function extents(g, factor) {
  const a = g.stepX;
  const b = g.stepY;
  const s = g.oddRowShift;
  const offsets = g.nRows >= 2 ? [0, s * a] : [0];
  const xLo = Math.min(...offsets) - a / 2;
  const xHi = (g.nCols - 1) * a + Math.max(...offsets) + a / 2;
  const rho = (2 * b) / 3;
  const yLo = -rho;
  const yHi = (g.nRows - 1) * b + rho;
  const d = a / factor;
  const height = Math.ceil((yHi - yLo) / d - 1e-9);
  const width = Math.ceil((xHi - xLo) / d - 1e-9);
  return { factor, height, width, shape: [height, width], stepUm: d, x0: xLo, y0: yLo, rho };
}

/** 4, or 2, or 1: the first whose raster has at most MAX_RASTER_PIXELS pixels. */
export function chooseFactor(g) {
  for (const f of FACTORS) {
    const e = extents(g, f);
    if (e.height * e.width <= MAX_RASTER_PIXELS) return f;
  }
  return FACTORS[FACTORS.length - 1];
}

/**
 * The raster of a hexagonal scan, or null for a square one, none, or one whose steps are unknown (the
 * extent of the raster cannot be computed from them).
 */
export function rasterSpec(g, factor) {
  if (!isHex(g) || !(g.stepX > 0) || !(g.stepY > 0)) return null;
  return extents(g, factor || chooseFactor(g));
}

/** Position of scan point (row, col) on the sample, micrometres. */
export function pointUm(g, row, col) {
  const shift = row % 2 === 1 ? g.oddRowShift * g.stepX : 0;
  return { x: col * g.stepX + shift, y: row * g.stepY };
}

/**
 * The scan point nearest to (x, y): the port of ScanGeometry.nearest. Rows next to the two that bracket y are
 * candidates too, because past a corner both brackets clip to the same edge row. A tie goes to the lower row
 * (strict '<' over ascending rows) and, within a row, to the lower column (ceil of ... - 0.5).
 */
export function nearestCell(g, x, y) {
  const sx = g.stepX;
  const sy = g.stepY;
  const lastR = g.nRows - 1;
  const lastC = g.nCols - 1;
  const rLo = clamp(Math.floor(y / sy), 0, lastR);
  const candidates = [clamp(rLo - 1, 0, lastR), rLo, clamp(rLo + 1, 0, lastR), clamp(rLo + 2, 0, lastR)];
  let best = null;
  for (const r of candidates) {
    const shift = r % 2 === 1 ? g.oddRowShift * sx : 0;
    const c = clamp(Math.ceil((x - shift) / sx - 0.5), 0, lastC);
    const ex = x - (c * sx + shift);
    const ey = y - r * sy;
    const d2 = ex * ex + ey * ey;
    if (best === null || d2 < best.d2) best = { row: r, col: c, d2 };
  }
  return best;
}

/** Physical centre of raster pixel (i, j). */
export function pixelCentreUm(spec, i, j) {
  return { x: spec.x0 + (j + 0.5) * spec.stepUm, y: spec.y0 + (i + 0.5) * spec.stepUm };
}

/** The cell shown at pixel (i, j), or null where no hexagon covers it (the transparent corners). */
export function cellAtPixel(g, spec, i, j) {
  const { x, y } = pixelCentreUm(spec, i, j);
  const { row, col } = nearestCell(g, x, y);
  const p = pointUm(g, row, col);
  const dx = Math.abs(x - p.x);
  const dy = Math.abs(y - p.y);
  const tol = EDGE_TOLERANCE * g.stepX;
  const inside = dx <= g.stepX / 2 + tol && dy <= spec.rho * (1 - dx / g.stepX) + tol;
  return inside ? { row, col } : null;
}

/**
 * The pixel that holds the centre of cell (row, col), for whoever needs a whole pixel. At factor 4 that
 * pixel belongs to the cell; at factor 2 and 1 it can lie in a neighbouring cell or in none, so whoever
 * needs the cell a pixel belongs to asks cellAtPixel.
 */
export function pixelOfCell(g, spec, row, col) {
  const { x, y } = pointUm(g, row, col);
  return { i: Math.floor((y - spec.y0) / spec.stepUm), j: Math.floor((x - spec.x0) / spec.stepUm) };
}

/**
 * Where the exact centre of cell (row, col) lies, as fractions of the displayed image. Not the centre of the
 * pixel that holds it: at factor 2 and 1 that can be up to half a step away from the cell.
 */
export function fractionOfCell(g, spec, row, col) {
  const { x, y } = pointUm(g, row, col);
  return { fx: (x - spec.x0) / (spec.width * spec.stepUm), fy: (y - spec.y0) / (spec.height * spec.stepUm) };
}

/** The pixel at a position given as fractions (0..1) of the displayed image, or null outside it. */
export function pixelFromFraction(spec, fx, fy) {
  if (!(fx >= 0 && fx < 1 && fy >= 0 && fy < 1)) return null;
  return { i: Math.floor(fy * spec.height), j: Math.floor(fx * spec.width) };
}

/** Where the centre of pixel (i, j) lies, as fractions of the displayed image. */
export function fractionOfPixel(spec, i, j) {
  return { fx: (j + 0.5) / spec.width, fy: (i + 0.5) / spec.height };
}
