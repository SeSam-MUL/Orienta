/**
 * Turning a drawn selection on the overview into a crop window.
 *
 * Pure geometry — no React, no canvas — so the masks can be checked against
 * hand-computed cases without a browser.
 *
 * Coordinates are DISPLAY-GRID pixels: `r` is a scan row, `c` a scan column,
 * both integers, both inclusive when they come from a drag.
 *
 * A mask is a flat Uint8Array of length rows*cols, row-major over the
 * bounding box, 1 = selected. `null` means "the whole box" — the rectangle
 * case — and every consumer must treat null that way rather than building an
 * all-ones array for nothing. That is also the contract of the backend's
 * `POST /api/ebsd/crop`, which reads a null mask as "no mask, take the box".
 */

/**
 * Bounding box (inclusive) of the drawn points, or null if nothing was drawn.
 *
 * A faithful hull of the points it is given — it does NOT clamp to the scan.
 * A window outside the scan is refused by the backend with an HTTP 400 rather
 * than silently shrunk, and bounding a drag to the grid belongs to the caller
 * that knows the grid.
 *
 * @param {{r:number,c:number}[]|null} points
 * @returns {{row0:number,col0:number,rows:number,cols:number}|null}
 */
export function boundsOf(points) {
  if (!points || points.length === 0) return null;
  let minR = Infinity; let maxR = -Infinity;
  let minC = Infinity; let maxC = -Infinity;
  for (const p of points) {
    if (p.r < minR) minR = p.r;
    if (p.r > maxR) maxR = p.r;
    if (p.c < minC) minC = p.c;
    if (p.c > maxC) maxC = p.c;
  }
  return {
    row0: minR,
    col0: minC,
    rows: maxR - minR + 1,
    cols: maxC - minC + 1,
  };
}

/**
 * A rectangle selects its whole bounding box, so it carries no mask.
 *
 * Returning null rather than an all-ones Uint8Array keeps the common case free
 * of a per-pixel array that says nothing.
 *
 * @returns {null}
 */
export function rectMask() {
  return null;
}

/**
 * How many pixels the selection actually covers.
 *
 * @param {Uint8Array|null} mask  null = the whole box
 * @param {{rows:number,cols:number}|null} bbox
 * @returns {number}
 */
export function countSelected(mask, bbox) {
  if (!bbox) return 0;
  if (!mask) return bbox.rows * bbox.cols;
  let n = 0;
  for (let i = 0; i < mask.length; i += 1) if (mask[i]) n += 1;
  return n;
}

/**
 * Bytes the cropped patterns will occupy in memory. 0 when unknown.
 *
 * @param {number} rows
 * @param {number} cols
 * @param {[number,number]|null} patternShape  detector [h, w]
 * @param {number} bytesPerPixel  sample depth, e.g. 1 for uint8
 * @returns {number}
 */
export function estimateBytes(rows, cols, patternShape, bytesPerPixel) {
  if (!patternShape || patternShape.length < 2) return 0;
  const [h, w] = patternShape;
  if (!h || !w) return 0;
  return rows * cols * h * w * (bytesPerPixel || 1);
}

const UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];

/**
 * Human-readable size — one decimal above bytes, as people say it aloud.
 *
 * @param {number} n
 * @returns {string}
 */
export function formatBytes(n) {
  if (!n || n <= 0) return '0 B';
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return unit === 0 ? `${Math.round(value)} B` : `${value.toFixed(1)} ${UNITS[unit]}`;
}

/**
 * Bytes per detector sample, from the dtype string the backend reports.
 *
 * `/api/ebsd/datasets` carries `dtype` per dataset ("uint8", "uint16",
 * "float32"), so the size estimate does not have to assume 8-bit patterns —
 * a uint16 detector would otherwise be reported at half its real size.
 * Unknown or unparseable input falls back to 1, which is what the estimate
 * assumed before dtype was available.
 *
 * @param {string|null|undefined} dtype
 * @returns {number}
 */
export function bytesPerSample(dtype) {
  const bits = /(\d+)$/.exec(String(dtype ?? ''));
  if (!bits) return 1;
  const n = Number(bits[1]) / 8;
  return Number.isFinite(n) && n >= 1 ? n : 1;
}

/**
 * The ellipse inscribed in the bounding box.
 *
 * Tested at pixel CENTRES: a scan pixel is a cell, not a point, so the cell at
 * (r, c) counts as inside when its middle is inside the ellipse. That is what
 * makes the axis pixels of an odd-sized box land inside and the corners not.
 *
 * The radii are half the box in CELLS (rows/2), while the centres only span
 * rows-1. That mismatch is deliberate: it is the ellipse inscribed in the box
 * as an area, so the outermost cell ON an axis still has its centre inside
 * (in a 5x5 box, (0,2) sits at 0.8 of the radius) while a corner does not.
 * Using (rows-1)/2 instead would put the axis extremes exactly on the rim and
 * leave them at the mercy of floating-point rounding.
 *
 * A null bbox means nothing was drawn — `boundsOf` returns null for an empty
 * path — and yields an empty mask, so this agrees with `countSelected`, which
 * answers 0 for the same input instead of throwing.
 *
 * @param {{row0:number,col0:number,rows:number,cols:number}|null} bbox
 * @returns {Uint8Array}
 */
export function ellipseMask(bbox) {
  if (!bbox) return new Uint8Array(0);
  const { rows, cols } = bbox;
  const mask = new Uint8Array(rows * cols);
  const cy = (rows - 1) / 2;
  const cx = (cols - 1) / 2;
  const ry = rows / 2;
  const rx = cols / 2;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const dy = (r - cy) / ry;
      const dx = (c - cx) / rx;
      if (dy * dy + dx * dx <= 1) mask[r * cols + c] = 1;
    }
  }
  return mask;
}

/**
 * Bresenham walk from (r0,c0) to (r1,c1), marking every cell the line touches.
 *
 * Endpoints are rounded: the walk steps by whole cells and stops on equality,
 * so a fractional endpoint would never be reached and the loop would not
 * terminate. Drag coordinates are already whole scan pixels; the rounding is
 * insurance. Non-finite input bails for the same reason — Task 10 derives grid
 * coordinates by dividing by a container size, and a container that is still
 * 0 wide on its first render would otherwise hang the tab rather than draw
 * nothing.
 *
 * Cells outside the box are dropped rather than wrapped: a caller may hand us
 * a box narrower than the hull of the points, and a wrap would paint the
 * opposite edge of the map.
 */
function markLine(mask, bbox, r0, c0, r1, c1) {
  if (!Number.isFinite(r0) || !Number.isFinite(c0)
      || !Number.isFinite(r1) || !Number.isFinite(c1)) return;
  const { row0, col0, rows, cols } = bbox;
  let r = Math.round(r0);
  let c = Math.round(c0);
  const rEnd = Math.round(r1);
  const cEnd = Math.round(c1);
  const dr = Math.abs(rEnd - r);
  const dc = Math.abs(cEnd - c);
  const sr = r < rEnd ? 1 : -1;
  const sc = c < cEnd ? 1 : -1;
  let err = dc - dr;
  for (;;) {
    const mr = r - row0;
    const mc = c - col0;
    if (mr >= 0 && mr < rows && mc >= 0 && mc < cols) mask[mr * cols + mc] = 1;
    if (r === rEnd && c === cEnd) break;
    const e2 = 2 * err;
    if (e2 > -dr) { err -= dr; c += sc; }
    if (e2 < dc) { err += dc; r += sr; }
  }
}

/**
 * The freehand polygon: the even-odd fill, plus the traced outline.
 *
 * The path is closed automatically — a user who lifts the mouse near the start
 * means a closed shape. Pixel centres decide the fill, for the same reason as
 * the ellipse.
 *
 * WHY THE SECOND PASS. The even-odd fill alone is inset by one on the bottom
 * and right, and not occasionally — always. At the last row of the box every
 * vertex has `r <= maxR`, so `(yi > y) !== (yj > y)` is false for every edge:
 * zero crossings, the whole row unselected. At the last column every `xCross`
 * is a convex combination of two vertex columns and therefore `<= maxC`, so
 * `x < xCross` is false: the whole column unselected. Since `boundsOf` makes
 * maxR and maxC the last row and column, EVERY lasso mask would be one
 * smaller than its own bounding box on two sides — a traced 4x4 square would
 * return 9 of its 16 pixels, and a lasso one row tall would return nothing at
 * all and earn the user an HTTP 400 for a drag that looked fine. The loss is
 * proportionally worst on small ROIs, which are the whole point of cropping.
 *
 * So the polygon edges are rasterised and OR-ed in. That is not a lopsided
 * patch: on the top and left the outline adds nothing, because those cells are
 * already inside the fill; on the bottom and right it adds exactly what the
 * half-open rule drops. The result is symmetric — every cell the user traced
 * is selected, whichever side of the shape it lies on.
 *
 * The one-character alternative does NOT work: `x <= xCross` makes the corner
 * (0,0) of the triangle (0,0)-(0,4)-(4,0) count both the vertical edge and the
 * hypotenuse — an even count, hence outside — losing a corner the user drew.
 *
 * @param {{r:number,c:number}[]|null} points
 * @param {{row0:number,col0:number,rows:number,cols:number}|null} bbox
 * @returns {Uint8Array}
 */
export function lassoMask(points, bbox) {
  if (!bbox) return new Uint8Array(0);
  const { row0, col0, rows, cols } = bbox;
  const mask = new Uint8Array(rows * cols);
  if (!points || points.length < 3) return mask;

  const n = points.length;
  // The interior, by the even-odd rule.
  for (let r = 0; r < rows; r += 1) {
    const y = row0 + r;
    for (let c = 0; c < cols; c += 1) {
      const x = col0 + c;
      let inside = false;
      for (let i = 0, j = n - 1; i < n; j = i, i += 1) {
        const yi = points[i].r; const xi = points[i].c;
        const yj = points[j].r; const xj = points[j].c;
        // Half-open crossing test: counts an edge once even when a vertex
        // lands exactly on the scan line.
        if ((yi > y) !== (yj > y)) {
          const xCross = xi + ((y - yi) / (yj - yi)) * (xj - xi);
          if (x < xCross) inside = !inside;
        }
      }
      if (inside) mask[r * cols + c] = 1;
    }
  }

  // The traced outline, closing edge included. See WHY THE SECOND PASS above.
  for (let i = 0, j = n - 1; i < n; j = i, i += 1) {
    markLine(mask, bbox, points[j].r, points[j].c, points[i].r, points[i].c);
  }
  return mask;
}
