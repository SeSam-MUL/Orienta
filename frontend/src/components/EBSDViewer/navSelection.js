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
