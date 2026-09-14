/**
 * A rectangle dragged upwards is the same rectangle.
 *
 * Two endpoints take one: /api/eds/region-quantify and
 * /api/eds/phase-map/assign-region. Drawing bottom-right to top-left sends
 * row_start > row_end and they disagree about what that means — the first
 * answers 400 with the names of fields the user never typed (logged
 * 2026-09-10 13:15:51), the second slices an empty array and reports 0 pixels
 * painted, which reads as "the button does nothing". /api/phasemap/region-stats
 * already normalises server-side, so the same drag behaved differently
 * depending on which map it was drawn on.
 *
 * Ordering it here, where the request is built, covers every caller: the drag
 * hooks (which happen to order their own output) and the four Region Average
 * number fields, which are typed by hand and never passed through one.
 */

// A coordinate this function is willing to reason about. An empty field or a
// word is NOT one: `Number('') || 0` and `Number('abc') || 0` both read as
// pixel 0, which turns "nothing typed here" into a corner of the scan and
// averages a region nobody asked for. Those go to the backend untouched, where
// they are a 422 naming the field.
const pixelOrNull = (v) => {
  if (typeof v === 'number') return Number.isFinite(v) ? Math.floor(v) : null;
  // A field holds a string. Everything else — null, undefined, an object —
  // is an absent coordinate, whatever `Number()` makes of it (`Number(null)`
  // is 0, which is exactly the guess this avoids).
  if (typeof v !== 'string' || v.trim() === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? Math.floor(n) : null;
};

/**
 * Order a rectangle so start <= end on both axes.
 * Coordinates are pixel indices, so they are coerced to integers.
 *
 * A rectangle with a coordinate that is not a number comes back exactly as it
 * arrived: there is nothing to order, and guessing a value here would hide the
 * one thing the user needs to be told.
 */
export function normalizeRect({ rowStart, rowEnd, colStart, colEnd }) {
  const r0 = pixelOrNull(rowStart);
  const r1 = pixelOrNull(rowEnd);
  const c0 = pixelOrNull(colStart);
  const c1 = pixelOrNull(colEnd);
  if (r0 === null || r1 === null || c0 === null || c1 === null) {
    return { rowStart, rowEnd, colStart, colEnd };
  }
  return {
    rowStart: Math.min(r0, r1),
    rowEnd: Math.max(r0, r1),
    colStart: Math.min(c0, c1),
    colEnd: Math.max(c0, c1),
  };
}
