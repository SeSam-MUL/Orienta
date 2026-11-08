/**
 * The scan geometry the backend reports, as the frontend reads it.
 *
 * `GET /api/ebsd/load`, `/info`, `/datasets`, `/metadata` and `/api/indexing/results` all carry
 * the same object (`scan_geometry`, from `backend/api/services/scan_geometry.py`): the kind of
 * lattice (`square` or `hex`), the index-space rectangle (`n_rows` x `n_cols`), the two steps in
 * micrometres and, for a hexagonal scan, how far every second row is offset (in steps).
 *
 * Parsing lives here and nowhere else, so that the store, the dataset list and the result list
 * cannot disagree about what a field means. `null` always means "the backend did not say": it is
 * not a square scan.
 */

/** A grid size: a finite positive integer given as a number or a numeric string, else null. */
function gridCount(value) {
  if (typeof value !== 'number' && typeof value !== 'string') return null;
  const n = Number(value); // '' and ' ' give 0, which is not positive
  return Number.isInteger(n) && n > 0 ? n : null;
}

/** A step in micrometres: a finite positive number (or numeric string), else null. */
function positiveStep(value) {
  if (typeof value !== 'number' && typeof value !== 'string') return null;
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/**
 * @param {object|null|undefined} raw  the backend's `scan_geometry`
 * @returns {{kind:'square'|'hex', nRows:number, nCols:number, gridShape:[number,number],
 *            stepX:number|null, stepY:number|null, stepsKnown:boolean, oddRowShift:number,
 *            shiftConfirmed:boolean}|null}
 */
export function parseScanGeometry(raw) {
  if (!raw || typeof raw !== 'object') return null;
  if (raw.kind !== 'square' && raw.kind !== 'hex') return null;
  const nRows = gridCount(raw.n_rows);
  const nCols = gridCount(raw.n_cols);
  if (nRows === null || nCols === null) return null;
  return {
    kind: raw.kind,
    nRows,
    nCols,
    gridShape: [nRows, nCols],
    stepX: positiveStep(raw.step_x),
    stepY: positiveStep(raw.step_y),
    stepsKnown: raw.steps_known !== false,
    oddRowShift: Number(raw.odd_row_shift) || 0,
    shiftConfirmed: raw.shift_confirmed !== false,
  };
}

/** True for a hexagonal scan. A missing geometry is not one. */
export const isHexScan = (geometry) => !!geometry && geometry.kind === 'hex';
