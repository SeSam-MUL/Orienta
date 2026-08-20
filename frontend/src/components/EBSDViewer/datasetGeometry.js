/**
 * Reading the navigation grid off an entry of `GET /api/ebsd/datasets`.
 *
 * That endpoint reports `navigation_shape` and `signal_shape` straight from the
 * live signal's axes_manager, which makes it the authority on how big a dataset
 * is — more so than a crop REQUEST, which only says what was asked for.
 *
 * Both shapes arrive in hyperspy order, (x, y) — columns first. Every consumer
 * in the frontend wants (rows, cols), and the backend's own `/info` does the
 * same flip (`pat_h = sig_shape[1]`, `pat_w = sig_shape[0]`). Doing it in one
 * tested place is what keeps a cropped dataset from being read on its parent's
 * grid.
 */

/**
 * @param {{navigation_shape?: number[], signal_shape?: number[], dtype?: string}|null} entry
 * @returns {{gridShape:[number,number], patternShape:[number,number]|null,
 *            patternCount:number, navigationShape:[number,number],
 *            dtype:string|null}|null}
 *          null when the entry carries no usable navigation shape — the caller
 *          then keeps the grid it has rather than writing a guess.
 */
export function datasetGeometry(entry) {
  const nav = entry?.navigation_shape;
  if (!Array.isArray(nav) || nav.length < 2) return null;
  const cols = Number(nav[0]);
  const rows = Number(nav[1]);
  if (!Number.isFinite(rows) || !Number.isFinite(cols) || rows <= 0 || cols <= 0) return null;

  const sig = entry.signal_shape;
  const patternShape = (Array.isArray(sig) && sig.length >= 2
    && Number.isFinite(Number(sig[0])) && Number.isFinite(Number(sig[1])))
    ? [Number(sig[1]), Number(sig[0])]
    : null;

  return {
    gridShape: [rows, cols],
    patternShape,
    patternCount: rows * cols,
    navigationShape: [cols, rows],
    dtype: entry.dtype || null,
  };
}
