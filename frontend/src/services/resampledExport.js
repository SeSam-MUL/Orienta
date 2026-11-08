/**
 * What a result export says about a hexagonal scan resampled to a square grid.
 *
 * `POST /api/indexing/export` with `ctf` or `ang_square` answers with the file and the header
 * `X-Orienta-Resampled`: ASCII JSON with the step of the square grid (`step_um`), its size (`rows`,
 * `cols`), how many measured points there were (`source_points`), how many cells were written
 * (`points`), how many measured points no cell took (`unused`) and whether the direction of the row
 * offset is known (`shift_confirmed`). The pages word it in the language of the reader.
 */

/** The name each format is downloaded as when the browser chooses the file name. */
export const EXPORT_FILE_NAMES = {
  h5: 'indexing_result.h5',
  ang: 'indexing_result.ang',
  ang_square: 'indexing_result_square.ang',
  ctf: 'indexing_result.ctf',
};

const HEADER = 'x-orienta-resampled';

const headerOf = (response) => {
  const headers = response?.headers;
  if (!headers) return null;
  if (typeof headers.get === 'function') return headers.get(HEADER) ?? null;
  return headers[HEADER] ?? headers['X-Orienta-Resampled'] ?? null;
};

const count = (value) => (Number.isInteger(value) && value > 0 ? value : null);
const optional = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);

/**
 * @param {object|undefined} response  an axios response of an export
 * @returns {{stepUm:number|null, rows:number, cols:number, sourcePoints:number|null,
 *            points:number|null, unused:number|null, shiftConfirmed:boolean}|null}
 *          null when the export was not resampled, or the header cannot be read
 */
export function readResampled(response) {
  const raw = headerOf(response);
  if (typeof raw !== 'string' || !raw) return null;
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  const rows = count(data?.rows);
  const cols = count(data?.cols);
  if (rows === null || cols === null) return null;
  return {
    stepUm: optional(data.step_um),
    rows,
    cols,
    sourcePoints: optional(data.source_points),
    points: optional(data.points),
    unused: optional(data.unused),
    shiftConfirmed: data.shift_confirmed !== false,
  };
}
