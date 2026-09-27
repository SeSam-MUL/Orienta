/**
 * One line of the export dialog's "written" list.
 *
 * A table is described by its rows; a picture by its size in pixels.
 * "phase_map.png — 462 rows" was the M5 tester's reading of a 21 x 22 px map:
 * the backend's `rows` on a PNG is the pixel count, and the dialog printed it
 * as rows of something.
 */
function num(v) {
  if (v === null || v === undefined || v === '') return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

export function fileLine(f, t, fmt = String) {
  const w = num(f?.width);
  const h = num(f?.height);
  if (w !== null && h !== null) {
    return t('export.doneImage', { name: f.name, width: fmt(w), height: fmt(h) });
  }
  return t('export.doneRows', { name: f.name, rows: fmt(f?.rows) });
}
