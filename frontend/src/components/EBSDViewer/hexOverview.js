// frontend/src/components/EBSDViewer/hexOverview.js
/**
 * The hexagonal branch of the EBSD viewer's overview, as pure functions.
 *
 * On a hexagonal scan the overview picture is the backend's display raster (GET /api/ebsd/overview with
 * display=raster), not the index grid. The viewer keeps navigating, marking and cropping in (row, col) of the
 * stored grid; these functions are the only places where the picture's pixels and the grid meet. `hexMap` is
 * what makeHexMap() returns. For a square scan the viewer never calls them (its own code is the square path).
 */

/**
 * The pointer at fractions (rx, ry) of the picture.
 * `r`/`c` are the measurement point under it (null in a corner no hexagon covers: nothing to navigate to);
 * `sr`/`sc` are the raster pixel, which is what a selection is drawn in (a drag may start or end in a corner).
 * Null outside the picture.
 */
export function overviewPointer(hexMap, rx, ry) {
  if (!(rx >= 0 && rx <= 1 && ry >= 0 && ry <= 1)) return null;
  const [h, w] = hexMap.shape;
  const i = Math.min(Math.floor(ry * h), h - 1);
  const j = Math.min(Math.floor(rx * w), w - 1);
  const cell = hexMap.cellAt(i, j);
  return { r: cell ? cell.row : null, c: cell ? cell.col : null, rx, ry, sr: i, sc: j };
}

/** Where the crosshair of measurement point (row, col) goes, in percent of the picture. */
export function crosshairPercent(hexMap, row, col) {
  const f = hexMap.fractionOf(row, col);
  return { x: f.fx * 100, y: f.fy * 100 };
}

/**
 * The crop window of the drawn shape: {bbox, mask} in index space (hexSelection.selectionFromPixels), or null.
 * `points` are the viewer's selection points ({r, c} raster pixels); no tool means the Shift+drag rectangle.
 */
export function selectionCrop(hexMap, tool, points) {
  const pixels = (points || []).map((p) => ({ i: p.r, j: p.c }));
  return hexMap.selection(tool || 'rect', pixels);
}

/** The numbers the ROI caption prints: the cells of the crop window on a hexagonal scan, else the drag itself. */
export function roiLabelBox(crop, roi) {
  if (!crop) return roi;
  const { row0, col0, rows, cols } = crop.bbox;
  return { startRow: row0, startCol: col0, endRow: row0 + rows - 1, endCol: col0 + cols - 1 };
}

/**
 * What a picture is fetched for: the fields of a hexagonal scan's geometry (as the backend states them, in
 * `ebsdInfo.scan_geometry`) that decide its raster, joined into one string. A picture carries the key it was
 * fetched under, and is shown only while the page holds the same key. Null for a square scan, a missing
 * geometry, or one that is not an object: a square scan's picture is bound to nothing.
 */
export function rasterKey(scanGeometry) {
  if (!scanGeometry || scanGeometry.kind !== 'hex') return null;
  const { n_rows: rows, n_cols: cols, step_x: sx, step_y: sy, odd_row_shift: shift } = scanGeometry;
  return `hex|${rows}|${cols}|${sx}|${sy}|${shift}`;
}

/**
 * Do the picture the backend drew and the map this page clicks through disagree? True when the page has a
 * hexagonal map and the answer carries no matching raster, or when the answer carries a raster and the page has
 * none (a click would land on the wrong measurement either way).
 */
export function overviewMismatch(hexMap, display) {
  if (hexMap) return !hexMap.matches(display);
  return !!display;
}

/**
 * The scale of the overview picture for the image export: how many units one picture pixel is, and the unit's name.
 * On a hexagonal scan a picture pixel is a fraction of the step and the value comes from the map in micrometres
 * (`hexMap.umPerPx`), so the unit is micrometres whatever unit the file reported its step in; null where the scan has
 * no known steps. A square scan keeps the file's own step and unit.
 */
export function overviewScale(hexMap, stepSize) {
  if (hexMap) return { unitsPerPixel: hexMap.geometry.stepsKnown ? hexMap.umPerPx : null, unitLabel: 'µm' };
  return { unitsPerPixel: stepSize?.x ?? null, unitLabel: stepSize?.units || 'µm' };
}
