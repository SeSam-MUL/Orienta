// frontend/src/components/common/scanPickerGeometry.js
/**
 * What the two scan pickers (Crystal Hint single pixel, the single-pixel phase test) need to know about their
 * picture, for a square scan and for a hexagonal one.
 *
 * Both pickers draw the scan overview in a box and let the user click a measurement point. On a square scan the
 * picture is the index grid (one pixel per point); on a hexagonal scan it is the backend's display raster
 * (GET /api/ebsd/overview?display=raster), and `hexMap` (makeHexMap) maps it to the points. With `hexMap = null`
 * every function here is the arithmetic the pickers had before.
 */
import { overviewMismatch } from '../EBSDViewer/hexOverview';

/** Width over height of the picture. */
export function pickerAspect(hexMap, nRows, nCols) {
  return hexMap ? hexMap.shape[1] / hexMap.shape[0] : nCols / Math.max(nRows, 1);
}

/** Width of the frame both pickers draw around their box (CSS px). The picture lies inside it. */
export const PICKER_BORDER = 1;

/**
 * The size of the area the picture and the crosshair are drawn in, for a picker box of w x h (border-box, frame of
 * PICKER_BORDER). A hexagonal scan puts about six raster pixels on a screen pixel, so there the click and the
 * crosshair use the area the picture really covers; a square scan keeps the box size, as before.
 */
export function pickerInner(hexMap, w, h) {
  return hexMap ? { w: w - 2 * PICKER_BORDER, h: h - 2 * PICKER_BORDER } : { w, h };
}

/**
 * The rectangle (screen px) a click is measured against: for a square scan the box as `getBoundingClientRect` gives
 * it, as before; for a hexagonal scan the area inside the frame, which is where the picture lies.
 */
export function pickerRect(el, hexMap) {
  const box = el.getBoundingClientRect();
  if (!hexMap) return box;
  return { left: box.left + el.clientLeft, top: box.top + el.clientTop, width: el.clientWidth, height: el.clientHeight };
}

/**
 * The measurement point under fractions (fx, fy) of the picture, as [row, col], or null where no point is: outside
 * the picture or, on a hexagonal scan, in a corner no hexagon covers. A square scan clamps to the grid, as before.
 */
export function pickAt(hexMap, nRows, nCols, fx, fy) {
  if (hexMap) {
    const cell = hexMap.cellAtFraction(fx, fy);
    return cell ? [cell.row, cell.col] : null;
  }
  const c = Math.max(0, Math.min(nCols - 1, Math.floor(fx * nCols)));
  const r = Math.max(0, Math.min(nRows - 1, Math.floor(fy * nRows)));
  return [r, c];
}

/** Where the crosshair of point `pixelIndex` (row * nCols + col) sits, as fractions {fx, fy} of the picture. */
export function markerAt(hexMap, nRows, nCols, pixelIndex) {
  const { row, col } = rowColOf(nCols, pixelIndex);
  if (hexMap) return hexMap.fractionOf(row, col);
  return { fx: (col + 0.5) / nCols, fy: (row + 0.5) / nRows };
}

/** The (row, col) a flat index stands for, for the caption under the picker. */
export function rowColOf(nCols, pixelIndex) {
  const cols = Math.max(nCols, 1);
  return { row: Math.floor(pixelIndex / cols), col: pixelIndex % cols };
}

/**
 * The picture a picker may show, or null (the picker then shows its "loading" text).
 *
 * `entry` is what a fetch left behind: `{ image, display, hexMap }`, where `hexMap` is the map the picker held when
 * it asked. A picture belongs to the map it was fetched for. When the picker has moved on to another one (another
 * measurement, a crop, a dataset switch) the old picture is withheld until the new answer is in, even if it has
 * the same size: a click on it would name a point of the wrong measurement. A hexagonal scan shows only a picture
 * that carries the raster of its own map. On a square scan this is the picture itself.
 */
export function shownPicture(hexMap, entry) {
  if (!entry || !entry.image) return null;
  if (entry.hexMap !== hexMap) return null;
  return overviewMismatch(hexMap, entry.display) ? null : entry.image;
}

/**
 * What a finished fetch leaves behind: `{ image, display, hexMap }` for the map the picker held when it asked.
 * A square picker that last held a hexagonal picture and gets no picture also records that (`afterHex`): the
 * hexagonal picture it held can never be shown again, so the picker must say the fetch failed, not wait. A square
 * picker that never held one has nothing to say (its failed fetch leaves the box empty, as it always did).
 */
export function pictureEntry(previous, hexMap, image, display) {
  const entry = { image: image || null, display: display || null, hexMap };
  if (!hexMap && !entry.image && previous && (previous.hexMap || previous.afterHex)) entry.afterHex = true;
  return entry;
}

/** What a failed fetch leaves behind: a hexagonal picker records the failure; a square one only after a hexagonal one. */
export function pictureAfterFailure(previous, hexMap) {
  return hexMap || (previous && (previous.hexMap || previous.afterHex))
    ? pictureEntry(previous, hexMap, null, null)
    : previous;
}

/**
 * Why a picker has no picture although its fetch has finished, as the key under `ebsdviewer:overview` of the
 * text to show instead of "loading": 'failed' when the request failed or the answer carries no picture,
 * 'rasterMismatch' when the picture does not carry this map's raster. Null while no answer for this map is in
 * and when the picture is fine. A square scan keeps its loading text, unless its picker held a hexagonal picture
 * before (see pictureEntry).
 */
export function pictureProblem(hexMap, entry) {
  if (!entry || entry.hexMap !== hexMap) return null;
  if (!hexMap) return entry.afterHex && !entry.image ? 'failed' : null;
  if (!entry.image) return 'failed';
  return overviewMismatch(hexMap, entry.display) ? 'rasterMismatch' : null;
}
