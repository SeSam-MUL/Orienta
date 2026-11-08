// frontend/src/services/hexMap.js
import { rasterSpec, cellAtPixel, pixelOfCell, pixelFromFraction, fractionOfCell } from './hexGeometry';
import { selectionFromPixels, lineCells, cellBoxToPixelBox } from './hexSelection';

/** `a` is a number within a millionth of `unit` of `b` (both sides are the same expression in doubles). */
const near = (a, b, unit) => typeof a === 'number' && Math.abs(a - b) <= 1e-6 * unit;

/**
 * What a page holds for a hexagonal scan: the raster and every conversion between it and the index grid,
 * in one frozen object. `null` for a square or missing geometry (or one whose steps are unknown), and that
 * null IS the "legacy path" switch: a caller writes `hexMap ? hexMap.cellAt(i, j) : { row: i, col: j }` and
 * nothing else changes for square scans.
 *
 * `fractionOf` gives the exact centre of a cell as fractions of the picture, for placing a marker. `pixelOf`
 * gives the whole pixel that holds that centre, for whoever needs a pixel; at factors below 4 that pixel can lie
 * in a neighbouring cell, so whoever needs the cell of a pixel uses `cellAt`.
 *
 * @param {object|null} geometry  what parseScanGeometry() returned
 */
export function makeHexMap(geometry) {
  const spec = rasterSpec(geometry);
  if (!spec) return null;
  return Object.freeze({
    geometry,
    spec,
    shape: spec.shape,
    umPerPx: spec.stepUm,
    cellAt: (i, j) => cellAtPixel(geometry, spec, i, j),
    pixelOf: (row, col) => pixelOfCell(geometry, spec, row, col),
    cellAtFraction: (fx, fy) => {
      const p = pixelFromFraction(spec, fx, fy);
      return p ? cellAtPixel(geometry, spec, p.i, p.j) : null;
    },
    fractionOf: (row, col) => fractionOfCell(geometry, spec, row, col),
    selection: (tool, points) => selectionFromPixels(geometry, spec, tool, points),
    line: (a, b, n) => lineCells(geometry, spec, a, b, n),
    boxToPixels: (box) => cellBoxToPixelBox(geometry, spec, box),
    /**
     * Is the raster the backend says it drew (`display` of its answer) this one: the same shape and factor, and
     * the same pixel size and origin on the sample? The origin matters because two scans can share a shape and
     * differ by a half step: a window cut from an even row and one cut from an odd row have opposite offsets of
     * their odd rows, so their rasters start at different x0 and a click would land half a step beside the cell.
     */
    matches: (display) => !!display && Array.isArray(display.shape)
      && display.shape[0] === spec.height && display.shape[1] === spec.width
      && display.factor === spec.factor
      && near(display.step_um, spec.stepUm, spec.stepUm)
      && near(display.x0, spec.x0, spec.stepUm)
      && near(display.y0, spec.y0, spec.stepUm),
  });
}
