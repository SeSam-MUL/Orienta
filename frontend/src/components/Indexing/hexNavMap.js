// frontend/src/components/Indexing/hexNavMap.js
/**
 * The hexagonal branch of the Indexing page's navigation map, as pure functions.
 *
 * On a hexagonal scan the picture is the backend's display raster. The page keeps its region in index space
 * ({rowStart, rowEnd, colStart, colEnd}, the end exclusive: the box of cells that is indexed); these functions are
 * where the picture and that box meet. `hexMap` is what makeHexMap() returns. On a square scan NavigationCanvas
 * never calls them.
 */
import { overviewPointer, rasterKey } from '../EBSDViewer/hexOverview';

/**
 * What the navigation picture is fetched for: the key of a hexagonal scan's geometry (the store's parsed form),
 * null for a square or missing geometry. A picture carries the key it was fetched under and is shown only while
 * the page holds the same key.
 */
export function navRasterKey(geometry) {
  if (!geometry || geometry.kind !== 'hex') return null;
  return rasterKey({
    kind: 'hex', n_rows: geometry.nRows, n_cols: geometry.nCols,
    step_x: geometry.stepX, step_y: geometry.stepY, odd_row_shift: geometry.oddRowShift,
  });
}

/**
 * Which measurement of a file the load answer (`ebsdInfo`) describes: the measurement and pattern source the file was
 * loaded with, and the APEX measurement it was paired with. Two measurements of one file have the same path and can
 * have the same geometry, so the path and the geometry alone cannot tell their pictures apart. Empty where the
 * answer says nothing (no copy of it, a file without choices).
 */
export function navMeasurementKey(info) {
  const variant = info?.variant;
  return [variant?.measurement ?? '', variant?.pattern_source ?? '', info?.apex?.measurement_id ?? ''].join('/');
}

/**
 * The index grid a geometry spans: its kind and its rows and columns, nothing of the steps. Saved regions are
 * index cells, so they stay valid for as long as this stays the same. Null for a missing geometry.
 */
export function navGridKey(geometry) {
  return geometry ? `${geometry.kind}|${geometry.nRows}|${geometry.nCols}` : null;
}

/**
 * The region a drag from `a` to `b` (fractions {rx, ry} of the picture) selects: the box of cells whose centres lie
 * in the dragged rectangle. A drag too small to hold a centre selects the cell shown by its middle pixel; one that
 * starts or ends outside the picture, or stays in a corner no hexagon covers, selects nothing (null).
 * The indexed set is the whole index box, so on a hexagonal scan it can reach half a cell past the dragged edge.
 */
export function regionFromDrag(hexMap, a, b) {
  if (!a || !b) return null;
  const pa = overviewPointer(hexMap, a.rx, a.ry);
  const pb = overviewPointer(hexMap, b.rx, b.ry);
  if (!pa || !pb) return null;
  const selection = hexMap.selection('rect', [{ i: pa.sr, j: pa.sc }, { i: pb.sr, j: pb.sc }]);
  if (!selection) return null;
  const { row0, col0, rows, cols } = selection.bbox;
  return { rowStart: row0, rowEnd: row0 + rows, colStart: col0, colEnd: col0 + cols };
}

/** The frame of a region, as fractions {x0, y0, x1, y1} of the picture: the raster box that covers its hexagons. */
export function regionFrame(hexMap, region) {
  const [h, w] = hexMap.shape;
  const box = hexMap.boxToPixels({
    row0: region.rowStart,
    col0: region.colStart,
    rows: region.rowEnd - region.rowStart,
    cols: region.colEnd - region.colStart,
  });
  return { x0: box.x / w, y0: box.y / h, x1: (box.x + box.w) / w, y1: (box.y + box.h) / h };
}

/**
 * A chemistry mask given per measurement point ({mask2d, shape: [rows, cols]}) on the raster:
 * {mask2d, shape: [h, w], skip}, where `skip` marks the pixels no hexagon covers (they are not part of the scan,
 * so they are neither selected nor excluded).
 */
export function maskToRaster(hexMap, maskData) {
  const [h, w] = hexMap.shape;
  const cols = maskData.shape[1];
  const mask2d = new Uint8Array(h * w);
  const skip = new Uint8Array(h * w);
  for (let i = 0; i < h; i += 1) {
    for (let j = 0; j < w; j += 1) {
      const cell = hexMap.cellAt(i, j);
      if (!cell) skip[i * w + j] = 1;
      else if (maskData.mask2d[cell.row * cols + cell.col]) mask2d[i * w + j] = 1;
    }
  }
  return { mask2d, shape: [h, w], skip };
}
