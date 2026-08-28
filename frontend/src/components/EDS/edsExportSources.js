/**
 * Turn EDS layers into off-screen canvases that the shared ImageExportDialog
 * can export: one map on its own, the composite overlay, or a labelled sheet
 * with every visible map on it.
 *
 * The per-layer drawing mirrors Tile.jsx / LayeredCanvas (same blend, opacity,
 * threshold-mask and mask-preview rules), so an exported map looks like the one
 * on screen rather than like a second interpretation of the same layer.
 *
 * Every builder returns `{ canvas, umPerPx }` — the picture AND how many
 * micrometres one of its pixels covers. That second half is the whole point of
 * the shape: a scale bar has to be sized from the raster the bitmap actually
 * IS, and only the builder knows that. Asking the page instead ("what area did
 * the bottom layer come from?") was right for the single-map path and wrong for
 * the other two, because they resample:
 *
 *   composite  every layer is stretched into the SCAN grid, so the scan step
 *              applies even when the bottom layer is a 1024-px SE image
 *              (measured 8.4785x apart on SampleB);
 *   montage    each panel is stretched into a fixed 320 px cell and gaps sit
 *              between them, so NO single number is true for the sheet.
 *
 * `umPerPx: null` means "this picture has no single scale" and every consumer
 * must then offer no scale bar. It is never a placeholder for "unknown, use
 * something plausible".
 */

import { buildMaskCanvas } from '../PhaseMap/maskCanvas';

/** Luma weights used everywhere in this codebase for threshold masking. */
function applyThreshold(ctx, w, h, threshold) {
  if (!threshold) return;
  const img = ctx.getImageData(0, 0, w, h);
  const { min, max } = threshold;
  for (let i = 0; i < img.data.length; i += 4) {
    const v = img.data[i] * 0.299 + img.data[i + 1] * 0.587 + img.data[i + 2] * 0.114;
    if (v < min || v > max) img.data[i + 3] = 0;
  }
  ctx.putImageData(img, 0, 0);
}

/** The bitmap a layer actually draws from — mask layers borrow their source. */
export function sourceBitmapFor(layer, bitmaps) {
  return layer.kind === 'mask' ? bitmaps.get(layer.isMaskFor) : bitmaps.get(layer.id);
}

function newCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  return c;
}

/** `x` when it is a usable physical size, else null. Never undefined/NaN. */
function asUmPerPx(v) {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/**
 * One layer at its own native resolution — what a single tile shows.
 *
 * Opacity is deliberately NOT applied here: on its own the map should be fully
 * opaque. Opacity only means something against the layers underneath it, which
 * is the composite's job.
 *
 * `umPerPx` is the pixel size of the raster the BITMAP is on — for a mask, the
 * raster of the layer it masks, since those are the pixels drawn. The canvas is
 * the bitmap's own size, so the value passes straight through.
 */
export function buildSingleCanvas(layer, bitmap, umPerPx = null) {
  if (!bitmap) throw new Error('This map has no image data yet');
  const scale = asUmPerPx(umPerPx);
  if (layer.kind === 'mask' && layer.threshold) {
    const m = buildMaskCanvas(bitmap, layer.threshold);
    const c = newCanvas(m.width, m.height);
    c.getContext('2d').drawImage(m, 0, 0);
    return { canvas: c, umPerPx: scale };
  }
  const c = newCanvas(bitmap.width, bitmap.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  applyThreshold(ctx, c.width, c.height, layer.threshold);
  return { canvas: c, umPerPx: scale };
}

/**
 * The stacked overlay at scan-grid resolution.
 *
 * Layers are drawn bottom-up with their blend and opacity; differing native
 * resolutions (EDS scan grid vs the higher-res SEM survey) are co-registered by
 * the explicit width/height in drawImage.
 *
 * `umPerPx` is therefore the SCAN raster's pixel size — `shape` is the scan
 * grid, so that is the raster of every pixel in the result, no matter which
 * layer sits at the bottom. Passing the bottom layer's own size (what the EDS
 * page did until 2026-08-27) was wrong by the area ratio whenever that layer
 * was the SE image, which is the default stack.
 */
export function buildCompositeCanvas({ layers, bitmaps, shape, umPerPx = null }) {
  if (!shape || !layers?.length) throw new Error('Nothing to export: no layers');
  const [H, W] = shape;
  const c = newCanvas(W, H);
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);

  let drew = 0;
  for (const l of layers) {
    if (!l.visible) continue;
    const bm = sourceBitmapFor(l, bitmaps);
    if (!bm) continue;
    if (l.kind === 'mask' && l.threshold) {
      ctx.globalAlpha = l.opacity ?? 1;
      ctx.globalCompositeOperation = (l.blend && l.blend !== 'normal') ? l.blend : 'source-over';
      ctx.drawImage(buildMaskCanvas(bm, l.threshold), 0, 0, c.width, c.height);
      drew += 1;
      continue;
    }
    ctx.globalAlpha = l.opacity ?? 1;
    ctx.globalCompositeOperation = (l.blend && l.blend !== 'normal') ? l.blend : 'source-over';
    ctx.drawImage(bm, 0, 0, c.width, c.height);
    applyThreshold(ctx, c.width, c.height, l.threshold);
    drew += 1;
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  if (!drew) throw new Error('Nothing to export: no visible layer has data yet');
  return { canvas: c, umPerPx: asUmPerPx(umPerPx) };
}

/** Columns for `n` panels: roughly square, never more than 4 wide. */
export function montageColumns(n) {
  if (n <= 1) return 1;
  return Math.min(4, Math.ceil(Math.sqrt(n)));
}

/**
 * Every visible map side by side on one sheet, each with its name above it.
 *
 * Cells all use the scan-grid aspect so the panels line up in a rectangular
 * grid; each bitmap is scaled into its cell exactly the way the composite
 * co-registers differing resolutions.
 *
 * `umPerPx` is ALWAYS null, and that is a fact about the picture rather than a
 * gap in what we know: the sheet is `cellWidth`-wide panels separated by gaps
 * and captioned, so a bar laid across it spans cell, gap and label in unknown
 * proportion — and the panels need not share a raster anyway. One bar cannot be
 * true here, so none is offered.
 */
export function buildMontageCanvas({ layers, bitmaps, shape, labelFor, cellWidth = 320, gap = 8, background = '#000000', labelColor = '#ffffff' }) {
  const usable = (layers || []).filter((l) => l.visible && sourceBitmapFor(l, bitmaps));
  if (!usable.length) throw new Error('Nothing to export: no visible map has data yet');

  const [H, W] = shape || [1, 1];
  const aspect = H / W;
  const cellW = Math.max(64, Math.round(cellWidth));
  const cellH = Math.max(32, Math.round(cellW * aspect));
  const labelH = Math.max(14, Math.round(cellH * 0.11));
  const fontPx = Math.max(9, Math.round(labelH * 0.68));

  const cols = montageColumns(usable.length);
  const rows = Math.ceil(usable.length / cols);
  const c = newCanvas(
    cols * cellW + (cols + 1) * gap,
    rows * (cellH + labelH) + (rows + 1) * gap,
  );
  const ctx = c.getContext('2d');
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, c.width, c.height);

  usable.forEach((l, i) => {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = gap + col * (cellW + gap);
    const y = gap + row * (cellH + labelH + gap);

    ctx.save();
    ctx.font = `600 ${fontPx}px sans-serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = labelColor;
    const name = String(labelFor ? labelFor(l) : (l.label ?? l.id));
    ctx.fillText(name, x + 2, y + labelH / 2, cellW - 4);
    ctx.restore();

    // Each panel is rendered through the single-map path, so a tile and its
    // panel in the sheet cannot drift apart.
    const panel = buildSingleCanvas(l, sourceBitmapFor(l, bitmaps)).canvas;
    ctx.drawImage(panel, x, y + labelH, cellW, cellH);
  });

  return { canvas: c, umPerPx: null };
}

/** Canvas -> data URL the export dialog can load. */
export function canvasToDataUrl(canvas) {
  return canvas.toDataURL('image/png');
}

/**
 * Tight bounding box of the non-empty pixels of a canvas, or null when it is
 * empty (or already essentially full).
 *
 * A phase map covers only the indexed part of its scan grid; the rest is
 * transparent/black. The page auto-zooms to that part, so an export of the
 * whole grid puts the map small inside a wide empty surround — measured on a
 * real result: content filled 48.6% of the composed image. Measuring the
 * canvas is more dependable than asking the view for its zoom box, which is
 * only reported in some modes.
 *
 * `fullIfAbove` guards against returning a near-identical box for maps that do
 * fill their grid — there is nothing to gain from a 1-pixel crop.
 */
export function contentBounds(canvas, { alpha = 8, luma = 12, fullIfAbove = 0.98 } = {}) {
  const w = canvas?.width | 0;
  const h = canvas?.height | 0;
  if (!w || !h) return null;
  const d = canvas.getContext('2d').getImageData(0, 0, w, h).data;

  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = (y * w + x) * 4;
      if (d[i + 3] <= alpha) continue;
      if (d[i] <= luma && d[i + 1] <= luma && d[i + 2] <= luma) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return null;

  const box = { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
  if ((box.width * box.height) / (w * h) >= fullIfAbove) return null;
  return box;
}
