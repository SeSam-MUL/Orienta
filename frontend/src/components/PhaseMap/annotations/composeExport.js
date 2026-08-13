/**
 * Composite the current Phase Maps view (layer bitmaps + on-canvas
 * annotations) into a single high-resolution PNG.
 *
 * Produces a Blob the caller can save via FileSaver or a temp <a>
 * download link.
 *
 * Inputs:
 *   layers:      Array<{id, opacity, blend, visible}> from layer stack
 *   bitmaps:     Map<id, ImageBitmap>
 *   shape:       { rows, cols } — native pixel shape of the map
 *   annotations: Array of {type, x, y, w, h, rotation, props}
 *   scale:       integer scale factor (1 = native, 2 = 2x DPI, ...)
 *   phaseStats:  same shape consumed by LegendAnnotation
 *
 * Output: Promise<Blob>
 */

import { BLEND_MAP } from '../layerSources';

const TYPE_DRAWERS = {
  legend:   drawLegend,
  scalebar: drawScalebar,
  title:    drawTitle,
  arrow:    drawArrow,
};

/**
 * The map itself as a canvas: the layer stack at native resolution, times
 * `scale`, with the IPF colour key beside it when one is shown.
 *
 * Annotations are deliberately NOT drawn here. The export dialog keeps them
 * editable on top of this picture and burns them in at save time, so dragging
 * one does not mean re-composing the whole map.
 */
export async function composeMapCanvas({
  layers, bitmaps, shape, scale = 2, ipfKey = null, contentBbox = null,
}) {
  if (!shape?.rows || !shape?.cols) throw new Error('composeMapCanvas: invalid shape');
  // Frame what the screen frames. The view auto-zooms to the indexed region,
  // and the annotations were arranged against THAT. Exporting the whole scan
  // grid instead put a title the user had placed over the map out in the empty
  // margin beside it.
  const src = (contentBbox && contentBbox.w > 0 && contentBbox.h > 0)
    ? { x: contentBbox.x || 0, y: contentBbox.y || 0, w: contentBbox.w, h: contentBbox.h }
    : { x: 0, y: 0, w: shape.cols, h: shape.rows };
  const mapW = Math.round(src.w * scale);
  const mapH = Math.round(src.h * scale);

  // The key rides in a column to the right, on its own light plate — the same
  // arrangement as on screen, so the exported figure reads like the view.
  const keyImg = ipfKey ? await loadKeyImage(ipfKey) : null;
  const keyW = keyImg ? Math.round(mapW * 0.24) : 0;

  const canvas = document.createElement('canvas');
  canvas.width = mapW + keyW;
  canvas.height = mapH;
  const ctx = canvas.getContext('2d');

  ctx.imageSmoothingEnabled = false;   // nearest neighbour: crisp data pixels
  ctx.fillStyle = '#1a1b26';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (const layer of layers) {
    if (!layer.visible) continue;
    const bmp = bitmaps?.get?.(layer.id);
    if (!bmp) continue;
    ctx.globalAlpha = Math.max(0, Math.min(1, layer.opacity ?? 1));
    ctx.globalCompositeOperation = BLEND_MAP[layer.blend] || 'source-over';
    ctx.drawImage(bmp, src.x, src.y, src.w, src.h, 0, 0, mapW, mapH);
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';

  if (keyImg) {
    ctx.fillStyle = '#f5f5f5';
    ctx.fillRect(mapW, 0, keyW, mapH);
    // Contain-fit with a margin, pinned to the top like the on-screen panel.
    const pad = Math.round(keyW * 0.06);
    const availW = keyW - 2 * pad;
    const availH = mapH - 2 * pad;
    const k = Math.min(availW / keyImg.width, availH / keyImg.height);
    const w = keyImg.width * k;
    const h = keyImg.height * k;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(keyImg, mapW + pad + (availW - w) / 2, pad, w, h);
    ctx.imageSmoothingEnabled = false;
  }

  // `mapCols/mapRows` say how many SCAN columns the picture spans — the scale
  // bar needs that, not the full grid, once the frame is cropped.
  return {
    canvas, mapWidth: mapW, mapHeight: mapH, keyWidth: keyW,
    mapCols: src.w, mapRows: src.h,
  };
}

function loadKeyImage(src) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    // A missing key must not sink the export — the map is the point.
    img.onerror = () => resolve(null);
    img.src = src.startsWith('data:') ? src : `data:image/png;base64,${src}`;
  });
}

export async function composeExport({
  layers, bitmaps, shape, annotations, scale = 2, phaseStats, stepX = null,
}) {
  if (!shape?.rows || !shape?.cols) {
    throw new Error('composeExport: invalid shape');
  }
  const W = shape.cols * scale;
  const H = shape.rows * scale;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  // Layer compositing — same order/blend the live LayeredCanvas uses.
  ctx.imageSmoothingEnabled = false;  // nearest-neighbour for crisp pixels
  ctx.fillStyle = '#1a1b26';
  ctx.fillRect(0, 0, W, H);
  for (const layer of layers) {
    if (!layer.visible) continue;
    const bmp = bitmaps?.get?.(layer.id);
    if (!bmp) continue;
    ctx.globalAlpha = Math.max(0, Math.min(1, layer.opacity ?? 1));
    ctx.globalCompositeOperation = BLEND_MAP[layer.blend] || 'source-over';
    ctx.drawImage(bmp, 0, 0, W, H);
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';

  drawAnnotationsOnto(ctx, annotations, {
    width: W, height: H, phaseStats, scale, stepX, scanCols: shape.cols,
  });

  // Encode to PNG Blob
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => {
      if (b) resolve(b);
      else reject(new Error('composeExport: canvas.toBlob returned null'));
    }, 'image/png');
  });
}

/**
 * Draw the map's annotations into any 2D context.
 *
 * Split out of composeExport so the image-export dialog can burn the SAME
 * annotations into its own render — one drawing path, so the figure the dialog
 * writes cannot drift from the one this module writes.
 *
 * `width`/`height` describe the box the normalised [0..1] annotation
 * coordinates refer to; `offsetX`/`offsetY` shift that box inside the target
 * canvas (the dialog adds a border around the image).
 */
export function drawAnnotationsOnto(ctx, annotations, {
  width, height, offsetX = 0, offsetY = 0,
  phaseStats, scale = 1, stepX = null, scanCols = null,
  // Multiplies every annotation font size. Default 1 keeps the long-standing
  // "Composite export" button rendering exactly as before; the image-export
  // dialog passes the magnification so a 4x figure gets 4x text instead of
  // hairline captions on a huge picture.
  textScale = 1,
} = {}) {
  const W = width;
  const H = height;
  for (const annot of annotations || []) {
    const drawer = TYPE_DRAWERS[annot.type];
    if (!drawer) continue;
    const ax = offsetX + annot.x * W;
    const ay = offsetY + annot.y * H;
    let aw = annot.w * W;
    let ah = annot.h * H;
    if (annot.type === 'scalebar' && stepX > 0 && scanCols > 0) {
      // Same rule as the on-screen frame: the box is the bar plus padding.
      // The stored width belongs to a box the bar never respected, and behind a
      // plate it would show as a stripe of colour running far past the bar.
      const barPx = ((annot.props?.lengthUm ?? 5) / stepX) / scanCols * W;
      const fs = (annot.props?.fontSize ?? 12) * textScale;
      const pad = 5 * textScale;
      aw = barPx + 2 * pad;
      ah = Math.max(3, fs * 0.4) + 2 + fs * 1.3 + 2 * pad;
    }
    const rot = ((annot.rotation || 0) * Math.PI) / 180;
    ctx.save();
    if (rot) {
      ctx.translate(ax + aw / 2, ay + ah / 2);
      ctx.rotate(rot);
      ctx.translate(-aw / 2, -ah / 2);
    } else {
      ctx.translate(ax, ay);
    }
    // The plate goes down before the content, for every type — one rule, so a
    // legend and a scale bar look the same way when both sit on one.
    fillPlate(ctx, annot, aw, ah);
    drawer(ctx, annot, aw, ah, {
      phaseStats, scale, stepX, textScale,
      canvasWidthPx: W, canvasHeightPx: H,
      scanCols,
    });
    ctx.restore();
  }
}

// ----- Per-annotation drawers -----------------------------------------------

/** The plate behind an annotation, in the exported picture. */
function fillPlate(ctx, annot, w, h) {
  const p = annot.props || {};
  const legacy = typeof p.background === 'string' && p.background ? p.background : null;
  let fill = null;
  if (p.bgOpacity == null && legacy) {
    fill = legacy;
  } else {
    const a = Math.max(0, Math.min(1, Number(p.bgOpacity) || 0));
    if (a > 0) {
      const hex = String(p.bgColor || '#000000').replace('#', '');
      const full = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex;
      const r = parseInt(full.slice(0, 2), 16) || 0;
      const g = parseInt(full.slice(2, 4), 16) || 0;
      const b = parseInt(full.slice(4, 6), 16) || 0;
      fill = `rgba(${r}, ${g}, ${b}, ${a})`;
    }
  }
  if (!fill) return;
  ctx.save();
  ctx.fillStyle = fill;
  const r = Math.min(4, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(r, 0);
  ctx.lineTo(w - r, 0);
  ctx.quadraticCurveTo(w, 0, w, r);
  ctx.lineTo(w, h - r);
  ctx.quadraticCurveTo(w, h, w - r, h);
  ctx.lineTo(r, h);
  ctx.quadraticCurveTo(0, h, 0, h - r);
  ctx.lineTo(0, r);
  ctx.quadraticCurveTo(0, 0, r, 0);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawLegend(ctx, annot, w, h, opts) {
  const { phaseStats } = opts || {};
  const p = annot.props || {};
  const fontSize = (p.fontSize ?? 11) * (opts?.textScale ?? 1);
  // The plate is drawn by `fillPlate` for every annotation type, before this
  // runs. Painting a second one here would ignore the user's opacity.

  const phases = phaseStats?.phases || [];
  const pad = 8;
  const lineH = fontSize * 1.4;
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textBaseline = 'middle';
  let y = pad + lineH / 2;
  for (const ph of phases) {
    if (y + lineH / 2 > h - pad) break;
    // colour square
    ctx.fillStyle = ph.color_hex;
    ctx.fillRect(pad, y - fontSize / 2, fontSize, fontSize);
    ctx.strokeStyle = '#11181a';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad + 0.5, y - fontSize / 2 + 0.5, fontSize - 1, fontSize - 1);
    // name
    ctx.fillStyle = '#eaeaea';
    const nameX = pad + fontSize + 6;
    const pctText = `${(ph.area_pct ?? 0).toFixed(1)}%`;
    const pctW = ctx.measureText(pctText).width;
    const nameMax = w - pad - pctW - 8 - nameX;
    let name = ph.name || '?';
    while (name.length > 0 && ctx.measureText(name).width > nameMax) {
      name = name.slice(0, -1);
    }
    if (name !== (ph.name || '?')) name = name.slice(0, -1) + '…';
    ctx.fillText(name, nameX, y);
    // pct
    ctx.fillStyle = '#a5a8b0';
    ctx.fillText(pctText, w - pad - pctW, y);
    y += lineH;
  }
}

function drawScalebar(ctx, annot, w, h, opts) {
  const p = annot.props || {};
  const fontSize = (p.fontSize ?? 12) * (opts?.textScale ?? 1);
  const barColor = p.barColor || '#ffffff';
  const textColor = p.textColor || '#ffffff';
  const lengthUm = p.lengthUm ?? 5;
  const barHeight = Math.max(3, fontSize * 0.4);
  const barY = (h - fontSize - 4) / 2;
  // Bar physical width: same logic as on-canvas ScalebarBody. Box width
  // is the maximum extent; bar shrinks to match the actual physical
  // length given stepX (µm/scan-pixel) and scanCols.
  let barW = w;
  if (opts?.stepX && opts?.scanCols && opts?.canvasWidthPx && w > 0) {
    const wantedFractionOfCanvas = (lengthUm / opts.stepX) / opts.scanCols;
    const wantedCanvasPx = wantedFractionOfCanvas * opts.canvasWidthPx;
    barW = Math.max(2, Math.min(w, wantedCanvasPx));
  }
  ctx.fillStyle = barColor;
  ctx.fillRect((w - barW) / 2, barY, barW, barHeight);
  // text
  ctx.font = `${fontSize}px sans-serif`;
  ctx.fillStyle = textColor;
  ctx.textBaseline = 'top';
  ctx.textAlign = 'center';
  // halo for legibility
  ctx.shadowColor = 'rgba(0,0,0,0.8)';
  ctx.shadowBlur = 4;
  ctx.fillText(`${lengthUm} µm`, w / 2, barY + barHeight + 2);
  ctx.shadowBlur = 0;
  ctx.textAlign = 'start';
}

function drawTitle(ctx, annot, w, h, opts) {
  const p = annot.props || {};
  const text = p.text ?? '';
  const fontSize = (p.fontSize ?? 16) * (opts?.textScale ?? 1);
  const color = p.color ?? '#ffffff';
  ctx.font = `600 ${fontSize}px sans-serif`;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0,0,0,0.7)';
  ctx.shadowBlur = 4;
  ctx.fillText(text, 6, h / 2);
  ctx.shadowBlur = 0;
}

function drawArrow(ctx, annot, w, h, opts) {
  const p = annot.props || {};
  const color = p.color ?? '#ffb86c';
  const fontSize = (p.fontSize ?? 11) * (opts?.textScale ?? 1);
  const label = p.label ?? 'ND';
  // Body of arrow: vertical line from bottom to ~80% up
  const cx = w / 2;
  const tipY = h * 0.2;
  const baseY = h * 0.9;
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2, h * 0.05);
  ctx.beginPath();
  ctx.moveTo(cx, baseY);
  ctx.lineTo(cx, tipY);
  ctx.stroke();
  // Arrowhead
  const ah = Math.max(6, h * 0.12);
  const aw = Math.max(4, w * 0.18);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(cx, tipY - ah);
  ctx.lineTo(cx - aw, tipY + ah * 0.2);
  ctx.lineTo(cx + aw, tipY + ah * 0.2);
  ctx.closePath();
  ctx.fill();
  // Label
  ctx.font = `700 ${fontSize * 1.4}px sans-serif`;
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.shadowColor = 'rgba(0,0,0,0.7)';
  ctx.shadowBlur = 3;
  ctx.fillText(label, cx, h);
  ctx.shadowBlur = 0;
  ctx.textAlign = 'start';
}

/** Convenience helper — triggers a browser download of the composed PNG. */
export async function downloadComposedExport(opts, filename = 'phase_map.png') {
  const blob = await composeExport(opts);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}
