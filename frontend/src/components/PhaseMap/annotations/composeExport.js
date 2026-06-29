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

  // Annotations — drawn on top, in DOM order. Coordinates are normalised
  // [0..1] of the canvas size, so they map identically into the export.
  for (const annot of annotations || []) {
    const drawer = TYPE_DRAWERS[annot.type];
    if (!drawer) continue;
    const ax = annot.x * W;
    const ay = annot.y * H;
    const aw = annot.w * W;
    const ah = annot.h * H;
    const rot = ((annot.rotation || 0) * Math.PI) / 180;
    ctx.save();
    if (rot) {
      ctx.translate(ax + aw / 2, ay + ah / 2);
      ctx.rotate(rot);
      ctx.translate(-aw / 2, -ah / 2);
    } else {
      ctx.translate(ax, ay);
    }
    drawer(ctx, annot, aw, ah, {
      phaseStats, scale, stepX,
      canvasWidthPx: W, canvasHeightPx: H,
      scanCols: shape.cols,
    });
    ctx.restore();
  }

  // Encode to PNG Blob
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => {
      if (b) resolve(b);
      else reject(new Error('composeExport: canvas.toBlob returned null'));
    }, 'image/png');
  });
}

// ----- Per-annotation drawers -----------------------------------------------

function drawLegend(ctx, annot, w, h, { phaseStats }) {
  const p = annot.props || {};
  const fontSize = (p.fontSize ?? 11);
  const bg = p.background || 'rgba(20,22,30,0.85)';
  ctx.fillStyle = bg;
  // rounded rect via path
  const r = 4;
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
  const fontSize = (p.fontSize ?? 12);
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

function drawTitle(ctx, annot, w, h) {
  const p = annot.props || {};
  const text = p.text ?? '';
  const fontSize = p.fontSize ?? 16;
  const color = p.color ?? '#ffffff';
  ctx.font = `600 ${fontSize}px sans-serif`;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(0,0,0,0.7)';
  ctx.shadowBlur = 4;
  ctx.fillText(text, 6, h / 2);
  ctx.shadowBlur = 0;
}

function drawArrow(ctx, annot, w, h) {
  const p = annot.props || {};
  const color = p.color ?? '#ffb86c';
  const fontSize = p.fontSize ?? 11;
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
