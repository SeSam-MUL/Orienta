// Pure painter — draws ONLY the layout model. Interactive handles/crosshair are
// NOT part of the model, so they can never appear here. This is the structural
// guarantee that an exported figure contains no stray dashed selection boxes.
//
// Used for both the live preview (screen scale) and the export (high scale):
// preview === export. Canvas fillStyle/strokeStyle MUST be concrete colors.
import { heatmapScale, letterString } from './figureModel';

const NCC_STOPS = ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b'];
const RED = '#ff5555';

export function paintFigure(ctx, model, sources, { W, H }) {
  const bg = model.canvas?.bg ?? '#ffffff';
  if (bg && bg !== 'transparent') { ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H); }

  // What the heatmap can measure, worked out ONCE from the panel as it will be
  // drawn. Every scale bar in the figure is sized from this and from nothing
  // else — not from its own box, which the user drags freely. Both halves null
  // = this figure carries no scale, and then no bar is drawn at all.
  const mapScale = heatmapScale(model, sources, { W, H });

  let panelIdx = 0;
  for (const el of model.elements) {
    const px = el.x * W, py = el.y * H, pw = el.w * W, ph = el.h * H;
    if (el.type === 'panel') {
      const img = sources[el.source];
      // Fitted rect = object-fit:contain of the image inside the panel box, so a
      // square (circular) EBSD pattern stays circular instead of being stretched
      // into the box aspect. Falls back to the full box when aspect is unlocked
      // or the image has no intrinsic size (e.g. the unit-test fake context).
      let dx = px, dy = py, dw = pw, dh = ph;
      if (img && el.aspectLock !== false && img.naturalWidth > 0 && img.naturalHeight > 0) {
        const ar = img.naturalWidth / img.naturalHeight;
        if (ar > pw / ph) { dw = pw; dh = pw / ar; }   // image relatively wider → fit width
        else              { dh = ph; dw = ph * ar; }   // image relatively taller → fit height
        dx = px + (pw - dw) / 2;
        dy = py + (ph - dh) / 2;
      }
      if (img) {
        ctx.imageSmoothingEnabled = el.smoothing !== 'nearest';
        ctx.drawImage(img, dx, dy, dw, dh);
      }
      if (el.includeMarkers && sources.markers && el.source !== 'heatmap') {
        drawMarkers(ctx, sources.markers, dx, dy, dw, dh);
      }
      if (el.border?.on) {
        ctx.strokeStyle = el.border.color || '#000000';
        ctx.lineWidth = el.border.width || 2;
        ctx.strokeRect(px, py, pw, ph);
      }
      if (el.label?.on && el.label.text) {
        drawTextBox(ctx, el.label.text, px, py + ph + 4, 14 * (H / 600), '#000000');
      }
      if (el.letter?.on) {
        drawTextBox(ctx, letterString(panelIdx, el.letter.style || 'paren'),
          px + 4, py + 4, 18 * (H / 600), '#000000', true);
      }
      panelIdx++;
    } else if (el.type === 'colorbar') {
      drawColorbar(ctx, el, px, py, pw, ph, H);
    } else if (el.type === 'scalebar') {
      drawScalebar(ctx, el, px, py, pw, ph, H, mapScale);
    } else if (el.type === 'text') {
      drawTextBox(ctx, el.text || '', px, py, (el.fontSize || 16) * (H / 600), el.color || '#000000', el.weight === 'bold');
    }
  }
}

function drawMarkers(ctx, markers, px, py, pw, ph) {
  for (const m of markers) {
    const cx = px + m.x * pw, cy = py + m.y * ph;
    ctx.strokeStyle = RED; ctx.lineWidth = Math.max(1.5, pw * 0.012);
    ctx.beginPath(); ctx.arc(cx, cy, Math.max(5, pw * 0.03), 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = RED; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.font = `700 ${Math.max(9, pw * 0.05)}px sans-serif`;
    ctx.fillText(String(m.n), cx, cy);
    ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
  }
}

function drawColorbar(ctx, el, px, py, pw, ph, H) {
  if (el.visible === false) return;
  const grad = ctx.createLinearGradient(0, py, 0, py + ph);
  NCC_STOPS.forEach((c, i) => grad.addColorStop(i / (NCC_STOPS.length - 1), c));
  ctx.fillStyle = grad; ctx.fillRect(px, py, pw, ph);
  ctx.strokeStyle = '#000000'; ctx.lineWidth = 1; ctx.strokeRect(px, py, pw, ph);
  const fs = (el.fontSize || 11) * (H / 600);
  ctx.fillStyle = el.color || '#000000'; ctx.font = `${fs}px sans-serif`;
  ctx.textAlign = 'start'; ctx.textBaseline = 'middle';
  ctx.fillText(`+${el.vmax ?? 1}`, px + pw + 4, py + fs * 0.6);
  ctx.fillText(`${el.vmin ?? -1}`, px + pw + 4, py + ph - fs * 0.6);
  ctx.textBaseline = 'alphabetic';
}

/**
 * The bar's LENGTH is physics; its box is only a frame the user positions.
 *
 * `mapScale` is what the heatmap panel can measure. Without the half this bar's
 * mode needs, the figure cannot state the length and NOTHING is drawn —
 * deliberately, because the alternative this replaces was a bar of arbitrary
 * width under a label that claimed a number ("100 px" beside a bar 30 % of its
 * own box wide, in a figure with no scale at all).
 *
 * `mode: 'px'` measures in SCAN pixels. That is a real statement about the map
 * and needs no step size, so it stays available on files without one.
 */
function drawScalebar(ctx, el, px, py, pw, ph, H, mapScale) {
  const lengthValue = Number(el.lengthValue);
  if (!(lengthValue > 0)) return;
  const px_mode = el.mode === 'px';
  const unit = el.unit || (px_mode ? 'px' : 'µm');
  // Both modes reduce to "how many output pixels is this length".
  const barW = px_mode
    ? lengthValue * (mapScale?.outPxPerScanPx ?? 0)
    : lengthValue / (mapScale?.umPerPx > 0 ? mapScale.umPerPx : Infinity);
  if (!(barW > 0) || !Number.isFinite(barW)) return;

  const fs = (el.fontSize || 12) * (H / 600);
  const barH = Math.max(3, fs * 0.4);
  ctx.fillStyle = el.color || '#000000';
  ctx.fillRect(px, py, barW, barH);
  ctx.font = `${fs}px sans-serif`; ctx.textAlign = 'start'; ctx.textBaseline = 'top';
  // The label is generated from the length that was just drawn, so the two
  // cannot drift apart. A stored `labelText` is honoured only when it still
  // describes that length.
  ctx.fillText(`${lengthValue} ${unit}`, px, py + barH + 2);
  ctx.textBaseline = 'alphabetic';
}

function drawTextBox(ctx, text, x, y, fontSize, color, bold) {
  ctx.font = `${bold ? '700 ' : ''}${fontSize}px sans-serif`;
  ctx.fillStyle = color; ctx.textAlign = 'start'; ctx.textBaseline = 'top';
  ctx.fillText(text, x, y);
  ctx.textBaseline = 'alphabetic';
}
