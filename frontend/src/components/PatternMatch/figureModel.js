// Pure, immutable figure-layout model + geometry/scale math. No React/DOM.
//
// A "model" is { canvas:{wPx,hPx,bg,dpi}, elements:[...] }. Every element has a
// normalized geometry { x, y, w, h } in [0..1] of the canvas. Element types:
//   panel    — an image source (experimental/simulated/ncc/heatmap)
//   colorbar — NCC color ramp
//   scalebar — µm / deg / px bar
//   text     — free caption / title
// The painter (paintFigure) renders ONLY this model, which is the structural
// guarantee that interactive handles/crosshair never reach the export.

export const SOURCE_META = {
  experimental: { label: 'Experimental' },
  simulated:    { label: 'Simulated' },
  ncc:          { label: 'NCC Δ' },
  heatmap:      { label: 'NCC map' },
};

const clamp01 = (v) => Math.max(0, Math.min(1, v));
let _idc = 0;
const newId = (t) => `${t}_${++_idc}`;

function makePanel(source, geom) {
  return {
    id: newId('panel'), type: 'panel', source, ...geom,
    border: { on: false, color: '#000000', width: 2 },
    label: { on: false, text: SOURCE_META[source]?.label || source },
    letter: { on: false, style: 'paren' },
    smoothing: 'nearest', aspectLock: true, includeMarkers: true,
  };
}

export function defaultModel(availableSources = []) {
  const n = Math.max(1, availableSources.length);
  // Auto-size the canvas to a single row of SQUARE panels so square EBSD
  // patterns fill their boxes with no letterbox and no wasted vertical margin.
  // (Previously the canvas was a fixed 2:1 and panel w/h were independent
  // fractions of it, which — with a stretch blit — squashed circular patterns
  // into ellipses; the painter now contains-fits, and this keeps the default
  // boxes square so there is nothing to letterbox.)
  const S = 460;            // panel side (px)
  const G = 36;             // gutter (px)
  const captionStrip = 30;  // room under the panels for labels / letters
  const wPx = n * S + (n + 1) * G;
  const hPx = S + 2 * G + captionStrip;
  const gx = G / wPx, gy = G / hPx;
  const w = S / wPx, h = S / hPx;
  const elements = availableSources.map((source, i) =>
    makePanel(source, { x: gx + i * (w + gx), y: gy, w, h }));
  return { canvas: { wPx, hPx, bg: '#ffffff', dpi: 300 }, elements };
}

export function addPanel(model, source) {
  // Square in PIXELS on the current canvas aspect (w*wPx == h*hPx).
  const { wPx = 1, hPx = 1 } = model.canvas || {};
  const w = 0.32;
  const h = Math.min(0.9, w * (wPx / hPx));
  return { ...model, elements: [...model.elements, makePanel(source, { x: 0.1, y: 0.1, w, h })] };
}

export function addElement(model, el) {
  return { ...model, elements: [...model.elements, { id: newId(el.type || 'el'), ...el }] };
}

export function removeElement(model, id) {
  return { ...model, elements: model.elements.filter((e) => e.id !== id) };
}

export function updateElement(model, id, patch) {
  return { ...model, elements: model.elements.map((e) => e.id === id ? deepMerge(e, patch) : e) };
}

function deepMerge(a, patch) {
  const out = { ...a };
  for (const k of Object.keys(patch)) {
    out[k] = (patch[k] && typeof patch[k] === 'object' && !Array.isArray(patch[k]))
      ? deepMerge(a[k] || {}, patch[k]) : patch[k];
  }
  return out;
}

export function moveElement(el, dx, dy) {
  return { ...el, x: clamp01(el.x + dx), y: clamp01(el.y + dy) };
}

export function resizeElement(el, handle, dx, dy, { aspectLock } = {}) {
  let { x, y, w, h } = el;
  const right = el.x + el.w, bottom = el.y + el.h;
  if (handle.includes('e')) w += dx;
  if (handle.includes('s')) h += dy;
  if (handle.includes('w')) { x += dx; w -= dx; }
  if (handle.includes('n')) { y += dy; h -= dy; }
  w = Math.max(0.03, w); h = Math.max(0.03, h);
  if (aspectLock) {
    const ratio = (el.w > 0 && el.h > 0) ? el.w / el.h : 1; // guard div-by-zero
    h = Math.max(0.03, w / ratio);                          // re-clamp after aspect
    // Keep the corner opposite the dragged handle anchored so the box grows
    // toward the handle (N/W handles otherwise drift down-right).
    if (handle.includes('n')) y = bottom - h;
    if (handle.includes('w')) x = right - w;
  }
  return { ...el, x: clamp01(x), y: clamp01(y), w: Math.min(w, 1), h: Math.min(h, 1) };
}

export function snap(value, others, tol) {
  let best = value, bestD = tol;
  for (const o of others) {
    const d = Math.abs(o - value);
    if (d < bestD) { bestD = d; best = o; }
  }
  return best;
}

export function alignElements(els, mode) {
  if (!els.length) return [];
  const xs = els.map(e => e.x), rs = els.map(e => e.x + e.w);
  const ys = els.map(e => e.y), bs = els.map(e => e.y + e.h);
  const minX = Math.min(...xs), maxR = Math.max(...rs), minY = Math.min(...ys), maxB = Math.max(...bs);
  return els.map((e) => {
    let patch = {};
    if (mode === 'left') patch = { x: minX };
    else if (mode === 'right') patch = { x: maxR - e.w };
    else if (mode === 'top') patch = { y: minY };
    else if (mode === 'bottom') patch = { y: maxB - e.h };
    else if (mode === 'hcenter') patch = { x: (minX + maxR) / 2 - e.w / 2 };
    else if (mode === 'vcenter') patch = { y: (minY + maxB) / 2 - e.h / 2 };
    return { id: e.id, patch };
  });
}

export function distributeElements(els, axis) {
  if (els.length < 3) return [];
  const key = axis === 'x' ? 'x' : 'y';
  const sorted = [...els].sort((a, b) => a[key] - b[key]);
  const first = sorted[0][key], last = sorted[sorted.length - 1][key];
  const gap = (last - first) / (sorted.length - 1);
  return sorted.map((e, i) => ({ id: e.id, patch: { [key]: first + i * gap } }));
}

export function sameSize(els, dim) {
  if (!els.length) return [];
  const v = els[0][dim];
  return els.map((e) => ({ id: e.id, patch: { [dim]: v } }));
}

// Pick a 1/2/5×10ⁿ µm length spanning ≈⅓ of the panel for the scale bar.
// `nativePx` is the panel's full native width in scan pixels (= map columns),
// `panelWidthFracOfNative` is the fraction of that native extent the panel shows
// (1.0 for a full map). Returns the nice length + its fraction of the panel.
//
// `fracOfPanel` is a HINT for the toolbar only. The painter must not size a bar
// from it: it is a fraction of the heatmap panel, while the bar is drawn inside
// the scale-bar element's own box, and the two stopped agreeing the moment the
// user dragged a resize handle — the bar changed length and the label did not.
export function niceScaleLength(stepUm, panelWidthFracOfNative, nativePx) {
  const span = nativePx * (panelWidthFracOfNative || 1);
  const totalUm = stepUm * span;
  if (!(totalUm > 0)) return { valueUm: 0, fracOfPanel: 0 }; // guard 0/negative/NaN
  const target = totalUm / 3;
  const pow = Math.pow(10, Math.floor(Math.log10(target)));
  const cands = [1, 2, 5, 10].map((m) => m * pow);
  let valueUm = cands[0];
  for (const c of cands) if (c <= target) valueUm = c;
  return { valueUm, fracOfPanel: (valueUm / stepUm) / span };
}

/**
 * Round a count down to a 1/2/5 x 10^n value, for a bar measured in map pixels.
 * Same ladder as the micrometre picker, so the two read alike.
 */
export function niceCount(raw) {
  const v = Number(raw);
  if (!(v > 0)) return 0;
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  let out = pow;
  for (const c of [1, 2, 5, 10].map((m) => m * pow)) if (c <= v) out = c;
  return Math.max(1, Math.round(out));
}

/**
 * The rectangle a panel's image is actually DRAWN in, in output pixels.
 *
 * The painter fits the image inside the panel box (object-fit: contain) so a
 * square pattern stays square, which means the drawn width is generally NOT the
 * box width. A scale bar sized against the box is stretched by exactly that
 * mismatch — on the default layout, a 4:3 heatmap in a wider box.
 *
 * Mirrors the fitting in `paintFigure`; kept here because it is geometry, and
 * because the toolbar needs it without a canvas.
 */
export function panelDrawnRect(el, img, { W, H }) {
  const px = el.x * W, py = el.y * H, pw = el.w * W, ph = el.h * H;
  if (!img || el.aspectLock === false
      || !(img.naturalWidth > 0) || !(img.naturalHeight > 0)) {
    return { x: px, y: py, w: pw, h: ph };
  }
  const ar = img.naturalWidth / img.naturalHeight;
  let dw, dh;
  if (ar > pw / ph) { dw = pw; dh = pw / ar; }
  else { dh = ph; dw = ph * ar; }
  return { x: px + (pw - dw) / 2, y: py + (ph - dh) / 2, w: dw, h: dh };
}

/**
 * Micrometres covered by ONE OUTPUT PIXEL of the heatmap panel, or null.
 *
 * This is the only quantity a scale bar on this figure may be sized from, and
 * everything it needs travels with `sources`: the decoded heatmap image, the
 * scan step (`stepUm`) and how many scan columns the heatmap spans (`mapCols`).
 *
 * `mapCols` is the heatmap's OWN column count, not the scan's: /ncc-heatmap
 * crops to the indexed bounding box and reports `n_cols` for the crop, with the
 * offset alongside. It also upscales the picture for click precision, which is
 * why the drawn width has to be measured rather than assumed.
 *
 * null means "this figure cannot state a length" and no bar is drawn. Until
 * 2026-08-27 every caller passed `stepUm: null`, so this was ALWAYS the case
 * and the button silently produced a bar labelled "100 px" whose width was
 * 30 % of its own box.
 */
export function heatmapUmPerOutputPx(model, sources, size) {
  return heatmapScale(model, sources, size).umPerPx;
}

/**
 * What the heatmap panel can measure: `{ umPerPx, outPxPerScanPx }`.
 *
 * `outPxPerScanPx` needs no step size — how wide one SCAN pixel is drawn is
 * pure geometry — so a figure from a file with no header geometry can still
 * carry an honest bar, in map pixels. `umPerPx` additionally needs the step.
 * Either can be null on its own.
 */
export function heatmapScale(model, sources, { W, H }) {
  const none = { umPerPx: null, outPxPerScanPx: null };
  const cols = Number(sources?.mapCols);
  if (!(cols > 0)) return none;
  const el = (model?.elements || []).find(
    (e) => e.type === 'panel' && e.source === 'heatmap',
  );
  if (!el) return none;
  const rect = panelDrawnRect(el, sources?.heatmap, { W, H });
  if (!(rect.w > 0)) return none;
  const outPxPerScanPx = rect.w / cols;
  const step = Number(sources?.stepUm);
  return {
    umPerPx: step > 0 ? (step * cols) / rect.w : null,
    outPxPerScanPx,
  };
}

// Bump the module id counter past any numeric suffix in a (loaded) model so
// freshly-added elements can't collide with ids restored from a preset.
export function reseedIdsFrom(model) {
  let max = 0;
  for (const e of model?.elements || []) {
    const m = /_(\d+)$/.exec(e.id || '');
    if (m) max = Math.max(max, Number(m[1]));
  }
  if (max > _idc) _idc = max;
}

export function letterString(index, style = 'paren') {
  const ch = String.fromCharCode(97 + index);
  if (style === 'paren') return `(${ch})`;
  if (style === 'rparen') return `${ch})`;
  if (style === 'upper') return ch.toUpperCase();
  return ch;
}
