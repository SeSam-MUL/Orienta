/**
 * Pure core of the image export: crop geometry, output sizing, scalebar
 * rounding, filenames, and the canvas render itself.
 *
 * Kept free of React and of any app knowledge so the maths can be tested
 * directly. `ImageExportDialog` owns the UI; this file owns the arithmetic and
 * the single place where pixels are actually produced.
 *
 * Coordinate conventions
 * ----------------------
 * `crop` is in IMAGE pixels: { x, y, width, height } with the origin top-left.
 * `output` is in EXPORT pixels. Annotations are drawn in output coordinates so
 * that a 2 px line stays 2 px on the written file rather than becoming a
 * hairline at 16x.
 */

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------

// Only what canvas.toBlob() can actually encode. TIFF is deliberately absent:
// browsers cannot write it, and offering it would fail at the last step.
export const FORMATS = [
  { id: 'png', mime: 'image/png', ext: 'png', lossy: false },
  { id: 'jpeg', mime: 'image/jpeg', ext: 'jpg', lossy: true },
  { id: 'webp', mime: 'image/webp', ext: 'webp', lossy: true },
];

export function formatById(id) {
  return FORMATS.find((f) => f.id === id) || FORMATS[0];
}

// ---------------------------------------------------------------------------
// Crop geometry
// ---------------------------------------------------------------------------

const MIN_CROP = 1;

/** Round a crop to whole image pixels and clamp it inside `natural`. */
export function clampCrop(crop, natural) {
  const NW = Math.max(1, Math.round(natural?.width || 1));
  const NH = Math.max(1, Math.round(natural?.height || 1));
  const num = (v, d) => (Number.isFinite(v) ? v : d);

  let w = Math.round(Math.max(MIN_CROP, Math.min(NW, num(crop?.width, NW))));
  let h = Math.round(Math.max(MIN_CROP, Math.min(NH, num(crop?.height, NH))));
  const x = Math.round(Math.max(0, Math.min(NW - w, num(crop?.x, 0))));
  const y = Math.round(Math.max(0, Math.min(NH - h, num(crop?.y, 0))));
  return { x, y, width: w, height: h };
}

export function fullCrop(natural) {
  return clampCrop({ x: 0, y: 0, width: natural?.width, height: natural?.height }, natural);
}

export function isFullCrop(crop, natural) {
  const f = fullCrop(natural);
  return crop.x === f.x && crop.y === f.y && crop.width === f.width && crop.height === f.height;
}

/**
 * Handles: 'move' plus the eight edge/corner grips named by compass point
 * ('nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w').
 *
 * `dx`/`dy` are the pointer delta in IMAGE pixels. Resizing pins the opposite
 * edge; dragging a grip past that edge collapses the rect to MIN_CROP rather
 * than flipping it inside out, which is the behaviour that keeps a fast drag
 * predictable.
 */
export function resizeCrop(handle, start, dx, dy, natural) {
  const s = clampCrop(start, natural);
  const NW = Math.max(1, Math.round(natural?.width || 1));
  const NH = Math.max(1, Math.round(natural?.height || 1));
  const ddx = Number.isFinite(dx) ? dx : 0;
  const ddy = Number.isFinite(dy) ? dy : 0;

  if (handle === 'move') {
    return clampCrop({ ...s, x: s.x + ddx, y: s.y + ddy }, natural);
  }

  let { x, y, width, height } = s;
  const right = s.x + s.width;
  const bottom = s.y + s.height;

  if (handle.includes('w')) {
    x = Math.max(0, Math.min(right - MIN_CROP, s.x + ddx));
    width = right - x;
  }
  if (handle.includes('e')) {
    width = Math.max(MIN_CROP, Math.min(NW - s.x, s.width + ddx));
  }
  if (handle.includes('n')) {
    y = Math.max(0, Math.min(bottom - MIN_CROP, s.y + ddy));
    height = bottom - y;
  }
  if (handle.includes('s')) {
    height = Math.max(MIN_CROP, Math.min(NH - s.y, s.height + ddy));
  }
  return clampCrop({ x, y, width, height }, natural);
}

// ---------------------------------------------------------------------------
// Output sizing
// ---------------------------------------------------------------------------

// Fractions included on purpose: a montage of every EDS map is already a few
// thousand pixels wide, so the useful direction there is DOWN, not up.
export const SCALE_FACTORS = [0.25, 0.5, 1, 2, 4, 8, 16];

// Browser canvas limits, measured rather than assumed (Chrome 2026-08-10):
// 16384x12288 = 201 MP encodes fine, 20000x15000 = 300 MP and 21120x17664 =
// 373 MP both make toBlob() hand back null — which surfaced as the useless
// "Encoding to image/png failed". Chrome's documented ceiling is 2^28 px of
// area; we stay a margin below it and also respect the 16384 per-side cap.
const MAX_OUTPUT_PX = 16384;
const MAX_OUTPUT_AREA = 240e6;

/**
 * Output pixel size for a crop.
 *
 * mode.type 'factor' multiplies the crop; 'width' targets an exact output width
 * and derives the height from the crop's aspect ratio. Both report the
 * `factor` actually achieved, and `limited` when the request had to be reduced
 * to something the browser can actually encode — the caller says so rather
 * than letting the export fail at the last step.
 */
export function outputSize(crop, mode) {
  const c = clampCrop(crop, { width: crop?.width, height: crop?.height });
  const aspect = c.height / c.width;

  let requested;
  if (mode?.type === 'width') {
    const t = Number(mode.targetWidth);
    requested = Number.isFinite(t) && t >= 1 ? Math.round(t) : c.width;
  } else {
    const f = Number(mode?.factor);
    requested = Math.round(c.width * (Number.isFinite(f) && f > 0 ? f : 1));
  }
  requested = Math.max(1, requested);

  let width = Math.min(MAX_OUTPUT_PX, requested);
  let height = Math.max(1, Math.round(width * aspect));
  if (height > MAX_OUTPUT_PX) {
    height = MAX_OUTPUT_PX;
    width = Math.max(1, Math.round(height / aspect));
  }
  if (width * height > MAX_OUTPUT_AREA) {
    // Shrink along the diagonal so the aspect ratio survives.
    const k = Math.sqrt(MAX_OUTPUT_AREA / (width * height));
    width = Math.max(1, Math.floor(width * k));
    height = Math.max(1, Math.floor(height * k));
  }
  return { width, height, factor: width / c.width, limited: width < requested };
}

/**
 * A sensible starting magnification for a given source.
 *
 * A flat 4x is right for a 120 px overview and absurd for an already-composed
 * multi-map sheet, so pick the largest sane factor that keeps the first offer
 * a manageable size. The user can still choose anything from the list.
 */
export function defaultFactor(natural) {
  const w = Math.max(1, natural?.width || 1);
  const candidates = SCALE_FACTORS.filter((f) => f <= 4);
  for (let i = candidates.length - 1; i >= 0; i -= 1) {
    if (w * candidates[i] <= 4000) return candidates[i];
  }
  return SCALE_FACTORS[0];
}

// ---------------------------------------------------------------------------
// Scalebar
// ---------------------------------------------------------------------------

const NICE_STEPS = [1, 2, 5];

/**
 * Choose a scalebar length that is a 1/2/5 x 10^n multiple and covers roughly
 * `targetFraction` of the visible width.
 *
 * Returns null when there is no physical scale to show. `lengthPx` is in IMAGE
 * pixels; the caller converts to output pixels.
 */
export function niceScalebar(unitsPerPixel, cropWidthPx, targetFraction = 0.25) {
  const upp = Number(unitsPerPixel);
  const w = Number(cropWidthPx);
  if (!Number.isFinite(upp) || upp <= 0 || !Number.isFinite(w) || w <= 0) return null;

  const rawUnits = upp * w * targetFraction;
  if (!(rawUnits > 0)) return null;

  const decade = Math.pow(10, Math.floor(Math.log10(rawUnits)));
  let best = NICE_STEPS[0] * decade;
  for (const s of NICE_STEPS) {
    const cand = s * decade;
    // `<=` so a tie goes to the LARGER rung: a target of exactly 1.5 decades
    // (which happens for the common 0.5 µm step) should give the longer, more
    // readable bar rather than the stubbier one.
    if (Math.abs(cand - rawUnits) <= Math.abs(best - rawUnits)) best = cand;
  }
  // A bar wider than the image is useless; step down the 1/2/5 ladder until it
  // fits. Each step shrinks `best` by at least 2x, so this terminates — the
  // counter is a backstop that gives up rather than hanging the render.
  for (let guard = 0; best / upp > w; guard += 1) {
    if (guard > 64 || !Number.isFinite(best) || best <= 0) return null;
    const dec = Math.pow(10, Math.floor(Math.log10(best)));
    // Nearest ladder rung at or below `best`, tolerant of float noise.
    const ratio = best / dec;
    const i = NICE_STEPS.findIndex((s) => s >= ratio - 1e-9);
    best = i > 0 ? NICE_STEPS[i - 1] * dec : 5 * dec / 10;
  }

  const decimals = best < 1 ? Math.min(3, Math.ceil(-Math.log10(best))) : 0;
  return { lengthUnits: best, lengthPx: best / upp, text: best.toFixed(decimals) };
}

// ---------------------------------------------------------------------------
// Filenames
// ---------------------------------------------------------------------------

// Reserved on Windows (a superset of the POSIX set). Spaces and hyphens are
// deliberately NOT in here: dataset names are full of both, and mangling them
// makes the suggested filename unrecognisable.
// String.fromCharCode(92) is the path separator; writing it as a literal here
// would need escaping that has already been mangled once in this file.
const ILLEGAL_FILENAME_CHARS = new Set(
  ['<', '>', ':', '"', '/', '|', '?', '*', String.fromCharCode(92)],
);

const KNOWN_EXT = /\.(png|jpe?g|webp)$/i;

/** Replace filesystem-illegal characters so the name can be written as-is. */
export function sanitiseFilename(name) {
  // Built from a set rather than a character class: the illegal set contains
  // both the regex escape character and the quote characters, and expressing it
  // as a literal is a well-known way to silently drop one of them.
  const mapped = Array.from(String(name ?? ''))
    .map((ch) => (ILLEGAL_FILENAME_CHARS.has(ch) || ch.charCodeAt(0) < 32 ? '_' : ch))
    .join('');
  return mapped
    .replace(/\s+/g, ' ')
    .replace(/^[.\s]+/, '')
    .replace(/[.\s]+$/, '')
    .slice(0, 180)
    .trim() || 'export';
}

/** Swap (or append) the extension so it matches the chosen format. */
export function buildFilename(base, formatId) {
  const f = formatById(formatId);
  const stem = sanitiseFilename(String(base ?? '').replace(KNOWN_EXT, ''));
  return `${stem}.${f.ext}`;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Draw the cropped, scaled image onto a fresh off-screen canvas.
 *
 * `filter` is a CSS filter string (e.g. 'brightness(1.2)') applied while
 * drawing, which is how the "as displayed" mode reproduces the viewer's
 * brightness slider — that correction lives only in CSS and is not present in
 * the source PNG. Pass null for a raw export.
 *
 * Lossy formats cannot store transparency, so the canvas is filled with
 * `background` first when one is given; otherwise it stays transparent.
 */
export function renderExportCanvas({
  image, crop, output, smoothing = false, filter = null, background = null,
  margins = null, marginColor = '#000000',
}) {
  if (!image) throw new Error('renderExportCanvas: no image');
  // Go through canvasSizeWithMargins so the render uses the SAME clamped
  // margins the dialog showed — otherwise a border the UI already reduced
  // would still be drawn at full size and blow the encode limit.
  const total = canvasSizeWithMargins(output, margins);
  const m = total.margins;
  const canvas = document.createElement('canvas');
  canvas.width = total.width;
  canvas.height = total.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('renderExportCanvas: 2D context unavailable');

  // The added border is opaque even for PNG: it is deliberate empty space to
  // put a caption on, and transparent space would be invisible everywhere it
  // is meant to be used.
  if (m.left || m.right || m.top || m.bottom) {
    ctx.fillStyle = marginColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  if (background) {
    ctx.fillStyle = background;
    ctx.fillRect(m.left, m.top, output.width, output.height);
  }

  ctx.imageSmoothingEnabled = !!smoothing;
  if (smoothing) ctx.imageSmoothingQuality = 'high';
  if (filter) ctx.filter = filter;
  ctx.drawImage(
    image,
    crop.x, crop.y, crop.width, crop.height,
    m.left, m.top, output.width, output.height,
  );
  ctx.filter = 'none';
  return canvas;
}

export const NO_MARGINS = Object.freeze({ top: 0, right: 0, bottom: 0, left: 0 });

/**
 * What the border FIELD offers per side, as a fraction of the image. Half the
 * image is already a very wide band to add by hand.
 *
 * This is a suggestion for the person typing, NOT a limit on the file. A
 * border fitted to a scale body the user placed beside the map may legitimately
 * be several times the map: a three-phase IPF key is 2.75x as tall as it is
 * wide, and beside a wide, short map it needs about 1.4 map heights above and
 * below (reported 2026-09-07). Clamping that away is what sliced it.
 *
 * The only HARD ceilings are the browser's, enforced in `canvasSizeWithMargins`
 * (`MAX_OUTPUT_PX`, `MAX_OUTPUT_AREA`), which also reports when it had to trim.
 */
export const BORDER_INPUT_MAX_FRACTION = 0.5;

/**
 * The wider of two margin sets, side by side.
 *
 * Used at save time to guarantee that whatever the caller says the figure
 * NEEDS (its colour key, its colour bars) still has room in the file, even
 * when the border was fitted earlier and the body has grown since. The border
 * is re-fitted only when the SET of scale bodies changes — deliberately, so
 * dragging one does not resize the picture under the hand — which leaves the
 * gap this closes: a body made taller after that point was simply cut off by
 * the canvas edge (2026-09-03: a three-phase IPF key exported as its middle
 * key, sliced top and bottom).
 */
export function widerMargins(a, b) {
  const A = a || NO_MARGINS;
  const B = b || NO_MARGINS;
  const pick = (k) => Math.max(
    Number.isFinite(A[k]) ? A[k] : 0,
    Number.isFinite(B[k]) ? B[k] : 0,
  );
  return { top: pick('top'), right: pick('right'), bottom: pick('bottom'), left: pick('left') };
}

/**
 * Margins are a FRACTION of the image on that axis (top/bottom of the height,
 * left/right of the width), so they survive a change of export resolution:
 * "20% at the top" stays 20% whether you export at 1x or at 8x. Returns whole
 * output pixels.
 */
export function marginPixels(margins, output) {
  // No upper clamp: the border is as wide as it was asked to be, and
  // `canvasSizeWithMargins` trims only against the real encode limits — and
  // says so via `limited`. A fraction cap here quietly cut off whatever had
  // been placed out there.
  const f = (v) => Math.max(0, Number.isFinite(v) ? v : 0);
  const mg = margins || NO_MARGINS;
  const w = Math.max(1, output?.width || 1);
  const h = Math.max(1, output?.height || 1);
  return {
    top: Math.round(f(mg.top) * h),
    bottom: Math.round(f(mg.bottom) * h),
    left: Math.round(f(mg.left) * w),
    right: Math.round(f(mg.right) * w),
  };
}

/**
 * Total canvas once the margins are added, clamped to what the browser can
 * still encode.
 *
 * The image area alone is already capped by `outputSize`, but margins are added
 * ON TOP of it — so without this a large border pushed the canvas past the
 * encode limit and the export died. When the total does not fit, the margins
 * are scaled down together and `limited` says so; the image itself is never
 * silently shrunk.
 */
export function canvasSizeWithMargins(output, margins) {
  let m = marginPixels(margins, output);
  let width = output.width + m.left + m.right;
  let height = output.height + m.top + m.bottom;
  let limited = false;

  const shrink = (k) => {
    limited = true;
    m = {
      top: Math.floor(m.top * k), bottom: Math.floor(m.bottom * k),
      left: Math.floor(m.left * k), right: Math.floor(m.right * k),
    };
    width = output.width + m.left + m.right;
    height = output.height + m.top + m.bottom;
  };

  // Per-side cap first, then the area cap; both are hard browser limits.
  if (width > MAX_OUTPUT_PX || height > MAX_OUTPUT_PX) {
    const room = Math.min(
      (MAX_OUTPUT_PX - output.width) / Math.max(1, m.left + m.right),
      (MAX_OUTPUT_PX - output.height) / Math.max(1, m.top + m.bottom),
    );
    shrink(Math.max(0, Math.min(1, room)));
  }
  if (width * height > MAX_OUTPUT_AREA) {
    // Solve for the factor k that brings the area back under the cap.
    for (let i = 0; i < 40 && width * height > MAX_OUTPUT_AREA; i += 1) {
      shrink(Math.sqrt(MAX_OUTPUT_AREA / (width * height)) * 0.999);
      if (!m.top && !m.bottom && !m.left && !m.right) break;
    }
  }

  return { width, height, origin: { x: m.left, y: m.top }, margins: m, limited };
}

/**
 * Burn the enabled annotations into `ctx`, in output coordinates.
 *
 * `geom` carries the mapping: { crop, output, sx, sy } where sx/sy convert an
 * image-pixel offset inside the crop to output pixels.
 *
 * Every annotation is skipped silently when its own data is absent — the caller
 * decides what is available (the pattern has no scalebar or ROI), and a missing
 * spec means "not requested", not "broken".
 */
export function drawAnnotations(ctx, spec, geom) {
  if (!ctx || !spec || !geom) return;
  const { crop, output, sx, sy } = geom;
  // `output` is the WHOLE canvas including any added border, so the caption and
  // the scale bar can be placed in that border. Crosshair and ROI mark points
  // in the image, so they are offset by where the image starts.
  const origin = geom.origin || { x: 0, y: 0 };
  // Scale the furniture with the image so a 16x export is not covered in
  // hairlines, but keep it readable on a 1x export too.
  const unit = Math.max(1, Math.round(Math.min(output.width, output.height) / 200));
  const toX = (imgX) => origin.x + (imgX - crop.x) * sx;
  const toY = (imgY) => origin.y + (imgY - crop.y) * sy;

  if (spec.roi) {
    const { startRow, startCol, endRow, endCol } = spec.roi;
    const x = toX(startCol);
    const y = toY(startRow);
    const w = (endCol - startCol) * sx;
    const h = (endRow - startRow) * sy;
    ctx.save();
    ctx.strokeStyle = spec.roiColor || '#f1fa8c';
    ctx.lineWidth = 2 * unit;
    ctx.strokeRect(x, y, w, h);
    ctx.restore();
  }

  if (spec.crosshair) {
    const x = toX(spec.crosshair.col + 0.5);
    const y = toY(spec.crosshair.row + 0.5);
    const arm = 7 * unit;
    ctx.save();
    ctx.strokeStyle = spec.crosshairColor || '#ff5555';
    ctx.lineWidth = 2 * unit;
    ctx.beginPath();
    ctx.moveTo(x - arm, y); ctx.lineTo(x + arm, y);
    ctx.moveTo(x, y - arm); ctx.lineTo(x, y + arm);
    ctx.stroke();
    ctx.restore();
  }

  if (spec.scalebar) {
    const L = scalebarLayout(ctx, spec, geom);
    ctx.save();
    ctx.font = fontSpec(L.st, L.fontPx);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    if (L.st.backgroundOpacity > 0) {
      ctx.fillStyle = rgba(L.st.background, L.st.backgroundOpacity);
      ctx.fillRect(L.x, L.y, L.w, L.h);
    }
    ctx.fillStyle = L.st.color;
    const barX = L.x + (L.w - L.barW) / 2;
    const barY = L.y + L.h - L.pad / 2 - L.barH;
    ctx.fillRect(barX, barY, L.barW, L.barH);
    ctx.fillText(L.label, L.x + L.w / 2, barY - Math.max(1, unit));
    ctx.restore();
  }

  if (spec.label) {
    const L = captionLayout(ctx, spec, geom);
    ctx.save();
    ctx.font = fontSpec(L.st, L.fontPx);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    if (L.st.backgroundOpacity > 0) {
      ctx.fillStyle = rgba(L.st.background, L.st.backgroundOpacity);
      ctx.fillRect(L.x, L.y, L.w, L.h);
    }
    ctx.fillStyle = L.st.color;
    L.lines.forEach((line, i) => {
      ctx.fillText(line, L.x + L.pad / 2, L.y + L.pad / 4 + i * L.lineH);
    });
    ctx.restore();
  }
}

// ---------------------------------------------------------------------------
// Annotation styling and layout
// ---------------------------------------------------------------------------

// Position is a free fraction of the output, not a fixed corner: the user drags
// these blocks around on the preview. `widthFrac` is the caption's wrap width.
export const DEFAULT_CAPTION_STYLE = Object.freeze({
  fontScale: 1,
  bold: true,
  color: '#ffffff',
  background: '#000000',
  backgroundOpacity: 0.55,   // 0 = no plate at all
  x: 0.02,
  y: 0.02,
  widthFrac: 0.55,
});

export const DEFAULT_SCALEBAR_STYLE = Object.freeze({
  fontScale: 1,
  barScale: 1,               // bar thickness
  bold: true,
  color: '#ffffff',
  background: '#000000',
  backgroundOpacity: 0.55,
  x: 0.68,
  y: 0.86,
  lengthUnits: null,         // null = pick automatically from the crop width
});

function fontSpec(st, fontPx) {
  return `${st.bold ? 600 : 400} ${fontPx}px sans-serif`;
}

function unitOf(output) {
  return Math.max(1, Math.round(Math.min(output.width, output.height) / 200));
}

const clampTo = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/**
 * Geometry of the caption block, in output pixels.
 *
 * Shared by the renderer and by the dialog's drag overlay, so the box you grab
 * on screen is exactly the box that gets drawn. Mutates ctx.font as a side
 * effect of measuring — every caller sets its own font before drawing anyway.
 */
export function captionLayout(ctx, spec, geom) {
  const st = { ...DEFAULT_CAPTION_STYLE, ...(spec.labelStyle || {}) };
  const { output } = geom;
  const unit = unitOf(output);
  const fontPx = Math.max(6, 11 * unit * st.fontScale);
  const pad = 8 * unit;
  ctx.font = fontSpec(st, fontPx);

  const maxTextW = Math.max(fontPx, st.widthFrac * output.width - pad);
  const lines = wrapText(ctx, spec.label, maxTextW);
  const textW = lines.reduce((m, l) => Math.max(m, ctx.measureText(l).width), 0);
  const lineH = fontPx * 1.25;

  const w = Math.min(output.width, textW + pad);
  // Height = the text itself plus symmetric padding. Using lines.length * lineH
  // added the line SPACING below the last line too, which left a visibly empty
  // strip under the text and made the block look far too tall.
  const h = Math.min(output.height, (lines.length - 1) * lineH + fontPx + pad / 2);
  return {
    st, lines, fontPx, pad, lineH, w, h,
    x: clampTo(st.x * output.width, 0, Math.max(0, output.width - w)),
    y: clampTo(st.y * output.height, 0, Math.max(0, output.height - h)),
  };
}

/** Geometry of the scale bar block, in output pixels. */
export function scalebarLayout(ctx, spec, geom) {
  const st = { ...DEFAULT_SCALEBAR_STYLE, ...(spec.scalebarStyle || {}) };
  const { output, sx } = geom;
  const unit = unitOf(output);
  const fontPx = Math.max(6, 11 * unit * st.fontScale);
  const pad = 8 * unit;
  const barW = spec.scalebar.lengthPx * sx;
  const barH = Math.max(1, 3 * unit * st.barScale);
  const label = `${spec.scalebar.text} ${spec.scalebar.unitLabel}`;
  ctx.font = fontSpec(st, fontPx);

  const w = Math.min(output.width, Math.max(barW, ctx.measureText(label).width) + pad);
  const h = Math.min(output.height, barH + fontPx + pad);
  return {
    st, label, fontPx, pad, barW, barH, w, h,
    x: clampTo(st.x * output.width, 0, Math.max(0, output.width - w)),
    y: clampTo(st.y * output.height, 0, Math.max(0, output.height - h)),
  };
}

/**
 * Break `text` into lines that each fit `maxWidth` under the context's current
 * font. Words are kept whole where possible; a single word too long for the
 * line is split rather than allowed to overflow.
 */
export function wrapText(ctx, text, maxWidth) {
  if (!(maxWidth > 0)) return [];
  const words = String(text ?? '').split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';

  const hardSplit = (word) => {
    const parts = [];
    let chunk = '';
    for (const ch of word) {
      if (chunk && ctx.measureText(chunk + ch).width > maxWidth) { parts.push(chunk); chunk = ch; }
      else chunk += ch;
    }
    if (chunk) parts.push(chunk);
    return parts;
  };

  for (const word of words) {
    const cand = line ? `${line} ${word}` : word;
    if (ctx.measureText(cand).width <= maxWidth) { line = cand; continue; }
    if (line) { lines.push(line); line = ''; }
    if (ctx.measureText(word).width <= maxWidth) { line = word; continue; }
    const parts = hardSplit(word);
    lines.push(...parts.slice(0, -1));
    line = parts[parts.length - 1] || '';
  }
  if (line) lines.push(line);
  return lines;
}

/**
 * Selectable scale bar lengths for the current crop: the 1/2/5 ladder, limited
 * to bars that actually fit and are not so short they vanish.
 */
export function scalebarOptions(unitsPerPixel, cropWidthPx) {
  const upp = Number(unitsPerPixel);
  const w = Number(cropWidthPx);
  if (!Number.isFinite(upp) || upp <= 0 || !Number.isFinite(w) || w <= 0) return [];
  const span = upp * w;
  const out = [];
  for (let e = -6; e <= 9; e += 1) {
    for (const s of [1, 2, 5]) {
      const v = s * Math.pow(10, e);
      if (v <= span && v >= span / 100) out.push(v);
    }
  }
  return out.sort((a, b) => a - b);
}

/** Format a bar length for a menu entry / the drawn label. */
export function formatUnits(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return '';
  const decimals = v < 1 ? Math.min(3, Math.ceil(-Math.log10(v))) : 0;
  return v.toFixed(decimals);
}

/**
 * '#rrggbb' + alpha -> 'rgba(r,g,b,a)'.
 *
 * Canvas has no way to set a fill colour and an alpha separately without
 * touching globalAlpha, which would also fade whatever is drawn next.
 */
export function rgba(hex, alpha) {
  const a = Math.min(1, Math.max(0, Number.isFinite(alpha) ? alpha : 1));
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex ?? ''));
  if (!m) return `rgba(0,0,0,${a})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/**
 * Shorten `text` with a trailing ellipsis until it fits `maxWidth` under the
 * context's current font. Binary search so a long caption costs ~log2(n)
 * measureText calls rather than one per character.
 *
 * Exported for testing; the annotation drawing is the only caller.
 */
export function ellipsise(ctx, text, maxWidth) {
  const s = String(text ?? '');
  if (!(maxWidth > 0)) return '';
  if (ctx.measureText(s).width <= maxWidth) return s;

  const ell = '…';
  if (ctx.measureText(ell).width > maxWidth) return '';

  let lo = 0;             // always fits (with the ellipsis)
  let hi = s.length;      // never fits
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ctx.measureText(s.slice(0, mid) + ell).width <= maxWidth) lo = mid;
    else hi = mid;
  }
  return s.slice(0, lo) + ell;
}

export function canvasToBlob(canvas, mime, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) { resolve(blob); return; }
        // The overwhelmingly common cause is an output the browser refuses to
        // encode, so say the size out loud instead of just "failed".
        const mp = (canvas.width * canvas.height / 1e6).toFixed(0);
        reject(new Error(
          `Could not encode ${canvas.width} × ${canvas.height} px (${mp} MP) as ${mime}. `
          + 'Choose a smaller resolution or crop a smaller area.',
        ));
      },
      mime,
      quality,
    );
  });
}

/** Load a base64 payload into an <img> ready for drawImage. */
export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not decode the image data'));
    img.src = src;
  });
}

/**
 * The region of an image that is not just its background, or null when there is
 * nothing to trim.
 *
 * A 3D view is captured at the size of its viewport, and the object usually sits
 * in the middle of a lot of empty backdrop — a master-pattern sphere measured
 * 3456 × 720 with the sphere covering a fraction of it. Opening the export on
 * the whole frame would repeat the "why is it so small" problem.
 *
 * The backdrop colour is read from the corners rather than assumed to be black,
 * because these viewers clear to the theme colour. `fullIfAbove` avoids
 * returning a near-identical box when the image is already tight.
 */
export async function trimUniformBackground(src, { tolerance = 16, fullIfAbove = 0.98 } = {}) {
  const img = await loadImage(src);
  const w = img.naturalWidth | 0;
  const h = img.naturalHeight | 0;
  if (!w || !h) return null;

  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, w, h).data;

  const at = (x, y) => (y * w + x) * 4;
  // Median of the four corners: one corner could carry a stray annotation.
  const corners = [at(0, 0), at(w - 1, 0), at(0, h - 1), at(w - 1, h - 1)];
  const mid = (i) => {
    const v = corners.map((o) => d[o + i]).sort((a, b) => a - b);
    return (v[1] + v[2]) / 2;
  };
  const br = mid(0);
  const bg = mid(1);
  const bb = mid(2);

  let minX = w;
  let minY = h;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = at(x, y);
      if (d[i + 3] < 8) continue;
      if (Math.abs(d[i] - br) <= tolerance
        && Math.abs(d[i + 1] - bg) <= tolerance
        && Math.abs(d[i + 2] - bb) <= tolerance) continue;
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

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

/** Plain browser download — the fallback when Electron's writer is absent. */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
}

/**
 * ArrayBuffer -> base64, chunked.
 *
 * String.fromCharCode(...bytes) on a whole image blows the argument limit and
 * throws for anything past a few hundred kB, which is exactly the size range a
 * 16x export lands in.
 */
export function bufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const CHUNK = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/**
 * Write the blob wherever the user wants it.
 *
 * With Electron present this opens a real save dialog and writes the bytes;
 * otherwise (browser start, or an Electron build predating the IPC handler) it
 * falls back to an ordinary download. Both are working paths, so the caller is
 * told which one ran rather than being left to guess.
 *
 * Returns null when the user cancels the save dialog.
 */
export async function saveImageBlob(blob, filename, formatId) {
  const api = typeof window !== 'undefined' ? window.electronAPI : null;
  if (api?.saveImage) {
    const base64 = bufferToBase64(await blob.arrayBuffer());
    const f = formatById(formatId);
    const path = await api.saveImage({
      defaultPath: filename,
      base64,
      filters: [{ name: `${f.id.toUpperCase()} image`, extensions: [f.ext] }],
    });
    return path ? { via: 'electron', path } : null;
  }
  downloadBlob(blob, filename);
  return { via: 'download', path: filename };
}

// ---------------------------------------------------------------------------
// Dragging annotations on the preview
// ---------------------------------------------------------------------------

const clamp01 = (v) => Math.min(1, Math.max(0, Number.isFinite(v) ? v : 0));
const clampScale = (v) => Math.min(8, Math.max(0.2, Number.isFinite(v) ? v : 1));
const clampFrac = (v) => Math.min(1, Math.max(0.05, Number.isFinite(v) ? v : 0.5));

// A drag delta of a quarter of the image doubles the size. Chosen so the
// handles feel responsive without a small twitch rescaling everything.
const SIZE_GAIN = 4;

/** Move an annotation. Deltas are fractions of the output box. */
export function moveAnnotation(style, dxFrac, dyFrac) {
  return {
    ...style,
    x: clamp01((style.x ?? 0) + (Number.isFinite(dxFrac) ? dxFrac : 0)),
    y: clamp01((style.y ?? 0) + (Number.isFinite(dyFrac) ? dyFrac : 0)),
  };
}

/**
 * Resize the caption by dragging a grip.
 *
 * Horizontal grips change the wrap width (so the text reflows), vertical grips
 * change the font size, corners do both. West/north grips also shift the origin
 * so the opposite edge stays where it was.
 */
export function resizeCaptionStyle(style, handle, dxFrac, dyFrac) {
  const dx = Number.isFinite(dxFrac) ? dxFrac : 0;
  const dy = Number.isFinite(dyFrac) ? dyFrac : 0;
  const st = { ...style };
  if (handle.includes('e')) st.widthFrac = clampFrac((style.widthFrac ?? 0.55) + dx);
  if (handle.includes('w')) {
    st.widthFrac = clampFrac((style.widthFrac ?? 0.55) - dx);
    st.x = clamp01((style.x ?? 0) + dx);
  }
  if (handle.includes('s')) st.fontScale = clampScale((style.fontScale ?? 1) * (1 + dy * SIZE_GAIN));
  if (handle.includes('n')) {
    st.fontScale = clampScale((style.fontScale ?? 1) * (1 - dy * SIZE_GAIN));
    st.y = clamp01((style.y ?? 0) + dy);
  }
  return st;
}

/**
 * Resize the scale bar by dragging a grip. The bar's LENGTH is physical and
 * fixed by the chosen units, so dragging scales the lettering and the bar
 * thickness together instead.
 */
export function resizeScalebarStyle(style, handle, dxFrac, dyFrac) {
  const dx = Number.isFinite(dxFrac) ? dxFrac : 0;
  const dy = Number.isFinite(dyFrac) ? dyFrac : 0;
  const st = { ...style };
  let d = 0;
  if (handle.includes('e')) d += dx;
  if (handle.includes('w')) { d -= dx; st.x = clamp01((style.x ?? 0) + dx); }
  if (handle.includes('s')) d += dy;
  if (handle.includes('n')) { d -= dy; st.y = clamp01((style.y ?? 0) + dy); }
  const f = 1 + d * SIZE_GAIN;
  st.fontScale = clampScale((style.fontScale ?? 1) * f);
  st.barScale = clampScale((style.barScale ?? 1) * f);
  return st;
}

// ---------------------------------------------------------------------------
// Panel sheets
// ---------------------------------------------------------------------------

/**
 * Lay several unrelated images out on one labelled sheet.
 *
 * Unlike the EDS montage these panels do NOT share a pixel grid — an NCC map,
 * an experimental pattern and a simulated one all have their own aspect — so
 * each is fitted (contain) into its cell instead of being stretched to a
 * common shape.
 *
 * panels: [{ label, src }] — entries without a src are skipped.
 */
export async function buildPanelSheet(panels, {
  cellWidth = 460, gap = 12, background = '#000000', labelColor = '#ffffff', columns = null,
} = {}) {
  const loaded = [];
  for (const p of panels || []) {
    if (!p?.src) continue;
    loaded.push({ label: p.label ?? '', img: await loadImage(p.src) });
  }
  if (!loaded.length) throw new Error('Nothing to export: none of the panels has an image');

  const cols = Math.max(1, columns ?? Math.min(4, loaded.length));
  const rows = Math.ceil(loaded.length / cols);
  const cellW = Math.max(64, Math.round(cellWidth));
  // Tallest aspect wins so no panel gets cropped; the others letterbox.
  const aspect = Math.max(...loaded.map((l) => (l.img.naturalHeight || 1) / (l.img.naturalWidth || 1)));
  const cellH = Math.max(48, Math.round(cellW * aspect));
  const labelH = Math.max(16, Math.round(cellH * 0.09));
  const fontPx = Math.max(11, Math.round(labelH * 0.72));

  const canvas = document.createElement('canvas');
  canvas.width = cols * cellW + (cols + 1) * gap;
  canvas.height = rows * (cellH + labelH) + (rows + 1) * gap;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  loaded.forEach((p, i) => {
    const x = gap + (i % cols) * (cellW + gap);
    const y = gap + Math.floor(i / cols) * (cellH + labelH + gap);

    ctx.save();
    ctx.font = `600 ${fontPx}px sans-serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = labelColor;
    ctx.fillText(p.label, x + 2, y + labelH / 2, cellW - 4);
    ctx.restore();

    const iw = p.img.naturalWidth || 1;
    const ih = p.img.naturalHeight || 1;
    const s = Math.min(cellW / iw, cellH / ih);
    const w = iw * s;
    const h = ih * s;
    ctx.drawImage(p.img, x + (cellW - w) / 2, y + labelH + (cellH - h) / 2, w, h);
  });

  return canvas;
}
