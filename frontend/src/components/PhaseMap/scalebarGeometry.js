/**
 * How long the on-screen scalebar has to be, in CSS pixels.
 *
 * This used to live inline in LayeredCanvas as a CSS percentage, which was
 * wrong twice over:
 *
 *   1. The percentage was applied to a bar inside a shrink-wrapped flex column,
 *      so it resolved against the LABEL's width instead of the map's. Measured
 *      on a real result: a "5 µm" bar came out 5.1 px where it should have been
 *      240 px, because 5/29 of the 30 px wide label box is 5.1 px.
 *   2. It ignored the auto-zoom. LayeredCanvas magnifies the indexed bounding
 *      box to fill the canvas, so a map pixel is bigger on screen than
 *      `wrapperWidth / nativeWidth` suggests.
 *
 * Working in pixels removes the containing-block ambiguity entirely, and the
 * bbox term makes the bar follow what is actually displayed. The export path
 * never had either problem — it draws into the exported bitmap, where one image
 * pixel is one map pixel — which is why the two disagreed.
 */

import { bboxContentRect } from '../EDS/mapCoords';

/** Round a physical length to a "nice" number: 1, 2 or 5 × 10^n. */
export function niceLength(rawMicrons) {
  if (!Number.isFinite(rawMicrons) || rawMicrons <= 0) return null;
  const exponent = Math.floor(Math.log10(rawMicrons));
  const base = 10 ** exponent;
  const norm = rawMicrons / base;
  let pick;
  if (norm < 1.5) pick = 1;
  else if (norm < 3.5) pick = 2;
  else if (norm < 7.5) pick = 5;
  else pick = 10;
  return pick * base;
}

const LADDER = [10, 5, 2, 1];

/**
 * A nice length that also fits within `cap` µm.
 *
 * Rounding up (norm ≥ 7.5 → 10) can overshoot, and at a length fraction near 1
 * that would draw a bar wider than the map it is measuring. Step down the
 * 1/2/5 ladder until it fits.
 */
export function niceLengthCapped(rawMicrons, cap) {
  let v = niceLength(rawMicrons);
  if (!v) return null;
  if (!Number.isFinite(cap) || cap <= 0) return v;
  for (let guard = 0; v > cap && guard < 40; guard += 1) {
    const exponent = Math.floor(Math.log10(v) + 1e-9);
    const base = 10 ** exponent;
    const idx = LADDER.indexOf(Math.round(v / base));
    v = idx >= 0 && idx < LADDER.length - 1 ? LADDER[idx + 1] * base : base / 2;
  }
  return v > 0 ? v : null;
}

export function formatMicrons(um) {
  if (um >= 1000) return `${(um / 1000).toFixed(um >= 10000 ? 0 : 1)} mm`;
  if (um >= 1) return `${um % 1 === 0 ? um : um.toFixed(1)} µm`;
  return `${(um * 1000).toFixed(um >= 0.1 ? 0 : 1)} nm`;
}

/**
 * Bar length in CSS pixels plus its label.
 *
 * @param nativeSize  {w,h} of the canvas buffer, in map pixels
 * @param contentBbox auto-zoom box in map pixels, or null when the whole grid
 *                    is shown
 * @param stepX       µm per map pixel
 * @param lengthFrac  requested bar length as a fraction of the displayed width
 * @param fitWidth    the wrapper's width in CSS pixels (the buffer's full width
 *                    on screen)
 * @param zoomScale   user zoom on top of that (1 = not zoomed). The map is
 *                    magnified by a CSS transform, so a µm covers that many
 *                    times more screen pixels — and that many times less of the
 *                    map is visible, which is what caps the length.
 * @returns {{lengthUm, label, barPx}} or null when nothing can be drawn
 */
export function umToScreenPx({
  um, nativeSize, contentBbox, stepX, fitWidth, zoomScale = 1,
}) {
  if (!(um > 0) || !nativeSize?.w || !(stepX > 0) || !(fitWidth > 0)) return null;
  const zoom = Number.isFinite(zoomScale) && zoomScale > 0 ? zoomScale : 1;

  // Columns of the map that are visible, and how many buffer columns they are
  // drawn across. Without a bbox both are the full buffer width, so the
  // magnification term is exactly 1.
  const dst = contentBbox ? bboxContentRect(nativeSize.w, nativeSize.h, contentBbox) : null;
  const visibleCols = contentBbox?.w || nativeSize.w;
  const bufferCols = dst?.w || nativeSize.w;
  if (!(visibleCols > 0) || !(bufferCols > 0)) return null;

  // map px -> buffer px -> CSS px -> zoomed CSS px
  const px = (um / stepX) * (bufferCols / visibleCols) * (fitWidth / nativeSize.w) * zoom;
  return (px > 0 && Number.isFinite(px)) ? px : null;
}

/** How much of the map is on screen right now, in µm across. */
export function visibleWidthUm({ nativeSize, contentBbox, stepX, zoomScale = 1 }) {
  if (!nativeSize?.w || !(stepX > 0)) return null;
  const zoom = Number.isFinite(zoomScale) && zoomScale > 0 ? zoomScale : 1;
  const visibleCols = contentBbox?.w || nativeSize.w;
  return (visibleCols * stepX) / zoom;
}

export function scalebarGeometry({
  nativeSize, contentBbox, stepX, lengthFrac, fitWidth, zoomScale = 1,
}) {
  const visibleUm = visibleWidthUm({ nativeSize, contentBbox, stepX, zoomScale });
  if (!visibleUm) return null;

  // Zooming in shows less of the map, so both the requested length and its cap
  // shrink with it — otherwise the bar would run off the visible area.
  const frac = Number.isFinite(lengthFrac) && lengthFrac > 0 ? lengthFrac : 0.2;
  const lengthUm = niceLengthCapped(visibleUm * frac, visibleUm);
  if (!lengthUm) return null;

  const barPx = umToScreenPx({
    um: lengthUm, nativeSize, contentBbox, stepX, fitWidth, zoomScale,
  });
  if (!barPx) return null;

  return { lengthUm, label: formatMicrons(lengthUm), barPx };
}
