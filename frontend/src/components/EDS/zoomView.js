/**
 * Zoom-view math for the EDS maps (composite overlay + "All Maps" tile grid).
 *
 * A view is `{ scale, cx, cy }` — magnification (>= 1) plus the NORMALISED
 * centre of the visible content in [0,1]^2 of the map.
 *
 * Normalised on purpose: the tiles carry different native resolutions (EDS and
 * BC on the EBSD scan grid, electron images at the higher SEM survey
 * resolution) and the composite overlay has a different box size again. A pan
 * stored in image pixels or CSS pixels would drift apart the moment two views
 * are synced; fractions of the map are the only frame all of them share.
 *
 * Rendering applies the view as a CSS transform (see `viewToTransform`), so no
 * canvas is ever re-drawn for a zoom and the shared LayeredCanvas stays
 * untouched. Pointer mapping goes through `zoomedRect`, which hands the
 * existing letterbox math in mapCoords.js a virtual rect — that keeps
 * crosshair sync, click-to-quantify, ROI and linescan pixel-exact while zoomed.
 */

export const MIN_SCALE = 1;
export const MAX_SCALE = 16;

/** Multiplicative step per wheel notch. */
export const WHEEL_STEP = 1.25;

export const IDENTITY_VIEW = Object.freeze({ scale: 1, cx: 0.5, cy: 0.5 });

/** True once the view is meaningfully zoomed in (guards float noise). */
export function isZoomed(view) {
  return !!view && view.scale > 1.001;
}

/**
 * Clamp a view into the legal range: scale in [MIN_SCALE, MAX_SCALE] and a
 * centre that keeps the content covering the whole box (no empty margin, so a
 * map can never be dragged off-screen). At scale 1 the centre is pinned to
 * (0.5, 0.5), which makes `clampView` the reset path as well.
 */
export function clampView(view) {
  const raw = view || IDENTITY_VIEW;
  const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, Number.isFinite(raw.scale) ? raw.scale : 1));
  const half = 0.5 / scale;
  const axis = (v) => {
    const n = Number.isFinite(v) ? v : 0.5;
    return Math.min(1 - half, Math.max(half, n));
  };
  return { scale, cx: axis(raw.cx), cy: axis(raw.cy) };
}

/**
 * The rectangle the zoomed content occupies, in the same coordinate space as
 * `rect` (a DOMRect from getBoundingClientRect). Feed this into
 * pointerToRowCol / rowColToContainerPx instead of the raw rect and every
 * existing mapping becomes zoom-aware without changing that math.
 *
 * Derivation: the content is scaled by `s` about the box centre and shifted so
 * that the content fraction (cx, cy) lands exactly on the box centre.
 */
export function zoomedRect(rect, view) {
  const v = clampView(view);
  const width = rect.width * v.scale;
  const height = rect.height * v.scale;
  const left = rect.left + rect.width / 2 - v.cx * width;
  const top = rect.top + rect.height / 2 - v.cy * height;
  return { left, top, width, height, right: left + width, bottom: top + height };
}

/**
 * CSS transform for a child element that exactly fills the host box.
 * Percentages in `translate` refer to the element's own size, which equals the
 * host size here — so no rect measurement is needed to render.
 *
 * `scale(s) translate(t)` applies the translate first: a content point at
 * fraction f sits at screen x = centre + s*((f - 0.5)*W + t). With
 * t = (0.5 - cx)*W the fraction cx lands on the centre, as `zoomedRect` assumes.
 */
export function viewToTransform(view) {
  const v = clampView(view);
  if (v.scale === 1 && v.cx === 0.5 && v.cy === 0.5) return 'none';
  const tx = (0.5 - v.cx) * 100;
  const ty = (0.5 - v.cy) * 100;
  return `scale(${v.scale}) translate(${tx}%, ${ty}%)`;
}

/**
 * Zoom by `factor` about a pointer sitting at fraction (px, py) of the host
 * box. The content under the pointer stays under the pointer.
 */
export function zoomAt(view, factor, px = 0.5, py = 0.5) {
  const v = clampView(view);
  const nextScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * factor));
  // Content fraction currently under the pointer.
  const fx = v.cx + (px - 0.5) / v.scale;
  const fy = v.cy + (py - 0.5) / v.scale;
  return clampView({
    scale: nextScale,
    cx: fx - (px - 0.5) / nextScale,
    cy: fy - (py - 0.5) / nextScale,
  });
}

/**
 * Pan by a pointer delta given in fractions of the host box (dx = movement in
 * px / box width). Dragging right moves the content right, i.e. the visible
 * centre moves left.
 */
export function panBy(view, dxFrac, dyFrac) {
  const v = clampView(view);
  return clampView({
    scale: v.scale,
    cx: v.cx - (Number.isFinite(dxFrac) ? dxFrac : 0) / v.scale,
    cy: v.cy - (Number.isFinite(dyFrac) ? dyFrac : 0) / v.scale,
  });
}

/** Wheel delta → zoom factor (up/away from user = zoom in). */
export function wheelFactor(deltaY) {
  return deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP;
}
