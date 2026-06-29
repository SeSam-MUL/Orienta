// Pure geometry helpers for the linked pattern crosshair + numbered markers.
// No React, no DOM access — unit-testable in the node environment.
//
// All "norm" coordinates are normalized [0..1] within the IMAGE CONTENT box
// (i.e. the actual pixels of an objectFit:contain image, excluding the
// letterbox bars). A norm coord therefore maps to the same detector position
// across the experimental / simulated / NCC-difference panels, which share one
// detector frame.

/** Content rect of an objectFit:contain image inside its box (px). */
export function contentBox({ naturalW, naturalH, boxW, boxH }) {
  if (!naturalW || !naturalH || !boxW || !boxH) return { x: 0, y: 0, w: boxW || 0, h: boxH || 0 };
  const scale = Math.min(boxW / naturalW, boxH / naturalH);
  const w = naturalW * scale;
  const h = naturalH * scale;
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h };
}

/** Pointer (clientX/Y) -> normalized [0..1] within the image content box, or null. */
export function pointerToNorm({ clientX, clientY }, rect, natural) {
  const cb = contentBox({ naturalW: natural.w, naturalH: natural.h, boxW: rect.width, boxH: rect.height });
  if (cb.w <= 0 || cb.h <= 0) return null; // not laid out yet — avoid divide-by-zero
  const px = clientX - rect.left - cb.x;
  const py = clientY - rect.top - cb.y;
  if (px < 0 || py < 0 || px > cb.w || py > cb.h) return null;
  return { x: px / cb.w, y: py / cb.h };
}

/** Normalized content coord -> percent-of-BOX (for absolute CSS overlay placement). */
export function normToBoxPct(norm, rect, natural) {
  const cb = contentBox({ naturalW: natural.w, naturalH: natural.h, boxW: rect.width, boxH: rect.height });
  return {
    left: ((cb.x + norm.x * cb.w) / rect.width) * 100,
    top: ((cb.y + norm.y * cb.h) / rect.height) * 100,
  };
}

/** Nearest marker within radiusNorm (Euclidean), else null. */
export function hitTestMarker(markers, norm, radiusNorm) {
  let best = null, bestD = radiusNorm * radiusNorm;
  for (const m of markers) {
    const dx = m.x - norm.x, dy = m.y - norm.y;
    const d = dx * dx + dy * dy;
    if (d <= bestD) { bestD = d; best = m.id; }
  }
  return best;
}

/** Reassign contiguous 1..N numbering in array order. */
export function renumber(markers) {
  return markers.map((m, i) => ({ ...m, n: i + 1 }));
}
