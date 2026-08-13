/**
 * Which layers decide the frame the map is drawn in.
 *
 * Index a region and the result covers only that region; every other layer —
 * an EDS element map, band contrast, an electron image — still comes at the
 * full scan size, because that is how those maps exist. Taking the frame from
 * everything on the canvas therefore made the picture jump back to the whole
 * scan the moment such a layer was added, leaving the indexed region as a small
 * bright square in the middle of data nobody asked to see.
 *
 * The region is a property of the RESULT, so only result-shaped layers get a
 * vote. The others are then cropped to it, which is what "index this area"
 * means for a figure.
 */

// Layers that live on the result's own grid and carry its mask: the indexing
// maps themselves, the analysis maps derived from them, and the per-pixel
// diagnostics. Everything else (h5 source images, EDS elements, virtual BSE)
// covers the full scan whatever the result did.
const ROI_SOURCES = new Set(['result', 'analysis', 'diagnostics', 'refinement']);

export function definesRoi(layer) {
  return !!layer && layer.visible !== false && ROI_SOURCES.has(layer.source);
}

/**
 * The layers to measure the frame from.
 *
 * Returns [] when none is visible — the caller then keeps the frame it had
 * rather than snapping to the full scan, so hiding the phase map for a moment
 * does not resize everything under the user's hands.
 */
export function roiDefiningLayers(layers) {
  return (layers || []).filter(definesRoi);
}
