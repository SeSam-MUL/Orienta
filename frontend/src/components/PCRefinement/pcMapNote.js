// What the PC Refinement page tells the user about a per-pixel PC map.
//
// A grid calibration leaves a map with one PC per scan pixel. The page's fields
// show ONE PC, so the page has to say that they show its mean, and what a PC
// typed into them or a Global PC Refine did to the map.

// The map the backend reports on /api/pc/status ({pc_per_pixel, pc_map_shape}),
// or null when the detector has one PC.
export function pcMapInfoOf(status) {
  if (!status?.pc_per_pixel) return null;
  const shape = Array.isArray(status.pc_map_shape) ? status.pc_map_shape : [];
  if (shape.length >= 2) return { rows: shape[0], cols: shape[1] };
  // a flat list of PCs has no rows and columns, and none are made up for it
  return shape.length === 1 ? { count: shape[0] } : { rows: '?', cols: '?' };
}

// The pc_map_action of a PC write ("shifted" | "replaced" | "none"): only the two
// that changed a map are worth a sentence.
export function pcMapActionOf(value) {
  return value === 'shifted' || value === 'replaced' ? value : null;
}

// The i18n key (under pcrefinement:pixelwise) and params of the one sentence to
// show: what just happened to the map wins over the standing description.
export function pcMapNote(info, action) {
  if (action === 'shifted') return { key: 'mapShifted', params: {} };
  if (action === 'replaced') return { key: 'mapReplaced', params: {} };
  if (info?.count != null) return { key: 'mapActiveFlat', params: { count: info.count } };
  if (info) return { key: 'mapActive', params: { rows: info.rows, cols: info.cols } };
  return null;
}
