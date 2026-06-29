/**
 * EDS-specific layer source catalogue + default-layer seeder.
 *
 * Kept independent from PhaseMap/layerSources.js so the two pages can evolve
 * separately. Pure data: no React, no IO, no async.
 */

/**
 * From a list of electron-image names (e.g. ["FSE/Oben links", "SE/Elektronenbild 1"]),
 * pick the one most likely to be a real SE image. Preference order:
 *   1. Path starting with "SE/" or "SE-" (real Secondary Electron).
 *   2. Path starting with "BSE/" or "BSE-" (Backscatter Electron, also useful).
 *   3. Path starting with "FSE/" or "FSE-" (Forward-Scatter detector — last resort).
 *   4. Any other electron image (first in list).
 * Returns null when the list is empty.
 */
export function pickPreferredElectronImage(electronImages) {
  if (!Array.isArray(electronImages) || electronImages.length === 0) return null;
  const prefixMatches = (prefix) =>
    electronImages.find((n) => typeof n === 'string' &&
      (n.startsWith(`${prefix}/`) || n.startsWith(`${prefix}-`)));
  return prefixMatches('SE') || prefixMatches('BSE') || prefixMatches('FSE') || electronImages[0];
}

export function buildEdsLayerSources({ elements = [], electronImages = [] } = {}) {
  const sources = [];
  for (const el of elements) {
    sources.push({
      id: `eds-${el}`,
      label: el,
      kind: 'eds-element',
      element: el,
      defaultBlend: 'screen',
      defaultOpacity: 0.7,
    });
  }
  for (const name of electronImages) {
    sources.push({
      id: `electron-${name}`,
      label: `SE Image (${name})`,
      kind: 'electron',
      electronName: name,
      defaultBlend: 'normal',
      defaultOpacity: 0.6,
    });
  }
  return sources;
}

export function defaultLayersFor({
  hasSE = false,
  hasBC = false,
  electronImageName = null,
  elements = [],
} = {}) {
  const out = [];
  if (hasSE && electronImageName) {
    out.push({
      id: `electron-${electronImageName}`,
      kind: 'electron',
      electronName: electronImageName,
      label: `SE Image (${electronImageName})`,
      visible: true, opacity: 0.6, blend: 'normal',
    });
  } else if (elements.length > 0 || hasBC) {
    // SE-fallback only worth seeding if we have ANY other context to show
    // alongside it. Pure-empty inputs return [] (the empty-state test pins this).
    out.push({
      id: 'vbse',
      kind: 'vbse',
      label: 'V-BSE (SE fallback)',
      visible: true, opacity: 0.55, blend: 'normal',
    });
  }
  if (hasBC) {
    out.push({
      id: 'bc',
      kind: 'bc',
      label: 'BC (Band Contrast)',
      visible: true, opacity: 0.5, blend: 'multiply',
    });
  }
  for (const el of elements.slice(0, 3)) {
    out.push({
      id: `eds-${el}`,
      kind: 'eds-element',
      element: el,
      label: el,
      visible: true, opacity: 0.7, blend: 'screen',
      displayMode: 'at',
    });
  }
  return out;
}

/**
 * Build a layer entry for EVERY available map (all electron images + BC + all
 * EDS elements), all visible. Used by the "All Maps" tile grid, which is
 * DECOUPLED from the curated/capped composite overlay stack — so it shows the
 * full element set, not just the first three. Ordering: preferred SE image
 * first, then the remaining electron images, then BC, then all elements.
 * Tiles render each bitmap on its own, so opacity/blend are full/normal.
 */
export function allMapsLayersFor({ elements = [], electronImages = [], hasBC = false } = {}) {
  const out = [];
  const preferred = pickPreferredElectronImage(electronImages);
  const orderedElectron = preferred
    ? [preferred, ...electronImages.filter((n) => n !== preferred)]
    : electronImages;
  for (const name of orderedElectron) {
    out.push({
      id: `electron-${name}`, kind: 'electron', electronName: name,
      label: `SE Image (${name})`, visible: true, opacity: 1, blend: 'normal',
    });
  }
  if (hasBC) {
    out.push({
      id: 'bc', kind: 'bc', label: 'BC (Band Contrast)',
      visible: true, opacity: 1, blend: 'normal',
    });
  }
  for (const el of elements) {
    out.push({
      id: `eds-${el}`, kind: 'eds-element', element: el, label: el,
      visible: true, opacity: 1, blend: 'normal', displayMode: 'at',
    });
  }
  return out;
}
