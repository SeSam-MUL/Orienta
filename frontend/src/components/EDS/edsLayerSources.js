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
  notQuantified = [],
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
  // The seeded layers are at%, where two families of one element (the
  // spectrum source's `Cu Kα1` and `Cu Lα1,2`) draw the same quantified map:
  // seed three different elements, not three labels. An element the
  // quantification leaves out (`notQuantified`, the /elements response: C on
  // the matrix-corrected spectrum source) has no at% map -- not a seed.
  const skip = new Set((notQuantified || []).map((l) => String(l).split(' ')[0]));
  const seenSymbols = new Set();
  const seeded = [];
  for (const el of elements) {
    const symbol = String(el).split(' ')[0];
    if (seenSymbols.has(symbol) || skip.has(symbol)) continue;
    seenSymbols.add(symbol);
    seeded.push(el);
    if (seeded.length === 3) break;
  }
  for (const el of seeded) {
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
 * The element labels to offer as maps in `displayMode`.
 *
 * On the spectrum source the element list carries one label per fitted
 * family -- the Oxford reference scan: 12 labels for AZtec's 8 windows, `Fe Kα1` AND
 * `Fe Lα1,2`. In counts mode each is its own net map. In wt% / at% mode both
 * would draw the same quantified Fe map, so only the line the quantification
 * uses (`lines`, the fit status's `fit.lines`: `{Fe: {label, family}}`) is
 * kept per element. An element whose chosen label is not in the list keeps
 * all its labels (never drop an element); without `lines` (window files, a
 * fit not ready) the list is returned unchanged.
 */
export function elementsForDisplayMode(elements = [], displayMode = 'counts', lines = null) {
  if (displayMode === 'counts' || !lines || typeof lines !== 'object') return elements;
  const present = new Set(elements);
  return elements.filter((label) => {
    const chosen = lines[String(label).split(' ')[0]]?.label;
    return !chosen || !present.has(chosen) || chosen === label;
  });
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
