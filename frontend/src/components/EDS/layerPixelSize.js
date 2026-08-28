/**
 * layerPixelSize.js — which physical pixel size belongs to which layer.
 *
 * The areas of an Aztec acquisition do not share a scale. Measured on a real
 * file (Arbeitsbereich 6):
 *
 *     EBSD / EDS      12 x 9      0.6579 um/px
 *     Electron Image  1024 x 768  0.0621 um/px
 *
 * a factor of 10.6, on a different field of view. The page used to hand ONE
 * global step size to the export dialog, so a scale bar burnt into an exported
 * SE image was wrong by that factor — silently, in a figure meant for a paper.
 *
 * ONE resolver for BOTH pages. The EDS page and the phase-map page grew their
 * own layer-id conventions (`electron-<name>` / `eds-<element>` here,
 * `se:<name>` / `eds:<element>` there) and only the EDS one had a pixel-size
 * rule; the phase map handed its scan step to every export, including exports
 * of an SE layer. Two conventions, one rule — a second copy of this table is
 * how the next 8.5x error gets in.
 *
 * Pure so the mapping can be tested without a backend or a canvas.
 */

/**
 * Strip the wrapper off a mask layer's id: `mask-<sourceId>-<stamp>`.
 *
 * A mask draws the pixels of the layer it masks, so its scale is that layer's
 * scale — but its ID names the mask. `mask-electron-SE/Elektronenbild 1-1756…`
 * matched none of the prefixes below and fell through to the scan step, which
 * put 0.5 um/px on a 1024-px-wide SE image.
 *
 * The source id may itself contain hyphens and slashes, so only the leading
 * `mask-` and the trailing `-<digits>` are removed. Returns the id unchanged
 * when it is not a mask id.
 */
export function maskSourceId(layerId) {
  const m = /^mask-(.+)-\d+$/.exec(String(layerId || ''));
  return m ? m[1] : String(layerId || '');
}

/**
 * Acquisition area a layer's image comes from.
 *
 * `layer` may be the id alone or the layer object; passing the object lets a
 * mask be resolved through its own `isMaskFor` rather than by parsing the id,
 * which is both cheaper and exact.
 */
export function areaForLayer(layer) {
  let id;
  if (layer && typeof layer === 'object') {
    id = String(layer.isMaskFor || layer.id || '');
  } else {
    id = maskSourceId(layer);
  }
  // EDS page ids, then phase-map page ids. Both name the same two areas.
  if (id.startsWith('electron-') || id.startsWith('se:')) return 'electron_image';
  if (id.startsWith('eds-') || id.startsWith('eds:')) return 'eds';
  // phase / ipf-* / bc / ci / uncertainty / vbse / kam / gos / the diagnostics
  // and refinement layers are all rendered on the EBSD scan raster.
  return 'ebsd';
}

/**
 * Physical pixel size for a layer, or null when it cannot be determined.
 *
 * Returns null rather than a fallback: a scale bar drawn from the wrong area's
 * pixel size is worse than no scale bar, because it looks authoritative.
 * The one substitution allowed is EBSD <-> EDS, which Aztec writes on the same
 * scan raster (verified equal on both real files).
 */
export function pixelSizeForLayer(layer, pixelSizes) {
  if (!pixelSizes) return null;
  const area = areaForLayer(layer);
  const direct = pixelSizes[area];
  if (direct && Number.isFinite(direct.x) && direct.x > 0) return direct;

  if (area === 'ebsd' || area === 'eds') {
    const other = pixelSizes[area === 'ebsd' ? 'eds' : 'ebsd'];
    if (other && Number.isFinite(other.x) && other.x > 0) return other;
  }
  return null;
}

/**
 * `pixelSizeForLayer(...).x`, or null — the form every export path wants.
 *
 * Exists so no caller has to remember the `?.x ?? null` dance; `undefined`
 * reaching `unitsPerPixel` reads as "no scale" in some checks and as NaN in
 * others.
 */
export function umPerPxForLayer(layer, pixelSizes) {
  const size = pixelSizeForLayer(layer, pixelSizes);
  return size && Number.isFinite(size.x) && size.x > 0 ? size.x : null;
}

/**
 * The pixel size of the SCAN raster — the grid the phase map, the element maps
 * and every composite are drawn on.
 *
 * `pixelSizeForLayer('phase', …)` says the same thing; this name says WHY at
 * the call site, where the question is "what raster is this canvas", not
 * "what layer is this".
 */
export function scanUmPerPx(pixelSizes) {
  return umPerPxForLayer('phase', pixelSizes);
}
