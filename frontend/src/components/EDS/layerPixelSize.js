/**
 * layerPixelSize.js — which physical pixel size belongs to which EDS layer.
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
 * Pure so the mapping can be tested without a backend or a canvas.
 */

/** Acquisition area a layer's image comes from. */
export function areaForLayer(layerId) {
  const id = String(layerId || '');
  if (id.startsWith('electron-')) return 'electron_image';
  if (id.startsWith('eds-')) return 'eds';
  // phase / ipf-* / bc / ci / uncertainty and friends are all rendered on the
  // EBSD scan raster.
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
export function pixelSizeForLayer(layerId, pixelSizes) {
  if (!pixelSizes) return null;
  const area = areaForLayer(layerId);
  const direct = pixelSizes[area];
  if (direct && Number.isFinite(direct.x) && direct.x > 0) return direct;

  if (area === 'ebsd' || area === 'eds') {
    const other = pixelSizes[area === 'ebsd' ? 'eds' : 'ebsd'];
    if (other && Number.isFinite(other.x) && other.x > 0) return other;
  }
  return null;
}
