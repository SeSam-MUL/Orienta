/**
 * Exporting the IPF colour key on its own.
 *
 * It was the only picture on the phase map page without a right-click export,
 * and it is the one a figure most often wants separately: the key is 2.75x as
 * tall as it is wide for three phases, so setting it beside a wide, short map
 * makes a sheet that is mostly empty paper.
 *
 * The two pieces live here rather than inline in the page because a full
 * PhaseMapPage mount is not testable in jsdom — the wiring is what breaks, so
 * the wiring is what a test has to be able to hold.
 */

/**
 * The key picture as a canvas at its native size.
 *
 * `opaque` gives it the white plate the on-screen panel shows; without it the
 * canvas keeps the backend's transparency, which a PNG carries through, so the
 * key can be laid onto a figure of one's own. Both are real choices — the key
 * is dark ink, and on a dark slide the transparent one looks empty.
 *
 * `makeCanvas` exists so a test can hand in a recording canvas: jsdom has no
 * 2D context.
 */
export function keyImageToCanvas(img, { opaque = true, makeCanvas } = {}) {
  if (!img?.width || !img?.height) {
    throw new Error('keyImageToCanvas: no key image');
  }
  const canvas = (makeCanvas || (() => document.createElement('canvas')))();
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('keyImageToCanvas: 2D context unavailable');
  if (opaque) {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  ctx.drawImage(img, 0, 0);
  return canvas;
}

/**
 * The two entries the key panel's context menu offers.
 *
 * `open(opaque, filename)` is called with what the entry stands for, so the
 * page keeps ownership of the export dialog and this stays a plain function.
 */
export function ipfKeyExportItems({ t, stem, open }) {
  const name = String(stem || 'phase-map').replace(/[\\/]/g, '_');
  return [
    {
      id: 'ipfkey-plate',
      label: t('imageexport:menuExportIpfKey'),
      onSelect: () => open(true, `${name}_ipf-key`),
    },
    {
      id: 'ipfkey-alpha',
      label: t('imageexport:menuExportIpfKeyAlpha'),
      onSelect: () => open(false, `${name}_ipf-key_transparent`),
    },
  ];
}
