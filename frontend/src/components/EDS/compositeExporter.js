/**
 * Render the visible layer stack to an off-screen canvas at native shape
 * resolution, applying each layer's blend + opacity + threshold mask, then
 * trigger a PNG download.
 *
 * Scalebar burn-in is out of scope for this iteration — the PhaseMap
 * scalebar is server-side rendered and not directly reusable from the
 * frontend. A future task can extract a `drawScalebar(ctx, opts)` helper
 * once one exists in the codebase.
 *
 * Threshold-mask logic mirrors LayeredCanvas / Tile (0.299/0.587/0.114
 * luma → pixels outside [min,max] become fully transparent).
 */
export async function exportComposite({ layers, bitmaps, shape, filename = 'eds-composite.png' }) {
  if (!shape || !layers || layers.length === 0) {
    throw new Error('Nothing to export: layers/shape missing');
  }
  const [H, W] = shape;
  const off = document.createElement('canvas');
  off.width = W;
  off.height = H;
  const ctx = off.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  for (const l of layers) {
    if (!l.visible) continue;
    const bm = bitmaps.get(l.id);
    if (!bm) continue;
    ctx.globalAlpha = l.opacity ?? 1;
    ctx.globalCompositeOperation = (l.blend && l.blend !== 'normal') ? l.blend : 'source-over';
    // Scale to the export canvas — layers may differ in native resolution
    // (EDS scan grid vs higher-res electron image); explicit w/h co-registers.
    ctx.drawImage(bm, 0, 0, W, H);
    // Apply threshold alpha-mask if present (matches Tile/LayeredCanvas logic).
    if (l.threshold) {
      const img = ctx.getImageData(0, 0, W, H);
      const { min, max } = l.threshold;
      for (let i = 0; i < img.data.length; i += 4) {
        const v = img.data[i] * 0.299 + img.data[i + 1] * 0.587 + img.data[i + 2] * 0.114;
        if (v < min || v > max) img.data[i + 3] = 0;
      }
      ctx.putImageData(img, 0, 0);
    }
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';

  const blob = await new Promise((res) => off.toBlob(res, 'image/png'));
  if (!blob) throw new Error('Canvas toBlob returned null');
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
}
