/**
 * buildMaskCanvas — synthesise a B/W mask canvas from a source bitmap and
 * a threshold {min, max}. Pixels with luminance inside the range become
 * fully white (and fully opaque), pixels outside become fully black
 * (and still fully opaque). The result is intended to be drawn over
 * other layers with `globalCompositeOperation = 'multiply'`, which then
 * gates the lower layers to the in-threshold region only.
 *
 * Luma formula matches LayeredCanvas / Tile (0.299/0.587/0.114).
 */
export function buildMaskCanvas(sourceBitmap, threshold) {
  const off = document.createElement('canvas');
  off.width = sourceBitmap.width;
  off.height = sourceBitmap.height;
  const ctx = off.getContext('2d');
  ctx.drawImage(sourceBitmap, 0, 0);
  const img = ctx.getImageData(0, 0, off.width, off.height);
  const { min, max } = threshold;
  for (let i = 0; i < img.data.length; i += 4) {
    const v = img.data[i] * 0.299 + img.data[i + 1] * 0.587 + img.data[i + 2] * 0.114;
    const pass = (v >= min && v <= max);
    img.data[i]     = pass ? 255 : 0;
    img.data[i + 1] = pass ? 255 : 0;
    img.data[i + 2] = pass ? 255 : 0;
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return off;
}
