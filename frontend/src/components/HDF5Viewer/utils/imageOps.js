/**
 * Pure image-processing helpers for the HDF5 viewer.
 * Framework-free — no React, no Zustand. Operate on ImageData / base64 PNG strings.
 */

/**
 * Apply a gamma-correction look-up-table to an ImageData object in-place.
 * Reads/writes the .data Uint8ClampedArray.
 *
 * Behaviour preserved verbatim from the original inline version in HDF5Viewer.jsx:
 * gammaPct is a percentage (e.g. 100 = identity, 50 = darker, 200 = brighter).
 * Values within 1% of identity are skipped as a fast path.
 *
 * @param {ImageData} imageData
 * @param {number} gammaPct - gamma * 100; 100 is identity
 * @returns {void} mutates imageData in-place
 */
export function applyGammaLUT(imageData, gammaPct) {
  const gammaVal = gammaPct / 100;
  if (Math.abs(gammaVal - 1.0) < 0.01) return;
  const lut = new Uint8Array(256);
  const exp = 1 / gammaVal;
  for (let i = 0; i < 256; i++) lut[i] = Math.min(255, Math.round(255 * Math.pow(i / 255, exp)));
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = lut[d[i]]; d[i + 1] = lut[d[i + 1]]; d[i + 2] = lut[d[i + 2]];
  }
}

/**
 * Apply global histogram-equalisation to an ImageData object in-place.
 *
 * Note: despite being toggled by an "CLAHE" UI checkbox in HDF5Viewer.jsx,
 * the implementation is plain global histogram equalisation, not tile-based
 * CLAHE. The (mislabelled) name is preserved for now — a future task will
 * either rename or replace with a real CLAHE implementation.
 *
 * @param {ImageData} imageData
 * @returns {void} mutates imageData in-place
 */
export function applyHistEq(imageData) {
  const d = imageData.data;
  const total = d.length / 4;
  const hist = new Float32Array(256);
  for (let i = 0; i < d.length; i += 4) {
    hist[Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2])]++;
  }
  const cdf = new Float32Array(256);
  cdf[0] = hist[0];
  for (let i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];
  const cdfMin = cdf.find((v) => v > 0) ?? 0;
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) lut[i] = Math.round(((cdf[i] - cdfMin) / (total - cdfMin)) * 255);
  for (let i = 0; i < d.length; i += 4) {
    d[i] = lut[d[i]]; d[i + 1] = lut[d[i + 1]]; d[i + 2] = lut[d[i + 2]];
  }
}

/**
 * Compute the standard deviation of luminance for a base64-encoded PNG.
 * Decodes via an offscreen Image + canvas (requires DOM).
 * Returns 0 if the image fails to load.
 *
 * @param {string} base64 - raw base64 PNG payload (no data: URI prefix)
 * @returns {Promise<number>} luminance standard deviation, or 0 on error
 */
export async function computeStdDev(base64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const lums = [];
      for (let i = 0; i < id.data.length; i += 4)
        lums.push(0.299 * id.data[i] + 0.587 * id.data[i + 1] + 0.114 * id.data[i + 2]);
      const mean = lums.reduce((s, v) => s + v, 0) / lums.length;
      const variance = lums.reduce((s, v) => s + (v - mean) ** 2, 0) / lums.length;
      resolve(Math.sqrt(variance));
    };
    img.onerror = () => resolve(0);
    img.src = `data:image/png;base64,${base64}`;
  });
}

/**
 * Classify an EBSD pattern image into a coarse quality bucket.
 * Decodes via an offscreen Image + canvas (requires DOM).
 *
 * Categories:
 *   "beam_off"    - mean luminance < 5
 *   "low_signal"  - mean luminance < 30
 *   "saturated"   - >15% of pixels at luminance > 250
 *   "hot_pixels"  - >1% of pixels brighter than mean + 4σ
 *   "ok"          - otherwise (also returned on decode error)
 *
 * @param {string} base64 - raw base64 PNG payload (no data: URI prefix)
 * @returns {Promise<"beam_off"|"low_signal"|"saturated"|"hot_pixels"|"ok">}
 */
export async function classifyPattern(base64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const d = id.data;
      let sum = 0, saturatedCount = 0;
      const total = d.length / 4;
      const lums = new Float32Array(total);
      for (let i = 0, j = 0; i < d.length; i += 4, j++) {
        const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        lums[j] = lum;
        sum += lum;
        if (lum > 250) saturatedCount++;
      }
      const mean = sum / total;
      let variance = 0;
      for (let i = 0; i < total; i++) variance += (lums[i] - mean) ** 2;
      const std = Math.sqrt(variance / total);
      let hotCount = 0;
      for (let i = 0; i < total; i++) if (lums[i] > mean + 4 * std) hotCount++;
      const satFrac = saturatedCount / total;
      const hotFrac = hotCount / total;
      if (mean < 5)          resolve('beam_off');
      else if (mean < 30)    resolve('low_signal');
      else if (satFrac > 0.15) resolve('saturated');
      else if (hotFrac > 0.01) resolve('hot_pixels');
      else                   resolve('ok');
    };
    img.onerror = () => resolve('ok');
    img.src = patternToImgSrc(base64);
  });
}


/**
 * Resolve a pattern string into something usable as `<img>.src` / `<a>.href`.
 *
 * The navigation hook now writes blob: URLs (binary transport, faster) but
 * older code paths (sessionStorage rehydration, old test fixtures) may still
 * pass a bare base64 PNG payload. Detecting the prefix keeps both forms
 * working through the same consumer code.
 *
 * @param {string | null | undefined} pattern
 * @returns {string} a URL/data-URL the browser can load directly
 */
export function patternToImgSrc(pattern) {
  if (!pattern) return '';
  if (pattern.startsWith('blob:') || pattern.startsWith('data:')) return pattern;
  return `data:image/png;base64,${pattern}`;
}
