// Canvas wrapper around the pure painter: build a canvas at the target
// resolution, paint the model, return a PNG/JPEG Blob. Also download + clipboard
// helpers. The painter (paintFigure) holds all drawing logic and is unit-tested
// separately with a recording fake context.
import { paintFigure } from './paintFigure';

/** Resolve the export pixel dimensions from the model + resolution options. */
export function targetDims(model, { scale = 2, targetWidthPx = null } = {}) {
  const { wPx, hPx } = model.canvas;
  if (targetWidthPx) return { W: Math.round(targetWidthPx), H: Math.round(targetWidthPx * (hPx / wPx)) };
  return { W: Math.round(wPx * scale), H: Math.round(hPx * scale) };
}

/** Decode a {key: base64|dataURL} map into HTMLImageElements (skips falsy). */
export function loadSources(b64map) {
  const entries = Object.entries(b64map).filter(([, v]) => !!v);
  return Promise.all(entries.map(([k, v]) => new Promise((res) => {
    const img = new Image();
    img.onload = () => res([k, img]);
    img.onerror = () => res([k, null]);
    img.src = v.startsWith('data:') ? v : `data:image/png;base64,${v}`;
  }))).then((pairs) => Object.fromEntries(pairs.filter(([, v]) => v)));
}

export async function composePatternFigure(model, sources, opts = {}) {
  // Require at least one element that will actually render: a non-panel element,
  // or a panel whose source image successfully decoded. This blocks exporting a
  // blank canvas (panel(s) whose image is missing) while still allowing e.g. a
  // scale-bar/colorbar/text-only figure.
  const renderable = model.elements.some((e) => e.type !== 'panel' || sources?.[e.source]);
  if (!renderable) throw new Error('Nothing to export: add at least one panel (with an available image).');
  const { W, H } = targetDims(model, opts);
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context unavailable.');
  paintFigure(ctx, model, sources, { W, H });
  const type = opts.format === 'jpeg' ? 'image/jpeg' : 'image/png';
  const q = opts.format === 'jpeg' ? (opts.jpegQuality ?? 0.9) : undefined;
  const blob = await new Promise((res) => canvas.toBlob(res, type, q));
  if (!blob) throw new Error('canvas.toBlob returned null');
  return blob;
}

export function downloadFigure(blob, filename = 'pattern-figure.png') {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
}

export async function copyFigureToClipboard(blob) {
  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
    throw new Error('Clipboard image copy not supported in this environment.');
  }
  await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
}
