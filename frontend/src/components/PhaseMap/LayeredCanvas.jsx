/**
 * LayeredCanvas — composites ImageBitmaps onto an offscreen canvas at
 * native map resolution, then scales to fit container. Redraws batched
 * via requestAnimationFrame so rapid state changes (slider drag) collapse
 * to one frame.
 *
 * Decorations (scalebar / title / IPF keys) are rendered as HTML/SVG
 * overlays positioned absolutely on top of the map. They update instantly
 * on slider drag without touching the canvas redraw path.
 *
 * Props:
 *   layers   — Layer[] (bottom-up: layers[0] drawn first)
 *   bitmaps  — Map<layerId, ImageBitmap>
 *   loading  — boolean (show spinner overlay)
 *   error    — string|null
 *   perLayerErrors — Map<id, string>|null
 *   scalebar — { enabled, position, length, fontSize, barColor, boxColor, boxAlpha }
 *   title    — string|null  (rendered at top center)
 *   stepX    — number (μm/pixel)
 *   ipfKeyImage   — base64 PNG | null (rendered bottom-right when IPF layer active)
 *   showIpfKey    — boolean (only show key when an IPF-* layer is in the stack)
 */
import { useRef, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { BLEND_MAP } from './layerSources';
import { buildMaskCanvas } from './maskCanvas';
import { colors } from '../../theme/components';

// Compute the bounding box of non-transparent pixels in the canvas.
// Returns null when the canvas is entirely transparent or full. Used to
// pan-centre the indexed region inside the canvas frame so a small ROI
// doesn't stick in one corner.
function computeAlphaBbox(ctx, w, h) {
  if (w <= 0 || h <= 0) return null;
  let imgData;
  try {
    imgData = ctx.getImageData(0, 0, w, h);
  } catch {
    return null;
  }
  const d = imgData.data;
  let minX = w, minY = h, maxX = -1, maxY = -1;
  // Scan rows first to find vertical extent (cheap early-exit on rows that
  // are entirely transparent).
  for (let y = 0; y < h; y++) {
    const rowOff = y * w * 4;
    let rowHasContent = false;
    for (let x = 0; x < w; x++) {
      if (d[rowOff + x * 4 + 3] > 0) {
        rowHasContent = true;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
      }
    }
    if (rowHasContent) {
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return null;
  const bbox = { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
  // If the bbox covers (essentially) the whole canvas, no centring needed.
  if (bbox.w >= w - 1 && bbox.h >= h - 1) return null;
  return bbox;
}

// Round a physical length to a "nice" number: 1, 2, 5 × 10^n.
function niceLength(rawMicrons) {
  if (!Number.isFinite(rawMicrons) || rawMicrons <= 0) return null;
  const exponent = Math.floor(Math.log10(rawMicrons));
  const base = Math.pow(10, exponent);
  const norm = rawMicrons / base;
  let pick;
  if (norm < 1.5) pick = 1;
  else if (norm < 3.5) pick = 2;
  else if (norm < 7.5) pick = 5;
  else pick = 10;
  return pick * base;
}

function formatMicrons(um) {
  if (um >= 1000) return `${(um / 1000).toFixed(um >= 10000 ? 0 : 1)} mm`;
  if (um >= 1) return `${um % 1 === 0 ? um : um.toFixed(1)} µm`;
  return `${(um * 1000).toFixed(um >= 0.1 ? 0 : 1)} nm`;
}

// Full 9-position map. The old logic only matched startsWith('upper') and
// endsWith('right'), so all 5 'center' positions (upper center / center left
// / center / center right / lower center) fell through to bottom-left,
// regardless of what the user picked in the dropdown.
const SCALEBAR_POS_STYLE = {
  'upper left':   { top: '3%',    left:  '3%' },
  'upper center': { top: '3%',    left:  '50%', transform: 'translateX(-50%)' },
  'upper right':  { top: '3%',    right: '3%' },
  'center left':  { top: '50%',   left:  '3%',  transform: 'translateY(-50%)' },
  'center':       { top: '50%',   left:  '50%', transform: 'translate(-50%, -50%)' },
  'center right': { top: '50%',   right: '3%',  transform: 'translateY(-50%)' },
  'lower left':   { bottom: '3%', left:  '3%' },
  'lower center': { bottom: '3%', left:  '50%', transform: 'translateX(-50%)' },
  'lower right':  { bottom: '3%', right: '3%' },
};

function ScalebarOverlay({ nativeSize, stepX, settings }) {
  if (!settings?.enabled || !nativeSize || !stepX || stepX <= 0) return null;
  const mapWidthUm = nativeSize.w * stepX;
  const raw = mapWidthUm * settings.length;
  const nice = niceLength(raw);
  if (!nice) return null;
  const widthFrac = nice / mapWidthUm;          // fraction of map width
  const widthPct = `${(widthFrac * 100).toFixed(2)}%`;
  const label = formatMicrons(nice);

  const pos = SCALEBAR_POS_STYLE[settings.position] ?? SCALEBAR_POS_STYLE['lower right'];

  return (
    <div style={{
      position: 'absolute', ...pos,
      opacity: 1,
      padding: '4px 8px',
      borderRadius: 3,
      color: settings.barColor,
      fontSize: `${settings.fontSize}pt`,
      fontFamily: 'sans-serif',
      lineHeight: 1.0,
      backgroundColor: hexWithAlpha(settings.boxColor, settings.boxAlpha),
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
      pointerEvents: 'none',
    }}>
      <div style={{
        width: widthPct, height: 4,
        background: settings.barColor,
      }} />
      <div>{label}</div>
    </div>
  );
}

function hexWithAlpha(hex, alpha) {
  // hex like #rrggbb → rgba(r,g,b,alpha)
  if (!hex || !hex.startsWith('#')) return hex;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function TitleOverlay({ text }) {
  if (!text) return null;
  return (
    <div style={{
      position: 'absolute',
      top: 8, left: 0, right: 0,
      display: 'flex', justifyContent: 'center',
      pointerEvents: 'none',
    }}>
      <span style={{
        background: 'rgba(0, 0, 0, 0.55)',
        color: '#fff',
        fontSize: '10pt',
        fontWeight: 500,
        padding: '3px 10px',
        borderRadius: 12,
        whiteSpace: 'nowrap',
      }}>
        {text}
      </span>
    </div>
  );
}

function IpfKeyOverlay({ imageBase64 }) {
  const { t } = useTranslation('phasemap');
  if (!imageBase64) return null;
  return (
    <div style={{
      position: 'absolute',
      bottom: 8, right: 8,
      // Was 110×110 — far too small to read the orientation triangle(s),
      // especially with multiple Laue groups (the backend PNG widens to
      // ~2*n_keys inches). Give it real room while staying a corner overlay.
      maxWidth: 220, maxHeight: 180,
      background: 'rgba(255,255,255,0.92)',
      borderRadius: 3,
      padding: 2,
      pointerEvents: 'none',
    }}>
      <img
        src={`data:image/png;base64,${imageBase64}`}
        alt={t('phasemap:canvasOverlay.ipfKeyAlt')}
        style={{ display: 'block', maxWidth: '100%', maxHeight: '100%' }}
      />
    </div>
  );
}

export default function LayeredCanvas({
  layers, bitmaps, bitmapVersion = 0,
  loading, error, perLayerErrors = null,
  scalebar = null, title = null, stepX = 1.0,
  ipfKeyImage = null, showIpfKey = false,
  hoverPixel = null,
}) {
  const { t } = useTranslation('phasemap');
  const visibleRef = useRef(null);
  const offscreenRef = useRef(null);
  const rafRef = useRef(null);
  const containerRef = useRef(null);
  // JS-measured fit size: width/height in CSS pixels at which we render
  // the canvas + overlays. Recomputed on container resize via ResizeObserver.
  const [fitSize, setFitSize] = useState(null);

  // Derive the native canvas size from the first available bitmap.
  // All layers should match (alignment is enforced upstream).
  //
  // `bitmaps` is a Map ref whose identity NEVER changes — its contents
  // mutate in place. To pick up new bitmaps we depend on `bitmapVersion`,
  // a counter the hook bumps on every cache mutation. Without this the
  // first bitmap that arrives would be invisible until the user wiggled
  // something else (changed a blend mode, dragged a slider, etc.).
  const nativeSize = useMemo(() => {
    for (const layer of layers) {
      // Mask layers have no bitmap of their own (they synthesise one from
      // isMaskFor at draw time). Skip them when probing for size.
      if (layer.kind === 'mask') continue;
      const bmp = bitmaps.get(layer.id);
      if (bmp) return { w: bmp.width, h: bmp.height };
    }
    return null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, bitmapVersion]);

  // Measure the container and compute the largest map-aspect box that fits.
  // Aspect-ratio CSS proved fragile (the wrapper collapsed to 0×0 under
  // certain flex parent configs), so we just measure and set explicit
  // pixel dimensions. ResizeObserver handles container resizes.
  useEffect(() => {
    if (!nativeSize || !containerRef.current) return;
    const el = containerRef.current;
    const recompute = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const aspectMap = nativeSize.w / nativeSize.h;
      const aspectBox = rect.width / rect.height;
      let w, h;
      if (aspectBox > aspectMap) {
        // box wider than map → height-constrained
        h = rect.height;
        w = h * aspectMap;
      } else {
        // box taller than map → width-constrained
        w = rect.width;
        h = w / aspectMap;
      }
      setFitSize({ w, h });
    };
    recompute();
    const obs = new ResizeObserver(recompute);
    obs.observe(el);
    return () => obs.disconnect();
  }, [nativeSize]);

  // Combined allocation + redraw effect. Splitting these into two effects
  // previously created a race: the allocation effect fired on nativeSize
  // change while the wrapper was still gated behind {nativeSize && fitSize}
  // and `visibleRef.current` was null, so the canvas dimension assignment
  // was silently skipped. When the wrapper later mounted, the canvas was
  // stuck at its default 300x150 buffer and the bitmap was drawn into the
  // upper-left corner only — producing the "grey square in upper-left"
  // bug. By doing allocation just-in-time inside the rAF callback we know
  // the canvas is in the DOM before we touch it.
  useEffect(() => {
    if (!nativeSize || !fitSize) return;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      const vis = visibleRef.current;
      if (!vis) return;
      if (!offscreenRef.current) offscreenRef.current = document.createElement('canvas');
      const off = offscreenRef.current;
      // Make sure both canvas buffers match native resolution every frame —
      // cheap (no-op if size unchanged) and bulletproof against ordering.
      if (off.width !== nativeSize.w || off.height !== nativeSize.h) {
        off.width = nativeSize.w; off.height = nativeSize.h;
      }
      if (vis.width !== nativeSize.w || vis.height !== nativeSize.h) {
        vis.width = nativeSize.w; vis.height = nativeSize.h;
      }
      // willReadFrequently: this offscreen canvas is read back via getImageData
      // (computeAlphaBbox) on every recomposite — the hint keeps it on a CPU
      // backing store so the readback doesn't stall on a GPU→CPU copy.
      const offCtx = off.getContext('2d', { willReadFrequently: true });
      offCtx.globalAlpha = 1;
      offCtx.globalCompositeOperation = 'source-over';
      // Crisp scaling — when EDS layers of differing native resolution are
      // scaled to the common composite size, keep them pixel-sharp rather
      // than blurred (matches the `imageRendering: pixelated` visible canvas).
      offCtx.imageSmoothingEnabled = false;
      offCtx.clearRect(0, 0, off.width, off.height);
      // Reusable scratch canvas for threshold-aware layers — we mask the bitmap
      // there so the alpha-mask doesn't interact with blend/opacity on the off
      // canvas. PhaseMap layers never set `threshold`, so this branch is dead
      // for PhaseMap usage (no regression). EDS scalar layers (BC, V-BSE,
      // EDS-element) use it via useEdsLayerStack.setThreshold.
      let scratch = null;
      for (const layer of layers) {
        if (!layer.visible) continue;
        // Mask layers: synthesise a B/W mask from the source layer's bitmap
        // + the captured threshold, then composite with the layer's blend
        // (default 'multiply' gates layers below to the in-threshold region).
        if (layer.kind === 'mask') {
          const srcBmp = bitmaps.get(layer.isMaskFor);
          if (!srcBmp) continue;
          const maskCanvas = buildMaskCanvas(srcBmp, layer.threshold);
          offCtx.globalAlpha = layer.opacity;
          offCtx.globalCompositeOperation = BLEND_MAP[layer.blend] ?? 'multiply';
          // Scale to the composite's native size — layers may differ in
          // resolution (EDS scan grid vs higher-res electron image). For
          // PhaseMap, where all layers already match, this is a no-op.
          offCtx.drawImage(maskCanvas, 0, 0, off.width, off.height);
          continue;
        }
        const bmp = bitmaps.get(layer.id);
        if (!bmp) continue;
        offCtx.globalAlpha = layer.opacity;
        offCtx.globalCompositeOperation = BLEND_MAP[layer.blend] ?? 'source-over';
        if (layer.threshold) {
          if (!scratch) scratch = document.createElement('canvas');
          if (scratch.width !== bmp.width || scratch.height !== bmp.height) {
            scratch.width = bmp.width; scratch.height = bmp.height;
          }
          // willReadFrequently: read back via getImageData just below for the
          // threshold alpha-mask — same GPU→CPU stall avoidance as offCtx.
          const sCtx = scratch.getContext('2d', { willReadFrequently: true });
          sCtx.globalAlpha = 1;
          sCtx.globalCompositeOperation = 'source-over';
          sCtx.clearRect(0, 0, scratch.width, scratch.height);
          sCtx.drawImage(bmp, 0, 0);
          const img = sCtx.getImageData(0, 0, scratch.width, scratch.height);
          const { min, max } = layer.threshold;
          for (let i = 0; i < img.data.length; i += 4) {
            const v = img.data[i] * 0.299 + img.data[i + 1] * 0.587 + img.data[i + 2] * 0.114;
            if (v < min || v > max) img.data[i + 3] = 0;
          }
          sCtx.putImageData(img, 0, 0);
          offCtx.drawImage(scratch, 0, 0, off.width, off.height);
        } else {
          offCtx.drawImage(bmp, 0, 0, off.width, off.height);
        }
      }
      offCtx.globalAlpha = 1;
      offCtx.globalCompositeOperation = 'source-over';

      // Auto-zoom-to-bbox: when the indexed region covers only a small
      // part of the result_shape (common for ROI runs — e.g. a 30x24
      // block inside a 90x120 grid), scale the non-transparent bbox up
      // to FILL the canvas with letterboxing where the aspect doesn't
      // match. Without this the rendered content sits in whatever
      // corner the ROI lived in and most of the canvas is empty black.
      const bbox = computeAlphaBbox(offCtx, off.width, off.height);
      const visCtx = vis.getContext('2d');
      visCtx.imageSmoothingEnabled = false;
      visCtx.clearRect(0, 0, vis.width, vis.height);
      if (bbox) {
        const visW = vis.width;
        const visH = vis.height;
        const bboxAspect = bbox.w / bbox.h;
        const visAspect = visW / visH;
        // Object-fit:contain semantics — scale bbox into canvas keeping
        // aspect, letterbox empty space at top/bottom or left/right.
        let dstW, dstH;
        if (bboxAspect > visAspect) {
          dstW = visW;
          dstH = visW / bboxAspect;
        } else {
          dstH = visH;
          dstW = visH * bboxAspect;
        }
        const dstX = Math.round((visW - dstW) / 2);
        const dstY = Math.round((visH - dstH) / 2);
        // drawImage(src, sx, sy, sw, sh, dx, dy, dw, dh) — crop to bbox,
        // scale and centre in the visible canvas.
        visCtx.drawImage(
          off,
          bbox.x, bbox.y, bbox.w, bbox.h,
          dstX, dstY, Math.round(dstW), Math.round(dstH),
        );
      } else {
        visCtx.drawImage(off, 0, 0);
      }
    });
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // bitmapVersion is the real change signal — bitmaps Map identity is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, bitmapVersion, nativeSize, fitSize]);

  // Filter errors to only the layers currently in the stack — otherwise
  // ghost errors from removed/replaced layers stay visible forever.
  const activeLayerIds = useMemo(() => new Set(layers.map((l) => l.id)), [layers]);
  const layerErrorMessages = perLayerErrors
    ? [...perLayerErrors.entries()]
        .filter(([id]) => activeLayerIds.has(id))
        .map(([id, msg]) => `${id}: ${msg}`)
    : [];

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative', width: '100%', height: '100%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: '#000', overflow: 'hidden',
      }}
    >
      {nativeSize && fitSize && (
        <div style={{
          position: 'relative',
          width: `${fitSize.w}px`,
          height: `${fitSize.h}px`,
        }}>
          <canvas
            ref={visibleRef}
            style={{
              width: '100%',
              height: '100%',
              display: 'block',
              imageRendering: 'pixelated',
            }}
          />
          <ScalebarOverlay nativeSize={nativeSize} stepX={stepX} settings={scalebar} />
          <TitleOverlay text={title} />
          {showIpfKey && <IpfKeyOverlay imageBase64={ipfKeyImage} />}
          {hoverPixel
            && Number.isFinite(hoverPixel.row)
            && Number.isFinite(hoverPixel.col)
            && nativeSize.w > 0 && nativeSize.h > 0 && (
            <div
              style={{
                position: 'absolute',
                left:  `${((hoverPixel.col + 0.5) / nativeSize.w) * 100}%`,
                top:   `${((hoverPixel.row + 0.5) / nativeSize.h) * 100}%`,
                width: 12,
                height: 12,
                transform: 'translate(-50%, -50%)',
                border: '2px solid #ffb86c',
                borderRadius: '50%',
                boxShadow: '0 0 0 1px rgba(0,0,0,0.6)',
                pointerEvents: 'none',
              }}
            />
          )}
        </div>
      )}
      {!nativeSize && !loading && !error && (
        <div style={{ color: colors.textSecondary, fontSize: '10pt', padding: 24, textAlign: 'center' }}>
          {t('phasemap:canvas.noLayers')}
        </div>
      )}
      {loading && (
        <div style={{ position: 'absolute', top: 8, right: 8, color: colors.textSecondary, fontSize: '9pt' }}>
          {t('phasemap:canvasOverlay.loading')}
        </div>
      )}
      {error && (
        <div style={{ color: colors.red, fontSize: '10pt', padding: 16 }}>
          {error}
        </div>
      )}
      {layerErrorMessages.length > 0 && (
        <div style={{
          position: 'absolute', bottom: 4, left: 4, right: 4,
          background: '#22000099', border: `1px solid ${colors.red}`,
          color: colors.red, padding: '4px 8px', borderRadius: 3,
          fontSize: '8pt', fontFamily: 'monospace',
          maxHeight: '20%', overflow: 'auto',
        }}>
          {layerErrorMessages.map((m) => <div key={m}>{'⚠'} {m}</div>)}
        </div>
      )}
    </div>
  );
}
