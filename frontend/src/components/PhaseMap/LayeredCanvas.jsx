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
import { bboxContentRect } from '../EDS/mapCoords';
import { roiDefiningLayers } from './roiFrame';
import { IDENTITY_VIEW, isZoomed, viewToTransform } from '../EDS/zoomView';
import { zoomRectPct } from '../EBSDViewer/zoomOverlay';
import { scalebarGeometry } from './scalebarGeometry';
import { colors } from '../../theme/components';

// Structural equality for content bboxes (or null). Used to gate the
// onContentBbox callback / local state update so we don't churn every frame.
function bboxEqual(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}

// CSS left/top percentages for the hover marker inside the canvas wrapper
// (which is sized to the full-shape aspect). Without a bbox the marker sits at
// the native-pixel centre as a fraction of the full grid. With a bbox the
// content is drawn object-fit:contain inside the buffer, so map the pixel
// through that inner letterbox (letterbox 2) — the wrapper already handles
// letterbox 1 by being sized to fitSize.
function markerPercent(row, col, nativeSize, contentBbox) {
  const { w: natW, h: natH } = nativeSize;
  let leftFrac, topFrac;
  if (contentBbox) {
    const dst = bboxContentRect(natW, natH, contentBbox);
    const fx = (col + 0.5 - contentBbox.x) / contentBbox.w;
    const fy = (row + 0.5 - contentBbox.y) / contentBbox.h;
    leftFrac = (dst.x + fx * dst.w) / natW;
    topFrac = (dst.y + fy * dst.h) / natH;
  } else {
    leftFrac = (col + 0.5) / natW;
    topFrac = (row + 0.5) / natH;
  }
  return { left: leftFrac * 100, top: topFrac * 100 };
}

// The marker sits OUTSIDE the zoom transform (a transformed 12px circle with a
// 2px border would render 32px thick at 16x), so its position is carried
// through the same view maths numerically — the trick zoomOverlay.js exists for.
function markerStyle(row, col, nativeSize, contentBbox, zoomView) {
  const p = markerPercent(row, col, nativeSize, contentBbox);
  const z = zoomRectPct({ left: p.left, top: p.top, width: 0, height: 0 }, zoomView);
  // Off the visible crop while zoomed in: drawing it would pin a stray dot to
  // the edge of the map.
  if (z.left < 0 || z.top < 0 || z.left > 100 || z.top > 100) return null;
  return { left: `${z.left}%`, top: `${z.top}%` };
}

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

function ScalebarOverlay({ nativeSize, contentBbox, stepX, settings, fitSize, zoomScale }) {
  if (!settings?.enabled) return null;
  // Pixels, not a percentage: the bar sits in a shrink-wrapped flex column, so
  // a percentage width resolved against the LABEL rather than the map — the bar
  // came out ~47x too short. scalebarGeometry also folds in the auto-zoom.
  const geom = scalebarGeometry({
    nativeSize,
    contentBbox,
    stepX,
    lengthFrac: settings.length,
    fitWidth: fitSize?.w,
    zoomScale,
  });
  if (!geom) return null;
  const { label, barPx } = geom;

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
      <div
        data-scalebar-bar
        style={{
          width: `${barPx}px`, height: 4,
          background: settings.barColor,
        }}
      />
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
  hoverPixel = null, onContentBbox = null,
  // The indexed region, straight from the result (null = the whole scan).
  // Everything is drawn inside it, so a region run behaves like a full one.
  roiBbox = null,
  // User zoom. Applied as a CSS transform to the CANVAS only, so the map
  // magnifies while the scalebar, title and IPF key keep their size and stay
  // anchored to the map box. Null renders exactly as before.
  zoomView = null,
}) {
  const view = zoomView || IDENTITY_VIEW;
  const zoomed = isZoomed(view);
  const { t } = useTranslation('phasemap');
  const visibleRef = useRef(null);
  const offscreenRef = useRef(null);
  const rafRef = useRef(null);
  const containerRef = useRef(null);
  // Auto-zoom bbox (native map pixels) or null when the full frame is drawn.
  // Computed inside the draw effect; kept in state so the hover marker can
  // map through it, and mirrored to the parent via onContentBbox so pointer
  // hit-testing uses the same bbox.
  const [contentBbox, setContentBbox] = useState(null);
  // Scratch canvas for measuring the result's own extent, and the last frame we
  // measured — see roiFrame.js for why the frame is the result's business.
  const roiCanvasRef = useRef(null);
  const lastRoiBboxRef = useRef(null);
  const lastBboxRef = useRef(null);
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
      // Layout metrics, NOT getBoundingClientRect: the client rect includes any
      // CSS transform on an ancestor, so once the user zooms in, measuring it
      // would feed the magnified size back into the fit and compound it.
      const rect = { width: el.clientWidth, height: el.clientHeight };
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
      // The region comes from the RESULT, not from the pixels on screen.
      // Measuring the drawing cannot work: an EDS element map, an electron
      // image and even band contrast (which falls back to the file's own BC)
      // exist at full scan size whatever the run covered, so the frame snapped
      // back to the whole scan the moment one of them was added — and with only
      // such a layer showing there was nothing left to measure at all.
      const roiLayers = roiDefiningLayers(layers).filter((l) => bitmaps.get(l.id));
      let bbox;
      if (roiBbox && roiBbox.w > 0 && roiBbox.h > 0) {
        bbox = roiBbox;
        lastRoiBboxRef.current = roiBbox;
      } else if (roiLayers.length > 0) {
        if (!roiCanvasRef.current) roiCanvasRef.current = document.createElement('canvas');
        const roi = roiCanvasRef.current;
        if (roi.width !== off.width || roi.height !== off.height) {
          roi.width = off.width; roi.height = off.height;
        }
        const roiCtx = roi.getContext('2d', { willReadFrequently: true });
        roiCtx.globalAlpha = 1;
        roiCtx.globalCompositeOperation = 'source-over';
        roiCtx.imageSmoothingEnabled = false;
        // Measured one layer at a time, because a layer that covers the whole
        // scan says nothing about the region. Band contrast is the case in
        // point: it sits in the "Indexing Result" group but falls back to the
        // file's own BC for every pixel, so a union with it would hand back the
        // full frame. `computeAlphaBbox` already answers null for full cover,
        // which is exactly "this layer does not constrain the frame".
        let acc = null;
        for (const layer of roiLayers) {
          const bmp = bitmaps.get(layer.id);
          if (!bmp) continue;
          roiCtx.clearRect(0, 0, roi.width, roi.height);
          roiCtx.drawImage(bmp, 0, 0, roi.width, roi.height);
          const b = computeAlphaBbox(roiCtx, roi.width, roi.height);
          if (!b) continue;
          acc = acc ? {
            x: Math.min(acc.x, b.x),
            y: Math.min(acc.y, b.y),
            w: Math.max(acc.x + acc.w, b.x + b.w) - Math.min(acc.x, b.x),
            h: Math.max(acc.y + acc.h, b.y + b.h) - Math.min(acc.y, b.y),
          } : b;
        }
        bbox = acc;
        // Only remember a real region: a full-frame result must not leave a
        // stale crop behind for the next stack that has nothing to say.
        lastRoiBboxRef.current = acc;
      } else {
        // Nothing from the result is showing right now (the user hid it, or
        // only source images are stacked). Keep the frame we had instead of
        // snapping to the full scan under their hands.
        bbox = lastRoiBboxRef.current ?? null;
      }
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

      // Publish the bbox (or null) so pointer/overlay mapping in the parent
      // and the local hover marker account for the auto-zoom. Gate on real
      // change (incl. null transitions) to avoid per-frame churn.
      if (!bboxEqual(bbox, lastBboxRef.current)) {
        lastBboxRef.current = bbox;
        setContentBbox(bbox);
        if (onContentBbox) onContentBbox(bbox);
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
        <div
          data-phasemap-map-box
          style={{
            position: 'relative',
            width: `${fitSize.w}px`,
            height: `${fitSize.h}px`,
            // The zoomed canvas is clipped to the map's own box rather than the
            // black letterbox area, so the decorations anchored to its corners
            // keep meaning what they say.
            overflow: 'hidden',
          }}
        >
          <canvas
            ref={visibleRef}
            style={{
              width: '100%',
              height: '100%',
              display: 'block',
              imageRendering: 'pixelated',
              // Percentage translates resolve against this element's own box,
              // which is exactly the box the page's pointer maths measures.
              transform: viewToTransform(view),
              transformOrigin: '50% 50%',
              willChange: zoomed ? 'transform' : 'auto',
            }}
          />
          <ScalebarOverlay
            nativeSize={nativeSize}
            contentBbox={contentBbox}
            stepX={stepX}
            settings={scalebar}
            fitSize={fitSize}
            zoomScale={view.scale}
          />
          <TitleOverlay text={title} />
          {showIpfKey && <IpfKeyOverlay imageBase64={ipfKeyImage} />}
          {hoverPixel
            && Number.isFinite(hoverPixel.row)
            && Number.isFinite(hoverPixel.col)
            && nativeSize.w > 0 && nativeSize.h > 0
            && markerStyle(hoverPixel.row, hoverPixel.col, nativeSize, contentBbox, view) && (
            <div
              style={{
                position: 'absolute',
                ...markerStyle(hoverPixel.row, hoverPixel.col, nativeSize, contentBbox, view),
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
