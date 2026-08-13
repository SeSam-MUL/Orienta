import { useRef, useEffect, useState } from 'react';
import { pointerToRowCol, rowColToContainerPx } from './mapCoords';
import { useCursorPublisher, useCursorSync } from './CursorSyncContext';
import { useRectangleDrag } from './hooks/useRectangleDrag';
import { buildMaskCanvas } from '../PhaseMap/maskCanvas';
import { IDENTITY_VIEW, isZoomed, viewToTransform, zoomedRect, wheelFactor } from './zoomView';
import { colors } from '../../theme/components';

// A drag shorter than this (in CSS px) still counts as a click, so panning a
// zoomed tile never fires click-to-quantify by accident.
const CLICK_SLOP_PX = 3;

export default function Tile({
  layer, bitmap, error, shape, onPixelClick, onRegionSelected,
  view = IDENTITY_VIEW, onZoomAt, onPan, onResetView, onContextMenu,
}) {
  const hostRef = useRef(null);
  const canvasRef = useRef(null);
  const publish = useCursorPublisher();
  const [crosshair, setCrosshair] = useState(null);

  // Latest view + zoom callbacks in refs: the wheel listener below is attached
  // natively (once) and the pointer handlers run at 60Hz — neither should
  // depend on a fresh closure.
  const viewRef = useRef(view);
  viewRef.current = view;
  const zoomCbRef = useRef(onZoomAt);
  zoomCbRef.current = onZoomAt;

  const drag = useRectangleDrag({
    shape,
    onRegion: onRegionSelected,
    mapRect: (r) => zoomedRect(r, viewRef.current),
  });

  // Pan bookkeeping: last pointer position plus the distance travelled, which
  // decides whether the following click counts.
  const panRef = useRef(null);
  const suppressClickRef = useRef(false);

  // Single-layer draw — no blending. Bitmap may be undefined while fetching.
  // For mask layers, `bitmap` is the SOURCE bitmap (provided by TileGrid) and
  // we render the B/W mask preview directly (no blending, no extra threshold
  // — the mask IS the visual).
  //
  // The canvas buffer is sized to the BITMAP's own resolution, not the page's
  // global `shape`. EDS layers come from two pixel grids — element/BC maps at
  // the EBSD scan grid, electron images at a higher-res SEM survey resolution.
  // Sizing every tile to one global `shape` drew the smaller bitmaps 1:1 into
  // the corner of an oversized canvas (the "mini in the corner" bug).
  useEffect(() => {
    if (!canvasRef.current || !bitmap) return;
    const c = canvasRef.current;
    c.width = bitmap.width; c.height = bitmap.height;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if (layer.kind === 'mask' && layer.threshold) {
      const m = buildMaskCanvas(bitmap, layer.threshold);
      ctx.drawImage(m, 0, 0);
      return;
    }
    ctx.drawImage(bitmap, 0, 0);
    // Threshold alpha-mask: pixels with luminance outside [min, max] become
    // fully transparent. Same 0.299/0.587/0.114 luma formula as LayeredCanvas.
    if (layer.threshold) {
      const img = ctx.getImageData(0, 0, c.width, c.height);
      const { min, max } = layer.threshold;
      for (let i = 0; i < img.data.length; i += 4) {
        const v = img.data[i] * 0.299 + img.data[i + 1] * 0.587 + img.data[i + 2] * 0.114;
        if (v < min || v > max) img.data[i + 3] = 0;
      }
      ctx.putImageData(img, 0, 0);
    }
  }, [bitmap, layer.threshold, layer.kind]);

  // Ctrl/Cmd + wheel zooms towards the cursor; a plain wheel is left alone so
  // it keeps scrolling the tile grid. Attached natively because React's
  // onWheel is passive — preventDefault() there would be ignored (and would
  // let Electron zoom the whole app instead).
  useEffect(() => {
    const el = hostRef.current;
    if (!el) return undefined;
    const onWheel = (e) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      if (!zoomCbRef.current) return;
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      zoomCbRef.current(
        wheelFactor(e.deltaY),
        (e.clientX - rect.left) / rect.width,
        (e.clientY - rect.top) / rect.height,
      );
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  useCursorSync((pos) => {
    if (!shape || !pos.hovering || !hostRef.current) { setCrosshair(null); return; }
    const rect = hostRef.current.getBoundingClientRect();
    const zr = zoomedRect(rect, viewRef.current);
    const p = rowColToContainerPx(pos.row, pos.col, zr, shape);
    if (!p) { setCrosshair(null); return; }
    const x = p.x + (zr.left - rect.left);
    const y = p.y + (zr.top - rect.top);
    // While zoomed the synced pixel can sit outside the visible crop.
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) { setCrosshair(null); return; }
    setCrosshair({ x, y });
  });

  const onMove = (e) => {
    if (!shape || !hostRef.current) return;
    const rect = hostRef.current.getBoundingClientRect();
    // Pan first — it must feel immediate and does not depend on pixel mapping.
    if (panRef.current) {
      if (e.buttons !== 1) { panRef.current = null; }
      else if (rect.width && rect.height) {
        const dx = e.clientX - panRef.current.x;
        const dy = e.clientY - panRef.current.y;
        panRef.current = {
          x: e.clientX, y: e.clientY,
          moved: panRef.current.moved + Math.abs(dx) + Math.abs(dy),
        };
        if (panRef.current.moved > CLICK_SLOP_PX) suppressClickRef.current = true;
        onPan?.(dx / rect.width, dy / rect.height);
      }
    }
    const out = pointerToRowCol(e, zoomedRect(rect, viewRef.current), shape);
    if (out) publish({ row: out.row, col: out.col, hovering: true, screenX: e.clientX, screenY: e.clientY });
  };
  const onLeave = (e) => {
    panRef.current = null;
    if (!shape) return;
    publish({ row: 0, col: 0, hovering: false, screenX: e.clientX, screenY: e.clientY });
  };
  const onDown = (e) => {
    // Shift+drag stays the ROI tool; plain drag pans once zoomed in.
    if (!e.shiftKey && e.button === 0 && isZoomed(viewRef.current) && onPan) {
      panRef.current = { x: e.clientX, y: e.clientY, moved: 0 };
      suppressClickRef.current = false;
    }
    drag.onPointerDown(e);
  };
  const onUp = (e) => {
    panRef.current = null;
    drag.onPointerUp(e);
  };
  const onClick = (e) => {
    if (suppressClickRef.current) { suppressClickRef.current = false; return; }
    if (!shape || !hostRef.current || !onPixelClick) return;
    const rect = hostRef.current.getBoundingClientRect();
    const out = pointerToRowCol(e, zoomedRect(rect, viewRef.current), shape);
    if (out) onPixelClick(out.row, out.col);
  };

  const zoomed = isZoomed(view);

  return (
    <div
      data-tile
      data-layer-id={layer.id}
      style={{
        background: colors.bgSecondary,
        // CSS keyword 'red' for error border — jsdom preserves the keyword in
        // style.border, while hex tokens get normalised to rgb(). Keeps tests
        // honest and still renders correctly in the browser.
        border: `1px solid ${error ? 'red' : colors.border}`,
        borderRadius: 5,
        padding: 6,
        display: 'flex',
        flexDirection: 'column',
        gap: 5,
        cursor: 'crosshair',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9pt', color: colors.text, fontWeight: 600 }}>
        <span>{layer.label}</span>
        <span style={{ color: colors.textSecondary, fontSize: '7.5pt' }}>
          {zoomed && (
            <span data-tile-zoom-badge style={{ color: colors.accent, fontWeight: 700, marginRight: 4 }}>
              {view.scale.toFixed(1)}×
            </span>
          )}
          {layer.blend} · {Math.round((layer.opacity ?? 1) * 100)}%
        </span>
      </div>
      <div
        ref={hostRef}
        data-tile-host
        onMouseDown={onDown}
        onMouseMove={(e) => { onMove(e); drag.onPointerMove(e); }}
        onMouseUp={onUp}
        onMouseLeave={onLeave}
        onClick={onClick}
        onDoubleClick={() => onResetView?.()}
        onContextMenu={(e) => {
          if (!onContextMenu) return;
          e.preventDefault();
          onContextMenu(e.clientX, e.clientY);
        }}
        style={{
          position: 'relative',
          width: '100%',
          overflow: 'hidden',
          borderRadius: 3,
          cursor: zoomed ? 'grab' : 'crosshair',
          // Aspect from the bitmap's own dims so the canvas (width/height 100%)
          // is never stretched. Falls back to `shape`, then a sane default,
          // while the bitmap is still fetching.
          aspectRatio: bitmap
            ? `${bitmap.width}/${bitmap.height}`
            : (shape ? `${shape[1]}/${shape[0]}` : '156/128'),
        }}
      >
        {/* Zoom wrapper: a CSS transform, so no canvas is ever re-drawn and
            the pixel data stays byte-identical to the unzoomed render. */}
        <div
          data-tile-zoom-layer
          style={{
            position: 'absolute', inset: 0,
            transform: viewToTransform(view),
            transformOrigin: '50% 50%',
            willChange: zoomed ? 'transform' : 'auto',
          }}
        >
          <canvas
            ref={canvasRef}
            style={{
              width: '100%', height: '100%',
              imageRendering: 'pixelated',
              opacity: Math.max(0.4, layer.opacity ?? 1),
              background: '#000',
            }}
          />
        </div>
        {crosshair && (
          <div
            data-tile-crosshair
            style={{
              position: 'absolute', left: crosshair.x, top: crosshair.y,
              width: 14, height: 14, transform: 'translate(-50%, -50%)',
              border: `1px solid ${colors.cyan}`, borderRadius: 2,
              boxShadow: '0 0 0 1px rgba(0,0,0,.45)',
              pointerEvents: 'none',
            }}
          />
        )}
        {drag.overlay && (
          <div data-roi-rect style={{
            position: 'absolute',
            left: Math.min(drag.overlay.x0, drag.overlay.x1),
            top:  Math.min(drag.overlay.y0, drag.overlay.y1),
            width:  Math.abs(drag.overlay.x1 - drag.overlay.x0),
            height: Math.abs(drag.overlay.y1 - drag.overlay.y0),
            border: `1px dashed ${colors.cyan}`,
            background: 'rgba(139,233,253,.08)',
            pointerEvents: 'none',
          }} />
        )}
        {error && (
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,.55)', color: colors.red,
            fontSize: '9pt', textAlign: 'center', padding: 8,
          }}>
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
