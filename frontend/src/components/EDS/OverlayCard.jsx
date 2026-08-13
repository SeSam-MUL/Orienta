import { useRef, useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import LayeredCanvas from '../PhaseMap/LayeredCanvas';
import MagnifierLens from './MagnifierLens';
import { pointerToRowCol, rowColToContainerPx } from './mapCoords';
import { useCursorPublisher, useCursorSync } from './CursorSyncContext';
import { useRectangleDrag } from './hooks/useRectangleDrag';
import { IDENTITY_VIEW, isZoomed, viewToTransform, zoomedRect, wheelFactor } from './zoomView';
import { colors } from '../../theme/components';

// Drags shorter than this still count as a click (see Tile.jsx).
const CLICK_SLOP_PX = 3;

export default function OverlayCard({
  layers, bitmaps, bitmapVersion = 0, errors, shape, onPixelClick, onRegionSelected,
  linescanMode = false, onLineComplete,
  swipe = { a: null, b: null }, onSwipeSplitChange,
  magnifierEnabled = false,
  view = IDENTITY_VIEW, onZoomAt, onPan, onResetView, onContextMenu,
}) {
  const inSwipeMode = !!(swipe?.a && swipe?.b);
  const hostRef = useRef(null);
  const publish = useCursorPublisher();
  const [crosshair, setCrosshair] = useState(null);
  // Container-relative cursor position for the magnifier lens. Local to this
  // component so 60Hz mouse moves don't bubble up and re-render the rest of
  // the EDS page.
  const [lensPos, setLensPos] = useState(null);

  // Latest zoom view + callback in refs — the wheel listener is attached once
  // natively and the pointer handlers must not depend on a fresh closure.
  const viewRef = useRef(view);
  viewRef.current = view;
  const zoomCbRef = useRef(onZoomAt);
  zoomCbRef.current = onZoomAt;

  const drag = useRectangleDrag({
    shape,
    onRegion: onRegionSelected,
    mapRect: (r) => zoomedRect(r, viewRef.current),
  });

  // Pan bookkeeping (plain drag once zoomed; shift stays ROI, linescan mode
  // keeps the line tool).
  const panRef = useRef(null);
  const suppressClickRef = useRef(false);

  // Linescan drag state: persistent endpoints in container-relative px so the
  // line keeps showing after release until the next start.
  const [linePts, setLinePts] = useState(null);   // { x0, y0, x1, y1 }
  const lineStartRef = useRef(null);              // { row, col, x, y }

  // Ctrl/Cmd + wheel zooms towards the cursor; a plain wheel is left untouched
  // so the page keeps scrolling. Native listener because React's onWheel is
  // passive and could not preventDefault (Electron would zoom the whole app).
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

  // Subscribe to cursor sync and rebuild crosshair position when a sibling
  // (Tile or another OverlayCard) publishes. shape-null → no crosshair.
  useCursorSync((pos) => {
    if (!shape || !pos.hovering || !hostRef.current) { setCrosshair(null); return; }
    const rect = hostRef.current.getBoundingClientRect();
    const zr = zoomedRect(rect, viewRef.current);
    const p = rowColToContainerPx(pos.row, pos.col, zr, shape);
    if (!p) { setCrosshair(null); return; }
    const x = p.x + (zr.left - rect.left);
    const y = p.y + (zr.top - rect.top);
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) { setCrosshair(null); return; }
    setCrosshair({ x, y });
  });

  const handleMouseMove = (e) => {
    if (!shape || !hostRef.current) return;
    const rect = hostRef.current.getBoundingClientRect();
    if (magnifierEnabled) {
      setLensPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    }
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

  const handleMouseLeave = (e) => {
    if (magnifierEnabled) setLensPos(null);
    panRef.current = null;
    if (!shape) return;
    publish({ row: 0, col: 0, hovering: false, screenX: e.clientX, screenY: e.clientY });
  };

  const handleClick = (e) => {
    // Suppress the click-to-quantify when in linescan mode — a click+drag in
    // linescan mode should ONLY publish the line, never trigger pixel quantify.
    if (linescanMode) return;
    if (suppressClickRef.current) { suppressClickRef.current = false; return; }
    if (!shape || !hostRef.current || !onPixelClick) return;
    const rect = hostRef.current.getBoundingClientRect();
    const out = pointerToRowCol(e, zoomedRect(rect, viewRef.current), shape);
    if (out) onPixelClick(out.row, out.col);
  };

  // Plain drag pans a zoomed overlay. Shift keeps the ROI tool and linescan
  // mode keeps the line tool, so nothing existing loses its gesture.
  const handleMouseDown = (e) => {
    if (linescanMode) { onLinePointerDown(e); return; }
    if (!e.shiftKey && e.button === 0 && isZoomed(viewRef.current) && onPan) {
      panRef.current = { x: e.clientX, y: e.clientY, moved: 0 };
      suppressClickRef.current = false;
    }
    drag.onPointerDown(e);
  };
  const handleMouseUp = (e) => {
    if (linescanMode) { onLinePointerUp(e); return; }
    panRef.current = null;
    drag.onPointerUp(e);
  };

  // --- Linescan drag handlers ---
  const onLinePointerDown = (e) => {
    if (!shape || !hostRef.current) return;
    const rect = hostRef.current.getBoundingClientRect();
    const out = pointerToRowCol(e, zoomedRect(rect, viewRef.current), shape);
    if (!out) return;
    lineStartRef.current = {
      row: out.row, col: out.col,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
    setLinePts({
      x0: lineStartRef.current.x, y0: lineStartRef.current.y,
      x1: lineStartRef.current.x, y1: lineStartRef.current.y,
    });
  };
  const onLinePointerMove = (e) => {
    if (!linescanMode || !lineStartRef.current || !hostRef.current) return;
    // M2 fix: if the user released the button outside the canvas, e.buttons === 0
    // on re-entry. Treat that as cancelling the drag.
    if (e.buttons === 0) {
      lineStartRef.current = null;
      setLinePts(null);
      return;
    }
    const rect = hostRef.current.getBoundingClientRect();
    setLinePts((p) => p && ({ ...p, x1: e.clientX - rect.left, y1: e.clientY - rect.top }));
  };
  const onLinePointerUp = (e) => {
    if (!lineStartRef.current || !hostRef.current || !shape) {
      lineStartRef.current = null;
      return;
    }
    const rect = hostRef.current.getBoundingClientRect();
    const end = pointerToRowCol(e, zoomedRect(rect, viewRef.current), shape);
    if (end) {
      onLineComplete?.({
        start: { row: lineStartRef.current.row, col: lineStartRef.current.col },
        end,
      });
    }
    lineStartRef.current = null;
    // NB: keep `linePts` so the dashed line remains visible until next drag.
  };

  const zoomed = isZoomed(view);

  return (
    <div
      ref={hostRef}
      data-overlay-card-host
      onMouseDown={handleMouseDown}
      onMouseMove={(e) => {
        handleMouseMove(e);
        (linescanMode ? onLinePointerMove : drag.onPointerMove)(e);
      }}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      onDoubleClick={() => { if (!linescanMode) onResetView?.(); }}
      onContextMenu={(e) => {
        if (!onContextMenu) return;
        e.preventDefault();
        onContextMenu(e.clientX, e.clientY);
      }}
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: shape ? `${shape[1]}/${shape[0]}` : '156/128',
        background: colors.bgSecondary,
        borderRadius: 4,
        overflow: 'hidden',
        maxHeight: '55vh',
        cursor: zoomed && !linescanMode ? 'grab' : 'crosshair',
      }}
    >
      {/* Zoom wrapper — a CSS transform, so LayeredCanvas keeps compositing
          exactly as before and nothing is re-rendered for a zoom. */}
      <div
        data-overlay-zoom-layer
        style={{
          position: 'absolute', inset: 0,
          transform: viewToTransform(view),
          transformOrigin: '50% 50%',
          willChange: zoomed ? 'transform' : 'auto',
        }}
      >
        {inSwipeMode ? (
          <SwipeCanvas
            bitmapA={bitmaps.get(swipe.a)}
            bitmapB={bitmaps.get(swipe.b)}
            shape={shape}
            splitX={swipe.splitX ?? 0.5}
            onSplitChange={onSwipeSplitChange}
          />
        ) : (
          <LayeredCanvas
            layers={layers}
            bitmaps={bitmaps}
            bitmapVersion={bitmapVersion}
            perLayerErrors={errors}
          />
        )}
      </div>
      {zoomed && (
        <div data-overlay-zoom-badge style={{
          position: 'absolute', left: 6, top: 6,
          padding: '1px 6px', borderRadius: 3,
          background: 'rgba(0,0,0,.55)', color: colors.accent,
          fontSize: '8pt', fontWeight: 700, pointerEvents: 'none', zIndex: 4,
        }}>
          {view.scale.toFixed(1)}×
        </div>
      )}
      {crosshair && (
        <div
          data-overlay-crosshair
          style={{
            position: 'absolute',
            left: crosshair.x, top: crosshair.y,
            width: 14, height: 14, transform: 'translate(-50%, -50%)',
            border: `1px solid ${colors.cyan}`,
            borderRadius: 2,
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
      {linePts && linescanMode && (
        <svg data-linescan-line style={{
          position: 'absolute', inset: 0, width: '100%', height: '100%',
          pointerEvents: 'none',
        }}>
          <line
            x1={linePts.x0} y1={linePts.y0}
            x2={linePts.x1} y2={linePts.y1}
            stroke={colors.cyan} strokeWidth="2" strokeLinecap="round"
            strokeDasharray="4 3"
          />
          <circle cx={linePts.x0} cy={linePts.y0} r="3" fill={colors.cyan} />
          <circle cx={linePts.x1} cy={linePts.y1} r="3" fill={colors.cyan} />
        </svg>
      )}
      <MagnifierLens
        // Pass the host ref; the lens resolves the underlying <canvas> in
        // its own effect (reading refs during render is forbidden by
        // react-hooks/refs).
        hostRef={hostRef}
        pos={lensPos}
        visible={magnifierEnabled}
      />
    </div>
  );
}

function SwipeCanvas({ bitmapA, bitmapB, shape, splitX, onSplitChange }) {
  const { t } = useTranslation('eds');
  const canvasRef = useRef(null);
  const hostRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !shape) return;
    c.width = shape[1]; c.height = shape[0];
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    const cutPx = c.width * splitX;
    // Scale each bitmap to fill the canvas — the A and B layers may be at
    // different native resolutions (e.g. a high-res electron image vs an
    // EDS scan-grid map). drawImage with explicit w/h co-registers them.
    if (bitmapA) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, cutPx, c.height);
      ctx.clip();
      ctx.drawImage(bitmapA, 0, 0, c.width, c.height);
      ctx.restore();
    }
    if (bitmapB) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(cutPx, 0, c.width - cutPx, c.height);
      ctx.clip();
      ctx.drawImage(bitmapB, 0, 0, c.width, c.height);
      ctx.restore();
    }
  }, [bitmapA, bitmapB, shape, splitX]);

  const onHandleDown = (e) => {
    e.preventDefault();
    setDragging(true);
    const move = (ev) => {
      if (!hostRef.current) return;
      const r = hostRef.current.getBoundingClientRect();
      const t = (ev.clientX - r.left) / r.width;
      onSplitChange?.(Math.max(0.05, Math.min(0.95, t)));
    };
    const up = () => {
      setDragging(false);
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  };

  return (
    <div ref={hostRef} data-swipe-canvas style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%', height: '100%',
          imageRendering: 'pixelated',
          background: '#000', borderRadius: 4,
        }}
      />
      <div
        data-swipe-handle
        onMouseDown={onHandleDown}
        style={{
          position: 'absolute', top: 0, bottom: 0,
          left: `calc(${splitX * 100}% - 4px)`,
          width: 8,
          cursor: 'col-resize',
          background: 'rgba(189,147,249,0.45)',
          border: dragging ? '1px solid #f8f8f2' : '1px solid rgba(255,255,255,0.4)',
          borderRadius: 2,
          zIndex: 2,
        }}
        title={t('swipe.handleTooltip')}
      />
    </div>
  );
}
