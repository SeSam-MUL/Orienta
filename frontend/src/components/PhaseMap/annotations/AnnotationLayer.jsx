import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Floating overlay on the Phase Maps canvas. Hosts user-positioned
 * Legend, Scalebar, Title text and North-Arrow widgets.
 *
 * Geometry: each annotation has normalised (x, y, w, h) in [0..1]
 * relative to the canvas bounding-rect. We render absolute-positioned
 * divs and translate normalised coords to pixel coords using a
 * containerRef'd parent.
 *
 * Interactions:
 *   - Drag (mousedown on body) -> updates (x, y)
 *   - Resize handles on the 4 corners -> updates (w, h)
 *   - Rotation handle above the box -> updates rotation
 *   - Delete via X button when selected
 *
 * Selection: clicking a widget selects it (border highlight) and shows
 * its props panel below the legend palette. Clicking the empty canvas
 * deselects.
 */

const HANDLE = 8;       // px — corner resize handle size
const ROT_OFFSET = 22;  // px — rotation handle distance above box

function clamp01(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/** Convert a mouse event to normalised (x, y) inside the container. */
function evToNorm(e, containerEl) {
  const rect = containerEl.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / Math.max(1, rect.width);
  const ny = (e.clientY - rect.top) / Math.max(1, rect.height);
  return { nx, ny, rect };
}

// ----- Annotation body renderers ---------------------------------------------

function LegendBody({ annot, phaseStats }) {
  const { t } = useTranslation('phasemap');
  const phases = phaseStats?.phases || [];
  const fontSize = annot.props?.fontSize ?? 11;
  return (
    <div style={{
      width: '100%', height: '100%',
      background: annot.props?.background || 'rgba(20,22,30,0.85)',
      borderRadius: 4, padding: '6px 8px',
      color: '#eaeaea', fontSize, lineHeight: 1.4,
      overflow: 'hidden',
      boxSizing: 'border-box',
    }}>
      {phases.length === 0 && (
        <div style={{ color: '#888', fontSize: fontSize - 1 }}>
          {t('phasemap:annotations.noPhasesYet')}
        </div>
      )}
      {phases.map((p) => (
        <div key={p.phase_id}
             style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 1 }}>
          <span style={{
            display: 'inline-block', width: fontSize, height: fontSize,
            background: p.color_hex, border: '1px solid #1118', borderRadius: 2,
            flex: '0 0 auto',
          }} />
          <span style={{
            flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>{p.name}</span>
          <span style={{ color: '#a5a8b0', fontSize: fontSize - 1 }}>
            {p.area_pct?.toFixed?.(1) ?? '?'}%
          </span>
        </div>
      ))}
    </div>
  );
}

function ScalebarBody({ annot, stepX, scanCols, canvasWidthPx }) {
  const lengthUm = annot.props?.lengthUm ?? 5;
  const fontSize = annot.props?.fontSize ?? 12;
  const barColor = annot.props?.barColor ?? '#ffffff';
  const textColor = annot.props?.textColor ?? '#ffffff';
  // Compute the PHYSICAL width the bar should occupy:
  //   lengthUm / stepX  -> scan pixels
  //   / scanCols        -> fraction of the scan grid
  //   * canvasWidthPx   -> canvas pixels
  // Divided by the annotation box's pixel width to express as a percent
  // of the BOX (so we can render with `width: NN%`). When we can't
  // derive (stepX or scanCols missing), fall back to 100% so the user
  // still sees a positionable bar.
  let barWidthPct = 100;
  if (stepX && scanCols && canvasWidthPx && annot.w > 0) {
    const wantedFractionOfCanvas = (lengthUm / stepX) / scanCols;
    const wantedCanvasPx = wantedFractionOfCanvas * canvasWidthPx;
    const boxPx = annot.w * canvasWidthPx;
    barWidthPct = Math.max(2, Math.min(100, (wantedCanvasPx / boxPx) * 100));
  }
  return (
    <div style={{
      width: '100%', height: '100%',
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', gap: 2,
      background: 'transparent', boxSizing: 'border-box',
    }}>
      <div style={{
        width: `${barWidthPct}%`,
        height: Math.max(3, fontSize * 0.4),
        background: barColor, borderRadius: 1,
      }} />
      <div style={{
        color: textColor, fontSize, fontFamily: 'sans-serif',
        textShadow: '0 0 4px rgba(0,0,0,0.8)',
      }}>
        {lengthUm} µm
      </div>
    </div>
  );
}

function TitleBody({ annot }) {
  const { t } = useTranslation('phasemap');
  const text = annot.props?.text ?? t('phasemap:annotations.defaultTitle');
  const fontSize = annot.props?.fontSize ?? 16;
  const color = annot.props?.color ?? '#ffffff';
  return (
    <div style={{
      width: '100%', height: '100%',
      display: 'flex', alignItems: 'center', justifyContent: 'flex-start',
      padding: '0 6px', boxSizing: 'border-box',
      color, fontSize, fontWeight: 600, fontFamily: 'sans-serif',
      textShadow: '0 0 4px rgba(0,0,0,0.7)',
      overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis',
    }}>
      {text}
    </div>
  );
}

function ArrowBody({ annot }) {
  const label = annot.props?.label ?? 'ND';
  const color = annot.props?.color ?? '#ffb86c';
  const fontSize = annot.props?.fontSize ?? 11;
  return (
    <svg
      width="100%" height="100%"
      viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet"
      style={{ overflow: 'visible' }}
    >
      <defs>
        <marker
          id="arrowhead" markerWidth="10" markerHeight="7"
          refX="9" refY="3.5" orient="auto"
        >
          <polygon points="0 0, 10 3.5, 0 7" fill={color} />
        </marker>
      </defs>
      <line x1="50" y1="90" x2="50" y2="20"
            stroke={color} strokeWidth="4" markerEnd="url(#arrowhead)" />
      <text x="50" y="98" textAnchor="middle"
            fill={color} fontSize={fontSize * 4}
            style={{ fontFamily: 'sans-serif', fontWeight: 700 }}>
        {label}
      </text>
    </svg>
  );
}

function renderBody(annot, ctx) {
  switch (annot.type) {
    case 'legend':   return <LegendBody annot={annot} phaseStats={ctx.phaseStats} />;
    case 'scalebar': return (
      <ScalebarBody
        annot={annot}
        stepX={ctx.stepX}
        scanCols={ctx.scanCols}
        canvasWidthPx={ctx.canvasWidthPx}
      />
    );
    case 'title':    return <TitleBody annot={annot} />;
    case 'arrow':    return <ArrowBody annot={annot} />;
    default:         return <div style={{ color: '#888' }}>?{annot.type}</div>;
  }
}

// ----- One annotation widget -------------------------------------------------

function AnnotationWidget({
  annot, isSelected, onSelect, onDelete, onUpdate, containerRef, containerSize, ctx,
}) {
  const { t } = useTranslation('phasemap');
  // Convert normalised geom to absolute pixel rect for rendering.
  // ``containerSize`` is the live ResizeObserver measurement from the
  // parent AnnotationLayer; falling back to the ref-snapshot keeps the
  // widget renderable even if size hasn't measured yet (rare).
  const rectW = containerSize?.width || containerRef.current?.getBoundingClientRect().width || 0;
  const rectH = containerSize?.height || containerRef.current?.getBoundingClientRect().height || 0;
  if (rectW <= 0 || rectH <= 0) return null;
  const left = annot.x * rectW;
  const top = annot.y * rectH;
  const width = annot.w * rectW;
  const height = annot.h * rectH;

  // ----- Drag (whole-body) ---------------------------------------------------
  const dragRef = useRef(null);
  const onMouseDownBody = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    onSelect(annot.id);
    const startX = e.clientX, startY = e.clientY;
    const startAx = annot.x, startAy = annot.y;
    dragRef.current = { mode: 'drag', startX, startY, startAx, startAy };
    window.addEventListener('mousemove', onWindowMove);
    window.addEventListener('mouseup', onWindowUp);
  };
  const onMouseDownResize = (corner) => (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    onSelect(annot.id);
    const startX = e.clientX, startY = e.clientY;
    dragRef.current = {
      mode: 'resize', corner,
      startX, startY,
      startAx: annot.x, startAy: annot.y, startAw: annot.w, startAh: annot.h,
    };
    window.addEventListener('mousemove', onWindowMove);
    window.addEventListener('mouseup', onWindowUp);
  };
  const onMouseDownRotate = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    onSelect(annot.id);
    const containerEl2 = containerRef.current;
    if (!containerEl2) return;
    const rectNow = containerEl2.getBoundingClientRect();
    const cx = left + width / 2;
    const cy = top + height / 2;
    const startAngle = Math.atan2(e.clientY - (rectNow.top + cy), e.clientX - (rectNow.left + cx)) * 180 / Math.PI;
    dragRef.current = {
      mode: 'rotate', cx, cy, rectLeft: rectNow.left, rectTop: rectNow.top,
      startRot: annot.rotation || 0, startAngle,
    };
    window.addEventListener('mousemove', onWindowMove);
    window.addEventListener('mouseup', onWindowUp);
  };

  const onWindowMove = useCallback((ev) => {
    const d = dragRef.current;
    if (!d) return;
    const containerEl2 = containerRef.current;
    if (!containerEl2) return;
    const rect2 = containerEl2.getBoundingClientRect();
    if (d.mode === 'drag') {
      const dx = (ev.clientX - d.startX) / rect2.width;
      const dy = (ev.clientY - d.startY) / rect2.height;
      onUpdate(annot.id, {
        x: clamp01(d.startAx + dx),
        y: clamp01(d.startAy + dy),
      });
    } else if (d.mode === 'resize') {
      const dx = (ev.clientX - d.startX) / rect2.width;
      const dy = (ev.clientY - d.startY) / rect2.height;
      let nx = d.startAx, ny = d.startAy, nw = d.startAw, nh = d.startAh;
      if (d.corner.includes('e')) nw = Math.max(0.02, d.startAw + dx);
      if (d.corner.includes('s')) nh = Math.max(0.02, d.startAh + dy);
      if (d.corner.includes('w')) { nx = clamp01(d.startAx + dx); nw = Math.max(0.02, d.startAw - dx); }
      if (d.corner.includes('n')) { ny = clamp01(d.startAy + dy); nh = Math.max(0.02, d.startAh - dy); }
      onUpdate(annot.id, { x: nx, y: ny, w: nw, h: nh });
    } else if (d.mode === 'rotate') {
      const ang = Math.atan2(
        ev.clientY - (d.rectTop + d.cy),
        ev.clientX - (d.rectLeft + d.cx),
      ) * 180 / Math.PI;
      onUpdate(annot.id, { rotation: d.startRot + (ang - d.startAngle) });
    }
  }, [annot.id, containerRef, onUpdate]);

  const onWindowUp = useCallback(() => {
    dragRef.current = null;
    window.removeEventListener('mousemove', onWindowMove);
    window.removeEventListener('mouseup', onWindowUp);
  }, [onWindowMove]);

  useEffect(() => () => {
    window.removeEventListener('mousemove', onWindowMove);
    window.removeEventListener('mouseup', onWindowUp);
  }, [onWindowMove, onWindowUp]);

  // ----- Render ---------------------------------------------------------------
  const wrapperStyle = {
    position: 'absolute',
    left, top, width, height,
    transform: `rotate(${annot.rotation || 0}deg)`,
    transformOrigin: 'center center',
    cursor: 'move',
    boxSizing: 'border-box',
    border: isSelected ? '1px dashed #50fa7b' : '1px dashed rgba(255,255,255,0.0)',
    userSelect: 'none',
  };
  const handleStyle = {
    position: 'absolute', width: HANDLE, height: HANDLE,
    background: '#50fa7b', border: '1px solid #1a1b26',
    boxSizing: 'border-box', pointerEvents: 'auto', cursor: 'nwse-resize',
  };

  return (
    <div style={wrapperStyle} onMouseDown={onMouseDownBody}>
      {renderBody(annot, ctx)}
      {isSelected && (
        <>
          {/* Corner handles */}
          <div style={{ ...handleStyle, left: -HANDLE/2, top: -HANDLE/2, cursor: 'nwse-resize' }}
               onMouseDown={onMouseDownResize('nw')} />
          <div style={{ ...handleStyle, right: -HANDLE/2, top: -HANDLE/2, cursor: 'nesw-resize' }}
               onMouseDown={onMouseDownResize('ne')} />
          <div style={{ ...handleStyle, left: -HANDLE/2, bottom: -HANDLE/2, cursor: 'nesw-resize' }}
               onMouseDown={onMouseDownResize('sw')} />
          <div style={{ ...handleStyle, right: -HANDLE/2, bottom: -HANDLE/2, cursor: 'nwse-resize' }}
               onMouseDown={onMouseDownResize('se')} />
          {/* Rotation handle (top) */}
          <div
            onMouseDown={onMouseDownRotate}
            title={t('phasemap:annotations.rotateTooltip')}
            style={{
              position: 'absolute', left: '50%', top: -ROT_OFFSET,
              transform: 'translateX(-50%)',
              width: HANDLE + 4, height: HANDLE + 4, borderRadius: '50%',
              background: '#bd93f9', border: '1px solid #1a1b26',
              cursor: 'grab', pointerEvents: 'auto',
            }}
          />
          {/* Delete X */}
          <button
            onClick={(e) => { e.stopPropagation(); onDelete(annot.id); }}
            onMouseDown={(e) => e.stopPropagation()}
            title={t('phasemap:annotations.deleteTooltip')}
            style={{
              position: 'absolute', right: -10, top: -ROT_OFFSET - 4,
              width: 16, height: 16, borderRadius: '50%',
              background: '#ff5555', color: '#fff', border: 'none',
              cursor: 'pointer', lineHeight: '14px', fontSize: 10, padding: 0,
            }}
          >×</button>
        </>
      )}
    </div>
  );
}

// ----- Layer host -------------------------------------------------------------

export default function AnnotationLayer({
  annotations, selectedId, onSelect, onDelete, onUpdate,
  containerRef, ctx,
}) {
  // Track container size so the inner widgets re-render after the
  // container measures (and on subsequent resizes). Without this the
  // widgets render once with rect={0,0,0,0} (because containerRef was
  // null on the first paint) and never measure again — invisible /
  // un-clickable widgets.
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => {
      const rect = el.getBoundingClientRect();
      setSize((prev) => (
        prev.width === rect.width && prev.height === rect.height
          ? prev
          : { width: rect.width, height: rect.height }
      ));
    };
    measure();  // initial
    let ro = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(measure);
      ro.observe(el);
    }
    window.addEventListener('resize', measure);
    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [containerRef]);

  // Merge canvas size into ctx so per-annotation drawers can use it.
  const enrichedCtx = { ...ctx, canvasWidthPx: size.width, canvasHeightPx: size.height };

  return (
    <div
      // Wrapper is full-bleed inside the canvas container; pointer-events
      // are PASS-THROUGH (so the underlying canvas still receives clicks
      // that miss any annotation), but each widget re-enables them.
      style={{
        position: 'absolute', inset: 0,
        pointerEvents: 'none', overflow: 'visible',
        zIndex: 5,
      }}
      onMouseDown={() => onSelect(null)}
    >
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
        {annotations.map((a) => (
          <div key={a.id} style={{ pointerEvents: 'auto', position: 'absolute', inset: 0 }}>
            <AnnotationWidget
              annot={a}
              isSelected={selectedId === a.id}
              onSelect={onSelect}
              onDelete={onDelete}
              onUpdate={onUpdate}
              containerRef={containerRef}
              containerSize={size}
              ctx={enrichedCtx}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
