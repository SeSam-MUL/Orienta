import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { umToScreenPx, niceLength } from '../scalebarGeometry';
import { formatScaleValue, stopsToGradient } from '../scaleFormat';
import { valueScaleFontPx } from './composeExport';

/**
 * How much room a scale bar actually needs, in pixels.
 *
 * The bar's length is physics — µm through step size, letterboxing and zoom —
 * so the widget cannot have an independent width. Storing one (the default was
 * 30 % of the map) produced a selection frame five times wider than the bar it
 * held, with resize handles that changed nothing visible. The frame is now
 * derived from the same number the bar is drawn with.
 *
 * Returns null when the geometry is not known yet; the caller then falls back
 * to the stored box rather than collapsing the widget to nothing.
 */
export function scalebarBoxSize(annot, ctx) {
  const barPx = umToScreenPx({
    um: annot.props?.lengthUm ?? 5,
    nativeSize: ctx?.mapNativeSize,
    contentBbox: ctx?.mapContentBbox,
    stepX: ctx?.stepX,
    fitWidth: ctx?.mapBoxWidthPx,
    zoomScale: ctx?.mapZoomScale,
  });
  if (!barPx) return null;
  const fontSize = annot.props?.fontSize ?? 12;
  // A plate needs padding of its own, or the colour would stop dead at the
  // bar's ends and read as clipped.
  const PAD = backgroundCss(annot.props) === 'transparent' ? 4 : 9;
  const barH = Math.max(3, fontSize * 0.4);
  return {
    w: barPx + 2 * PAD,
    // bar + gap + one line of text
    h: barH + 2 + fontSize * 1.3 + 2 * PAD,
  };
}

/**
 * The plate an annotation sits on: a colour and how much of it comes through.
 *
 * One rule for all four types, because "put something behind it so it reads on
 * a bright map" is the same wish for a legend, a scale bar, a title and an
 * arrow. Opacity 0 — the default everywhere except the legend — means no plate
 * at all, so existing figures keep the look they had.
 */
export function backgroundCss(props) {
  // Legends used to store a ready-made CSS colour. Honour it until the user
  // touches the new controls, or their plate would vanish on upgrade.
  const legacy = props?.background;
  const opacity = props?.bgOpacity;
  if (opacity == null && typeof legacy === 'string' && legacy) return legacy;
  const a = Math.max(0, Math.min(1, Number(opacity) || 0));
  if (a <= 0) return 'transparent';
  const hex = String(props?.bgColor || '#000000').replace('#', '');
  const full = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex;
  const r = parseInt(full.slice(0, 2), 16) || 0;
  const g = parseInt(full.slice(2, 4), 16) || 0;
  const b = parseInt(full.slice(4, 6), 16) || 0;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

const PLATE_RADIUS = 4;

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

/**
 * On the map an annotation stays on the map — dragged off, it would simply be
 * invisible. In the export dialog the picture can have a border around it, and
 * putting the colour key BESIDE the data instead of on top of it is the normal
 * way to compose a figure, so there the box opens up.
 *
 * Still bounded: a body that ran away by a whole picture width could not be
 * grabbed again. Coordinates stay relative to the MAP either way, so widening
 * the border does not move anything.
 */
const OUTSIDE_REACH = 1.5;
function clampOutside(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(-OUTSIDE_REACH, Math.min(1 + OUTSIDE_REACH, v));
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
      background: backgroundCss(annot.props),
      borderRadius: PLATE_RADIUS, padding: '6px 8px',
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

function ScalebarBody({ annot, ctx, containerSize }) {
  const { t } = useTranslation('phasemap');
  const noScaleLabel = t('phasemap:annotations.scalebarNoScale');
  const lengthUm = annot.props?.lengthUm ?? 5;
  const fontSize = annot.props?.fontSize ?? 12;
  const barColor = annot.props?.barColor ?? '#ffffff';
  const textColor = annot.props?.textColor ?? '#ffffff';

  // How long is `lengthUm` on this screen, right now?
  //
  // The previous version worked it out as a fraction of the SCAN GRID
  // (lengthUm/stepX/scanCols) and multiplied by `canvasWidthPx` — a value the
  // page never passed, so the whole branch was skipped and the bar simply
  // filled its box at 100%. The label said 5 µm and the bar was whatever width
  // the box had been dragged to.
  //
  // Two things that maths missed even with the value supplied: the map is drawn
  // letterboxed inside the container (so a fraction of the CONTAINER is not a
  // fraction of the map), and the view auto-zooms to the indexed region, which
  // magnifies it. `umToScreenPx` folds in both, plus any user zoom, exactly the
  // way the layer-stack scalebar does — the two are checked against each other.
  //
  // The exported figure is a different picture (the full grid, unzoomed), and
  // composeExport keeps its own correct maths for that.
  const barPx = umToScreenPx({
    um: lengthUm,
    nativeSize: ctx?.mapNativeSize,
    contentBbox: ctx?.mapContentBbox,
    stepX: ctx?.stepX,
    fitWidth: ctx?.mapBoxWidthPx,
    zoomScale: ctx?.mapZoomScale,
  });
  // In PIXELS, and deliberately NOT clamped to the widget box. The box is a
  // frame the user drags around for positioning; the bar's length is physics.
  // Clamping it to the box (what the previous version did) made the bar stop
  // growing while the label still claimed 5 µm — measured 13 % short at 3.8x
  // zoom. If the bar outgrows its frame it now visibly sticks out, which tells
  // the user to widen the frame or pick a shorter length.
  // Without a usable geometry it falls back to filling the box rather than
  // stating a length it cannot back up: with no usable geometry it says so
  // rather than drawing a full-width bar under a micrometre label. That
  // combination — a bar of arbitrary length beneath a confident "5 µm" — is
  // the exact failure this annotation exists to avoid, and the map has no
  // step size to back it whenever `ctx.stepX` is null (see `knownStepX` in
  // PhaseMapPage: a 1.0 placeholder is not a measurement).
  return (
    <div style={{
      width: '100%', height: '100%',
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', gap: 2,
      background: backgroundCss(annot.props), borderRadius: PLATE_RADIUS,
      boxSizing: 'border-box',
    }}>
      {barPx ? (
        <>
          <div style={{
            width: `${barPx}px`,
            flexShrink: 0,
            height: Math.max(3, fontSize * 0.4),
            background: barColor, borderRadius: 1,
          }} />
          <div style={{
            color: textColor, fontSize, fontFamily: 'sans-serif',
            textShadow: '0 0 4px rgba(0,0,0,0.8)',
          }}>
            {lengthUm} µm
          </div>
        </>
      ) : (
        <div
          data-scalebar-noscale
          style={{
            color: textColor, fontSize: Math.max(9, fontSize * 0.85),
            fontFamily: 'sans-serif', fontStyle: 'italic', opacity: 0.85,
            textAlign: 'center', padding: '0 4px',
            textShadow: '0 0 4px rgba(0,0,0,0.8)',
          }}
        >
          {noScaleLabel}
        </div>
      )}
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
      background: backgroundCss(annot.props), borderRadius: PLATE_RADIUS,
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
    <div style={{
      width: '100%', height: '100%',
      background: backgroundCss(annot.props), borderRadius: PLATE_RADIUS,
      boxSizing: 'border-box',
    }}>
      <svg
        width="100%" height="100%"
        viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet"
        style={{ overflow: 'visible', display: 'block' }}
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
    </div>
  );
}

/**
 * The IPF colour key as a body you can move.
 *
 * The picture is the very PNG the backend rendered for THIS map (direction,
 * phase filter and the user's colour picks are already in it), handed down
 * through the context — never stored on the annotation, which would freeze a
 * key that later belongs to a different map.
 */
function ColorKeyBody({ annot, ctx }) {
  const raw = ctx?.ipfKeyImage;
  const src = raw ? (raw.startsWith('data:') ? raw : `data:image/png;base64,${raw}`) : null;
  return (
    <div style={{
      width: '100%', height: '100%', boxSizing: 'border-box',
      background: backgroundCss(annot.props), borderRadius: PLATE_RADIUS,
      padding: 4, display: 'flex', alignItems: 'center', justifyContent: 'center',
      overflow: 'hidden',
    }}>
      {src ? (
        <img
          src={src} alt=""
          draggable={false}
          style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', display: 'block' }}
        />
      ) : null}
    </div>
  );
}

/**
 * One value scale as a body you can move: caption, colour bar, ends + middle.
 *
 * Mirrors what `drawValueScale` burns into the file, and reads its numbers from
 * the same live layer, so the preview cannot promise a bar the export does not
 * draw.
 */
function ValueScaleBody({ annot, ctx, box }) {
  const p = annot.props || {};
  const entry = (ctx?.scaleLegends || []).find((e) => e?.id === p.layerId);
  const scale = entry?.scale;
  // Sized from the body, exactly as the exporter sizes it — so what is on
  // screen is what lands in the file.
  const fontSize = valueScaleFontPx(box?.w ?? 0, box?.h ?? 0, p.textScale);
  const color = p.textColor ?? '#ffffff';
  const plate = {
    width: '100%', height: '100%', boxSizing: 'border-box',
    background: backgroundCss(annot.props), borderRadius: PLATE_RADIUS,
    padding: 4, overflow: 'hidden',
  };
  if (!scale || !Number.isFinite(scale.min) || !Number.isFinite(scale.max)) {
    return <div style={{ ...plate, color: '#888', fontSize: fontSize * 0.9 }}>—</div>;
  }
  return (
    <div style={{
      ...plate,
      display: 'flex', flexDirection: 'column', gap: 2,
      color, fontSize, fontFamily: 'sans-serif', lineHeight: 1.2,
    }}>
      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flexShrink: 0 }}>
        {entry.label}
      </div>
      <div style={{ display: 'flex', alignItems: 'stretch', gap: 4, flex: 1, minHeight: 0 }}>
        <div style={{
          width: '34%', minWidth: 6,
          border: '1px solid rgba(0,0,0,0.55)', borderRadius: 1,
          background: stopsToGradient(scale.stops),
        }} />
        <div style={{
          display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
          fontVariantNumeric: 'tabular-nums',
        }}>
          <span>{formatScaleValue(scale.max)}</span>
          <span>{formatScaleValue((scale.min + scale.max) / 2)}</span>
          <span>{formatScaleValue(scale.min)}</span>
        </div>
      </div>
      {scale.unit ? (
        <div style={{ flexShrink: 0 }}>{scale.unit}</div>
      ) : null}
    </div>
  );
}

// `box` is the widget's own size in screen pixels — the scale body sizes its
// lettering from it, the way the exporter sizes it from the box in the file.
function renderBody(annot, ctx, containerSize, box) {
  switch (annot.type) {
    case 'legend':   return <LegendBody annot={annot} phaseStats={ctx.phaseStats} />;
    case 'scalebar': return (
      <ScalebarBody annot={annot} ctx={ctx} containerSize={containerSize} />
    );
    case 'title':      return <TitleBody annot={annot} />;
    case 'arrow':      return <ArrowBody annot={annot} />;
    case 'colorkey':   return <ColorKeyBody annot={annot} ctx={ctx} />;
    case 'valuescale': return <ValueScaleBody annot={annot} ctx={ctx} box={box} />;
    default:           return <div style={{ color: '#888' }}>?{annot.type}</div>;
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
  // A scale bar's frame hugs the bar; every other annotation keeps the box the
  // user dragged.
  const naturalBox = annot.type === 'scalebar' ? scalebarBoxSize(annot, ctx) : null;
  // NOTE: "nothing to draw yet" is decided AFTER every hook below has run.
  // Bailing out here (what this did before) skipped the drag hooks on the very
  // first paint, and the next paint — once the container had measured — called
  // more hooks than the one before it, which React refuses to reconcile.
  // On the map that first paint always had a measured container, so it never
  // showed; in the export dialog host and layer mount together and it crashed
  // the page.
  const measured = rectW > 0 && rectH > 0;
  const left = annot.x * rectW;
  const top = annot.y * rectH;
  const width = naturalBox ? naturalBox.w : annot.w * rectW;
  const height = naturalBox ? naturalBox.h : annot.h * rectH;

  // ----- Drag (whole-body) ---------------------------------------------------
  const dragRef = useRef(null);
  const handlersRef = useRef({ move: null, up: null });
  const winMove = useRef((ev) => handlersRef.current.move?.(ev)).current;
  const winUp = useRef(() => handlersRef.current.up?.()).current;
  const onMouseDownBody = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    onSelect(annot.id);
    const startX = e.clientX, startY = e.clientY;
    const startAx = annot.x, startAy = annot.y;
    dragRef.current = { mode: 'drag', startX, startY, startAx, startAy };
    window.addEventListener('mousemove', winMove);
    window.addEventListener('mouseup', winUp);
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
      // For the scale bar: where the drag started, in the units the bar is
      // actually measured in.
      startUm: annot.props?.lengthUm ?? 5,
      startBarPx: naturalBox ? naturalBox.w - 8 : 0,
    };
    window.addEventListener('mousemove', winMove);
    window.addEventListener('mouseup', winUp);
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
    window.addEventListener('mousemove', winMove);
    window.addEventListener('mouseup', winUp);
  };

  const clampPos = ctx?.allowOutside ? clampOutside : clamp01;
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
        x: clampPos(d.startAx + dx),
        y: clampPos(d.startAy + dy),
      });
    } else if (d.mode === 'resize') {
      // A scale bar has no free width — its length is µm through the step
      // size. Dragging a corner therefore changes the LENGTH, snapped to the
      // 1/2/5 ladder a reader expects, and the frame follows. Before this the
      // handles moved a box the bar ignored, which is what "the box does not
      // work" meant.
      if (annot.type === 'scalebar' && d.startBarPx > 0) {
        const grow = d.corner.includes('e') ? 1 : -1;
        const nextPx = Math.max(8, d.startBarPx + grow * (ev.clientX - d.startX));
        const raw = (d.startUm || 5) * (nextPx / d.startBarPx);
        const snapped = niceLength(raw);
        if (snapped && snapped !== annot.props?.lengthUm) {
          onUpdate(annot.id, { props: { lengthUm: snapped } });
        }
        return;
      }
      const dx = (ev.clientX - d.startX) / rect2.width;
      const dy = (ev.clientY - d.startY) / rect2.height;
      let nx = d.startAx, ny = d.startAy, nw = d.startAw, nh = d.startAh;
      if (d.corner.includes('e')) nw = Math.max(0.02, d.startAw + dx);
      if (d.corner.includes('s')) nh = Math.max(0.02, d.startAh + dy);
      if (d.corner.includes('w')) { nx = clampPos(d.startAx + dx); nw = Math.max(0.02, d.startAw - dx); }
      if (d.corner.includes('n')) { ny = clampPos(d.startAy + dy); nh = Math.max(0.02, d.startAh - dy); }
      onUpdate(annot.id, { x: nx, y: ny, w: nw, h: nh });
    } else if (d.mode === 'rotate') {
      const ang = Math.atan2(
        ev.clientY - (d.rectTop + d.cy),
        ev.clientX - (d.rectLeft + d.cx),
      ) * 180 / Math.PI;
      onUpdate(annot.id, { rotation: d.startRot + (ang - d.startAngle) });
    }
  }, [annot.id, containerRef, onUpdate, clampPos]);

  const onWindowUp = useCallback(() => {
    dragRef.current = null;
    window.removeEventListener('mousemove', winMove);
    window.removeEventListener('mouseup', winUp);
  }, [winMove, winUp]);

  // The window listeners are these two forwarders, whose identity never
  // changes; they look up the current handlers when they fire. Registering the
  // handlers themselves ends the drag as soon as the parent hands down a fresh
  // onUpdate, because the cleanup below then sees "different function" and
  // unregisters mid-gesture. The map's parent memoises its callbacks so it got
  // away with it; the export dialog's does not, and there the annotation
  // stopped after the first mouse-move.
  handlersRef.current.move = onWindowMove;
  handlersRef.current.up = onWindowUp;

  useEffect(() => () => {
    window.removeEventListener('mousemove', winMove);
    window.removeEventListener('mouseup', winUp);
  }, [winMove, winUp]);

  // ----- Render ---------------------------------------------------------------
  if (!measured) return null;
  const wrapperStyle = {
    position: 'absolute',
    left, top, width, height,
    transform: `rotate(${annot.rotation || 0}deg)`,
    transformOrigin: 'center center',
    cursor: 'move',
    boxSizing: 'border-box',
    border: isSelected ? '1px dashed #50fa7b' : '1px dashed rgba(255,255,255,0.0)',
    userSelect: 'none',
    // The annotation is the only part of this layer that takes the pointer.
    pointerEvents: 'auto',
  };
  const handleStyle = {
    position: 'absolute', width: HANDLE, height: HANDLE,
    background: '#50fa7b', border: '1px solid #1a1b26',
    boxSizing: 'border-box', pointerEvents: 'auto', cursor: 'nwse-resize',
  };

  return (
    <div data-annotation-widget style={wrapperStyle} onMouseDown={onMouseDownBody}>
      {renderBody(annot, ctx, containerSize, { w: width, h: height })}
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
  // How wide the MAP is drawn right now, as opposed to the container it sits
  // letterboxed inside. Measured here rather than taken from `ctx`, because the
  // page builds `ctx` in its render body and `LayeredCanvas` refits itself from
  // its own ResizeObserver: after a window resize the map's drawn width changes
  // without the page re-rendering, and a bar sized from the frozen value states
  // a length it no longer has. 0 = there is no map box in this tree (the export
  // dialog reuses these widgets over its preview), and then `ctx` is the live
  // answer after all.
  const [mapBoxWidth, setMapBoxWidth] = useState(0);
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
      const box = el.querySelector?.('[data-phasemap-map-box]');
      const w = box ? box.getBoundingClientRect().width : 0;
      setMapBoxWidth((prev) => (prev === w ? prev : w));
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

  // Deselect on a press that missed every annotation. This used to be a
  // full-bleed div with pointer events on; that div also ate the map's clicks.
  // Listening on the container instead leaves the map fully usable.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const onDown = (e) => {
      if (e.target.closest?.('[data-annotation-widget]')) return;
      onSelect(null);
    };
    el.addEventListener('mousedown', onDown);
    return () => el.removeEventListener('mousedown', onDown);
  }, [containerRef, onSelect]);

  // Merge canvas size into ctx so per-annotation drawers can use it — and let
  // the live map-box measurement win over whatever the page froze in.
  const enrichedCtx = {
    ...ctx,
    canvasWidthPx: size.width,
    canvasHeightPx: size.height,
    ...(mapBoxWidth > 0 ? { mapBoxWidthPx: mapBoxWidth } : null),
  };

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
    >
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
        {/* inset:0 makes each of these cover the ENTIRE map, so they must stay
            transparent to the pointer. With `auto` here — which is how it was —
            a single annotation laid a full-screen click catcher over the map and
            neither left- nor right-click reached it any more. The widget inside
            re-enables pointer events for its own box. */}
        {annotations.map((a) => (
          <div key={a.id} style={{ pointerEvents: 'none', position: 'absolute', inset: 0 }}>
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
