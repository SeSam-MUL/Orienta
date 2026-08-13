import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { pointerToNorm, normToBoxPct } from './markerGeometry';
import { IDENTITY_VIEW, isZoomed, viewToTransform } from '../EDS/zoomView';
import { useWheelZoom } from '../common/useWheelZoom';
import { zoomRectPct } from '../EBSDViewer/zoomOverlay';

// Wraps a single pattern <img> and overlays the linked crosshair + numbered
// markers as ABSOLUTELY-POSITIONED DOM elements. These overlays are never drawn
// into the image and are not part of the export model — so they can never leak
// into an exported figure.
const RED = '#ff5555';
// A drag that travels further than this is a pan, not a click — otherwise
// panning a zoomed panel would drop a marker on release.
const CLICK_SLOP_PX = 3;

/**
 * Zoom is opt-in: pass `view` plus the callbacks and the panel becomes
 * zoomable; leave them out and this renders exactly as it always did.
 *
 * The transform sits on the <img>, not on the wrapper, for two reasons. The
 * pointer maths reads the IMG's client rect, which a CSS transform already
 * folds in — so `pointerToNorm` stays zoom-correct with no extra term. And the
 * overlays stay outside it, because a 1px crosshair inside a 16x transform
 * would render 16px thick; they are moved numerically instead.
 */
export default function LinkedPatternImage({
  src, alt, markers = [], hover = null, onHover, onClick,
  naturalSize = null, style = {},
  view = null, onZoomAt, onPan, onResetView,
}) {
  const { t } = useTranslation(['patternmatch', 'phasemap']);
  const wrapRef = useRef(null);
  const imgRef = useRef(null);
  const [natural, setNatural] = useState(naturalSize || { w: 1, h: 1 });

  const activeView = view || IDENTITY_VIEW;
  const zoomed = isZoomed(activeView);
  // The wheel listener is attached once; refs keep it reading current values.
  const viewRef = useRef(activeView);
  viewRef.current = activeView;
  const panRef = useRef(null);
  const suppressClickRef = useRef(false);

  const bindWheel = useWheelZoom(onZoomAt);

  // Measure the IMG: its client rect carries the zoom transform, so everything
  // below keeps working unchanged whether or not the panel is zoomed.
  const rectNatural = () => {
    const im = imgRef.current?.getBoundingClientRect?.();
    // Fall back to the wrapper when the image has no measurable box — before it
    // has loaded, and in tests, where only the wrapper is given a size.
    const r = (im && im.width > 0 && im.height > 0)
      ? im
      : (wrapRef.current?.getBoundingClientRect?.() || { left: 0, top: 0, width: 1, height: 1 });
    return { rect: { left: r.left, top: r.top, width: r.width, height: r.height }, natural };
  };
  const handleMove = (e) => {
    if (panRef.current) {
      if (e.buttons !== 1) {
        panRef.current = null;
      } else {
        const r = wrapRef.current?.getBoundingClientRect();
        if (r?.width && r?.height) {
          const dx = e.clientX - panRef.current.x;
          const dy = e.clientY - panRef.current.y;
          panRef.current = {
            x: e.clientX, y: e.clientY,
            moved: panRef.current.moved + Math.abs(dx) + Math.abs(dy),
          };
          if (panRef.current.moved > CLICK_SLOP_PX) suppressClickRef.current = true;
          onPan?.(dx / r.width, dy / r.height);
          return;
        }
      }
    }
    const { rect, natural: n } = rectNatural();
    onHover?.(pointerToNorm({ clientX: e.clientX, clientY: e.clientY }, rect, n));
  };
  const handleDown = (e) => {
    if (e.button === 0 && zoomed && onPan) {
      panRef.current = { x: e.clientX, y: e.clientY, moved: 0 };
      suppressClickRef.current = false;
    }
  };
  const handleUp = () => { panRef.current = null; };
  const handleClick = (e) => {
    // A drag that panned must not also drop a marker.
    if (suppressClickRef.current) { suppressClickRef.current = false; return; }
    const { rect, natural: n } = rectNatural();
    const norm = pointerToNorm({ clientX: e.clientX, clientY: e.clientY }, rect, n);
    if (norm) onClick?.(norm);
  };

  const boxRect = wrapRef.current?.getBoundingClientRect?.() || { width: 1, height: 1 };
  const placement = (nx, ny) => {
    const p = normToBoxPct({ x: nx, y: ny }, { width: boxRect.width, height: boxRect.height }, natural);
    if (!zoomed) return p;
    // Overlays live outside the transform, so carry their percentages through
    // the same view maths the transform applies.
    const z = zoomRectPct({ left: p.left, top: p.top, width: 0, height: 0 }, activeView);
    return { left: z.left, top: z.top };
  };

  return (
    <div ref={(el) => { wrapRef.current = el; bindWheel(el); }} data-linked-wrap title={t('patternmatch:linked.imageTooltip')}
      style={{
        position: 'relative', display: 'inline-block', lineHeight: 0, ...style,
        // After the caller's style on purpose: clipping the magnified image
        // to this box is correctness, not decoration — a caller must not be
        // able to switch it off by accident.
        overflow: 'hidden',
      }}
      onMouseMove={handleMove} onMouseLeave={() => { onHover?.(null); panRef.current = null; }}
      onMouseDown={handleDown} onMouseUp={handleUp} onClick={handleClick}>
      <img ref={imgRef} src={src} alt={alt} draggable={false}
        onLoad={(e) => { if (!naturalSize && e.target.naturalWidth) setNatural({ w: e.target.naturalWidth, h: e.target.naturalHeight }); }}
        style={{
          width: '100%', height: '100%', objectFit: 'contain', display: 'block',
          cursor: zoomed ? 'grab' : 'crosshair',
          transform: viewToTransform(activeView),
          transformOrigin: '50% 50%',
          willChange: zoomed ? 'transform' : 'auto',
          userSelect: 'none',
        }} />
      {hover && (() => {
        const p = placement(hover.x, hover.y);
        return (
          <div data-crosshair style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
            <div style={{ position: 'absolute', left: 0, right: 0, top: `${p.top}%`, height: 1, background: RED }} />
            <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${p.left}%`, width: 1, background: RED }} />
            <div style={{ position: 'absolute', left: `${p.left}%`, top: `${p.top}%`, width: 6, height: 6, borderRadius: '50%', background: RED, transform: 'translate(-50%,-50%)' }} />
          </div>
        );
      })()}
      {markers.map((m) => {
        const p = placement(m.x, m.y);
        return (
          <div key={m.id} data-marker={m.n} title={t('patternmatch:linked.markerTooltip', { n: m.n })}
            style={{ position: 'absolute', left: `${p.left}%`, top: `${p.top}%`, width: 16, height: 16,
              marginLeft: -8, marginTop: -8, borderRadius: '50%', border: `2px solid ${RED}`,
              color: RED, fontSize: 9, fontWeight: 700, display: 'flex', alignItems: 'center',
              justifyContent: 'center', pointerEvents: 'none', boxShadow: '0 0 0 1px rgba(0,0,0,0.6)' }}>
            {m.n}
          </div>
        );
      })}
      {zoomed && (
        <div
          data-pattern-zoom-badge
          onClick={(e) => { e.stopPropagation(); onResetView?.(); }}
          title={t('phasemap:hoverTips.zoomReset')}
          style={{
            position: 'absolute', top: 4, left: 4, padding: '1px 6px', borderRadius: 9,
            background: 'rgba(0,0,0,.6)', color: '#fff', fontSize: '8pt',
            lineHeight: 1.5, cursor: 'pointer', userSelect: 'none',
          }}
        >
          {activeView.scale.toFixed(1)}×
        </div>
      )}
    </div>
  );
}
