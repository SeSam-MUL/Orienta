import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { pointerToNorm, normToBoxPct } from './markerGeometry';

// Wraps a single pattern <img> and overlays the linked crosshair + numbered
// markers as ABSOLUTELY-POSITIONED DOM elements. These overlays are never drawn
// into the image and are not part of the export model — so they can never leak
// into an exported figure.
const RED = '#ff5555';

export default function LinkedPatternImage({
  src, alt, markers = [], hover = null, onHover, onClick,
  naturalSize = null, style = {},
}) {
  const { t } = useTranslation('patternmatch');
  const wrapRef = useRef(null);
  const [natural, setNatural] = useState(naturalSize || { w: 1, h: 1 });

  const rectNatural = () => {
    const r = wrapRef.current?.getBoundingClientRect?.() || { left: 0, top: 0, width: 1, height: 1 };
    return { rect: { left: r.left, top: r.top, width: r.width, height: r.height }, natural };
  };
  const handleMove = (e) => {
    const { rect, natural: n } = rectNatural();
    onHover?.(pointerToNorm({ clientX: e.clientX, clientY: e.clientY }, rect, n));
  };
  const handleClick = (e) => {
    const { rect, natural: n } = rectNatural();
    const norm = pointerToNorm({ clientX: e.clientX, clientY: e.clientY }, rect, n);
    if (norm) onClick?.(norm);
  };

  const boxRect = wrapRef.current?.getBoundingClientRect?.() || { width: 1, height: 1 };
  const placement = (nx, ny) =>
    normToBoxPct({ x: nx, y: ny }, { width: boxRect.width, height: boxRect.height }, natural);

  return (
    <div ref={wrapRef} data-linked-wrap title={t('patternmatch:linked.imageTooltip')}
      style={{ position: 'relative', display: 'inline-block', lineHeight: 0, ...style }}
      onMouseMove={handleMove} onMouseLeave={() => onHover?.(null)} onClick={handleClick}>
      <img src={src} alt={alt}
        onLoad={(e) => { if (!naturalSize && e.target.naturalWidth) setNatural({ w: e.target.naturalWidth, h: e.target.naturalHeight }); }}
        style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block', cursor: 'crosshair' }} />
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
    </div>
  );
}
