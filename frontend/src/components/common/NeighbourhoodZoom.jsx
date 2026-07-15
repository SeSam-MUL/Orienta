import { useEffect, useRef, useState } from 'react';
import { colors as C } from '../../theme/tokens';

/**
 * Click-anchored neighbourhood magnifier + pixel nudge for the pattern-match
 * dialogs. Shows a (2·HALF+1)² data-pixel crop around the selected pixel from
 * a full-grid layer PNG (IPF preferred — tiny mis-indexed nests read as
 * colour-continuity breaks there), nearest-neighbour upscaled with a centre
 * crosshair, plus arrow buttons / keyboard nudge (1 px steps) so 2-5 px nests
 * can be reached precisely without pixel-perfect map clicks.
 *
 * Deliberately NOT the MagnifierLens: that is a cursor-follow hover lens;
 * this is a click-anchored, keyboard-nudgeable fixed crop.
 *
 * Props:
 *   imageB64 — base64 PNG at NATIVE map resolution (full grid) | null
 *   shape    — [rows, cols] of the grid (nudge bounds)
 *   pixel    — {row, col} | null
 *   onNudge  — (dRow, dCol) => void
 *   caption  — translated caption string
 *   labels   — {up, down, left, right, tip} translated aria/tooltip strings
 *   kind / kinds / onKindChange — optional source-layer switch (radio
 *     semantics): kinds = [{id, label}], active id = kind. Lets the user
 *     flip the lens between IPF-Z/Y/X and phase colours — some nests only
 *     read as breaks in ONE of those maps.
 */
const HALF = 7;               // crop = (2*HALF+1)^2 data pixels
const SCALE = 9;              // canvas px per data px

export default function NeighbourhoodZoom({ imageB64, shape, pixel, onNudge,
                                            caption, labels = {},
                                            kind, kinds, onKindChange }) {
  const canvasRef = useRef(null);
  const [img, setImg] = useState(null);

  useEffect(() => {
    if (!imageB64) { setImg(null); return undefined; }
    const el = new Image();
    el.onload = () => setImg(el);
    el.src = `data:image/png;base64,${imageB64}`;
    return () => { el.onload = null; };
  }, [imageB64]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !img || !pixel) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#111119';
    ctx.fillRect(0, 0, cv.width, cv.height);
    const n = 2 * HALF + 1;
    // drawImage clips out-of-bounds source regions itself; anchor the dest
    // so the selected pixel always sits in the centre cell.
    ctx.drawImage(img, pixel.col - HALF, pixel.row - HALF, n, n,
                  0, 0, n * SCALE, n * SCALE);
    ctx.strokeStyle = '#ffb86c';
    ctx.lineWidth = 2;
    ctx.strokeRect(HALF * SCALE + 0.5, HALF * SCALE + 0.5, SCALE - 1, SCALE - 1);
  }, [img, pixel]);

  // Keyboard nudge — window-level so it works without focusing the widget,
  // but NEVER while a form control has focus (the dialogs contain selects +
  // the reference-pixel number inputs whose arrow keys must keep working).
  useEffect(() => {
    if (!pixel) return undefined;
    const onKey = (e) => {
      const tag = document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      // A focused layer-radio must not double-drive the anchor.
      if (document.activeElement?.getAttribute?.('role') === 'radio') return;
      const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0],
                  ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
      if (!d) return;
      e.preventDefault();
      onNudge?.(d[0], d[1]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [pixel, onNudge]);

  if (!pixel) return null;
  const rows = shape?.[0] ?? Infinity;
  const cols = shape?.[1] ?? Infinity;
  const n = 2 * HALF + 1;
  // Click-to-jump: a click on a zoom cell moves the MAIN selection there
  // (same semantics as clicking the parent map, but pixel-precise). The
  // dialog's nudge handler clamps, so multi-cell deltas are safe.
  const onCanvasClick = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = Math.min(n - 1, Math.max(0,
      Math.floor((e.clientX - rect.left) / rect.width * n)));
    const cy = Math.min(n - 1, Math.max(0,
      Math.floor((e.clientY - rect.top) / rect.height * n)));
    if (cx === HALF && cy === HALF) return;      // clicked the current pixel
    onNudge?.(cy - HALF, cx - HALF);
  };
  const btn = (glyph, label, dr, dc, disabled) => (
    <button key={glyph} onClick={() => onNudge?.(dr, dc)} disabled={disabled}
      aria-label={label} title={labels.tip || label}
      style={{
        width: 26, height: 20, fontSize: '8pt', padding: 0,
        background: 'transparent', border: `1px solid ${C.border}`,
        borderRadius: 3, color: disabled ? '#44475a' : C.text,
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}>
      {glyph}
    </button>
  );
  return (
    <div style={{ marginTop: 6, textAlign: 'center' }}>
      {img && (
        <canvas ref={canvasRef} width={n * SCALE} height={n * SCALE}
          onClick={onCanvasClick}
          title={labels.clickTip || labels.tip || ''}
          style={{
            width: n * SCALE, height: n * SCALE, borderRadius: 3,
            border: `1px solid ${C.border}`, display: 'block', margin: '0 auto',
            cursor: 'crosshair',
          }} />
      )}
      <div style={{ display: 'flex', gap: 4, justifyContent: 'center', marginTop: 4 }}>
        {btn('◀', labels.left || 'left', 0, -1, pixel.col <= 0)}
        {btn('▲', labels.up || 'up', -1, 0, pixel.row <= 0)}
        {btn('▼', labels.down || 'down', 1, 0, pixel.row >= rows - 1)}
        {btn('▶', labels.right || 'right', 0, 1, pixel.col >= cols - 1)}
      </div>
      {Array.isArray(kinds) && kinds.length > 1 && (
        <div role="radiogroup" aria-label={labels.layerGroup || 'Zoom map layer'}
          style={{ display: 'flex', gap: 3, justifyContent: 'center',
                   alignItems: 'center', marginTop: 6 }}>
          {labels.layerLead && (
            <span style={{ fontSize: '7pt', color: '#6272a4', marginRight: 1 }}>
              {labels.layerLead}
            </span>
          )}
          {kinds.map((k) => {
            const active = k.id === kind;
            return (
              <button key={k.id} role="radio" aria-checked={active}
                onClick={() => onKindChange?.(k.id)}
                title={labels.layerTip || ''}
                style={{
                  height: 18, fontSize: '7.5pt', padding: '0 6px',
                  marginLeft: k.sep ? 6 : 0,
                  background: active ? 'rgba(139, 233, 253, 0.12)' : 'transparent',
                  border: `1px solid ${active ? C.accent : C.border}`,
                  borderRadius: 3, color: active ? C.accent : '#6272a4',
                  fontWeight: active ? 600 : 400, cursor: 'pointer',
                }}>
                {k.label}
              </button>
            );
          })}
        </div>
      )}
      {caption && (
        <div style={{ fontSize: '7pt', color: '#6272a4', marginTop: 2 }}>{caption}</div>
      )}
    </div>
  );
}
