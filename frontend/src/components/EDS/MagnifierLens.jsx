/**
 * MagnifierLens — circular 4x zoom lens that follows the cursor over a
 * source canvas (the Composite Overlay or the swipe canvas). Renders into
 * its own canvas overlay (no React reconciliation on every cursor move)
 * and is purely visual (pointerEvents: none).
 *
 * The lens accepts a `hostRef` (the OverlayCard's host div) and resolves
 * the first <canvas> under it inside an effect. We do not read the ref
 * during render (which the react-hooks/refs rule forbids).
 */
import { useRef, useEffect } from 'react';
import { colors } from '../../theme/components';

const LENS_SIZE = 120;
const ZOOM = 4;

export default function MagnifierLens({ hostRef, pos, visible }) {
  const lensRef = useRef(null);
  useEffect(() => {
    if (!visible || !lensRef.current || !pos) return;
    const host = hostRef?.current;
    const sourceCanvas = host?.querySelector('canvas') ?? null;
    if (!sourceCanvas) return;
    const lc = lensRef.current;
    lc.width = LENS_SIZE;
    lc.height = LENS_SIZE;
    const ctx = lc.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    const sx = pos.x - LENS_SIZE / (2 * ZOOM);
    const sy = pos.y - LENS_SIZE / (2 * ZOOM);
    ctx.clearRect(0, 0, LENS_SIZE, LENS_SIZE);
    try {
      ctx.drawImage(
        sourceCanvas,
        sx, sy, LENS_SIZE / ZOOM, LENS_SIZE / ZOOM,
        0, 0, LENS_SIZE, LENS_SIZE,
      );
    } catch {
      // Ignore drawImage errors (e.g. source canvas is 0x0 during mount).
    }
  }, [pos, visible, hostRef]);
  if (!visible || !pos) return null;
  return (
    <canvas
      ref={lensRef}
      data-magnifier-lens
      style={{
        position: 'absolute',
        pointerEvents: 'none',
        left: pos.x - LENS_SIZE / 2,
        top: pos.y - LENS_SIZE / 2,
        width: LENS_SIZE,
        height: LENS_SIZE,
        borderRadius: '50%',
        border: `2px solid ${colors.cyan}`,
        boxShadow: '0 6px 18px rgba(0,0,0,.6)',
        zIndex: 3,
      }}
    />
  );
}
