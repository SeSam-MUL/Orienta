import { useRef, useEffect, useState } from 'react';
import { pointerToRowCol, rowColToContainerPx } from './mapCoords';
import { useCursorPublisher, useCursorSync } from './CursorSyncContext';
import { useRectangleDrag } from './hooks/useRectangleDrag';
import { buildMaskCanvas } from '../PhaseMap/maskCanvas';
import { colors } from '../../theme/components';

export default function Tile({ layer, bitmap, error, shape, onPixelClick, onRegionSelected }) {
  const hostRef = useRef(null);
  const canvasRef = useRef(null);
  const publish = useCursorPublisher();
  const [crosshair, setCrosshair] = useState(null);
  const drag = useRectangleDrag({ shape, onRegion: onRegionSelected });

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

  useCursorSync((pos) => {
    if (!shape || !pos.hovering || !hostRef.current) { setCrosshair(null); return; }
    const rect = hostRef.current.getBoundingClientRect();
    const { x, y } = rowColToContainerPx(pos.row, pos.col, rect, shape);
    setCrosshair({ x, y });
  });

  const onMove = (e) => {
    if (!shape || !hostRef.current) return;
    const rect = hostRef.current.getBoundingClientRect();
    const out = pointerToRowCol(e, rect, shape);
    if (out) publish({ row: out.row, col: out.col, hovering: true, screenX: e.clientX, screenY: e.clientY });
  };
  const onLeave = (e) => {
    if (!shape) return;
    publish({ row: 0, col: 0, hovering: false, screenX: e.clientX, screenY: e.clientY });
  };
  const onClick = (e) => {
    if (!shape || !hostRef.current || !onPixelClick) return;
    const rect = hostRef.current.getBoundingClientRect();
    const out = pointerToRowCol(e, rect, shape);
    if (out) onPixelClick(out.row, out.col);
  };

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
          {layer.blend} · {Math.round((layer.opacity ?? 1) * 100)}%
        </span>
      </div>
      <div
        ref={hostRef}
        data-tile-host
        onMouseDown={drag.onPointerDown}
        onMouseMove={(e) => { onMove(e); drag.onPointerMove(e); }}
        onMouseUp={drag.onPointerUp}
        onMouseLeave={onLeave}
        onClick={onClick}
        style={{
          position: 'relative',
          width: '100%',
          // Aspect from the bitmap's own dims so the canvas (width/height 100%)
          // is never stretched. Falls back to `shape`, then a sane default,
          // while the bitmap is still fetching.
          aspectRatio: bitmap
            ? `${bitmap.width}/${bitmap.height}`
            : (shape ? `${shape[1]}/${shape[0]}` : '156/128'),
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: '100%', height: '100%',
            imageRendering: 'pixelated',
            borderRadius: 3,
            opacity: Math.max(0.4, layer.opacity ?? 1),
            background: '#000',
          }}
        />
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
