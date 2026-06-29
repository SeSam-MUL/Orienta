import { useRef, useState, useCallback } from 'react';
import { pointerToRowCol } from '../mapCoords';

/**
 * Shift+drag to draw a rectangle ROI on a letterboxed canvas. On pointer-up
 * calls onRegion({ rowStart, colStart, rowEnd, colEnd }) with normalised
 * (start <= end) coords in image-pixel space.
 *
 * Plain (non-shift) drags are ignored — caller's regular click handler still
 * fires for pixel-quantify.
 */
export function useRectangleDrag({ shape, onRegion }) {
  const startRef = useRef(null);
  const rectRef = useRef(null);
  const [overlay, setOverlay] = useState(null);

  const onPointerDown = useCallback((e) => {
    if (!e.shiftKey || !shape) return;
    const r = e.currentTarget.getBoundingClientRect();
    rectRef.current = r;
    const out = pointerToRowCol(e, r, shape);
    if (!out) return;
    startRef.current = { ...out, clientX: e.clientX, clientY: e.clientY };
    setOverlay({
      x0: e.clientX - r.left, y0: e.clientY - r.top,
      x1: e.clientX - r.left, y1: e.clientY - r.top,
    });
  }, [shape]);

  const onPointerMove = useCallback((e) => {
    if (!startRef.current || !rectRef.current) return;
    // Snapshot rect into the closure — React 19 may double-invoke the updater
    // (StrictMode) and rectRef.current may have been nulled by pointerUp by
    // the time the second invocation runs.
    const rect = rectRef.current;
    const nx1 = e.clientX - rect.left;
    const ny1 = e.clientY - rect.top;
    setOverlay((o) => (o ? { ...o, x1: nx1, y1: ny1 } : o));
  }, []);

  const onPointerUp = useCallback((e) => {
    if (!startRef.current || !rectRef.current || !shape) {
      startRef.current = null; rectRef.current = null; setOverlay(null);
      return;
    }
    const end = pointerToRowCol(e, rectRef.current, shape);
    if (end) {
      const start = startRef.current;
      onRegion?.({
        rowStart: Math.min(start.row, end.row),
        colStart: Math.min(start.col, end.col),
        rowEnd:   Math.max(start.row, end.row),
        colEnd:   Math.max(start.col, end.col),
      });
    }
    startRef.current = null;
    rectRef.current = null;
    setOverlay(null);
  }, [shape, onRegion]);

  return { onPointerDown, onPointerMove, onPointerUp, overlay };
}
