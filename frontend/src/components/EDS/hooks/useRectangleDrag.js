import { useRef, useState, useCallback } from 'react';
import { pointerToRowCol } from '../mapCoords';

/**
 * Shift+drag to draw a rectangle ROI on a letterboxed canvas. On pointer-up
 * calls onRegion({ rowStart, colStart, rowEnd, colEnd }) with normalised
 * (start <= end) coords in image-pixel space.
 *
 * Plain (non-shift) drags are ignored — caller's regular click handler still
 * fires for pixel-quantify.
 *
 * `mapRect` (optional) translates the host rect into the rect the map content
 * actually occupies — that is how a zoomed view stays pixel-exact (see
 * zoomView.zoomedRect). The drawn overlay keeps using the HOST rect, so the
 * dashed rectangle still sits where the cursor is. Omitting it (the default)
 * is byte-for-byte the previous behaviour.
 */
export function useRectangleDrag({ shape, onRegion, contentBbox = null, mapRect = null }) {
  const startRef = useRef(null);
  const rectRef = useRef(null);
  const hostRectRef = useRef(null);
  const [overlay, setOverlay] = useState(null);
  // Keep the latest bbox in a ref so the memoised callbacks always read the
  // current auto-zoom without needing to re-create on every bbox change.
  const bboxRef = useRef(contentBbox);
  bboxRef.current = contentBbox;
  const mapRectRef = useRef(mapRect);
  mapRectRef.current = mapRect;

  const onPointerDown = useCallback((e) => {
    if (!e.shiftKey || !shape) return;
    const host = e.currentTarget.getBoundingClientRect();
    const r = mapRectRef.current ? mapRectRef.current(host) : host;
    rectRef.current = r;
    hostRectRef.current = host;
    const out = pointerToRowCol(e, r, shape, bboxRef.current);
    if (!out) return;
    startRef.current = { ...out, clientX: e.clientX, clientY: e.clientY };
    setOverlay({
      x0: e.clientX - host.left, y0: e.clientY - host.top,
      x1: e.clientX - host.left, y1: e.clientY - host.top,
    });
  }, [shape]);

  const onPointerMove = useCallback((e) => {
    if (!startRef.current || !hostRectRef.current) return;
    // Snapshot rect into the closure — React 19 may double-invoke the updater
    // (StrictMode) and hostRectRef.current may have been nulled by pointerUp by
    // the time the second invocation runs.
    const rect = hostRectRef.current;
    const nx1 = e.clientX - rect.left;
    const ny1 = e.clientY - rect.top;
    setOverlay((o) => (o ? { ...o, x1: nx1, y1: ny1 } : o));
  }, []);

  const onPointerUp = useCallback((e) => {
    if (!startRef.current || !rectRef.current || !shape) {
      startRef.current = null; rectRef.current = null; hostRectRef.current = null; setOverlay(null);
      return;
    }
    const end = pointerToRowCol(e, rectRef.current, shape, bboxRef.current);
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
    hostRectRef.current = null;
    setOverlay(null);
  }, [shape, onRegion]);

  return { onPointerDown, onPointerMove, onPointerUp, overlay };
}
