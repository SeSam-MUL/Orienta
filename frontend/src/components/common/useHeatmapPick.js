import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Pick a pixel on the NCC heatmap by clicking OR by dragging the crosshair.
 *
 * Selecting a pixel makes the dialog re-fetch the pattern match for it, which
 * re-renders a simulated pattern on the backend and takes seconds. Committing
 * on every mousemove would therefore fire a request storm, so a drag only moves
 * a local preview and the selection is committed once, on release. A plain
 * click is just a zero-length drag and goes through the same path.
 *
 * Listeners live on the window so the crosshair keeps following the pointer
 * outside the image (the position is clamped to the grid) and so releasing
 * anywhere still commits.
 *
 * Once the map is zoomed, dragging means something else: there the picture has
 * to be movable, so `onPan` takes over the drag. Two ways to still place the
 * point remain — a click, and Shift+drag. Shift rather than grabbing the
 * crosshair itself, because at a high zoom the crosshair is often outside the
 * visible crop, and a gesture you cannot reach is no gesture.
 *
 * Pass `onPan` exactly when panning should win — with it absent this behaves
 * exactly as it always did.
 *
 * @param imgRef  ref to the heatmap <img>
 * @param dims    {rows, cols} of the (cropped) heatmap
 * @param offset  {row, col} of the crop inside the full grid
 * @param onPick  called once per click/drag with
 *                {row, col, localRow, localCol}
 * @param onPan   optional (dxFrac, dyFrac) => void — when given, dragging pans
 *                instead of moving the point
 * @param boxRef  the element the pan fractions are measured against (the
 *                wrapper, not the transformed image)
 * @returns {{preview, dragging, panning, onMouseDown}}
 */
// A press that travels no further than this is a click, not a drag.
const CLICK_SLOP_PX = 3;

export function useHeatmapPick({ imgRef, dims, offset, onPick, onPan = null, boxRef = null }) {
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const draggingRef = useRef(false);
  // The handlers are attached once per drag; refs keep them reading current
  // values without re-subscribing on every render.
  const dimsRef = useRef(dims);
  dimsRef.current = dims;
  const offsetRef = useRef(offset);
  offsetRef.current = offset;
  const pickRef = useRef(onPick);
  pickRef.current = onPick;
  const panRef = useRef(onPan);
  panRef.current = onPan;
  // Set while a drag is panning rather than moving the point.
  const panStateRef = useRef(null);

  const toPixel = useCallback((e) => {
    const img = imgRef.current;
    const d = dimsRef.current;
    if (!img || !d?.rows || !d?.cols) return null;
    const rect = img.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const clamp = (v, hi) => Math.min(Math.max(0, v), hi - 1);
    const localRow = clamp(Math.floor((e.clientY - rect.top) / rect.height * d.rows), d.rows);
    const localCol = clamp(Math.floor((e.clientX - rect.left) / rect.width * d.cols), d.cols);
    const off = offsetRef.current || { row: 0, col: 0 };
    return {
      row: localRow + (off.row || 0),
      col: localCol + (off.col || 0),
      localRow,
      localCol,
    };
  }, [imgRef]);

  useEffect(() => {
    if (!dragging) return undefined;
    const onMove = (e) => {
      if (!draggingRef.current) return;
      const ps = panStateRef.current;
      if (ps) {
        const box = (boxRef?.current || imgRef.current)?.getBoundingClientRect?.();
        if (!box?.width || !box?.height) return;
        const dx = e.clientX - ps.x;
        const dy = e.clientY - ps.y;
        ps.x = e.clientX;
        ps.y = e.clientY;
        ps.moved += Math.abs(dx) + Math.abs(dy);
        panRef.current?.(dx / box.width, dy / box.height);
        return;
      }
      const p = toPixel(e);
      if (p) setPreview(p);
    };
    const onUp = (e) => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      setDragging(false);
      const ps = panStateRef.current;
      panStateRef.current = null;
      if (ps) {
        // A pan that went nowhere was a click: still pick the pixel under it,
        // so selecting stays possible while zoomed in.
        if (ps.moved <= CLICK_SLOP_PX) {
          const p = toPixel(e);
          if (p) pickRef.current?.(p);
        }
        return;
      }
      // Prefer the release position; fall back to the last preview when the
      // pointer left the document (no usable coordinates in the event).
      const p = toPixel(e) || preview;
      setPreview(null);
      if (p) pickRef.current?.(p);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [dragging, toPixel, preview, boxRef, imgRef]);

  const onMouseDown = useCallback((e) => {
    // Left button only — right-click opens the export menu.
    if (e.button !== 0) return;
    const p = toPixel(e);
    if (!p) return;
    e.preventDefault();   // no text/image drag while picking
    draggingRef.current = true;
    setDragging(true);
    if (panRef.current && !e.shiftKey) {
      // Zoomed in: the drag moves the picture, so no preview point is armed.
      // Shift is the exception — it keeps dragging the point.
      panStateRef.current = { x: e.clientX, y: e.clientY, moved: 0 };
      return;
    }
    setPreview(p);
  }, [toPixel]);

  return { preview, dragging, panning: !!onPan, onMouseDown };
}

export default useHeatmapPick;
