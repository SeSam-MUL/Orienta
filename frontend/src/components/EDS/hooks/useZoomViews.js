import { useState, useCallback, useRef, useMemo } from 'react';
import { IDENTITY_VIEW, clampView, zoomAt, panBy, isZoomed } from '../zoomView';

export const SYNC_ALL = 'all';
export const SYNC_SINGLE = 'single';

/**
 * Zoom state for every map view on the EDS page (the composite overlay and one
 * entry per tile), with a sync switch:
 *
 *   'all'    — one shared view drives every map; zooming any of them moves all.
 *   'single' — each map keeps its own view; zooming touches only that one.
 *
 * Switching modes discards nothing: the per-map views survive a trip through
 * 'all' and come back unchanged. Going single → all seeds the shared view from
 * the map that was zoomed last, so flipping the switch never makes the view
 * jump somewhere the user did not ask for.
 */
export function useZoomViews(initialMode = SYNC_ALL) {
  const [mode, setModeState] = useState(initialMode);
  const [shared, setShared] = useState(IDENTITY_VIEW);
  const [perId, setPerId] = useState(() => ({}));
  const lastTouched = useRef(null);

  const viewFor = useCallback(
    (id) => (mode === SYNC_ALL ? shared : (perId[id] || IDENTITY_VIEW)),
    [mode, shared, perId],
  );

  const update = useCallback((id, fn) => {
    lastTouched.current = id;
    if (mode === SYNC_ALL) {
      setShared((v) => clampView(fn(v)));
      return;
    }
    setPerId((m) => ({ ...m, [id]: clampView(fn(m[id] || IDENTITY_VIEW)) }));
  }, [mode]);

  const zoomAtPointer = useCallback(
    (id, factor, px, py) => update(id, (v) => zoomAt(v, factor, px, py)),
    [update],
  );
  const pan = useCallback(
    (id, dxFrac, dyFrac) => update(id, (v) => panBy(v, dxFrac, dyFrac)),
    [update],
  );
  const resetOne = useCallback((id) => update(id, () => IDENTITY_VIEW), [update]);

  const resetAll = useCallback(() => {
    setShared(IDENTITY_VIEW);
    setPerId({});
  }, []);

  const setMode = useCallback((next) => {
    if (next === mode) return;
    if (next === SYNC_ALL) {
      const seed = lastTouched.current ? perId[lastTouched.current] : null;
      if (seed) setShared(clampView(seed));
    }
    setModeState(next);
  }, [mode, perId]);

  const anyZoomed = useMemo(
    () => (mode === SYNC_ALL ? isZoomed(shared) : Object.values(perId).some(isZoomed)),
    [mode, shared, perId],
  );

  return { mode, setMode, viewFor, zoomAtPointer, pan, resetOne, resetAll, anyZoomed };
}
