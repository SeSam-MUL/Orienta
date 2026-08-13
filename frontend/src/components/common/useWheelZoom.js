import { useCallback, useRef } from 'react';
import { wheelFactor } from '../EDS/zoomView';

/**
 * Wheel-to-zoom on one element, reported as a factor plus the pointer position
 * as a fraction of the element's box — the shape `useZoomViews.zoomAtPointer`
 * wants.
 *
 * A native listener with `{passive: false}` rather than React's `onWheel`,
 * which is registered passively: `preventDefault()` would be ignored there, the
 * panel behind would scroll, and in Electron the whole app would zoom instead.
 *
 * Returns a CALLBACK REF, not a plain effect on a ref object. A first attempt
 * used `useEffect([ref])`, which runs once on mount — and the NCC map only
 * appears once its image has loaded, so at that moment `ref.current` was still
 * null and the listener was never attached. The symptom was subtle: wheeling
 * over the map zoomed a different panel, because the only listener that saw the
 * event belonged to somebody else. A callback ref fires when the node actually
 * arrives and again when it leaves.
 *
 * The callback is read through a ref, so a parent re-rendering on every frame
 * of a zoom does not detach and re-attach the listener.
 */
export function useWheelZoom(onZoomAt) {
  const cbRef = useRef(onZoomAt);
  cbRef.current = onZoomAt;
  const detachRef = useRef(null);

  return useCallback((el) => {
    detachRef.current?.();
    detachRef.current = null;
    if (!el) return;
    const onWheel = (e) => {
      if (!cbRef.current) return;
      e.preventDefault();
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height) return;
      cbRef.current(
        wheelFactor(e.deltaY),
        (e.clientX - r.left) / r.width,
        (e.clientY - r.top) / r.height,
      );
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    detachRef.current = () => el.removeEventListener('wheel', onWheel);
  }, []);
}

export default useWheelZoom;
