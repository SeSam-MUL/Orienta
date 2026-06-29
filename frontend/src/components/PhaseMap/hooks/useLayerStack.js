/**
 * useLayerStack — React adapter over the pure layerStackReducer.
 *
 * Responsibilities:
 *   - Holds the reducer state via useReducer.
 *   - Fetches layer pixel data (PNG via various endpoints) on demand.
 *   - Caches decoded ImageBitmap per layer id (LRU, max 16 entries).
 *   - Invalidates on cleanup-param change for result-source layers only.
 *   - Exposes addLayer / removeLayer / setOpacity / setBlend / setVisibility /
 *     reorder / applyPreset / clear methods.
 */
import { useReducer, useRef, useCallback, useEffect } from 'react';
import { layerStackReducer, initialState } from '../layerStackReducer';
import { findLayerDef } from '../layerSources';
import { applyPreset } from '../presets';
import { phaseMapApi, ebsdApi, h5Api, analysisApi } from '../../../services/api';

const CACHE_SIZE = 16;

/** Convert a black background to alpha=0 on a decoded ImageBitmap.
 *  Returns a new ImageBitmap. Pixels with RGB all ≤ 8 → alpha=0. */
async function blackToAlpha(bitmap) {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i] <= 8 && d[i + 1] <= 8 && d[i + 2] <= 8) d[i + 3] = 0;
  }
  ctx.putImageData(img, 0, 0);
  bitmap.close();
  return canvas.transferToImageBitmap();
}

async function pngBase64ToBitmap(b64, { keyToAlpha = false } = {}) {
  const blob = await fetch(`data:image/png;base64,${b64}`).then((r) => r.blob());
  let bitmap = await createImageBitmap(blob);
  if (keyToAlpha) bitmap = await blackToAlpha(bitmap);
  return bitmap;
}

/** Dispatch the right endpoint per layer source. Returns base64 PNG.
 *  Routes off `layer.source` (set by addLayer) and `layer.id`. Static catalog
 *  lookups happen at addLayer time, not here — so dynamic ids (ci_<phase>,
 *  eds:<el>, se:<name>) work without further registration. */
async function fetchLayerImage({ layer, cleanupParams }) {
  switch (layer.source) {
    case 'result': {
      // Backend `kind` matches the layer id for all result-source entries
      // (phase, ipf-x/y/z, bc, ci, ci_<phase>, uncertainty, ci-threshold).
      // Per-layer params (e.g. band_min/band_max for ci-threshold) live
      // on the layer object and are merged into the request alongside the
      // global cleanupParams. cleanupParams win conflicts (rare).
      const mergedParams = { ...(layer.params || {}), ...cleanupParams };
      const res = await phaseMapApi.layer(layer.id, mergedParams);
      return { base64: res.data.image, keyToAlpha: false };
    }
    case 'diagnostics':
    case 'refinement': {
      // Phase-A diagnostics + Phase-B refinement layers are served by the
      // same /api/phasemap/layer endpoint, but their backend `kind` is
      // underscored (forward_ncc) while the catalog id is hyphenated
      // (forward-ncc). Resolve the kind from the static catalog.
      const def = findLayerDef(layer.id);
      const kind = def?.kind ?? layer.id;
      const res = await phaseMapApi.layer(kind, cleanupParams);
      // The backend already emits a transparent-background RGBA PNG
      // (NaN / non-indexed pixels → alpha 0), so no colour-keying needed.
      return { base64: res.data.image, keyToAlpha: false };
    }
    case 'analysis': {
      const res = await analysisApi.getMap(layer.id);
      return { base64: res.data.image, keyToAlpha: true };
    }
    case 'ebsd': {
      if (layer.id === 'vbse') {
        const res = await ebsdApi.virtualBSE();
        return { base64: res.data.image, keyToAlpha: true };
      }
      throw new Error(`Unsupported ebsd layer: ${layer.id}`);
    }
    case 'h5': {
      if (layer.id.startsWith('eds:')) {
        const element = layer.id.slice(4);
        const res = await h5Api.getEDSMap(element, 'hot', '');
        return { base64: res.data.image, keyToAlpha: true };
      }
      if (layer.id.startsWith('se:')) {
        const name = layer.id.slice(3);
        const res = await h5Api.getElectronImage(name);
        return { base64: res.data.image, keyToAlpha: true };
      }
      throw new Error(`Unsupported h5 layer: ${layer.id}`);
    }
    default:
      throw new Error(`Unknown source: ${layer.source}`);
  }
}

export function useLayerStack({ cleanupParams, resetSignal, frameSig }) {
  const [state, dispatch] = useReducer(layerStackReducer, initialState);
  const cacheRef = useRef(new Map());           // layerId → ImageBitmap
  const cacheOrderRef = useRef([]);             // LRU order (most recent at end)
  const fetchingRef = useRef(new Set());        // layer ids currently in-flight
  const errorRef = useRef(new Map());           // layerId → error string
  // Remember the last user-chosen "mode" (single quick-mode layer or
  // preset) so a result-change can re-apply it instead of forcing the
  // user back to Phase every time they click a different gallery entry.
  // Shape: { type: 'single', id } | { type: 'preset', name } | null
  const lastModeRef = useRef(null);

  // Monotonic epoch, bumped on every cacheFlush(). A layer fetch captures the
  // epoch at start and refuses to write its decoded bitmap if the epoch has
  // moved on — i.e. a result/file switch flushed the cache while this fetch
  // was still in flight. Without this guard a late fetch for the PREVIOUS
  // result repopulates cacheRef with a stale bitmap (e.g. 'phase' exists in
  // both presets), so the fetch effect below sees the id as "already cached"
  // and never refetches the new result's layer — the map renders the wrong
  // grid forever. (Worse with a shape mismatch: 174×145 result vs 151×186
  // open file → LayeredCanvas locks onto the stale native size.)
  const epochRef = useRef(0);

  // Rerender ping AND change-detection signal. Bumped whenever the
  // cache or error Maps are mutated. Consumers (LayeredCanvas) include
  // it in their useMemo / useEffect deps so they recompute when bitmaps
  // arrive — the Map reference itself is stable so React can't see the
  // mutation by reference equality alone.
  const [bitmapVersion, force] = useReducer((x) => x + 1, 0);

  const cacheTouch = useCallback((id) => {
    const order = cacheOrderRef.current;
    const idx = order.indexOf(id);
    if (idx !== -1) order.splice(idx, 1);
    order.push(id);
  }, []);

  const cacheSet = useCallback((id, bitmap) => {
    const cache = cacheRef.current;
    const order = cacheOrderRef.current;
    if (cache.has(id)) {
      try { cache.get(id).close(); } catch { /* ignore */ }
    }
    cache.set(id, bitmap);
    cacheTouch(id);
    while (order.length > CACHE_SIZE) {
      const evictId = order.shift();
      const evictBmp = cache.get(evictId);
      cache.delete(evictId);
      try { evictBmp?.close(); } catch { /* ignore */ }
    }
    force();
  }, [cacheTouch]);

  const cacheFlush = useCallback((predicate = null) => {
    // Invalidate any in-flight fetch so its late write is dropped (see epochRef).
    epochRef.current += 1;
    const cache = cacheRef.current;
    const order = cacheOrderRef.current;
    const errors = errorRef.current;
    for (const [id, bmp] of cache.entries()) {
      if (!predicate || predicate(id)) {
        try { bmp.close(); } catch { /* ignore */ }
        cache.delete(id);
        const idx = order.indexOf(id);
        if (idx !== -1) order.splice(idx, 1);
      }
    }
    // Also flush matching error entries — otherwise stale errors from
    // long-removed layers stick around in the LayeredCanvas overlay.
    for (const id of [...errors.keys()]) {
      if (!predicate || predicate(id)) errors.delete(id);
    }
    force();
  }, []);

  const fetchLayer = useCallback(async (layer) => {
    if (fetchingRef.current.has(layer.id)) return;
    if (cacheRef.current.has(layer.id)) { cacheTouch(layer.id); return; }
    fetchingRef.current.add(layer.id);
    const epoch = epochRef.current;  // snapshot; a flush mid-flight bumps this
    try {
      const { base64, keyToAlpha } = await fetchLayerImage({ layer, cleanupParams });
      if (!base64) throw new Error('Empty image payload');
      const bitmap = await pngBase64ToBitmap(base64, { keyToAlpha });
      // Result/file switched while we were fetching → this bitmap belongs to
      // the previous result. Drop it so it can't poison the cache and stall
      // the new result's render. The fetch effect re-fires for uncached layers.
      if (epoch !== epochRef.current) {
        try { bitmap.close(); } catch { /* ignore */ }
        return;
      }
      cacheSet(layer.id, bitmap);
      errorRef.current.delete(layer.id);
    } catch (err) {
      errorRef.current.set(layer.id, err?.response?.data?.detail ?? err.message ?? 'fetch failed');
      force();
    } finally {
      fetchingRef.current.delete(layer.id);
    }
  }, [cleanupParams, cacheSet, cacheTouch]);

  // Whenever the layer list changes, ensure all visible layers have bitmaps.
  useEffect(() => {
    Promise.all(
      state.layers
        .filter((l) => l.visible && !cacheRef.current.has(l.id))
        .map((l) => fetchLayer(l))
    );
  }, [state.layers, fetchLayer]);

  // Cleanup-param change → invalidate result-source layers only.
  // (BC/IPF colour doesn't depend on cleanup, but phase/ci/uncertainty do.)
  const cleanupSignature = JSON.stringify(cleanupParams ?? {});
  useEffect(() => {
    const affected = new Set(['phase', 'ci', 'uncertainty']);
    cacheFlush((id) => affected.has(id) || id.startsWith('ci_'));
  }, [cleanupSignature, cacheFlush]);

  // Frame (coordinate-system) change → invalidate ONLY the orientation-colored
  // layers so they refetch through the new R_user. Phase/CI/BC/EDS are
  // orientation-independent and must NOT be flushed (avoids needless refetch).
  useEffect(() => {
    if (frameSig == null) return;
    const affected = new Set(['ipf-x', 'ipf-y', 'ipf-z']);
    cacheFlush((id) => affected.has(id));
  }, [frameSig, cacheFlush]);

  // Result/file change → flush everything, re-apply the user's last
  // chosen mode (if any). Without this remembering, every gallery click
  // forced the user back to Phase, which was extremely annoying when
  // they were trying to compare IPF maps across multiple results.
  useEffect(() => {
    cacheFlush();
    // Only seed when we actually have a result to render. On initial
    // mount with no result, leave the stack empty so users see the
    // friendly "No layers loaded" placeholder instead of a fetch error.
    if (resetSignal == null) {
      dispatch({ type: 'CLEAR' });
      return;
    }
    const last = lastModeRef.current;
    if (last?.type === 'single') {
      // Re-apply the last quick-mode pick. inferDynamicDef handles
      // dynamic ids; findLayerDef handles static ones.
      const def = findLayerDef(last.id) ?? inferDynamicDef(last.id);
      if (def) {
        dispatch({
          type: 'REPLACE_ALL',
          layers: [{
            id: last.id, label: def.label ?? last.id, source: def.source,
            opacity: 1.0, blend: 'normal', visible: true,
            key: `${last.id}-${Date.now()}`,
          }],
        });
        return;
      }
    }
    if (last?.type === 'preset') {
      try {
        dispatch({ type: 'REPLACE_ALL', layers: applyPreset(last.name) });
        return;
      } catch { /* fall through to default */ }
    }
    // First result of the session, or last mode no longer valid: fall
    // back to the phase_default preset.
    dispatch({ type: 'REPLACE_ALL', layers: applyPreset('phase_default') });
  }, [resetSignal, cacheFlush]);

  // -------------------- public API --------------------
  /** Infer source + sensible defaults for dynamic-id layers (per-phase CI,
   *  EDS elements, electron images) that aren't in the static catalog. */
  function inferDynamicDef(id) {
    if (id.startsWith('ci_')) {
      return { id, label: `CI — ${id.slice(3)}`, source: 'result',
               defaultBlend: 'normal', defaultOpacity: 0.6 };
    }
    if (id.startsWith('eds:')) {
      return { id, label: `EDS: ${id.slice(4)}`, source: 'h5',
               defaultBlend: 'screen', defaultOpacity: 0.5 };
    }
    if (id.startsWith('se:')) {
      return { id, label: `SE: ${id.slice(3)}`, source: 'h5',
               defaultBlend: 'normal', defaultOpacity: 1.0 };
    }
    return null;
  }

  const addLayer = useCallback((id) => {
    const def = findLayerDef(id) ?? inferDynamicDef(id);
    if (!def) {
      console.warn('[useLayerStack] unknown layer id:', id);
      return;
    }
    const layer = {
      id, label: def.label ?? id, source: def.source,
      opacity: def.defaultOpacity ?? 1.0,
      blend: def.defaultBlend ?? 'normal',
      visible: true,
      // Kind-specific request params (e.g. band_min/band_max for ci-threshold,
      // shallow-copied so reducer mutations stay safe). Merged into the
      // /api/phasemap/layer query at fetch time.
      params: def.params ? { ...def.params } : undefined,
      key: `${id}-${Date.now()}`,
    };
    dispatch({ type: 'ADD', layer });
  }, []);

  const removeLayer = useCallback((id) => {
    // Also drop the bitmap + any stored error so the user can re-add the
    // layer fresh without ghost state lingering.
    cacheFlush((cid) => cid === id);
    dispatch({ type: 'REMOVE', id });
  }, [cacheFlush]);
  const setOpacity = useCallback((id, value) => dispatch({ type: 'SET_OPACITY', id, value }), []);
  const setBlend = useCallback((id, value) => dispatch({ type: 'SET_BLEND', id, value }), []);
  const setVisibility = useCallback((id, value) => dispatch({ type: 'SET_VISIBILITY', id, value }), []);
  const setThreshold = useCallback(
    (id, value) => dispatch({ type: 'SET_THRESHOLD', id, value }),
    []
  );
  const setLayerParams = useCallback(
    (id, value) => {
      // Setting params changes the backend request → invalidate the
      // cached bitmap so the next render fetches the layer fresh.
      cacheFlush((cid) => cid === id);
      dispatch({ type: 'SET_LAYER_PARAMS', id, value });
    },
    [cacheFlush]
  );
  const reorder = useCallback((from, to) => dispatch({ type: 'REORDER', from, to }), []);
  const usePreset = useCallback((name) => {
    // Replacing the whole stack: flush everything so errors/bitmaps for
    // layers no longer in the stack don't bleed through.
    cacheFlush();
    dispatch({ type: 'REPLACE_ALL', layers: applyPreset(name) });
    // Remember the user's choice so we re-apply it when they switch
    // gallery entries.
    lastModeRef.current = { type: 'preset', name };
  }, [cacheFlush]);

  /** Replace the entire stack with a single layer at default opacity/blend.
   *  Replaces the old "Display Mode" dropdown — one click resets the view
   *  to just this layer. */
  const setSingleLayer = useCallback((id) => {
    const def = findLayerDef(id) ?? inferDynamicDef(id);
    if (!def) {
      console.warn('[useLayerStack] unknown layer id:', id);
      return;
    }
    cacheFlush();
    // Remember the user's choice so we re-apply it on result change.
    lastModeRef.current = { type: 'single', id };
    dispatch({
      type: 'REPLACE_ALL',
      layers: [{
        id, label: def.label ?? id, source: def.source,
        // Single-layer modes always render at full opacity / normal blend,
        // regardless of the layer's catalog defaults (which were tuned for
        // stacking on top of a base layer).
        opacity: 1.0,
        blend: 'normal',
        visible: true,
        key: `${id}-${Date.now()}`,
      }],
    });
  }, [cacheFlush]);
  const clear = useCallback(() => dispatch({ type: 'CLEAR' }), []);

  return {
    layers: state.layers,
    bitmaps: cacheRef.current,
    bitmapVersion,
    fetching: fetchingRef.current,
    errors: errorRef.current,
    addLayer, removeLayer, setOpacity, setBlend, setVisibility, setThreshold, setLayerParams, reorder,
    usePreset, setSingleLayer, clear,
  };
}
