/**
 * useEdsLayerStack — standalone layer-stack hook for the EDS page.
 *
 * Why not reuse PhaseMap's useLayerStack?
 *   1. ID format conflict. T3's edsLayerSources emits ids like `eds-Al Kα1`
 *      and `electron-SE1`. PhaseMap's useLayerStack only recognises `eds:*`
 *      and `se:*` shapes and would silently drop our ids in addLayer.
 *   2. displayMode-aware fetching. EDS maps need
 *      `edsApi.getMap(element, mode, ...)` (different endpoint, different
 *      namespace). PhaseMap's fetcher calls `h5Api.getEDSMap(element, ...)`
 *      with no mode parameter.
 *
 * What it shares with PhaseMap:
 *   - `layerStackReducer` (same action shapes, so T8's `LayerStackPanel`
 *     can sit on top of either hook without modification).
 *   - LRU bitmap cache pattern (max 16 entries, `.close()` on eviction).
 *   - bitmapVersion ping for canvas consumers.
 */
import { useReducer, useRef, useCallback, useEffect } from 'react';
import { layerStackReducer, initialState } from '../../PhaseMap/layerStackReducer';
import { edsApi, ebsdApi, h5Api, phaseMapApi } from '../../../services/api';
import useEdsColorStore from '../../../stores/useEdsColorStore';
import { isCropWarning } from '../../common/CropWarningChip';

const CACHE_SIZE = 16;

async function pngBase64ToBitmap(b64) {
  const blob = await fetch(`data:image/png;base64,${b64}`).then((r) => r.blob());
  return createImageBitmap(blob);
}

/** Route by `layer.kind` (preferred) with id-prefix fallback for safety.
 *  Throws on unknown kind — the per-layer error path catches it. */
async function fetchLayer(layer, displayMode) {
  if (layer.kind === 'eds-element') {
    // Parse element symbol out of the full label (e.g. "Al Kα1" → "Al").
    const labelStr = (layer.element || '').replace(/^Window Integral\s*/i, '').trim();
    const symbol = labelStr.split(/\s/)[0] || labelStr;
    const color = useEdsColorStore.getState().getColor(symbol) || '';
    // cmap argument is ignored by the backend when color is non-empty;
    // we still pass 'gray' as a sane fallback rather than 'hot' to avoid
    // confusion if the colour resolver ever yields the empty string.
    return edsApi.getMap(layer.element, displayMode, 'gray', color);
  }
  if (layer.kind === 'electron') {
    // 'dataset' — this stack renders the ACTIVE dataset, so under a crop the
    // electron image must arrive cut to the same physical region as the EDS
    // maps beside it.
    return h5Api.getElectronImage(layer.electronName, 'dataset');
  }
  if (layer.kind === 'vbse') return ebsdApi.virtualBSE('gray');
  if (layer.kind === 'bc')   return ebsdApi.bandContrast('gray');
  if (
    layer.kind === 'phase' ||
    (typeof layer.kind === 'string' && layer.kind.startsWith('ipf')) ||
    layer.kind === 'ci' ||
    (typeof layer.id === 'string' && layer.id.startsWith('ci_'))
  ) {
    return phaseMapApi.layer(layer.id, {});
  }
  throw new Error(`Unsupported EDS layer kind: ${layer.kind || layer.id}`);
}

export function useEdsLayerStack({ initialLayers, displayMode, cacheSize = CACHE_SIZE }) {
  const [state, dispatch] = useReducer(layerStackReducer, initialState);
  const cacheRef    = useRef(new Map());     // layerId → ImageBitmap
  const orderRef    = useRef([]);            // LRU order (most-recent at end)
  const fetchingRef = useRef(new Set());     // ids currently in-flight
  const errorRef    = useRef(new Map());     // layerId → error string
  const sourceRef   = useRef(new Map());     // layerId → backend `source` (provenance)
  // layerId → the backend's `crop` verdict, kept ONLY where a layer could not
  // follow the active crop faithfully. An electron image lives on its own,
  // finer grid: a file that cannot place the two areas hands back the FULL
  // image, and a window running past that area's edge yields a clamped
  // cut-out. Either way the user has to be told — neither is visible in the
  // picture.
  const cropRef     = useRef(new Map());     // layerId → { cropped, exact, reason }
  const displayModeRef = useRef(displayMode);
  const shapeRef    = useRef(null);          // [H, W] — the page's interaction grid
  const shapeFromGridRef = useRef(false);    // true once `shapeRef` came from a scan-grid layer
  const [bitmapVersion, force] = useReducer((x) => x + 1, 0);

  // Reactive subscription to the element→colour map. EDS element maps are
  // backend-rendered PNGs with the colour BAKED IN (see fetchLayer), so a
  // colour edit cannot be re-tinted client-side — the affected layers must
  // be refetched. fetchLayer still reads the colour via getState() at fetch
  // time; this subscription only exists to drive the invalidation effect.
  const elementColors      = useEdsColorStore((s) => s.colors);
  const prevColorsRef      = useRef(null);          // baseline for diffing
  const pendingColorIdsRef = useRef(new Set());     // ids awaiting debounced flush
  const colorFlushTimerRef = useRef(null);          // debounce timer handle

  const cacheTouch = useCallback((id) => {
    const o = orderRef.current;
    const i = o.indexOf(id);
    if (i !== -1) o.splice(i, 1);
    o.push(id);
  }, []);

  const cacheSet = useCallback((id, bitmap) => {
    const c = cacheRef.current;
    const o = orderRef.current;
    if (c.has(id)) {
      try { c.get(id).close(); } catch { /* ignore */ }
    }
    c.set(id, bitmap);
    cacheTouch(id);
    while (o.length > cacheSize) {
      const evictId = o.shift();
      try { c.get(evictId)?.close(); } catch { /* ignore */ }
      c.delete(evictId);
    }
    force();
  }, [cacheTouch, cacheSize]);

  const cacheFlush = useCallback((predicate) => {
    const c = cacheRef.current;
    const o = orderRef.current;
    const errs = errorRef.current;
    const srcs = sourceRef.current;
    for (const [id, bmp] of c.entries()) {
      if (predicate(id)) {
        try { bmp.close(); } catch { /* ignore */ }
        c.delete(id);
        const i = o.indexOf(id);
        if (i !== -1) o.splice(i, 1);
      }
    }
    for (const id of [...errs.keys()]) {
      if (predicate(id)) errs.delete(id);
    }
    // Drop stale provenance alongside the cache/error entries so a file switch
    // (REPLACE_ALL flush) or removeLayer never shows a previous file's source.
    for (const id of [...srcs.keys()]) {
      if (predicate(id)) srcs.delete(id);
    }
    for (const id of [...cropRef.current.keys()]) {
      if (predicate(id)) cropRef.current.delete(id);
    }
    force();
  }, []);

  const invalidateBitmapsForLayer = useCallback((id) => {
    cacheFlush((cid) => cid === id);
  }, [cacheFlush]);

  // Seed / re-seed on initialLayers reference change (EDS page feeds this
  // from useDefaultLayers — each file-open produces a fresh array).
  useEffect(() => {
    if (Array.isArray(initialLayers)) {
      // A fresh initialLayers array means a new file (or file switch) — the
      // scan grid may differ, so drop the cached interaction shape and let
      // the next scan-grid fetch re-establish it.
      shapeRef.current = null;
      shapeFromGridRef.current = false;
      // Flush ALL cached bitmaps. Layer ids are stable across files
      // (`eds-Al`, `bc`, `electron-SE1`, …), so without this the fetch loop
      // would skip every re-seeded layer (`cacheRef.has(l.id)` is true) and
      // keep rendering the PREVIOUS file's pixels under the new file's labels.
      cacheFlush(() => true);
      dispatch({ type: 'REPLACE_ALL', layers: initialLayers });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialLayers]);

  // Mirror displayMode into a ref so async fetches can detect a mode change
  // that happened between fetch-start and bitmap-decode.
  useEffect(() => {
    displayModeRef.current = displayMode;
  }, [displayMode]);

  // displayMode change → drop EDS-element bitmaps so the fetch loop
  // re-requests them with the new mode. Non-EDS layers stay cached.
  useEffect(() => {
    const edsIds = new Set(
      state.layers.filter((l) => l.kind === 'eds-element').map((l) => l.id),
    );
    if (edsIds.size > 0) cacheFlush((cid) => edsIds.has(cid));
    // intentionally NOT depending on state.layers — we only want this to
    // run on displayMode change. New layers go through the addLayer path
    // which triggers a fetch on its own.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayMode]);

  // colour change → flush ONLY the EDS-element bitmaps whose colour actually
  // changed, so the fetch loop re-requests them with the new colour baked in.
  // Mirrors the displayMode invalidation above. Debounced because the colour
  // picker (<input type=color>) fires onChange continuously while dragging.
  useEffect(() => {
    // First run = record baseline; never flush on mount (avoids a double-fetch
    // right after seeding, which already fetches via the visible-layer loop).
    if (prevColorsRef.current === null) {
      prevColorsRef.current = elementColors;
      return;
    }
    const prev = prevColorsRef.current;
    const next = elementColors;
    prevColorsRef.current = next;

    for (const l of state.layers) {
      if (l.kind !== 'eds-element') continue;
      // Derive the element symbol exactly as fetchLayer does.
      const labelStr = (l.element || '').replace(/^Window Integral\s*/i, '').trim();
      const symbol = labelStr.split(/\s/)[0] || labelStr;
      if ((prev?.[symbol] || '') !== (next?.[symbol] || '')) {
        pendingColorIdsRef.current.add(l.id);
      }
    }
    if (pendingColorIdsRef.current.size === 0) return;

    if (colorFlushTimerRef.current) clearTimeout(colorFlushTimerRef.current);
    colorFlushTimerRef.current = setTimeout(() => {
      const ids = pendingColorIdsRef.current;
      pendingColorIdsRef.current = new Set();
      colorFlushTimerRef.current = null;
      cacheFlush((cid) => ids.has(cid));
    }, 120);
    // only depends on elementColors — see the displayMode effect's rationale.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elementColors]);

  const doFetch = useCallback(async (layer) => {
    if (fetchingRef.current.has(layer.id)) return;
    if (cacheRef.current.has(layer.id)) { cacheTouch(layer.id); return; }
    fetchingRef.current.add(layer.id);
    const startedMode = displayMode;
    try {
      const res = await fetchLayer(layer, startedMode);
      const b64 = res?.data?.image;
      if (!b64) throw new Error('Empty image payload');
      const bitmap = await pngBase64ToBitmap(b64);
      // Stale-mode guard: if displayMode changed during the await, the bitmap
      // we just decoded reflects the OLD mode for an EDS-element layer. Drop
      // it so the visible refetch (kicked off by the mode-change invalidation)
      // can fill the cache with current data.
      if (layer.kind === 'eds-element' && startedMode !== displayModeRef.current) {
        try { bitmap.close(); } catch { /* ignore */ }
        return;
      }
      // `shape` is the page's INTERACTION grid — quantify / region / probe /
      // linescan coords are all in the EBSD scan grid. Adopt it from a
      // scan-grid layer (EDS element / BC / V-BSE / phase / IPF / CI), never
      // from an electron image (a separate, higher-res SEM survey image that
      // fetches first and would otherwise put every click on the wrong pixel).
      if (res?.data?.shape) {
        const isGrid = layer.kind !== 'electron';
        if (isGrid && !shapeFromGridRef.current) {
          shapeRef.current = res.data.shape;
          shapeFromGridRef.current = true;
        } else if (!isGrid && !shapeRef.current) {
          shapeRef.current = res.data.shape;   // electron-only fallback
        }
      }
      cacheSet(layer.id, bitmap);
      errorRef.current.delete(layer.id);
      // Record provenance (e.g. BC layer's "h5oina" native vs "computed" FFT
      // pattern-quality) so the panel can surface which one is displayed.
      if (res?.data?.source) sourceRef.current.set(layer.id, res.data.source);
      // Only a complaint is worth keeping: "cropped and exact" is the answer
      // for every layer whether or not a crop is active, so it would say
      // nothing. `exact: false` counts as a complaint — see CropWarningChip.
      if (isCropWarning(res?.data?.crop)) {
        cropRef.current.set(layer.id, res.data.crop);
      } else {
        cropRef.current.delete(layer.id);
      }
    } catch (err) {
      errorRef.current.set(
        layer.id,
        err?.response?.data?.detail ?? err.message ?? 'fetch failed',
      );
      force();
    } finally {
      fetchingRef.current.delete(layer.id);
    }
  }, [displayMode, cacheSet, cacheTouch]);

  // Visible layers without a cached bitmap → fetch. bitmapVersion is in
  // the deps so an invalidation (which bumps the version) immediately
  // re-runs the loop for the now-missing entry. Mask layers (kind='mask')
  // are derived from another layer's bitmap and have no remote source —
  // skip them.
  useEffect(() => {
    Promise.all(
      state.layers
        .filter((l) => l.visible && l.kind !== 'mask' && !cacheRef.current.has(l.id))
        .map((l) => doFetch(l)),
    );
  }, [state.layers, doFetch, bitmapVersion]);

  // -------------------- public API --------------------
  const addLayer = useCallback((layer) => dispatch({ type: 'ADD', layer }), []);
  const removeLayer = useCallback((id) => {
    cacheFlush((cid) => cid === id);
    dispatch({ type: 'REMOVE', id });
  }, [cacheFlush]);
  const setOpacity = useCallback((id, value) => dispatch({ type: 'SET_OPACITY', id, value }), []);
  const setBlend = useCallback((id, value) => dispatch({ type: 'SET_BLEND', id, value }), []);
  const setVisibility = useCallback((id, value) => dispatch({ type: 'SET_VISIBILITY', id, value }), []);
  const setThreshold = useCallback((id, value) => dispatch({ type: 'SET_THRESHOLD', id, value }), []);
  const addMaskFromThreshold = useCallback(
    (sourceId) => dispatch({ type: 'ADD_MASK_FROM_THRESHOLD', sourceId }),
    [],
  );
  const reorder = useCallback((from, to) => dispatch({ type: 'REORDER', from, to }), []);
  const clear = useCallback(() => dispatch({ type: 'CLEAR' }), []);

  // Unmount cleanup — ImageBitmap is GPU-backed and not garbage-collected
  // until .close() is called. Without this, every EDS-page visit leaks up
  // to CACHE_SIZE bitmaps.
  useEffect(() => () => {
    if (colorFlushTimerRef.current) clearTimeout(colorFlushTimerRef.current);
    for (const bmp of cacheRef.current.values()) {
      try { bmp.close(); } catch { /* ignore */ }
    }
    cacheRef.current.clear();
    orderRef.current.length = 0;
  }, []);

  return {
    layers: state.layers,
    bitmaps: cacheRef.current,
    bitmapVersion,
    errors: errorRef.current,
    layerSources: sourceRef.current,
    layerCropStatus: cropRef.current,
    shape: shapeRef.current,
    addLayer,
    removeLayer,
    setOpacity,
    setBlend,
    setVisibility,
    setThreshold,
    addMaskFromThreshold,
    reorder,
    clear,
    invalidateBitmapsForLayer,
  };
}
