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
import { useReducer, useRef, useCallback, useEffect, useState } from 'react';
import { layerStackReducer, initialState } from '../layerStackReducer';
import { findLayerDef } from '../layerSources';
import { isCropWarning } from '../../common/CropWarningChip';
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
async function fetchLayerImage({ layer, cleanupParams, colorOverrides }) {
  switch (layer.source) {
    case 'result': {
      // Backend `kind` matches the layer id for all result-source entries
      // (phase, ipf-x/y/z, bc, ci, ci_<phase>, uncertainty, ci-threshold).
      // Per-layer params (e.g. band_min/band_max for ci-threshold) live
      // on the layer object and are merged into the request alongside the
      // global cleanupParams. cleanupParams win conflicts (rare).
      const mergedParams = { ...(layer.params || {}), ...cleanupParams };
      // Phase colouring is the only result layer that honours the user's
      // colour picks. Thread the overrides through so the LIVE map matches
      // the swatches, the legend and the /render export. Gated on a
      // non-empty map so plain results don't carry a needless "{}" param.
      if (layer.id === 'phase' && colorOverrides && Object.keys(colorOverrides).length > 0) {
        mergedParams.color_overrides = JSON.stringify(colorOverrides);
      }
      const res = await phaseMapApi.layer(layer.id, mergedParams);
      // `scale` is present only where the colours mean a number (CI, band
      // contrast, the diagnostics). It carries the range ACTUALLY painted, so
      // the legend states what the map shows rather than a nominal 0..1.
      return { base64: res.data.image, keyToAlpha: false, scale: res.data.scale ?? null };
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
      return { base64: res.data.image, keyToAlpha: false, scale: res.data.scale ?? null };
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
        // 'dataset': same reason as the SE underlay below — this element map
        // is composited onto the active dataset's phase map, so it has to be
        // cut to the same window. A full-scan map stretched onto a cropped
        // grid puts every feature in the wrong place, silently.
        const res = await h5Api.getEDSMap(element, 'hot', '', 'dataset');
        return { base64: res.data.image, keyToAlpha: true };
      }
      if (layer.id.startsWith('se:')) {
        const name = layer.id.slice(3);
        // 'dataset': same reason as in useEdsLayerStack — the phase map is
        // the active dataset's, so its SE underlay must follow the crop.
        const res = await h5Api.getElectronImage(name, 'dataset');
        // `crop` says whether the image could follow the active crop. Passed
        // on so the layer row can warn; a full-scan image silently stacked
        // under a cropped phase map would misplace every feature on it.
        return { base64: res.data.image, keyToAlpha: true, crop: res.data.crop };
      }
      throw new Error(`Unsupported h5 layer: ${layer.id}`);
    }
    default:
      throw new Error(`Unknown source: ${layer.source}`);
  }
}

/**
 * Does a change of the post-processing cleanup params change what this layer
 * LOOKS LIKE — i.e. must its cached bitmap be dropped so it re-fetches?
 *
 * This list has to match the kinds that `/api/phasemap/layer` renders through
 * `effective_pid_2d` / `ipf_valid_2d` (backend/api/routes/phase_map.py):
 *
 *   phase                     painted straight from effective_pid_2d
 *   ipf-x / ipf-y / ipf-z     the COLOUR is cleanup-independent, the ALPHA is
 *                             not: `alpha[:] = (effective_pid_2d >= 0)` then
 *                             `alpha &= ipf_valid_2d`
 *   bc                        same alpha rule when BC comes from the xmap
 *   ci / ci_<phase> / uncertainty
 *
 * Deliberately NOT here: `grain-boundaries` (misorientation is computed before
 * cleanup is applied and the endpoint returns before touching it),
 * `ci-threshold` (paints its own band, ignores effective ids), and every
 * non-result source — EDS elements, SE images, analysis maps, virtual BSE.
 *
 * Leaving an affected kind out of this list is SILENT and reads to the user as
 * "the sliders do nothing": the layer simply keeps its cached bitmap forever.
 * That was the 2026-09-03 bug — ipf-* and bc were missing, on the reasoning
 * that IPF *colour* does not depend on cleanup. Measured symptom: 30+ requests
 * carried a non-zero ci_threshold and not one of them was a /layer request.
 */
export function cleanupAffectsLayer(id) {
  const s = String(id ?? '');
  return (
    s === 'phase' || s === 'bc' || s === 'ci' || s === 'uncertainty'
    || s.startsWith('ipf-') || s.startsWith('ci_')
  );
}

export function useLayerStack({ cleanupParams, resetSignal, frameSig, colorOverrides = null }) {
  const [state, dispatch] = useReducer(layerStackReducer, initialState);
  const cacheRef = useRef(new Map());           // layerId → ImageBitmap
  // layerId → { min, max, unit, cmap, stops } for layers whose colours mean a
  // number. Kept beside the bitmaps so a legend can never outlive its map.
  const scaleRef = useRef(new Map());
  // layerId → the backend's `crop` verdict, kept ONLY where a layer could not
  // follow the active crop faithfully — refused outright, or clamped at the
  // edge of its own acquisition area (today: the electron images).
  const cropRef = useRef(new Map());
  const cacheOrderRef = useRef([]);             // LRU order (most recent at end)
  const fetchingRef = useRef(new Set());        // layer ids currently in-flight
  const errorRef = useRef(new Map());           // layerId → error string
  // Remember the last user-chosen "mode" (single quick-mode layer or
  // preset) so a result-change can re-apply it instead of forcing the
  // user back to Phase every time they click a different gallery entry.
  // Shape: { type: 'single', id } | { type: 'preset', name } | null
  const lastModeRef = useRef(null);
  // What each quick mode looked like when the user last left it, so switching
  // IPF-Z -> IPF-X -> IPF-Z brings the stack back instead of throwing the
  // user's work away. Keyed by mode id; cleared whenever the underlying result
  // changes, because those layers belong to that result's data.
  const modeStacksRef = useRef(new Map());
  // Which quick mode the current stack belongs to. Not derivable from the
  // stack: as soon as a second layer is added the stack is no longer "one
  // layer named X", but the user is still working inside mode X.
  const [activeMode, setActiveMode] = useState(null);
  // Ref twin of activeMode, so the switch callback reads the current value
  // without having to be rebuilt on every change.
  const activeModeRef = useRef(null);
  // The reducer state, readable from callbacks without listing it as a dep.
  const layersRef = useRef([]);

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
  layersRef.current = state.layers;

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
        // The legend goes with the map it describes: a bar left behind would
        // state the range of a picture that is no longer on screen.
        scaleRef.current.delete(id);
        cropRef.current.delete(id);
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
      const { base64, keyToAlpha, scale, crop } = await fetchLayerImage({ layer, cleanupParams, colorOverrides });
      if (!base64) throw new Error('Empty image payload');
      const bitmap = await pngBase64ToBitmap(base64, { keyToAlpha });
      // Result/file switched while we were fetching → this bitmap belongs to
      // the previous result. Drop it so it can't poison the cache and stall
      // the new result's render. The fetch effect re-fires for uncached layers.
      if (epoch !== epochRef.current) {
        try { bitmap.close(); } catch { /* ignore */ }
        // A newer flush (rapid layer switching, colour edit, result switch)
        // superseded this fetch. Ping a re-render so the fetch effect
        // re-evaluates: if this layer is still visible + uncached it gets
        // re-fetched. Without this, switching away-and-back to a layer while
        // its first fetch is in flight strands it with no bitmap and no
        // pending request (blank canvas until some later interaction).
        force();
        return;
      }
      cacheSet(layer.id, bitmap);
      if (scale) scaleRef.current.set(layer.id, scale);
      else scaleRef.current.delete(layer.id);
      // Only a complaint is worth keeping — see cropRef's declaration.
      if (isCropWarning(crop)) cropRef.current.set(layer.id, crop);
      else cropRef.current.delete(layer.id);
      errorRef.current.delete(layer.id);
    } catch (err) {
      errorRef.current.set(layer.id, err?.response?.data?.detail ?? err.message ?? 'fetch failed');
      force();
    } finally {
      fetchingRef.current.delete(layer.id);
    }
  }, [cleanupParams, colorOverrides, cacheSet, cacheTouch]);

  // Colour overrides are a fetch input for the phase layer only. Track the
  // last-seen signature so a colour edit drops the stale cached phase bitmap
  // *inside this same effect* — before the fetch loop reads the cache. Doing
  // the flush here (rather than in a separate effect) avoids relying on
  // cross-effect ordering and guarantees the phase layer re-fetches with the
  // new overrides. No-op when overrides are unchanged (identical signature).
  const colorSignature = JSON.stringify(colorOverrides ?? {});
  const prevColorSigRef = useRef(colorSignature);

  // Whenever the layer list (or colour overrides) change, ensure all visible
  // layers have bitmaps.
  useEffect(() => {
    if (prevColorSigRef.current !== colorSignature) {
      prevColorSigRef.current = colorSignature;
      cacheFlush((id) => id === 'phase');
    }
    Promise.all(
      state.layers
        // Skip layers already cached, currently in flight, or in a known
        // error state. Excluding errored layers is what keeps this safe to
        // re-run on every cache mutation (bitmapVersion) — a persistently
        // failing layer is retried only after a flush clears its error, not
        // in a tight loop.
        .filter((l) => l.visible
          && !cacheRef.current.has(l.id)
          && !fetchingRef.current.has(l.id)
          && !errorRef.current.has(l.id))
        .map((l) => fetchLayer(l))
    );
    // bitmapVersion is in deps so the effect re-runs after ANY cache mutation
    // (flush / set / dropped-fetch force()). That re-run is how a layer
    // stranded by the rapid-switch race gets re-fetched, and how frame/cleanup
    // flushes reliably reload their invalidated layers.
  }, [state.layers, fetchLayer, colorSignature, cacheFlush, bitmapVersion]);

  // Cleanup-param change → drop the cached bitmap of every layer the backend
  // renders differently under those params (see cleanupAffectsLayer).
  const cleanupSignature = JSON.stringify(cleanupParams ?? {});
  useEffect(() => {
    cacheFlush(cleanupAffectsLayer);
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
    // The parked stacks belong to the result they were built on.
    modeStacksRef.current = new Map();
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

  // Change what a layer shows without losing how it is set up.
  const retypeLayer = useCallback((id, nextId) => {
    if (!nextId || nextId === id) return;
    const def = findLayerDef(nextId) ?? inferDynamicDef(nextId);
    if (!def) {
      console.warn('[useLayerStack] unknown layer id:', nextId);
      return;
    }
    // The outgoing layer's bitmap is nobody's business now; dropping it also
    // clears any error it had recorded.
    cacheFlush((cid) => cid === id);
    dispatch({
      type: 'RETYPE',
      id,
      layer: {
        id: nextId, label: def.label ?? nextId, source: def.source,
        opacity: def.defaultOpacity ?? 1.0,
        blend: def.defaultBlend ?? 'normal',
        visible: true,
        params: def.params ? { ...def.params } : undefined,
        key: `${nextId}-${Date.now()}`,
      },
    });
  }, [cacheFlush]);

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

  /** Switch to a quick mode.
   *
   *  The FIRST visit to a mode starts it as a single layer at full opacity —
   *  the old "Display Mode" behaviour. After that the mode remembers whatever
   *  the user built there: extra layers, opacities, blends, order. Leaving a
   *  mode stores its stack; coming back restores it.
   *
   *  This used to replace the stack unconditionally, so anything added on top
   *  of IPF-Z was gone the moment you looked at IPF-X and came back. */
  const setSingleLayer = useCallback((id) => {
    const def = findLayerDef(id) ?? inferDynamicDef(id);
    if (!def) {
      console.warn('[useLayerStack] unknown layer id:', id);
      return;
    }
    // Park the stack we are leaving under its own mode.
    const leaving = activeModeRef.current;
    if (leaving && leaving !== id && layersRef.current.length) {
      modeStacksRef.current.set(leaving, layersRef.current);
    }
    if (leaving === id) return;   // already there — do not reset it

    cacheFlush();
    lastModeRef.current = { type: 'single', id };
    activeModeRef.current = id;
    setActiveMode(id);

    const saved = modeStacksRef.current.get(id);
    if (saved?.length) {
      dispatch({ type: 'REPLACE_ALL', layers: saved });
      return;
    }
    dispatch({
      type: 'REPLACE_ALL',
      layers: [{
        id, label: def.label ?? id, source: def.source,
        // A mode's FIRST layer renders at full opacity / normal blend,
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
    scales: scaleRef.current,
    cropStatus: cropRef.current,
    bitmapVersion,
    fetching: fetchingRef.current,
    errors: errorRef.current,
    addLayer, removeLayer, retypeLayer, setOpacity, setBlend, setVisibility, setThreshold, setLayerParams, reorder,
    usePreset, setSingleLayer, clear,
    activeMode,
    // Targeted invalidation for callers that mutate the backend result in
    // place (e.g. the pseudo-symmetry grain flip changes orientations →
    // IPF layers must refetch, same pattern as the frameSig effect above).
    cacheFlush,
  };
}
