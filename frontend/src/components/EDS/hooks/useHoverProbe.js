import { useCallback, useRef, useState } from 'react';
import { edsApi } from '../../../services/api';
import { edsDetailText } from '../edsDetailText';

const CACHE_MAX = 32;

/**
 * Debounced + LRU-cached probe fetcher. Multiple rapid cursor moves coalesce
 * into one request after the hover settles; repeat visits to the same pixel
 * are served from cache. Cache key includes displayMode so a mode change
 * forces a refetch for the same pixel.
 */
export function useHoverProbe({ displayMode = 'at_pct', debounceMs = 60 } = {}) {
  const cacheRef = useRef(new Map());      // key -> data
  const timerRef = useRef(null);
  const inFlightRef = useRef(null);
  const [probe, setProbe] = useState(null);
  const [error, setError] = useState(null);

  const requestProbe = useCallback((row, col) => {
    const key = `${row},${col},${displayMode}`;
    if (cacheRef.current.has(key)) {
      // Cache hit: serve immediately, no debounce or fetch.
      setProbe(cacheRef.current.get(key));
      setError(null);
      return;
    }
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      // Cancel any earlier in-flight fetch by marking its token stale.
      if (inFlightRef.current) inFlightRef.current.cancel = true;
      const token = { cancel: false };
      inFlightRef.current = token;
      try {
        const res = await edsApi.probe(row, col, displayMode);
        if (token.cancel) return;
        const data = res.data;
        // An answer whose EDS part was left out (`eds_status`: the spectrum
        // fit still running, or failed) is not cached: the values arrive
        // later, and a cached "pending" would outlive them.
        if (!data?.eds_status) {
          // LRU insert: evict oldest if at capacity.
          if (cacheRef.current.size >= CACHE_MAX) {
            const firstKey = cacheRef.current.keys().next().value;
            cacheRef.current.delete(firstKey);
          }
          cacheRef.current.set(key, data);
        }
        setProbe(data);
        setError(null);
      } catch (e) {
        if (token.cancel) return;
        setError(edsDetailText(e) || e?.message || 'probe failed');
        setProbe(null);
      }
    }, debounceMs);
  }, [displayMode, debounceMs]);

  const clear = useCallback(() => {
    setProbe(null);
    setError(null);
  }, []);

  return { probe, error, requestProbe, clear };
}
