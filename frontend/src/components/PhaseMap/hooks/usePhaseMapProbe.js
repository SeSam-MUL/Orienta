import { useCallback, useRef, useState } from 'react';
import { phaseMapApi } from '../../../services/api';

const CACHE_MAX = 32;

/**
 * Debounced + LRU-cached probe fetcher for the phase map. Multiple rapid cursor
 * moves coalesce into one request after the hover settles; repeat visits to the
 * same pixel are served from cache. PhaseMap probes are mode-agnostic, so the
 * cache key is just `${row},${col}` (no displayMode like the EDS probe hook).
 */
export function usePhaseMapProbe({ debounceMs = 60 } = {}) {
  const cacheRef = useRef(new Map());      // key -> data
  const timerRef = useRef(null);
  const inFlightRef = useRef(null);
  const [probe, setProbe] = useState(null);
  const [error, setError] = useState(null);

  const requestProbe = useCallback((row, col) => {
    const key = `${row},${col}`;
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
        const res = await phaseMapApi.probe(row, col);
        if (token.cancel) return;
        const data = res.data;
        // LRU insert: evict oldest if at capacity.
        if (cacheRef.current.size >= CACHE_MAX) {
          const firstKey = cacheRef.current.keys().next().value;
          cacheRef.current.delete(firstKey);
        }
        cacheRef.current.set(key, data);
        setProbe(data);
        setError(null);
      } catch (e) {
        if (token.cancel) return;
        setError(e?.response?.data?.detail || e?.message || 'probe failed');
        setProbe(null);
      }
    }, debounceMs);
  }, [debounceMs]);

  const clear = useCallback(() => {
    setProbe(null);
    setError(null);
  }, []);

  return { probe, error, requestProbe, clear };
}
