/**
 * useDiagnosticsQuery - fetch summary + browser results for the Anomaly Browser.
 *
 * Matches the in-memory-cache pattern used by useLayerStack: results are cached
 * per (result_id, metric, sort, range, n) and re-fetched only on key change.
 *
 * IMPORTANT: only *successful, non-null* responses are cached. A 404 (diagnostics
 * not yet computed) returns `null` from forwardDiagApi.summary/browser — caching
 * that null would permanently mask a later compute. Instead, summary re-fetches
 * every time the drawer opens (the `refreshKey` arg becomes part of the effect
 * deps), so a fresh compute is always picked up.
 */
import { useEffect, useRef, useState } from 'react';
import { forwardDiagApi } from '../../../services/api';

const _cache = new Map();   // key → { value, ts }
const _MAX = 50;
const _put = (k, v) => {
  if (_cache.size > _MAX) _cache.delete(_cache.keys().next().value);
  _cache.set(k, { value: v, ts: Date.now() });
};

/**
 * Fetch the diagnostics summary for a result.
 *
 * @param {string|undefined} resultId  active indexing result id
 * @param {number}           refreshKey  bump this (e.g. on drawer open) to force
 *                                       a re-fetch even if a prior call 404'd.
 * @returns {{summary, error, loading}}
 *   - loading === true   → request in flight
 *   - summary === null   → backend returned 404 (diagnostics not computed)
 *   - summary present    → diagnostics available
 */
export function useDiagnosticsSummary(resultId, refreshKey = 0) {
  const [summary, setSummary] = useState(undefined);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setError(null);
    if (!resultId) { setSummary(undefined); setLoading(false); return; }
    const k = `summary:${resultId}`;
    // Serve only cached *non-null* summaries. A cached null is never stored
    // (see below), so a hit here always means "diagnostics exist".
    if (_cache.has(k)) {
      setSummary(_cache.get(k).value);
      setLoading(false);
      return;
    }
    setLoading(true);
    forwardDiagApi.summary(resultId).then((s) => {
      if (cancelled) return;
      // Only cache successful (non-null) summaries. Caching a 404-null would
      // permanently shadow a later compute for this result_id.
      if (s) _put(k, s);
      setSummary(s);   // s may be null → "not computed" state
      setLoading(false);
    }).catch((e) => {
      if (!cancelled) { setError(e.message || String(e)); setLoading(false); }
    });
    return () => { cancelled = true; };
  }, [resultId, refreshKey]);
  return { summary, error, loading };
}

export function useDiagnosticsBrowser(resultId, opts) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const k = `browser:${resultId}:${opts?.sort || ''}:${opts?.n || ''}:${opts?.range_min ?? ''}:${opts?.range_max ?? ''}`;
  const lastKeyRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    if (!resultId) { setData(null); return; }
    if (lastKeyRef.current === k) return;
    lastKeyRef.current = k;
    if (_cache.has(k)) { setData(_cache.get(k).value); return; }
    setLoading(true);
    forwardDiagApi.browser(resultId, opts).then((d) => {
      if (cancelled) return;
      // Only cache truthy responses (parity with summary; browser throws on
      // 404 today but stay defensive in case that ever changes).
      if (d) _put(k, d);
      setData(d);
      setLoading(false);
    }).catch((e) => { if (!cancelled) { setError(e.message || String(e)); setLoading(false); } });
    return () => { cancelled = true; };
  }, [k, resultId]);

  return { data, error, loading };
}

export function clearDiagnosticsCache(resultId) {
  if (!resultId) { _cache.clear(); return; }
  for (const key of Array.from(_cache.keys())) {
    if (key.includes(`:${resultId}`)) _cache.delete(key);
  }
}
