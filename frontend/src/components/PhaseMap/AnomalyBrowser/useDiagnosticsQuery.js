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

/**
 * @param {number} refreshKey  bumped by the drawer every time it opens. A request that failed is
 *   asked again then (a refusal can pass: the diagnostics get computed, a result is loaded), while
 *   a request that worked is not repeated just because the drawer was reopened. (It is not a
 *   dependency of the request effect itself: its cleanup would cancel a request still in flight
 *   when the drawer mounts and opens in the same breath.)
 */
export function useDiagnosticsBrowser(resultId, opts, refreshKey = 0) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const k = `browser:${resultId}:${opts?.sort || ''}:${opts?.n || ''}:${opts?.range_min ?? ''}:${opts?.range_max ?? ''}`;
  const lastKeyRef = useRef(null);
  const failedRef = useRef(false);
  const [retry, setRetry] = useState(0);

  // Reopening the drawer after a failure asks again.
  useEffect(() => {
    if (!failedRef.current) return;
    failedRef.current = false;
    lastKeyRef.current = null;
    setRetry((n) => n + 1);
  }, [refreshKey]);

  useEffect(() => {
    let cancelled = false;
    if (!resultId) { setData(null); setError(null); return; }
    if (lastKeyRef.current === k) return;
    lastKeyRef.current = k;
    failedRef.current = false;
    setError(null);
    if (_cache.has(k)) { setData(_cache.get(k).value); return; }
    setLoading(true);
    forwardDiagApi.browser(resultId, opts).then((d) => {
      if (cancelled) return;
      // Only cache truthy responses (parity with summary; browser throws on
      // 404 today but stay defensive in case that ever changes).
      if (d) _put(k, d);
      setData(d);
      setLoading(false);
    }).catch((e) => {
      if (cancelled) return;
      // The error itself, not its message: a coded refusal carries `code` (see _fwdDiagError),
      // which the drawer words in the language of the page. The rows of the previous request
      // belong to another sort or tab and must not stay on screen next to the reason.
      setError(e || new Error('request failed'));
      setData(null);
      setLoading(false);
      // A failure is not an answer to remember (see the refreshKey effect above).
      failedRef.current = true;
    });
    return () => { cancelled = true; };
  }, [k, resultId, retry]);

  return { data, error, loading };
}

export function clearDiagnosticsCache(resultId) {
  if (!resultId) { _cache.clear(); return; }
  for (const key of Array.from(_cache.keys())) {
    if (key.includes(`:${resultId}`)) _cache.delete(key);
  }
}
