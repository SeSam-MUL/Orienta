/**
 * useGridNavigation — owns row/col/index/pattern fetch state for HDF5Viewer.
 *
 * The race-safe pattern fetch (navSeqRef) is preserved verbatim from the
 * original component: every fetch bumps a sequence counter, late responses
 * see a mismatched counter and silently abandon their writes. Without this,
 * fast slider scrubbing causes earlier (slower) responses to overwrite the
 * pattern from a later (faster) one.
 *
 * Transport: `h5Api.getPatternBinary` returns the PNG as a Blob, which is
 * wrapped in `URL.createObjectURL`. The store now holds either a `blob:` URL
 * (this hook's hot path) or a bare base64 string (legacy writers like
 * sessionStorage hydration). PatternCanvas + the export/copy paths detect
 * which form and render accordingly. This saves ~33% bandwidth (vs base64
 * over JSON) plus the JSON parse — measurable on arrow-key sweeps.
 *
 * Prefetch: after each successful navigation, ±1 neighbor is fire-and-forget
 * fetched into an in-memory cache. The cache is pruned on every navigation
 * (anything farther than ±1 from the current index gets `URL.revokeObjectURL`d
 * to avoid leaks). On unmount everything is revoked.
 *
 * Returned API:
 *   row, col, index, pattern  — current position (mirrored from the store)
 *   patternLoading            — fetch in flight
 *   navigate(row, col, index) — jump to a specific cell
 *   navigateByIndex(index)    — convenience: derives row/col from cols
 *   navigateRow(row), navigateCol(col) — incremental navigation
 *
 * The store remains the source of truth for currentRow/Col/Index/Pattern —
 * other modules (EBSDViewer, IndexingPage) read from it.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import useDataStore from '../../../stores/useDataStore';
import { h5Api } from '../../../services/api';

export function useGridNavigation({ gridShape, patternCount, isOpen }) {
  const currentRow     = useDataStore((s) => s.currentRow);
  const currentCol     = useDataStore((s) => s.currentCol);
  const currentIndex   = useDataStore((s) => s.currentIndex);
  const currentPattern = useDataStore((s) => s.currentPattern);
  const setPosition    = useDataStore((s) => s.setPosition);

  const [patternLoading, setPatternLoading] = useState(false);

  // Race-safety: counter that bumps on every fetch; late responses bail out.
  const navSeqRef = useRef(0);

  // Mounted flag: flipped to false in the unmount cleanup (BEFORE the cache is
  // revoked) so that any in-flight prefetch / fetch can detect "we are gone"
  // at every async boundary and refuse to insert into the cache. Without this,
  // a fast unmount during a quick tab-switch can leave a Blob URL stranded in
  // the map after the cleanup already iterated over it — silent leak.
  const mountedRef = useRef(true);

  // objectUrlsRef caches blob URLs by index. revokeOld() prunes anything
  // outside ±1 of the current index, so the cache holds at most ~3 entries
  // at rest, but can briefly hold up to ~5 during rapid navigation while
  // in-flight prefetches from the previous step settle. Acceptable for
  // the use case.
  const objectUrlsRef = useRef(new Map());

  const rows = gridShape?.[0] || 1;
  const cols = gridShape?.[1] || 1;
  const maxIndex = Math.max(0, (patternCount ?? 0) - 1);

  /** Fetch the binary pattern for `idx`, return its blob: URL (cached). */
  const fetchPatternBinary = useCallback(async (idx) => {
    const cached = objectUrlsRef.current.get(idx);
    if (cached) return cached;
    const res = await h5Api.getPatternBinary(idx);
    // If we unmounted while the request was in flight, don't create a URL
    // and don't insert into the cache — the unmount cleanup already ran and
    // anything we add now would leak. The Blob itself is GC'd normally.
    if (!mountedRef.current) return null;
    const url = URL.createObjectURL(res.data);
    // Re-check cache after the await — a parallel fetch for the same idx
    // could have completed first; if so prefer the existing URL and revoke
    // ours to avoid a leaked object. Also re-check mount: extremely small
    // race between the createObjectURL above and unmount running.
    if (!mountedRef.current) {
      URL.revokeObjectURL(url);
      return null;
    }
    const existing = objectUrlsRef.current.get(idx);
    if (existing) {
      URL.revokeObjectURL(url);
      return existing;
    }
    objectUrlsRef.current.set(idx, url);
    return url;
  }, []);

  /** Revoke all cached URLs whose index is farther than ±1 from `currentIdx`. */
  const revokeOld = useCallback((currentIdx) => {
    for (const [idx, url] of objectUrlsRef.current.entries()) {
      if (Math.abs(idx - currentIdx) > 1) {
        URL.revokeObjectURL(url);
        objectUrlsRef.current.delete(idx);
      }
    }
  }, []);

  /**
   * Fetch the pattern at flat-index `index` and write it to the store at
   * (row, col, index). Race-safe: only the most recent invocation gets to
   * commit its result. Schedules ±1 neighbor prefetch (fire-and-forget) and
   * prunes far-away cache entries on success.
   */
  const fetchAndSetPattern = useCallback(async (row, col, index) => {
    if (!patternCount || patternCount === 0) return;
    const seq = ++navSeqRef.current;
    setPatternLoading(true);
    try {
      const url = await fetchPatternBinary(index);
      // Bail if either (a) a newer fetch superseded us, or (b) we unmounted
      // while the request was in flight. Both must be checked — navSeqRef
      // alone doesn't catch unmount.
      if (navSeqRef.current !== seq || !mountedRef.current) return;
      setPosition(row, col, index, url);
      // Schedule prefetch + cleanup off the critical path. Prefetch failures
      // are non-fatal (it's an optimization, not a contract).
      setTimeout(() => {
        if (!mountedRef.current) return;
        [index - 1, index + 1].forEach((n) => {
          if (n >= 0 && n <= maxIndex && !objectUrlsRef.current.has(n)) {
            fetchPatternBinary(n).catch((err) => {
              console.warn(`prefetch failed for pattern ${n}:`, err);
            });
          }
        });
        revokeOld(index);
      }, 0);
    } catch {
      if (navSeqRef.current !== seq || !mountedRef.current) return;
      setPosition(row, col, index, null);
    } finally {
      if (navSeqRef.current === seq && mountedRef.current) setPatternLoading(false);
    }
  }, [patternCount, setPosition, fetchPatternBinary, revokeOld, maxIndex]);

  // Cleanup all object URLs on unmount to avoid leaks. Flip mountedRef FIRST
  // so any in-flight prefetch/fetch sees us as gone at its next await boundary
  // and refuses to insert a fresh URL into the (now-revoked) cache.
  useEffect(() => {
    const cache = objectUrlsRef.current;
    return () => {
      mountedRef.current = false;
      for (const url of cache.values()) URL.revokeObjectURL(url);
      cache.clear();
    };
  }, []);

  const navigate = useCallback((row, col, index) => {
    fetchAndSetPattern(row, col, index);
  }, [fetchAndSetPattern]);

  const navigateByIndex = useCallback((newIndex) => {
    const idx = Math.max(0, Math.min(newIndex, maxIndex));
    fetchAndSetPattern(Math.floor(idx / cols), idx % cols, idx);
  }, [cols, fetchAndSetPattern, maxIndex]);

  const navigateRow = useCallback((newRow) => {
    const r = Math.max(0, Math.min(newRow, rows - 1));
    fetchAndSetPattern(r, currentCol, Math.min(r * cols + currentCol, maxIndex));
  }, [cols, currentCol, fetchAndSetPattern, maxIndex, rows]);

  const navigateCol = useCallback((newCol) => {
    const c = Math.max(0, Math.min(newCol, cols - 1));
    fetchAndSetPattern(currentRow, c, Math.min(currentRow * cols + c, maxIndex));
  }, [cols, currentRow, fetchAndSetPattern, maxIndex]);

  return {
    row: currentRow,
    col: currentCol,
    index: currentIndex,
    pattern: currentPattern,
    patternLoading,
    rows,
    cols,
    maxIndex,
    navigate,
    navigateByIndex,
    navigateRow,
    navigateCol,
    fetchAndSetPattern,
    setPatternLoading,  // exposed for the "auto-fetch on file open" effect
  };
}

export default useGridNavigation;
