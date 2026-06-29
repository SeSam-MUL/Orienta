/**
 * useBookmarks — per-file bookmark list, persisted in localStorage.
 *
 * Storage key: `hdf5_bm_v1_${filePath}` (versioned so future schema changes
 * — e.g. adding labels or notes to bookmarks — become a one-line migration
 * by bumping the version. Bookmarks stored under earlier unversioned keys
 * are intentionally not migrated; re-creating a handful of bookmarks is a
 * cheaper one-time cost than maintaining a migration path).
 *
 * Returned API:
 *   bookmarks   — array of { index, row, col }
 *   add(bm)     — append, no-op if index already bookmarked
 *   remove(idx) — remove by flat-index
 *   clear()     — drop all
 *   isBookmarked(index) — bool
 *
 * Uses functional setState in add/remove/clear so the persisted snapshot is
 * always the post-update value, not whatever was captured in the closure.
 */
import { useCallback, useEffect, useState } from 'react';

const storageKey = (filePath) => `hdf5_bm_v1_${filePath}`;

export function useBookmarks(filePath) {
  const [bookmarks, setBookmarks] = useState([]);

  // Load from localStorage when filePath changes.
  useEffect(() => {
    if (!filePath) { setBookmarks([]); return; }
    try {
      const raw = localStorage.getItem(storageKey(filePath));
      setBookmarks(raw ? JSON.parse(raw) : []);
    } catch {
      setBookmarks([]);
    }
  }, [filePath]);

  const persist = useCallback((next) => {
    if (!filePath) return;
    try { localStorage.setItem(storageKey(filePath), JSON.stringify(next)); }
    catch { /* quota or disabled storage — ignore, in-memory state still works */ }
  }, [filePath]);

  const add = useCallback((bm) => {
    setBookmarks((prev) => {
      if (prev.some((b) => b.index === bm.index)) return prev;
      const next = [...prev, bm];
      persist(next);
      return next;
    });
  }, [persist]);

  const remove = useCallback((index) => {
    setBookmarks((prev) => {
      const next = prev.filter((b) => b.index !== index);
      persist(next);
      return next;
    });
  }, [persist]);

  const clear = useCallback(() => {
    setBookmarks([]);
    persist([]);
  }, [persist]);

  const isBookmarked = useCallback(
    (index) => bookmarks.some((b) => b.index === index),
    [bookmarks],
  );

  return { bookmarks, add, remove, clear, isBookmarked };
}

export default useBookmarks;
