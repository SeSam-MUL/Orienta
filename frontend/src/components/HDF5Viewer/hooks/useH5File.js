/**
 * useH5File — manages the H5OINA backend-file lifecycle for HDF5Viewer.
 *
 * Single source of truth for "is the backend H5 session open?" — replaces
 * three earlier conflicting useEffects in HDF5Viewer.jsx:
 *   1) auto-sync from /api/h5/status on mount (file already open server-side)
 *   2) pre-fill the path-input when isFileOpen flips true
 *   3) auto-fetch index 0 when the file was opened externally
 *
 * Returned API:
 *   isOpen     — bool, mirror of useDataStore.isFileOpen
 *   filePath   — current file path (from store)
 *   openLoading — true while POST /api/h5/open is in flight
 *   openError  — last user-facing open error (string), or null
 *   open(path) — POST /api/h5/open + auto-fetch first pattern
 *   close()    — POST /api/h5/close + clear store
 *   syncFromStatus() — call on mount to pick up an externally-opened file
 *
 * The hook deliberately does NOT auto-trigger open(filePath) from a useEffect:
 *   - Open is user-initiated (click, Enter key, or Browse dialog).
 *   - Auto-sync is a separate one-shot, not a reactive watcher.
 *   This avoids the "open-loop" bug the original code had where the path-input
 *   change feedback reopened the same file.
 */
import { useState, useCallback, useEffect, useRef } from 'react';
import i18n from '../../../i18n';
import useDataStore from '../../../stores/useDataStore';
import { h5Api } from '../../../services/api';

export function useH5File() {
  const isOpen   = useDataStore((s) => s.isFileOpen);
  const filePath = useDataStore((s) => s.filePath);
  const setFileData = useDataStore((s) => s.setFileData);
  const clearFile   = useDataStore((s) => s.clearFile);
  const setPosition = useDataStore((s) => s.setPosition);

  const [openLoading, setOpenLoading] = useState(false);
  const [openError,   setOpenError]   = useState(null);

  // Track unmount so async tasks don't write into a dead component.
  const aliveRef = useRef(true);
  useEffect(() => () => { aliveRef.current = false; }, []);

  /**
   * Open a file. Returns the response data on success; throws on failure.
   * Also fetches the first pattern (index 0) so the canvas isn't empty.
   */
  const open = useCallback(async (path) => {
    const trimmed = (path ?? '').trim();
    if (!trimmed) return null;
    setOpenLoading(true);
    setOpenError(null);
    try {
      const res = await h5Api.open(trimmed);
      if (!aliveRef.current) return res.data;
      setFileData(res.data);
      if (res.data?.pattern_count > 0) {
        try {
          const patRes = await h5Api.getPattern(0);
          if (!aliveRef.current) return res.data;
          setPosition(0, 0, 0, patRes.data?.image ?? null);
        } catch {
          if (aliveRef.current) setPosition(0, 0, 0, null);
        }
      }
      return res.data;
    } catch (err) {
      const msg = err?.response?.data?.detail ?? err?.message ?? i18n.t('hdf5viewer:errors.openFailed');
      if (aliveRef.current) setOpenError(msg);
      throw err;
    } finally {
      if (aliveRef.current) setOpenLoading(false);
    }
  }, [setFileData, setPosition]);

  const close = useCallback(async () => {
    try { await h5Api.close(); } catch { /* swallow — we clear locally regardless */ }
    if (aliveRef.current) {
      clearFile();
      setOpenError(null);
    }
  }, [clearFile]);

  /**
   * One-shot: if the backend already has a file open (e.g. EBSD Viewer
   * loaded it), populate the React store + first pattern. Returns true if
   * a sync happened. Safe to call multiple times — early-exits if already
   * synced.
   */
  const syncFromStatus = useCallback(async () => {
    if (isOpen) return false;
    try {
      const res = await h5Api.status();
      if (!aliveRef.current || !res.data?.is_open) return false;
      const backendPath = res.data.file_path;
      const openRes = await h5Api.open(backendPath);
      if (!aliveRef.current) return false;
      setFileData(openRes.data);
      if (openRes.data?.pattern_count > 0) {
        try {
          const patRes = await h5Api.getPattern(0);
          if (!aliveRef.current) return true;
          setPosition(0, 0, 0, patRes.data?.image ?? null);
        } catch {
          if (aliveRef.current) setPosition(0, 0, 0, null);
        }
      }
      return true;
    } catch {
      return false;
    }
  // isOpen intentionally excluded from deps: this is a one-shot bootstrap.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setFileData, setPosition]);

  return {
    isOpen,
    filePath,
    openLoading,
    openError,
    setOpenError,
    open,
    close,
    syncFromStatus,
  };
}

export default useH5File;
