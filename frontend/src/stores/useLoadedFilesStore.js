/**
 * Shared loaded-files registry mirror.
 *
 * The backend keeps an append-only list of every H5OINA/H5 file the user has
 * loaded this session (GET /api/ebsd/loaded-files). Several places show that
 * list: the header <FileSwitcher/>, the EBSD viewer's "Loaded Files" sidebar
 * panel, and the Indexing page's dataset dropdown.
 *
 * Before this store each of those kept its OWN copy with its OWN refresh
 * triggers, so pruning the list in one place left the others showing stale
 * entries (the "the File dropdown still lists files I removed" bug). They now
 * all subscribe to `files` here and call the same mutating actions, so one
 * refresh updates every consumer at once.
 *
 * This is a thin mirror of backend state — the backend list is authoritative.
 */

import { create } from 'zustand';
import { ebsdApi } from '../services/api';

const useLoadedFilesStore = create((set, get) => ({
  // [{ path, name, active }] — exactly what /loaded-files returns.
  files: [],

  /**
   * Re-fetch the registry from the backend. Every consumer subscribes to
   * `files`, so a single refresh re-renders all of them. Swallows errors by
   * emptying the list (backend unreachable → nothing to switch between),
   * matching the previous per-component behaviour.
   */
  refresh: async () => {
    try {
      const res = await ebsdApi.loadedFiles();
      set({ files: Array.isArray(res.data?.files) ? res.data.files : [] });
      return true;
    } catch {
      set({ files: [] });
      return false;
    }
  },

  // --- Mutations: hit the backend, then refresh so all consumers update. ---
  // These intentionally let the API error propagate so the caller can surface
  // it (toast / inline message); the list is left unchanged on failure.

  /** Remove a single file (backend refuses to remove the active file). */
  remove: async (path) => {
    await ebsdApi.removeLoadedFile(path);
    await get().refresh();
  },

  /** Drop everything except the active file. */
  clearKeepActive: async () => {
    await ebsdApi.clearLoadedFiles();
    await get().refresh();
  },

  /** Drop every entry, including the active file (full reset). */
  clearAll: async () => {
    await ebsdApi.clearAllLoadedFiles();
    await get().refresh();
  },
}));

export default useLoadedFilesStore;
