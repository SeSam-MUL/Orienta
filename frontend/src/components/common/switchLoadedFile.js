// Opens a file that is already loaded: the one sequence of the header's file switcher (FileSwitcher.jsx) and of every
// other place that switches to a loaded file (the PC page's "Find PC for it"). The backend re-loads the file, the
// pairing notice is shown, the shared data store is refreshed, the registry is read again.
import { ebsdApi, isSupersededLoad } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import useLoadedFilesStore from '../../stores/useLoadedFilesStore';
import { toast } from '../../stores/useToastStore';
import { announcePairingDropped } from '../EBSDViewer/apexLoad';

/**
 * @param {string} path the loaded file to open
 * @param {(key: string, options?: object) => string} tt translates a key of the `ebsdviewer` namespace
 * @returns {Promise<{opened: boolean, filePath: (string|null)}>} `opened` is false when a load that started after
 *   this one and succeeded decides which file the backend ends on; `filePath` is the path the switch answer names.
 *   Rejects with the request's error.
 */
export default async function switchLoadedFile(path, tt) {
  const res = await ebsdApi.switchFile(path);
  // Another load or switch (the viewer's file list, the Indexing dropdown) started after this
  // one and succeeded: its answer describes the file the backend ends on, this one is the older.
  // (Waits for a later one that is still pending; a later one that fails supersedes nothing.)
  const superseded = await isSupersededLoad(res);
  // A measurement of an APEX file whose metadata file cannot be used any more is reloaded without it.
  announcePairingDropped(res?.data, tt, { warn: (m) => toast.warning(m, 15000) });
  // `ebsdInfo` is the viewer's copy of the load answer: its grid and scan geometry are what the
  // overview clicks and the crop windows map through, and nothing but a load or a switch
  // rewrites it. The switch answer describes the dataset now active, so it replaces the
  // previous file's wherever the switch was made (this header sits on three pages).
  // Only where a copy exists: setEBSDLoaded also marks the viewer as loaded, and after a
  // renderer reload nothing but the viewer's own first visit may do that (it hydrates the
  // dataset list, overview, pattern and file path only while ebsdLoaded is false).
  if (!superseded && res?.data && typeof res.data === 'object' && useDataStore.getState().ebsdInfo) {
    useDataStore.getState().setEBSDLoaded?.(res.data);
  }
  await useDataStore.getState().syncFromBackend();   // refresh global store → all pages re-read
  await useLoadedFilesStore.getState().refresh();    // shared registry → all switcher views re-read
  const named = res?.data && typeof res.data === 'object' ? res.data.file_path : null;
  return { opened: !superseded, filePath: typeof named === 'string' && named ? named : null };
}
