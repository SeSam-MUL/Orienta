import { create } from 'zustand';
import { frameApi } from '../services/api';

const LS_PREFIX = 'frameSpec::';

function lsKey(sourceFile) { return `${LS_PREFIX}${sourceFile || ''}`; }

const useFrameStore = create((set, get) => ({
  sourceFile: null,
  spec: null,
  frameSig: null,
  loaded: false,

  applyFromBackend: ({ source_file, spec, frame_sig }) =>
    set({ sourceFile: source_file, spec, frameSig: frame_sig, loaded: true }),

  // Pull the active file's frame from the backend. If the backend has only the
  // default but localStorage has a saved spec for this file, re-push it so the
  // user's setting survives a backend restart.
  loadForFile: async (sourceFile) => {
    try {
      const { data } = await frameApi.get();
      set({ sourceFile: data.source_file, spec: data.spec, frameSig: data.frame_sig, loaded: true });
      const saved = localStorage.getItem(lsKey(data.source_file));
      const isDefaultPreset =
        data.spec?.rotation?.mode === 'preset' && data.spec?.rotation?.preset === 'identity'
        && data.spec?.apply_to_export === false;
      if (saved && isDefaultPreset) {
        try { await get().setSpec(JSON.parse(saved)); } catch { /* ignore */ }
      }
    } catch {
      set({ loaded: true });
    }
  },

  setSpec: async (spec) => {
    const { data } = await frameApi.set(spec);
    set({ sourceFile: data.source_file, spec: data.spec, frameSig: data.frame_sig, loaded: true });
    try { localStorage.setItem(lsKey(data.source_file), JSON.stringify(data.spec)); } catch { /* quota */ }
  },
}));

// Cross-window sync: a sibling window persisting a spec fires a `storage` event.
// Re-pull from the backend (the source of truth) so this window's IPF/PF update.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (e) => {
    if (e.key && e.key.startsWith(LS_PREFIX)) {
      const { loadForFile, sourceFile } = useFrameStore.getState();
      loadForFile(sourceFile);
    }
  });
}

export default useFrameStore;
