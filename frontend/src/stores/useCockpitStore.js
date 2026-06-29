/**
 * useCockpitStore — UI state for the H5OINA cockpit.
 *
 * Holds: active map layer, layer catalog, panel collapse state, splitter sizes,
 * selection mask. Per-pixel data (pattern, spectrum, aztec) is fetched via hooks
 * and lives on those hooks — this store is purely UI state.
 */
import { create } from 'zustand';

const SPLITTER_KEY = 'h5cockpit:splitters';
const PANEL_KEY = 'h5cockpit:panels';

const loadSplitters = () => {
  try {
    return JSON.parse(localStorage.getItem(SPLITTER_KEY)) ?? null;
  } catch { return null; }
};
const loadPanels = () => {
  try {
    return JSON.parse(localStorage.getItem(PANEL_KEY)) ?? null;
  } catch { return null; }
};

const DEFAULT_SPLITTERS = { rail: 160, inspector: 420 };
const DEFAULT_PANELS = { pattern: true, spectrum: true, counts: true, aztec: true };

const useCockpitStore = create((set, get) => ({
  activeLayerId: null,
  layerCatalog: null,
  selection: { mask: null, kind: null },  // mask: Uint8Array, kind: 'lasso'|'rect'
  splitters: { ...DEFAULT_SPLITTERS, ...(loadSplitters() ?? {}) },
  panels: { ...DEFAULT_PANELS, ...(loadPanels() ?? {}) },

  setActiveLayer: (id) => set({ activeLayerId: id }),
  setLayerCatalog: (catalog) => set({ layerCatalog: catalog }),
  setSelection: (mask, kind) => set({ selection: { mask, kind } }),
  clearSelection: () => set({ selection: { mask: null, kind: null } }),

  setSplitter: (name, size) => set((s) => {
    const next = { ...s.splitters, [name]: size };
    localStorage.setItem(SPLITTER_KEY, JSON.stringify(next));
    return { splitters: next };
  }),
  togglePanel: (name) => set((s) => {
    const next = { ...s.panels, [name]: !s.panels[name] };
    localStorage.setItem(PANEL_KEY, JSON.stringify(next));
    return { panels: next };
  }),

  resetUI: () => {
    localStorage.removeItem(SPLITTER_KEY);
    localStorage.removeItem(PANEL_KEY);
    set({ splitters: DEFAULT_SPLITTERS, panels: DEFAULT_PANELS });
  },
}));

export default useCockpitStore;
