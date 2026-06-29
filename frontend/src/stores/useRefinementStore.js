import { create } from 'zustand';
import { phaseRefinementApi } from '../services/api';

const useRefinementStore = create((set, get) => ({
  fileStem: null,
  searchDir: null,
  summary: null,
  selectedPixel: null,
  overrideHistory: [],
  redoStack: [],
  mode: 'click',       // per-pixel override (the only implemented mode; UI exposes it as "Pixel")
  ciThreshold: 0.3,

  loadFile: async (fileStem, searchDir) => {
    const res = await phaseRefinementApi.summary(fileStem, searchDir);
    set({ fileStem, searchDir, summary: res.data, selectedPixel: null });
  },

  selectPixel: async (row, col) => {
    const { fileStem, searchDir } = get();
    const res = await phaseRefinementApi.pixel(fileStem, row, col, searchDir);
    set({ selectedPixel: { row, col, phases: res.data.phases } });
  },

  overridePixels: async (pixels, source) => {
    const { fileStem, searchDir } = get();
    await phaseRefinementApi.override(fileStem, pixels, source, searchDir);
    const res = await phaseRefinementApi.summary(fileStem, searchDir);
    set((state) => ({
      summary: res.data,
      overrideHistory: [...state.overrideHistory, { pixels, source }],
      redoStack: [],
    }));
  },

  setMode: (mode) => set({ mode }),
  setCiThreshold: (t) => set({ ciThreshold: t }),
  reset: () =>
    set({
      fileStem: null,
      summary: null,
      selectedPixel: null,
      overrideHistory: [],
      redoStack: [],
    }),
}));

export default useRefinementStore;
