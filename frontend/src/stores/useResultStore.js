/**
 * Results store - tracks indexing results, phase maps, grain data.
 */

import { create } from 'zustand';

const useResultStore = create((set, get) => ({
  // --- Indexing ---
  indexingResult: null,
  indexingMethod: null,
  resultsList: [],  // [{id, label, method, phases, mean_ci, n_indexed, timestamp}]

  // --- Phase Map ---
  phaseMapImage: null,
  ipfDirection: 'Z',

  // --- Analysis ---
  analysisLoaded: false,
  analysisShape: null,
  hasGrains: false,
  currentMapType: 'bc',
  currentMapImage: null,

  // --- Actions ---
  setIndexingResult: (result, method) => {
    // Backend returns phases as [{id, name}, ...] — .join(', ') on objects
    // produces "[object Object], ..." so extract names first.
    const phaseNames = Array.isArray(result.phases)
      ? result.phases
          .map((p) => (typeof p === 'string' ? p : p?.name))
          .filter(Boolean)
      : [];
    const entry = {
      id: result.result_id || `${method}_${Date.now()}`,
      label: [
        method,
        phaseNames.length > 0 ? phaseNames.join(', ') : null,
        result.mean_ci != null ? `CI: ${result.mean_ci.toFixed(3)}` : null,
        `${result.n_indexed ?? '?'} px`,
      ].filter(Boolean).join(' — '),
      method,
      phases: result.phases || [],
      mean_ci: result.mean_ci ?? null,
      n_indexed: result.n_indexed ?? 0,
      timestamp: new Date().toLocaleTimeString(),
      result_id: result.result_id || null,
    };
    set((state) => ({
      indexingResult: result,
      indexingMethod: method,
      resultsList: [...state.resultsList, entry],
    }));
  },

  setPhaseMap: (image, direction) => set({
    phaseMapImage: image,
    ipfDirection: direction,
  }),

  setAnalysis: (data) => set({
    analysisLoaded: true,
    analysisShape: data.shape,
    hasGrains: data.has_grains || false,
  }),

  setCurrentMap: (type, image) => set({
    currentMapType: type,
    currentMapImage: image,
  }),

  clearResults: () => set({
    indexingResult: null,
    indexingMethod: null,
    resultsList: [],
    phaseMapImage: null,
    analysisLoaded: false,
    analysisShape: null,
    hasGrains: false,
    currentMapType: 'bc',
    currentMapImage: null,
  }),
}));

export default useResultStore;
