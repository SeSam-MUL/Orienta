/**
 * Global data store using Zustand
 *
 * Replaces PyQt5 signals for inter-module state management.
 * Tracks: current file, position, EDS data, loaded results.
 */

import { create } from 'zustand';
import { ebsdApi, isFileLoadInFlight } from '../services/api';
import { parseScanGeometry } from '../services/scanGeometry';

const useDataStore = create((set, get) => ({
  // --- File State ---
  isFileOpen: false,
  filePath: null,
  formatType: null,
  gridShape: [0, 0],
  // The scan geometry (square / hexagonal, steps, row offset) of the dataset `gridShape` is the
  // grid of. Written by EVERY writer of gridShape in the same set() call, so the two cannot
  // disagree; null = the backend did not say (not "square").
  scanGeometry: null,
  // True when the square map in `gridShape` was resampled from a hexagonal scan (an OIM-form
  // hexagonal file is held as a square map, so `scanGeometry` is honestly square and cannot say
  // it). Written next to the geometry by the same writers; null = the backend did not say.
  hexResampled: null,
  patternCount: 0,
  patternShape: [0, 0],

  // --- Features ---
  hasPatterns: false,
  hasRawPatterns: false,
  hasEDS: false,
  hasElectronImages: false,
  edsElements: [],
  electronImages: [],

  // --- Navigation ---
  currentRow: 0,
  currentCol: 0,
  currentIndex: 0,
  currentPattern: null, // Base64 string

  // --- EBSD Data ---
  ebsdLoaded: false,
  ebsdInfo: null,

  // --- Cross-page actions ---
  pendingChemMask: false,  // When true, Indexing page should auto-select chemistry mask mode
  pendingSimXtal: null,    // When set, Simulation page should pre-fill crystalFile with this path
  pendingPhaseMapIndexing: false,  // When true, Indexing page should pull /phase-map and offer per-phase pixel masks

  // --- Forward-diagnostics state ---
  diagnosticsComputed: {},                  // { [result_id]: { bandwidth, computed_at } }
  diagnosticsJob: null,                     // { result_id, job_id, progress, state, current_phase } | null

  // --- Refinement state ---
  refinementComputed: {},                   // { [result_id]: { lambda, computed_at, refined_result_id, summary?, stage_timings? } }
  refinementJob: null,                      // { result_id, job_id, state, current_stage, progress, outer_iter } | null
  refinementLambdaPending: 0.01,            // λ pending in the slider (log-scale UI)

  // --- Metadata (central, from /api/ebsd/metadata) ---
  metadata: null,       // Full metadata object from backend
  detector: null,       // { has_detector, shape, pc, sample_tilt, camera_tilt, ... }
  stepSize: null,       // { x, y, units }
  // Per-acquisition-area pixel size: { ebsd|eds|electron_image: {x,y,units} }.
  // The electron images sit on the SEM raster, the maps on the scan raster —
  // measured 10.6x apart — so a scale bar must pick the right one.
  pixelSizes: null,
  beamEnergy: null,     // kV or null
  axesRepr: '',         // Human-readable axes manager string
  phases: [],           // Phase names from loaded data

  // --- Actions ---
  setFileData: (data) => set({
    isFileOpen: true,
    filePath: data.file_path,
    formatType: data.format_type,
    gridShape: data.grid_shape,
    scanGeometry: parseScanGeometry(data.scan_geometry),
    hexResampled: data.hex_resampled ?? null,
    patternCount: data.pattern_count,
    patternShape: data.pattern_shape,
    hasPatterns: data.has_patterns,
    hasRawPatterns: data.has_raw_patterns,
    hasEDS: data.has_eds,
    hasElectronImages: data.has_electron_images,
    edsElements: data.eds_elements || [],
    electronImages: data.electron_images || [],
  }),

  clearFile: () => set({
    isFileOpen: false,
    filePath: null,
    formatType: null,
    gridShape: [0, 0],
    scanGeometry: null,
    hexResampled: null,
    patternCount: 0,
    patternShape: [0, 0],
    hasPatterns: false,
    hasRawPatterns: false,
    hasEDS: false,
    hasElectronImages: false,
    edsElements: [],
    electronImages: [],
    currentRow: 0,
    currentCol: 0,
    currentIndex: 0,
    currentPattern: null,
    metadata: null,
    detector: null,
    stepSize: null,
    pixelSizes: null,
    beamEnergy: null,
    axesRepr: '',
    phases: [],
  }),

  setPosition: (row, col, index, pattern) => set({
    currentRow: row,
    currentCol: col,
    currentIndex: index,
    currentPattern: pattern,
  }),

  setEBSDLoaded: (info) => set({
    ebsdLoaded: true,
    ebsdInfo: info,
  }),

  // Counts the times the backend's dataset changed without the viewer asking (a result was activated
  // on another page and brought up its file or variant). The viewer re-reads its own state on it.
  fileChangedElsewhere: 0,
  noteFileChangedElsewhere: () => set((s) => ({ fileChangedElsewhere: s.fileChangedElsewhere + 1 })),

  // The navigation grid alone, for when it changes WITHOUT the file changing.
  // Cropping is the first thing that does that: a deepcopy kept its parent's
  // grid, so `gridShape` used to be written once by setFileData and never
  // again. Deliberately narrower than setFileData — it must not touch
  // filePath, the EDS/electron-image flags or anything else about the file,
  // and staying narrow means a field added to setFileData later cannot be
  // silently cleared here.
  //
  // The geometry goes with the grid and is NOT kept when the caller has none: a crop of a
  // hexagonal scan can start on an odd row and so has the opposite offset, and the parent's
  // geometry left standing next to the crop's grid is exactly the stale copy to avoid.
  // The resampled flag is the dataset's own, too (a crop of a resampled map is resampled).
  setNavigationGrid: ({ gridShape, patternShape, patternCount, scanGeometry, hexResampled }) => set((st) => ({
    gridShape,
    scanGeometry: scanGeometry ?? null,
    hexResampled: hexResampled ?? null,
    patternCount,
    patternShape: patternShape || st.patternShape,
  })),

  setPendingChemMask: (val) => set({ pendingChemMask: val }),
  setPendingSimXtal: (path) => set({ pendingSimXtal: path }),
  setPendingPhaseMapIndexing: (val) => set({ pendingPhaseMapIndexing: val }),

  setDiagnosticsComputed: (result_id, info) =>
    set((s) => ({
      diagnosticsComputed: { ...s.diagnosticsComputed, [result_id]: info },
    })),

  clearDiagnosticsComputed: (result_id) =>
    set((s) => {
      const c = { ...s.diagnosticsComputed };
      delete c[result_id];
      return { diagnosticsComputed: c };
    }),

  setDiagnosticsJob: (job) => set({ diagnosticsJob: job }),

  setRefinementComputed: (result_id, info) =>
    set((s) => ({
      refinementComputed: { ...s.refinementComputed, [result_id]: info },
    })),

  clearRefinementComputed: (result_id) =>
    set((s) => {
      const c = { ...s.refinementComputed };
      delete c[result_id];
      return { refinementComputed: c };
    }),

  setRefinementJob: (job) => set({ refinementJob: job }),
  setRefinementLambda: (v) => set({ refinementLambdaPending: v }),

  setMetadata: (meta) => set({
    metadata: meta,
    detector: meta.detector || null,
    stepSize: meta.step_size || null,
    pixelSizes: meta.pixel_sizes || null,
    beamEnergy: meta.beam_energy || null,
    axesRepr: meta.axes_repr || '',
    phases: meta.phases || [],
  }),

  /**
   * Sync the store from whatever the backend currently has loaded.
   *
   * Useful when the file was loaded by a path other than the React
   * "Open File" button (direct API call, test-env script, Electron open
   * dialog, ...) — without this the StatusBar and other store-driven
   * widgets stay stale while page-local state (IndexingPage, EDSPage,
   * PCRefinement) reads the backend directly and appears correct.
   *
   * Called from App.jsx on mount and again when the backend reconnects
   * after a health-check failure. Safe to call when no file is loaded —
   * the /info, /detector, /metadata endpoints now all return 200 empty
   * in that case (BUG-A fix).
   *
   * `force` re-reads also when the path is the one in the store: another variant of the open file
   * has the same path, and another measurement can have another grid.
   *
   * Returns true if a file was synced, false if backend has nothing.
   */
  syncFromBackend: async ({ force = false } = {}) => {
    // A load or switch in flight is about to set this state itself; reading the
    // backend half-way through would race it.
    if (isFileLoadInFlight()) return false;
    try {
      // A FAILED info request is not an answer: it rejects into the catch below
      // and leaves the store as it is. Only a reply that says loaded:false may
      // clear an open file.
      const [infoRes, metaRes] = await Promise.all([
        ebsdApi.info(),
        ebsdApi.getMetadata().catch(() => ({ data: {} })),
      ]);
      const info = infoRes.data || {};
      const meta = metaRes.data || {};

      if (!info.loaded) {
        // Backend has no file → ensure store is empty. An EDS-only file is open without a signal.
        if (get().isFileOpen && !info.open_file) get().clearFile();
        return false;
      }

      // Only update if something actually differs — avoids a render loop
      const current = get();
      if (!force && current.isFileOpen && current.filePath === info.file_path) {
        return true; // already in sync
      }

      // Populate from the available endpoints (info has core shape;
      // metadata has detector / step / phases; the ebsd/load endpoint
      // returns a richer object but is not safe to call again here).
      set({
        isFileOpen: true,
        filePath: info.file_path,
        formatType: meta.format_type || null,
        gridShape: (info.data_shape && info.data_shape.length >= 2)
          ? [info.data_shape[0], info.data_shape[1]] : [0, 0],
        patternCount: (info.data_shape && info.data_shape.length >= 2)
          ? info.data_shape[0] * info.data_shape[1] : 0,
        patternShape: (info.data_shape && info.data_shape.length >= 4)
          ? [info.data_shape[2], info.data_shape[3]] : [0, 0],
        scanGeometry: parseScanGeometry(info.scan_geometry),
        hexResampled: info.hex_resampled ?? null,
        hasPatterns: true,
        hasRawPatterns: info.has_raw_patterns === true,
        hasEDS: Array.isArray(meta.eds_elements) && meta.eds_elements.length > 0,
        // /metadata does not list them: a forced re-read of the open file keeps what it has
        hasElectronImages: force ? current.hasElectronImages
          : Array.isArray(meta.electron_images) && meta.electron_images.length > 0,
        edsElements: meta.eds_elements || [],
        electronImages: force ? current.electronImages : (meta.electron_images || []),
      });

      if (meta.detector || meta.step_size || meta.phases) {
        get().setMetadata(meta);
      }
      return true;
    } catch {
      return false;
    }
  },
}));

export default useDataStore;
