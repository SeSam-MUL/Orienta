import { create } from 'zustand';

/**
 * @typedef {'none' | 'header' | 'refined' | 'inherited' | 'manual'} PCStatus
 */

/**
 * @typedef {Object} BatchFileEntry
 * @property {string} file_path
 * @property {string} file_name
 * @property {boolean} loaded
 * @property {PCStatus} pc_status
 * @property {number[] | null} pc_value - [x, y, z]
 * @property {string | null} pc_source_file
 * @property {boolean} preprocessing_done
 * @property {Object | null} preprocessing_config
 * @property {boolean} eds_available
 * @property {string[]} eds_elements
 * @property {boolean} roi_configured
 * @property {boolean} phases_configured
 * @property {boolean} ready
 * @property {BatchJobConfig} config
 */

/**
 * @typedef {Object} BatchJobConfig
 * @property {'full' | 'roi' | 'chem'} selection_mode
 * @property {Object | null} region
 * @property {Object | null} chem_filters
 * @property {'and' | 'or'} chem_combine
 * @property {Array<{name: string, path: string, method: string}>} phases
 * @property {'hough' | 'spherical' | 'dictionary'} method
 * // Hough
 * @property {number} n_bands
 * @property {number} t_sigma
 * @property {number} r_sigma
 * // Dictionary
 * @property {'ncc' | 'zncc'} metric
 * @property {number} keep_n
 * @property {number} resolution
 * @property {number} energy_kv
 * // Spherical
 * @property {number} bandwidth
 * @property {number} nregions
 * @property {boolean} refine
 * // Preprocessing
 * @property {boolean} frame_averaging
 * @property {number} frame_averaging_window
 * @property {boolean} background_removal
 * @property {'dynamic' | 'static'} background_method
 * @property {number} static_bg_row
 * @property {number} static_bg_col
 * @property {boolean} gauss_background
 * @property {number} circular_mask
 * @property {number} nregions_ahe
 * // Export
 * @property {boolean} auto_export
 * @property {'h5' | 'ang' | 'ctf'} export_format
 * @property {string} export_dir
 * @property {boolean} include_eds_in_export
 * @property {boolean} store_n_best
 * @property {boolean} refine_after_indexing
 */

/** @returns {BatchJobConfig} */
const defaultConfig = () => ({
  selection_mode: 'full',
  region: null,
  chem_filters: null,
  chem_combine: 'and',
  phases: [],
  method: 'spherical',
  // Hough
  n_bands: 12,
  t_sigma: 2.0,
  r_sigma: 2.0,
  // Dictionary
  metric: 'ncc',
  keep_n: 20,
  resolution: 5.0,
  energy_kv: 20,
  // Spherical
  bandwidth: 88,
  nregions: 10,
  refine: true,
  // Preprocessing
  frame_averaging: false,
  frame_averaging_window: 3,
  background_removal: false,
  background_method: 'dynamic',
  static_bg_row: 0,
  static_bg_col: 0,
  gauss_background: true,
  circular_mask: 0,
  nregions_ahe: 10,
  // Export
  auto_export: true,
  // export_formats: subset of {'h5_rich', 'h5_light', 'ang', 'ctf'}.
  // Default is the "safe" set: light h5 (for in-app viewers) + .ang/.ctf
  // (for MTEX). 'h5_rich' duplicates the source file so it is opt-in.
  export_formats: ['h5_light', 'ang', 'ctf'],
  export_dir: '',
  include_eds_in_export: true,
  store_n_best: false,
  refine_after_indexing: false,

  // Post-processing filters applied to the winning phase assignment after
  // indexing completes. 0 disables. Defaults = off so existing batches don't
  // suddenly start rewriting their maps.
  postproc_ci_threshold: 0,
  postproc_uncertainty_threshold: 0,
  postproc_min_cluster_size: 0,
});

/**
 * Derive file_name from a full file path.
 * @param {string} filePath
 * @returns {string}
 */
const fileNameFromPath = (filePath) => {
  const normalized = filePath.replace(/\\/g, '/');
  return normalized.split('/').pop() || filePath;
};

/**
 * Create a default BatchFileEntry for a given path.
 * @param {string} filePath
 * @returns {BatchFileEntry}
 */
const makeFileEntry = (filePath) => ({
  file_path: filePath,
  file_name: fileNameFromPath(filePath),
  loaded: false,
  pc_status: 'none',
  pc_value: null,
  pc_source_file: null,
  preprocessing_done: false,
  preprocessing_config: null,
  eds_available: false,
  eds_elements: [],
  roi_configured: false,
  phases_configured: false,
  ready: false,
  config: defaultConfig(),
});

/**
 * Compute the ready flag for a single file entry.
 * A file is ready when its PC is known and at least one phase is configured.
 * @param {BatchFileEntry} entry
 * @returns {boolean}
 */
const isReady = (entry) =>
  entry.pc_status !== 'none' && entry.config.phases.length > 0;

const useBatchConfigStore = create((set, get) => ({
  /** @type {BatchFileEntry[]} */
  files: [],

  /** @type {number} Index of the currently selected file, or -1 when no file is selected. */
  activeFileIndex: -1,

  /** @type {number} Current workflow step (1–7). */
  currentStep: 1,

  /** @type {'hough' | 'spherical' | 'dictionary'} Global default indexing method. */
  method: 'spherical',

  /** @type {Array<{name: string, path: string, method: string}>} Default phases applied to all files. */
  globalPhases: [],

  /** @type {Object} Default preprocessing config applied to all files. */
  globalPreprocessing: {},

  // ---------------------------------------------------------------------------
  // File management
  // ---------------------------------------------------------------------------

  /**
   * Add one or more files by their full paths.
   * Duplicate paths (already present in the store) are silently ignored.
   * @param {string[]} filePaths
   */
  addFiles: (filePaths) => {
    set((state) => {
      const existingPaths = new Set(state.files.map((f) => f.file_path));
      const newEntries = filePaths
        .filter((p) => !existingPaths.has(p))
        .map(makeFileEntry);

      if (newEntries.length === 0) return state;

      const files = [...state.files, ...newEntries];
      const activeFileIndex =
        state.activeFileIndex === -1 ? 0 : state.activeFileIndex;

      return { files, activeFileIndex };
    });
  },

  /**
   * Remove the file at the given index.
   * Adjusts activeFileIndex when the removed file was selected or before the cursor.
   * @param {number} index
   */
  removeFile: (index) => {
    set((state) => {
      if (index < 0 || index >= state.files.length) return state;

      const files = state.files.filter((_, i) => i !== index);

      let activeFileIndex = state.activeFileIndex;
      if (files.length === 0) {
        activeFileIndex = -1;
      } else if (activeFileIndex >= files.length) {
        activeFileIndex = files.length - 1;
      } else if (activeFileIndex > index) {
        activeFileIndex -= 1;
      }

      return { files, activeFileIndex };
    });
  },

  /** Remove all files and reset selection. */
  clearFiles: () => set({ files: [], activeFileIndex: -1 }),

  // ---------------------------------------------------------------------------
  // Navigation
  // ---------------------------------------------------------------------------

  /**
   * Set the active (selected) file.
   * @param {number} index
   */
  setActiveFile: (index) => {
    set((state) => {
      if (index < -1 || index >= state.files.length) return state;
      return { activeFileIndex: index };
    });
  },

  /**
   * Navigate to a workflow step (1–7).
   * @param {number} step
   */
  setStep: (step) => {
    if (step >= 1 && step <= 7) {
      set({ currentStep: step });
    }
  },

  // ---------------------------------------------------------------------------
  // Method
  // ---------------------------------------------------------------------------

  /**
   * Set the global indexing method and propagate it to every file's config.
   * @param {'hough' | 'spherical' | 'dictionary'} method
   */
  setMethod: (method) => {
    set((state) => ({
      method,
      files: state.files.map((f) => ({
        ...f,
        config: { ...f.config, method },
      })),
    }));
  },

  // ---------------------------------------------------------------------------
  // Per-file config updates
  // ---------------------------------------------------------------------------

  /**
   * Merge a partial config object into a single file's config.
   * Recomputes the ready flag afterwards.
   * @param {number} index
   * @param {Partial<BatchJobConfig>} partialConfig
   */
  updateFileConfig: (index, partialConfig) => {
    set((state) => {
      if (index < 0 || index >= state.files.length) return state;

      const files = state.files.map((f, i) => {
        if (i !== index) return f;
        const config = { ...f.config, ...partialConfig };
        const phases_configured = config.phases.length > 0;
        const updated = { ...f, config, phases_configured };
        return { ...updated, ready: isReady(updated) };
      });

      return { files };
    });
  },

  /**
   * Update the PC (pattern center) information for a single file.
   * @param {number} index
   * @param {{ pc_value?: number[] | null, pc_status?: PCStatus, pc_source_file?: string | null }} pcInfo
   */
  updateFilePCStatus: (index, { pc_value, pc_status, pc_source_file }) => {
    set((state) => {
      if (index < 0 || index >= state.files.length) return state;

      const files = state.files.map((f, i) => {
        if (i !== index) return f;
        const updated = {
          ...f,
          pc_value: pc_value !== undefined ? pc_value : f.pc_value,
          pc_status: pc_status !== undefined ? pc_status : f.pc_status,
          pc_source_file:
            pc_source_file !== undefined ? pc_source_file : f.pc_source_file,
        };
        return { ...updated, ready: isReady(updated) };
      });

      return { files };
    });
  },

  /**
   * Update file metadata that becomes available after a quick-load scan
   * (e.g. whether EDS data exists, grid dimensions, pattern shape).
   * @param {number} index
   * @param {{ loaded?: boolean, eds_available?: boolean, eds_elements?: string[], grid_shape?: number[], pattern_shape?: number[] }} metadata
   */
  updateFileMetadata: (
    index,
    { loaded, eds_available, eds_elements, grid_shape, pattern_shape, size_mb }
  ) => {
    set((state) => {
      if (index < 0 || index >= state.files.length) return state;

      const files = state.files.map((f, i) => {
        if (i !== index) return f;
        const updated = {
          ...f,
          loaded: loaded !== undefined ? loaded : f.loaded,
          eds_available:
            eds_available !== undefined ? eds_available : f.eds_available,
          eds_elements:
            eds_elements !== undefined ? eds_elements : f.eds_elements,
          grid_shape: grid_shape !== undefined ? grid_shape : f.grid_shape,
          pattern_shape:
            pattern_shape !== undefined ? pattern_shape : f.pattern_shape,
          size_mb: size_mb !== undefined ? size_mb : f.size_mb,
        };
        return { ...updated, ready: isReady(updated) };
      });

      return { files };
    });
  },

  // ---------------------------------------------------------------------------
  // Batch config propagation
  // ---------------------------------------------------------------------------

  /**
   * Copy selected parts of a source file's config to one or more target files.
   * @param {number} sourceIndex - index of the file to copy from
   * @param {number[]} targetIndices - indices of files to copy to
   * @param {'pc' | 'preprocessing' | 'phases' | 'roi' | 'all'} what - which part to copy
   */
  applyConfigToSelected: (sourceIndex, targetIndices, what) => {
    set((state) => {
      if (sourceIndex < 0 || sourceIndex >= state.files.length) return state;

      const source = state.files[sourceIndex];
      const targetSet = new Set(targetIndices);

      const files = state.files.map((f, i) => {
        if (i === sourceIndex || !targetSet.has(i)) return f;

        let updated = { ...f };

        if (what === 'pc' || what === 'all') {
          updated = {
            ...updated,
            pc_value: source.pc_value,
            pc_status: source.pc_status === 'refined' ? 'inherited' : source.pc_status,
            pc_source_file: source.file_path,
          };
        }

        if (what === 'preprocessing' || what === 'all') {
          const preprocessingKeys = [
            'frame_averaging',
            'frame_averaging_window',
            'background_removal',
            'background_method',
            'static_bg_row',
            'static_bg_col',
            'gauss_background',
            'circular_mask',
            'nregions_ahe',
          ];
          const preprocessingPatch = Object.fromEntries(
            preprocessingKeys.map((k) => [k, source.config[k]])
          );
          updated = {
            ...updated,
            preprocessing_config: { ...preprocessingPatch },
            preprocessing_done: source.preprocessing_done,
            config: { ...updated.config, ...preprocessingPatch },
          };
        }

        if (what === 'phases' || what === 'all') {
          const phases = [...source.config.phases];
          updated = {
            ...updated,
            phases_configured: phases.length > 0,
            config: { ...updated.config, phases },
          };
        }

        if (what === 'roi' || what === 'all') {
          updated = {
            ...updated,
            roi_configured: source.roi_configured,
            config: {
              ...updated.config,
              selection_mode: source.config.selection_mode,
              region: source.config.region,
              chem_filters: source.config.chem_filters,
              chem_combine: source.config.chem_combine,
            },
          };
        }

        return { ...updated, ready: isReady(updated) };
      });

      return { files };
    });
  },

  /**
   * Overwrite the phases list on every file with the provided phases array.
   * Also updates globalPhases.
   * @param {Array<{name: string, path: string, method: string}>} phases
   */
  applyGlobalPhases: (phases) => {
    set((state) => ({
      globalPhases: phases,
      files: state.files.map((f) => {
        const phases_configured = phases.length > 0;
        const config = { ...f.config, phases: [...phases] };
        const updated = { ...f, config, phases_configured };
        return { ...updated, ready: isReady(updated) };
      }),
    }));
  },

  /**
   * Apply a preprocessing config patch to every file.
   * Also updates globalPreprocessing.
   * @param {Partial<BatchJobConfig>} config - preprocessing fields to apply
   */
  applyGlobalPreprocessing: (config) => {
    const preprocessingKeys = [
      'frame_averaging',
      'frame_averaging_window',
      'background_removal',
      'background_method',
      'static_bg_row',
      'static_bg_col',
      'gauss_background',
      'circular_mask',
      'nregions_ahe',
    ];

    const patch = Object.fromEntries(
      Object.entries(config).filter(([k]) => preprocessingKeys.includes(k))
    );

    set((state) => ({
      globalPreprocessing: { ...state.globalPreprocessing, ...patch },
      files: state.files.map((f) => ({
        ...f,
        preprocessing_config: { ...patch },
        config: { ...f.config, ...patch },
      })),
    }));
  },

  /**
   * Update globalPhases without propagating to individual files.
   * Use applyGlobalPhases when you want to push to all files.
   * @param {Array<{name: string, path: string, method: string}>} phases
   */
  setGlobalPhases: (phases) => set({ globalPhases: phases }),

  // ---------------------------------------------------------------------------
  // Derived queries
  // ---------------------------------------------------------------------------

  /**
   * Return only the files whose ready flag is true.
   * @returns {BatchFileEntry[]}
   */
  getReadyFiles: () => get().files.filter((f) => f.ready),

  /**
   * Recompute the ready flag for every file based on current state.
   * Call this after any bulk operation that may have changed readiness conditions.
   */
  computeReadiness: () => {
    set((state) => ({
      files: state.files.map((f) => ({ ...f, ready: isReady(f) })),
    }));
  },
}));

export default useBatchConfigStore;
