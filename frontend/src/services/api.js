/**
 * API Client for Orienta Backend
 *
 * Every FastAPI endpoint gets a corresponding function here.
 * Base URL is configurable for dev (localhost:8000) vs Electron (bundled).
 */

import axios from 'axios';

// In dev mode with Vite proxy, use relative URLs. In Electron/production, use full URL.
const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000, // 5 min for long operations
});

// --- Health ---
export const healthCheck = () => api.get('/api/health');

// --- System ---
/** Fetch GPU detection / VRAM info from /api/system/gpu. */
export async function getGpuStatus() {
  const r = await api.get('/api/system/gpu');
  return r.data;
}

// --- HDF5 Viewer ---
export const h5Api = {
  open: (path) => api.post('/api/h5/open', { path }),
  close: () => api.post('/api/h5/close'),
  forceClose: () => api.post('/api/h5/force-close'),
  status: () => api.get('/api/h5/status'),
  getPattern: (index, type = 'processed') =>
    api.get(`/api/h5/pattern/${index}`, { params: { pattern_type: type } }),
  // Binary PNG bytes for hot navigation paths — ~33% smaller than the base64
  // route plus skips the JSON parse. Returns a Blob instead of parsed JSON.
  getPatternBinary: (index) =>
    api.get(`/api/h5/pattern/${index}/binary`, { responseType: 'blob' }),
  getPatternByPos: (row, col) => api.get(`/api/h5/pattern/pos/${row}/${col}`),
  navigate: (row, col) => api.post('/api/h5/navigate', { row, col }),
  getEDSElements: (scope = 'file') =>
    api.get('/api/h5/eds/elements', { params: { scope } }),
  getEDSMap: (element, cmap = 'hot', color = '', scope = 'file') =>
    api.get(`/api/h5/eds/map/${element}`, { params: { cmap, color, scope } }),
  getEDSPixel: (row, col) => api.get(`/api/h5/eds/pixel/${row}/${col}`),
  // `scope` picks WHICH view of the electron image you get. Omit it (the
  // default, 'file') for the H5 cockpit: the image exactly as the file holds
  // it. Pass 'dataset' from the EDS / Phase Map layer stacks: under a crop the
  // image comes back cut to the same physical region as every other layer, so
  // an SE image no longer shows the whole sample under a cropped EDS map.
  getElectronList: (scope) =>
    api.get('/api/h5/electron/list', { params: scope ? { scope } : undefined }),
  getElectronImage: (name, scope) =>
    api.get(`/api/h5/electron/${name}`, { params: scope ? { scope } : undefined }),
  getMinimap: () => api.get('/api/h5/minimap'),
  getTree: (maxDepth = 5) => api.get('/api/h5/tree', { params: { max_depth: maxDepth } }),
  getTreeNode: (path) => api.get('/api/h5/tree/node', { params: { path } }),
  getAttributes: (path = '/') => api.get('/api/h5/attributes', { params: { path } }),
  // Backend-side scans (Task 4): replace per-pattern fetch loops with one call.
  scanQuality: (samples = 50, bins = 20) =>
    api.get('/api/h5/scan/quality', { params: { samples, bins } }),
  scanDefects: (samples = 60) =>
    api.get('/api/h5/scan/defects', { params: { samples } }),
  // --- Cockpit additions (2026-05-06) ---
  getScalarMap: (path, cmap = 'gray') =>
    api.get('/api/h5/scalar-map', { params: { path, cmap } }),
  getPhaseMap: () => api.get('/api/h5/phase-map'),
  getIPFMap: (direction = 'Z') =>
    api.get('/api/h5/ipf-map', { params: { direction } }),
  getEDSSpectrum: (row, col) => api.get(`/api/h5/eds/spectrum/${row}/${col}`),
  getEDSHeader: () => api.get('/api/h5/eds/header'),
  getEBSDHeader: () => api.get('/api/h5/ebsd/header'),
  getBackground: (type = 'processed') =>
    api.get('/api/h5/background', { params: { type } }),
  getAztecPixel: (row, col) => api.get(`/api/h5/aztec/pixel/${row}/${col}`),
  getLayers: () => api.get('/api/h5/layers'),
};

// --- EBSD Viewer ---
export const ebsdApi = {
  load: (path, useKikuchipy = true) =>
    api.post('/api/ebsd/load', { path, use_kikuchipy: useKikuchipy }),

  /**
   * Load an EBSD file with stage-based progress reporting.
   *
   * @param {string} path - Absolute file path
   * @param {object} options
   * @param {(state: object) => void} [options.onProgress] - called with each polled
   *   snapshot (fields: stage, stage_idx, stage_total, elapsed_seconds, message;
   *   plus stage='error' on the error path)
   * @param {number} [options.pollIntervalMs=250] - polling cadence in ms
   * @param {boolean} [options.useKikuchipy=true] - passes through to safe_loader
   * @returns {Promise} resolves with the POST /api/ebsd/load axios response on success
   */
  loadWithProgress: (
    path,
    { onProgress, pollIntervalMs = 250, useKikuchipy = true } = {},
  ) => {
    // crypto.randomUUID() is available in all modern browsers + Node 14.17+.
    // It returns an RFC4122 v4 UUID. Fallback for paranoia / very old envs.
    const requestId =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : 'fallback-' +
          Math.random().toString(16).slice(2) +
          '-' +
          Date.now().toString(16);

    let pollTimer = null;
    let stopped = false;
    // Backend-down detection via wall clock: track the last time we got any
    // HTTP reply from the backend (200 OR 404 — both prove the server is up
    // and the event loop responded). If we go more than STALE_THRESHOLD_MS
    // without any reply, the backend is unreachable and we fire an error.
    //
    // Why wall-clock instead of consecutive-count: during a heavy load the
    // backend event loop can be GIL-blocked for several seconds at a time
    // (cold-start imports, dask graph construction in to_thread workers).
    // A consecutive-count detector + short per-request timeout fired a
    // false positive on this exact path (session 2026-05-27) — backend
    // was alive but couldn't reply to progress polls within 1.5 s during
    // a cold start. The wall-clock approach tolerates that, while still
    // catching a genuinely dead backend within 10 s.
    //
    // Note: 404 from the progress endpoint is normal at the very start
    // (load hasn't written its first stage yet) AND proves the backend is
    // up and routing requests, so it counts as a successful contact.
    const STALE_THRESHOLD_MS = 10000;
    // Initialise to "just contacted" so the threshold starts ticking from
    // the moment loadWithProgress is invoked, not from epoch 0.
    let lastBackendContactAt = Date.now();
    // Last elapsed the BACKEND reported. Used so a failure can say how long
    // the load actually ran instead of the hardcoded 0 it used to report
    // ("Load failed … 0.0 s elapsed" while the load had been running for
    // minutes, which made a slow load look like an instant failure).
    let lastElapsedSeconds = 0;
    // Set when the stale detector has already produced a good error message,
    // so aborting the POST below doesn't overwrite it with "canceled".
    let staleErrorEmitted = false;
    // Lets the stale detector cancel the in-flight POST. Without it, dropping
    // the POST's own timeout (see below) would leave the request hanging
    // forever against a backend that has genuinely died.
    const abortController = new AbortController();

    const poll = async () => {
      if (stopped) return;
      try {
        // Per-request timeout (3000 ms) — overrides the 5-min instance
        // default so a stalled TCP socket can't hang a poll forever.
        // The actual backend-down decision is made by the wall-clock
        // STALE_THRESHOLD_MS check below; this timeout just bounds how
        // long each individual await can sit before we move on.
        const r = await api.get(`/api/ebsd/load/progress/${requestId}`, {
          timeout: 3000,
        });
        lastBackendContactAt = Date.now();
        if (typeof r.data?.elapsed_seconds === 'number') {
          lastElapsedSeconds = r.data.elapsed_seconds;
        }
        if (!stopped && onProgress) {
          onProgress(r.data);
        }
      } catch (e) {
        // Any HTTP response (including 404) means the backend replied;
        // only network errors / timeouts leave lastBackendContactAt frozen.
        if (e?.response?.status !== undefined) {
          lastBackendContactAt = Date.now();
        }
      }
      const silentMs = Date.now() - lastBackendContactAt;
      if (silentMs >= STALE_THRESHOLD_MS && !stopped && onProgress) {
        const silentSec = Math.round(silentMs / 1000);
        onProgress({
          stage: 'error',
          stage_idx: 0,
          stage_total: 4,
          elapsed_seconds: silentMs / 1000,
          message: `Backend not responding (no reply for ${silentSec} s) — check that uvicorn is running on port 8000`,
          error: 'Backend not responding',
        });
        // Stop polling to avoid spamming a dead endpoint; the modal is
        // now in error state and the user clicks Close to dismiss.
        stopped = true;
        staleErrorEmitted = true;
        // The POST has no timeout of its own, so cancel it here — otherwise
        // it would wait forever on a backend that is not coming back.
        abortController.abort();
      }
      if (!stopped) {
        pollTimer = setTimeout(poll, pollIntervalMs);
      }
    };

    // Kick off the first poll on the next tick so the POST has a chance to
    // register the first stage. setTimeout(...,0) is enough.
    pollTimer = setTimeout(poll, 0);

    const postPromise = api.post(
      '/api/ebsd/load',
      {
        path,
        use_kikuchipy: useKikuchipy,
        request_id: requestId,
      },
      {
        // NO client-side deadline for the load itself. The axios instance
        // defaults to 5 minutes, which is a fine ceiling for ordinary calls
        // but wrong here: a large scan on a slow or networked disk can take
        // longer, and aborting the request does NOT stop the backend — it
        // keeps loading, finishes, and holds the file, while the user is
        // told "Load failed". A genuinely dead backend is still caught
        // within STALE_THRESHOLD_MS by the progress poll above, which aborts
        // this request via abortController.
        timeout: 0,
        signal: abortController.signal,
      },
    );

    return postPromise.then(
      (response) => {
        stopped = true;
        if (pollTimer) clearTimeout(pollTimer);
        return response;
      },
      (error) => {
        stopped = true;
        if (pollTimer) clearTimeout(pollTimer);
        // Synthesise an error progress event so the modal can render the
        // failure message — the backend may not have written stage='error'
        // yet (or the request never reached it). Skipped when the stale
        // detector already reported a better message and aborted us.
        if (onProgress && !staleErrorEmitted) {
          const detail =
            error?.response?.data?.detail || error?.message || 'Unknown error';
          onProgress({
            stage: 'error',
            stage_idx: 0,
            stage_total: 4,
            // How long the BACKEND said it had been loading, not 0 — a slow
            // load that fails should not look like an instant failure.
            elapsed_seconds: lastElapsedSeconds,
            message: detail,
            error: detail,
          });
        }
        throw error;
      },
    );
  },

  info: () => api.get('/api/ebsd/info'),
  getPattern: (row, col, config) => api.get(`/api/ebsd/pattern/${row}/${col}`, config),
  getPatternAtlas: (step = 4) => api.get(`/api/ebsd/pattern-atlas`, { params: { step } }),
  select: (indices) => api.post('/api/ebsd/select', { indices }),
  getDetector: () => api.get('/api/ebsd/detector'),
  getMetadata: () => api.get('/api/ebsd/metadata'),

  // Multi-dataset management
  datasets: () => api.get('/api/ebsd/datasets'),
  selectDataset: (name) => api.post('/api/ebsd/select-dataset', { name }),

  // Multi-file management — list every loaded H5OINA file and switch the
  // active one (switchFile re-loads the chosen file from disk).
  loadedFiles: () => api.get('/api/ebsd/loaded-files'),
  switchFile: (path) => api.post('/api/ebsd/switch-file', { path }),
  removeLoadedFile: (path) => api.post('/api/ebsd/loaded-files/remove', { path }),
  clearLoadedFiles: () => api.post('/api/ebsd/loaded-files/clear'),
  clearAllLoadedFiles: () => api.post('/api/ebsd/loaded-files/clear-all'),
  deepcopy: (name = '') => api.post('/api/ebsd/deepcopy', { name }),

  // Cut the active dataset down to a drawn selection. `mask` is null for a
  // rectangle — the backend reads that as "no mask, take the whole box" —
  // and a flat row-major boolean array of rows*cols for a free shape.
  crop: (payload) => api.post('/api/ebsd/crop', payload),
  // The active dataset's crop window, or { window: null } when it is a full scan.
  getCrop: () => api.get('/api/ebsd/crop'),
  // Write the active crop to `path` as a standalone file. The BACKEND does the
  // writing — this only hands it a destination — so `path` must be a real
  // filesystem path, not a download name.
  exportCrop: (path) => api.post('/api/ebsd/crop/export', { path }),
  deleteDataset: (name) => api.delete(`/api/ebsd/dataset/${encodeURIComponent(name)}`),

  // Signal processing (operate on active dataset in-place)
  frameAverage: (windowSize = 3) =>
    api.post('/api/ebsd/frame-average', { window_size: windowSize }),
  backgroundRemoval: (method = 'dynamic', staticBgRow = 0, staticBgCol = 0) =>
    api.post('/api/ebsd/background-removal', {
      method,
      static_bg_row: staticBgRow,
      static_bg_col: staticBgCol,
    }),
  autocontrast: () => api.post('/api/ebsd/autocontrast'),
  clahe: (kernelSize = 8, requestId = null) =>
    api.post('/api/ebsd/clahe', { kernel_size: kernelSize, request_id: requestId }),
  // Poll progress of a long processing op (CLAHE) by its request_id.
  processingProgress: (requestId) =>
    api.get(`/api/ebsd/processing-progress/${requestId}`),

  // Circular detector signal mask
  getSignalMask: () => api.get('/api/ebsd/signal-mask'),
  setSignalMask: (enabled, radiusFraction = 1.0) =>
    api.post('/api/ebsd/signal-mask', { enabled, radius_fraction: radiusFraction }),

  // Navigation overview
  overview: (mode = 'mean') =>
    api.get('/api/ebsd/overview', { params: { mode } }),

  // Virtual EBSD-derived images (used by EDS Analysis)
  virtualBSE: (cmap = 'gray', color = '', roi = '') =>
    api.get('/api/ebsd/virtual-bse', { params: { cmap, color, roi } }),
  bandContrast: (cmap = 'gray', color = '') =>
    api.get('/api/ebsd/band-contrast', { params: { cmap, color } }),
};

// --- PC Refinement ---
export const pcApi = {
  addPattern: (row, col) => api.post('/api/pc/pattern/add', { row, col }),
  removePattern: (idx) => api.post(`/api/pc/pattern/remove?idx=${idx}`),
  loadPhase: (cifPath) => api.post('/api/pc/phase/load', { cif_path: cifPath }),
  setDetector: (shape, pc, options = {}) => {
    const {
      sampleTilt = 70,
      cameraTilt = 0,
      binning = 1,
      detectorTilt = 0.0,
      azimuthal = 0.0,
      pixelSize = null,
      // PC values flowing through the app are in BRUKER convention
      // because that's what kikuchipy's EBSDDetector stores internally
      // and what /api/calibration/<dataset> + /api/pc/detector/info both
      // return on the way OUT. Defaulting this to 'tsl' here would make
      // kikuchipy interpret incoming numbers as TSL and convert them to
      // Bruker (PCy -> 1-PCy), so a no-op Apply silently flips PCy and
      // breaks pattern simulation. See investigation 2026-05-22.
      convention = 'bruker',
    } = options;
    return api.post('/api/pc/detector/set', {
      shape,
      pc,
      sample_tilt: sampleTilt,
      camera_tilt: cameraTilt,
      binning,
      detector_tilt: detectorTilt,
      azimuthal,
      pixel_size: pixelSize,
      convention,
    });
  },
  indexPattern: (row, col) => api.post('/api/pc/index-pattern', { row, col }),

  /**
   * Render a forward-simulated EBSP for a calibration pattern using
   * trial geometry. Returns experimental + simulated PNGs and NCC so
   * the PC Refinement page can visualize whether the trial geometry is
   * correct (and not just self-consistent with the Hough bands).
   *
   * All PC values are in BRUKER convention (matches the rest of the app).
   */
  renderPreview: ({
    patternIdx, shtPath, pc,
    sampleTilt, detectorTilt = 0.0, azimuthal = 0.0,
    binning = 1, pixelSize = null, maxBandwidth = 128,
    orientationEulerDeg = null,
  }) => api.post('/api/pc/render-preview', {
    pattern_idx: patternIdx,
    sht_path: shtPath,
    pc,
    sample_tilt: sampleTilt,
    detector_tilt: detectorTilt,
    azimuthal,
    binning,
    pixel_size: pixelSize,
    max_bandwidth: maxBandwidth,
    orientation_euler_deg: orientationEulerDeg,
  }),
  indexAll: () => api.post('/api/pc/index-all'),
  getPatternResult: (idx) => api.get(`/api/pc/pattern/${idx}/result`),
  getPatternImage: (idx) => api.get(`/api/pc/pattern/${idx}/image`),
  updatePC: (pcx, pcy, pcz, currentPatternIdx = null) =>
    api.post('/api/pc/detector/update-pc', {
      pcx, pcy, pcz, current_pattern_idx: currentPatternIdx,
    }),
  updateTilt: (sampleTilt, detectorTilt, azimuthal, currentPatternIdx = null) =>
    api.post('/api/pc/detector/update-tilt', {
      sample_tilt: sampleTilt,
      detector_tilt: detectorTilt,
      azimuthal,
      current_pattern_idx: currentPatternIdx,
    }),
  detectorInfo: () => api.get('/api/pc/detector/info'),
  updateParams: (params) => api.post('/api/pc/params/update', params),
  status: () => api.get('/api/pc/status'),
  optimize: (patterns, method = 'PSO', searchLimit = 0.05) =>
    api.post('/api/pc/optimize', { patterns, method, search_limit: searchLimit }),
  getOptimizeStatus: (taskId) => api.get(`/api/pc/optimize/${taskId}`),
  analyzeDrift: (indices) => api.post('/api/pc/drift', { indices }),
  runGridCalibration: (gridStep, mode = 'plane', method = 'PSO', searchLimit = 0.05) =>
    api.post('/api/pc/calibrate/grid', {
      grid_step: gridStep, mode, method, search_limit: searchLimit,
    }),
};

// --- Calibration ---
export const calibrationApi = {
  getEntry: (dataset) => api.get(`/api/calibration/${encodeURIComponent(dataset)}`),
  propagateToParent: (dataset) =>
    api.post(`/api/calibration/${encodeURIComponent(dataset)}/propagate-to-parent`),
};

// --- Simulation ---
export const simApi = {
  getConfig: () => api.get('/api/simulation/config'),
  updateConfig: (section, key, value) =>
    api.post('/api/simulation/config/update', { section, key, value }),
  // Backend caches for 5 minutes because the underlying WSL/EMsoft/OpenCL
  // checks take ~5 seconds. Pass { force: true } from the Settings "Re-Check"
  // button to bypass the cache.
  systemStatus: ({ force = false } = {}) =>
    api.get('/api/simulation/system-status' + (force ? '?force=true' : '')),
  start: (params) => api.post('/api/simulation/start', params),
  startGpu: (params) => api.post('/api/simulation/start-gpu', params),
  // Auto-route a single job: ask the backend which engine ('emsoft' | 'gpu')
  // fits this .xtal at this effective dmin (symmetry + reflection-count
  // heuristic). Response: { engine, reason, reflections, point_group_order,
  // emsoft_available }. Used only when the engine toggle is set to 'auto'.
  recommendEngine: (xtal, dmin = 0.05) =>
    api.get('/api/simulation/recommend-engine', { params: { xtal, dmin } }),
  getStatus: (taskId) => api.get(`/api/simulation/status/${taskId}`),
  stop: (taskId) => api.post(`/api/simulation/stop/${taskId}`),
  history: () => api.get('/api/simulation/history'),
  clearHistory: () => api.post('/api/simulation/history/clear'),
  getNmlTemplate: (params) => api.get('/api/simulation/nml-template', { params }),
  scanMissing: (outputType = 'both', ekev = 20.0) =>
    api.get('/api/simulation/scan-missing', { params: { output_type: outputType, ekev } }),
  batchStart: (params) => api.post('/api/simulation/batch/start', params),
  // GPU batch — same payload + same status polling as batchStart; only the
  // start URL differs (POST /api/simulation/batch/start-gpu). Used when the
  // Simulation page engine toggle is set to 'gpu'.
  batchStartGpu: (params) => api.post('/api/simulation/batch/start-gpu', params),
  batchStatus: (batchId) => api.get(`/api/simulation/batch/status/${batchId}`),
  batchCancel: (batchId) => api.post(`/api/simulation/batch/cancel/${batchId}`),
  getLog: (taskId) => api.get(`/api/simulation/log/${taskId}`, { responseType: 'text' }),
  serverConfig: () => api.get('/api/simulation/server-config'),
  updateServerConfig: (cfg) => api.post('/api/simulation/server-config', cfg),
  testServerConnection: () => api.post('/api/simulation/server-config/test'),
  crystalPicker: (kv = 20) => api.get('/api/simulation/crystal-picker', { params: { kv } }),
};

// --- Indexing ---
export const indexApi = {
  start: (params) => api.post('/api/indexing/start', params),
  getStatus: (taskId) => api.get(`/api/indexing/status/${taskId}`),
  stop: (taskId) => api.post(`/api/indexing/stop/${taskId}`),
  releaseGpu: () => api.post('/api/indexing/release-gpu'),
  gpuStatus: () => api.get('/api/indexing/gpu-status'),
  getLastResult: () => api.get('/api/indexing/result/last'),
  methods: () => api.get('/api/indexing/methods'),
  discoverFiles: (method, materialHint = '', currentPc = null) =>
    api.get(`/api/indexing/files/${method}`, {
      params: {
        material_hint: materialHint,
        current_pc: currentPc ? currentPc.join(',') : '',
      },
    }),
  exportUrl: (format = 'h5', includeEds = true, includeDetector = true) =>
    `${API_BASE}/api/indexing/export?format=${format}&include_eds=${includeEds}&include_detector=${includeDetector}`,
  exportResult: (format = 'h5', includeEds = true, includeDetector = true) =>
    api.post('/api/indexing/export', { format, include_eds: includeEds, include_detector: includeDetector }, { responseType: 'blob' }),
  // Batch indexing
  batchStart: (datasets, autoExport = true, exportDir = '', cleanup = true) =>
    api.post('/api/indexing/batch/start', { datasets, auto_export: autoExport, export_dir: exportDir, cleanup_after_export: cleanup }),
  batchStatus: () => api.get('/api/indexing/batch/status'),
  batchStop: () => api.post('/api/indexing/batch/stop'),
  refine: (params) => api.post('/api/indexing/refine', params),
  previewPreprocessing: (params) => api.post('/api/indexing/preview-preprocessing', params),
  // EDS pre-flight: is the loaded file's chemistry usable for these phases?
  // `phaseFiles` must be the SAME paths that go into eds_phase_strengths (i.e.
  // already remapped to the selected dictionary path for Dictionary runs), so
  // the per-phase verdicts line up with what the run will actually weight.
  edsPreflight: (phaseFiles = []) =>
    api.post('/api/indexing/eds-preflight', { phase_files: phaseFiles }),
  // Per-dataset PC storage
  storePC: (datasetName, pc) => api.post(`/api/indexing/pc/store?dataset_name=${encodeURIComponent(datasetName)}`, pc),
  getStoredPCs: () => api.get('/api/indexing/pc/stored'),
  // Pattern Matches Viewer. When ``comparePhases=true`` the response
  // additionally carries ``phase_results`` — per-phase R-scores +
  // simulated patterns for the same pixel, sorted best R first. The
  // Pattern Match Quality dialog uses it for the misindex-diagnose
  // phase-navigator (2026-05-26). Costs ~0.5 s extra per click on a
  // 12-phase result after the per-result backends are warm.
  patternMatch: (row, col, rank = 0, maxBandwidth = null, aperture = 'auto',
                 apertureRadius = 1.0, comparePhases = false) => api.get(
    `/api/indexing/pattern-match`,
    { params: {
      row, col, rank,
      ...(maxBandwidth != null ? { max_bandwidth: maxBandwidth } : {}),
      aperture, aperture_radius: apertureRadius,
      ...(comparePhases ? { compare_phases: true } : {}),
    } },
  ),
  // Universal manual pseudo-symmetry flip: candidate orientations (current +
  // crystallographic pseudo-variants + Hough) rendered + render-NCC, and grain
  // propagation of the chosen correction.
  // refRow/refCol: optional free reference pixel — its stored orientation
  // joins the gallery as a rendered candidate (foreign-basin fixes).
  // reindex: fresh full orientation search for the pixel's own phase
  // (first call per result pays a ~6 s backend warmup).
  patternMatchVariants: (row, col, { maxBandwidth = 128, aperture = 'auto', apertureRadius = 1.0, refRow = null, refCol = null, reindex = false } = {}) =>
    api.get('/api/indexing/pattern-match/variants', { params: {
      row, col, max_bandwidth: maxBandwidth, aperture, aperture_radius: apertureRadius,
      ...(refRow != null && refCol != null ? { ref_row: refRow, ref_col: refCol } : {}),
      ...(reindex ? { reindex: true } : {}),
    } }),
  // propagateSimilar: ALSO fix every other same-phase grain map-wide that
  // sits at the same wrong orientation — each sibling render-verified.
  applyVariantToGrain: ({ row, col, quat, thresholdDeg = 5.0, maxTotalDeg = 15.0, refine = false, propagateSimilar = false }) =>
    api.post('/api/indexing/pattern-match/apply-to-grain', {
      row, col, quat, threshold_deg: thresholdDeg, max_total_deg: maxTotalDeg, refine,
      propagate_similar: propagateSimilar,
    }),
  // Manual per-grain PHASE reassignment from the Compare-phases view (the
  // surgical sibling of the map-wide Phase Verification). Undo shares
  // /phase-reassign/undo.
  assignPhaseToGrain: ({ row, col, targetPhaseId, thresholdDeg = 5.0 }) =>
    api.post('/api/indexing/pattern-match/assign-phase', {
      row, col, target_phase_id: targetPhaseId, threshold_deg: thresholdDeg,
    }),
  undoGrainFlip: () =>
    api.post('/api/indexing/pattern-match/undo-grain', {}),
  // Map-wide pseudo-symmetry variant unification (per-grain render-NCC
  // verified; one-level undo shares the grain-flip undo slot).
  unifyPseudosymVariants: () =>
    api.post('/api/indexing/pseudosym/unify', {}),
  // Render-verified phase check (Stage A, read-only) + grain-based phase
  // reassignment (Stage B) + one-level undo. Fixes chemically-degenerate
  // phases stealing pixels of another phase (judge by render-NCC).
  phaseCheck: () =>
    api.post('/api/indexing/phase-check', {}),
  phaseReassign: () =>
    api.post('/api/indexing/phase-reassign', {}),
  phaseReassignUndo: () =>
    api.post('/api/indexing/phase-reassign/undo', {}),
  // Phase B — Forward-NCC quality map
  forwardNccStatus: () => api.get('/api/indexing/forward-ncc/status'),
  forwardNccCompute: (maxBandwidth = 256, force = false) => api.post(
    '/api/indexing/forward-ncc/compute',
    null,
    { params: { max_bandwidth: maxBandwidth, force } },
  ),
  forwardNccHeatmap: () => api.get('/api/indexing/forward-ncc/heatmap'),
  nccHeatmap: () => api.get('/api/indexing/ncc-heatmap'),
  singlePixelPhaseTest: (params) =>
    api.post('/api/indexing/single-pixel-phase-test', params),
  // Non-blocking single-pixel phase test: start a background job, poll its
  // progress, and cancel it. Replaces the synchronous singlePixelPhaseTest
  // above for runs that can take minutes (the synchronous one times out).
  phaseTestStart: (params) =>
    api.post('/api/indexing/single-pixel-phase-test/start', params),
  phaseTestRemask: (params) =>
    api.post('/api/indexing/single-pixel-phase-test/remask', params),
  phaseTestProgress: (jobId) =>
    api.get(`/api/indexing/single-pixel-phase-test/progress/${jobId}`),
  phaseTestCancel: (jobId) =>
    api.post(`/api/indexing/single-pixel-phase-test/cancel/${jobId}`),
  phaseTestPhases: () =>
    api.get('/api/indexing/single-pixel-phase-test/phases'),
  // Lightweight per-pixel EDS chemistry so the dialog's EDS readout can follow
  // the crosshair live (no GPU/detector/pattern read involved).
  phaseTestPixelChemistry: (pixelIndex) =>
    api.get(`/api/indexing/single-pixel-phase-test/pixel-chemistry/${pixelIndex}`),
  // Result registry
  listResults: () => api.get('/api/indexing/results'),
  activateResult: (id) => api.post(`/api/indexing/results/activate/${id}`),
  deactivateResult: () => api.post('/api/indexing/results/deactivate'),
  deleteResult: (id) => api.delete(`/api/indexing/results/${id}`),
};

// --- Phase Map ---
export const phaseMapApi = {
  render: (params = {}) => api.get('/api/phasemap/render', { params }),
  /**
   * Fetch a single transparent-BG RGBA layer for frontend compositing.
   * @param {string} kind  phase | ipf-x | ipf-y | ipf-z | bc | ci | ci_<phase> | uncertainty
   * @param {object} cleanupParams  Optional cleanup filter passthrough.
   */
  layer: (kind, cleanupParams = {}) =>
    api.get('/api/phasemap/layer', { params: { kind, ...cleanupParams } }),
  /** Fetch the IPF colour key triangles for the active xmap (transparent BG PNG).
   *  phaseFilter (phase id, -1 = all) restricts the key to one phase so it
   *  matches a phase-filtered IPF layer. orientation 'vertical' stacks the
   *  triangles in a column (for the side panel next to the map). */
  ipfKey: (direction = 'Z', phaseFilter = -1, orientation = 'horizontal', colorOverrides = null) =>
    api.get('/api/phasemap/ipf-key', {
      params: {
        direction, phase_filter: phaseFilter, orientation,
        // The swatch above each triangle names a phase, so it has to carry the
        // user's colour picks like the map and the legend do.
        ...(colorOverrides && Object.keys(colorOverrides).length > 0
          ? { color_overrides: JSON.stringify(colorOverrides) }
          : {}),
      },
    }),
  // Export takes the full render-params object so the file on disk matches the
  // live preview (direction, cleanup, scalebar, title, confidence overlay…).
  export: (outputPath, format = 'png', dpi = 300, viewParams = {}) =>
    api.post('/api/phasemap/export', {
      output_path: outputPath, format, dpi, ...viewParams,
    }),
  directions: () => api.get('/api/phasemap/available-directions'),
  availableMaps: () => api.get('/api/phasemap/available-maps'),
  // Phase id -> name -> colour mapping for the active result (legend panel).
  phaseLegend: () => api.get('/api/phasemap/phase-legend'),
  // Filtered stats: only PRESENT phases + pixel-count + area-% + median CI
  // + suspect flag for likely misindex candidates. Drives the rebuilt
  // PhaseLegend.jsx panel.
  // ``includeEmpty=true`` also returns PhaseList entries with zero
  // indexed pixels (so the user can audit which input phases didn't
  // win anywhere). Default keeps the trimmed-down view from feature A.
  phaseStats: (includeEmpty = false, colorOverrides = null) =>
    api.get('/api/phasemap/phase-stats', {
      params: {
        ...(includeEmpty ? { include_empty: true } : {}),
        ...(colorOverrides && Object.keys(colorOverrides).length > 0
          ? { color_overrides: JSON.stringify(colorOverrides) }
          : {}),
      },
    }),
  // Phase-pair adjacency matrix: how often two phases share a 4-neighbour
  // border. Reveals chemical degeneracy hotspots (high counts between
  // similar phases = likely misindex, not a real interface).
  phaseAdjacency: () => api.get('/api/phasemap/phase-adjacency'),
  // Cleanup: when fileStem/searchDir resolve to a <stem>_multiphase.h5 the
  // call writes into that checkpoint; otherwise it mutates the in-memory
  // indexing result so /render + downstream exports see the cleaned map.
  applyCleanup: ({
    fileStem = null, searchDir = null,
    ciThreshold = 0, uncertaintyThreshold = 0, minClusterSize = 0,
    fillUnindexed = false, modalFilterSize = 0,
    target = 'auto',
  } = {}) =>
    api.post('/api/phasemap/apply-cleanup', {
      file_stem: fileStem, search_dir: searchDir,
      ci_threshold: ciThreshold, uncertainty_threshold: uncertaintyThreshold,
      min_cluster_size: minClusterSize,
      fill_unindexed: fillUnindexed,
      modal_filter_size: modalFilterSize,
      target,
    }),
  // Tier-2 inspection endpoints (probe / region-stats / linescan).
  probe: (row, col) => api.post('/api/phasemap/probe', { row, col }),
  regionStats: (body) => api.post('/api/phasemap/region-stats', body),
  linescan: (body) => api.post('/api/phasemap/linescan', body),
};

// --- Analysis ---
export const analysisApi = {
  load: (xmapPath) => api.post('/api/analysis/load', { xmap_path: xmapPath }),
  status: () => api.get('/api/analysis/status'),
  reconstructGrains: (threshold = 5.0, minSize = 5, minIntercept = 1.5) =>
    api.post('/api/analysis/grains/reconstruct', { misorientation_threshold: threshold, min_grain_size: minSize, min_intercept_um: minIntercept }),
  getTaskStatus: (taskId) => api.get(`/api/analysis/task/${taskId}`),
  getMap: (mapType, cmap = 'viridis') =>
    api.get(`/api/analysis/map/${mapType}`, { params: { cmap } }),
  grainSize: () => api.post('/api/analysis/grain-size'),
  deformation: () => api.post('/api/analysis/deformation'),
  texture: () => api.post('/api/analysis/texture'),
  bcGmm: () => api.post('/api/analysis/bc_gmm'),
  textureComponents: (preset = 'FCC_Rolling') =>
    api.get('/api/analysis/texture/components', { params: { preset } }),
  exportExcel: (outputPath, name = 'Dataset') =>
    api.post('/api/analysis/export/excel', { output_path: outputPath, dataset_name: name }),
  downloadExcel: (name = 'Dataset') =>
    `${API_BASE}/api/analysis/export/excel-download?name=${encodeURIComponent(name)}`,
  batch: (folderPath, outputPath) =>
    api.post('/api/analysis/batch', { folder_path: folderPath, output_path: outputPath }),
  qualityStats: () => api.get('/api/analysis/quality-stats'),
  applyQualityFilter: (bcMin = 0, bandsMin = 0) =>
    api.post('/api/analysis/quality-filter', { bc_min: bcMin, bands_min: bandsMin }),
  pcDriftImage: () => api.get('/api/analysis/pc-drift-image'),
  pcDriftToSeed: (targetDataset = null) =>
    api.post('/api/analysis/pc-drift-to-seed', { target_dataset: targetDataset }),
  aztecComparison: () => api.get('/api/analysis/aztec-comparison'),
};

// --- ML Hub ---
export const mlApi = {
  status: () => api.get('/api/ml/status'),
  models: () => api.get('/api/ml/models'),
  predict: (patternIndices) => api.post('/api/ml/predict', { pattern_indices: patternIndices }),
  // Train a phase classifier. Pass '__last_indexing__' to pull samples from
  // the most recent indexing result, then train. Returns { task_id }.
  train: ({ trainingDataPath = '__last_indexing__', epochs = 50, batchSize = 32, ciThreshold = 0.3 } = {}) =>
    api.post('/api/ml/train', {
      training_data_path: trainingDataPath, epochs, batch_size: batchSize, ci_threshold: ciThreshold,
    }),
  trainStatus: (taskId) => api.get(`/api/ml/train/${taskId}`),
  clearStore: () => api.post('/api/ml/store/clear'),
};

// --- Batch v2 (Multi-Phase) ---
export const batchV2Api = {
  scanFolder: (folder, recursive = true) =>
    api.post('/api/batch-v2/scan-folder', { folder, recursive }),
  quickLoad: (filePath) =>
    api.post('/api/batch-v2/quick-load', { file_path: filePath }),
  copyPC: (sourceDataset, targetDatasets) =>
    api.post('/api/batch-v2/copy-pc', { source_dataset: sourceDataset, target_datasets: targetDatasets }),
  pcStatus: (fileNames = '') =>
    api.get('/api/batch-v2/pc-status', { params: { files: fileNames } }),
  unload: (datasetName = '') =>
    api.post('/api/batch-v2/unload', null, { params: { dataset_name: datasetName } }),
  create: (files, phases, config = {}) =>
    api.post('/api/batch-v2/create', { files, phases, config }),
  start: (batchId) => api.post(`/api/batch-v2/${batchId}/start`),
  pause: (batchId) => api.post(`/api/batch-v2/${batchId}/pause`),
  resume: (batchId) => api.post(`/api/batch-v2/${batchId}/resume`),
  stop: (batchId) => api.post(`/api/batch-v2/${batchId}/stop`),
  status: (batchId) => api.get(`/api/batch-v2/${batchId}/status`),
  jobs: (batchId) => api.get(`/api/batch-v2/${batchId}/jobs`),
  retryFailed: (batchId) => api.post(`/api/batch-v2/${batchId}/retry-failed`),
  deleteBatch: (batchId) => api.delete(`/api/batch-v2/${batchId}`),
  list: () => api.get('/api/batch-v2/list'),
  report: (batchId) => api.get(`/api/batch-v2/${batchId}/report`),
};

// --- Phase Refinement (file-stem based per-pixel phase override) ---
// Renamed from `refinementApi` to `phaseRefinementApi` to coexist with the new
// `refinementApi` below, which handles Phase B joint R+PC refinement on
// result_id. Distinct feature, distinct shape — only the URL prefix is shared
// (`/api/refinement/...` vs `/api/refinement/{file_stem}/...`).
export const phaseRefinementApi = {
  summary: (fileStem, searchDir) =>
    api.get(`/api/refinement/${encodeURIComponent(fileStem)}/summary`, { params: { search_dir: searchDir } }),
  pixel: (fileStem, row, col, searchDir) =>
    api.get(`/api/refinement/${encodeURIComponent(fileStem)}/pixel/${row}/${col}`, { params: { search_dir: searchDir } }),
  override: (fileStem, pixels, source, searchDir) =>
    api.post(`/api/refinement/${encodeURIComponent(fileStem)}/override`, { pixels, source }, { params: { search_dir: searchDir } }),
  overrideRegion: (fileStem, region, searchDir) =>
    api.post(`/api/refinement/${encodeURIComponent(fileStem)}/override-region`, region, { params: { search_dir: searchDir } }),
};

// --- Database ---
export const dbApi = {
  browse: (category = 'all', material = '') =>
    api.get('/api/database/browse', { params: { category, material } }),
  cacheStatus: () => api.get('/api/database/cache/status'),
  categories: () => api.get('/api/database/categories'),

  // File management
  addCif: (filePath, materialName = '') =>
    api.post('/api/database/add-cif', { file_path: filePath, material_name: materialName }),
  cifInfo: (filename) => api.get(`/api/database/cif/${encodeURIComponent(filename)}/info`),
  xtalInfo: (stem) => api.get(`/api/database/xtal/${encodeURIComponent(stem)}/info`),
  addXtal: (filePath, materialName = '') =>
    api.post('/api/database/add-xtal', { file_path: filePath, material_name: materialName }),
  convertCifToXtal: (cifPath, outputDir = '') =>
    api.post('/api/database/convert-cif-to-xtal', { cif_path: cifPath, output_dir: outputDir }),

  // Build database (parse all CIFs into crystal_database.xlsx)
  build: (skipOnline = true) =>
    api.post('/api/database/build', { skip_online: skipOnline }),
  buildStatus: (taskId) =>
    api.get(`/api/database/build/status/${taskId}`),

  // Read built database entries (rich crystal metadata)
  entries: () => api.get('/api/database/entries'),

  // Update cell values in crystal_database.xlsx
  updateEntries: (updates) => api.patch('/api/database/entries/update', { updates }),

  // Master pattern / SHT thumbnail preview
  preview: (filename) => api.get(`/api/database/preview/${encodeURIComponent(filename)}`),

  // Master pattern of a .sht reconstructed onto a sphere grid (3D viewer).
  // Returns { filename, grid, phi[], theta[], bandwidth, file_bandwidth, meta }.
  sphere: (filename, params = {}) =>
    api.get(`/api/database/sphere/${encodeURIComponent(filename)}`, { params }),

  // Full-unit-cell crystal structure of a local .cif/.xtal (3D ball-and-stick viewer).
  // Returns { source, lattice, cell_vectors, space_group, atoms[], bonds[], polyhedra[], meta }.
  structure: (filename) =>
    api.get(`/api/database/structure/${encodeURIComponent(filename)}`),

  // Provenance + simulation parameters for a .sht (File Info panel).
  shtInfo: (filename) => api.get(`/api/database/sht/${encodeURIComponent(filename)}/info`),

  // DWF reference table (Debye-Waller factors)
  getDwf: () => api.get('/api/database/dwf'),
  updateDwf: (updates) => api.patch('/api/database/dwf', { updates }),

  // Resolve cascade dependencies before deleting
  resolveCascade: (files) =>
    api.post('/api/database/resolve-cascade', { files }),

  // Delete files — deleteFrom: "local" | "server" | "everywhere"
  deleteFiles: (files, { deleteFrom = 'local', deleteFromServer } = {}) =>
    api.delete('/api/database/files', {
      data: {
        files,
        delete_from: deleteFromServer !== undefined ? (deleteFromServer ? 'everywhere' : 'local') : deleteFrom,
      },
    }),

  // Update CIF metadata — only include fields the caller actually passed.
  // Backend treats an absent/empty field as "leave unchanged" and an edit
  // to one field shouldn't even transmit the other value.
  updateCifMeta: (filename, payload) => {
    const body = {};
    if (payload?.doi !== undefined) body.doi = payload.doi;
    if (payload?.reference !== undefined) body.reference = payload.reference;
    return api.patch(`/api/database/cif/${encodeURIComponent(filename)}/meta`, body);
  },

  // Batch parse CIF files
  batchCifInfo: (filenames) => api.post('/api/database/cif/batch-info', { filenames }),

  // Sync
  syncStatus: () => api.get('/api/database/sync/status'),
  sync: () => api.post('/api/database/sync'),
  resolveConflicts: (resolutions) => api.post('/api/database/sync/resolve', { resolutions }),

  // Selective transfer (any category: cif/xtal/sht/h5/master/dictionary).
  // files: [{ name, category, material }]. `signal` allows aborting an in-flight
  // request (used by the bulk-upload Cancel button).
  upload: (files, { overwrite = false, signal } = {}) =>
    api.post('/api/database/upload', { files, overwrite }, { signal }),
  download: (files, { overwrite = false, signal } = {}) =>
    api.post('/api/database/download', { files, overwrite }, { signal }),
};

// --- EDS ---
export const edsApi = {
  elements: () => api.get('/api/eds/elements'),
  getMap: (element, mode = 'counts', cmap = 'hot', color = '') =>
    api.get(`/api/eds/map/${element}`, { params: { mode, cmap, color } }),
  quantifyPixel: (row, col, mode = 'counts') =>
    api.post('/api/eds/quantify/pixel', { row, col, display_mode: mode }),
  probe: (row, col, displayMode = 'at_pct') =>
    api.post('/api/eds/probe', { row, col, display_mode: displayMode }),
  linescan: (body) => api.post('/api/eds/linescan', body),
  regionQuantify: (rowStart, rowEnd, colStart, colEnd, mode = 'at_pct') =>
    api.post('/api/eds/region-quantify', {
      row_start: rowStart, row_end: rowEnd,
      col_start: colStart, col_end: colEnd,
      display_mode: mode,
    }),
  suggestPhases: (row, col) =>
    api.post('/api/eds/suggest-phases', { row, col }),
  cifPhases: () => api.get('/api/eds/cif-phases'),
  // Options object rather than positional args: the request grew a mode,
  // a cluster count and a phase selection, and positional arguments for
  // that many optional fields are a bug waiting to happen.
  autoClassify: (opts = {}) =>
    api.post('/api/eds/auto-classify', {
      tolerance: opts.tolerance ?? 15.0,
      min_score: opts.minScore ?? 0.3,
      ...(opts.mode ? { mode: opts.mode } : {}),
      ...(opts.nClusters != null ? { n_clusters: opts.nClusters } : {}),
      ...(opts.phaseKeys ? { phase_keys: opts.phaseKeys } : {}),
    }),
  getPhaseMap: (includeImage = true) =>
    api.get('/api/eds/phase-map', { params: { include_image: includeImage } }),
  clearPhaseMap: () => api.delete('/api/eds/phase-map'),
  assignRegion: (rowStart, rowEnd, colStart, colEnd, phaseIndex) =>
    api.post('/api/eds/phase-map/assign-region', {
      row_start: rowStart, row_end: rowEnd,
      col_start: colStart, col_end: colEnd,
      phase_index: phaseIndex,
    }),
  assignPolygon: (vertices, phaseIndex) =>
    api.post('/api/eds/phase-map/assign-polygon', {
      vertices, phase_index: phaseIndex,
    }),
  phaseMapIndexingConfig: () => api.get('/api/eds/phase-map/indexing-config'),
  displayModes: () => api.get('/api/eds/display-modes'),
  chemistryMask: (filters, combine = 'and', margin_px = 0) =>
    api.post('/api/eds/chemistry-mask', {
      filters: filters.map(f => ({
        element: f.element,
        operator: f.operator,
        min_val: f.min,
        max_val: f.max,
        unit: f.unit || 'at_pct',
      })),
      combine,
      margin_px,
    }),
};

// --- Install Wizard ---
export const installApi = {
  wslStatus: () => api.get('/api/install/wsl-status'),
  installWsl: (distro = 'Ubuntu-22.04', repair = false, brokenDistro = '') =>
    api.post('/api/install/wsl-install', null, {
      params: { distro, repair, broken_distro: brokenDistro },
    }),
  createUser: (username, password) =>
    api.post('/api/install/wsl-create-user', { username, password }),
  resetPassword: (username, password) =>
    api.post('/api/install/wsl-reset-password', { username, password }),
  validatePassword: (password) =>
    api.post('/api/install/validate-password', { password }),
};

// --- Dictionary GPU (stand-alone tool) ---
export const dictionaryGpuApi = {
  generate: (payload) => api.post('/api/dictionary-gpu/generate', payload),
  progress: (taskId) => api.get(`/api/dictionary-gpu/progress/${taskId}`),
  list: () => api.get('/api/dictionary-gpu/list'),
  delete: (name) => api.delete(`/api/dictionary-gpu/${encodeURIComponent(name)}`),
};

// Named function exports for Task 15 contract (used by React page in Task 16).
export const generateDictionaryGpu = (payload) => dictionaryGpuApi.generate(payload);
export const getDictionaryGpuProgress = (taskId) => dictionaryGpuApi.progress(taskId);
export const listDictionaryGpu = () => dictionaryGpuApi.list();
export const deleteDictionaryGpu = (name) => dictionaryGpuApi.delete(name);

// --- Forward NCC Diagnostics ---
//
// Uses fetch (not axios) so we can:
//   - return null cleanly on 404 from /summary ("not computed yet" — not an error)
//   - surface FastAPI's `detail.error` strings verbatim in thrown messages
// Base URL is taken from the same API_BASE constant as the axios client above
// (relative in dev so Vite proxies, absolute in Electron via VITE_API_URL).
const FWD_DIAG_BASE = `${API_BASE}/api/forward-diagnostics`;

function _qs(obj) {
  // Skip undefined / null / empty-string so backend defaults kick in.
  const parts = [];
  for (const [k, v] of Object.entries(obj || {})) {
    if (v === undefined || v === null || v === '') continue;
    parts.push(`${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
  }
  return parts.length ? `?${parts.join('&')}` : '';
}

async function _fwdDiagError(response) {
  // FastAPI raises HTTPException(detail={"error": "..."}), which serialises to
  // {"detail": {"error": "..."}}. Fall through to plain {"error": "..."} or
  // HTTP status if neither is present.
  let body = null;
  try { body = await response.json(); } catch (_) { /* non-JSON body */ }
  const msg = body?.detail?.error || body?.error || `HTTP ${response.status}`;
  return new Error(msg);
}

export const forwardDiagApi = {
  compute: async (result_id, max_bandwidth, force_recompute = false) => {
    const r = await fetch(`${FWD_DIAG_BASE}/compute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result_id, max_bandwidth, force_recompute }),
    });
    if (!r.ok) throw await _fwdDiagError(r);
    return r.json();
  },

  progress: async (job_id) => {
    const r = await fetch(`${FWD_DIAG_BASE}/progress${_qs({ job_id })}`);
    if (!r.ok) throw await _fwdDiagError(r);
    return r.json();
  },

  // Returns null on 404 so callers can treat "not computed yet" as a state,
  // not an exception.
  summary: async (result_id) => {
    const r = await fetch(`${FWD_DIAG_BASE}/summary${_qs({ result_id })}`);
    if (r.status === 404) return null;
    if (!r.ok) throw await _fwdDiagError(r);
    return r.json();
  },

  browser: async (result_id, opts = {}) => {
    const { sort, n, range_min, range_max } = opts;
    const r = await fetch(`${FWD_DIAG_BASE}/browser${_qs({
      result_id, sort, n, range_min, range_max,
    })}`);
    if (!r.ok) throw await _fwdDiagError(r);
    return r.json();
  },

  thumbnail: async (result_id, row, col, size = 64) => {
    const r = await fetch(`${FWD_DIAG_BASE}/thumbnail${_qs({
      result_id, row, col, size,
    })}`);
    if (!r.ok) throw await _fwdDiagError(r);
    return r.json();
  },

  cancel: async (job_id) => {
    const r = await fetch(`${FWD_DIAG_BASE}/cancel${_qs({ job_id })}`, {
      method: 'POST',
    });
    if (!r.ok) throw await _fwdDiagError(r);
    return r.json();
  },
};

/**
 * Refinement API — Phase B (joint R+PC refinement).
 * Spec: docs/superpowers/specs/2026-05-14-joint-r-pc-refinement-design.md
 *
 * Mirrors `forwardDiagApi` above (fetch-based; null on 404 from /summary;
 * throws FastAPI's detail.error verbatim).
 */
const _qsRef = (obj) =>
  Object.entries(obj || {})
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&');

const REFINEMENT_BASE = `${API_BASE}/api/refinement`;

async function _refError(r) {
  let body = null;
  try { body = await r.json(); } catch { /* not JSON */ }
  const msg = body?.detail?.error || body?.error || `HTTP ${r.status}`;
  throw new Error(msg);
}

export const refinementApi = {
  compute: async (result_id, smoothness_lambda, force_full_recompute = false) => {
    const r = await fetch(`${REFINEMENT_BASE}/compute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result_id, smoothness_lambda, force_full_recompute }),
    });
    if (!r.ok) return _refError(r);
    return r.json();
  },
  resmooth: async (result_id, smoothness_lambda) => {
    const r = await fetch(`${REFINEMENT_BASE}/resmooth`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result_id, smoothness_lambda }),
    });
    if (!r.ok) return _refError(r);
    return r.json();
  },
  progress: async (job_id) => {
    const qs = _qsRef({ job_id });
    const r = await fetch(`${REFINEMENT_BASE}/progress${qs ? `?${qs}` : ''}`);
    if (!r.ok) return _refError(r);
    return r.json();
  },
  summary: async (result_id) => {
    const qs = _qsRef({ result_id });
    const r = await fetch(`${REFINEMENT_BASE}/summary${qs ? `?${qs}` : ''}`);
    if (r.status === 404) return null;
    if (!r.ok) return _refError(r);
    return r.json();
  },
  cancel: async (job_id) => {
    const qs = _qsRef({ job_id });
    const r = await fetch(`${REFINEMENT_BASE}/cancel${qs ? `?${qs}` : ''}`, {
      method: 'POST',
    });
    if (!r.ok) return _refError(r);
    return r.json();
  },
};

// --- Settings ---
export const settingsApi = {
  get: () => api.get('/api/settings'),
  saveServer: (config) => api.put('/api/settings/server', config),
  savePaths: (paths) => api.put('/api/settings/paths', paths),
  testServer: (database_root) => api.post('/api/settings/server/test', { database_root }),
  createServerDirs: (database_root) => api.post('/api/settings/server/create-dirs', { database_root }),
  // API keys: stored per-machine in user-config. Pass empty string to clear.
  // Only provided keys are updated — omit fields to leave them unchanged.
  saveApiKeys: (keys) => api.put('/api/settings/api-keys', keys),
  // Test a configured key. Currently supports name='materials_project'.
  testApiKey: (name) => api.post('/api/settings/api-keys/test', { name }),
};

// --- Crystal Hint ---
// Detects crystal symmetry from EBSD patterns + suggests candidate phases
// from the local library filtered by sample chemistry. Companion endpoints
// for presets and library inventory.
export const crystalHintApi = {
  // Mode 1: analyse one pixel — symmetry + lattice + ranked candidates
  // avgRadius: 0=single pixel, 1=3x3 average, 2=5x5 average. Averaging
  // neighbouring patterns roughly doubles the symmetry NCC on real EBSD
  // data by suppressing per-pixel noise.
  analyzePixel: ({
    pixelIndex,
    elements = [],
    presetKey = null,
    crystalSystemFilter = null,
    strictChemistry = true,
    patternType = 'processed',
    avgRadius = 0,
    edsWeighting = 'soft',
  } = {}) =>
    api.post('/api/crystal-hint/analyze-pixel', {
      pixel_index: pixelIndex,
      elements,
      preset_key: presetKey,
      crystal_system_filter: crystalSystemFilter,
      strict_chemistry: strictChemistry,
      pattern_type: patternType,
      avg_radius: avgRadius,
      eds_weighting: edsWeighting,
    }),
  // All material presets (AA226, AA6061, etc.)
  listPresets: () => api.get('/api/crystal-hint/presets'),
  getPreset: (key) => api.get(`/api/crystal-hint/presets/${encodeURIComponent(key)}`),
  // Full local CIF/SHT library inventory (for debug + reference UI)
  listLibrary: () => api.get('/api/crystal-hint/library'),
  // Mode 2: aggregate analysis across a region (ROI or whole scan)
  // avgRadius: per-pixel neighborhood averaging — same semantics as Mode 1.
  analyzeRegion: ({
    elements = [],
    presetKey = null,
    roiType = 'whole',
    rect = null,            // [r_min, r_max, c_min, c_max]
    polygon = null,         // [[row, col], ...]
    maxPixels = 256,
    avgRadius = 0,
  } = {}) =>
    api.post('/api/crystal-hint/analyze-region', {
      elements,
      preset_key: presetKey,
      roi_type: roiType,
      avg_radius: avgRadius,
      rect,
      polygon,
      max_pixels: maxPixels,
    }),
  // Mode 3: compare detected symmetry vs indexed phase symmetry
  qualityCheck: ({
    roiType = 'whole',
    rect = null,
    polygon = null,
    maxPixels = 256,
    avgRadius = 0,
  } = {}) =>
    api.post('/api/crystal-hint/quality-check', {
      roi_type: roiType, rect, polygon, max_pixels: maxPixels,
      avg_radius: avgRadius,
    }),
  // Download a CIF from an external DB (COD only for v1) and validate it.
  // Optionally persists it into Database/CIF_Library/. Does NOT trigger
  // SHT generation — the user must run that explicitly from Simulation page.
  downloadCif: ({ source = 'COD', structureId, save = true, overwrite = false } = {}) =>
    api.post('/api/crystal-hint/download-cif', {
      source, structure_id: structureId, save, overwrite,
    }),
  // External-DB search (COD + Materials Project). Slow (~1-5s) — run in
  // parallel with analyzePixel for non-blocking UX. Accepts an optional
  // AbortSignal so callers can cancel stale searches (e.g. when the user
  // rapidly clicks new pixels — without cancel, the backend keeps doing
  // 1-5s of COD+MP work for every stale click).
  externalSearch: ({
    elements = [],
    crystalSystem = null,
    aLowA = null,
    aHighA = null,
    maxResults = 20,
    nFoldScores = null,
    dObservedA = null,
    signal = undefined,
  } = {}) =>
    api.post('/api/crystal-hint/external-search', {
      elements,
      crystal_system: crystalSystem,
      a_low_A: aLowA,
      a_high_A: aHighA,
      max_results: maxResults,
      // Optional pattern-derived hints for per-phase fit ranking.
      // Backend applies symmetry_fit + dspacing_fit and resorts.
      n_fold_scores: nFoldScores,
      d_observed_A: dObservedA,
    }, { signal }),
};

// --- WebSocket ---
export const createWebSocket = (onMessage) => {
  const wsBase = API_BASE || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
  const wsUrl = wsBase.replace(/^http/, 'ws') + '/ws';
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    // Guard against non-JSON frames (ping text, health probe, etc.) so a
    // single malformed message doesn't throw into the event loop and
    // silently break subsequent pushes.
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (err) {
      console.warn('WebSocket: ignoring non-JSON message:', event.data);
      return;
    }
    try {
      onMessage(data);
    } catch (err) {
      console.error('WebSocket onMessage handler threw:', err);
    }
  };

  ws.onopen = () => {
    ws.__opened = true;
    console.log('WebSocket connected');
  };

  // A WebSocket error event carries no detail — it is always a bare Event. The
  // only case worth reporting is a socket that was already carrying traffic and
  // then broke; a handshake that never completed (backend still starting, or
  // React's dev double-mount tearing the socket down) is routine and reconnects
  // on its own, so logging it as an error was pure noise on every page load.
  ws.onerror = () => {
    if (ws.__opened && !ws.__closingIntentionally) {
      console.warn('WebSocket connection lost — reconnecting.');
    }
  };

  return ws;
};

/**
 * Close a socket without the "closed before the connection is established"
 * warning the browser logs when you close one mid-handshake.
 */
export const closeWebSocket = (ws) => {
  if (!ws) return;
  ws.__closingIntentionally = true;
  if (ws.readyState === WebSocket.CONNECTING) {
    ws.addEventListener('open', () => ws.close(), { once: true });
    return;
  }
  if (ws.readyState === WebSocket.OPEN) ws.close();
};

// --- Reference Frame / Coordinate System ---
export const frameApi = {
  get: () => api.get('/api/frame'),
  set: (spec) => api.put('/api/frame', { spec }),
};

export const stateApi = {
  version: () => api.get('/api/state-version'),
};

export const poleFigureApi = {
  image: ({ phaseId, hkl, mode = 'both', subsample = 20000 } = {}) =>
    api.get('/api/pole-figure', { params: { phase_id: phaseId, hkl, mode, subsample } }),
};

export default api;
