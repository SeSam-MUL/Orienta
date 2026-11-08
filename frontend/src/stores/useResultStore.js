/**
 * Results store - tracks indexing results, phase maps, grain data.
 */

import { formatTime } from '../i18n/formatDateTime';
import { create } from 'zustand';
import {
  DEFAULT_GRAIN_STATS_SETTINGS, DEFAULT_STATS_FIELDS, RECON_FIELDS, settingsFromStatus, samePhases, batchStatsFromGrain,
} from '../components/Analysis/grainStats';
import { DEFAULT_RX_CRITERIA, DEFAULT_RX_PHASE, rxCriteriaFromBackend, rxPhaseOf } from '../components/Analysis/rxCriteria';
import { DEFAULT_DEFORMATION_PARAMS, deformationFromBackend } from '../components/Analysis/deformationSettings';
import {
  DEFAULT_TEXTURE_PRESET, DEFAULT_TEXTURE_TOLERANCE, texturePresetOf, textureToleranceFromBackend,
} from '../components/Analysis/textureSettings';

/** What the Batch & Export tab shows about its batch and its Excel export; they live here so that they survive leaving the tab. */
export const EMPTY_BATCH_RUN = Object.freeze({
  taskId: null, running: false, progress: null, log: '', msg: null, error: false, warn: false, result: null,
  settings: null,      // what the batch ran with: { edgeRule, ecdBasis, statsPhase, recon, rx, rxPending, deformation }
  currentFile: null,   // the file the running batch is at (the backend's "Processing file i/n: name"), else null
});
export const EMPTY_EXPORT_RUN = Object.freeze({
  taskId: null, running: false, path: null, msg: null, error: false, warn: false, warnings: [], renamed: null,
  rxApplied: null,     // the RX criteria the export sent before it ran ({gos, gbc, gkam, radius}), else null
  textureApplied: null,       // the texture choice it sent ({preset, tolerance, phase}), else null
  deformationApplied: null,   // the Deformation parameters it sent ({order, threshold, hagb}), else null
});

/**
 * The four ways the analysis part of the store is reset, and what each leaves alone (pinned field by field in
 * useResultStore.resets.test.js, so a new field has to be classified there):
 * - `clearResults`: everything, except the display choice `ipfDirection`.
 * - `applyBackendStatus({loaded: false})`: what is bound to the map that is gone (the map, the grains, their
 *   statistics, the phases, the reconstruction task, the export). A batch is NOT bound to the map (it works
 *   on a folder), nor are the indexing results, the page's map choice or the chain's busy flag.
 * - `applyBackendStatus` of a swapped map (another `dataset_token` / phase list): the same map-bound state is
 *   replaced by what the answer says; the settings are back at the defaults.
 * - a load on the Data tab (`setAnalysis` + `resetGrainStatsSettings` + `bumpGrainMaps`): as a swap; the
 *   reconstruction task id goes too (the backend's run for the replaced map fails by itself).
 * The RX criteria (`rxCriteria`, typed as strings, and `rxCriteriaEdited`, typed but not yet sent with "Classify")
 * are the backend's per-map settings like the statistics settings: every path but `applyBackendStatus` of the
 * same map puts them back at the defaults (the backend resets them on a load or a file switch). A status answer of
 * the same map takes the backend's values unless the user has typed since the last "Classify" (the page asks on
 * every tab change, and a typed value must not be taken back).
 * `rxPhase` (the phase the RX analysis works on, 'most_measured' or a phase id) is chosen with the criteria and travels with
 * them: it is pending (`rxCriteriaEdited`) until "Classify" sends it, and a status answer (`rx_phase`) is adopted when
 * nothing is pending. `deformationParams` (KAM order, KAM threshold, HAGB limit, typed as strings) and
 * `deformationEdited` work the same way with "Analyze Deformation" and the status answer's `deformation_params`.
 * `reconEdited`: a reconstruction parameter of the Grain tab was typed since the last reconstruction; the status answer's
 * `recon_params` are the old grains' and are not taken then. A reconstruction (its end) clears it.
 * `mapEpoch` counts the times the map the page shows was replaced or lost (a load, a swapped map, `loaded: false`, a
 * clear): the tabs that keep results of their own in component state are keyed on it, so the results of the old map go.
 * The map-bound extras that a new map clears (`mapBound`): statistics, export run, the reconstruction outcome.
 * `texturePhase` (the Texture tab's phase choice, 'most_measured' or a phase id of the map), `texturePreset` and
 * `textureTolerance` (the typed tolerance angle) are per-map like the statistics phase: every path but a status answer
 * of the same map puts them back at the defaults. A status answer that carries the backend's choice (`texture_phase`,
 * `texture_preset`, `texture_tolerance_deg`) is adopted, unless a pick is pending (`textureEdited`: chosen or typed on the
 * tab and not applied by a run yet) or the phase id is not a phase of the map.
 */
const DEFAULT_TEXTURE_PHASE = 'most_measured';
const TEXTURE_DEFAULTS = {
  texturePhase: DEFAULT_TEXTURE_PHASE,
  texturePreset: DEFAULT_TEXTURE_PRESET,
  textureTolerance: DEFAULT_TEXTURE_TOLERANCE,
  textureEdited: false,
};
const textureChoiceOf = (value, phaseOptions) => (
  Number.isInteger(value) && (phaseOptions || []).some((p) => p.id === value) ? value : DEFAULT_TEXTURE_PHASE
);

// The RX criteria with their phase, and the Deformation parameters: what every path that replaces the map puts back.
const RX_DEFAULTS = {
  rxCriteria: { ...DEFAULT_RX_CRITERIA },
  rxPhase: DEFAULT_RX_PHASE,
  rxCriteriaEdited: false,
};
const DEFORMATION_DEFAULTS = {
  deformationParams: { ...DEFAULT_DEFORMATION_PARAMS },
  deformationEdited: false,
};

const mapBound = () => ({
  grainStats: null,
  exportRun: { ...EMPTY_EXPORT_RUN },
  reconProgress: null,
  reconOutcome: null,
});

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
  // The phases of the loaded map as [{id, name}] (from /load and /status): the phase select offers them.
  phaseOptions: [],
  // Changes exactly when the backend's map is replaced (GET /status `dataset_token`); null = not known yet.
  datasetToken: null,
  // Facts of the loaded map from /load and /status: whether it has band contrast of its own (null = not known),
  // and the name of its file without the folder (null = not known, or not from a file). Per-map like the phases.
  hasNativeBc: null,
  sourceName: null,
  // The step of the loaded map in micrometres (null = not known).
  stepSize: null,
  // The Texture tab's phase: 'most_measured' (the phase with the most indexed pixels) or a phase id of the map.
  texturePhase: DEFAULT_TEXTURE_PHASE,
  texturePreset: DEFAULT_TEXTURE_PRESET,
  textureTolerance: DEFAULT_TEXTURE_TOLERANCE,
  // A phase, preset or tolerance was chosen or typed on the tab and no run has applied it yet: a status answer must not
  // take it back (as `rxCriteriaEdited` does for the RX criteria). Cleared by a run, and by anything that replaces the map.
  textureEdited: false,
  // Raised whenever the grains or the basis of a grain map change; every map viewer fetches again.
  grainsVersion: 0,
  // Raised whenever the map the page shows is replaced or lost: the results kept in the tabs belong to the old one.
  mapEpoch: 0,
  // A reconstruction parameter of the Grain tab was typed and no reconstruction has used it yet.
  reconEdited: false,
  // The Grain tab's reconstruction parameters and what the backend uses for the grain-size
  // statistics, the ECD map and the Excel export. One object that the tab and "Run Complete
  // Analysis" both read; it lives here and not in the tab because the backend keeps the statistics
  // settings after the tab is left. Further settings (the phase) join it.
  grainStatsSettings: { ...DEFAULT_GRAIN_STATS_SETTINGS },
  // The Batch tab's own edge rule, ECD basis and phase ({edgeRule, ecdBasis, statsPhase: null | 'most_measured'}).
  // null until the batch tab is used or changed: then it starts as the Grain tab's settings (a single phase there
  // becomes all phases here). It stays when the tab is left and is not reset per map.
  batchStatsSettings: null,
  // The four criteria of the Recrystallization tab as typed ({gosThreshold, gbcFraction, gkamThreshold,
  // minRadius}); the tab, "Run Complete Analysis" and the Batch tab all read them from here. `rxCriteriaEdited`:
  // typed since the backend last said (or was told) which criteria it holds.
  rxCriteria: { ...DEFAULT_RX_CRITERIA },
  rxCriteriaEdited: false,
  // The phase of the RX analysis ('most_measured' or a phase id of the map); pending with the criteria (rxCriteriaEdited).
  rxPhase: DEFAULT_RX_PHASE,
  // The three parameters of the Deformation tab as typed ({kamOrder, kamThreshold, hagbMin}); `deformationEdited`: typed since
  // the backend last said (or was told) which parameters it holds.
  deformationParams: { ...DEFAULT_DEFORMATION_PARAMS },
  deformationEdited: false,
  // The statistics block of the Grain tab (what grainStatsFrom reads from the backend's answers). It
  // lives here so that it survives leaving the tab and so that "Run Complete Analysis" can fill it.
  grainStats: null,
  // Raised whenever a reconstruction starts (from the tab or the chain) or a map is loaded: a
  // /grain-size answer asked for before that belongs to other grains and is dropped.
  grainSizeEpoch: 0,
  // "Run Complete Analysis" is running: the Grain tab treats it like its own reconstruction.
  runAllBusy: false,
  // The Grain tab's own reconstruction is running ("Run Complete Analysis" waits for it, as the tab waits for the chain).
  grainReconBusy: false,
  // The id of the reconstruction task the backend is running (the Grain tab polls it, and picks it up again
  // when it is opened after having been left); null = none.
  reconTaskId: null,
  // How far that task is, and how the last one ended ({error, text}); shown by the Grain tab, written by the page's watch.
  reconProgress: null,
  reconOutcome: null,
  batchRun: { ...EMPTY_BATCH_RUN },
  exportRun: { ...EMPTY_EXPORT_RUN },
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
      timestamp: formatTime(),
      result_id: result.result_id || null,
      scan_geometry: result.scan_geometry ?? undefined,
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
    phaseOptions: Array.isArray(data.phase_options) ? data.phase_options : [],
    datasetToken: data.dataset_token ?? null,     // unknown: the next /status answer is adopted, not taken as a swap
    hasNativeBc: typeof data.has_native_bc === 'boolean' ? data.has_native_bc : null,
    sourceName: typeof data.source_name === 'string' && data.source_name ? data.source_name : null,
    stepSize: Number.isFinite(data.step_size) ? data.step_size : null,
  }),

  // The page asks the backend what it holds (GET /status) when it becomes active: a reconstruction may
  // have finished while the Grain tab was away, and a map may have been loaded or switched on another page.
  // One update. When the grains appeared or went, or the map is another one, the maps drawn from them are
  // fetched again (version), answers still on their way are dropped (epoch) and the statistics on screen,
  // which belong to the earlier grains, go. With no grains there are no statistics.
  // The backend lost its map (`loaded: false`, e.g. it was restarted): nothing the page shows is true any more.
  // A map swapped for another one (another `dataset_token`, or another list of phases) is a new map: the
  // statistics settings are back at their defaults (the backend resets them on a load) and a phase id of
  // the old map means nothing.
  applyBackendStatus: (d) => set((state) => {
    if (!d.loaded) {
      if (!state.analysisLoaded && !state.hasGrains) return {};     // nothing was known, nothing is lost
      return {
        analysisLoaded: false,
        analysisShape: null,
        hasGrains: false,
        phaseOptions: [],
        datasetToken: null,
        hasNativeBc: null,
        sourceName: null,
        stepSize: null,
        ...TEXTURE_DEFAULTS,
        ...mapBound(),
        grainStatsSettings: { ...state.grainStatsSettings, statsPhase: null },
        reconEdited: false,
        ...RX_DEFAULTS,
        ...DEFORMATION_DEFAULTS,
        reconTaskId: null,
        grainReconBusy: false,
        grainsVersion: state.grainsVersion + 1,
        grainSizeEpoch: state.grainSizeEpoch + 1,
        mapEpoch: state.mapEpoch + 1,
      };
    }
    const hasGrains = !!d.has_grains;
    const shape = d.shape ?? null;
    const sameShape = !state.analysisShape || !shape
      || (state.analysisShape.length === shape.length && state.analysisShape.every((n, i) => n === shape[i]));
    const token = d.dataset_token ?? null;
    const swapped = token !== null && state.datasetToken !== null && token !== state.datasetToken;
    const phaseOptions = Array.isArray(d.phase_options) ? d.phase_options : state.phaseOptions;
    const otherPhases = state.phaseOptions.length > 0 && !samePhases(state.phaseOptions, phaseOptions);
    const newMap = swapped || otherPhases;
    const changed = hasGrains !== state.hasGrains || !sameShape || newMap;
    const base = newMap ? { ...state.grainStatsSettings, ...DEFAULT_STATS_FIELDS } : state.grainStatsSettings;
    const rxEdited = newMap ? false : state.rxCriteriaEdited;
    const rxBase = newMap ? { ...DEFAULT_RX_CRITERIA } : state.rxCriteria;
    const rxCriteria = rxEdited ? rxBase : rxCriteriaFromBackend(rxBase, d);
    // the RX phase: a pick not yet sent stays; otherwise the backend's remembered choice (an answer without it keeps ours)
    const rxPhase = newMap ? DEFAULT_RX_PHASE
      : (rxEdited || !('rx_phase' in d) ? rxPhaseOf(state.rxPhase, phaseOptions) : rxPhaseOf(d.rx_phase, phaseOptions));
    const defEdited = newMap ? false : state.deformationEdited;
    const defBase = newMap ? { ...DEFAULT_DEFORMATION_PARAMS } : state.deformationParams;
    const deformationParams = defEdited ? defBase : deformationFromBackend(defBase, d.deformation_params);
    const reconEdited = newMap ? false : state.reconEdited;
    const next = {
      analysisLoaded: true,
      analysisShape: shape,
      hasGrains,
      phaseOptions,
      datasetToken: token ?? state.datasetToken,
      // what the answer says about the map; an answer without it keeps what is known, unless the map is another one
      hasNativeBc: typeof d.has_native_bc === 'boolean' ? d.has_native_bc : (newMap ? null : state.hasNativeBc),
      sourceName: 'source_name' in d ? (d.source_name || null) : (newMap ? null : state.sourceName),
      stepSize: Number.isFinite(d.step_size) ? d.step_size : (newMap ? null : state.stepSize),
      // a pick not yet run stays; otherwise the backend's remembered choice (the real answer always carries it)
      texturePhase: (!newMap && state.textureEdited) || !('texture_phase' in d)
        ? (newMap ? DEFAULT_TEXTURE_PHASE : textureChoiceOf(state.texturePhase, phaseOptions))
        : textureChoiceOf(d.texture_phase, phaseOptions),
      texturePreset: (!newMap && state.textureEdited) || !('texture_preset' in d)
        ? (newMap ? DEFAULT_TEXTURE_PRESET : state.texturePreset)
        : texturePresetOf(d.texture_preset),
      textureTolerance: (!newMap && state.textureEdited) || !('texture_tolerance_deg' in d)
        ? (newMap ? DEFAULT_TEXTURE_TOLERANCE : state.textureTolerance)
        : textureToleranceFromBackend(state.textureTolerance, d.texture_tolerance_deg),
      textureEdited: newMap ? false : state.textureEdited,
      grainStatsSettings: settingsFromStatus(base, d, phaseOptions, reconEdited),
      reconEdited,
      rxPhase,
      ...(deformationParams !== state.deformationParams || defEdited !== state.deformationEdited
        ? { deformationParams, deformationEdited: defEdited } : {}),
      ...(rxCriteria !== state.rxCriteria || rxEdited !== state.rxCriteriaEdited ? { rxCriteria, rxCriteriaEdited: rxEdited } : {}),
      ...(!hasGrains || changed ? { grainStats: null } : {}),
      ...(newMap ? { ...mapBound(), mapEpoch: state.mapEpoch + 1 } : {}),
      ...(changed ? { grainsVersion: state.grainsVersion + 1, grainSizeEpoch: state.grainSizeEpoch + 1 } : {}),
    };
    if ('reconstruct' in d) {
      // the reconstruction the backend is running; a task this page knew that the backend no longer lists is over
      const running = d.reconstruct?.task_id ?? null;
      next.reconTaskId = running;
      next.grainReconBusy = running !== null ? true : (state.reconTaskId !== null ? false : state.grainReconBusy);
    }
    return next;
  }),

  // Grains exist now (or are new): hasGrains and the version move in ONE update, so a grain layer
  // is requested once, not once for each.
  markGrainsReconstructed: () => set((state) => ({ hasGrains: true, reconEdited: false, grainsVersion: state.grainsVersion + 1 })),

  // The Texture tab's phase choice ('most_measured' or a phase id).
  setTexturePhase: (texturePhase) => set({ texturePhase, textureEdited: true }),
  setTexturePreset: (texturePreset) => set({ texturePreset, textureEdited: true }),
  // `textureTolerance` is the text of the field (it may be empty or half typed while the user types)
  setTextureTolerance: (textureTolerance) => set({ textureTolerance, textureEdited: true }),

  // A texture run was sent with this choice and the backend remembers it now: the pick is no longer pending, unless
  // the user has picked something else meanwhile.
  // `sent` = { phase, preset, tolerance } as the request carried them (tolerance: the text of the field).
  markTextureApplied: (sent) => set((state) => (
    state.texturePhase === sent.phase && state.texturePreset === sent.preset && state.textureTolerance === sent.tolerance
      ? { textureEdited: false } : {}
  )),

  // The grains are unchanged but the maps drawn from them are not (the ECD basis changed).
  setGrainStats: (grainStats) => set({ grainStats }),

  // A reconstruction starts: the statistics of the old grains go, and so does any answer still on its way.
  beginReconstruction: () => set((state) => ({
    grainStats: null, reconOutcome: null, grainSizeEpoch: state.grainSizeEpoch + 1,
  })),

  setReconProgress: (reconProgress) => set({ reconProgress }),

  // The page's watch saw the reconstruction task finish: the grains exist (flag and version in one update),
  // the statistics are the task's, the task is over. Or it failed: only the outcome says so.
  finishReconstruction: ({ stats, text }) => set((state) => ({
    grainStats: stats, hasGrains: true, reconEdited: false, grainsVersion: state.grainsVersion + 1,
    grainSizeEpoch: state.grainSizeEpoch + 1, reconTaskId: null, grainReconBusy: false, reconProgress: null, reconOutcome: { error: false, text },
  })),
  failReconstruction: (text) => set({
    reconTaskId: null, grainReconBusy: false, reconProgress: null, reconOutcome: { error: true, text },
  }),

  setRunAllBusy: (runAllBusy) => set({ runAllBusy }),

  setGrainReconBusy: (grainReconBusy) => set({ grainReconBusy }),

  // The reconstruction task that is running (null: none). The busy flag follows it.
  // It raises the epoch: a /status answer asked before (and applied after) must not take the id away again.
  setReconTaskId: (reconTaskId) => set((state) => ({
    reconTaskId, grainReconBusy: reconTaskId !== null, reconProgress: reconTaskId !== null ? 0 : null,
    grainSizeEpoch: state.grainSizeEpoch + 1,
  })),

  // Patch (or function of the previous run) of what the Batch tab shows about its batch / export.
  setBatchRun: (patch) => set((state) => ({
    batchRun: { ...state.batchRun, ...(typeof patch === 'function' ? patch(state.batchRun) : patch) },
  })),
  setExportRun: (patch) => set((state) => ({
    exportRun: { ...state.exportRun, ...(typeof patch === 'function' ? patch(state.exportRun) : patch) },
  })),

  bumpGrainMaps: () => set((state) => ({ grainsVersion: state.grainsVersion + 1 })),

  // A change of a reconstruction parameter is a pending choice (`reconEdited`): the status answer must not take it back.
  setGrainStatsSettings: (patch) => set((state) => ({
    grainStatsSettings: { ...state.grainStatsSettings, ...patch },
    ...(RECON_FIELDS.some((field) => field in patch) ? { reconEdited: true } : {}),
  })),

  // A change on the Batch tab. The first change (or the first use of the tab) starts from the Grain tab's settings.
  setBatchStatsSettings: (patch) => set((state) => ({
    batchStatsSettings: { ...(state.batchStatsSettings ?? batchStatsFromGrain(state.grainStatsSettings)), ...patch },
  })),
  // The tab was opened: the Grain tab's settings are copied once, never again (so they cannot move under the batch).
  ensureBatchStatsSettings: () => set((state) => (
    state.batchStatsSettings ? {} : { batchStatsSettings: batchStatsFromGrain(state.grainStatsSettings) }
  )),

  // The user typed an RX criterion (patch of the four typed fields): the backend does not hold it yet.
  setRxCriteria: (patch) => set((state) => ({
    rxCriteria: { ...state.rxCriteria, ...patch },
    rxCriteriaEdited: true,
  })),

  // The phase of the RX analysis. It is sent with the criteria, so it is pending with them.
  setRxPhase: (rxPhase) => set({ rxPhase, rxCriteriaEdited: true }),

  // The user typed a Deformation parameter (patch of the three typed fields): the backend does not hold it yet.
  setDeformationParams: (patch) => set((state) => ({
    deformationParams: { ...state.deformationParams, ...patch },
    deformationEdited: true,
  })),

  // POST /api/analysis/deformation answered with `deformation_params`: the backend holds them now, so what is typed is
  // no longer pending.
  adoptDeformationUsed: (held) => set((state) => ({
    deformationParams: deformationFromBackend(state.deformationParams, held),
    deformationEdited: false,
  })),

  // POST /api/analysis/rx answered: the backend holds the criteria it says it used (the fields of the answer),
  // so what is typed is no longer pending.
  adoptRxCriteriaUsed: (answer) => set((state) => ({
    rxCriteria: rxCriteriaFromBackend(state.rxCriteria, answer),
    rxCriteriaEdited: false,
  })),

  // A new map was loaded: the backend's statistics settings are back at their defaults and the
  // statistics on screen belong to another map. The reconstruction parameters are the user's and stay.
  resetGrainStatsSettings: () => set((state) => ({
    grainStatsSettings: { ...state.grainStatsSettings, ...DEFAULT_STATS_FIELDS },
    reconEdited: false,
    ...TEXTURE_DEFAULTS,
    ...RX_DEFAULTS,
    ...DEFORMATION_DEFAULTS,
    mapEpoch: state.mapEpoch + 1,
    ...mapBound(),
    // the backend drops a run for the map that was replaced (its task fails by itself): the id goes now
    reconTaskId: null,
    grainReconBusy: false,
    grainSizeEpoch: state.grainSizeEpoch + 1,
  })),

  setCurrentMap: (type, image) => set({
    currentMapType: type,
    currentMapImage: image,
  }),

  clearResults: () => set((state) => ({
    indexingResult: null,
    indexingMethod: null,
    resultsList: [],
    phaseMapImage: null,
    analysisLoaded: false,
    analysisShape: null,
    hasGrains: false,
    phaseOptions: [],
    datasetToken: null,
    hasNativeBc: null,
    sourceName: null,
    stepSize: null,
    ...TEXTURE_DEFAULTS,
    grainsVersion: 0,
    mapEpoch: state.mapEpoch + 1,
    reconEdited: false,
    grainStatsSettings: { ...DEFAULT_GRAIN_STATS_SETTINGS },
    ...RX_DEFAULTS,
    ...DEFORMATION_DEFAULTS,
    grainStats: null,
    grainSizeEpoch: 0,
    runAllBusy: false,
    grainReconBusy: false,
    reconTaskId: null,
    reconProgress: null,
    reconOutcome: null,
    batchRun: { ...EMPTY_BATCH_RUN },
    exportRun: { ...EMPTY_EXPORT_RUN },
    batchStatsSettings: null,
    currentMapType: 'bc',
    currentMapImage: null,
  })),
}));

export default useResultStore;
