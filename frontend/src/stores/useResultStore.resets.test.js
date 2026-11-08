// What each way of resetting the store clears, keeps and changes, field by field.
//
// The store has several reset paths (`clearResults`, a status answer `loaded: false`, a status answer of a swapped map,
// a load on the Data tab). Each clears a different part by hand. This test lists EVERY field against each path, so a
// field that is added later fails here until somebody has decided, for each path, whether it goes. The rules are in the
// comment above `mapBound` in useResultStore.js; the differences are on purpose:
// - clearResults wipes everything but `ipfDirection` (a display choice).
// - "loaded: false" wipes what is bound to the lost map. NOT the batch (it works on a folder), the indexing results,
//   the map choice of the page, nor the chain's busy flag.
// - a swapped map and a load on the Data tab replace the map-bound state by the new map's. A load clears the
//   reconstruction task id (the backend's run for the replaced map fails by itself), a swap takes it from the answer.
import { describe, it, expect, beforeEach } from 'vitest';
import useResultStore, { EMPTY_BATCH_RUN, EMPTY_EXPORT_RUN } from './useResultStore';
import { DEFAULT_GRAIN_STATS_SETTINGS } from '../components/Analysis/grainStats';
import { DEFAULT_RX_CRITERIA } from '../components/Analysis/rxCriteria';
import { DEFAULT_DEFORMATION_PARAMS } from '../components/Analysis/deformationSettings';

const dataOf = (s) => Object.fromEntries(Object.entries(s).filter(([, v]) => typeof v !== 'function'));
const INITIAL = dataOf(useResultStore.getState());       // as the module made it
const state = () => useResultStore.getState();

/** A value for every field that is not its initial one. A new field has no entry here and fails the first test. */
const DIRTY = {
  indexingResult: { result_id: 'r1' },
  indexingMethod: 'hough',
  resultsList: [{ id: 'r1' }],
  phaseMapImage: 'img',
  ipfDirection: 'X',
  analysisLoaded: true,
  analysisShape: [4, 5],
  hasGrains: true,
  phaseOptions: [{ id: 9, name: 'Zed' }],
  datasetToken: 'tok',
  hasNativeBc: true,
  sourceName: 'dirty.h5oina',
  stepSize: 0.25,
  texturePhase: 3,
  texturePreset: 'BCC_Rolling',
  textureTolerance: '4.5',
  textureEdited: true,
  grainsVersion: 7,
  mapEpoch: 4,
  reconEdited: true,
  grainStatsSettings: { ...DEFAULT_GRAIN_STATS_SETTINGS, gbThreshold: '9.5', edgeRule: 'keep_all', ecdBasis: 'measured_pixels', statsPhase: 9 },
  rxCriteria: { ...DEFAULT_RX_CRITERIA, gosThreshold: '3.5', minRadius: '2' },
  rxCriteriaEdited: true,
  rxPhase: 3,
  deformationParams: { ...DEFAULT_DEFORMATION_PARAMS, kamOrder: '3', hagbMin: '20' },
  deformationEdited: true,
  grainStats: { count: 5 },
  grainSizeEpoch: 8,
  runAllBusy: true,
  grainReconBusy: true,
  reconTaskId: 't7',
  reconProgress: 0.5,
  reconOutcome: { error: true, text: 'x' },
  batchRun: { ...EMPTY_BATCH_RUN, taskId: 'b1', running: true, msg: 'm' },
  exportRun: { ...EMPTY_EXPORT_RUN, msg: 'e', warnings: ['w'] },
  batchStatsSettings: { edgeRule: 'keep_all', ecdBasis: 'measured_pixels', statsPhase: 'most_measured' },
  currentMapType: 'ipf',
  currentMapImage: 'map',
};

const SWAPPED = {
  loaded: true, shape: [40, 60], has_grains: false, phase_options: [{ id: 1, name: 'Al' }], dataset_token: 'new',
  has_native_bc: false, source_name: 'new.ctf', step_size: 0.5,
  edge_rule: 'matlab_script', ecd_basis: 'cell_area', stats_phase: null, recon_params: null, reconstruct: null,
};

const ALL = Object.keys(DIRTY);
const without = (...lists) => ALL.filter((k) => !lists.flat().includes(k));

const PATHS = {
  clearResults: {
    run: () => state().clearResults(),
    cleared: without(['ipfDirection', 'mapEpoch']),
    changed: ['mapEpoch'],
    kept: ['ipfDirection'],
  },
  'status loaded: false': {
    run: () => state().applyBackendStatus({ loaded: false }),
    cleared: [
      'analysisLoaded', 'analysisShape', 'hasGrains', 'phaseOptions', 'datasetToken', 'hasNativeBc', 'sourceName', 'stepSize',
      'texturePhase', 'texturePreset', 'textureTolerance', 'textureEdited', 'grainStats', 'grainReconBusy', 'reconTaskId', 'reconProgress', 'reconOutcome', 'exportRun', 'rxCriteria', 'rxCriteriaEdited',
      'rxPhase', 'deformationParams', 'deformationEdited', 'reconEdited',
    ],
    changed: ['grainsVersion', 'grainSizeEpoch', 'grainStatsSettings', 'mapEpoch'],
    kept: [
      'indexingResult', 'indexingMethod', 'resultsList', 'phaseMapImage', 'ipfDirection', 'runAllBusy', 'batchRun',
      'batchStatsSettings', 'currentMapType', 'currentMapImage',
    ],
  },
  'status of a swapped map': {
    run: () => state().applyBackendStatus(SWAPPED),
    cleared: [
      'grainStats', 'exportRun', 'reconOutcome', 'reconProgress', 'reconTaskId', 'grainReconBusy', 'rxCriteria',
      'rxCriteriaEdited', 'texturePhase', 'texturePreset', 'textureTolerance', 'textureEdited', 'rxPhase', 'deformationParams',
      'deformationEdited', 'reconEdited',
    ],
    changed: [
      'analysisShape', 'hasGrains', 'phaseOptions', 'datasetToken', 'hasNativeBc', 'sourceName', 'stepSize', 'grainsVersion',
      'grainStatsSettings', 'grainSizeEpoch', 'mapEpoch',
    ],
    kept: [
      'analysisLoaded', 'indexingResult', 'indexingMethod', 'resultsList', 'phaseMapImage', 'ipfDirection', 'runAllBusy',
      'batchRun', 'batchStatsSettings', 'currentMapType', 'currentMapImage',
    ],
  },
  'load on the Data tab': {
    run: () => {
      state().setAnalysis({
        shape: [40, 60], has_grains: false, phase_options: [{ id: 1, name: 'Al' }], has_native_bc: false, source_name: 'new.ctf', step_size: 0.5,
      });
      state().resetGrainStatsSettings();
      state().bumpGrainMaps();
    },
    cleared: [
      'datasetToken', 'grainStats', 'exportRun', 'reconOutcome', 'reconProgress', 'reconTaskId', 'grainReconBusy',
      'rxCriteria', 'rxCriteriaEdited', 'texturePhase', 'texturePreset', 'textureTolerance', 'textureEdited', 'rxPhase',
      'deformationParams', 'deformationEdited', 'reconEdited',
    ],
    changed: [
      'analysisShape', 'hasGrains', 'phaseOptions', 'hasNativeBc', 'sourceName', 'stepSize', 'grainsVersion', 'grainStatsSettings',
      'grainSizeEpoch', 'mapEpoch',
    ],
    kept: [
      'analysisLoaded', 'runAllBusy', 'batchRun', 'batchStatsSettings', 'indexingResult', 'indexingMethod',
      'resultsList', 'phaseMapImage', 'ipfDirection', 'currentMapType', 'currentMapImage',
    ],
  },
};

describe('the store: every field is classified for every reset path', () => {
  beforeEach(() => {
    state().clearResults();
    useResultStore.setState({ ...DIRTY });
  });

  it('the test knows every field (a new field fails here until it has a value and a place in each path)', () => {
    expect(Object.keys(dataOf(state())).sort()).toEqual([...ALL].sort());
    expect(Object.keys(INITIAL).sort()).toEqual([...ALL].sort());
  });

  it('the dirty state differs from the initial one in every field (the classification below can tell them apart)', () => {
    ALL.forEach((k) => expect(DIRTY[k], k).not.toEqual(INITIAL[k]));
  });

  Object.entries(PATHS).forEach(([name, path]) => {
    describe(name, () => {
      it('lists every field exactly once', () => {
        const listed = [...path.cleared, ...path.changed, ...path.kept];
        expect([...listed].sort()).toEqual([...ALL].sort());
      });

      it('clears, changes and keeps exactly what it says', () => {
        path.run();
        const now = dataOf(state());
        path.cleared.forEach((k) => expect(now[k], `${k} should be cleared`).toEqual(INITIAL[k]));
        path.changed.forEach((k) => expect(now[k], `${k} should be changed`).not.toEqual(DIRTY[k]));
        path.kept.forEach((k) => expect(now[k], `${k} should be kept`).toEqual(DIRTY[k]));
      });
    });
  });
});
