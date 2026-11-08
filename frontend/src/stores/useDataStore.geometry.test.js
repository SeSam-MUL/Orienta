// @vitest-environment jsdom
/**
 * `gridShape` has three writers in the store (setFileData, setNavigationGrid, syncFromBackend) and
 * the geometry of the scan has to be written by the same call as the grid, or a crop of a
 * hexagonal scan shows its own grid next to its parent's row offset.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

const info = vi.fn();
const getMetadata = vi.fn();
vi.mock('../services/api', () => ({
  ebsdApi: { info: (...a) => info(...a), getMetadata: (...a) => getMetadata(...a) },
  isFileLoadInFlight: () => false,
}));

import useDataStore from './useDataStore';
import { parseScanGeometry } from '../services/scanGeometry';

const hex = (rows, cols, shift) => ({
  kind: 'hex', n_rows: rows, n_cols: cols, step_x: 0.5, step_y: 0.4330127, steps_known: true,
  odd_row_shift: shift, shift_confirmed: false,
});
const PARENT = hex(6, 5, 0.5);
const CROP = hex(3, 3, -0.5);          // a crop that starts on an odd row: the offset reverses

/** The invariant: the geometry describes the grid the store holds. */
function expectInStep() {
  const s = useDataStore.getState();
  expect(s.scanGeometry).not.toBeNull();
  expect(s.scanGeometry.gridShape).toEqual(s.gridShape);
}

beforeEach(() => {
  info.mockReset();
  getMetadata.mockReset();
  getMetadata.mockResolvedValue({ data: {} });
  useDataStore.getState().clearFile();
});

describe('the three writers of gridShape keep the geometry in step', () => {
  it('setFileData writes both from the load answer', () => {
    useDataStore.getState().setFileData({
      file_path: '/d/scan.up1', grid_shape: [6, 5], scan_geometry: PARENT, pattern_count: 30,
    });
    expectInStep();
    expect(useDataStore.getState().scanGeometry.oddRowShift).toBe(0.5);
  });

  it('setNavigationGrid replaces the geometry with the new dataset\'s, never keeps the parent\'s', () => {
    useDataStore.getState().setFileData({ file_path: '/d/scan.up1', grid_shape: [6, 5], scan_geometry: PARENT });
    useDataStore.getState().setNavigationGrid({
      gridShape: [3, 3], patternShape: [8, 8], patternCount: 9, scanGeometry: parseScanGeometry(CROP),
    });
    expectInStep();
    expect(useDataStore.getState().scanGeometry.oddRowShift).toBe(-0.5);
  });

  it('setNavigationGrid without a geometry leaves "unknown", not the parent\'s', () => {
    useDataStore.getState().setFileData({ file_path: '/d/scan.up1', grid_shape: [6, 5], scan_geometry: PARENT });
    useDataStore.getState().setNavigationGrid({ gridShape: [3, 3], patternShape: [8, 8], patternCount: 9 });
    expect(useDataStore.getState().scanGeometry).toBeNull();
  });

  it('syncFromBackend writes both from /info', async () => {
    info.mockResolvedValue({ data: {
      loaded: true, file_path: '/d/scan.up1', data_shape: [6, 5, 8, 8], scan_geometry: PARENT } });
    expect(await useDataStore.getState().syncFromBackend()).toBe(true);
    expectInStep();
  });

  it('a square file written by the same paths is square', () => {
    useDataStore.getState().setFileData({
      file_path: '/d/a.h5oina', grid_shape: [2, 3],
      scan_geometry: { kind: 'square', n_rows: 2, n_cols: 3, step_x: 1, step_y: 1 } });
    expect(useDataStore.getState().scanGeometry.kind).toBe('square');
  });

  it('a load answer without a geometry (the HDF5 viewer, an old backend) is unknown', () => {
    useDataStore.getState().setFileData({ file_path: '/d/a.h5', grid_shape: [2, 3] });
    expect(useDataStore.getState().scanGeometry).toBeNull();
  });

  it('clearFile forgets it', () => {
    useDataStore.getState().setFileData({ file_path: '/d/scan.up1', grid_shape: [6, 5], scan_geometry: PARENT });
    useDataStore.getState().clearFile();
    expect(useDataStore.getState().scanGeometry).toBeNull();
  });
});

// "Resampled from a hexagonal scan": an OIM-form hexagonal file is held as a SQUARE map (its
// geometry is honestly square), and the backend says so with `hex_resampled`. The flag follows the
// grid through the same three writers as the geometry; null = the backend did not say.
describe('the resampled flag follows the grid through the same writers', () => {
  const SQUARE = { kind: 'square', n_rows: 4, n_cols: 3, step_x: 1, step_y: 1 };

  it('setFileData writes it from the load answer', () => {
    useDataStore.getState().setFileData({
      file_path: '/d/a.h5', grid_shape: [4, 3], scan_geometry: SQUARE, hex_resampled: true });
    expect(useDataStore.getState().hexResampled).toBe(true);
    useDataStore.getState().setFileData({
      file_path: '/d/b.h5', grid_shape: [4, 3], scan_geometry: SQUARE, hex_resampled: false });
    expect(useDataStore.getState().hexResampled).toBe(false);
  });

  it('a load answer without the flag leaves "unknown", not false', () => {
    useDataStore.getState().setFileData({ file_path: '/d/a.h5', grid_shape: [4, 3], scan_geometry: SQUARE });
    expect(useDataStore.getState().hexResampled).toBeNull();
  });

  it('setNavigationGrid replaces it with the new dataset\'s, never keeps the parent\'s', () => {
    useDataStore.getState().setFileData({
      file_path: '/d/a.h5', grid_shape: [4, 3], scan_geometry: SQUARE, hex_resampled: true });
    useDataStore.getState().setNavigationGrid({
      gridShape: [2, 2], patternShape: [8, 8], patternCount: 4,
      scanGeometry: parseScanGeometry({ ...SQUARE, n_rows: 2, n_cols: 2 }), hexResampled: false });
    expect(useDataStore.getState().hexResampled).toBe(false);
  });

  it('setNavigationGrid without the flag leaves "unknown", not the parent\'s', () => {
    useDataStore.getState().setFileData({
      file_path: '/d/a.h5', grid_shape: [4, 3], scan_geometry: SQUARE, hex_resampled: true });
    useDataStore.getState().setNavigationGrid({ gridShape: [2, 2], patternShape: [8, 8], patternCount: 4 });
    expect(useDataStore.getState().hexResampled).toBeNull();
  });

  it('syncFromBackend writes it from /info', async () => {
    info.mockResolvedValue({ data: {
      loaded: true, file_path: '/d/a.h5', data_shape: [4, 3, 8, 8], scan_geometry: SQUARE,
      hex_resampled: true } });
    expect(await useDataStore.getState().syncFromBackend()).toBe(true);
    expect(useDataStore.getState().hexResampled).toBe(true);
  });

  it('clearFile forgets it', () => {
    useDataStore.getState().setFileData({
      file_path: '/d/a.h5', grid_shape: [4, 3], scan_geometry: SQUARE, hex_resampled: true });
    useDataStore.getState().clearFile();
    expect(useDataStore.getState().hexResampled).toBeNull();
  });
});
