import { describe, it, expect } from 'vitest';
import { datasetGeometry, ebsdInfoWithGrid } from './datasetGeometry';

const PARENT = {
  kind: 'hex', n_rows: 6, n_cols: 5, step_x: 0.5, step_y: 0.4330127, odd_row_shift: 0.5,
  shift_confirmed: false, steps_known: true,
};
const CROP = { ...PARENT, n_rows: 3, n_cols: 3, odd_row_shift: -0.5 };

describe('datasetGeometry carries the dataset\'s scan geometry', () => {
  it('parses it for the store and keeps the backend\'s object for ebsdInfo', () => {
    const g = datasetGeometry({ name: 'crop', navigation_shape: [3, 3], signal_shape: [8, 8], scan_geometry: CROP });
    expect(g.scanGeometry).toMatchObject({ kind: 'hex', gridShape: [3, 3], oddRowShift: -0.5 });
    expect(g.scanGeometryRaw).toBe(CROP);
  });

  it('reports none when the backend did not', () => {
    const g = datasetGeometry({ navigation_shape: [3, 3] });
    expect(g.scanGeometry).toBeNull();
    expect(g.scanGeometryRaw).toBeNull();
  });
});

describe('ebsdInfoWithGrid', () => {
  const info = {
    grid_shape: [6, 5], scan_geometry: PARENT, file_path: '/d/scan.up1', dtype: 'uint8',
    pattern_shape: [8, 8], has_eds: false,
  };

  it('applyGrid replaces the geometry of ebsdInfo along with its grid', () => {
    const geom = datasetGeometry({ navigation_shape: [3, 3], signal_shape: [8, 8], scan_geometry: CROP });
    const out = ebsdInfoWithGrid(info, geom);
    expect(out.grid_shape).toEqual([3, 3]);
    expect(out.scan_geometry).toBe(CROP);                 // not the parent's
    expect(out.file_path).toBe('/d/scan.up1');            // everything else about the file stays
  });

  it('a dataset without a geometry leaves null, not the parent\'s', () => {
    const geom = datasetGeometry({ navigation_shape: [3, 3], signal_shape: [8, 8] });
    expect(ebsdInfoWithGrid(info, geom).scan_geometry).toBeNull();
  });

  it('works from nothing', () => {
    const geom = datasetGeometry({ navigation_shape: [3, 3], scan_geometry: CROP });
    expect(ebsdInfoWithGrid(null, geom).grid_shape).toEqual([3, 3]);
  });
});

describe('the resampled flag travels with the grid of a dataset', () => {
  const info = { grid_shape: [6, 5], hex_resampled: true, file_path: '/d/scan.h5' };

  it('datasetGeometry returns the flag of the entry: true, false, or null when not said', () => {
    expect(datasetGeometry({ navigation_shape: [3, 3], hex_resampled: true }).hexResampled).toBe(true);
    expect(datasetGeometry({ navigation_shape: [3, 3], hex_resampled: false }).hexResampled).toBe(false);
    expect(datasetGeometry({ navigation_shape: [3, 3] }).hexResampled).toBeNull();
    expect(datasetGeometry({ navigation_shape: [3, 3], hex_resampled: null }).hexResampled).toBeNull();
  });

  it('ebsdInfoWithGrid replaces the flag along with the grid, never keeps the parent\'s', () => {
    const own = datasetGeometry({ navigation_shape: [3, 3], hex_resampled: false });
    expect(ebsdInfoWithGrid(info, own).hex_resampled).toBe(false);
    const unsaid = datasetGeometry({ navigation_shape: [3, 3] });
    expect(ebsdInfoWithGrid(info, unsaid).hex_resampled).toBeNull();
  });
});
