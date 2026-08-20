import { describe, it, expect } from 'vitest';
import { datasetGeometry } from './datasetGeometry';

describe('datasetGeometry', () => {
  it('flips hyperspy (x, y) into (rows, cols)', () => {
    const g = datasetGeometry({
      name: 'Scan1_crop1',
      navigation_shape: [10, 8],     // 10 cols, 8 rows
      signal_shape: [156, 128],      // 156 wide, 128 high
      dtype: 'uint8',
    });
    expect(g.gridShape).toEqual([8, 10]);
    expect(g.patternShape).toEqual([128, 156]);
    expect(g.patternCount).toBe(80);
    expect(g.navigationShape).toEqual([10, 8]);
    expect(g.dtype).toBe('uint8');
  });

  it('keeps the pattern shape optional', () => {
    const g = datasetGeometry({ navigation_shape: [4, 3] });
    expect(g.gridShape).toEqual([3, 4]);
    expect(g.patternShape).toBeNull();
    expect(g.dtype).toBeNull();
  });

  it('returns null rather than a guess when the navigation shape is unusable', () => {
    expect(datasetGeometry(null)).toBeNull();
    expect(datasetGeometry(undefined)).toBeNull();
    expect(datasetGeometry({})).toBeNull();
    expect(datasetGeometry({ navigation_shape: [12] })).toBeNull();
    expect(datasetGeometry({ navigation_shape: [0, 5] })).toBeNull();
    expect(datasetGeometry({ navigation_shape: ['a', 'b'] })).toBeNull();
  });
});
