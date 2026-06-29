import { describe, it, expect } from 'vitest';
import { downsampleForWidth, energyToX, xToEnergy } from './spectrumPlot';

describe('spectrumPlot', () => {
  it('downsampleForWidth keeps every-Nth point', () => {
    const arr = Array.from({ length: 2048 }, (_, i) => i);
    const ds = downsampleForWidth(arr, 600);
    expect(ds.length).toBeLessThanOrEqual(600);
    expect(ds[0]).toBe(0);
    expect(ds[ds.length - 1]).toBe(2047);
  });
  it('energyToX maps 0 keV to 0 and energyMax to width', () => {
    expect(energyToX(0, 0, 20, 800)).toBe(0);
    expect(energyToX(20, 0, 20, 800)).toBe(800);
    expect(energyToX(10, 0, 20, 800)).toBe(400);
  });
  it('xToEnergy is the inverse', () => {
    expect(xToEnergy(400, 0, 20, 800)).toBeCloseTo(10);
  });
});
