/**
 * Region Average: is its composition a sum or a mean?
 *
 * Before anything is computed the panel reads the file's fit state. A failed
 * fit counted as "summed", so the panel promised a summed composition where
 * `/region-quantify` answers 500.
 */
import { describe, it, expect } from 'vitest';
import { regionAverageIsSummed } from './regionAverage';
import { COMPOSITION_SUMMED_NETS } from './compositionMethod';

describe('regionAverageIsSummed', () => {
  it('a running or finished fit gives summed compositions', () => {
    expect(regionAverageIsSummed(null, 'running')).toBe(true);
    expect(regionAverageIsSummed(null, 'ready')).toBe(true);
  });

  it('a failed fit gives no composition, and is not called summed', () => {
    expect(regionAverageIsSummed(null, 'failed')).toBe(false);
  });

  it('a file without a cube, or no answer yet, gives the mean of pixels', () => {
    expect(regionAverageIsSummed(null, 'none')).toBe(false);
    expect(regionAverageIsSummed(null, null)).toBe(false);
  });

  it('an answer says it itself, whatever the state', () => {
    const summed = { composition_method: COMPOSITION_SUMMED_NETS };
    const mean = { composition_method: 'mean_of_pixel_at_pct' };
    expect(regionAverageIsSummed(summed, 'failed')).toBe(true);
    expect(regionAverageIsSummed(mean, 'ready')).toBe(false);
  });
});
