import { describe, it, expect } from 'vitest';
import { phaseFitsTooltip } from './phaseFits';

// A t() that returns the key plus its parameters, so the test sees what is passed.
const t = (key, o = {}) => `${key.split(':')[1].replace('preview.', '')}|${Object.values(o).join(',')}`;

describe('phaseFitsTooltip', () => {
  it('is null for one phase or none (nothing to compare)', () => {
    expect(phaseFitsTooltip(undefined, 'A', t)).toBeNull();
    expect(phaseFitsTooltip(null, 'A', t)).toBeNull();
    expect(phaseFitsTooltip([{ name: 'A', ci: 0.5, fit: 1, n_bands: 6, score: 12 }], 'A', t)).toBeNull();
  });

  it('lists every phase in order, marks only the winner, and ends with the rule', () => {
    const lines = phaseFitsTooltip([
      { name: 'A', ci: 0.5, fit: 1, n_bands: 6, score: 12 },
      { name: 'B', ci: null, fit: null, n_bands: 1, score: null },
    ], 'A', t).split('\n');
    expect(lines).toEqual([
      'phaseFitsTitle|',
      'phaseFitsRow|A,0.500,1.00,6,12.0' + 'phaseFitsWinner|',
      'phaseFitsNone|B,1',
      'phaseFitsRule|',
    ]);
  });
});
