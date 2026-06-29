import { describe, it, expect } from 'vitest';
import { EDS_LINES, getElementLines } from './edsLines';

describe('edsLines', () => {
  it('Fe Kα1 is at 6.40 keV', () => {
    expect(EDS_LINES.Fe.Ka1).toBeCloseTo(6.404, 2);
  });
  it('getElementLines returns array sorted by energy', () => {
    const lines = getElementLines('Fe');
    expect(lines.length).toBeGreaterThanOrEqual(3);
    for (let i = 1; i < lines.length; i++) {
      expect(lines[i].energy).toBeGreaterThanOrEqual(lines[i-1].energy);
    }
  });
  it('unknown element returns empty array', () => {
    expect(getElementLines('Xx')).toEqual([]);
  });
});
