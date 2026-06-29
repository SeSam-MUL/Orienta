import { describe, it, expect } from 'vitest';
import { disambiguateNames } from './disambiguateNames';

describe('disambiguateNames', () => {
  it('returns the input unchanged for 0 or 1 names', () => {
    expect(disambiguateNames([])).toEqual([]);
    expect(disambiguateNames(['only one'])).toEqual(['only one']);
    expect(disambiguateNames(null)).toEqual([]);
  });

  it('strips the common prefix and suffix, keeping the discriminating middle', () => {
    const names = [
      'HIgh_MG_EBSD Al-10Mg_Guss Arbeitsbereich 2 Elementverteilungsdaten 2',
      'HIgh_MG_EBSD 5182-10Mg_Guss Arbeitsbereich 2 Elementverteilungsdaten 4',
      'HIgh_MG_EBSD 5182-10Mg_WB-6mm_AC Arbeitsbereich 1 Elementverteilungsdaten 11',
    ];
    const out = disambiguateNames(names);
    // common prefix "HIgh_MG_EBSD" stripped; tails differ so nothing common there
    expect(out[0]).toBe('Al-10Mg_Guss Arbeitsbereich 2 Elementverteilungsdaten 2');
    expect(out[1]).toBe('5182-10Mg_Guss Arbeitsbereich 2 Elementverteilungsdaten 4');
    expect(out[2]).toBe('5182-10Mg_WB-6mm_AC Arbeitsbereich 1 Elementverteilungsdaten 11');
    // every label is distinct
    expect(new Set(out).size).toBe(3);
  });

  it('strips a common suffix when present', () => {
    const names = [
      'Sample A Arbeitsbereich 1 Elementverteilungsdaten 1',
      'Sample B Arbeitsbereich 1 Elementverteilungsdaten 1',
    ];
    const out = disambiguateNames(names);
    expect(out).toEqual(['A', 'B']);
  });

  it('never returns an empty label (falls back to full name)', () => {
    // identical names → middle would be empty → fall back to full name
    const out = disambiguateNames(['same name', 'same name']);
    expect(out).toEqual(['same name', 'same name']);
  });

  it('leaves already-distinct names readable', () => {
    const out = disambiguateNames(['Scan1', 'HiGainNi', 'LoGainNi']);
    expect(out).toEqual(['Scan1', 'HiGainNi', 'LoGainNi']);
  });
});
