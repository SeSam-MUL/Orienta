import { describe, it, expect } from 'vitest';
import { pickShtForPhase } from './previewSht';

// The shape /api/indexing/files/spherical returns, trimmed to what matters.
const AL = {
  path: 'C:\\lib\\EBSD_SHT_Database\\Al\\Al (Al) [cF4] {20kV}.sht',
  filename: 'Al (Al) [cF4] {20kV}.sht', material: 'Al', display_label: 'Al', formula: 'Al',
};
const NI = {
  path: 'C:\\lib\\EBSD_SHT_Database\\Ni\\Ni (Ni) [cF4] {20kV}.sht',
  filename: 'Ni (Ni) [cF4] {20kV}.sht', material: 'Ni', display_label: 'Ni', formula: 'Ni',
};
const ALPHA = {
  path: 'C:\\lib\\EBSD_SHT_Database\\alpha-AlFeMnSi\\Mn0.5Fe0.5Al5Si0.68 (sd_0302719) [cI168] {20kV}.sht',
  filename: 'Mn0.5Fe0.5Al5Si0.68 (sd_0302719) [cI168] {20kV}.sht',
  material: 'alpha-AlFeMnSi', display_label: 'Mn0.5Fe0.5Al5Si0.68', formula: 'Mn0.5Fe0.5Al5Si0.68',
};

describe('pickShtForPhase', () => {
  it('takes the loaded phase, not the first file in the list', () => {
    // The workshop case: Al listed first, Ni loaded for the calibration.
    expect(pickShtForPhase([AL, NI], 'Ni')).toBe(NI.path);
  });

  it('matches case-insensitively and on the formula', () => {
    expect(pickShtForPhase([AL, ALPHA], 'mn0.5fe0.5al5si0.68')).toBe(ALPHA.path);
  });

  it('falls back to the leading formula of the file name', () => {
    const bare = { path: 'C:\\x\\Ni (Ni) [cF4] {20kV}.sht' };
    expect(pickShtForPhase([AL, bare], 'Ni')).toBe(bare.path);
  });

  it('finds a master by the CIF stem the phase was loaded under', () => {
    // Review finding: alpha loaded from sd_0302719.cif is called "sd_0302719",
    // which matches neither its material folder nor its formula.
    expect(pickShtForPhase([AL, ALPHA], 'sd_0302719')).toBe(ALPHA.path);
  });

  it('still offers a preview when the phase has no master of its own', () => {
    expect(pickShtForPhase([AL, NI], 'Fe')).toBe(AL.path);
  });

  it('returns null when there is nothing to choose', () => {
    expect(pickShtForPhase([], 'Ni')).toBeNull();
    expect(pickShtForPhase(null, 'Ni')).toBeNull();
  });
});
