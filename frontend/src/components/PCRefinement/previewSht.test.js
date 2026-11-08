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
    // Alpha loaded from sd_0302719.cif is called "sd_0302719",
    // which matches neither its material folder nor its formula.
    expect(pickShtForPhase([AL, ALPHA], 'sd_0302719')).toBe(ALPHA.path);
  });

  it("gives a phase without a master of its own none, not another phase's", () => {
    // Alpha-/beta-Ti from their own CIFs were once previewed against
    // the Mg master (NCC 0.016), which reads as a bad pattern centre.
    expect(pickShtForPhase([AL, NI], 'titan-alpha-mp-46')).toBeNull();
    expect(pickShtForPhase([AL, NI], 'Fe')).toBeNull();
  });

  it('does not decide on a material folder several masters share', () => {
    // The real library: the "Al" folder holds Al, Al6Fe, Al7FeCu2, Al2Cu, MnAl6, pi.
    const al6fe = {
      path: 'C:\\lib\\EBSD_SHT_Database\\Al\\Al6Fe (Al6Fe_mp-570001_symmetrized) {20kV}.sht',
      filename: 'Al6Fe (Al6Fe_mp-570001_symmetrized) {20kV}.sht',
      material: 'Al', display_label: 'Al₆Fe', formula: 'Al6Fe',
    };
    expect(pickShtForPhase([al6fe, AL], 'Al')).toBe(AL.path);       // its own label decides
    expect(pickShtForPhase([al6fe, { ...AL, display_label: 'x', formula: 'x', filename: 'x.sht', path: 'C:\\x.sht' }], 'Al')).toBeNull();
  });

  it('takes the phase\'s own master when it was simulated at two beam energies', () => {
    // One phase, 10 kV and 20 kV masters, matched both on every
    // rule and so got none. They are the same phase's own masters.
    const at = (kv) => ({ path: `C:\\l\\Ni\\Ni (Ni) [cF4] {${kv}kV}.sht`, filename: `Ni (Ni) [cF4] {${kv}kV}.sht`,
      material: 'Ni', display_label: 'Ni', formula: 'Ni' });
    expect(pickShtForPhase([AL, at(20), at(10)], 'Ni')).toBe(at(20).path);
  });

  it('does not decide on a label two masters carry', () => {
    // alpha- and beta-Ti masters are both labelled "Ti"; a phase named just
    // "Ti" could be either.
    const alphaTi = { path: 'C:\\l\\Ti (alpha-Ti_COD-9008517) [hP6] {20kV}.sht', filename: 'Ti (alpha-Ti_COD-9008517) [hP6] {20kV}.sht', material: 'Ti', display_label: 'Ti', formula: 'Ti' };
    const betaTi = { path: 'C:\\l\\Ti (beta-Ti_COD-9008554) [cI2] {20kV}.sht', filename: 'Ti (beta-Ti_COD-9008554) [cI2] {20kV}.sht', material: 'Ti', display_label: 'Ti', formula: 'Ti' };
    expect(pickShtForPhase([alphaTi, betaTi], 'Ti')).toBeNull();
    expect(pickShtForPhase([alphaTi, betaTi], 'beta-Ti_COD-9008554')).toBe(betaTi.path);
  });

  it('returns null when there is nothing to choose', () => {
    expect(pickShtForPhase([], 'Ni')).toBeNull();
    expect(pickShtForPhase(null, 'Ni')).toBeNull();
  });
});
