import { describe, it, expect } from 'vitest';
import { buildEdsPhaseStrengths, edsPriorActive } from './edsPriorParams';

// The request shape buildParams() ships for the EDS chemistry prior. The
// on-state must be 1.0 for EVERY phase (a partial application is exactly the
// bug the per-phase slider caused); the off-state must omit the field so the
// payload stays identical to a run without the feature.
describe('buildEdsPhaseStrengths', () => {
  const paths = ['C:/lib/Al.sht', 'C:/lib/Si.sht'];

  it('switch ON → 1.0 for every phase', () => {
    expect(buildEdsPhaseStrengths({ enabled: true, edsAvailable: true, phasePaths: paths }))
      .toEqual({ 'C:/lib/Al.sht': 1.0, 'C:/lib/Si.sht': 1.0 });
  });

  it('switch OFF → null, so the field is omitted entirely', () => {
    expect(buildEdsPhaseStrengths({ enabled: false, edsAvailable: true, phasePaths: paths }))
      .toBeNull();
  });

  it('no EDS in the file → null even when the switch is on', () => {
    expect(buildEdsPhaseStrengths({ enabled: true, edsAvailable: false, phasePaths: paths }))
      .toBeNull();
  });

  it('no phases selected → null', () => {
    expect(buildEdsPhaseStrengths({ enabled: true, edsAvailable: true, phasePaths: [] }))
      .toBeNull();
    expect(buildEdsPhaseStrengths({ enabled: true, edsAvailable: true }))
      .toBeNull();
  });

  it('never emits a value other than 1.0 — there is no partial state', () => {
    const out = buildEdsPhaseStrengths({ enabled: true, edsAvailable: true, phasePaths: paths });
    expect(new Set(Object.values(out))).toEqual(new Set([1.0]));
  });

  it('keys are the paths given (already dictionary-remapped by the caller)', () => {
    // Dictionary runs substitute the selected dictionary path for the master
    // path; the backend looks the weight up by that substituted key.
    const out = buildEdsPhaseStrengths({
      enabled: true, edsAvailable: true, phasePaths: ['C:/lib/Al_dict_5deg.h5'],
    });
    expect(Object.keys(out)).toEqual(['C:/lib/Al_dict_5deg.h5']);
  });

  it('drops empty path entries', () => {
    const out = buildEdsPhaseStrengths({
      enabled: true, edsAvailable: true, phasePaths: ['C:/lib/Al.sht', '', null],
    });
    expect(out).toEqual({ 'C:/lib/Al.sht': 1.0 });
  });
});

describe('edsPriorActive', () => {
  it('mirrors whether the field would be sent', () => {
    const p = ['a.sht'];
    expect(edsPriorActive({ enabled: true, edsAvailable: true, phasePaths: p })).toBe(true);
    expect(edsPriorActive({ enabled: false, edsAvailable: true, phasePaths: p })).toBe(false);
    expect(edsPriorActive({ enabled: true, edsAvailable: false, phasePaths: p })).toBe(false);
    expect(edsPriorActive({ enabled: true, edsAvailable: true, phasePaths: [] })).toBe(false);
  });
});
