import { describe, it, expect } from 'vitest';
import { classifyPair, detectPhaseDegeneracy, planReduction } from './phaseDegeneracy';

// Fixtures — shaped like discover_files entries.
const al6fe = {
  path: '/db/Al6Fe.sht', formula: 'Al6Fe', filename: 'Al6Fe.sht',
  laue_class: 'mmm', centering: 'C', pearson: 'oS28', space_group_number: 63,
  lattice_a: 7.430, lattice_b: 6.440, lattice_c: 8.816,
  lattice_alpha: 90, lattice_beta: 90, lattice_gamma: 90,
};
const mnal6 = {
  path: '/db/MnAl6.sht', formula: 'MnAl6', filename: 'MnAl6.sht',
  laue_class: 'mmm', centering: 'C', pearson: 'oS28', space_group_number: 63,
  lattice_a: 7.539, lattice_b: 6.460, lattice_c: 8.819,
  lattice_alpha: 90, lattice_beta: 90, lattice_gamma: 90,
};
const alphaIm3a = {
  path: '/db/aA.sht', formula: 'Mn0.52Fe1.08Al4.85Si0.55', filename: 'aA.sht',
  laue_class: 'm-3', centering: 'I', pearson: 'cI168', space_group_number: 204,
  lattice_a: 12.50, lattice_b: 12.50, lattice_c: 12.50,
  lattice_alpha: 90, lattice_beta: 90, lattice_gamma: 90,
};
const alphaIm3b = { ...alphaIm3a, path: '/db/aB.sht', formula: 'Mn0.5Fe0.5Al5Si0.68',
  filename: 'aB.sht', lattice_a: 12.56, lattice_b: 12.56, lattice_c: 12.56 };
const alphaIm3c = { ...alphaIm3a, path: '/db/aC.sht', formula: 'Mn0.5Fe0.5Al5Si0.68b',
  filename: 'aC.sht' };
const alphaPm3 = {
  path: '/db/aMn.sht', formula: 'Al4.01Mn1Si0.74', filename: 'aMn.sht',
  laue_class: 'm-3', centering: 'P', pearson: 'cP138', space_group_number: 200,
  lattice_a: 12.643, lattice_b: 12.643, lattice_c: 12.643,
  lattice_alpha: 90, lattice_beta: 90, lattice_gamma: 90,
};
const al = {
  path: '/db/Al.sht', formula: 'Al', filename: 'Al.sht',
  laue_class: 'm-3m', centering: 'F', pearson: 'cF4', space_group_number: 225,
  lattice_a: 4.049, lattice_b: 4.049, lattice_c: 4.049,
  lattice_alpha: 90, lattice_beta: 90, lattice_gamma: 90,
};

describe('classifyPair', () => {
  it('flags an exact duplicate (same path) as hard', () => {
    expect(classifyPair(al6fe, { ...mnal6, path: al6fe.path })).toBe('hard');
  });
  it('flags Al6Fe vs MnAl6 (Cmcm, lattice within 1.5%) as hard', () => {
    expect(classifyPair(al6fe, mnal6)).toBe('hard');
  });
  it('flags Im-3 vs Pm-3 alpha phases (same Laue, different centering) as soft', () => {
    expect(classifyPair(alphaIm3a, alphaPm3)).toBe('soft');
  });
  it('returns none for two structurally distinct phases', () => {
    expect(classifyPair(al, alphaIm3a)).toBe('none');   // m-3m vs m-3
  });
  it('uses the SG-number + Pearson fallback when lattice data is missing', () => {
    const a = { path: '/x', space_group_number: 204, pearson: 'cI168' };
    const b = { path: '/y', space_group_number: 204, pearson: 'cI168' };
    const c = { path: '/z', space_group_number: 204, pearson: 'cF16' };
    expect(classifyPair(a, b)).toBe('hard');
    expect(classifyPair(a, c)).toBe('none');
  });
});

describe('detectPhaseDegeneracy', () => {
  it('merges three Im-3 alpha entries into one hard cluster of 3', () => {
    const { clusters } = detectPhaseDegeneracy([alphaIm3a, alphaIm3b, alphaIm3c]);
    const hard = clusters.filter(c => c.tier === 'hard');
    expect(hard).toHaveLength(1);
    expect(hard[0].members.sort()).toEqual([0, 1, 2]);
  });
  it('reports a soft pair and marks both phases in byPhaseIndex', () => {
    const { clusters, byPhaseIndex } = detectPhaseDegeneracy([alphaIm3a, alphaPm3]);
    expect(clusters).toHaveLength(1);
    expect(clusters[0].tier).toBe('soft');
    expect(byPhaseIndex[0].tier).toBe('soft');
    expect(byPhaseIndex[1].tier).toBe('soft');
  });
  it('returns no clusters for a clean distinct selection', () => {
    const { clusters } = detectPhaseDegeneracy([al, alphaIm3a]);
    expect(clusters).toHaveLength(0);
  });
  it('suppresses a soft annotation when the phase is already in a hard cluster', () => {
    // alphaIm3a/b are hard with each other AND soft vs alphaPm3.
    const { clusters } = detectPhaseDegeneracy([alphaIm3a, alphaIm3b, alphaPm3]);
    expect(clusters.filter(c => c.tier === 'hard')).toHaveLength(1);
    expect(clusters.filter(c => c.tier === 'soft')).toHaveLength(0);
  });
});

describe('planReduction', () => {
  it('keeps the lowest index and returns the rest', () => {
    expect(planReduction({ tier: 'hard', members: [5, 1, 3] })).toEqual([3, 5]);
  });
  it('returns nothing for a soft cluster', () => {
    expect(planReduction({ tier: 'soft', members: [0, 1] })).toEqual([]);
  });
});

describe('classifyPair — angle-tolerance guard', () => {
  it('rejects two otherwise-hard phases when a cell angle diverges > 1.5°', () => {
    // Same Laue class + centering + lattice axes as al6fe, but beta = 95°.
    const skewed = { ...al6fe, path: '/db/x.sht', lattice_beta: 95 };
    expect(classifyPair(skewed, al6fe)).toBe('none');
  });
});

describe('detectPhaseDegeneracy — empty / null / single input', () => {
  it('returns empty result for null without throwing', () => {
    const r = detectPhaseDegeneracy(null);
    expect(r.clusters).toEqual([]);
    expect(r.byPhaseIndex).toEqual({});
  });
  it('returns empty result for undefined without throwing', () => {
    const r = detectPhaseDegeneracy(undefined);
    expect(r.clusters).toEqual([]);
    expect(r.byPhaseIndex).toEqual({});
  });
  it('returns empty result for an empty array without throwing', () => {
    const r = detectPhaseDegeneracy([]);
    expect(r.clusters).toEqual([]);
    expect(r.byPhaseIndex).toEqual({});
  });
  it('returns empty result for a single-phase array without throwing', () => {
    const r = detectPhaseDegeneracy([al6fe]);
    expect(r.clusters).toEqual([]);
    expect(r.byPhaseIndex).toEqual({});
  });
});

describe('detectPhaseDegeneracy — byPhaseIndex partners for a hard cluster', () => {
  it('lists the partner formula on each member of a hard cluster', () => {
    const { byPhaseIndex } = detectPhaseDegeneracy([al6fe, mnal6]);
    expect(byPhaseIndex[0].tier).toBe('hard');
    expect(byPhaseIndex[0].partners).toContain('MnAl6');
    expect(byPhaseIndex[1].tier).toBe('hard');
    expect(byPhaseIndex[1].partners).toContain('Al6Fe');
  });
});

describe('classifyPair — partial lattice falls through to fallback', () => {
  it('uses the SG-number + Pearson fallback when one phase has a zero axis', () => {
    const partial = {
      path: '/db/partial.sht', space_group_number: 204, pearson: 'cI168',
      lattice_a: 0, lattice_b: 12.5, lattice_c: 12.5,
    };
    const full = {
      path: '/db/full.sht', space_group_number: 204, pearson: 'cI168',
      lattice_a: 12.5, lattice_b: 12.5, lattice_c: 12.5,
    };
    expect(classifyPair(partial, full)).toBe('hard');
  });
  it('uses the fallback when one phase is missing an axis entirely', () => {
    const partial = {
      path: '/db/nob.sht', space_group_number: 204, pearson: 'cI168',
      lattice_a: 12.5, lattice_c: 12.5,
    };
    const full = {
      path: '/db/full2.sht', space_group_number: 204, pearson: 'cI168',
      lattice_a: 12.5, lattice_b: 12.5, lattice_c: 12.5,
    };
    expect(classifyPair(partial, full)).toBe('hard');
  });
});
