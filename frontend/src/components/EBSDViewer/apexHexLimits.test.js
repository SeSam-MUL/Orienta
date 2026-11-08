// @vitest-environment node
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { groupHexLimits } from './apexHexLimits';

const FIXTURE = JSON.parse(readFileSync(
  fileURLToPath(new URL('./__fixtures__/hexLimits.json', import.meta.url)), 'utf8'));

describe('groupHexLimits', () => {
  const g = groupHexLimits(FIXTURE);
  const byCategory = Object.fromEntries(g.categories.map((c) => [c.category, c.items.map((i) => i.key)]));

  it('keeps the backend\'s order of categories, and only those that have a limit', () => {
    // `cleanup` stays in the backend's list of categories but holds no limit any more: it is not returned
    const withALimit = FIXTURE.categories.filter((c) => Object.values(FIXTURE.groups).flat().some((i) => i.category === c));
    expect(FIXTURE.categories).toHaveLength(5);
    expect(FIXTURE.categories).not.toContain('export');
    expect(FIXTURE.categories).not.toContain('batch');
    expect(withALimit).not.toContain('cleanup');
    expect(withALimit).toHaveLength(FIXTURE.categories.length - 1);
    expect(g.categories.map((c) => c.category)).toEqual(withALimit);
    const trimmed = { ...FIXTURE, categories: [...FIXTURE.categories, 'a_future_category'] };
    expect(groupHexLimits(trimmed).categories.map((c) => c.category)).toEqual(withALimit);
  });

  it('groups the 10 limits of the backend by category (hand-counted from hex_guard.GUARDED: 12 routes, the 3 crystal-hint routes being one limit)', () => {
    expect(Object.values(byCategory).flat()).toHaveLength(10);
    expect(g.categories).toHaveLength(4);                             // preprocessing, grains, maps, refinement
    expect(g.categories.some((c) => c.category === 'export' || c.category === 'batch')).toBe(false);
    expect(byCategory.preprocessing).toEqual(['nlpar', 'crystal_hint_averaging']);
    expect(byCategory.grains_and_boundaries).toEqual(
      ['grain_phase_assignment', 'grain_reconstruction', 'grain_size', 'deformation_analysis']);
    expect(byCategory.maps).toEqual(['analysis_map']);
    expect(byCategory.cleanup).toBeUndefined();                       // no limit left: the category is not shown
    expect(byCategory.refinement).toEqual(['pixelwise_pc_calibration', 'refinement_compute', 'refinement_resmooth']);
    expect(byCategory.export).toBeUndefined();                        // the result is written as a file now
    expect(byCategory.batch).toBeUndefined();                         // the batches run and write: no batch limit
  });

  it('merges the three crystal-hint routes, which share one key, into one partial limit with its parameter', () => {
    const keys = Object.values(byCategory).flat();
    expect(keys.filter((k) => k === 'crystal_hint_averaging')).toHaveLength(1);
    const averaging = g.categories.find((c) => c.category === 'preprocessing').items
      .find((i) => i.key === 'crystal_hint_averaging');
    expect(averaging.partial).toBe(true);
    expect(averaging.parameters).toEqual(['avg_radius']);
    expect(averaging.affects).toEqual([]);
  });

  it('carries what a partial limit refuses, as the gate\'s own constants say it', () => {
    const hint = g.categories.find((c) => c.category === 'preprocessing').items
      .find((i) => i.key === 'crystal_hint_averaging');
    expect([hint.affects, hint.parameters]).toEqual([[], ['avg_radius']]);
    // the analysis maps are refused whole: nothing to list
    const amap = g.categories.find((c) => c.category === 'maps').items.find((i) => i.key === 'analysis_map');
    expect([amap.partial, amap.affects, amap.parameters]).toEqual([false, [], []]);
  });

  it('names the things that still work, and no limit has an effect left', () => {
    expect(g.works).toEqual(['image_export', 'file_export', 'global_pc_refinement']);
    expect(g.categories.flatMap((c) => c.items).every((i) => !('effect' in i))).toBe(true);
  });

  it('unites the affected kinds of items that share a key', () => {
    const merged = groupHexLimits({
      categories: ['maps'],
      groups: { a: [{ key: 'k', category: 'maps', partial: true, affects: ['x'], parameters: ['q'] }],
        b: [{ key: 'k', category: 'maps', partial: true, affects: ['y'], parameters: ['p', 'q'] }] },
      not_limited: [],
    });
    expect(merged.categories[0].items).toEqual([
      { key: 'k', category: 'maps', partial: true, affects: ['x', 'y'], parameters: ['q', 'p'] }]);
  });

  it('keeps two limits that differ in their key apart, in the order the backend lists them', () => {
    const apart = groupHexLimits({
      categories: ['maps'],
      groups: { a: [{ key: 'k1', category: 'maps', partial: false, affects: [], parameters: [] },
        { key: 'k2', category: 'maps', partial: true, affects: ['z'], parameters: [] }] },
      not_limited: [],
    });
    expect(apart.categories[0].items.map((i) => [i.key, i.partial, i.affects])).toEqual([
      ['k1', false, []], ['k2', true, ['z']]]);
  });

  it('is null for an answer that is not the list', () => {
    for (const bad of [null, undefined, {}, { groups: {} }, 'x', { groups: {}, categories: 3 }]) {
      expect(groupHexLimits(bad)).toBeNull();
    }
  });
});
