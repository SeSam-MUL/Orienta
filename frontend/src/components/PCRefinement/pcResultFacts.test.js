// What the Result panel reads out of a search result: hand-calculated expectations.
import { describe, it, expect } from 'vitest';
import {
  warningOf, needsDecision, secondPositionOf, contrastOf, patternCountOf, patternSpread,
} from './pcResultFacts';

describe('warningOf', () => {
  it('takes the backend\'s quality.warning', () => {
    expect(warningOf({ quality: { warning: 'edge', axes: ['y', 'z'] } })).toEqual({ kind: 'edge', axes: ['y', 'z'] });
    expect(warningOf({ quality: { warning: 'notAMinimum' } })).toEqual({ kind: 'notAMinimum' });
    expect(warningOf({ quality: { warning: 'secondPosition' } })).toEqual({ kind: 'secondPosition' });
    expect(warningOf({ quality: { warning: null } })).toBeNull();
  });

  it('a result with quality is not judged again from its search record', () => {
    expect(warningOf({ quality: { warning: null }, search: { on_edge: ['x'] } })).toBeNull();
  });

  it('a result without quality is judged from its search record: edge first, then no lowest point', () => {
    expect(warningOf({ search: { on_edge: ['x'], confidence: { reasons: ['notAMinimum'] } } }))
      .toEqual({ kind: 'edge', axes: ['x'] });
    expect(warningOf({ search: { on_edge: [], confidence: { reasons: ['notAMinimum'] } } })).toEqual({ kind: 'notAMinimum' });
    expect(warningOf({ search: { on_edge: [], confidence: { reasons: [] } } })).toBeNull();
    expect(warningOf({})).toBeNull();
    expect(warningOf(null)).toBeNull();
  });

  it('asks what to do only for the edge and for no lowest point, not for a second position', () => {
    expect(needsDecision({ quality: { warning: 'edge', axes: ['x'] } })).toBe(true);
    expect(needsDecision({ quality: { warning: 'notAMinimum' } })).toBe(true);
    expect(needsDecision({ quality: { warning: 'secondPosition' } })).toBe(false);
    expect(needsDecision({ quality: { warning: null } })).toBe(false);
  });
});

describe('secondPositionOf, contrastOf, patternCountOf', () => {
  it('reads the second position from the quality, else from the search, and only when it is three numbers', () => {
    const s = { pc: [0.36, 0.44, 0.74], cost: 8.9 };
    expect(secondPositionOf({ quality: { second_position: s } })).toEqual(s);
    expect(secondPositionOf({ search: { second_position: s } })).toEqual(s);
    expect(secondPositionOf({ quality: { second_position: { pc: [0.3, 0.4], cost: 1 } } })).toBeNull();
    expect(secondPositionOf({ quality: {}, search: { second_position: null } })).toBeNull();
    expect(secondPositionOf({ quality: { second_position: { pc: [0.3, 0.4, 0.5], cost: NaN } } }).cost).toBeNull();
  });

  it('reads the contrast from the quality, else from the search, and nothing that is not a number', () => {
    expect(contrastOf({ quality: { contrast: 0.8 } })).toBe(0.8);
    expect(contrastOf({ quality: { contrast: null }, search: { confidence: { contrast: 0.5 } } })).toBeNull();
    expect(contrastOf({ search: { confidence: { contrast: 0.5 } } })).toBe(0.5);
    expect(contrastOf({})).toBeNull();
  });

  it('counts the patterns of the quality, else of the search', () => {
    expect(patternCountOf({ quality: { n_patterns: 9 } })).toBe(9);
    expect(patternCountOf({ search: { n_patterns: 5 } })).toBe(5);
    expect(patternCountOf({})).toBeNull();
  });
});

describe('patternSpread', () => {
  // ranges per axis, exact in binary: x 1.0 - 0.5 = 0.5, y 0.5 - 0.25 = 0.25, z 0.625 - 0.5 = 0.125
  const per = [[0.5, 0.25, 0.5], [1.0, 0.5, 0.625], [0.75, 0.375, 0.5]];

  it('is the largest range per axis, and in pixels x counts columns, y and z rows', () => {
    // ranges 0.5, 0.25, 0.125; pattern 100 rows x 200 columns: 100, 25, 12.5 pixels
    expect(patternSpread({ search: { per_pattern: per }, pattern_size: [100, 200] })).toEqual({ value: 0.5, pixels: 100 });
  });

  it('takes the largest range in pixels even where it is not the largest in PC units', () => {
    // pattern 400 rows x 100 columns: x 0.5 x 100 = 50, y 0.25 x 400 = 100, z 0.125 x 400 = 50: y wins in pixels
    expect(patternSpread({ search: { per_pattern: per }, pattern_size: [400, 100] })).toEqual({ value: 0.5, pixels: 100 });
  });

  it('has no pixels without a pattern size', () => {
    expect(patternSpread({ search: { per_pattern: per } })).toEqual({ value: 0.5, pixels: null });
    expect(patternSpread({ search: { per_pattern: per }, pattern_size: [0, 100] }).pixels).toBeNull();
  });

  it('is null for fewer than two patterns with a PC of their own, or none kept', () => {
    expect(patternSpread({ search: { per_pattern: [[0.5, 0.25, 0.5]] } })).toBeNull();
    expect(patternSpread({ search: { per_pattern: [[0.5, 0.25, 0.5], [null, 0.3, 0.5]] } })).toBeNull();
    expect(patternSpread({ search: { per_pattern: null } })).toBeNull();
    expect(patternSpread({})).toBeNull();
  });

  it('leaves a pattern without a PC of its own out and uses the others', () => {
    expect(patternSpread({ search: { per_pattern: [[0.5, 0.25, 0.5], [null, 0.3, 0.5], [0.75, 0.25, 0.5]] } }).value).toBe(0.25);
  });
});
