import { describe, it, expect } from 'vitest';
import { parseScanGeometry, isHexScan } from './scanGeometry';

const HEX = {
  kind: 'hex', n_rows: 6, n_cols: 5, grid_shape: [6, 5], step_x: 0.5, step_y: 0.4330127,
  steps_known: true, odd_row_shift: 0.5, shift_confirmed: false, has_valid_mask: false, n_valid: 30,
};

describe('parseScanGeometry', () => {
  it('reads a hexagonal scan field by field', () => {
    expect(parseScanGeometry(HEX)).toEqual({
      kind: 'hex', nRows: 6, nCols: 5, gridShape: [6, 5], stepX: 0.5, stepY: 0.4330127,
      stepsKnown: true, oddRowShift: 0.5, shiftConfirmed: false,
    });
  });

  it('reads a square scan and defaults the optional flags to "known"', () => {
    const g = parseScanGeometry({ kind: 'square', n_rows: 2, n_cols: 3, step_x: 1, step_y: 1 });
    expect(g).toMatchObject({ kind: 'square', gridShape: [2, 3], oddRowShift: 0, shiftConfirmed: true, stepsKnown: true });
  });

  it('answers null, never a guess, for anything it cannot read', () => {
    for (const bad of [null, undefined, 'hex', 3, {}, { kind: 'triangle', n_rows: 2, n_cols: 2 },
      { kind: 'hex', n_rows: 0, n_cols: 4 }, { kind: 'hex', n_rows: 'a', n_cols: 4 }]) {
      expect(parseScanGeometry(bad)).toBeNull();
    }
  });

  it('reads a step that is not a finite positive number as null, not NaN', () => {
    const base = { kind: 'square', n_rows: 2, n_cols: 3 };
    // missing, zero, negative, not a number, infinite, an array, a boolean, an empty string
    for (const bad of [undefined, null, 0, -1, 'abc', Infinity, NaN, [1], true, '']) {
      const g = parseScanGeometry({ ...base, step_x: bad, step_y: bad });
      expect(g.stepX).toBeNull();
      expect(g.stepY).toBeNull();
    }
    expect(parseScanGeometry({ ...base, step_x: 0.25, step_y: '0.5' })).toMatchObject({ stepX: 0.25, stepY: 0.5 });
  });

  it('reads a grid size only from a positive whole number or numeric string', () => {
    for (const bad of [[3], true, false, 2.5, -2, 0, '', ' ', '2.5', Infinity, NaN, {}, null, undefined]) {
      expect(parseScanGeometry({ kind: 'square', n_rows: bad, n_cols: 3 })).toBeNull();
      expect(parseScanGeometry({ kind: 'square', n_rows: 3, n_cols: bad })).toBeNull();
    }
    expect(parseScanGeometry({ kind: 'square', n_rows: '4', n_cols: 3 }).gridShape).toEqual([4, 3]);
  });

  it('keeps "the steps are placeholders" and "the offset is an assumption" visible', () => {
    const g = parseScanGeometry({ ...HEX, steps_known: false });
    expect(g.stepsKnown).toBe(false);
    expect(parseScanGeometry(HEX).shiftConfirmed).toBe(false);
  });
});

describe('isHexScan', () => {
  it('is true for hexagonal only; an unknown geometry is not hexagonal', () => {
    expect(isHexScan(parseScanGeometry(HEX))).toBe(true);
    expect(isHexScan(parseScanGeometry({ kind: 'square', n_rows: 2, n_cols: 2 }))).toBe(false);
    expect(isHexScan(null)).toBe(false);
    expect(isHexScan(undefined)).toBe(false);
  });
});
