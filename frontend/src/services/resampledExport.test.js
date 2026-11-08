import { describe, it, expect } from 'vitest';
import { readResampled, EXPORT_FILE_NAMES } from './resampledExport';

// The header as the backend writes it: ASCII JSON, compact separators.
const HEADER = '{"step_um":0.5,"rows":7,"cols":9,"source_points":60,"points":63,"unused":2,"shift_confirmed":false}';

describe('readResampled', () => {
  it('reads the numbers of a plain header object, whichever way the name is spelled', () => {
    for (const name of ['x-orienta-resampled', 'X-Orienta-Resampled']) {
      expect(readResampled({ headers: { [name]: HEADER } })).toEqual({
        stepUm: 0.5, rows: 7, cols: 9, sourcePoints: 60, points: 63, unused: 2, shiftConfirmed: false,
      });
    }
  });

  it('reads it from a headers object with get() (what axios gives)', () => {
    const headers = { get: (n) => (n.toLowerCase() === 'x-orienta-resampled' ? HEADER : undefined) };
    expect(readResampled({ headers })).toMatchObject({ rows: 7, cols: 9 });
  });

  it.each([
    ['no header', { headers: {} }],
    ['no response', undefined],
    ['no headers', {}],
    ['text that is not JSON', { headers: { 'x-orienta-resampled': 'rows=7' } }],
    ['JSON without a grid', { headers: { 'x-orienta-resampled': '{"step_um":0.5}' } }],
    ['a grid of zero rows', { headers: { 'x-orienta-resampled': '{"rows":0,"cols":4}' } }],
    ['a fractional grid', { headers: { 'x-orienta-resampled': '{"rows":2.5,"cols":4}' } }],
  ])('says null for %s, never a made-up grid', (_name, response) => {
    expect(readResampled(response)).toBeNull();
  });

  it('keeps a missing optional number as null, not 0', () => {
    expect(readResampled({ headers: { 'x-orienta-resampled': '{"rows":2,"cols":3}' } }))
      .toEqual({ stepUm: null, rows: 2, cols: 3, sourcePoints: null, points: null, unused: null, shiftConfirmed: true });
  });
});

describe('EXPORT_FILE_NAMES', () => {
  it('are the names the backend gives the files (the end of its default name)', () => {
    expect(EXPORT_FILE_NAMES).toEqual({
      h5: 'indexing_result.h5', ang: 'indexing_result.ang',
      ang_square: 'indexing_result_square.ang', ctf: 'indexing_result.ctf',
    });
  });
});
