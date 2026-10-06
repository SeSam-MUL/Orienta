// @vitest-environment node
//
// The pure part of the reflector-family table: what a click turns the choice
// into, how a coded backend error is worded, how a size is written.
import { describe, it, expect } from 'vitest';
import {
  specKey, tickedFamilies, specAfterToggle, specAfterAdd, currentSpec,
  reflectorError, formatSize, summaryKey,
} from './reflectorSpec';

const fam = (hkl, selected, extra = {}) => ({ hkl, selected, effective: selected, ...extra });
const TABLE = {
  families: [
    fam([1, 1, 1], true), fam([2, 0, 0], true), fam([2, 2, 0], true),
    fam([2, 2, 2], true, { effective: false }), fam([4, 2, 2], false),
  ],
};

describe('which families a click leaves ticked', () => {
  it('lists the ticked families, parallel-ignored ones included (they ARE ticked)', () => {
    expect(tickedFamilies(TABLE)).toEqual([[1, 1, 1], [2, 0, 0], [2, 2, 0], [2, 2, 2]]);
  });

  it('unticking removes exactly that family and makes the choice explicit', () => {
    expect(specAfterToggle(TABLE, [1, 1, 1])).toEqual({
      mode: 'custom', families: [[2, 0, 0], [2, 2, 0], [2, 2, 2]],
    });
  });

  it('ticking adds it at the end', () => {
    expect(specAfterToggle(TABLE, [4, 2, 2]).families.at(-1)).toEqual([4, 2, 2]);
  });

  it('adding a family twice does not repeat it', () => {
    expect(specAfterAdd(TABLE, [1, 1, 1]).families).toHaveLength(4);
    expect(specAfterAdd(TABLE, [3, 3, 1]).families).toHaveLength(5);
  });

  it('the spec sent for duplicate checks is the ticked list', () => {
    expect(currentSpec(TABLE)).toEqual({ mode: 'custom', families: tickedFamilies(TABLE) });
    expect(currentSpec(null)).toBeNull();
  });
});

describe('errors', () => {
  // A stand-in for i18next: returns the key with its params so the test sees
  // which key was asked for; '' for the keys it "does not know".
  const known = new Set(['reflectorFamilies.errors.forbidden', 'reflectorFamilies.errors.too_few',
    'reflectorFamilies.errors.generic']);
  const t = (key, o = {}) => {
    if (!known.has(key)) return o.defaultValue ?? key;
    const { defaultValue, ...p } = o;
    return `${key}${Object.keys(p).length ? JSON.stringify(p) : ''}`;
  };
  const axiosError = (detail) => ({ response: { data: { detail } } });

  it('uses the translated text of the code, with its parameters', () => {
    const txt = reflectorError(t, axiosError({ code: 'forbidden', message: 'x', params: { hkl: [1, 0, 0] } }));
    expect(txt).toBe('reflectorFamilies.errors.forbidden{"hkl":"{1 0 0}"}');
  });

  it('falls back to the backend message for a code it has no text for', () => {
    expect(reflectorError(t, axiosError({ code: 'brand_new', message: 'The server says', params: {} })))
      .toBe('The server says');
  });

  it('a plain string detail or a network error is a generic message', () => {
    expect(reflectorError(t, axiosError('boom'))).toBe('boom');
    expect(reflectorError(t, new Error('Network Error'))).toBe('Network Error');
    expect(reflectorError(t, {})).toBe('reflectorFamilies.errors.generic');
  });

  it('a coded object that is not an axios error (the stale spec of a table) works too', () => {
    expect(reflectorError(t, { code: 'too_few', message: 'm', params: { kept: 1 } }))
      .toBe('reflectorFamilies.errors.too_few{"kept":1}');
  });
});

describe('small helpers', () => {
  it('the registry key is the lower-case CIF stem, from a path or a name', () => {
    expect(specKey('C:\\lib\\Al7FeCu2.cif')).toBe('al7fecu2');
    expect(specKey('/home/x/Ni.CIF')).toBe('ni');
    expect(specKey('austenite')).toBe('austenite');
  });

  it('sizes are written for comparison', () => {
    expect(formatSize(0)).toBeNull();
    expect(formatSize(null)).toBeNull();
    expect(formatSize(3 * 1024 ** 2)).toBe('3 MiB');
    expect(formatSize(1.449 * 1024 ** 3)).toBe('1.45 GiB');
  });

  it('the headline names the three states', () => {
    expect(summaryKey('custom')).toBe('summaryCustom');
    expect(summaryKey('auto')).toBe('summaryAuto');
    expect(summaryKey('default')).toBe('summaryDefault');
    expect(summaryKey(undefined)).toBe('summaryDefault');
  });
});
