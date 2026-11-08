import { describe, it, expect } from 'vitest';
import { problemText } from './problemText';

const err = (code, detail = 'English prose') => ({
  response: { data: { detail }, headers: code ? { 'x-orienta-code': code } : {} },
});

// A stand-in for i18next's t: a dictionary of "ns:errors.code" -> text, defaultValue otherwise.
const makeT = (dict) => (key, opts) => (key in dict ? dict[key] : opts?.defaultValue);

describe('problemText falls back to the common namespace', () => {
  const t = makeT({
    'common:errors.hex_neighbourhood_unsupported': 'one sentence for every page',
    'phasemap:errors.pc_missing': 'the page\'s own sentence',
    'phasemap:errors.hex_neighbourhood_unsupported': 'the page says it differently',
  });

  it('uses the common text for a code the page has none for', () => {
    expect(problemText(err('hex_neighbourhood_unsupported'), t, 'indexing')).toBe('one sentence for every page');
  });

  it('prefers the page\'s own text over the common one', () => {
    expect(problemText(err('hex_neighbourhood_unsupported'), t, 'phasemap')).toBe('the page says it differently');
  });

  it('keeps the page\'s own codes working', () => {
    expect(problemText(err('pc_missing'), t, 'phasemap')).toBe('the page\'s own sentence');
  });

  it('still ends in the server\'s prose for a code nobody has a text for, and without a code', () => {
    expect(problemText(err('someFutureCode'), t, 'indexing')).toBe('English prose');
    expect(problemText(err(null), t, 'indexing')).toBe('English prose');
  });

  it('does not look in common twice when it is the namespace asked', () => {
    const seen = [];
    const spy = (key, opts) => { seen.push(key); return opts.defaultValue; };
    problemText(err('hex_pc_grid_pending'), spy, 'common');
    expect(seen).toEqual(['common:errors.hex_pc_grid_pending', 'ebsdviewer:errors.hex_pc_grid_pending']);
  });
});
