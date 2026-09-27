/**
 * An API error in the reader's language, without breaking the 159 readers
 * that expect a plain string in `detail`.
 */
import { describe, it, expect } from 'vitest';
import { problemText, problemCode, problemProse } from './problemText';

const t = (key, opts = {}) => {
  const table = {
    'phasemap:errors.noResultOrDataset':
      'Kein Indizierungsergebnis und kein Analyse-Datensatz geladen.',
  };
  const hit = table[key];
  return hit === undefined ? (opts.defaultValue ?? key) : hit;
};

const err = (code, detail) => ({
  response: {
    headers: code ? { 'x-orienta-code': code } : {},
    data: detail === undefined ? {} : { detail },
  },
  message: 'Request failed',
});

describe('problemText', () => {
  it('translates on the header code', () => {
    expect(problemText(err('noResultOrDataset', 'No indexing result…'), t, 'phasemap'))
      .toBe('Kein Indizierungsergebnis und kein Analyse-Datensatz geladen.');
  });

  it('falls back to the backend sentence for a code with no text', () => {
    expect(problemText(err('somethingElse', 'Backend said this'), t, 'phasemap'))
      .toBe('Backend said this');
  });

  it('falls back for an older backend that sends no header at all', () => {
    expect(problemText(err(null, 'Backend said this'), t, 'phasemap'))
      .toBe('Backend said this');
  });

  it('falls back to the axios message when there is no response', () => {
    expect(problemText({ message: 'Network Error' }, t, 'phasemap')).toBe('Network Error');
    expect(problemText(null, t, 'phasemap')).toBe('');
  });

  it('reads a header from a fetch Headers object as well as from axios', () => {
    const withHeaders = {
      response: {
        headers: { get: (k) => (k === 'x-orienta-code' ? 'noResultOrDataset' : null) },
        data: { detail: 'No indexing result…' },
      },
    };
    expect(problemText(withHeaders, t, 'phasemap'))
      .toBe('Kein Indizierungsergebnis und kein Analyse-Datensatz geladen.');
  });

  it('does not print [object Object] when a route answers with a dict', () => {
    const dictDetail = {
      response: { headers: {}, data: { detail: { code: 'x', message: 'a sentence' } } },
    };
    expect(problemProse(dictDetail)).toBe('a sentence');
  });

  it('reports the code, or null', () => {
    expect(problemCode(err('abc', 'x'))).toBe('abc');
    expect(problemCode(err(null, 'x'))).toBe(null);
    expect(problemCode({})).toBe(null);
  });
});
