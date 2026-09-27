import { describe, it, expect } from 'vitest';
import { reasonKey, describeFailure } from './addonReasons';

// A stand-in for i18next's `t`: it reports what was asked for, so a test can
// tell "looked it up and got a sentence" apart from "printed the key".
const t = (key, opts) => {
  const table = {
    'addons:reason.addon_not_enabled': 'This add-on is switched off',
    'addons:reason.addon_failed': 'The analysis failed',
    'addons:failure.unknownTitle': 'The request was refused',
    'addons:failure.noDetail': 'The server gave no further explanation.',
  };
  if (key in table) return table[key];
  return opts && 'defaultValue' in opts ? opts.defaultValue : key;
};

describe('reasonKey', () => {
  it('turns a backend code into a namespaced key', () => {
    expect(reasonKey('addon_not_enabled')).toBe('addons:reason.addon_not_enabled');
  });

  it('is null for nothing at all, so a caller can tell', () => {
    expect(reasonKey(undefined)).toBe(null);
    expect(reasonKey(null)).toBe(null);
    expect(reasonKey('')).toBe(null);
  });

  it('refuses a code that is not one, instead of building a key from it', () => {
    // These arrive from the wire. A key built out of a path or a sentence
    // would be looked up, missed, and printed raw at the user.
    expect(reasonKey('reason.with.dots')).toBe(null);
    expect(reasonKey('Not a code')).toBe(null);
    expect(reasonKey(42)).toBe(null);
  });
});

describe('describeFailure', () => {
  it('titles a known code and keeps the server’s sentence as the detail', () => {
    const out = describeFailure(
      { reason: 'addon_not_enabled', detail: 'Enable it with POST …' }, t);
    expect(out.title).toBe('This add-on is switched off');
    expect(out.detail).toBe('Enable it with POST …');
  });

  it('keeps the add-on’s OWN words, which no locale can carry', () => {
    // The detail of a failed run is the add-on's message -- the only thing in
    // the response that says what actually went wrong. Replacing it with a
    // translated summary would throw away the one sentence worth reading.
    const detail = 'bc-gmm / addon.bc_gmm: ValueError: fewer pixels than components';
    const out = describeFailure({ reason: 'addon_failed', detail }, t);
    expect(out.title).toBe('The analysis failed');
    expect(out.detail).toBe(detail);
  });

  it('falls back to the server’s detail for a code it does not know', () => {
    // A twelfth code shipped by a newer backend must still say something
    // true. A generic apology would hide a sentence the server wrote.
    const out = describeFailure(
      { reason: 'brand_new_code', detail: 'Something specific happened.' }, t);
    expect(out.title).toBe('Something specific happened.');
    // And NOT again as the detail: a panel printing both said the same
    // sentence twice, the second line pretending to add information.
    expect(out.detail).toBe('');
  });

  it('is still usable when there is no reason at all', () => {
    const out = describeFailure({ detail: 'Plain HTTP error text' }, t);
    expect(out.title).toBe('Plain HTTP error text');
    expect(out.detail).toBe('');
  });

  it('never returns an empty TITLE, whatever it is given', () => {
    // Rendered into a panel. An empty headline is a blank box that looks like
    // a rendering bug. An empty DETAIL is fine — the caller renders nothing
    // for it, which is how the duplicated-sentence bug was fixed.
    for (const failure of [{}, { reason: '' }, { detail: '' }, null, undefined]) {
      const out = describeFailure(failure, t);
      expect(out.title.length).toBeGreaterThan(0);
      expect(typeof out.detail).toBe('string');
    }
  });

  it('with nothing at all, says it was refused AND that nothing was said', () => {
    const out = describeFailure({}, t);
    expect(out.title).toBe('The request was refused');
    expect(out.detail).toBe('The server gave no further explanation.');
  });

  it('does not print a raw key when a locale is missing the string', () => {
    // t() falling through returns the key; showing "addons:reason.x" to a
    // user is worse than showing the English sentence the server sent.
    const bare = (key, opts) => (opts && 'defaultValue' in opts
      ? opts.defaultValue : key);
    const out = describeFailure(
      { reason: 'addon_not_enabled', detail: 'The server sentence.' }, bare);
    expect(out.title).not.toContain('addons:');
    expect(out.title).toBe('The server sentence.');
  });
});
