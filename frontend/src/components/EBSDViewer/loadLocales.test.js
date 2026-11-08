// @vitest-environment node
//
// A refusal of /load (a measurement or pattern source the file does not hold, a file without
// patterns, a variant only kikuchipy's own reader can read) travels as a code in X-Orienta-Code.
// The same codes now come from more routes than the viewer's load (the Indexing start and export,
// the crop export, the Phase Map layers, the Analysis page), so their four sentences are kept once,
// under errors.* of the `common` locale, which every page's namespace falls back to
// (services/problemText.js); the viewer keeps no second copy.
// The list of codes comes from the backend (fixtures/loadCodes.json is generated from
// load_choices.LOAD_CODES, and tests/test_load_variant_routes.py holds the fixture against the
// backend and the backend against the source, so it cannot go stale). A code the backend has and a
// locale lacks turns this red.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const read = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');
const json = (rel) => JSON.parse(read(rel));
const LANGS = ['en', 'de', 'ja', 'zh'];
const locale = (lang, ns = 'common') => json(`../../locales/${lang}/${ns}.json`);
const CODES = json('./__fixtures__/loadCodes.json').codes;

describe('the fixture is not empty (a test over an empty list proves nothing)', () => {
  it('has the load codes', () => {
    expect(CODES.length).toBeGreaterThanOrEqual(3);
    expect(CODES).toContain('loadVariantNeedsNativeReader');
    expect(CODES).toContain('variantNotSupported');
  });
});

describe.each(LANGS)('%s', (lang) => {
  const errors = locale(lang).errors;

  it('words every load refusal code under common errors.*', () => {
    const missing = CODES.filter((c) => typeof errors?.[c] !== 'string' || errors[c].trim().length < 10);
    expect(missing).toEqual([]);
  });

  it('has no text for a load code the backend no longer has', () => {
    const have = Object.keys(errors).filter((k) => /^load[A-Z]/.test(k) || k === 'variantNotSupported');
    expect(have.filter((k) => !CODES.includes(k))).toEqual([]);
  });

  it('keeps no second copy in the viewer namespace (one text, shown everywhere)', () => {
    const viewer = locale(lang, 'ebsdviewer').errors;
    expect(CODES.filter((c) => c in viewer)).toEqual([]);
  });
});

describe('German addresses the user formally', () => {
  it('never says du/dein', () => {
    const d = locale('de').errors;
    const offenders = CODES.filter((c) => /\b(du|dein\w*|dich|dir)\b/i.test(d[c]));
    expect(offenders).toEqual([]);
  });

  it('says "Sie" where it gives an instruction', () => {
    const d = locale('de').errors;
    expect(d.loadVariantNeedsNativeReader).toMatch(/\bSie\b/);
    expect(d.loadVariantUnknown).toMatch(/\bSie\b/);
  });
});

describe('Chinese punctuation', () => {
  it('uses full-width commas and colons in the load texts, as the neighbouring keys do', () => {
    const z = locale('zh').errors;
    expect(CODES.filter((c) => /[,:;]/.test(z[c]))).toEqual([]);
  });
});
