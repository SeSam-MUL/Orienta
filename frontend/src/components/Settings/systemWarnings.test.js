import { describe, it, expect } from 'vitest';
import { translateSystemWarnings, warningKey } from './systemWarnings';
import en from '../../locales/en/settings.json';
import de from '../../locales/de/settings.json';
import ja from '../../locales/ja/settings.json';
import zh from '../../locales/zh/settings.json';

/** A `t` that behaves like i18next: unknown key comes back as the key. */
function makeT(dict) {
  return (key, opts = {}) => {
    const path = key.replace(/^settings:/, '').split('.');
    let node = dict;
    for (const p of path) node = node?.[p];
    if (typeof node !== 'string') return key;
    return node.replace(/\{\{(\w+)\}\}/g, (_, name) =>
      (opts[name] === undefined ? `{{${name}}}` : String(opts[name])));
  };
}

// Shaped like the real backend payload (simulation/system_check.py): the one
// coded warning the backend sends today.
const ITEM = {
  code: 'emsoftNotOffered',
  values: {},
  message: "EMsoft is not set up on this computer. It is optional — Orienta's own simulation engine needs no EMsoft.",
};

describe('translateSystemWarnings', () => {
  it('replaces a coded warning with the translated sentence', () => {
    const out = translateSystemWarnings([ITEM.message], [ITEM], makeT(de));
    expect(out).toHaveLength(1);
    expect(out[0]).toContain('nicht eingerichtet');
    expect(out[0]).not.toBe(ITEM.message);
  });

  it('leaves prose-only warnings exactly as they are', () => {
    const prose = 'EMMCOpenCL not found';
    expect(translateSystemWarnings([prose], [ITEM], makeT(de))).toEqual([prose]);
  });

  it('keeps order and mixes both kinds', () => {
    const out = translateSystemWarnings(
      ['No GPU found — using CPU-only pipeline', ITEM.message], [ITEM], makeT(de));
    expect(out[0]).toBe('No GPU found — using CPU-only pipeline');
    expect(out[1]).toContain('nicht eingerichtet');
  });

  it('an unknown code falls back to the English message, never to the key', () => {
    const unknown = { code: 'somethingNew', values: {}, message: 'Backend says something new' };
    const out = translateSystemWarnings([unknown.message], [unknown], makeT(de));
    expect(out).toEqual([unknown.message]);
  });

  it('respects an exists() check when one is given', () => {
    const out = translateSystemWarnings([ITEM.message], [ITEM], makeT(de), () => false);
    expect(out).toEqual([ITEM.message]);
  });

  it('survives missing or malformed input', () => {
    expect(translateSystemWarnings(undefined, undefined, makeT(de))).toEqual([]);
    expect(translateSystemWarnings(['x'], null, makeT(de))).toEqual(['x']);
    expect(translateSystemWarnings(['x'], [null, {}], makeT(de))).toEqual(['x']);
  });
});

describe('the four languages carry the sentence', () => {
  it.each([['en', en], ['de', de], ['ja', ja], ['zh', zh]])(
    '%s has the code', (lang, dict) => {
      expect(dict.systemStatus?.warnings?.emsoftNotOffered, `${lang}.emsoftNotOffered`).toBeTruthy();
    });

  it('the retired GPU blocksize sentences are gone from every language', () => {
    // The backend no longer sends them (the master is always computed on the
    // CPU); a leftover string would be dead text in four files.
    for (const [lang, dict] of [['en', en], ['de', de], ['ja', ja], ['zh', zh]]) {
      const warnings = dict.systemStatus?.warnings ?? {};
      expect(warnings.gpuBlocksizeReduced, lang).toBeUndefined();
      expect(warnings.gpuBlocksizeMinimal, lang).toBeUndefined();
    }
  });

  it('each language says it in its own words', () => {
    const say = (dict) => translateSystemWarnings([ITEM.message], [ITEM], makeT(dict))[0];
    expect(new Set([en, de, ja, zh].map(say)).size).toBe(4);
  });

  it('the key helper matches where the sentences actually live', () => {
    expect(warningKey('emsoftNotOffered')).toBe('settings:systemStatus.warnings.emsoftNotOffered');
    expect(makeT(en)(warningKey('emsoftNotOffered'))).toContain('needs no EMsoft');
  });
});
