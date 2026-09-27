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

// Shaped like the real backend payload: `gb` is a preformatted string, so a
// whole number cannot lose its decimal on the way through JSON + i18next.
const ITEM = {
  code: 'gpuBlocksizeMinimal',
  values: { gb: '7.9', blocksize: 8 },
  message: 'GPU has 7.9 GB — using blocksize=8 for EMEBSDmasterOpenCL. Complex crystals (>20 atoms) may still fail.',
};

const WHOLE = {
  code: 'gpuBlocksizeReduced',
  values: { gb: '8.0', blocksize: 16 },
  message: 'GPU has 8.0 GB — using blocksize=16 for EMEBSDmasterOpenCL (reduced from 32).',
};

describe('translateSystemWarnings', () => {
  it('replaces a coded warning with the translated sentence', () => {
    const out = translateSystemWarnings([ITEM.message], [ITEM], makeT(de));
    expect(out).toHaveLength(1);
    expect(out[0]).toContain('Blockgröße 8');
    expect(out[0]).toContain('7.9 GB');
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
    expect(out[1]).toContain('Blockgröße');
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

describe('the four languages carry both sentences', () => {
  const codes = ['gpuBlocksizeReduced', 'gpuBlocksizeMinimal'];

  it.each([['en', en], ['de', de], ['ja', ja], ['zh', zh]])(
    '%s has every code, with both placeholders', (lang, dict) => {
      for (const code of codes) {
        const text = dict.systemStatus?.warnings?.[code];
        expect(text, `${lang}.${code}`).toBeTruthy();
        expect(text, `${lang}.${code} must name the memory`).toContain('{{gb}}');
        expect(text, `${lang}.${code} must name the blocksize`).toContain('{{blocksize}}');
      }
    });

  it('a whole number keeps its decimal all the way to the screen', () => {
    // The GPU-memory row one line away prints toFixed(1); a JSON 8.0 would
    // arrive here as `8` and the two lines would disagree.
    for (const [lang, dict] of [['en', en], ['de', de], ['ja', ja], ['zh', zh]]) {
      const out = translateSystemWarnings([WHOLE.message], [WHOLE], makeT(dict));
      expect(out[0], lang).toContain('8.0');
    }
  });

  it('each language says it in its own words', () => {
    const say = (dict) => translateSystemWarnings([ITEM.message], [ITEM], makeT(dict))[0];
    expect(new Set([en, de, ja, zh].map(say)).size).toBe(4);
  });

  it('the key helper matches where the sentences actually live', () => {
    expect(warningKey('gpuBlocksizeMinimal')).toBe('settings:systemStatus.warnings.gpuBlocksizeMinimal');
    expect(makeT(en)(warningKey('gpuBlocksizeMinimal'), { gb: 7.9, blocksize: 8 }))
      .toContain('7.9 GB');
  });
});
