// @vitest-environment jsdom
/**
 * `export_other_dataset`: a rich .h5 whose header and patterns are not the scan the result was
 * computed on is refused with this code. It is worded from `common.errors` in all four languages,
 * so the Indexing and Phase Map pages (whose namespaces do not hold it) find it.
 */
import { describe, it, expect } from 'vitest';
import { blobProblemText } from './problemText';
import en from '../locales/en/common.json';
import de from '../locales/de/common.json';
import ja from '../locales/ja/common.json';
import zh from '../locales/zh/common.json';

const CODE = 'export_other_dataset';
const LOCALES = { en, de, ja, zh };

const tOf = (lang) => (key, opts) => {
  const [ns, path] = key.split(':');
  if (ns !== 'common') return opts?.defaultValue;
  return path.split('.').reduce((node, part) => node?.[part], LOCALES[lang]) ?? opts?.defaultValue;
};

const refusal = () => ({
  message: 'Request failed with status code 400',
  response: {
    status: 400,
    headers: { 'x-orienta-code': CODE },
    data: new Blob([JSON.stringify({ detail: 'English prose of the server' })], { type: 'application/json' }),
  },
});

describe(`the code ${CODE}`, () => {
  for (const lang of Object.keys(LOCALES)) {
    it(`${lang}: has a sentence that says what to do and that nothing was written`, () => {
      const text = LOCALES[lang].errors[CODE];
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(60);
    });

    it(`${lang}: the Indexing export shows that sentence, not the server's prose`, async () => {
      const shown = await blobProblemText(refusal(), tOf(lang), 'indexing');
      expect(shown).toBe(LOCALES[lang].errors[CODE]);
      expect(shown).not.toContain('English prose');
    });
  }

  it('de: addresses the reader with Sie', () => {
    expect(de.errors[CODE]).toMatch(/\bWählen Sie\b/);
    expect(de.errors[CODE]).toMatch(/\bspeichern Sie\b/);
    expect(de.errors[CODE]).not.toMatch(/\b(wähle|speichere|dein|deine)\b/i);
  });
});
