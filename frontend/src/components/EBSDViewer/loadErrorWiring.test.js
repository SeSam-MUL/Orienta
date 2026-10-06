// @vitest-environment node
//
// The viewer's load error goes through problemText, so a coded refusal (a
// hexagonal EDAX UP1/UP2 scan) reads in the language of the page in the toast, the
// log and the error line, not only in the progress dialog. EBSDViewer cannot be
// mounted in a test, so this pins the one call and the four locale texts.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const SRC = readFileSync(
  fileURLToPath(new URL('./EBSDViewer.jsx', import.meta.url)), 'utf8');
const locale = (lang) => JSON.parse(readFileSync(
  fileURLToPath(new URL(`../../locales/${lang}/ebsdviewer.json`, import.meta.url)), 'utf8'));

describe('the viewer load error', () => {
  it('is worded by problemText', () => {
    const i = SRC.indexOf('setLoadError(msg)');
    expect(i).toBeGreaterThan(-1);
    expect(SRC.slice(Math.max(0, i - 400), i)).toMatch(/problemText\(err, t, 'ebsdviewer'\)/);
  });
  for (const lang of ['en', 'de', 'ja', 'zh']) {
    it(`${lang} has the text for edaxHexUpUnsupported`, () => {
      const text = locale(lang).errors?.edaxHexUpUnsupported;
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(20);
    });
  }
});
