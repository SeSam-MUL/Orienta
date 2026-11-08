// @vitest-environment node
//
// `POST /api/ebsd/crop` refuses a dataset without a recorded scan geometry with the code
// `crop_geometry_unknown`. The viewer worded that refusal from the server's English prose; it now
// goes through problemText, so the user reads the sentence in the language of the page. EBSDViewer
// cannot be mounted in a test, so this pins the one call, runs the refusal through the real
// problemText with the real locale files, and checks the four texts.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { problemText } from '../../../services/problemText';

const read = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');
const SRC = read('../EBSDViewer.jsx');
const locale = (lang, ns) => JSON.parse(read(`../../../locales/${lang}/${ns}.json`));

const refusal = {
  response: {
    headers: { 'x-orienta-code': 'crop_geometry_unknown' },
    data: { detail: 'English prose from the server' },
  },
};

// i18next's t over the real locale files: "ns:errors.code" -> text, defaultValue when missing.
const makeT = (lang) => (key, opts) => {
  const [ns, path] = key.split(':');
  let node = locale(lang, ns);
  for (const part of path.split('.')) node = node?.[part];
  return typeof node === 'string' ? node : opts?.defaultValue;
};

describe('a crop refused for a missing scan geometry', () => {
  it('is worded by problemText in the crop handler, not from the raw detail', () => {
    const start = SRC.indexOf("log(t('crop.failed'");
    expect(start).toBeGreaterThan(-1);
    const call = SRC.slice(start, SRC.indexOf('));', start));
    expect(call).toContain("problemText(e, t, 'ebsdviewer')");
    expect(call).not.toContain('response?.data?.detail');
  });

  const expected = {
    en: /scan geometry/i,
    de: /Scan-Geometrie|Scangeometrie/i,
    ja: /スキャン/,
    zh: /扫描/,
  };
  for (const lang of ['en', 'de', 'ja', 'zh']) {
    it(`${lang}: the refusal reads in the page's language, not in the server's prose`, () => {
      const text = problemText(refusal, makeT(lang), 'ebsdviewer');
      expect(text).not.toBe('English prose from the server');
      expect(text).toBe(locale(lang, 'common').errors.crop_geometry_unknown);
      expect(text).toMatch(expected[lang]);
    });
  }
});
