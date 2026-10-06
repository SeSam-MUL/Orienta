// @vitest-environment node
//
// The "Reflector rows" select counts ROWS of the reflector list (every
// symmetry-equivalent reflector, both signs, is one row), not families: PyEBSDIndex
// merges the rows into families afterwards. Its tooltip said "families" in the
// first sentence while the label next to it said rows. The tooltip is read from
// the real locale files, so a reworded or reverted text fails here.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const tip = (lang) => JSON.parse(readFileSync(
  fileURLToPath(new URL(`../../locales/${lang}/indexing.json`, import.meta.url)), 'utf8',
)).hoverTips.reflectorCount;

// The unit word of each language and what the first sentence must NOT count.
const ROW = { en: /rows/i, de: /Zeilen/i, ja: /行/, zh: /行/ };
const FAMILY_FIRST = {
  en: /^How many reflector families/,
  de: /^Wie viele Reflektorfamilien/,
  ja: /^この相が Hough のバンド三重項ライブラリに与える反射面ファミリーの数/,
  zh: /^该物相为 Hough 的带三元组库贡献多少个衍射面族/,
};

describe('hoverTips.reflectorCount', () => {
  for (const lang of ['en', 'de', 'ja', 'zh']) {
    it(`${lang}: the first sentence counts rows, not families`, () => {
      const first = tip(lang).split('\n')[0];
      expect(first).toMatch(ROW[lang]);
      expect(tip(lang)).not.toMatch(FAMILY_FIRST[lang]);
    });
  }
  it('en: says rows are symmetry-equivalent reflectors merged into families later', () => {
    expect(tip('en')).toMatch(/symmetry.equivalent/);
    expect(tip('en')).toMatch(/PyEBSDIndex/);
  });
});
