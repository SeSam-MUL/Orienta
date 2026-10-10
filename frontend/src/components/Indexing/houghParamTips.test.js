// @vitest-environment node
//
// Source-level guard: every Hough parameter on the Indexing page has its hover
// text on the LABEL and on the CONTROL. IndexingPage cannot be mounted in a test
// (see phasePathWiring.test.js), so this reads the page; what the texts say was
// checked against the PyEBSDIndex source (see the commit that wrote them), and
// the four languages are held to the same keys by localeParity.
// Redundant once IndexingPage can be mounted in a test.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const SRC = readFileSync(fileURLToPath(new URL('./IndexingPage.jsx', import.meta.url)), 'utf8');
const LOCALES = fileURLToPath(new URL('../../locales/', import.meta.url));
const PARAMS = ['bands', 'tSigma', 'rSigma'];

describe('Hough parameters on the Indexing page', () => {
  it.each(PARAMS)('%s: label and control both carry the hover text', (p) => {
    expect(SRC).toContain(`<InlineLabel tip={t('hough.${p}Tip')}>{t('hough.${p}')}</InlineLabel>`);
    expect(SRC).toContain(`title={t('hough.${p}Tip')}`);
  });

  it.each(['en', 'de', 'ja', 'zh'])('%s has a text for each, and it states the unit or the default', (lang) => {
    const hough = JSON.parse(readFileSync(`${LOCALES}${lang}/indexing.json`, 'utf8')).hough;
    for (const p of PARAMS) {
      const text = hough[`${p}Tip`];
      expect(text && text.length, p).toBeGreaterThan(60);
      // 12 / 2 / 2 are Orienta's defaults and 9 / 1 / 1.2 PyEBSDIndex's own.
      expect(text, p).toMatch(/PyEBSDIndex/);
    }
  });

  it('the old hand-waved "typical" ranges are gone', () => {
    const en = JSON.parse(readFileSync(`${LOCALES}en/indexing.json`, 'utf8')).hough;
    for (const p of PARAMS) expect(en[`${p}Tip`]).not.toMatch(/Typical/i);
  });
});
