// @vitest-environment jsdom
// The Analysis page refuses all its maps on a hexagonal dataset with its own code. The code must come
// out as a sentence in every language of the application, through the real i18n instance and the real
// locale files (not a stub of `t`), and the sentence must not say the maps compare neighbours: bc and
// ipf_* compare nothing, the page just draws them on the index grid.
import { describe, it, expect, afterEach } from 'vitest';
import i18n from '../i18n';
import { codedProblemText, problemText } from './problemText';
import commonEn from '../locales/en/common.json';

const CODE = 'hex_analysis_page_pending';
const PROSE = 'BACKEND ENGLISH SENTENCE';
const err = (code) => ({ response: { data: { detail: PROSE }, headers: { 'x-orienta-code': code } } });
const LANGS = ['en', 'de', 'ja', 'zh'];

afterEach(async () => { await i18n.changeLanguage('en'); });

describe('hex_analysis_page_pending', () => {
  it.each(LANGS)('is worded in %s, on the page that asks and on the shared namespace, not as a key or as the backend prose', async (lang) => {
    await i18n.changeLanguage(lang);
    for (const ns of ['analysis', 'common']) {
      const text = codedProblemText(err(CODE), i18n.getFixedT(lang, ns), ns);
      expect(text, `${lang}/${ns}`).toBeTruthy();
      expect(text).not.toContain(CODE);
      expect(text).not.toContain('errors.');
      expect(text).not.toBe(PROSE);
      expect(text.length).toBeGreaterThan(40);
      // i18n falls back to English, so a missing translation would still be "a text"
      if (lang !== 'en') expect(text, `${lang}/${ns} is the English fallback`).not.toBe(commonEn.errors[CODE]);
    }
  });

  it('gives four different sentences', async () => {
    const seen = new Set();
    for (const lang of LANGS) {
      await i18n.changeLanguage(lang);
      seen.add(problemText(err(CODE), i18n.getFixedT(lang, 'analysis'), 'analysis'));
    }
    expect(seen.size).toBe(4);
  });

  it('says in English that the page cannot draw the maps, and names no neighbours', async () => {
    await i18n.changeLanguage('en');
    const text = problemText(err(CODE), i18n.getFixedT('en', 'analysis'), 'analysis');
    expect(text).toBe('The maps of the Analysis page cannot be drawn for a hexagonal scan yet: the page draws every map on the index grid, which puts every second row half a step off.');
    expect(text).not.toMatch(/neighbour|compares/);
  });

  it('the old neighbourhood sentence is still the one for what really compares neighbours', async () => {
    await i18n.changeLanguage('en');
    const old = problemText(err('hex_neighbourhood_unsupported'), i18n.getFixedT('en', 'analysis'), 'analysis');
    expect(old).toMatch(/four neighbours/);
  });
});
