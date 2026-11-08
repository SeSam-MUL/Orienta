// @vitest-environment jsdom
// An APEX refusal (a pairing that conflicts with the pattern file, a metadata file that is gone) can
// come from any route that loads a file: the header switcher, the Indexing dropdown, the Phase Map
// gallery's activation. Their texts live with the viewer (errors.* of `ebsdviewer`), so a page that
// words a code in ITS namespace and then in `common` never found them and showed the backend's English
// sentence in a German window. The lookup is one function (`codedText`), and this goes through the
// real i18n instance with the real locale files.
import { describe, it, expect, afterEach } from 'vitest';
import i18n from '../i18n';
import { problemText, codedText, blobProblemText } from './problemText';
import viewer from '../locales/en/ebsdviewer.json';
import viewerDe from '../locales/de/ebsdviewer.json';
import viewerJa from '../locales/ja/ebsdviewer.json';
import viewerZh from '../locales/zh/ebsdviewer.json';
import commonEn from '../locales/en/common.json';

const PROSE = 'BACKEND ENGLISH SENTENCE';
const err = (code) => ({
  response: { data: { detail: PROSE }, headers: code ? { 'x-orienta-code': code } : {} },
});
const TEXT = { en: viewer, de: viewerDe, ja: viewerJa, zh: viewerZh };
const CODES = ['apexGuidMismatch', 'apexUnreadable', 'apexPairingLost', 'apexPatternCountMismatch'];
const PAGES = ['indexing', 'phasemap', 'analysis', 'pcrefinement', 'common'];

afterEach(async () => { await i18n.changeLanguage('en'); });

describe.each(Object.keys(TEXT))('in %s', (lang) => {
  it.each(PAGES)('a page whose namespace is %s words an APEX code, not the backend sentence', async (ns) => {
    await i18n.changeLanguage(lang);
    const t = i18n.getFixedT(lang, ns);
    for (const code of CODES) {
      const want = TEXT[lang].errors[code];
      expect(want, `${lang} has no text for ${code}`).toBeTruthy();
      expect(problemText(err(code), t, ns)).toBe(want);
      expect(codedText(code, t, ns)).toBe(want);
    }
  });

  it('words the code of a file download (blob) refusal the same way', async () => {
    await i18n.changeLanguage(lang);
    const t = i18n.getFixedT(lang, 'indexing');
    expect(await blobProblemText(err('apexUnreadable'), t, 'indexing')).toBe(TEXT[lang].errors.apexUnreadable);
  });
});

describe('the order of the lookup', () => {
  it('a text of the page itself still wins over the shared ones', async () => {
    await i18n.changeLanguage('en');
    i18n.addResourceBundle('en', 'ownpage', { errors: { apexUnreadable: 'this page says it its own way' } });
    try {
      expect(problemText(err('apexUnreadable'), i18n.getFixedT('en', 'ownpage'), 'ownpage'))
        .toBe('this page says it its own way');
    } finally {
      i18n.removeResourceBundle('en', 'ownpage');
    }
  });

  it('the shared load sentences in common still win over the viewer\'s namespace', async () => {
    await i18n.changeLanguage('en');
    expect(problemText(err('loadVariantUnknown'), i18n.getFixedT('en', 'indexing'), 'indexing'))
      .toBe(commonEn.errors.loadVariantUnknown);
  });

  it('a code nobody has a text for ends in the backend sentence, and so does no code', () => {
    const t = i18n.getFixedT('en', 'indexing');
    expect(problemText(err('someFutureCode'), t, 'indexing')).toBe(PROSE);
    expect(problemText(err(null), t, 'indexing')).toBe(PROSE);
  });
});
