// @vitest-environment jsdom
/**
 * The load refusals (`variantNotSupported` and its three neighbours) reach the screen on many
 * pages: the Indexing start and its export dropdown (a blob request), the Phase Map result export,
 * the viewer's crop save and file switch, the progress window of a load, the Analysis page. Each
 * surface has to put the translated sentence in front of the user, in every language, instead of
 * the backend's English prose or "Request failed with status code 400".
 *
 * The four sentences live once, in `common:errors.*`, which every page's namespace falls back to.
 * The pages are too heavy to mount here, so each surface is pinned in two halves: the real
 * lookup (real locale files, an axios-shaped error with the code header) and the call in the page
 * source that makes it.
 */
import '@testing-library/jest-dom/vitest';
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import i18n from '../../../i18n';
import LoadProgressModal from '../LoadProgressModal';
import { blobProblemText, codedText, problemText } from '../../../services/problemText';

afterEach(() => cleanup());

const read = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');
const locale = (lang, ns) => JSON.parse(read(`../../../locales/${lang}/${ns}.json`));
const LANGS = ['en', 'de', 'ja', 'zh'];
const CODES = JSON.parse(read('../__fixtures__/loadCodes.json')).codes;

// i18next's t over the real locale files: "ns:errors.code" -> text, defaultValue when missing.
const makeT = (lang) => (key, opts) => {
  const [ns, path] = key.split(':');
  let node = locale(lang, ns);
  for (const part of path.split('.')) node = node?.[part];
  return typeof node === 'string' ? node : opts?.defaultValue;
};

const PROSE = 'start_indexing cannot read the processed patterns; English prose from the server';
const coded = (code) => ({
  message: 'Request failed with status code 400',
  response: { status: 400, headers: { 'x-orienta-code': code }, data: { detail: PROSE } },
});
const codedBlob = (code) => ({
  message: 'Request failed with status code 400',
  response: {
    status: 400,
    headers: { 'x-orienta-code': code },
    data: new Blob([JSON.stringify({ detail: PROSE })], { type: 'application/json' }),
  },
});

/** The source of `file` from the first occurrence of `anchor` for `span` characters. */
const around = (file, anchor, span = 700) => {
  const src = read(file);
  const i = src.indexOf(anchor);
  expect(i, `${anchor} is in ${file}`).toBeGreaterThan(-1);
  return src.slice(i, i + span);
};

describe('one text, in common, for every page', () => {
  for (const lang of LANGS) {
    it(`${lang}: every load code is worded in common and not duplicated in ebsdviewer`, () => {
      const common = locale(lang, 'common').errors;
      const viewer = locale(lang, 'ebsdviewer').errors;
      for (const code of CODES) {
        expect(typeof common[code], `${lang} common ${code}`).toBe('string');
        expect(common[code].trim().length).toBeGreaterThan(20);
        expect(viewer[code], `${lang} ebsdviewer must not keep a second copy of ${code}`).toBeUndefined();
      }
    });

    for (const ns of ['indexing', 'ebsdviewer', 'phasemap', 'analysis', 'common']) {
      it(`${lang}/${ns}: variantNotSupported reads translated, never as the server's prose`, () => {
        const text = problemText(coded('variantNotSupported'), makeT(lang), ns);
        expect(text).toBe(locale(lang, 'common').errors.variantNotSupported);
        expect(text).not.toMatch(/English prose|start_indexing/);
      });
    }
  }

  it('zh words a diffraction pattern as 花样 in all four sentences, as the rest of the app does', () => {
    const z = locale('zh', 'common').errors;
    for (const code of CODES) expect(z[code], code).not.toContain('图案');
    expect(z.variantNotSupported).toContain('花样来源');
  });

  it('de says Sie', () => {
    const d = locale('de', 'common').errors;
    expect(d.variantNotSupported).toMatch(/\bSie\b/);
    expect(d.loadVariantUnknown).toMatch(/\bSie\b/);
    for (const code of CODES) expect(d[code]).not.toMatch(/\b(du|dein\w*|dich|dir)\b/i);
  });
});

describe('Indexing page', () => {
  it('start: a coded refusal goes through problemText with the indexing namespace', () => {
    expect(around('../../Indexing/IndexingPage.jsx', "await indexApi.start(params)", 900))
      .toMatch(/problemText\(err, t, 'indexing'\)/);
  });

  it('export dropdown: reads the refusal of a blob request with blobProblemText, not the raw detail', () => {
    const handler = around('../../Indexing/IndexingPage.jsx', 'indexApi.exportResult(fmt', 1500);
    expect(handler).toMatch(/blobProblemText\(err, t, 'indexing'\)/);
    expect(handler).not.toContain('err.response?.data?.detail');
  });

  for (const lang of LANGS) {
    it(`${lang}: a refused export (blob body) shows the translated sentence`, async () => {
      const text = await blobProblemText(codedBlob('variantNotSupported'), makeT(lang), 'indexing');
      expect(text).toBe(locale(lang, 'common').errors.variantNotSupported);
      expect(text).not.toMatch(/English prose|Request failed/);
    });
  }

  it('a run that ends on a refusal inside the background task shows its code translated', () => {
    // the poll of /status: the task state carries error_code (a LoadChoiceError raised in the run)
    expect(around('../../Indexing/IndexingPage.jsx', "t('messages.indexingFailed'", 400))
      .toMatch(/codedText\(d\.error_code, t, 'indexing'\) \|\| d\.error/);
    for (const lang of LANGS) {
      const text = codedText('variantNotSupported', makeT(lang), 'indexing');
      expect(text).toBe(locale(lang, 'common').errors.variantNotSupported);
    }
    expect(codedText(undefined, makeT('en'), 'indexing')).toBeNull();     // no code: the prose is shown
  });

  it('a refused export without a code shows the JSON detail of the blob, not the axios message', async () => {
    const err = codedBlob(undefined);
    err.response.headers = {};
    expect(await blobProblemText(err, makeT('en'), 'indexing')).toBe(PROSE);
  });
});

describe('Phase Map', () => {
  it('result export (a blob request) uses blobProblemText', () => {
    const handler = around('../../PhaseMap/PhaseMapPage.jsx', "responseType: 'blob'", 500);
    expect(handler).toMatch(/blobProblemText\(err, t, 'phasemap'\)/);
  });

  it('activating a result shows the coded text', () => {
    expect(around('../../PhaseMap/PhaseMapPage.jsx', "console.warn('[PhaseMap] activating result failed", 400))
      .toMatch(/problemText\(err, t, 'phasemap'\)/);
  });

  it('a layer fetch (band contrast) is worded by problemText with the phasemap namespace', () => {
    expect(around('../../PhaseMap/hooks/useLayerStack.js', 'Translated here rather than at the display', 600))
      .toMatch(/problemText\(err, tRef\.current, 'phasemap'\)/);
  });
});

describe('EBSD viewer', () => {
  it('crop save uses problemText, not the raw detail', () => {
    const handler = around('../EBSDViewer.jsx', "log(t('crop.saveCropFailed'", 200);
    expect(handler).toMatch(/problemText\(e, t, 'ebsdviewer'\)/);
    expect(handler).not.toContain('response?.data?.detail');
  });

  it('switching the loaded file uses problemText, not the raw detail', () => {
    const src = read('../EBSDViewer.jsx');
    const i = src.indexOf("t('logMessages.switchFailedDefault')");
    expect(src.slice(i - 200, i + 40)).toMatch(/problemText\(err, t, 'ebsdviewer'\)/);
  });
});

describe('Analysis page', () => {
  it('every reader that can meet a BC-quality refusal puts the coded text in front', () => {
    const src = read('../../Analysis/AnalysisPage.jsx');
    for (const failKey of ['analysis:map.failedToLoad', 'analysis:data.loadFailed',
      'analysis:data.qualityFilterFailed', 'analysis:deformation.failed', 'analysis:deformation.gmmFailed',
      'analysis:rx.failed']) {
      const i = src.indexOf(`t('${failKey}')`);
      expect(i, failKey).toBeGreaterThan(-1);
      expect(src.slice(i - 160, i), failKey).toContain('codedError(err, t)');
    }
    expect(src).toContain("codedProblemText(err, t, 'analysis')");
  });
});

describe('load progress window', () => {
  const failed = (code) => ({
    stage: 'error', stage_idx: 0, stage_total: 4, elapsed_seconds: 0.1,
    message: 'prose message', error: 'prose error', code,
  });

  afterEach(async () => { await i18n.changeLanguage('en'); });

  for (const lang of ['en', 'de']) {
    for (const code of CODES) {
      it(`${lang}: ${code} is worded from common.errors`, async () => {
        await i18n.changeLanguage(lang);
        render(<LoadProgressModal isOpen progressState={failed(code)} onClose={() => {}} />);
        const text = screen.getByTestId('progress-error').textContent;
        expect(text).toBe(locale(lang, 'common').errors[code]);
        expect(text).not.toMatch(/prose/);
      });
    }
  }
});
