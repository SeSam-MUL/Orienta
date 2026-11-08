/**
 * What the remove and clear buttons of the Loaded Files panel promise.
 *
 * Removing a file from the list discards its crops and its derived datasets (a
 * background-removed or CLAHE dataset is minutes of compute and is not on disk), and keeps a PC
 * the user set (it holds no signal and cannot be read from the file again). A button that does
 * not say so invites the loss; this test holds the three tooltips to it in every language, and
 * holds the panel to the keys.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const LOCALES = path.resolve(HERE, '../../locales');
const KEYS = ['removeTooltip', 'clearListTooltip', 'clearAllTooltip'];

// Hand-written: the word each language uses for "derived datasets", and the PC.
const DERIVED = { en: /derived datasets/i, de: /abgeleitete Datensätze/, ja: /派生データセット/, zh: /派生数据集/ };
const KEPT = {
  en: /PC you set is kept/,
  de: /gesetztes Pattern Center \(PC\) bleibt erhalten/,
  ja: /設定した PC は保持されます/,
  zh: /已设置的 PC 会保留/,
};

describe('loaded files: remove / clear tooltips', () => {
  for (const lang of ['en', 'de', 'ja', 'zh']) {
    const loaded = JSON.parse(
      fs.readFileSync(path.join(LOCALES, lang, 'ebsdviewer.json'), 'utf8'),
    ).loadedFiles;
    for (const key of KEYS) {
      it(`${lang}: ${key} says derived datasets are discarded and a set PC is kept`, () => {
        expect(loaded[key]).toMatch(DERIVED[lang]);
        expect(loaded[key]).toMatch(KEPT[lang]);
      });
    }
  }

  it('the viewer panel shows exactly these keys as the titles of its three buttons', () => {
    const src = fs.readFileSync(path.join(HERE, 'EBSDViewer.jsx'), 'utf8');
    for (const key of KEYS) {
      expect(src).toContain(`title={t('loadedFiles.${key}')}`);
    }
  });
});
