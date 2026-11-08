/**
 * The refusal to index a file without a pattern centre is shown in the reader's
 * language on every page that can start a run: the Indexing page, its batch
 * dialog, and the Batch workflow. The backend sends the code in a header and the
 * English sentence as `detail`; each starter has to translate on the code.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { problemText } from '../../services/problemText';

const root = path.resolve(__dirname, '..', '..');
const read = (p) => fs.readFileSync(path.join(root, p), 'utf-8');
const locale = (lang, ns) => JSON.parse(read(`locales/${lang}/${ns}.json`));

const err = {
  response: {
    headers: { 'x-orienta-code': 'pc_missing' },
    data: { detail: "'scan' carries no pattern centre (PC): English prose" },
  },
  message: 'Request failed with status code 400',
};

const tFor = (lang) => (key, opts = {}) => {
  const [ns, rest] = key.split(':');
  let v = locale(lang, ns);
  for (const part of rest.split('.')) v = v?.[part];
  return v === undefined ? opts.defaultValue : v;
};

describe('pc_missing', () => {
  for (const ns of ['indexing', 'batch']) {
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      it(`${ns}/${lang}: the code is translated, not the English prose`, () => {
        const text = problemText(err, tFor(lang), ns);
        expect(text).toBe(locale(lang, ns).errors.pc_missing);
        expect(text).not.toMatch(/English prose/);
        expect(text.length).toBeGreaterThan(40);
      });
    }
  }

  it('every page that starts a run translates on the code', () => {
    expect(read('components/Indexing/IndexingPage.jsx')).toMatch(/problemText\(err, t, 'indexing'\)/);
    expect(read('components/Indexing/BatchIndexingDialog.jsx')).toMatch(/problemText\(err, t, 'indexing'\)/);
    expect(read('components/Batch/BatchWorkflow.jsx')).toMatch(/problemText\(e, t, 'batch'\)/);
  });
});
