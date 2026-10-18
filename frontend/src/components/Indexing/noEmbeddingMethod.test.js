// The "Embedding" indexing method was removed from the UI: the backend has
// answered 400 for it ever since the encoder / FAISS path was never wired up.
// IndexingPage is not rendered by any test, so this reads the sources: the
// method list, the parameter group and the translations must stay gone.
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const SRC = path.resolve(import.meta.dirname, '..', '..');
const read = (rel) => fs.readFileSync(path.join(SRC, rel), 'utf8');
const PAGE = read('components/Indexing/IndexingPage.jsx');

describe('no Embedding indexing method in the UI', () => {
  it('the method list is exactly Hough / Dictionary / Spherical', () => {
    const start = PAGE.indexOf('const methodOptions = useMemo(');
    expect(start, 'methodOptions moved or was renamed').toBeGreaterThan(-1);
    const end = PAGE.indexOf('], [t]);', start);
    const values = [...PAGE.slice(start, end).matchAll(/value: '([a-z]+)'/g)].map((m) => m[1]);
    expect(values).toEqual(['hough', 'dictionary', 'spherical']);
  });

  it('the page has no embedding branch, parameter group or state', () => {
    expect(PAGE).not.toMatch(/embedding/i);
    expect(PAGE).not.toMatch(/faiss/i);
  });

  for (const lng of ['en', 'de', 'ja', 'zh']) {
    it(`${lng}/indexing.json has no embedding keys`, () => {
      const doc = JSON.parse(read(`locales/${lng}/indexing.json`));
      expect(doc.embedding).toBeUndefined();
      expect(doc.methods.embedding).toBeUndefined();
      expect(doc.methods.embeddingTip).toBeUndefined();
      expect(doc.method.tooltip).not.toMatch(/embedding/i);
    });
  }
});
