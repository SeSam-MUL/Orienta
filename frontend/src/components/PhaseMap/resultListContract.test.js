// @vitest-environment node
/**
 * The Phase Map page reads `GET /api/indexing/results` and decides from it
 * which result is on screen and which refined result belongs to it. The
 * backend builds that list in `list_results`. If the two disagree on a field
 * name the page just sees `undefined` -- no error anywhere, only an Original/
 * Refined switch that never appears -- and a test that feeds the page a
 * hand-written list cannot see it. So this test reads both sources.
 *
 * Source-level on purpose, like `DatabaseBrowser/syncResultContract.test.js`:
 * the defect is "two files disagree".
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const page = readFileSync(join(HERE, 'PhaseMapPage.jsx'), 'utf8');
const backend = readFileSync(
  join(HERE, '..', '..', '..', '..', 'backend', 'api', 'routes', 'indexing.py'), 'utf8');

// The fields the page's result-pairing logic reads off a listed entry.
const FIELDS_THE_PAGE_USES = ['id', 'is_active', 'is_refined', 'refinement_of'];

/** Keys of the entry dicts `list_results` builds (its own function body only). */
function backendKeys() {
  const start = backend.indexOf('@router.get("/results")');
  expect(start, 'indexing.py must have the GET /results route').toBeGreaterThan(-1);
  const rest = backend.slice(start + 10);
  const next = rest.search(/\n@router\./);
  const body = next > -1 ? rest.slice(0, next) : rest;
  return new Set([...body.matchAll(/"(\w+)":/g)].map((m) => m[1]));
}

/** Entry fields read in the page by the `r` / `rs` callbacks over the list. */
function pageUses(field) {
  return new RegExp(`\\br\\.${field}\\b`).test(page);
}

describe('/api/indexing/results: the page and list_results agree on the entry fields', () => {
  it('the backend entry has every field the page pairs results by', () => {
    const keys = backendKeys();
    const missing = FIELDS_THE_PAGE_USES.filter((f) => !keys.has(f));
    expect(missing, `list_results lacks ${missing.join(', ')}; it builds {${[...keys].join(', ')}}`)
      .toEqual([]);
  });

  it('the page really reads each of those fields (the list above cannot go stale)', () => {
    const unused = FIELDS_THE_PAGE_USES.filter((f) => !pageUses(f));
    expect(unused, `PhaseMapPage no longer reads r.${unused.join(', r.')}`).toEqual([]);
  });

  it('the page reads no other field of a listed entry for the pairing', () => {
    // Everything the pairing helper (`noteResults`) copies from an entry.
    const m = /setResultList\(list\.map\(\(r\) => \(\{([\s\S]*?)\}\)\)\);/.exec(page);
    expect(m, 'PhaseMapPage must build resultList in noteResults').toBeTruthy();
    const read = new Set([...m[1].matchAll(/\br\.(\w+)/g)].map((x) => x[1]));
    const keys = backendKeys();
    const orphans = [...read].filter((f) => !keys.has(f));
    expect(orphans, `noteResults reads r.${orphans.join(', r.')} which list_results does not send`)
      .toEqual([]);
  });
});
