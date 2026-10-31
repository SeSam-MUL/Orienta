// @vitest-environment node
/**
 * The discovery produces `linked_cif` (and `file_type`); the Indexing page's
 * phase-map hand-off joins on them. They must agree on the names and on the
 * rule behind them, and nothing else checks that they do: a record field the
 * backend renames reads as `undefined` in the page, every phase then looks
 * "missing", and no test of either side alone can see it. (The repo pattern:
 * `DatabaseBrowser/syncResultContract.test.js` reads both sources.)
 *
 * Source-level on purpose -- "two files disagree" is invisible to a render test
 * of either one.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..', '..', '..', '..');
const handoff = readFileSync(join(HERE, 'phaseMapHandoff.js'), 'utf8');
const controller = readFileSync(join(REPO, 'indexing_controller.py'), 'utf8');
const route = readFileSync(join(REPO, 'backend', 'api', 'routes', 'indexing.py'), 'utf8');

/** The text between two markers (the markers must exist). */
function between(text, start, end) {
  const i = text.indexOf(start);
  expect(i, `${start} not found`).toBeGreaterThan(-1);
  const j = text.indexOf(end, i);
  expect(j, `${end} not found after ${start}`).toBeGreaterThan(i);
  return text.slice(i, j);
}

const enrich = between(controller, 'def enrich_phase_entries', 'def discover_files_for_method');
const discover = between(controller, 'def discover_files_for_method', 'def find_matching_dicts');

/** Record fields the backend's enrichment sets. */
function producedFields() {
  return new Set([...enrich.matchAll(/entry\["(\w+)"\]\s*=/g)].map((m) => m[1]));
}

describe('discovery records and the phase-map hand-off agree', () => {
  it('the join field the hand-off reads is one the discovery produces', () => {
    const read = new Set([...handoff.matchAll(/\bf\.(linked_\w+)/g)].map((m) => m[1]));
    expect([...read]).toEqual(['linked_cif']);
    expect(producedFields()).toContain('linked_cif');
  });

  it('the file kinds the Dictionary selection tells apart are the ones the discovery writes', () => {
    for (const kind of ['master', 'dictionary']) {
      expect(handoff).toContain(`'${kind}'`);
      expect(discover).toContain(`entry["file_type"] = "${kind}"`);
    }
  });

  it('the discovery and the routing join go through the same identity rule', () => {
    // enrich: linked_cif comes from strict_linked_cif, not the display rule
    const linkedBlock = enrich.slice(enrich.indexOf('from phase_metadata import strict_linked_cif'));
    expect(linkedBlock).toContain('strict_linked_cif(Path(entry["path"])');
    expect(linkedBlock).toContain('entry["linked_cif"]');
    expect(enrich).not.toMatch(/\bfind_linked_cif\(/);
    // routing: the join of the request's files
    const join_ = between(route, 'def phase_map_file_links', 'def first_file_per_cif');
    expect(join_).toContain('strict_linked_cif(');
    expect(join_).not.toContain('find_linked_cif(');
  });
});
