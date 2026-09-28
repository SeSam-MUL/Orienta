// @vitest-environment node
/**
 * No two files under src/ may differ only in case.
 *
 * Windows and macOS resolve `./ElementFacets` and `./elementFacets` to the
 * same file; Linux and CI do not. Measured here: a component named
 * `ElementFacets.jsx` next to the logic module `elementFacets.js` made
 * `import ElementFacets from './ElementFacets'` hand back the logic module,
 * which has no default export, and seven tests died on "Element type is
 * invalid ... got: undefined". On a Linux runner the same code would have
 * worked, so the pair could have shipped and broken only for the developer
 * who does not have it in their editor's import cache.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SRC = path.join(path.dirname(fileURLToPath(import.meta.url)));

function collisionsIn(dir) {
  const seen = new Map();
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const lower = entry.name.toLowerCase();
    if (seen.has(lower)) out.push(`${dir}: ${seen.get(lower)} vs ${entry.name}`);
    else seen.set(lower, entry.name);
    if (entry.isDirectory()) out.push(...collisionsIn(path.join(dir, entry.name)));
  }
  return out;
}

describe('file names', () => {
  it('never differ only in case', () => {
    expect(collisionsIn(SRC)).toEqual([]);
  });
});
