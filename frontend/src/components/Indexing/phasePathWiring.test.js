// @vitest-environment node
//
// Source-level guard for the wiring of "add a phase by path", for the same
// reason and with the same limits as `phaseListWiring.test.js`: IndexingPage
// cannot be mounted in a test, and these two connections are where this
// feature can silently stop working while every unit test stays green.
//
//  * Without `onAddPath` on the page's <PhaseDropdown>, the picker shows the old
//    "+ Add file manually…" link, which needs Electron: in a browser
//    (start_app.py) the user has no way to add a phase of their own.
//  * Without merging `userAddedRef` into the freshly fetched list, a method
//    switch or a retry replaces the file list and the phase the user just
//    added vanishes from the picker.
//
// What the two do is tested against the real functions and component in
// `phasePath.test.js` and `phaseDropdownPath.test.jsx`. DELETE THIS FILE the
// day IndexingPage can be mounted in a test.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const SRC = readFileSync(
  fileURLToPath(new URL('./IndexingPage.jsx', import.meta.url)), 'utf8');

describe('add-by-path is wired into the Indexing page', () => {
  it('the phase picker receives onAddPath', () => {
    const i = SRC.indexOf('<PhaseDropdown');
    expect(i).toBeGreaterThan(-1);
    expect(SRC.slice(i, SRC.indexOf('/>', i))).toMatch(/onAddPath=\{handleAddPhasePath\}/);
  });

  it('a fresh listing is merged with the files the user added', () => {
    const i = SRC.indexOf('onLoaded: (files, groups) =>');
    expect(i).toBeGreaterThan(-1);
    const body = SRC.slice(i, i + 400);
    expect(body).toMatch(/mergeUserAdded\(/);
    expect(body).toMatch(/userAddedRef\.current/);
  });

  it('an own file that is a library file collapses into the entry and the selection moves with it', () => {
    const i = SRC.indexOf('onLoaded: (files, groups) =>');
    const body = SRC.slice(i, i + 1400);
    expect(body).toMatch(/collapseUserAdded\(/);
    expect(body).toMatch(/remapCollapsed\(/);
    expect(body).toMatch(/setPhaseFiles\(moved\.phaseFiles\)/);
  });
});
