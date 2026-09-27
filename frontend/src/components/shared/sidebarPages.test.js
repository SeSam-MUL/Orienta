/**
 * One list of shortcuts, and everything that shows them reads it.
 *
 * On 2026-09-25 there were two lists. The handler used SIDEBAR_PAGES inside
 * App.jsx; the Dashboard kept a hand-written copy. They had drifted until
 * SEVEN of the ten digits named a different page than the one they open —
 * Ctrl+3 said "Analysis" and opened EDS, Ctrl+9 said "ML predictor" and
 * opened Phase Maps, Ctrl+0 said "PC refinement" and opened Analysis — and
 * the dashboard advertised a Ctrl+E that no handler has ever had.
 *
 * Each list was internally consistent, which is why nobody noticed. The only
 * thing that can notice is a test that holds one against the other.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { SIDEBAR_PAGES, pageShortcuts } from './sidebarPages';

const SRC = path.resolve(import.meta.dirname, '..', '..');
const read = (rel) => fs.readFileSync(path.join(SRC, rel), 'utf8');

describe('the list itself', () => {
  it('gives every digit to exactly one page', () => {
    const digits = SIDEBAR_PAGES.filter((p) => p.shortcut).map((p) => p.shortcut);
    expect(new Set(digits).size).toBe(digits.length);
  });

  it('covers 1-9 and 0, and nothing else', () => {
    expect(pageShortcuts().map((p) => p.shortcut).sort())
      .toEqual(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
  });

  it('never hands a shortcut to a section heading', () => {
    for (const p of SIDEBAR_PAGES) {
      if (p.sectionKey) expect(p.shortcut).toBeUndefined();
    }
  });
});

describe('everything that displays a shortcut reads that list', () => {
  it('the dashboard does not keep its own copy', () => {
    const dash = read('components/Dashboard/Dashboard.jsx');
    expect(dash).toMatch(/pageShortcuts\(\)/);
    // The shape of the old bug: a digit written next to a hand-picked label.
    const handWritten = [...dash.matchAll(/"kbd">Ctrl\+([0-9])</g)].map((m) => m[1]);
    expect(handWritten, 'a digit is hard-coded in the dashboard again').toEqual([]);
  });

  it('the handler reads it too', () => {
    const app = read('App.jsx');
    expect(app).toMatch(/SIDEBAR_PAGES\.find\(p => p\.shortcut === e\.key\)/);
    // And does not define a second one.
    expect(app).not.toMatch(/const SIDEBAR_PAGES = \[/);
  });

  it('advertises no shortcut the handler cannot serve', () => {
    // Ctrl+E was on the dashboard for months with nothing behind it. The
    // non-page keys the handler really has are H and B, checked here so this
    // list cannot grow silently either.
    const app = read('App.jsx');
    const dash = read('components/Dashboard/Dashboard.jsx');
    const letters = [...dash.matchAll(/"kbd">Ctrl\+([A-Z])</g)].map((m) => m[1]);
    for (const letter of letters) {
      expect(
        app.includes(`=== '${letter.toLowerCase()}'`),
        `the dashboard shows Ctrl+${letter}, but App.jsx has no handler for it`,
      ).toBe(true);
    }
    expect(letters.length).toBeGreaterThan(0);   // the scan must find something
  });
});
