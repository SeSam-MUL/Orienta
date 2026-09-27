// @vitest-environment node
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { SIDEBAR_PAGES } from './components/shared/sidebarPages';

// Cross-module navigation goes through a string id. A wrong one used to leave
// the window blank AND the target page unmounted, so whatever it was supposed
// to load — the result gallery, in the case that started this — never loaded.
// App.jsx now refuses unknown ids; this keeps the senders honest so the button
// works rather than merely failing quietly.
const SRC = path.resolve(__dirname);

function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (/\.(jsx?|tsx?)$/.test(e.name)) out.push(p);
  }
  return out;
}

describe('navigate-to targets', () => {
  it('every dispatched page id is a real page', () => {
    // Read the list, do not scan for it. This used to grep App.jsx for
    // `{ id: '...' }`; when the list moved to its own module on 2026-09-25 the
    // grep still matched a few unrelated objects left in App.jsx, so the
    // `size > 5` control below did not fire -- the set simply lost most of the
    // real pages and the test started reporting honest ids as offenders.
    const ids = new Set(SIDEBAR_PAGES.filter((p) => !p.sectionKey).map((p) => p.id));
    // Handled outside the page router: opens the HDF5 viewer overlay.
    ids.add('h5viewer');
    // Name a few, rather than count: a count cannot tell "the list moved"
    // from "the list shrank".
    for (const id of ['dashboard', 'ebsdviewer', 'eds', 'phasemap', 'settings']) {
      expect(ids.has(id), `the page list no longer contains ${id}`).toBe(true);
    }

    const offenders = [];
    for (const file of walk(SRC)) {
      if (file.endsWith('App.jsx')) continue;
      const text = fs.readFileSync(file, 'utf8');
      for (const m of text.matchAll(/navigate-to'[^)]*?page:\s*'([a-z0-9_-]+)'/gs)) {
        if (!ids.has(m[1])) offenders.push(`${path.relative(SRC, file)} → "${m[1]}"`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
