// @vitest-environment node
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

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
    const app = fs.readFileSync(path.join(SRC, 'App.jsx'), 'utf8');
    const ids = new Set([...app.matchAll(/\{\s*id:\s*'([a-z0-9_]+)'/g)].map((m) => m[1]));
    // Handled outside the page router: opens the HDF5 viewer overlay.
    ids.add('h5viewer');
    expect(ids.size).toBeGreaterThan(5);

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
