/**
 * `toast` is an OBJECT — nobody may call it as a function.
 *
 * The store exports `{ success, error, info, warning }`. A bare `toast(...)`
 * throws "toast is not a function" — which is what the user saw after a
 * successful phase check (2026-09-07): the check returned 200, the count
 * reached the button, and then the informational toast threw inside the
 * try-block, so the catch showed the TypeError as an error toast and the
 * Phase Check layer was never added. Two older `toast(...)` calls had been
 * sitting on the reassign path with the same defect, waiting for a
 * pseudo-symmetric receiving phase to trigger them.
 *
 * Both halves were "valid" on their own; only the call site against the store
 * was wrong. So this test holds the two together: every `toast.<x>(` used in
 * the components must exist on the store, and no component may call `toast(`
 * bare.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { toast } from './useToastStore';

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(here, '..');

function walk(dir, out = []) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(p, out);
    else if (/\.(jsx?|tsx?)$/.test(ent.name) && !/\.test\./.test(ent.name)) out.push(p);
  }
  return out;
}

const importsToast = (src) =>
  /import\s*\{[^}]*\btoast\b[^}]*\}\s*from\s*['"][^'"]*useToastStore['"]/.test(src);

describe('toast call contract', () => {
  const files = walk(SRC).filter((f) => !f.endsWith('useToastStore.js'));
  const users = files.filter((f) => importsToast(fs.readFileSync(f, 'utf8')));

  it('finds the components that use the store (sanity)', () => {
    expect(users.length).toBeGreaterThan(0);
  });

  it('no component calls toast(...) as a function', () => {
    const offenders = [];
    for (const f of users) {
      const src = fs.readFileSync(f, 'utf8');
      const lines = src.split('\n');
      lines.forEach((line, i) => {
        if (/(?<![.\w])toast\(/.test(line)) offenders.push(`${path.relative(SRC, f)}:${i + 1}`);
      });
    }
    expect(offenders, 'bare toast(...) calls').toEqual([]);
  });

  it('every toast.<method>( used in a component exists on the store', () => {
    const missing = new Set();
    for (const f of users) {
      const src = fs.readFileSync(f, 'utf8');
      for (const m of src.matchAll(/\btoast\.(\w+)\(/g)) {
        if (typeof toast[m[1]] !== 'function') missing.add(`${path.relative(SRC, f)} -> toast.${m[1]}`);
      }
    }
    expect([...missing]).toEqual([]);
  });
});
