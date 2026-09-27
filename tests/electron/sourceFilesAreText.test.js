/**
 * No source file may contain a raw NUL byte.
 *
 * `EdsPreflightPanel.jsx` held one inside a string literal
 * (`const I18N_MISS = '<NUL>miss'`). git classifies a file with a NUL as binary,
 * so that component had **no readable diff at all** — `--numstat` reported `-` `-`
 * instead of line counts, and every review of it since was reading nothing. The
 * value it needed is the same written as the escape `\u0000`.
 *
 * It guards every source root rather than that one file, because the next person
 * to need a sentinel value will reach for the same trick — and the trick is not
 * JavaScript-specific, so Python is scanned too. (An earlier version read only
 * frontend/src and electron/ while claiming "the whole tree"; ROOTS below is
 * what is actually checked.)
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const REPO = path.resolve(import.meta.dirname, '..', '..');
const ROOTS = [
  'frontend/src', 'electron',
  'backend', 'simulation', 'analysis', 'tools', 'scripts', 'tests',
];
const EXTENSIONS = new Set([
  '.js', '.jsx', '.ts', '.tsx', '.json', '.css', '.html', '.py',
]);
const SKIP_DIRS = new Set([
  'node_modules', 'dist', 'build', '.vite', 'coverage',
  '__pycache__', '.pytest_cache', 'data', 'fixtures',
]);

function sourceFiles(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue;
      out.push(...sourceFiles(path.join(dir, entry.name)));
    } else if (EXTENSIONS.has(path.extname(entry.name))) {
      out.push(path.join(dir, entry.name));
    }
  }
  return out;
}

const FILES = ROOTS.flatMap((rel) => {
  const dir = path.join(REPO, rel);
  return fs.existsSync(dir) ? sourceFiles(dir) : [];
});

describe('source files are text', () => {
  it('finds the files it claims to check', () => {
    // A silent zero would make the guard below vacuous.
    expect(FILES.length).toBeGreaterThan(200);
  });

  it('really does reach every root, not just the JS ones', () => {
    // Without this, adding a root that does not exist (a rename, a worktree
    // without it) silently shrinks the scan while the count above still passes.
    for (const rel of ROOTS) {
      const dir = path.join(REPO, rel);
      if (!fs.existsSync(dir)) continue;
      const prefix = path.join(REPO, rel) + path.sep;
      expect(FILES.some((f) => f.startsWith(prefix)), `no file scanned under ${rel}`)
        .toBe(true);
    }
    // and the Python side is genuinely in scope
    expect(FILES.some((f) => f.endsWith('.py'))).toBe(true);
  });

  it('none contains a raw NUL byte', () => {
    const offenders = [];
    for (const file of FILES) {
      const buffer = fs.readFileSync(file);
      const at = buffer.indexOf(0);
      if (at !== -1) {
        const line = buffer.subarray(0, at).toString('utf8').split('\n').length;
        offenders.push(`${path.relative(REPO, file)}:${line}`);
      }
    }
    expect(offenders, 'git treats these as binary, so they have no readable diff')
      .toEqual([]);
  });

  it('the file this came from still carries the value, as an escape', () => {
    const source = fs.readFileSync(
      path.join(REPO, 'frontend/src/components/Indexing/EdsPreflightPanel.jsx'), 'utf8',
    );
    expect(source).toContain('\\u0000miss');
    expect(source).not.toContain('\u0000');
    // and the escape really is the same string the raw byte was
    // eslint-disable-next-line no-eval
    expect(eval("'\\u0000miss'")).toBe(String.fromCharCode(0) + 'miss');
  });
});
