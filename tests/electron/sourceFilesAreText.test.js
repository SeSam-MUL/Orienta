/**
 * No source file may contain a raw control character: not a NUL, and not any of
 * the others either.
 *
 * WHY IT GREW. On 2026-09-27 the same mistake was found four times in one day --
 * a DOI regex, an mp-key regex, `suggestNoMatchWire.test.jsx`, and the NUL below.
 * Every one of them was an escape whose BACKSLASH was eaten in transit (a shell
 * heredoc, a tool boundary), leaving the byte the escape stands for. The worst
 * case was `/\b(naechste|...)\b/i`, which became `/<BS>(naechste|...)<BS>/i`: a
 * regex that matches a literal backspace, so the guard against transliterated
 * umlauts matched nothing from its first day and passed green while doing it.
 * A raw control byte is invisible in review, and the code holding it keeps
 * running -- it just stops meaning what it says.
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
  // `docs/` because the user guides ship, and prose is where a swallowed escape
  // is hardest to see: nobody runs a document.
  'docs',
];

//: The markdown at the repository root is not under any of those roots, and it is
//: the part users read most: CHANGELOG.md travels INSIDE the runtime package, and
//: README/INSTALL/NOTICE are the public face. Scanned non-recursively.
const ROOT_FILES = fs.existsSync(REPO)
  ? fs.readdirSync(REPO).filter((n) => n.toLowerCase().endsWith('.md'))
    .map((n) => path.join(REPO, n))
  : [];
const EXTENSIONS = new Set([
  '.js', '.jsx', '.ts', '.tsx', '.json', '.css', '.html', '.py',
  // Markdown too: a swallowed escape in a document is just as invisible, and
  // these files are read by users (the changelog and the guides ship).
  '.md',
]);

//: Tab, LF, CR are the only control characters a source file may hold.
const ALLOWED_CONTROL = new Set([0x09, 0x0a, 0x0d]);

//: What each byte most likely used to be. Naming them is the point: the failure
//: then says "you probably meant \b" instead of printing something invisible.
const PROBABLY_MEANT = new Map([
  [0x07, '\\a'], [0x08, '\\b'], [0x0b, '\\v'], [0x0c, '\\f'], [0x1b, '\\e'],
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

const FILES = [
  ...ROOTS.flatMap((rel) => {
    const dir = path.join(REPO, rel);
    return fs.existsSync(dir) ? sourceFiles(dir) : [];
  }),
  ...ROOT_FILES,
];

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
    // ... as is markdown, added after a swallowed escape turned up in prose
    expect(FILES.some((f) => f.endsWith('.md'))).toBe(true);
  });

  it('none contains any other control character', () => {
    const offenders = [];
    for (const file of FILES) {
      const buffer = fs.readFileSync(file);
      for (let i = 0; i < buffer.length; i += 1) {
        const byte = buffer[i];
        // NUL has its own test above, with its own reason (git calls the file
        // binary). Everything else printable is fine; 0x7f is DEL.
        if (byte === 0 || ALLOWED_CONTROL.has(byte)) continue;
        if (byte >= 0x20 && byte !== 0x7f) continue;
        const line = buffer.subarray(0, i).toString('utf8').split('\n').length;
        const hex = `0x${byte.toString(16).padStart(2, '0')}`;
        const hint = PROBABLY_MEANT.get(byte);
        offenders.push(`${path.relative(REPO, file)}:${line} ${hex}`
          + (hint ? ` — probably meant ${hint}` : ''));
        break;  // one report per file is enough to go and look
      }
    }
    expect(offenders, 'a raw control byte is invisible in review, and the code '
      + 'holding it keeps running while no longer meaning what it says')
      .toEqual([]);
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
