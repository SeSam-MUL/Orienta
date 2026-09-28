/**
 * This tree is LF. A file that flips to CRLF loses its history.
 *
 * WHAT IT COSTS, measured on this branch: 37 files under `frontend/src`
 * were written back with Windows line endings, among them
 * `IndexingPage.jsx` (8010 diff lines for 16 real ones),
 * `DatabasePage.jsx` (3324 for 12) and `App.jsx` (1601 for 32). The diff
 * becomes unreadable, `git blame` stops working for those files, and the
 * public port would carry every one of them as a full rewrite.
 *
 * THE CAUSE, found by testing rather than guessing: Python's
 * `pathlib.Path.write_text` translates `\n` to `\r\n` on Windows unless it
 * is given `newline=''`. Every patch script that read a file, edited the
 * string and wrote it back flipped that file. The editor tooling does not
 * -- files created and only ever edited that way came out pure LF, which
 * is how the cause was narrowed down to one call.
 *
 * THE EXCEPTIONS ARE NAMED AND MAY ONLY SHRINK. Seven files were already
 * CRLF before this work began -- five wholly, two mixed -- and they belong
 * to another area. Normalising them would put somebody else's whole file
 * into this branch's diff, which is the very thing this test exists to
 * prevent. They are listed so nobody "fixes" them by accident, and so the
 * list is visible when somebody finally does.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SRC = path.dirname(fileURLToPath(import.meta.url));
// The repository root, two levels up from `frontend/src`.
const REPO = path.resolve(SRC, '..', '..');
const DOCS = path.join(REPO, 'docs', 'user-guide');
const EXTS = new Set(['.js', '.jsx', '.json', '.css', '.html']);

/** Already CRLF before this branch. Only ever remove entries from this. */
const KNOWN = new Set([
  'components/EBSDViewer/edsOnlyNavigation.test.jsx',
  'components/EDS/SwipeCompareController.jsx',
  'components/EDS/hooks/useDefaultLayers.js',
  'components/EDS/hooks/useEdsLayerStack.js',
  'components/PhaseMap/annotations/AnnotationToolbar.jsx',
  // mixed endings, not wholly CRLF
  'components/EDS/hooks/useEdsLayerStack.test.js',
  'components/EDS/hooks/useSuggestPhases.js',
]);

function walk(dir, out = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p, out);
    else if (EXTS.has(path.extname(e.name))) out.push(p);
  }
  return out;
}

const files = walk(SRC);
const relOf = (p) => path.relative(SRC, p).replace(/\\/g, '/');

/**
 * THE INDEX, NOT THE WORKING TREE.
 *
 * This guard used to read the checked-out bytes. That is only the same question
 * where `core.autocrlf` is off — and off is NOT git's default on Windows. In the
 * public repository, which has `core.autocrlf=true`, the working tree is CRLF by
 * design and this test reported **710 offenders** (`App.css` and 709 more) on a
 * tree whose content is pure LF. It was a guard bound to the machine it was
 * written on: green here, red for anyone who clones with git's own defaults.
 *
 * What cannot be argued with is the INDEX: `git ls-files --eol` reports it as
 * `i/lf` or `i/crlf` whatever the checkout does. That is the byte sequence a
 * diff, `git blame` and the port all see, which is what the paragraph above is
 * actually about.
 */
const indexEol = (() => {
  const out = execFileSync('git', ['ls-files', '--eol', '-z', '--', SRC],
                           { cwd: REPO, encoding: 'utf8', maxBuffer: 64 << 20 });
  const map = new Map();
  for (const entry of out.split('\0')) {
    if (!entry) continue;
    // `i/lf    w/crlf  attr/                 \tpath`
    const tab = entry.indexOf('\t');
    if (tab < 0) continue;
    const eol = /(^|\s)i\/(\S+)/.exec(entry.slice(0, tab));
    if (!eol) continue;
    map.set(path.relative(SRC, path.join(REPO, entry.slice(tab + 1))).replace(/\\/g, '/'),
            eol[2]);
  }
  return map;
})();

/**
 * A file git has no opinion about is not judged: `i/none` means "no line
 * endings at all" (an empty file) and `i/-text` means git treats it as binary.
 * Anything not in the map is untracked, and this guard is about what the
 * repository carries, not about a scratch file someone left lying around.
 */
export const crlfish = (eol) => eol === 'crlf' || eol === 'mixed';
const hasCrlf = (rel) => crlfish(indexEol.get(rel));

describe('line endings under frontend/src', () => {
  it('there are files to check', () => {
    // A walk that found nothing would pass forever.
    expect(files.length).toBeGreaterThan(400);
    // ... and so would an index view that came back empty, which is what happens
    // if `git` is missing or this is not a checkout. Then the guard cannot judge,
    // and "cannot judge" must not read as "clean".
    expect(indexEol.size, 'git ls-files reported nothing — the guard is blind')
      .toBeGreaterThan(400);
  });

  it('the verdict can say yes as well as no', () => {
    /**
     * The positive control, and it has to live here rather than on real files.
     * It used to be implicit: the old third test asserted the KNOWN entries were
     * still CRLF, so a detector that always said "no" failed it. That only worked
     * in a repository that HAS a CRLF file in its index, and the public one has
     * none — so on the side where this guard matters most, nothing proved it could
     * still see. A blind `() => false` passes every other test in this file.
     */
    expect(crlfish('crlf')).toBe(true);
    expect(crlfish('mixed')).toBe(true);
    expect(crlfish('lf')).toBe(false);
    expect(crlfish('none')).toBe(false);
    expect(crlfish(undefined)).toBe(false);
    // ... and the parser really produced git's vocabulary, not empty strings.
    const seen = new Set(indexEol.values());
    expect([...seen].every((v) => ['lf', 'crlf', 'mixed', 'none', '-text'].includes(v)),
      `unexpected eol values from git: ${[...seen]}`).toBe(true);
    expect(seen.has('lf'), 'no file reported as lf — the parse is wrong').toBe(true);
  });

  it('nothing outside the known list uses CRLF', () => {
    const offenders = files.map(relOf)
      .filter((r) => !KNOWN.has(r))
      .filter((r) => hasCrlf(r));
    expect(offenders).toEqual([]);
  });

  it('every entry of the known list still names a file, and the list has not grown', () => {
    /**
     * This used to also require each entry to be STILL CRLF, and that made the
     * exception list a fact about one repository. Measured: all seven are
     * `i/crlf` or `i/mixed` in the development tree and all seven are `i/lf` in
     * the public one, because `core.autocrlf=true` normalised them on the way in.
     * The public repository is CLEANER here, and the old assertion called that
     * staleness and went red — a guard that fails because the problem went away.
     *
     * An entry that has become LF is the outcome the list exists to make
     * possible, so it is not a failure. What is still checked is what actually
     * protects the list: a path that no longer exists (the file was deleted or
     * moved, and the exception is now a blanket over nothing), and growth —
     * because adding an entry has to be a visible, reviewed change rather than
     * something a patch script does on the way past.
     */
    const missing = [...KNOWN].filter((r) => !fs.existsSync(path.join(SRC, r)));
    expect(missing, 'exceptions that no longer name a file').toEqual([]);
    expect(KNOWN.size,
      'the exception list grew — a new CRLF file needs a reason, not an entry')
      .toBeLessThanOrEqual(7);
  });
});

describe('line endings under docs/user-guide', () => {
  /**
   * Added because the defect MOVED. The guard above was written for
   * `frontend/src`, the cause -- a patch script writing a file back --
   * was not, and the next thing it flipped was the user manual, which is
   * not under `frontend/src`. A guard drawn around the place a bug
   * happened to appear teaches the bug to appear elsewhere.
   *
   * No exceptions here: every file in this directory is LF, and the two
   * that were not were both written this week by that same script.
   */
  const docs = fs.existsSync(DOCS)
    ? fs.readdirSync(DOCS).filter((f) => f.endsWith('.md'))
    : [];

  it('there are guides to check', () => {
    expect(docs.length).toBeGreaterThan(10);
  });

  it('none of them uses CRLF', () => {
    // Its own `ls-files` call: DOCS is outside `frontend/src`, so the map above
    // does not cover it, and reusing that map would have quietly checked nothing.
    const out = execFileSync('git', ['ls-files', '--eol', '-z', '--', DOCS],
                             { cwd: REPO, encoding: 'utf8', maxBuffer: 16 << 20 });
    const rows = out.split('\0').filter(Boolean);
    expect(rows.length, 'git listed no guides — the check would be vacuous')
      .toBeGreaterThan(10);
    const offenders = rows
      .filter((row) => /(^|\s)i\/(crlf|mixed)/.test(row.slice(0, row.indexOf('\t'))))
      .map((row) => row.slice(row.indexOf('\t') + 1));
    expect(offenders).toEqual([]);
  });
});
