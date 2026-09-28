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
const hasCrlf = (p) => fs.readFileSync(p).includes('\r\n');

describe('line endings under frontend/src', () => {
  it('there are files to check', () => {
    // A walk that found nothing would pass forever.
    expect(files.length).toBeGreaterThan(400);
  });

  it('nothing outside the known list uses CRLF', () => {
    const offenders = files.map(relOf)
      .filter((r) => !KNOWN.has(r))
      .filter((r) => hasCrlf(path.join(SRC, r)));
    expect(offenders).toEqual([]);
  });

  it('the known list may only shrink -- every entry still exists and is still CRLF', () => {
    // An entry that has been cleaned up, or deleted, should leave the list.
    // Stale exceptions are how an exception list becomes a blanket.
    const stale = [...KNOWN].filter((r) => {
      const p = path.join(SRC, r);
      return !fs.existsSync(p) || !hasCrlf(p);
    });
    expect(stale).toEqual([]);
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
    const offenders = docs
      .filter((f) => hasCrlf(path.join(DOCS, f)));
    expect(offenders).toEqual([]);
  });
});
