// @vitest-environment node
/**
 * `DatabasePage` produces the sync result; `SyncUploadDialog` displays it.
 * They must agree on the field names, and nothing else checks that they do.
 *
 * They did not. Commit 279cc84c (2026-08-13) renamed the parent's counter from
 * `uploaded` to `transferred` — it had to, because the same commit added the
 * download direction and the count is no longer always an upload — but the
 * dialog kept reading `result.uploaded`. React renders `undefined` as nothing,
 * so from that day every completed sync showed a bare "uploaded" with no number
 * in front of it, in both directions. For fourteen days.
 *
 * Neither side's tests noticed: the dialog's test fed it `{ uploaded: 70 }`,
 * a shape the parent had stopped producing. A green test on a hand-written
 * fixture is exactly as strong as the fixture, which is why this file reads the
 * two sources instead.
 *
 * Source-level assertions on purpose — the same reasoning as
 * `EDS/PhaseMapCanvas.mount.test.jsx`: the defect is "two files disagree", and
 * no render test of either one alone can see it.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const page = readFileSync(join(HERE, 'DatabasePage.jsx'), 'utf8');
const dialog = readFileSync(join(HERE, 'SyncUploadDialog.jsx'), 'utf8');

/** Keys of the `totals` object literal the parent hands down as `result`. */
function producedKeys() {
  const m = /const totals = \{([^}]*)\}/.exec(page);
  expect(m, 'DatabasePage must build a `totals` literal').toBeTruthy();
  return new Set([...m[1].matchAll(/(\w+)\s*:/g)].map((x) => x[1]));
}

/** Fields the dialog reads off `result`. */
function consumedKeys() {
  return new Set([...dialog.matchAll(/\bresult\.(\w+)/g)].map((m) => m[1]));
}

describe('the sync result travels from DatabasePage to SyncUploadDialog', () => {
  it('the dialog reads only fields the page actually produces', () => {
    const produced = producedKeys();
    const consumed = consumedKeys();
    const orphans = [...consumed].filter((k) => !produced.has(k));
    expect(orphans, `SyncUploadDialog reads result.${orphans.join(', result.')} `
      + `but DatabasePage produces {${[...produced].join(', ')}}`).toEqual([]);
  });

  it('the count is named `transferred`, because it is not always an upload', () => {
    // Pinned by name: the download direction shares this counter, so a rename
    // back to `uploaded` would be wrong on half the runs even if both sides
    // agreed on it.
    expect(producedKeys()).toContain('transferred');
    expect(consumedKeys()).toContain('transferred');
    expect(consumedKeys()).not.toContain('uploaded');
  });
});

describe('both transfer directions are wired end to end', () => {
  it('the page can open the dialog in download mode', () => {
    expect(page).toMatch(/setSyncMode\('download'\)/);
    expect(page).toMatch(/mode=\{syncMode\}/);
  });

  it('the dialog honours that mode instead of ignoring the prop', () => {
    // The committed dialog took no `mode` at all, so "Download All" opened a
    // window headed "Sync Upload" that listed the LOCAL files — and then
    // downloaded. The counts it offered were for the other direction.
    expect(dialog).toMatch(/mode = 'upload'/);
    expect(dialog).toMatch(/mode === 'download' \? 'syncDownload' : 'syncUpload'/);
    expect(dialog).toMatch(/downloadable/);
  });

  it('the page supplies the per-direction counts the dialog asks for', () => {
    expect(page).toMatch(/uploadable:/);
    expect(page).toMatch(/downloadable:/);
  });
});
