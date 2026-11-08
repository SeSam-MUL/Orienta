/**
 * Save dialogs open where the user last saved — tested by running the module.
 *
 * The M5 tester's report: every save dialog started in Downloads. The fix is
 * a remembered folder; these tests pin what is remembered, when it is
 * ignored, and how a caller's own path wins.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const MODULE = path.resolve(import.meta.dirname, '..', '..', 'electron', 'save_dialogs.js');
const sd = require(MODULE);

let home;
beforeEach(() => { home = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-savedir-')); });
afterEach(() => { fs.rmSync(home, { recursive: true, force: true }); });

describe('rememberDir / readLastDir', () => {
  it('remembers the folder of a saved file and reads it back', () => {
    const dir = path.join(home, 'report');
    fs.mkdirSync(dir);
    expect(sd.rememberDir(home, path.join(dir, 'figure-1.png'))).toBe(dir);
    expect(sd.readLastDir(home)).toBe(dir);
    expect(fs.existsSync(sd.lastDirFile(home))).toBe(true);
  });

  it('remembers a chosen folder itself when told it is one', () => {
    const dir = path.join(home, 'series');
    fs.mkdirSync(dir);
    expect(sd.rememberDir(home, dir, { isDirectory: true })).toBe(dir);
    expect(sd.readLastDir(home)).toBe(dir);
  });

  it('forgets a folder that no longer exists', () => {
    const dir = path.join(home, 'gone');
    fs.mkdirSync(dir);
    sd.rememberDir(home, path.join(dir, 'x.png'));
    fs.rmSync(dir, { recursive: true });
    expect(sd.readLastDir(home)).toBeNull();
  });

  it('does not probe a network share (a dead one would block the main process)', () => {
    fs.mkdirSync(path.join(home, 'electron'), { recursive: true });
    const unc = '\\\\server\\share\\figures';        // \\server\share\figures
    fs.writeFileSync(sd.lastDirFile(home), `${unc}\n`);
    expect(sd.readLastDir(home)).toBe(unc);
  });

  it('is null with no memory, and never throws', () => {
    expect(sd.readLastDir(home)).toBeNull();
    expect(sd.readLastDir(null)).toBeNull();
    expect(sd.rememberDir(null, 'x.png')).toBeNull();
    expect(sd.rememberDir(home, '')).toBeNull();
  });
});

describe('resolveDefaultPath', () => {
  const lastDir = path.join('C:', 'work', 'report');
  const fallbackDir = path.join('C:', 'Users', 'me', 'Downloads');

  it('places a bare file name in the remembered folder', () => {
    expect(sd.resolveDefaultPath({ requested: 'image.png', lastDir, fallbackDir }))
      .toBe(path.join(lastDir, 'image.png'));
  });

  it('falls back to Downloads when nothing is remembered', () => {
    expect(sd.resolveDefaultPath({ requested: 'image.png', lastDir: null, fallbackDir }))
      .toBe(path.join(fallbackDir, 'image.png'));
    expect(sd.resolveDefaultPath({ requested: 'image.png' })).toBe('image.png');
  });

  it('lets an absolute path from the caller win', () => {
    const abs = path.resolve(path.join('D:', 'scan', 'phase-map.png'));
    expect(sd.resolveDefaultPath({ requested: abs, lastDir, fallbackDir })).toBe(abs);
  });

  it('opens on the folder alone when no name is given', () => {
    expect(sd.resolveDefaultPath({ requested: null, lastDir, fallbackDir })).toBe(lastDir);
    expect(sd.resolveDefaultPath({ requested: '', lastDir: null, fallbackDir })).toBe(fallbackDir);
    expect(sd.resolveDefaultPath({})).toBeUndefined();
  });

  it('leaves a relative path with directories alone', () => {
    const rel = path.join('sub', 'image.png');
    expect(sd.resolveDefaultPath({ requested: rel, lastDir, fallbackDir })).toBe(rel);
  });
});

describe('open dialogs keep their own folder', () => {
  it('remembers the folder of an opened file apart from the save folder', () => {
    const scans = path.join(home, 'scans');
    const report = path.join(home, 'report');
    fs.mkdirSync(scans);
    fs.mkdirSync(report);
    sd.rememberDir(home, path.join(report, 'figure.png'));
    expect(sd.rememberDir(home, path.join(scans, 'map.h5oina'), { kind: 'open' })).toBe(scans);
    expect(sd.readLastDir(home, 'open')).toBe(scans);
    expect(sd.readLastDir(home)).toBe(report);
    expect(sd.lastDirFile(home, 'open')).not.toBe(sd.lastDirFile(home));
  });

  it('remembers a chosen folder of scans itself', () => {
    const batch = path.join(home, 'batch');
    fs.mkdirSync(batch);
    sd.rememberDir(home, batch, { isDirectory: true, kind: 'open' });
    expect(sd.readLastDir(home, 'open')).toBe(batch);
    expect(sd.readLastDir(home)).toBeNull();
  });
});

describe('resolveOpenPath', () => {
  const lastDir = path.join('C:', 'data', 'scans');

  it('starts in the folder of the last opened file', () => {
    expect(sd.resolveOpenPath({ lastDir })).toBe(lastDir);
  });

  it('lets an absolute path from the caller win', () => {
    const abs = path.resolve(path.join('D:', 'other', 'map.h5oina'));
    expect(sd.resolveOpenPath({ requested: abs, lastDir })).toBe(abs);
  });

  it('ignores a relative or empty request, and is undefined with no memory', () => {
    expect(sd.resolveOpenPath({ requested: 'map.h5oina', lastDir })).toBe(lastDir);
    expect(sd.resolveOpenPath({ requested: '  ', lastDir })).toBe(lastDir);
    expect(sd.resolveOpenPath({})).toBeUndefined();
  });
});
