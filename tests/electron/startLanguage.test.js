/**
 * The setup wizard's language, handed to the app once — tested by running it.
 *
 * Every test works in its own temporary directory; nothing here touches
 * %APPDATA%\Orienta on this machine.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const startLanguage = require('../../electron/start_language.js');

let dir;
beforeEach(() => { dir = fs.mkdtempSync(path.join(os.tmpdir(), 'orienta-lang-')); });
afterEach(() => { fs.rmSync(dir, { recursive: true, force: true }); });

describe('start language', () => {
  it('carries nothing when the wizard recorded nothing', () => {
    expect(startLanguage.queryFor(dir)).toBe('');
  });

  it('carries an automatic language without the explicit flag', () => {
    expect(startLanguage.record(dir, 'de', false)).toBe(true);
    expect(startLanguage.queryFor(dir)).toBe('?lang=de');
  });

  it('carries an explicit choice with the flag', () => {
    startLanguage.record(dir, 'ja', true);
    expect(startLanguage.queryFor(dir)).toBe('?lang=ja&langExplicit=1');
  });

  it('never lets a later automatic language downgrade an explicit choice', () => {
    startLanguage.record(dir, 'zh', true);
    startLanguage.record(dir, 'en', false);
    expect(startLanguage.queryFor(dir)).toBe('?lang=zh&langExplicit=1');
  });

  it('lets a later explicit choice replace an earlier one', () => {
    startLanguage.record(dir, 'zh', true);
    startLanguage.record(dir, 'de', true);
    expect(startLanguage.queryFor(dir)).toBe('?lang=de&langExplicit=1');
  });

  it('refuses a language the app does not speak, and writes nothing', () => {
    expect(startLanguage.record(dir, 'fr', true)).toBe(false);
    expect(fs.existsSync(startLanguage.fileIn(dir))).toBe(false);
    expect(startLanguage.queryFor(dir)).toBe('');
  });

  it('is handed over once: consumed() removes it', () => {
    startLanguage.record(dir, 'de', true);
    startLanguage.consumed(dir);
    expect(startLanguage.queryFor(dir)).toBe('');
    startLanguage.consumed(dir); // a second call is harmless
  });

  it('survives until consumed — a failed load does not lose the choice', () => {
    startLanguage.record(dir, 'de', true);
    startLanguage.queryFor(dir);
    expect(startLanguage.queryFor(dir)).toBe('?lang=de&langExplicit=1');
  });

  it('ignores a damaged file instead of throwing', () => {
    fs.writeFileSync(startLanguage.fileIn(dir), '{not json');
    expect(startLanguage.queryFor(dir)).toBe('');
    fs.writeFileSync(startLanguage.fileIn(dir), JSON.stringify({ lang: '../x' }));
    expect(startLanguage.queryFor(dir)).toBe('');
  });

  it('creates its directory when it is not there yet', () => {
    const nested = path.join(dir, 'Orienta');
    startLanguage.record(nested, 'de', false);
    expect(startLanguage.queryFor(nested)).toBe('?lang=de');
  });
});

describe('where the shell keeps it', () => {
  it('sits beside the data-folder pointer, not inside the data folder', () => {
    const main = fs.readFileSync(path.join(__dirname, '../../electron/main.js'), 'utf8');
    expect(main).toMatch(/function startLanguageDir\(\)\s*\{\s*return path\.dirname\(homePointerFile\(\)\);/);
    expect(main).not.toMatch(/startLanguage\.\w+\(app\.getPath\('userData'\)/);
  });
});

/**
 * The language the APP runs in, remembered for the next start.
 *
 * The splash screen runs before any page, so it used the operating system's
 * language. The M5 tester ran Orienta in German on a Mac set to English and
 * met an English splash at every start. The app now records its language and
 * the shell prefers that.
 */
describe('app language', () => {
  it('is nothing until the app has said something', () => {
    expect(startLanguage.appLanguage(dir)).toBe(null);
  });

  it('is remembered, and read back', () => {
    expect(startLanguage.recordApp(dir, 'de')).toBe(true);
    expect(startLanguage.appLanguage(dir)).toBe('de');
  });

  it('survives the hand-off file being consumed', () => {
    // This is the whole point: start-language.json is deleted once the page
    // has loaded, so the shell cannot use it on the next start.
    startLanguage.record(dir, 'ja', true);
    startLanguage.recordApp(dir, 'ja');
    startLanguage.consumed(dir);
    expect(startLanguage.read(dir)).toBe(null);
    expect(startLanguage.appLanguage(dir)).toBe('ja');
  });

  it('refuses a language the app does not have', () => {
    expect(startLanguage.recordApp(dir, 'fr')).toBe(false);
    expect(startLanguage.recordApp(dir, '')).toBe(false);
    expect(startLanguage.appLanguage(dir)).toBe(null);
  });

  it('follows the latest change', () => {
    startLanguage.recordApp(dir, 'de');
    startLanguage.recordApp(dir, 'zh');
    expect(startLanguage.appLanguage(dir)).toBe('zh');
  });

  it('does not rewrite the file when nothing changed', () => {
    startLanguage.recordApp(dir, 'de');
    const before = fs.statSync(startLanguage.appFileIn(dir)).mtimeMs;
    expect(startLanguage.recordApp(dir, 'de')).toBe(true);
    expect(fs.statSync(startLanguage.appFileIn(dir)).mtimeMs).toBe(before);
  });

  it('reads as nothing when the file is corrupt, rather than throwing', () => {
    fs.writeFileSync(startLanguage.appFileIn(dir), 'not json');
    expect(startLanguage.appLanguage(dir)).toBe(null);
    fs.writeFileSync(startLanguage.appFileIn(dir), '{"lang":"kl"}');
    expect(startLanguage.appLanguage(dir)).toBe(null);
  });

  it('creates the directory if the app speaks before anything else wrote there', () => {
    const fresh = path.join(dir, 'not', 'there', 'yet');
    expect(startLanguage.recordApp(fresh, 'ja')).toBe(true);
    expect(startLanguage.appLanguage(fresh)).toBe('ja');
  });
});
