/**
 * The fallback message must not deny a feature that ships.
 *
 * `setupMissing` said, in all four languages, that the setup assistant "is not
 * part of this build yet" — while `electron/setup/index.html` is in the repo and
 * in the package. `loadSetupPlaceholder` loads that file when it EXISTS and only
 * falls through to this message when it is missing or unreadable, i.e. on a
 * damaged installation. So the sentence described the build when it should have
 * described the installation, and it sent the reader looking for a newer release
 * that would not have helped.
 *
 * These tests pin the two halves that can rot apart: the wizard is really there,
 * and the message really talks about this installation.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const requireCjs = createRequire(import.meta.url);

const ELECTRON = path.resolve(import.meta.dirname, '..', '..', 'electron');
const STRINGS = fs.readFileSync(path.join(ELECTRON, 'strings.js'), 'utf8');
const MAIN = fs.readFileSync(path.join(ELECTRON, 'main.js'), 'utf8');
const LANGS = ['en', 'de', 'ja', 'zh'];

// electron/strings.js is CommonJS; same helper the other Electron tests use.
const { STRINGS: TABLE, t } = requireCjs('../../electron/strings.js');

describe('setupMissing', () => {
  it('the wizard it used to deny is actually shipped', () => {
    // If this ever fails the message was right and the fix is wrong.
    expect(fs.existsSync(path.join(ELECTRON, 'setup', 'index.html'))).toBe(true);
  });

  it('fires only when the wizard file is missing, which is why the wording matters', () => {
    // loadSetupPlaceholder: loads the wizard when it exists, and the message is
    // the fallback below that check.
    const fn = MAIN.slice(MAIN.indexOf('function loadSetupPlaceholder'));
    const existsAt = fn.indexOf('fs.existsSync(wizard)');
    const loadAt = fn.indexOf('window.loadFile(wizard)');
    expect(existsAt).toBeGreaterThan(-1);
    expect(loadAt).toBeGreaterThan(existsAt);
  });

  it.each(LANGS)('%s no longer claims the build lacks the assistant', (lang) => {
    const message = t(lang, 'setupMissing');
    expect(message).toBeTruthy();
    expect(message).not.toMatch(/not part of this build/i);
    expect(message).not.toMatch(/noch nicht enthalten/);
    expect(message).not.toMatch(/まだ含まれていません/);
    expect(message).not.toMatch(/尚未包含/);
  });

  it.each(LANGS)('%s says the installation is incomplete and names the way out', (lang) => {
    const message = t(lang, 'setupMissing');
    // the reader needs the remedy, in their language
    const remedy = {
      en: /reinstall/i, de: /neu/i, ja: /再インストール/, zh: /重新安装/,
    }[lang];
    expect(message).toMatch(remedy);
    expect(message).toMatch(/INSTALL\.md/);
  });

  it('all four languages have it, and none was left as English', () => {
    const seen = LANGS.map((l) => t(l, 'setupMissing'));
    expect(new Set(seen).size).toBe(LANGS.length);
    for (const lang of LANGS) {
      expect(TABLE[lang]).toHaveProperty('setupMissing');
    }
  });

  it('no comment in main.js still says the wizard has not shipped', () => {
    // A regex written around the exact phrase that was deleted proves nothing.
    // The first pass removed "until the wizard ships" and left TWO others --
    // including the doc comment of loadSetupPlaceholder itself, which said
    // "Task 12 ... replaces this with the real wizard ... until that file
    // exists". So match the CLAIM, in every phrasing found in the file.
    const stale = [
      /until the wizard ships/i,
      /Task 12 (?:of the installer plan )?(?:replaces|ships)/i,
      /until that file exists/i,
      /real wizard[^.]*\bnot\b/i,
    ];
    const hits = stale.filter((re) => re.test(MAIN)).map(String);
    expect(hits, 'main.js still claims the shipped wizard is missing').toEqual([]);
  });

  it('and the doc comment describes what the function actually does', () => {
    const fn = MAIN.slice(MAIN.indexOf('* The page shown when there is nothing installed'));
    const doc = fn.slice(0, fn.indexOf('*/'));
    expect(doc).toMatch(/setup[/\\]index\.html/);
    expect(doc).toMatch(/missing or unreadable|damaged/i);
  });

  it('strings.js still parses as one table per language', () => {
    for (const lang of LANGS) expect(STRINGS).toContain(`  ${lang}: {`);
  });
});
