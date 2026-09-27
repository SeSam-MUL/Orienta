/**
 * The wizard's vocabulary, held against the code that produces it.
 *
 * The setup was originally written with its own English sentences, separate
 * from electron/strings.js and translated nowhere — so a German user met
 * `ENOTFOUND` twenty minutes into their first install. The defence is not
 * "remember to translate": it is that every failure `installer.js` can return
 * has a code, and this suite fails if any code has no sentence, in any of the
 * four languages the rest of the application speaks.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const HERE = path.resolve(import.meta.dirname, '..', '..', 'electron');
const LOCALES = JSON.parse(fs.readFileSync(path.join(HERE, 'setup', 'locales.json'), 'utf8'));
const SETUP_JS = fs.readFileSync(path.join(HERE, 'setup', 'setup.js'), 'utf8');
const INDEX_HTML = fs.readFileSync(path.join(HERE, 'setup', 'index.html'), 'utf8');

const { CODES } = await import(path.join(HERE, 'setup', 'installer.js'))
  .then((m) => m.default || m);

const LANGS = ['en', 'de', 'ja', 'zh'];

/** Every leaf key, dotted. */
function keysOf(object, prefix = '') {
  return Object.entries(object).flatMap(([key, value]) => (
    value && typeof value === 'object' && !Array.isArray(value)
      ? keysOf(value, `${prefix}${key}.`)
      : [`${prefix}${key}`]
  ));
}

describe('the four languages', () => {
  it('are all present', () => {
    for (const lang of LANGS) expect(LOCALES[lang], lang).toBeTruthy();
  });

  it('have exactly the same keys', () => {
    const reference = keysOf(LOCALES.en).sort();
    for (const lang of LANGS.slice(1)) {
      const theirs = keysOf(LOCALES[lang]).sort();
      expect({ lang, missing: reference.filter((k) => !theirs.includes(k)) })
        .toEqual({ lang, missing: [] });
      expect({ lang, extra: theirs.filter((k) => !reference.includes(k)) })
        .toEqual({ lang, extra: [] });
    }
  });

  it('never leave a string empty', () => {
    // An empty string renders as a blank line, which reads as a broken page
    // rather than as a missing translation.
    for (const lang of LANGS) {
      for (const key of keysOf(LOCALES[lang])) {
        const value = key.split('.').reduce((o, k) => o[k], LOCALES[lang]);
        expect(String(value).trim(), `${lang}.${key}`).not.toBe('');
      }
    }
  });

  it('use the same placeholders in every language', () => {
    // A translation that drops {{size}} renders "About GB", and one that
    // invents {{groesse}} renders it literally.
    const placeholders = (s) => (String(s).match(/\{\{\w+\}\}/g) || []).sort();
    for (const key of keysOf(LOCALES.en)) {
      const read = (lang) => key.split('.').reduce((o, k) => o[k], LOCALES[lang]);
      const reference = placeholders(read('en'));
      for (const lang of LANGS.slice(1)) {
        expect(placeholders(read(lang)), `${lang}.${key}`).toEqual(reference);
      }
    }
  });
});

describe('every failure the installer can produce', () => {
  it('has a sentence in all four languages', () => {
    for (const code of Object.values(CODES)) {
      for (const lang of LANGS) {
        expect(LOCALES[lang].errors[code], `${lang}.errors.${code}`).toBeTruthy();
      }
    }
  });

  it('has no sentence for a code that does not exist', () => {
    // The other direction: a stale entry means a code was renamed and the
    // wizard now falls back to English for the new one without anyone noticing.
    const codes = new Set(Object.values(CODES));
    for (const key of Object.keys(LOCALES.en.errors)) {
      expect(codes.has(key), `errors.${key} matches no code in CODES`).toBe(true);
    }
  });
});

describe('every step the installer announces', () => {
  it('has a label in all four languages', () => {
    // Read from setup.js's own STEPS list rather than repeated here, so the
    // two cannot drift.
    const listed = SETUP_JS.match(/const STEPS = \[([^\]]+)\]/);
    expect(listed).toBeTruthy();
    const steps = listed[1].match(/'(\w+)'/g).map((s) => s.slice(1, -1));
    expect(steps.length).toBeGreaterThan(4);
    for (const step of steps) {
      for (const lang of LANGS) {
        expect(LOCALES[lang].steps[step], `${lang}.steps.${step}`).toBeTruthy();
      }
    }
  });

  it('matches the steps runSetup actually emits', () => {
    const installer = fs.readFileSync(path.join(HERE, 'setup', 'installer.js'), 'utf8');
    const body = installer.slice(installer.indexOf('async function runSetup'));
    const emitted = new Set(
      [...body.matchAll(/currentStep = '(\w+)'/g)].map((m) => m[1]));
    for (const step of emitted) {
      expect(LOCALES.en.steps[step], `runSetup emits '${step}' with no label`).toBeTruthy();
    }
  });
});

describe('the wizard page', () => {
  it('writes no user-visible English of its own', () => {
    // Everything the user reads goes through t(). Text sitting in the HTML
    // would be a second vocabulary, translated nowhere — which is the defect
    // this whole file exists to prevent.
    const words = INDEX_HTML
      .replace(/<!--[\s\S]*?-->/g, '')
      .replace(/<(script|style)[\s\S]*?<\/\1>/g, '')
      .replace(/<[^>]+>/g, '\n')
      .split('\n').map((s) => s.trim()).filter(Boolean);
    // Two kinds of literal are allowed, and only these two. The product name,
    // once as <title> and once as the <h1> placeholder t('title') overwrites
    // on load. And the four language names in the picker, each written in its
    // own language — "German" is no use to someone who cannot read the page it
    // is currently written on, so those are deliberately NOT translated.
    const allowed = new Set(['Orienta', 'English', 'Deutsch', '日本語', '中文']);
    const unexpected = [...new Set(words)].filter((w) => !allowed.has(w));
    expect(unexpected).toEqual([]);
  });

  it('refuses remote script and inline style', () => {
    // This page downloads gigabytes; none of it may become script here.
    expect(INDEX_HTML).toMatch(/Content-Security-Policy/);
    expect(INDEX_HTML).toMatch(/default-src 'none'/);
    expect(INDEX_HTML).toMatch(/script-src 'self'/);
    expect(INDEX_HTML).not.toMatch(/unsafe-inline/);
  });

  it('has an element for every id setup.js reaches for', () => {
    // getElementById on a missing id returns null and the wizard dies at the
    // first property access — on the one screen the user has.
    const ids = new Set([...SETUP_JS.matchAll(/\$\('([\w]+)'\)/g)].map((m) => m[1]));
    const present = new Set([...INDEX_HTML.matchAll(/\bid="([\w]+)"/g)].map((m) => m[1]));
    // `probing` is created by setup.js itself.
    present.add('probing');
    const missing = [...ids].filter((id) => !present.has(id));
    expect(missing).toEqual([]);
  });

  it('loads its stylesheet and script as separate files', () => {
    expect(INDEX_HTML).toMatch(/<link rel="stylesheet" href="setup\.css">/);
    expect(INDEX_HTML).toMatch(/<script src="setup\.js"><\/script>/);
  });
});

describe('the bridge', () => {
  it('exposes exactly what the wizard calls, and the main process handles it', () => {
    const preload = fs.readFileSync(path.join(HERE, 'preload.js'), 'utf8');
    const main = fs.readFileSync(path.join(HERE, 'main.js'), 'utf8');

    const used = new Set([...SETUP_JS.matchAll(/window\.setupAPI\.(\w+)/g)].map((m) => m[1]));
    expect(used.size).toBeGreaterThan(4);

    const bridge = preload.slice(preload.indexOf("exposeInMainWorld('setupAPI'"));
    // Both registrars: the setup channels go through `handleSetup`, which
    // wraps ipcMain.handle with a check that the caller is the wizard window.
    const channels = new Set([
      ...main.matchAll(/ipcMain\.handle\('([\w:]+)'/g),
      ...main.matchAll(/handleSetup\('([\w:]+)'/g),
    ].map((m) => m[1]));

    for (const method of used) {
      expect(bridge, `setupAPI.${method} is not exposed`).toMatch(new RegExp(`\\b${method}:`));
    }
    // ...and each channel the bridge invokes is actually handled. A missing
    // handler rejects with "No handler registered", which the wizard would
    // show as an unexpected error on the failure screen.
    for (const [, channel] of bridge.matchAll(/ipcRenderer\.invoke\('([\w:]+)'/g)) {
      expect(channels.has(channel), `no ipcMain.handle for '${channel}'`).toBe(true);
    }
  });

  it('hands back an unsubscribe rather than leaking a listener per retry', () => {
    const preload = fs.readFileSync(path.join(HERE, 'preload.js'), 'utf8');
    const bridge = preload.slice(preload.indexOf("exposeInMainWorld('setupAPI'"));
    expect(bridge).toMatch(/removeListener\('setup:progress'/);
    expect(SETUP_JS).toMatch(/const stop = window\.setupAPI\.onProgress/);
    expect(SETUP_JS).toMatch(/stop\(\);/);
  });
});

describe('choosing a language', () => {
  it('offers every language the vocabulary has, and no others', () => {
    // A <option> for a language locales.json does not carry renders raw keys;
    // a language present in the file with no option is unreachable, since the
    // system locale is the only other way in.
    const offered = [...INDEX_HTML.matchAll(/<option value="(\w+)"/g)].map((m) => m[1]);
    expect(offered.sort()).toEqual([...LANGS].sort());
  });

  it('names each language IN that language', () => {
    // "German" is no use to someone who cannot read the page it is written on.
    for (const [, value, label] of INDEX_HTML.matchAll(/<option value="(\w+)">([^<]+)</g)) {
      const native = { en: 'English', de: 'Deutsch', ja: '日本語', zh: '中文' }[value];
      expect(label.trim(), value).toBe(native);
    }
  });

  it('re-renders rather than reloading', () => {
    // A reload during a twenty-minute install would restart the renderer while
    // the install carried on in the main process, and would throw away the
    // probe the choose screen is built from.
    expect(SETUP_JS).toMatch(/function applyLanguage/);
    expect(SETUP_JS).not.toMatch(/location\.reload/);
    // ...and it redraws whichever screen is actually on show.
    const body = SETUP_JS.slice(SETUP_JS.indexOf('function applyLanguage'),
      SETUP_JS.indexOf('async function main'));
    for (const screen of ['choose', 'failed', 'working', 'done']) {
      expect(body, `a language switch leaves the ${screen} screen stale`)
        .toMatch(new RegExp(`'${screen}'`));
    }
  });

  it('starts from the system language', () => {
    expect(SETUP_JS).toMatch(/applyLanguage\(ctx\.locale\)/);
  });
});
