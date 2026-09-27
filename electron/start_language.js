/**
 * The setup wizard's language, handed to the application once.
 *
 * With nothing stored the app starts in ENGLISH, whatever Windows or the wizard
 * spoke. The app keeps its language in the localStorage of its own origin
 * (http://127.0.0.1:8000), which the wizard -- a local page -- cannot write. So
 * the wizard records its language in a file, and the first load of the app
 * carries it in the URL, once.
 *
 * WHERE the file lives matters, and the first version got it wrong: it was in
 * the shell's userData, which sits INSIDE the data folder. A user who picked a
 * different data folder in the wizard relaunched into a process whose userData
 * was under the new folder, found no file, and started in English -- the exact
 * thing this exists to prevent. It now lives beside the data-folder pointer
 * (%APPDATA%\Orienta), which does not move when the data folder does.
 *
 * `explicit` separates "the user picked this in the wizard" from "the wizard
 * started in this". An explicit choice is never downgraded by a later
 * automatic one, and only an explicit choice overrides a language already
 * chosen inside the app (see frontend/src/i18n/startLanguage.js).
 *
 * No Electron here, so it can be tested by running it.
 */
const fs = require('node:fs');
const path = require('node:path');

const SUPPORTED = ['en', 'de', 'ja', 'zh'];

function fileIn(dir) {
  return path.join(dir, 'start-language.json');
}

function read(dir) {
  try {
    const value = JSON.parse(fs.readFileSync(fileIn(dir), 'utf8'));
    return value && SUPPORTED.includes(value.lang) ? value : null;
  } catch {
    return null;
  }
}

/** Record the wizard's language. Returns whether it was written. */
function record(dir, lang, explicit) {
  if (!SUPPORTED.includes(lang)) return false;
  const previous = read(dir);
  const next = (previous && previous.explicit && !explicit)
    ? previous
    : { lang, explicit: Boolean(explicit) };
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(fileIn(dir), JSON.stringify(next));
  return true;
}

/**
 * The query string to append to the first load, or ''. Does NOT delete the
 * file: that happens in `consumed()` once the page has actually loaded, so a
 * load that fails does not lose the choice.
 */
function queryFor(dir) {
  const value = read(dir);
  if (!value) return '';
  return `?lang=${value.lang}${value.explicit ? '&langExplicit=1' : ''}`;
}

function consumed(dir) {
  try { fs.rmSync(fileIn(dir), { force: true }); } catch { /* nothing to remove */ }
}

/**
 * The language the APP is running in, kept for the next start.
 *
 * Separate from start-language.json, which is a hand-off: the wizard writes
 * it, the first page load carries it, and `consumed()` deletes it. This one
 * is a memory and must survive, because the thing that needs it runs BEFORE
 * any page does — the splash screen, which used `app.getLocale()` and so
 * spoke the operating system's language. The M5 tester had Orienta in German
 * on a Mac set to English and met an English splash every start.
 *
 * Written by the app whenever the user changes language, read by the shell
 * on the next launch. A missing or unreadable file simply means "ask the OS",
 * which is what happened before this existed.
 */
function appFileIn(dir) {
  return path.join(dir, 'app-language.json');
}

/** What language did the app last run in? `null` when nothing is recorded. */
function appLanguage(dir) {
  try {
    const value = JSON.parse(fs.readFileSync(appFileIn(dir), 'utf8'));
    return value && SUPPORTED.includes(value.lang) ? value.lang : null;
  } catch {
    return null;
  }
}

/** Remember the app's language. Returns whether it was written. */
function recordApp(dir, lang) {
  if (!SUPPORTED.includes(lang)) return false;
  if (appLanguage(dir) === lang) return true;   // nothing to rewrite
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(appFileIn(dir), JSON.stringify({ lang }));
  return true;
}

module.exports = {
  SUPPORTED, fileIn, read, record, queryFor, consumed,
  appFileIn, appLanguage, recordApp,
};
