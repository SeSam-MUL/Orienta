/**
 * The four languages, held against each other.
 *
 * Every user-facing string exists four times, and nothing in the build notices
 * when one of the four is missing: i18next quietly falls back to the key or to
 * English, so a German user reads "settings:removeData.askLibrary" and nobody
 * hears about it. These tests compare the files themselves.
 *
 * They also pin two things that went wrong while writing them:
 *   * a platform named to users who cannot have it (WSL and NVIDIA on a Mac),
 *     which is what T5/T9 of the macOS plan is about;
 *   * placeholders that exist in one language and not another, which produces
 *     a sentence with a hole in it rather than an error.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const LOCALES = path.resolve(import.meta.dirname);
const LANGS = ['en', 'de', 'ja', 'zh'];

const read = (lang, file) =>
  JSON.parse(fs.readFileSync(path.join(LOCALES, lang, file), 'utf8'));

const namespaces = fs.readdirSync(path.join(LOCALES, 'en')).filter((f) => f.endsWith('.json'));

/** Every leaf key, dotted, sorted. */
function leaves(node, prefix = '') {
  const out = [];
  for (const [key, value] of Object.entries(node)) {
    const here = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === 'object' && !Array.isArray(value)) out.push(...leaves(value, here));
    else out.push(here);
  }
  return out.sort();
}

/** {{placeholders}} in a string. */
const placeholders = (text) =>
  [...String(text).matchAll(/\{\{(\w+)[^}]*\}\}/g)].map((m) => m[1]).sort();

function leafValue(node, dotted) {
  return dotted.split('.').reduce((acc, key) => (acc == null ? acc : acc[key]), node);
}

describe.each(namespaces)('%s', (file) => {
  const en = read('en', file);
  const enKeys = leaves(en);

  it.each(LANGS.filter((l) => l !== 'en'))('%s has exactly the same keys as en', (lang) => {
    const other = leaves(read(lang, file));
    const missing = enKeys.filter((k) => !other.includes(k));
    const extra = other.filter((k) => !enKeys.includes(k));
    expect({ missing, extra }).toEqual({ missing: [], extra: [] });
  });

  it('uses the same placeholders in every language', () => {
    const problems = [];
    for (const key of enKeys) {
      const expected = placeholders(leafValue(en, key));
      if (expected.length === 0) continue;
      for (const lang of LANGS.filter((l) => l !== 'en')) {
        const actual = placeholders(leafValue(read(lang, file), key));
        if (JSON.stringify(actual) !== JSON.stringify(expected)) {
          problems.push(`${lang}/${file}:${key} has ${JSON.stringify(actual)}, en has ${JSON.stringify(expected)}`);
        }
      }
    }
    expect(problems).toEqual([]);
  });
});

describe('platforms we do not name to people who cannot have them', () => {
  /**
   * Every place that may say "WSL" at all, and why.
   *
   * A list against a list, rather than a clever regular expression: this is a
   * decision per string, and the point is that a NEW string naming a Windows
   * feature has to be added here deliberately, by someone who then asks what
   * a Mac user reads. (The regex version of this test passed on German and
   * failed on Chinese for the same sentence — which is how it would have gone
   * on for the next string too.)
   */
  const MAY_SAY_WSL = [
    // The install wizard. Rendered only for platform.os === 'windows'
    // (InstallWizardSection.jsx:822); off Windows it shows "no WSL needed".
    /^settings\.json:install\./,
    // System status. The WSL row is dropped off Windows (systemStatusRows).
    /^settings\.json:systemStatus\.(rows\.wsl|recheckTooltip)$/,
    // Password hints belonging to that same wizard.
    /^settings\.json:hoverTips\.(confirmPassword|newPassword)$/,
    // Manual path overrides: these ARE WSL paths when there is a WSL.
    /^settings\.json:manualPaths\.(emsoftBinTooltip|emsphinxDirTooltip|hint)$/,
    // The simulation page says WSL is NOT needed for the built-in engine, and
    // names all three platforms for the EMsoft one.
    /^simulation\.json:(header\.subtitleOurs|engine\.oursTooltip|engine\.emsoftTooltip(Available|Unavailable)|actions\.startTooltipOurs)$/,
    // Spherical indexing: names Windows, Linux and macOS explicitly.
    /^indexing\.json:methods\.sphericalTip$/,
    // The status bar's WSL dot. StatusBar.jsx pushes it only when the backend
    // reports platform_os === 'windows', so these three strings cannot reach
    // a Mac or a Linux machine at all.
    /^shell\.json:status\.wsl(Ok|Distro|Missing)$/,
  ];

  it.each(LANGS)('%s says WSL only where we decided it may', (lang) => {
    const offenders = [];
    for (const file of namespaces) {
      const data = read(lang, file);
      for (const key of leaves(data)) {
        if (!/WSL/.test(String(leafValue(data, key)))) continue;
        const id = `${file}:${key}`;
        if (!MAY_SAY_WSL.some((re) => re.test(id))) offenders.push(id);
      }
    }
    expect(offenders).toEqual([]);
  });

  it.each(LANGS)('%s never reports a missing NVIDIA card without naming the platform', (lang) => {
    // "No NVIDIA GPU detected" on a Mac reads as a fault the user could fix.
    const offenders = [];
    for (const file of namespaces) {
      const data = read(lang, file);
      for (const key of leaves(data)) {
        const value = String(leafValue(data, key));
        if (!/No NVIDIA|Keine NVIDIA|NVIDIA GPU detected/.test(value)) continue;
        if (/^settings\.json:install\./.test(`${file}:${key}`)) continue;   // the Windows-only wizard
        if (/Windows|macOS|Linux|Mac/.test(value)) continue;
        offenders.push(`${file}:${key}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});

/**
 * Whole-branch review (2026-09-26), "smaller" findings: two collections
 * tooltips (`databasebrowser.json:manageCollectionsTooltip` and
 * `indexing.json:adoptCollection`) cited `docs/user-guide/Phase-
 * Collections.md` by its REPO-RELATIVE path — the only two strings in the
 * whole locale tree that named a filesystem path at all. An installed user
 * (packaged app, no `docs/` directory next to the executable) reads a
 * dangling reference to a file that is not there. Named the page instead
 * ("the guide" / a quoted title); this guards against a new string
 * reintroducing a repo path.
 */
describe('no user-facing string names a repo-relative file path', () => {
  // A path segment ("docs/", "src/", a filename with a slash before it, or a
  // bare `.md`/`.py`/`.jsx` reference) is what an installed user cannot open.
  // Deliberately broad — this whole category should stay empty, not grow a
  // list of exceptions the way MAY_SAY_WSL above does for a legitimate case.
  const PATH_LIKE = /\b[\w-]+\/[\w./-]*\.(md|py|jsx?|json|txt)\b/i;

  it.each(LANGS)('%s never points a user at a path inside the repository', (lang) => {
    const offenders = [];
    for (const file of namespaces) {
      const data = read(lang, file);
      for (const key of leaves(data)) {
        const value = String(leafValue(data, key));
        if (PATH_LIKE.test(value)) offenders.push(`${file}:${key} -> ${value}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
