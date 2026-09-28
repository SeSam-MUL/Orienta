/**
 * The word "collection" must not reach the screen.
 *
 * Sebastian did not like it, and with the manager dialog gone the library
 * calls these things GROUPS. The routes, the JSON files in
 * `Database/Collections` and the code's own identifiers keep their names --
 * renaming those would be churn with a migration attached. This is about
 * what a person reads.
 *
 * Checked over the strings that a SHIPPED component still renders, rather
 * than over whole namespaces: they hold leftovers from the retired dialog,
 * and failing on those would say "delete this file" when the answer is
 * "nobody shows it".
 *
 * IT USED TO READ ONE NAMESPACE, AND THAT WAS THE BUG IN THE GUARD.
 * `collections.json` was the file the word was expected in, so that is the
 * file it opened -- and eleven live strings in `databasebrowser.json` and
 * `indexing.json` went on saying "collection" in all four languages,
 * including a dropdown labelled "Move N files to collection…", a tooltip
 * describing the retired exclusive-folder model, and two pointing at a
 * user guide deleted in the same commit that added this guard's subject.
 *
 * A guard drawn around the place a defect appeared teaches the defect to
 * appear elsewhere -- the same shape as `lineEndings.test.js`, which had to
 * be widened from `frontend/src` to the manual for exactly this reason.
 * This one now reads every namespace a shipped file renders from.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(HERE, '..');
const LANGS = ['en', 'de', 'ja', 'zh'];

/** Every source file that ships (tests and the locale JSON excluded). */
function shippedSources() {
  const out = [];
  const walk = (dir) => {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) {
        if (e.name !== 'locales') walk(p);
        continue;
      }
      if (!/\.(js|jsx)$/.test(e.name) || /\.test\.(js|jsx)$/.test(e.name)) continue;
      out.push(p);
    }
  };
  walk(SRC);
  return out;
}

/**
 * Every dotted key a shipped file mentions, in either spelling.
 *
 * `t('collections:counts.showAll')` and `t('groups.menuRemove')` both
 * count -- a component picks its namespace up from `useTranslation(...)`,
 * so the bare form is just as live as the qualified one, and the old
 * regex only saw the qualified one. Resolving a bare key against every
 * namespace over-matches slightly: a key that exists in a namespace
 * nobody renders it from is checked anyway. That way round is safe -- the
 * worst it can do is make the guard stricter about a string nobody sees.
 */
function keysMentioned() {
  const found = new Set();
  for (const p of shippedSources()) {
    const src = fs.readFileSync(p, 'utf-8');
    for (const m of src.matchAll(/['"`]([a-z]+:)?([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)['"`]/g)) {
      found.add(m[2]);
    }
  }
  return [...found];
}

const WORD = /collection|Sammlung|コレクション|合集/i;

/**
 * What a reader actually sees: the string with its placeholders taken out.
 *
 * The first version of this check failed on
 * `{{usable}} of {{total}} phases · {{collection}}` in all four languages
 * -- and that renders as the GROUP'S NAME. A placeholder is an identifier
 * in the code, not a word on screen, and a guard that cannot tell the
 * difference sends somebody renaming variables for nothing.
 */
const asRendered = (s) => s.replace(/\{\{[^}]*\}\}/g, '');

/** A key's value, or its `_one`/`_other` pair for an i18next plural. */
function valuesAt(dict, key) {
  const parts = key.split('.');
  const leaf = parts[parts.length - 1];
  const parent = parts.slice(0, -1).reduce((o, k) => (o || {})[k], dict) || {};
  const direct = parts.reduce((o, k) => (o || {})[k], dict);
  if (typeof direct === 'string') return [direct];
  return Object.entries(parent)
    .filter(([k]) => k === `${leaf}_one` || k === `${leaf}_other`)
    .map(([, v]) => v)
    .filter((v) => typeof v === 'string');
}

const namespaces = (lang) => fs
  .readdirSync(path.join(SRC, 'locales', lang))
  .filter((f) => f.endsWith('.json'))
  .map((f) => f.replace(/\.json$/, ''));

describe('the interface says "group"', () => {
  it('finds the strings that are actually rendered', () => {
    // A walk that matched nothing would pass forever.
    const keys = keysMentioned();
    expect(keys.length).toBeGreaterThan(200);
    expect(keys).toContain('counts.showAll');
    // ... and it reaches beyond the one namespace it used to read.
    expect(namespaces('en').length).toBeGreaterThan(5);
    expect(namespaces('en')).toContain('databasebrowser');
  });

  it('can still see the word when it is there', () => {
    // The guard that cannot fail is the one that let eleven strings
    // through. This asserts the detector, not the strings.
    expect(WORD.test(asRendered('Move 3 files to collection…'))).toBe(true);
    expect(WORD.test(asRendered('Sammlung „{{name}}“'))).toBe(true);
    expect(WORD.test(asRendered('{{collection}} phases'))).toBe(false);
  });

  for (const lang of LANGS) {
    it(`${lang} uses no word for "collection" in anything on screen`, () => {
      const keys = keysMentioned();
      const offenders = [];
      for (const ns of namespaces(lang)) {
        const dict = JSON.parse(fs.readFileSync(
          path.join(SRC, 'locales', lang, `${ns}.json`), 'utf-8'));
        for (const key of keys) {
          for (const v of valuesAt(dict, key)) {
            if (WORD.test(asRendered(v))) offenders.push(`${ns}:${key}: ${v}`);
          }
        }
      }
      expect(offenders).toEqual([]);
    });
  }
});
