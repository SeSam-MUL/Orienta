/**
 * German quotation marks come in pairs, and the wrong half is easy to type.
 *
 * German opens with „ and closes with “. A keyboard gives you " for both,
 * so the usual mistake is to open correctly and close with a straight
 * quote -- which looks almost right and is wrong in every sentence it
 * appears in. Four strings in `phaselibrary.json` had it, all written in
 * one sitting, all read past twice.
 *
 * This project has form here: the changelog records `Zurueck` written in
 * ASCII and a truncated `Hinzu`, both found late by a person rather than a
 * test. Typography is not decoration in an app that ships in four
 * languages; it is the part nobody proofreads.
 *
 * Checked over EVERY German namespace, not only this feature's.
 */
import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DE = path.join(path.dirname(fileURLToPath(import.meta.url)), 'de');

function strings(obj, prefix = '') {
  const out = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v && typeof v === 'object') out.push(...strings(v, `${prefix}${k}.`));
    else if (typeof v === 'string') out.push([`${prefix}${k}`, v]);
  }
  return out;
}

const files = fs.readdirSync(DE).filter((f) => f.endsWith('.json'));

describe('German quotation marks', () => {
  it('there are German files to check', () => {
    // A walk that found nothing would pass forever.
    expect(files.length).toBeGreaterThan(5);
  });

  for (const file of files) {
    const dict = JSON.parse(fs.readFileSync(path.join(DE, file), 'utf-8'));

    it(`${file} closes every „ with a “`, () => {
      const bad = strings(dict)
        .filter(([, v]) => (v.match(/„/g) || []).length
          !== (v.match(/“/g) || []).length)
        .map(([k, v]) => `${k}: ${v}`);
      expect(bad).toEqual([]);
    });

    it(`${file} uses German quotes, not straight ones`, () => {
      // THE GUARD COULD NOT SEE A STRING QUOTED ENTIRELY WITH ". It looked
      // for a „ and then checked how it closed, so a sentence that never
      // opened correctly in the first place -- `Auf "Ordner hinzufügen"
      // klicken` -- passed. Twenty of them did, across twelve namespaces,
      // while this file's own docstring said typography is the part
      // nobody proofreads.
      //
      // A pair of straight quotes around anything is the mistake; a lone
      // " (an inch mark, a stray) is left alone rather than guessed at.
      const bad = strings(dict)
        .filter(([, v]) => /"[^"]+"/.test(v))
        .map(([k, v]) => `${k}: ${v}`);
      expect(bad).toEqual([]);
    });

    it(`${file} never closes a German quote with a straight one`, () => {
      // The specific mistake: „…" -- opened right, closed with the
      // keyboard's own character.
      const bad = strings(dict)
        .filter(([, v]) => /„[^„“]*"/.test(v))
        .map(([k, v]) => `${k}: ${v}`);
      expect(bad).toEqual([]);
    });
  }
});
