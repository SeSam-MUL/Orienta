// @vitest-environment jsdom
/**
 * The glossary: every term explained, in every language, and reachable
 * without a mouse.
 *
 * A first-time-user agent read "cF4" and "Cu" on a card about aluminium and
 * concluded the entry was wrong. `Cu` there is the structure type -- pure Al
 * and pure Ni both have it -- and nothing on the screen said so. Eight terms
 * are like that; these tests check that all eight are explained rather than
 * a convenient subset.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';
import i18n from '../../i18n';
import GlossaryLegend, { GlossaryTerm } from './GlossaryLegend';
import { GLOSSARY, GLOSSARY_KEYS, glossaryEntry } from './glossary';
import en from '../../locales/en/phaselibrary.json';
import de from '../../locales/de/phaselibrary.json';
import ja from '../../locales/ja/phaselibrary.json';
import zh from '../../locales/zh/phaselibrary.json';

afterEach(cleanup);

describe('the terms the page uses without explaining them', () => {
  it('are the eight §2.3 names, plus the two the user loop added', () => {
    expect(GLOSSARY_KEYS).toEqual([
      'pearson', 'prototype', 'itNumber', 'cif', 'xtal', 'master', 'sht', 'ht',
      // `placements` is in the HEADLINE of the page and was the one term of
      // art the glossary did not define; a reader said so in as many words.
      // `by name only` is a match tag the reader is expected to act on.
      'placements', 'labelOnly',
    ]);
  });

  it('says the things a domain reader corrected', () => {
    // Three factual errors, found by a reader who knows the field:
    //   - the centring list omitted S, and the FIRST data row on the page is
    //     `mS102`, so the glossary could not explain the first symbol shown
    //   - "the number has one" is true of the group and false of what a
    //     reader needs: the IT number fixes neither setting nor origin, and
    //     this library holds that exact case (Fd-3m against Fd-3m O1)
    //   - an EMsoft master is computed over an energy RANGE, not at one
    //     energy, which is the answer to "can I use the 20 kV master at 15?"
    expect(en.glossary.pearson.text).toMatch(/S side-centred/);
    expect(en.glossary.pearson.text).toMatch(/trigonal/);
    expect(en.glossary.itNumber.text).toMatch(/not the setting or the origin/i);
    expect(en.glossary.itNumber.text).toMatch(/Fd-3m O1/);
    expect(en.glossary.master.text).toMatch(/RANGE of beam energies/);
    for (const doc of [de, ja, zh]) {
      expect(doc.glossary.master.text).not.toMatch(/one beam energy|einer Strahlenergie/);
    }
  });

  it('every one has a name and an explanation in all four languages', () => {
    for (const [lng, doc] of [['en', en], ['de', de], ['ja', ja], ['zh', zh]]) {
      for (const key of GLOSSARY_KEYS) {
        const entry = doc.glossary?.[key];
        expect(entry, `${lng}/${key}`).toBeTruthy();
        expect(entry.name, `${lng}/${key} name`).toBeTruthy();
        // Long enough to be an explanation rather than a restatement of the
        // term. The shortest real one is about 90 characters.
        expect(entry.text.length, `${lng}/${key} text`).toBeGreaterThan(40);
      }
    }
  });

  it('an unknown term is a programming error, not a blank tooltip', () => {
    expect(() => glossaryEntry('nope')).toThrow(/no glossary entry/);
  });

  it('marks the abbreviations as abbreviations and the terms as terms', () => {
    // Not decoration: a screen reader treats <abbr> differently.
    const abbrs = GLOSSARY.filter((g) => g.abbr).map((g) => g.key);
    expect(abbrs).toEqual(['itNumber', 'cif', 'xtal', 'sht', 'ht']);
  });
});

describe('the explanations say what this library actually contains', () => {
  it('the structure type explanation names the case that misled a reader', () => {
    expect(en.glossary.prototype.text).toMatch(/aluminium and pure nickel/i);
    expect(en.glossary.prototype.text).toContain('Cu');
  });

  it('the Pearson explanation decodes the symbol aluminium really has', () => {
    expect(en.glossary.pearson.text).toContain('cF4');
  });

  it('ht names the one phase here that carries it', () => {
    // Measured: exactly one, `Al4Fe1.7Si (τ11)`, whose label is
    // `Fe1.8Al4.4Si0.6 ht`.
    for (const doc of [en, de, ja, zh]) {
      expect(doc.glossary.ht.text).toContain('Fe1.8Al4.4Si0.6 ht');
    }
  });
});

describe('on screen', () => {
  it('a term carries its explanation where it stands', async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<GlossaryTerm termKey="pearson" />);
    const el = document.querySelector('[data-glossary="pearson"]');
    expect(el.tagName).toBe('SPAN');                  // a term, not an abbr
    expect(el.getAttribute('title')).toContain('cF4');
    expect(el.textContent).toBe('Pearson symbol');
  });

  it('an abbreviation is an <abbr>', async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<GlossaryTerm termKey="sht" />);
    expect(document.querySelector('[data-glossary="sht"]').tagName).toBe('ABBR');
  });

  it('a term can be written differently from its glossary name', async () => {
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<GlossaryTerm termKey="cif">.cif file</GlossaryTerm>);
    expect(document.querySelector('[data-glossary="cif"]').textContent)
      .toBe('.cif file');
  });

  it('the legend holds all eight and needs no mouse', async () => {
    // A tooltip that appears only on hover is not a keyboard path (§2.9).
    // <details> is focusable and operable without a line of JavaScript.
    await act(async () => { await i18n.changeLanguage('en'); });
    render(<GlossaryLegend />);
    const box = screen.getByTestId('glossary');
    expect(box.tagName).toBe('DETAILS');
    expect(box.querySelector('summary')).toBeTruthy();
    for (const key of GLOSSARY_KEYS) {
      expect(box.querySelector(`[data-glossary-entry="${key}"]`), key).toBeTruthy();
    }
  });

  it('switches language with the rest of the app', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    render(<GlossaryLegend />);
    expect(screen.getByTestId('glossary').textContent).toContain('Strukturtyp');
    await act(async () => { await i18n.changeLanguage('en'); });
  });
});
