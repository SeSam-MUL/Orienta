import { describe, expect, it } from 'vitest';
import i18n from '../../i18n';
import { errorCode, errorText, ladderChoices, ladderText, splitDoiOrCitation, translateFitWarnings } from './cifWarnings';

describe('splitDoiOrCitation', () => {
  it.each([
    ['10.1000/abc', { doi: '10.1000/abc', reference: '' }],
    ['  10.1107/S0021889811047258  ', { doi: '10.1107/S0021889811047258', reference: '' }],
    ['https://doi.org/10.1000/abc', { doi: '10.1000/abc', reference: '' }],
    ['http://dx.doi.org/10.1000/abc', { doi: '10.1000/abc', reference: '' }],
    ['doi: 10.1000/abc', { doi: '10.1000/abc', reference: '' }],
    ['Smith, J. (2020). Acta 1, 2.', { doi: '', reference: 'Smith, J. (2020). Acta 1, 2.' }],
    // A sentence that merely contains a DOI is a citation, not a DOI.
    ['Smith 2020, doi 10.1000/abc', { doi: '', reference: 'Smith 2020, doi 10.1000/abc' }],
    ['10.5', { doi: '', reference: '10.5' }],
    ['10.12/abc', { doi: '', reference: '10.12/abc' }],
    ['10.123/abc', { doi: '10.123/abc', reference: '' }],
    ['   ', { doi: '', reference: '' }],
    [undefined, { doi: '', reference: '' }],
  ])('%j', (text, expected) => {
    expect(splitDoiOrCitation(text)).toEqual(expected);
  });
});

describe('errorText / errorCode', () => {
  it('reads a string detail, an object detail and a plain error', () => {
    expect(errorText({ response: { data: { detail: 'plain' } } })).toBe('plain');
    expect(errorText({ response: { data: { detail: { code: 'x', message: 'coded' } } } })).toBe('coded');
    expect(errorText({ message: 'network down' })).toBe('network down');
    expect(errorCode({ response: { data: { detail: { code: 'reference_missing', message: 'm' } } } })).toBe('reference_missing');
    expect(errorCode({ response: { data: { detail: 'plain' } } })).toBe('');
    expect(errorCode(null)).toBe('');
  });
});

describe('translateFitWarnings', () => {
  const info = {
    fit_warning_codes: ['no_reference', 'p1_hidden_symmetry'],
    fit_warnings: ['No reference or DOI found.', 'Stored as space group P 1, but ... Fm-3m (225) at 0.01 A.'],
    detected_space_group: { symbol: 'Fm-3m', number: 225, symprec: 0.01 },
  };

  it('translates every code in all four languages', async () => {
    const seen = new Set();
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      await i18n.changeLanguage(lang);
      const t = i18n.getFixedT(lang, 'crystaldatabase');
      const [ref, p1] = translateFitWarnings(t, info);
      expect(ref).not.toBe('No reference or DOI found.');          // not the backend fallback
      expect(ref).not.toMatch(/^fitWarnings\./);                    // not a bare key
      expect(p1).toContain('Fm-3m');
      expect(p1).toContain('225');
      expect(p1).toMatch(/0[.,]01/);                                // the whole ladder, not one tolerance
      expect(p1).toMatch(/0[.,]1/);
      seen.add(ref);
    }
    expect(seen.size).toBe(4);                                      // four different texts
    await i18n.changeLanguage('en');
  });

  it('names the ladder, not one group, where the group depends on the tolerance', async () => {
    const ambiguous = {
      fit_warning_codes: ['p1_ambiguous_symmetry'],
      fit_warnings: ['x'],
      p1_ambiguous: true,
      symmetry_ladder: [
        { symprec: 0.01, number: 8, symbol: 'Cm' },
        { symprec: 0.05, number: 225, symbol: 'Fm-3m' },
      ],
    };
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      await i18n.changeLanguage(lang);
      const t = i18n.getFixedT(lang, 'crystaldatabase');
      const [text] = translateFitWarnings(t, ambiguous);
      expect(text).toContain('0.01 Å: Cm (8); 0.05 Å: Fm-3m (225)');
      expect(text).not.toMatch(/\{\{/);
    }
    await i18n.changeLanguage('en');
  });

  it('names the number and the symbol a contradictory header gives, in every language', async () => {
    const info = {
      fit_warning_codes: ['contradictory_header'],
      fit_warnings: ['x'],
      header_contradiction: { number: 1, symbol: 'F m -3 m' },
    };
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      await i18n.changeLanguage(lang);
      const t = i18n.getFixedT(lang, 'crystaldatabase');
      const [text] = translateFitWarnings(t, info);
      expect(text).toContain('F m -3 m');
      expect(text).toContain('1');
      expect(text).not.toMatch(/\{\{/);
      expect(text).not.toBe('x');
      // The refusal of the save route has its own text in the same language.
      expect(t('symmetriseReasons.contradictory_header')).not.toMatch(/^symmetriseReasons\./);
    }
    await i18n.changeLanguage('en');
  });

  it('says what could not be checked, and shows a failed rung as a question mark', async () => {
    const info = { fit_warning_codes: ['p1_check_failed'], fit_warnings: ['x'], p1_check_error: 'RuntimeError: spglib got no cell' };
    for (const lang of ['en', 'de', 'ja', 'zh']) {
      await i18n.changeLanguage(lang);
      const t = i18n.getFixedT(lang, 'crystaldatabase');
      const [text] = translateFitWarnings(t, info);
      expect(text).toContain('RuntimeError: spglib got no cell');
      expect(text).not.toBe('x');
      expect(text).not.toContain('{{');
    }
    await i18n.changeLanguage('en');
    expect(ladderText([{ symprec: 0.01, number: 225, symbol: 'Fm-3m' }, { symprec: 0.1, number: null, symbol: '', error: 'e' }]))
      .toBe('0.01 \u00c5: Fm-3m (225); 0.1 \u00c5: ?');
    expect(ladderChoices([{ symprec: 0.01, number: null, symbol: '' }, { symprec: 0.05, number: 225, symbol: 'Fm-3m' }]).map(r => r.symprec)).toEqual([0.05]);
  });

  it('offers one choice per distinct group above P 1, in ladder order', () => {
    const ladder = [
      { symprec: 0.01, number: 8, symbol: 'Cm' },
      { symprec: 0.02, number: 8, symbol: 'Cm' },
      { symprec: 0.05, number: 225, symbol: 'Fm-3m' },
      { symprec: 0.1, number: 1, symbol: 'P1' },
    ];
    expect(ladderChoices(ladder).map(r => [r.symprec, r.number])).toEqual([[0.01, 8], [0.05, 225]]);
    expect(ladderChoices(undefined)).toEqual([]);
  });

  it('keeps the German text German', async () => {
    await i18n.changeLanguage('de');
    const t = i18n.getFixedT('de', 'crystaldatabase');
    expect(translateFitWarnings(t, info)[0]).toContain('unveröffentlicht');
    await i18n.changeLanguage('en');
  });

  it('falls back to the English sentence for a code this build does not know', async () => {
    await i18n.changeLanguage('de');
    const t = i18n.getFixedT('de', 'crystaldatabase');
    const out = translateFitWarnings(t, { fit_warning_codes: ['from_the_future'], fit_warnings: ['Something new.'] });
    expect(out).toEqual(['Something new.']);
    await i18n.changeLanguage('en');
  });

  it('shows the sentences of an older backend that sends no codes', () => {
    const t = i18n.getFixedT('en', 'crystaldatabase');
    expect(translateFitWarnings(t, { fit_warnings: ['Old text.'] })).toEqual(['Old text.']);
    expect(translateFitWarnings(t, null)).toEqual([]);
  });
});
