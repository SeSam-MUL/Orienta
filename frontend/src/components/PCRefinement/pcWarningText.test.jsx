/**
 * The PC refine reliability warning, in the reader's language.
 *
 * The M5 tester ran the app in German and read the whole warning in English
 * (report, section 4): "Refined PC moved 0.082 from the starting PC …". It is
 * written by the backend, so it stayed English whatever language was chosen.
 */
import { describe, it, expect } from 'vitest';
import { pcWarningText } from './PCRefinement';

const t = (key, opts = {}) => {
  const table = {
    'warnings.pcDrift': 'Das verfeinerte PC hat sich um {{deviation}} vom Start-PC entfernt',
    'warnings.pcOnBoxEdge': 'Am Rand des Suchbereichs in {{axes}}',
    'warnings.pcUnclear': 'Nicht eindeutig (Kontrast {{contrast}})',
  };
  const hit = table[key];
  if (hit === undefined) return opts.defaultValue ?? key;
  return hit.replace(/{{(\w+)}}/g, (_, k) => opts[k] ?? '');
};

describe('pcWarningText', () => {
  it('translates on the codes and fills their numbers', () => {
    expect(pcWarningText({
      pc_warning: 'Refined PC moved 0.082 from the starting PC (…)',
      pc_warning_codes: [{ code: 'pcDrift', params: { deviation: '0.082' } }],
    }, t)).toBe('Das verfeinerte PC hat sich um 0.082 vom Start-PC entfernt');
  });

  it('joins several warnings the way the backend did', () => {
    const out = pcWarningText({
      pc_warning: 'a  b',
      pc_warning_codes: [
        { code: 'pcOnBoxEdge', params: { axes: 'y, z' } },
        { code: 'pcDrift', params: { deviation: '0.082' } },
      ],
    }, t);
    expect(out).toBe('Am Rand des Suchbereichs in y, z  Das verfeinerte PC hat sich um 0.082 vom Start-PC entfernt');
  });

  it('says nothing about clarity while the search does not judge: the backend sends no pcUnclear then', () => {
    // state "notJudged" brings no pcUnclear code; the page then says only that the result is to be checked.
    expect(pcWarningText({ pc_warning: null, pc_warning_codes: [] }, t)).toBe('');
    expect(pcWarningText({
      pc_warning: 'English fallback',
      pc_warning_codes: [{ code: 'pcUnclear', params: { contrast: 0.1 } }],
    }, t)).toBe('Nicht eindeutig (Kontrast 0.1)');
  });

  it('falls back to the English sentence when a code has no text', () => {
    // An older frontend against a newer backend must say something true
    // rather than print a translation key.
    expect(pcWarningText({
      pc_warning: 'Something the backend explained in English',
      pc_warning_codes: [{ code: 'aCodeFromTheFuture' }],
    }, t)).toBe('Something the backend explained in English');
  });

  it('falls back when the backend sent no codes at all', () => {
    expect(pcWarningText({ pc_warning: 'older backend' }, t)).toBe('older backend');
    expect(pcWarningText({}, t)).toBe('');
    expect(pcWarningText(null, t)).toBe('');
  });
});
