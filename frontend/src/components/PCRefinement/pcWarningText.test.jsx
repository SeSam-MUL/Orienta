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
    'warnings.smallPatterns': 'Kleine Muster ({{h}}×{{w}} px)',
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
        { code: 'smallPatterns', params: { h: 60, w: 60 } },
        { code: 'pcDrift', params: { deviation: '0.082' } },
      ],
    }, t);
    expect(out).toBe('Kleine Muster (60×60 px)  Das verfeinerte PC hat sich um 0.082 vom Start-PC entfernt');
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
