/**
 * Export warnings in the reader's language.
 *
 * The backend has sent a stable `code` next to its English `message` since
 * this export was built, and the dialog printed the message: a German user
 * read English warnings under a German heading, which is what the M5 tester
 * reported. This pins that the code decides, and that the English sentence
 * is still there when a code has no text.
 */
import { describe, it, expect } from 'vitest';
import { warnText } from './ExportDialog';

const t = (key, opts = {}) => {
  const table = {
    'export.warn.no_step_size':
      'Diese Datei enthält keine Schrittweite für das EDS- oder EBSD-Gitter.',
  };
  const hit = table[key];
  return hit === undefined ? (opts.defaultValue ?? key) : hit;
};

describe('warnText', () => {
  it('translates on the code', () => {
    expect(warnText(
      { code: 'no_step_size', message: 'This file carries no step size…' }, t,
    )).toBe('Diese Datei enthält keine Schrittweite für das EDS- oder EBSD-Gitter.');
  });

  it('keeps the English sentence for a warning that quotes an exception', () => {
    // These still carry their detail inside the prose; the backend does not
    // send the pieces separately yet, so the fallback is the whole truth.
    expect(warnText(
      { code: 'xlsx_failed', message: 'summary.xlsx could not be created (disk full).' }, t,
    )).toBe('summary.xlsx could not be created (disk full).');
  });

  it('handles a warning with no code and no warning at all', () => {
    expect(warnText({ message: 'plain' }, t)).toBe('plain');
    expect(warnText({}, t)).toBe('');
    expect(warnText(null, t)).toBe('');
  });
});
