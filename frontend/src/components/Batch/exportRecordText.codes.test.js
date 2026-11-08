// @vitest-environment jsdom
/**
 * The words of the export record for the notes and reasons that a .up1/.up2 source and a failed
 * post-processing step add: edsNotCopied, cleanupFailed, autoAssignmentFailed (notes) and
 * richNeedsHdf5Source (a reason). Each is worded in all four languages, German with "Sie", and the
 * technical detail of an exception follows the translated sentence unchanged.
 */
import { describe, it, expect } from 'vitest';
import i18n from '../../i18n';
import { noteText, reasonText } from './exportRecordText';

const LANGS = ['en', 'de', 'ja', 'zh'];
const DETAIL = 'boom <x> & "y"';

const noteOf = (code, params = {}) => ({ code, params, text: 'English fallback sentence' });

describe('export record texts for a .up1 source and for failed post-processing', () => {
  for (const lang of LANGS) {
    const t = i18n.getFixedT(lang, 'batch');

    it(`${lang}: edsNotCopied and richNeedsHdf5Source are worded, not the English fallback`, () => {
      const note = noteText(noteOf('edsNotCopied'), t);
      const reason = reasonText(noteOf('richNeedsHdf5Source'), t);
      for (const text of [note, reason]) {
        expect(text).not.toBe('English fallback sentence');
        expect(text).not.toContain('.up1');                      // no file type: it holds for any non-HDF5 file
        expect(text).not.toContain('.up2');
        expect(text).not.toContain('exports.');                  // never a translation key
      }
    });

    it(`${lang}: cleanupFailed and autoAssignmentFailed carry the technical detail`, () => {
      for (const code of ['cleanupFailed', 'autoAssignmentFailed']) {
        const text = noteText(noteOf(code, { detail: DETAIL }), t);
        expect(text).not.toBe('English fallback sentence');
        expect(text).toContain(DETAIL);                           // unescaped, as the page shows it
        expect(text).not.toContain('{{');
      }
    });
  }

  it('de: speaks to the reader with Sie', () => {
    const t = i18n.getFixedT('de', 'batch');
    expect(reasonText(noteOf('richNeedsHdf5Source'), t)).toContain('Verwenden Sie die leichte .h5');
  });

  it('the four languages word "another file format" for the rich .h5 reason', () => {
    // by hand: the sentence "A rich .h5 needs an HDF5 source; this scan's patterns are in another
    // file format. Use the light .h5." in each language
    const expected = {
      en: 'another file format',
      de: 'einem anderen Dateiformat',
      ja: '別のファイル形式',
      zh: '其他文件格式',
    };
    for (const lang of LANGS) {
      const t = i18n.getFixedT(lang, 'batch');
      expect(reasonText(noteOf('richNeedsHdf5Source'), t)).toContain(expected[lang]);
      expect(noteText(noteOf('edsNotCopied'), t)).toContain(expected[lang]);
    }
  });

  it('a code this build does not know shows the backend sentence', () => {
    const t = i18n.getFixedT('de', 'batch');
    expect(noteText(noteOf('aFutureCode'), t)).toBe('English fallback sentence');
  });

  it('a detail-note without its detail shows the backend sentence, not a half-filled one', () => {
    const t = i18n.getFixedT('en', 'batch');
    expect(noteText(noteOf('cleanupFailed', {}), t)).toBe('English fallback sentence');
  });
});
