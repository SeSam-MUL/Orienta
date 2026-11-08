// @vitest-environment jsdom
/**
 * The Indexing page's "chemistry mask failed" line.
 *
 * The chemistry-mask catch put `err.response.data.detail` into the sentence.
 * On a file with a spectrum cube that detail is an object while the fit runs
 * (503) or after it failed (500), and the line read
 * "Mask failed: [object Object]".
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import i18n from '../../i18n';
import { chemMaskErrorText } from './chemMaskError';

const fitError = (status, detail) => Object.assign(new Error(`status ${status}`), {
  response: { status, data: { detail } },
});
const RUNNING = fitError(503, { code: 'eds_spectrum_fit_running', progress: 0.42,
                                message: 'The spectra are still being evaluated.' });
const FAILED = fitError(500, { code: 'eds_spectrum_fit_failed', error: 'cube unreadable',
                               message: 'The spectrum fit failed: cube unreadable.' });

describe('chemMaskErrorText', () => {
  const en = i18n.getFixedT('en', 'indexing');
  const de = i18n.getFixedT('de', 'indexing');

  it('the fit still running reads as a sentence, not [object Object]', () => {
    const text = chemMaskErrorText(RUNNING, en);
    expect(text).not.toContain('[object Object]');
    expect(text).toMatch(/^Mask failed: Evaluating the spectra of this map.* 42 %$/);
  });

  it('a failed fit says why, in the page\'s language', () => {
    const text = chemMaskErrorText(FAILED, de);
    expect(text).not.toContain('[object Object]');
    expect(text.startsWith('Maske fehlgeschlagen: ')).toBe(true);
    expect(text).toContain('cube unreadable');
  });

  it('a string detail and a network error read as before', () => {
    expect(chemMaskErrorText(fitError(400, 'No EDS data loaded'), en))
      .toBe('Mask failed: No EDS data loaded');
    expect(chemMaskErrorText(new Error('Network Error'), en))
      .toBe('Mask failed: Network Error');
  });

  it('the page uses it in the chemistry-mask catch', () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const src = readFileSync(resolve(here, 'IndexingPage.jsx'), 'utf8');
    const fn = src.slice(src.indexOf('async function applyChemMask'),
                         src.indexOf('const handleEdsLayersChange'));
    expect(fn).toMatch(/catch \(err\) \{\s*setChemPreview\(chemMaskErrorText\(err, t\)\);/);
    expect(fn).not.toMatch(/data\?\.detail/);
  });
});
