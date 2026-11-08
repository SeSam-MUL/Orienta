// @vitest-environment jsdom
//
// What the experimental panel of the Pattern Match window says when it has no pattern. The answer of
// /api/indexing/pattern-match carries `experimental_error_code` when the result's own file was refused
// with a code (its metadata pairing is lost, or it opened as another variant); the panel words the code
// in the language of the page instead of the general "not available". The texts are the real ones.
import { describe, it, expect, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import i18n from '../../i18n';
import { experimentalMissingText } from './experimentalPanelText';

const t = (key, options) => i18n.t(key, options);
afterEach(async () => { await i18n.changeLanguage('en'); });

describe('experimentalMissingText', () => {
  it('words the code of the refusal', () => {
    expect(experimentalMissingText({ experimental_error_code: 'loadCameUpAsOther' }, t))
      .toBe('The file opened, but as another measurement or pattern source than the one this result was '
        + 'created on. The result was not activated, and the viewer now shows the other one.');
    expect(experimentalMissingText({ experimental_error_code: 'apexPairingLost' }, t))
      .toMatch(/metadata file this pattern file was paired with is not known any more/);
  });

  it('does so in German as well', async () => {
    await i18n.changeLanguage('de');
    expect(experimentalMissingText({ experimental_error_code: 'loadVariantUnknown' }, t))
      .toMatch(/Die Datei enthält die gewählte Messung oder Mustervariante nicht/);
  });

  it('keeps the general text without a code, and for a code this build has no text for', () => {
    const general = i18n.t('phasemap:matches.notAvailable');
    expect(general).not.toBe('phasemap:matches.notAvailable');
    expect(experimentalMissingText({ experimental_error_code: null }, t)).toBe(general);
    expect(experimentalMissingText({}, t)).toBe(general);
    expect(experimentalMissingText(null, t)).toBe(general);
    expect(experimentalMissingText({ experimental_error_code: 'someCodeFromTheFuture' }, t)).toBe(general);
  });

  it('is what the window shows where it has no experimental pattern', () => {
    const page = readFileSync(path.join(import.meta.dirname, 'PhaseMapPage.jsx'), 'utf8');
    expect(page).toContain("import { experimentalMissingText } from './experimentalPanelText';");
    expect(page).toContain('{experimentalMissingText(matchData, t)}</div>}');
    expect(page).not.toContain("{t('phasemap:matches.notAvailable')}</div>}");
  });
});
