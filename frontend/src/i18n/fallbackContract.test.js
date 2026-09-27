// @vitest-environment jsdom
/**
 * The fallbacks, against the REAL i18n instance.
 *
 * Three helpers on this branch translate a backend code and fall back to the
 * backend's English prose. Each was first written with `defaultValue: ''` and
 * each had a green unit test saying the fallback worked — because all three
 * tests mocked `t` as `(key, opts) => opts.defaultValue ?? key`, which is not
 * what this project's i18next does: `returnEmptyString: false` makes an empty
 * defaultValue not a value, so the KEY came back and the prose was lost.
 *
 * The mocked tests still have their place; this one exists because a mock
 * cannot catch a wrong assumption about the thing it mocks. It uses the real
 * instance and a key that certainly does not exist.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import i18n from './index';
import { detailText } from '../components/Indexing/EdsPreflightPanel';
import { warnText } from '../components/EDS/ExportDialog';
import { pcWarningText } from '../components/PCRefinement/PCRefinement';

beforeAll(async () => { await i18n.changeLanguage('de'); });

const t = (key, opts) => i18n.t(key, opts);

describe('a code the UI has no text for', () => {
  it('EDS pre-flight falls back to the backend prose', () => {
    expect(detailText(
      { code: 'aCodeNoBuildHas', detail: 'the loader said why' }, t,
    )).toBe('the loader said why');
  });

  it('edsUnavailable, which has no key on purpose, shows the loader message', () => {
    expect(detailText(
      { code: 'edsUnavailable', detail: 'No EDS data in this file' }, t,
    )).toBe('No EDS data in this file');
  });

  it('an export warning falls back to its English sentence', () => {
    // 13 of the 20 export codes have no text yet; they must read as English
    // sentences, not as "export.warn.map_png_failed".
    expect(warnText(
      { code: 'map_png_failed', message: 'phase_map.png could not be rendered (disk full).' },
      (k, o) => i18n.t(`eds:${k}`, o),
    )).toBe('phase_map.png could not be rendered (disk full).');
  });

  it('the PC warning falls back to its English sentence', () => {
    expect(pcWarningText(
      { pc_warning: 'Refined PC moved 0.082 …', pc_warning_codes: [{ code: 'fromTheFuture' }] },
      (k, o) => i18n.t(`pcrefinement:${k}`, o),
    )).toBe('Refined PC moved 0.082 …');
  });
});

describe('a code the UI does know', () => {
  it('is translated, not passed through', () => {
    const out = detailText(
      { code: 'coverageIncomplete', detail: 'some phases cannot be judged chemically' },
      (k, o) => i18n.t(`indexing:${k}`, o),
    );
    expect(out).toBe('Einige Phasen sind chemisch nicht beurteilbar');
  });

  it('fills the parameters the backend sent', () => {
    const out = detailText(
      { code: 'notMeasured', params: { elements: 'Fe, Mn' }, detail: 'not measured: Fe, Mn' },
      (k, o) => i18n.t(`indexing:${k}`, o),
    );
    expect(out).toContain('Fe, Mn');
    expect(out).not.toContain('not measured');
  });
});
