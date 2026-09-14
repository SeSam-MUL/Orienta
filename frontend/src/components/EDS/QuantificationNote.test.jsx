// @vitest-environment jsdom
/**
 * The quantification provenance, on its way to a user.
 *
 * WHY THIS FILE EXISTS. `eds_utils` has attached a `.provenance` to every
 * wt%/at% result since 2026-09-12 -- the k-factor model, the calibration
 * standard, that there is NO ZAF matrix correction, every extrapolated line,
 * every known spectral overlap. The routes pulled the numbers out with
 * `float(...)` and dropped all of it. Built, correct, and never displayed:
 * the same shape as the classify warnings next door, and the same cost. The
 * numbers on screen look like an analysis and are a semi-quantitative
 * estimate, and nothing said so.
 *
 * WHAT THE NOTE HAS TO GET RIGHT, because it decides the shape:
 *
 *  - the no-ZAF caveat is PERMANENT -- it is true of every number this page
 *    will ever show. A banner would be noise by the second scan and get
 *    skipped, which is the same as not having it. So: one collapsed line;
 *  - it must raise its voice only when there is something SPECIFIC: an
 *    overlap, an extrapolated line, a beam voltage outside the calibration.
 *    A permanent amber is a permanent ignored amber;
 *  - no provenance -> render NOTHING, so the place the real one appears does
 *    not train the eye to skip it;
 *  - and the wire has to be real. The contract test at the bottom reads the
 *    key the backend writes, because a component that renders a field nobody
 *    sends is exactly the defect this file exists to prevent.
 */
import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, screen, fireEvent } from '@testing-library/react';

import en from '../../locales/en/eds.json';
import de from '../../locales/de/eds.json';
import ja from '../../locales/ja/eds.json';
import zh from '../../locales/zh/eds.json';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

import QuantificationNote from './QuantificationNote';

afterEach(cleanup);

/** What backend/api/routes/eds.py:_quant_provenance actually serialises. */
const CLEAN = {
  summary: 'k-factors: ... NO ZAF matrix correction -- treat as semi-quantitative.',
  method: 'standardless Cliff-Lorimer, C_i proportional to k_i * I_i',
  matrix_correction: null,
  beam_kv: 20.0,
  calibration: {
    standard: 'SampleB alpha-Al(Fe,Mn)Si interior vs Oxford Aztec Tru-Q',
    matrix: 'Al-rich intermetallic in an Al extrusion alloy',
    beam_kv: 20.0,
  },
  lines: { Al: { line: 'K', k: 0.7733 } },
  spectral_overlaps: [],
  warnings: [],
};

const WITH_OVERLAP = {
  ...CLEAN,
  spectral_overlaps: [{
    window: ['Zn', 'L'],
    interferer: ['Cu', 'L'],
    effect: 'Zn L reads high wherever Cu is present',
  }],
  warnings: ['Zn L window overlaps Cu L emission: Zn L reads high wherever Cu is present'],
};

describe('QuantificationNote', () => {
  it('renders nothing at all without provenance', () => {
    const { container } = render(<QuantificationNote provenance={null} />);
    expect(container.textContent).toBe('');
  });

  it('always offers the caveat, even when there is nothing specific to say', () => {
    render(<QuantificationNote provenance={CLEAN} />);
    // headline present, and it is the zero-caveat wording
    const button = screen.getByRole('button');
    expect(button.textContent).toContain('quantNote.headline');
    expect(button.textContent).toContain('"count":0');
    // the standing caveat is behind one click, not shouted
    expect(button.getAttribute('aria-expanded')).toBe('false');
    fireEvent.click(button);
    expect(screen.getByText(/quantNote\.noZaf/)).toBeTruthy();
    expect(screen.getByText(/quantNote\.calibratedOn/).textContent)
      .toContain('Al-rich intermetallic');
  });

  it('counts the specific caveats and names the overlapping windows', () => {
    render(<QuantificationNote provenance={WITH_OVERLAP} />);
    const button = screen.getByRole('button');
    // one overlap + one warning about it
    expect(button.textContent).toContain('"count":2');
    fireEvent.click(button);
    const overlap = screen.getByText(/quantNote\.overlap/);
    expect(overlap.textContent).toContain('Zn L');
    expect(overlap.textContent).toContain('Cu L');
  });

  it('is quiet when clean and loud only when it has something to say', () => {
    // read both BEFORE unmounting — cleanup() empties the container
    const { container: quiet } = render(<QuantificationNote provenance={CLEAN} />);
    const quietColour = quiet.querySelector('button').style.color;
    const quietText = quiet.textContent;
    cleanup();
    const { container: loud } = render(<QuantificationNote provenance={WITH_OVERLAP} />);
    const loudColour = loud.querySelector('button').style.color;
    const loudText = loud.textContent;
    expect(loudColour).not.toBe(quietColour);
    expect(loudText.startsWith('⚠')).toBe(true);   // warning sign
    expect(quietText.startsWith('ⓘ')).toBe(true);  // info sign
  });
});

describe('an assumed beam voltage', () => {
  const ASSUMED = {
    ...CLEAN,
    beam_kv_assumed: true,
    warnings: ['beam voltage not recorded in the file header — 20 kV '
      + 'assumed; the overvoltage of every k-factor depends on it'],
  };

  it('is counted, coloured and named in the reader language', () => {
    render(<QuantificationNote provenance={ASSUMED} />);
    const button = screen.getByRole('button');
    // one specific caveat, even though the raw warning is filtered out below
    expect(button.textContent).toContain('"count":1');
    expect(button.textContent.startsWith('⚠')).toBe(true);
    fireEvent.click(button);
    expect(screen.getByText(/quantNote\.beamAssumed/).textContent)
      .toContain('20');
  });

  it('does not also print the raw English sentence', () => {
    render(<QuantificationNote provenance={ASSUMED} />);
    fireEvent.click(screen.getByRole('button'));
    // the translated line replaces it rather than doubling it
    expect(screen.queryByText(/overvoltage of every k-factor/)).toBeNull();
  });

  it('says nothing when the voltage was actually read', () => {
    render(<QuantificationNote provenance={{ ...CLEAN, beam_kv_assumed: false }} />);
    fireEvent.click(screen.getByRole('button'));
    expect(screen.queryByText(/quantNote\.beamAssumed/)).toBeNull();
  });
});

describe('the wire it depends on', () => {
  it('every language has the keys the component asks for', () => {
    const need = ['headline_zero', 'headline_one', 'headline_other',
      'noZaf', 'calibratedOn', 'overlap', 'beamAssumed'];
    for (const [name, bundle] of Object.entries({ en, de, ja, zh })) {
      expect(bundle.quantNote, `${name} has no quantNote block`).toBeTruthy();
      for (const key of need) {
        expect(bundle.quantNote[key], `${name}.quantNote.${key}`).toBeTruthy();
      }
    }
  });

  it('the interpolations the component passes are the ones the strings use', () => {
    // calibratedOn gets {matrix, kv}; overlap gets {window, interferer};
    // headline_* get {count}. A string that interpolates something the
    // component never passes renders "{{...}}" at the user.
    const expected = {
      calibratedOn: ['matrix', 'kv'],
      beamAssumed: ['kv'],
      overlap: ['window', 'interferer'],
      headline_one: [],
      headline_zero: [],
    };
    for (const [name, bundle] of Object.entries({ en, de, ja, zh })) {
      for (const [key, allowed] of Object.entries(expected)) {
        const used = [...bundle.quantNote[key].matchAll(/\{\{(\w+)\}\}/g)]
          .map((m) => m[1]);
        for (const token of used) {
          expect(allowed, `${name}.quantNote.${key} interpolates {{${token}}}`)
            .toContain(token);
        }
      }
      expect([...bundle.quantNote.headline_other.matchAll(/\{\{(\w+)\}\}/g)]
        .map((m) => m[1])).toEqual(['count']);
    }
  });

  it('the component reads the key the backend writes', () => {
    // _quant_provenance() in backend/api/routes/eds.py serialises exactly
    // these; the shape above is a copy of it. If the backend renames one,
    // this is the test that should fail rather than the note going blank.
    for (const key of ['summary', 'matrix_correction', 'calibration',
      'spectral_overlaps', 'warnings']) {
      expect(Object.prototype.hasOwnProperty.call(CLEAN, key)).toBe(true);
    }
    render(<QuantificationNote provenance={CLEAN} />);
    fireEvent.click(screen.getByRole('button'));
    // it actually used calibration.matrix, not a placeholder
    expect(screen.getByText(/quantNote\.calibratedOn/).textContent)
      .toContain(CLEAN.calibration.matrix);
  });
});
