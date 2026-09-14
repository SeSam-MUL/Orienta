// @vitest-environment jsdom
/**
 * A CIF the library could not read must reach the panel, not only the log.
 *
 * `_entry_from_cif` returns None for anything unreadable, which is right —
 * one broken download must not cost the rest of the library. But the file is
 * ON DISK, in the folder the user put it in, and nothing on screen said it
 * had been skipped: the phase simply was not offered, and even the "why is
 * there no match" explanation could not name it, because it reasons about
 * library entries and this file never became one.
 *
 * Written like `suggestNoMatchWire.test.jsx`: the real hook is driven with
 * the real response body and the text is read off the real component with
 * i18n initialised, so renaming the field on either side fails the test.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  edsApi: { suggestPhases: vi.fn() },
}));

import { edsApi } from '../../services/api';
import { useSuggestPhases } from './hooks/useSuggestPhases';
import SuggestLibrarySkipped from './SuggestLibrarySkipped';
import en from '../../locales/en/eds.json';
import de from '../../locales/de/eds.json';
import ja from '../../locales/ja/eds.json';
import zh from '../../locales/zh/eds.json';

/** Verbatim from cif_phase_library.load_cif_phase_library_with_skips. */
const SKIPPED = [
  { file: 'broken.cif', reason: 'unreadable: Invalid CIF file with no structures!' },
  { file: 'empty.cif', reason: "parsed to an empty composition ('') — no element to match chemistry against" },
];

function Panel() {
  const { librarySkipped, run } = useSuggestPhases();
  return (
    <div>
      <button type="button" onClick={() => run(0, 0)}>run</button>
      <SuggestLibrarySkipped skipped={librarySkipped} />
    </div>
  );
}

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

describe('library_skipped reaches the panel', () => {
  it('names how many files were skipped, with the reasons in the tooltip', async () => {
    edsApi.suggestPhases.mockResolvedValue({
      data: {
        suggestions: [{ name: 'Al', score: 0.8 }],
        atomic_pct: { Al: 100 },
        library_source: 'cif',
        library_size: 23,
        library_skipped: SKIPPED,
      },
    });

    render(<Panel />);
    // Nothing before the request — the badge must not appear on an empty state.
    expect(screen.queryByTestId('suggest-library-skipped')).toBeNull();

    await act(async () => { screen.getByText('run').click(); });

    const badge = await screen.findByTestId('suggest-library-skipped');
    expect(badge.textContent).toContain('2');
    // The reason is a backend sentence (a parser message) and cannot be
    // translated, so it lives in the tooltip rather than in the locale files.
    expect(badge.title).toContain('broken.cif');
    expect(badge.title).toContain('Invalid CIF file');
    expect(badge.title).toContain('empty.cif');
  });

  it('stays invisible when every CIF read', async () => {
    edsApi.suggestPhases.mockResolvedValue({
      data: {
        suggestions: [], atomic_pct: {}, library_source: 'cif',
        library_size: 23, library_skipped: [],
      },
    });
    render(<Panel />);
    await act(async () => { screen.getByText('run').click(); });
    expect(screen.queryByTestId('suggest-library-skipped')).toBeNull();
  });

  it('survives a backend that does not send the field yet', async () => {
    edsApi.suggestPhases.mockResolvedValue({
      data: { suggestions: [], atomic_pct: {}, library_source: 'default' },
    });
    render(<Panel />);
    await act(async () => { screen.getByText('run').click(); });
    expect(screen.queryByTestId('suggest-library-skipped')).toBeNull();
  });
});

/**
 * Verbatim from `cif_phase_library._suspect_rows`: the file IS offered as a
 * phase — the database spreadsheet supplies it — but from a row the database
 * builder wrote using the same parse the CIF reader now refuses.
 */
const SUSPECT = [{
  file: 'sd_1816951.cif',
  code: 'suspect_listed_row',
  reason:
    'sd_1816951.cif parses to 2 different compositions (Mg2Cu, Mg4Cu); '
    + 'refusing it rather than letting data-block order decide. Check the file. '
    + 'The phase is still offered, from the database spreadsheet, with a '
    + 'composition that came from the same parse -- do not trust it.',
}];

describe('a suspect spreadsheet row is not reported as "not read"', () => {
  it('gets its own line, because the phase IS offered', async () => {
    edsApi.suggestPhases.mockResolvedValue({
      data: {
        suggestions: [{ name: 'Mg4Cu', score: 0.9 }],
        atomic_pct: { Mg: 80, Cu: 20 },
        library_source: 'cif',
        library_size: 36,
        library_skipped: SUSPECT,
      },
    });
    render(<Panel />);
    await act(async () => { screen.getByText('run').click(); });

    const badge = await screen.findByTestId('suggest-library-suspect');
    expect(badge.textContent).toContain('1');
    expect(badge.title).toContain('sd_1816951.cif');
    expect(badge.title).toContain('Mg2Cu');
    // "N CIFs not read" would send the user looking for a missing phase that
    // is sitting in the list in front of them.
    expect(screen.queryByTestId('suggest-library-skipped')).toBeNull();
  });

  it('counts the two kinds separately when both occur', async () => {
    edsApi.suggestPhases.mockResolvedValue({
      data: {
        suggestions: [], atomic_pct: {}, library_source: 'cif',
        library_size: 36, library_skipped: [...SKIPPED, ...SUSPECT],
      },
    });
    render(<Panel />);
    await act(async () => { screen.getByText('run').click(); });

    expect((await screen.findByTestId('suggest-library-skipped')).textContent)
      .toContain('2');
    expect((await screen.findByTestId('suggest-library-suspect')).textContent)
      .toContain('1');
  });
});

describe('the string exists in every language', () => {
  it.each([['en', en], ['de', de], ['ja', ja], ['zh', zh]])(
    '%s has both plural forms and interpolates the count', (lang, bundle) => {
      for (const key of ['librarySkipped', 'librarySuspect']) {
        const one = bundle.suggest[`${key}_one`];
        const other = bundle.suggest[`${key}_other`];
        expect(one, `${lang} ${key}_one`).toBeTruthy();
        expect(other, `${lang} ${key}_other`).toBeTruthy();
        expect(other, `${lang} ${key} plural must carry the number`)
          .toContain('{{count}}');
      }
      expect(bundle.suggest.librarySuspect_one,
        `${lang} must not reuse the "not read" wording — the phase IS offered`)
        .not.toBe(bundle.suggest.librarySkipped_one);
    });
});
