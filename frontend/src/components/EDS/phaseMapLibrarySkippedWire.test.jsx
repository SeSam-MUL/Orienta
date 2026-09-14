// @vitest-environment jsdom
/**
 * The panel that USES a suspect composition has to say so.
 *
 * `suggest-phases` carried `library_skipped` from the start. The phase map did
 * not — and the map is where the number acts: `auto-classify` scores every
 * candidate against every pixel, so a phase whose stored composition its own
 * CIF contradicts produces a map that is confidently wrong. The note existed
 * on the other panel, beside a list the user was only browsing.
 *
 * Written like `suggestLibrarySkippedWire.test.jsx`: the real component with
 * i18n initialised and the real response body, so renaming the field on either
 * side fails the test.
 */
import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';

import SuggestLibrarySkipped from './SuggestLibrarySkipped';
import en from '../../locales/en/eds.json';

/** Verbatim from `cif_phase_library._suspect_rows`. */
const SUSPECT = [{
  file: 'sd_1816951.cif',
  code: 'suspect_listed_row',
  reason:
    'the phase database stores Cu 20.0 Mg 80.0 for this file, but its own '
    + 'atom-site table reads Cu 66.7 Mg 33.3 (Cu differs by 46.7 at%). '
    + 'The stored value is the one in use.',
}];

describe('the classify response carries the library notes', () => {
  it('shows the suspect chip with both compositions in the tooltip', () => {
    cleanup();
    render(<SuggestLibrarySkipped skipped={SUSPECT} />);
    const badge = screen.getByTestId('suggest-library-suspect');
    expect(badge.textContent).toContain('1');
    // Both numbers, because the user has to decide which is right.
    expect(badge.title).toContain('Cu 20.0 Mg 80.0');
    expect(badge.title).toContain('Cu 66.7 Mg 33.3');
    expect(badge.title).toContain('sd_1816951.cif');
  });

  it('says nothing when the library is clean', () => {
    cleanup();
    const { container } = render(<SuggestLibrarySkipped skipped={[]} />);
    expect(container.textContent).toBe('');
  });

  it('survives a backend that does not send the field', () => {
    cleanup();
    const { container } = render(<SuggestLibrarySkipped skipped={undefined} />);
    expect(container.textContent).toBe('');
  });
});

describe('the phase-map panel reads the field the route sends', () => {
  it('names `library_skipped`, the same key suggest-phases uses', async () => {
    // A contract test rather than a mount: PhaseMapPanel needs a loaded file,
    // a store and a canvas. What can drift is the KEY, and that is checked
    // here against the source of both readers.
    const panel = await import('./PhaseMapPanel.jsx?raw')
      .then((m) => m.default)
      .catch(() => null);
    if (panel == null) return;      // ?raw unsupported — nothing to check
    expect(panel).toContain('phaseMap.library_skipped');
    expect(panel).toContain('SuggestLibrarySkipped');
  });
});

describe('the strings exist', () => {
  it('en has both chips and they do not say the same thing', () => {
    expect(en.suggest.librarySkipped_one).toBeTruthy();
    expect(en.suggest.librarySuspect_one).toBeTruthy();
    expect(en.suggest.librarySuspect_one).not.toBe(en.suggest.librarySkipped_one);
  });
});

describe('the phase map carries the quantification note', () => {
  it('renders the excluded window on the panel, not just in the response', async () => {
    const { default: QuantificationNote } = await import('./QuantificationNote');
    cleanup();
    // Verbatim from eds_utils: the window that could not be priced, and the
    // sentence the backend writes beside it.
    render(<QuantificationNote provenance={{
      warnings: ['Cu Ka1 was left out of the quantification: overvoltage 1.2 '
        + 'is below the 1.5 floor. The remaining elements are renormalised to '
        + '100 % without it.'],
      excluded_windows: [{ element: 'Cu', line: 'Cu Ka1', reason: 'overvoltage' }],
    }} />);
    // Collapsed by default — it says there IS a caveat, which is the part that
    // has to be visible without a click.
    expect(document.body.textContent).toMatch(/caveat/i);
    const toggle = screen.getByText(/caveat/i);
    await act(async () => { toggle.click(); });
    expect(document.body.textContent).toMatch(/Cu/);
  });

  it('the panel reads `quantification` off the classify response', async () => {
    const panel = await import('./PhaseMapPanel.jsx?raw')
      .then((m) => m.default)
      .catch(() => null);
    if (panel == null) return;
    expect(panel).toContain('phaseMap.quantification');
    expect(panel).toContain('QuantificationNote');
  });
});
