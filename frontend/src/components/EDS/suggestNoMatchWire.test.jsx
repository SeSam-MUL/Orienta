// @vitest-environment jsdom
/**
 * The whole line from the backend answer to the words on screen.
 *
 * The 2026-08-25 report was "75 at% Mg / 25 at% Si shows no Mg2Si although
 * the CIF is there". It was there, it scored 0.92, and the enrichment gate
 * rejected it — but the panel said "No phases suggested for this
 * composition." and nothing else, so there was nothing to act on. The
 * backend now sends `cif_no_match`; this file makes sure the user sees it.
 *
 * Written like `edsRequestWire.test.jsx`: the hook is driven with the REAL
 * response body and the text is read off the REAL rendered component, with
 * i18n initialised (vite.config.js setupFiles), so this fails if the field
 * is renamed on either side, if the code has no string, or if the numbers
 * are dropped on the way. Asserting against a mocked `t` would have proved
 * nothing about the locale files.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  edsApi: { suggestPhases: vi.fn() },
}));

import { edsApi } from '../../services/api';
import { useSuggestPhases } from './hooks/useSuggestPhases';
import SuggestNoMatch, { reasonText } from './SuggestNoMatch';
import en from '../../locales/en/eds.json';
import de from '../../locales/de/eds.json';
import ja from '../../locales/ja/eds.json';
import zh from '../../locales/zh/eds.json';

/** Verbatim from backend/api/services/cif_phase_library.explain_no_match. */
const REASON = {
  code: 'not_enriched_over_background',
  phase: 'Mg2Si_MP-mp-1367.cif',
  element: 'Si',
  measured_at_pct: 25.0,
  background_at_pct: 30.0,
  required_at_pct: 39.0,
  score_without_map: 0.917,
  message: 'Mg2Si_MP-mp-1367.cif fits this pixel (0.92) but was rejected: '
    + 'Si is 25.0 at% here against a map background of 30.0 at%.',
};

/** The panel, reduced to the two pieces this line runs through. */
function Panel() {
  const { suggestions, cifNoMatch, run } = useSuggestPhases();
  return (
    <div>
      <button type="button" onClick={() => run(0, 0)}>run</button>
      {suggestions && suggestions.length === 0 && (
        <SuggestNoMatch reason={cifNoMatch} />
      )}
    </div>
  );
}

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

describe('cif_no_match reaches the panel', () => {
  it('renders the reason for an empty suggestion list', async () => {
    edsApi.suggestPhases.mockResolvedValue({
      data: {
        suggestions: [],
        atomic_pct: { Mg: 75.0, Si: 25.0 },
        library_source: 'default',
        library_size: 28,
        cif_no_match: REASON,
      },
    });

    render(<Panel />);
    await act(async () => { screen.getByText('run').click(); });

    const box = await screen.findByTestId('suggest-no-match');
    // The facts the user needs: which phase, which element, and the two
    // numbers that decided it.
    expect(box.textContent).toContain('Mg2Si_MP-mp-1367.cif');
    expect(box.textContent).toContain('Si');
    expect(box.textContent).toContain('25');
    expect(box.textContent).toContain('39');
    // Translated, not the backend's English prose passed through.
    expect(box.textContent).not.toBe(REASON.message);
  });

  it('shows nothing when the backend sends no reason', async () => {
    edsApi.suggestPhases.mockResolvedValue({
      data: { suggestions: [], atomic_pct: {}, library_source: 'cif' },
    });

    render(<Panel />);
    await act(async () => { screen.getByText('run').click(); });

    expect(screen.queryByTestId('suggest-no-match')).toBeNull();
  });
});

describe('reasonText', () => {
  const t = (key, opts) => {
    const path = key.split('.').reduce((o, k) => (o || {})[k], en);
    return path.replace(/{{(\w+)}}/g, (_, name) => String(opts[name]));
  };

  it('has a string for every code the backend can send', () => {
    // Mirrors the `code` values in cif_phase_library.explain_no_match. A code
    // with no string falls through to the backend's English prose, which is
    // the fallback, not the plan.
    const codes = [
      'not_enriched_over_background', 'below_min_score',
      'no_candidate_phase', 'empty_library', 'no_chemistry',
    ];
    codes.forEach((code) => {
      const text = reasonText({ code, message: 'FALLBACK' }, t);
      expect(text, code).not.toBe('FALLBACK');
      expect(text, code).toBeTruthy();
    });
  });

  it('falls back to the backend prose for an unknown code', () => {
    expect(reasonText({ code: 'invented_later', message: 'FALLBACK' }, t))
      .toBe('FALLBACK');
  });

  it('names the threshold next to the score', () => {
    const text = reasonText(
      { code: 'below_min_score', phase: 'Si.cif', score: 0.21, min_score: 0.3 }, t);
    expect(text).toContain('0.21');
    expect(text).toContain('0.3');
  });

  it('is empty without a reason', () => {
    expect(reasonText(null, t)).toBe('');
  });
});

describe('the way out it offers is real', () => {
  /**
   * The first version of this string said "read the phase map instead". It
   * is not an escape: auto_classify_pixels and the cluster path compute the
   * SAME map-wide background and call the SAME scorer at the same
   * min_score. Measured on the fixture this message is written for (60x60,
   * mostly 70/30 Mg-Si, library Mg2Si + Si) the phase grid comes back
   * {-1: 3600} with every score at 0.050, the veto floor. Naming the region
   * by hand does work: assign_region_phase painted 3339 px.
   */
  const CLAIMS_THE_MAP_WOULD_DO_IT = [
    /read the phase map instead/i,
    /nutze die Phasenkarte/i,
  ];

  it('does not send the user to a tool that applies the same veto', () => {
    Object.values({ en, de, ja, zh }).forEach((loc) => {
      CLAIMS_THE_MAP_WOULD_DO_IT.forEach((pattern) => {
        expect(loc.suggest.noMatch.notEnriched).not.toMatch(pattern);
      });
    });
  });

  it('offers naming the region by hand, in every language', () => {
    expect(en.suggest.noMatch.notEnriched).toMatch(/by hand/i);
    expect(de.suggest.noMatch.notEnriched).toMatch(/von Hand/);
    expect(ja.suggest.noMatch.notEnriched).toMatch(/手動/);
    expect(zh.suggest.noMatch.notEnriched).toMatch(/手动/);
  });
});

describe('locales', () => {
  it('write German with real umlauts, like the rest of the file', () => {
    // ASCII German has been shipped by mistake before (CLAUDE.md,
    // 2026-08-26); the file itself writes Größe, für, ausschließlich.
    Object.values(de.suggest.noMatch).forEach((s) => {
      expect(s).not.toMatch(/(naechste|koennte|waere|fuer|groesse|ausschliesslich)/i);
    });
  });

  it('carry the same noMatch keys in all four languages', () => {
    const keys = (o) => Object.keys(o.suggest.noMatch).sort();
    expect(keys(de)).toEqual(keys(en));
    expect(keys(ja)).toEqual(keys(en));
    expect(keys(zh)).toEqual(keys(en));
  });

  it('interpolate the same placeholders in all four languages', () => {
    const slots = (s) => (s.match(/{{\w+}}/g) || []).sort();
    Object.keys(en.suggest.noMatch).forEach((k) => {
      [de, ja, zh].forEach((loc) => {
        expect(slots(loc.suggest.noMatch[k]), k)
          .toEqual(slots(en.suggest.noMatch[k]));
      });
    });
  });
});
