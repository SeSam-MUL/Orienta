// @vitest-environment jsdom
/**
 * A window the backend could not price must read as "no value", not as zero.
 *
 * Since 2026-09-12 `/api/eds/quantify/pixel` answers `wt_pct: null` /
 * `at_pct: null` for an element whose window has no usable k-factor, and puts
 * the reason in `quantification.excluded_windows` / `.warnings`. The page then
 * coerced both back with `?? 0` — so the table showed a confident 0.00 %,
 * which is a measurement nobody made, and the reason was nowhere: the
 * QuantificationNote was mounted for the region panel only.
 *
 * Driven through the REAL page pieces (the row mapping the page uses and the
 * real QuantTable / QuantificationNote) so a renamed field or a re-introduced
 * coercion fails here.
 */
import React from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react';

import QuantTable, { quantRowsFromResponse } from './quantTable';
import QuantificationNote from './QuantificationNote';

/** Verbatim shape of a /quantify/pixel answer with one dropped window. */
const RESPONSE = {
  row: 1,
  col: 2,
  display_mode: 'at_pct',
  data: {
    Al: { counts: 1000.0, wt_pct: 100.0, at_pct: 100.0 },
    Xx: { counts: 50.0, wt_pct: null, at_pct: null },
  },
  quantification: {
    summary: 'k-factors: ...',
    method: 'standardless Cliff-Lorimer',
    matrix_correction: null,
    beam_kv: 20.0,
    calibration: { standard: 'SampleB', matrix: 'Al', beam_kv: 20.0 },
    lines: { Al: { line: 'K', k: 1.0 } },
    spectral_overlaps: [],
    excluded_windows: [
      { element: 'Xx', line: 'K', reason: "no K line data for element 'Xx'" },
    ],
    warnings: [
      "Xx K was left out of the quantification: no K line data for element "
      + "'Xx'. The remaining elements are renormalised to 100 % without it.",
    ],
  },
};

beforeEach(() => { cleanup(); });

describe('an excluded window in the pixel table', () => {
  it('is not turned back into a measured zero', () => {
    const rows = quantRowsFromResponse(RESPONSE);
    const xx = rows.find((r) => r.element === 'Xx');
    expect(xx.counts).toBe(50.0);
    expect(xx.wt_pct).toBeNull();
    expect(xx.at_pct).toBeNull();
  });

  it('shows the empty marker rather than 0.00', () => {
    render(<QuantTable data={quantRowsFromResponse(RESPONSE)} />);
    const cells = Array.from(document.querySelectorAll('tbody tr'))
      .find((tr) => tr.textContent.includes('Xx'))
      .querySelectorAll('td');
    // element | counts | wt% | at%
    expect(cells[2].textContent).not.toMatch(/0\.00/);
    expect(cells[3].textContent).not.toMatch(/0\.00/);
    expect(cells[2].textContent.trim()).toBeTruthy();
  });

  it('still sums the elements that WERE quantified', () => {
    render(<QuantTable data={quantRowsFromResponse(RESPONSE)} />);
    expect(document.querySelector('tfoot').textContent).toContain('100.0');
  });

  it('names the dropped window in the note beside the table', async () => {
    render(<QuantificationNote provenance={RESPONSE.quantification} />);
    const toggle = screen.getByRole('button');
    await act(async () => { fireEvent.click(toggle); });
    expect(document.body.textContent).toContain('Xx');
  });
});
