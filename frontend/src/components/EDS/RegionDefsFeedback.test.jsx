// @vitest-environment jsdom
/**
 * What four first-time readers caught, pinned so it cannot come back.
 *
 * Every case here was a real defect found by reading the screen cold, not by
 * reading the code: a preview that measured on different data than the run it
 * previewed, a legend that disagreed with the colours it explained, tooltips
 * that showed a different number than the row above them, and English prose
 * surfacing in a four-language app.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

const previewRegionDefs = vi.fn();
vi.mock('../../services/api', () => ({
  edsApi: { previewRegionDefs: (...a) => previewRegionDefs(...a) },
}));

import RegionDefs, { reasonText } from './RegionDefs';
import { ENRICHED_AT, DEPLETED_AT, enrichmentKind } from './RegionInspector';

const DEFS = [{ name: 'a', elements: [{ element: 'Si', min_at_pct: 20 }] }];

function setup(over = {}) {
  const props = {
    defs: DEFS, setDefs: vi.fn(), elements: ['Al', 'Fe', 'Si'], allPhases: [],
    clusterRemainder: true, setClusterRemainder: vi.fn(),
    elementWeights: {}, setElementWeights: vi.fn(),
    onReclassify: vi.fn(), busy: false, ...over,
  };
  render(<RegionDefs {...props} />);
  return props;
}

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

describe('the preview measures what the run will measure', () => {
  it('sends the smoothing along', async () => {
    // Without this the preview silently used the module default while the
    // run used the Scale slider, so the two disagreed by exactly the amount
    // the user had just changed — while the endpoint promised they could not.
    previewRegionDefs.mockResolvedValue({ data: { definitions: [], unclaimed_pixels: 0, unclaimed_percentage: 0 } });
    setup({ scale: 9 });
    fireEvent.click(screen.getByText('defs.preview'));
    await waitFor(() => expect(previewRegionDefs).toHaveBeenCalledWith(
      expect.objectContaining({ scale: 9 })));
  });

  it('re-previews when the smoothing changes', async () => {
    previewRegionDefs.mockResolvedValue({ data: { definitions: [], unclaimed_pixels: 0, unclaimed_percentage: 0 } });
    setup({ scale: 3 });
    fireEvent.click(screen.getByText('defs.preview'));
    await waitFor(() => expect(previewRegionDefs).toHaveBeenCalledWith(
      expect.objectContaining({ scale: 3 })));
    cleanup();
    setup({ scale: 7 });
    fireEvent.click(screen.getByText('defs.preview'));
    await waitFor(() => expect(previewRegionDefs).toHaveBeenLastCalledWith(
      expect.objectContaining({ scale: 7 })));
  });
});

describe('the preview says WHAT it caught, not only how much', () => {
  it('shows the composition of the claimed pixels', async () => {
    // 202 px of matrix and 202 px of particle read identically until you
    // see what is in them. The backend already sends this.
    previewRegionDefs.mockResolvedValue({ data: {
      definitions: [{ name: 'a', n_pixels: 202, percentage: 1.9,
                      overlap_pixels: 0, reason: null,
                      mean_at_pct: { Al: 58.1, Si: 40.3 } }],
      unclaimed_pixels: 0, unclaimed_percentage: 0 } });
    setup();
    fireEvent.click(screen.getByText('defs.preview'));
    await waitFor(() => expect(
      screen.getByText(/Al 58\.1 · Si 40\.3 at%/)).toBeTruthy());
  });
});

describe('reasonText', () => {
  it('translates a code it knows', () => {
    const t = (k) => (k === 'defs.reason.nomatch' ? 'kein Pixel passt' : k);
    expect(reasonText(t, { reason_code: 'nomatch', reason: 'no pixels match' }))
      .toBe('kein Pixel passt');
  });

  it('falls back to the backend prose for a code it does not know', () => {
    // A reason the frontend has not learned must still say something.
    const t = (k) => k;
    expect(reasonText(t, { reason_code: 'from-the-future', reason: 'because' }))
      .toBe('because');
  });

  it('keeps the clause reason as prose, because it names an element', () => {
    const t = (k) => k;
    expect(reasonText(t, { reason_code: 'rule', reason: 'Mg not measured here' }))
      .toBe('Mg not measured here');
  });

  it('says something even with no reason at all', () => {
    expect(reasonText((k) => k, {})).toBe('defs.previewNone');
  });
});

describe('the enrichment legend and the colours agree', () => {
  it('depletion is the reciprocal of enrichment, not a rounder number', () => {
    // The legend said 0.8 while the code used 0.77; between them the words
    // and the colour contradicted each other.
    expect(DEPLETED_AT).toBeCloseTo(1 / ENRICHED_AT, 2);
  });

  it('colours match the thresholds exactly at the boundary', () => {
    expect(enrichmentKind(ENRICHED_AT)).toBe('enriched');
    expect(enrichmentKind(DEPLETED_AT)).toBe('depleted');
    expect(enrichmentKind(1.0)).toBe('flat');
  });

  it('an unmeasurable factor is not silently called flat', () => {
    expect(enrichmentKind(null)).toBe('unknown');
    expect(enrichmentKind(Infinity)).toBe('unknown');
  });
});
