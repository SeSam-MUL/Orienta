// @vitest-environment jsdom
/**
 * The region inspector.
 *
 * It reports measurements, so the things worth pinning are the ones that would
 * quietly mislead: an enrichment bar that puts depletion on the enriched side,
 * a "loading" state that keeps the previous region on screen as if it were
 * the one just clicked, and a candidate list that hides how far off it is.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

import RegionInspector, {
  enrichmentBarPct, enrichmentKind,
} from './RegionInspector';

const DETAIL = {
  region_id: 5,
  n_pixels: 68,
  percentage: 0.63,
  color: '#88ddf8',
  phase_index: 1,
  n_pieces: 2,
  pieces: [52, 16],
  elements: [
    { element: 'Si', at_pct: 56.0, spread: 6.4, enrichment: 7.69 },
    { element: 'Al', at_pct: 42.6, spread: 6.4, enrichment: 0.50 },
    { element: 'Zn', at_pct: 0.2, spread: 0.3, enrichment: null },
  ],
  neighbours: [
    { region_id: 2, color: '#ccbbaa', shared_edge_px: 27, gap_at_pct: 18.0 },
  ],
  candidates: [
    { phase_index: 3, cif_filename: 'Mn2(AlSi)5.cif', formula: 'Mn2Al5Si5',
      gap_at_pct: 5.38 },
    { phase_index: 1, cif_filename: 'Si.cif', formula: 'Si', gap_at_pct: 7.84 },
  ],
};

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

describe('enrichmentBarPct', () => {
  it('puts 1x exactly in the middle, so depletion reads as clearly as enrichment', () => {
    expect(enrichmentBarPct(1)).toBe(50);
  });

  it('is symmetric: 2x and 0.5x sit the same distance either side', () => {
    expect(enrichmentBarPct(2) - 50).toBeCloseTo(50 - enrichmentBarPct(0.5), 6);
  });

  it('clamps rather than running off the bar', () => {
    expect(enrichmentBarPct(1e6)).toBe(100);
    expect(enrichmentBarPct(1e-6)).toBe(0);
  });

  it('treats a missing factor as neutral instead of throwing', () => {
    expect(enrichmentBarPct(null)).toBe(50);
    expect(enrichmentBarPct(0)).toBe(50);
    expect(enrichmentBarPct(NaN)).toBe(50);
  });
});

describe('enrichmentKind', () => {
  it('uses the same 1.3x the classifier gates on', () => {
    expect(enrichmentKind(1.3)).toBe('enriched');
    expect(enrichmentKind(1.29)).toBe('flat');
  });

  it('calls the reciprocal depleted', () => {
    expect(enrichmentKind(0.5)).toBe('depleted');
    expect(enrichmentKind(0.9)).toBe('flat');
  });

  it('does not guess when there is no background to compare against', () => {
    expect(enrichmentKind(null)).toBe('unknown');
  });
});

describe('the inspector', () => {
  it('shows size and how many pieces the region is in', () => {
    render(<RegionInspector detail={DETAIL} />);
    expect(screen.getByText(/inspector\.headline/)).toBeTruthy();
    expect(screen.getByText(/inspector\.pieces/)).toBeTruthy();
  });

  it('lists every element with its at%, spread and enrichment', () => {
    render(<RegionInspector detail={DETAIL} />);
    expect(screen.getByText('Si')).toBeTruthy();
    expect(screen.getByText('56.0')).toBeTruthy();
    expect(screen.getByText('7.69×')).toBeTruthy();
    expect(screen.getByText('0.50×')).toBeTruthy();
  });

  it('shows a dash where there is no background to compare against', () => {
    render(<RegionInspector detail={DETAIL} />);
    expect(screen.getByText('—')).toBeTruthy();
  });

  it('names each candidate with how far off it is, not just its rank', () => {
    render(<RegionInspector detail={DETAIL} />);
    expect(screen.getByText('5.4')).toBeTruthy();
    expect(screen.getByText('7.8')).toBeTruthy();
  });

  it('names the region when a candidate is clicked', () => {
    const onAssign = vi.fn();
    render(<RegionInspector detail={DETAIL} phaseIndex={1} onAssign={onAssign} />);
    fireEvent.click(screen.getByText('Mn2(AlSi)5.cif'));
    expect(onAssign).toHaveBeenCalledWith(3);
  });

  it('offers to take the name off only when it has one', () => {
    render(<RegionInspector detail={DETAIL} phaseIndex={1} />);
    expect(screen.getByText('inspector.clearName')).toBeTruthy();
    cleanup();
    render(<RegionInspector detail={DETAIL} phaseIndex={-1} />);
    expect(screen.queryByText('inspector.clearName')).toBeNull();
  });

  it('jumps to a neighbour when it is clicked — the merge decision path', () => {
    const onSelect = vi.fn();
    render(<RegionInspector detail={DETAIL} onSelectRegion={onSelect} />);
    fireEvent.click(screen.getByText(/inspector\.neighbourGap/));
    expect(onSelect).toHaveBeenCalledWith(2);
  });

  it('says a region touches nothing rather than showing an empty column', () => {
    render(<RegionInspector detail={{ ...DETAIL, neighbours: [] }} />);
    expect(screen.getByText('inspector.noNeighbours')).toBeTruthy();
  });

  it('shows the error instead of stale numbers', () => {
    render(<RegionInspector detail={DETAIL} error="no regions on this map" />);
    expect(screen.getByText('no regions on this map')).toBeTruthy();
    expect(screen.queryByText('Si')).toBeNull();
  });

  it('does not present a previous region as the one being loaded', () => {
    render(<RegionInspector detail={null} loading />);
    expect(screen.getByText('inspector.loading')).toBeTruthy();
  });

  it('keeps showing the current region while a refresh is in flight', () => {
    // Blanking on every refresh would make the panel flicker on each merge.
    render(<RegionInspector detail={DETAIL} loading />);
    expect(screen.getByText('Si')).toBeTruthy();
  });

  it('says plainly when nothing is selected', () => {
    render(<RegionInspector detail={null} />);
    expect(screen.getByText('inspector.empty')).toBeTruthy();
  });

  it('disables naming while an operation is in flight', () => {
    render(<RegionInspector detail={DETAIL} phaseIndex={1} busy />);
    // getByText lands on the inner span; the button is its parent.
    expect(screen.getByText('Mn2(AlSi)5.cif').closest('button').disabled)
      .toBe(true);
  });
});
