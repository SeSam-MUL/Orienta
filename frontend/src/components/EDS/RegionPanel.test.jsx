// @vitest-environment jsdom
/**
 * The region panel: name a group, or fix the grouping.
 *
 * What matters here is that the panel never decides anything itself. It shows
 * what the backend grouped, ranks the candidate names so the plausible ones
 * are near the top, and sends one command per user action. A ranking bug would
 * quietly put the wrong phase under the user's cursor, so it is pinned.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

import RegionPanel, { describeComposition } from './RegionPanel';

const STRUCTURES = [
  { region_id: 0, n_pixels: 3036, percentage: 28.1, color: '#aabbcc',
    mean_at_pct: { Al: 97.0, Si: 1.2 }, phase_index: 0,   // C/O are
    // excluded by the backend: they take no part in the grouping.
    cif_filename: 'Al.cif', formula: 'Al' },
  { region_id: 1, n_pixels: 70, percentage: 0.7, color: '#ccbbaa',
    mean_at_pct: { Al: 60.3, Si: 38.0 }, phase_index: 0,
    cif_filename: 'Al.cif', formula: 'Al' },
  { region_id: 2, n_pixels: 68, percentage: 0.6, color: '#bbccaa',
    mean_at_pct: { Si: 56.0, Al: 42.6 }, phase_index: 1,
    cif_filename: 'Si.cif', formula: 'Si' },
];

const ALL_PHASES = [
  { phase_index: 0, cif_filename: 'Al.cif', formula: 'Al',
    composition: { Al: 100 }, n_pixels: 3106 },
  { phase_index: 1, cif_filename: 'Si.cif', formula: 'Si',
    composition: { Si: 100 }, n_pixels: 68 },
  { phase_index: 2, cif_filename: 'AlSi.cif', formula: 'AlSi',
    composition: { Al: 50, Si: 50 }, n_pixels: 0 },
];

function makeHandle(over = {}) {
  return {
    phaseMap: { loaded: true, regions: STRUCTURES, all_phases: ALL_PHASES },
    regionBusy: false,
    selectedRegionId: null,
    setSelectedRegionId: vi.fn(),
    scale: 5, setScale: vi.fn(),
    nClusters: null, setNClusters: vi.fn(),
    handleMergeRegions: vi.fn(),
    handleSplitRegion: vi.fn(),
    handleGrowRegion: vi.fn(),
    handleSnapEdges: vi.fn(),
    ...over,
  };
}

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

describe('describeComposition', () => {
  it('leads with the strongest elements', () => {
    expect(describeComposition({ Si: 9.5, Al: 81.1, Fe: 4.2 }))
      .toBe('Al 81.1 · Si 9.5 · Fe 4.2');
  });

  it('drops trace elements that only add noise to the row', () => {
    expect(describeComposition({ Al: 97, Zn: 0.02 })).toBe('Al 97.0');
  });

  it('survives a region with no measured chemistry', () => {
    expect(describeComposition({})).toBe('');
    expect(describeComposition(undefined)).toBe('');
  });
});

// Candidate ranking moved to the backend (`_region_detail`) when the
// inspector started showing the at% gap: two rankings could disagree about
// which phase sits under the user's cursor. The inspector's tests cover it.

describe('the panel', () => {
  it('lists every region with its composition, not its phase name', () => {
    render(<RegionPanel handle={makeHandle()} />);
    expect(screen.getByText('Al 97.0 · Si 1.2')).toBeTruthy();
    expect(screen.getByText('Al 60.3 · Si 38.0')).toBeTruthy();
    expect(screen.getByText('Si 56.0 · Al 42.6')).toBeTruthy();
  });

  it('selects a region when its row is clicked', () => {
    const h = makeHandle();
    render(<RegionPanel handle={h} />);
    fireEvent.click(screen.getByText('Al 60.3 · Si 38.0'));
    expect(h.setSelectedRegionId).toHaveBeenCalledWith(1);
  });

      it('merges the selected region with another', () => {
    const h = makeHandle({ selectedRegionId: 1 });
    render(<RegionPanel handle={h} />);
    fireEvent.change(screen.getByLabelText('regions.merge'), {
      target: { value: '2' },
    });
    fireEvent.click(screen.getByText('regions.mergeGo'));
    expect(h.handleMergeRegions).toHaveBeenCalledWith(1, 2);
  });

  it('never offers a region as a merge target with itself', () => {
    const h = makeHandle({ selectedRegionId: 1 });
    render(<RegionPanel handle={h} />);
    const opts = [...screen.getByLabelText('regions.merge').options]
      .map((o) => o.value).filter(Boolean);
    expect(opts).not.toContain('1');
    expect(opts).toEqual(['0', '2']);
  });

  it('splits the selected region locally', () => {
    const h = makeHandle({ selectedRegionId: 0 });
    render(<RegionPanel handle={h} />);
    fireEvent.click(screen.getByText('regions.splitInto:{"count":3}'));
    expect(h.handleSplitRegion).toHaveBeenCalledWith(0, 3);
  });

  it('moves one boundary in and out', () => {
    const h = makeHandle({ selectedRegionId: 2 });
    render(<RegionPanel handle={h} />);
    fireEvent.click(screen.getByText('+1 px'));
    expect(h.handleGrowRegion).toHaveBeenCalledWith(2, 1);
    fireEvent.click(screen.getByText('−1 px'));
    expect(h.handleGrowRegion).toHaveBeenCalledWith(2, -1);
  });

  it('snaps every boundary onto the chemistry, map-wide', () => {
    const h = makeHandle();
    render(<RegionPanel handle={h} />);
    fireEvent.change(screen.getByLabelText('regions.snap'), {
      target: { value: '4' },
    });
    fireEvent.click(screen.getByText('regions.snapGo'));
    expect(h.handleSnapEdges).toHaveBeenCalledWith(4);
  });

  it('shows the per-region tools only once one is selected', () => {
    render(<RegionPanel handle={makeHandle()} />);
    expect(screen.queryByText('regions.mergeGo')).toBeNull();
    cleanup();
    render(<RegionPanel handle={makeHandle({ selectedRegionId: 0 })} />);
    expect(screen.getByText('regions.mergeGo')).toBeTruthy();
  });

  it('says so plainly when the map has no regions', () => {
    render(<RegionPanel handle={makeHandle({
      phaseMap: { loaded: true, regions: [], all_phases: ALL_PHASES },
    })} />);
    expect(screen.getByText('regions.none')).toBeTruthy();
  });

  it('renders nothing at all without a map', () => {
    const { container } = render(
      <RegionPanel handle={makeHandle({ phaseMap: { loaded: false } })} />);
    expect(container.firstChild).toBeNull();
  });

  it('disables every action while one is in flight', () => {
    const h = makeHandle({ selectedRegionId: 0, regionBusy: true });
    render(<RegionPanel handle={h} />);
    expect(screen.getByText('regions.snapGo').disabled).toBe(true);
    expect(screen.getByText('+1 px').disabled).toBe(true);
  });
});
