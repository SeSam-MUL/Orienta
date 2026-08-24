// @vitest-environment jsdom
/**
 * The structure panel: name a group, or fix the grouping.
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

import StructurePanel, { describeComposition, rankCandidates } from './StructurePanel';

const STRUCTURES = [
  { structure_id: 0, n_pixels: 3036, percentage: 28.1, color: '#aabbcc',
    mean_at_pct: { Al: 97.0, Si: 1.2 }, phase_index: 0,   // C/O are
    // excluded by the backend: they take no part in the grouping.
    cif_filename: 'Al.cif', formula: 'Al' },
  { structure_id: 1, n_pixels: 70, percentage: 0.7, color: '#ccbbaa',
    mean_at_pct: { Al: 60.3, Si: 38.0 }, phase_index: 0,
    cif_filename: 'Al.cif', formula: 'Al' },
  { structure_id: 2, n_pixels: 68, percentage: 0.6, color: '#bbccaa',
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
    phaseMap: { loaded: true, structures: STRUCTURES, all_phases: ALL_PHASES },
    structureBusy: false,
    selectedStructureId: null,
    setSelectedStructureId: vi.fn(),
    scale: 5, setScale: vi.fn(),
    nClusters: null, setNClusters: vi.fn(),
    handleAssignStructure: vi.fn(),
    handleMergeStructures: vi.fn(),
    handleSplitStructure: vi.fn(),
    handleGrowStructure: vi.fn(),
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

  it('survives a structure with no measured chemistry', () => {
    expect(describeComposition({})).toBe('');
    expect(describeComposition(undefined)).toBe('');
  });
});

describe('rankCandidates', () => {
  it('puts the closest nominal composition first', () => {
    const ranked = rankCandidates(
      { mean_at_pct: { Al: 50, Si: 50 } }, ALL_PHASES);
    expect(ranked[0].cif_filename).toBe('AlSi.cif');
  });

  it('ranks a nearly pure structure onto the pure phase', () => {
    const ranked = rankCandidates({ mean_at_pct: { Al: 97, Si: 1.2 } }, ALL_PHASES);
    expect(ranked[0].cif_filename).toBe('Al.cif');
  });

  it('offers phases the classifier placed nowhere', () => {
    // The whole point of naming by hand: AlSi.cif holds 0 px and must still
    // be offered, or the correction the user wants is impossible.
    const ranked = rankCandidates({ mean_at_pct: { Al: 50, Si: 50 } }, ALL_PHASES);
    expect(ranked.map((p) => p.cif_filename)).toContain('AlSi.cif');
  });

  it('keeps every candidate when the structure has no chemistry to rank by', () => {
    expect(rankCandidates({ mean_at_pct: {} }, ALL_PHASES)).toHaveLength(3);
  });

  it('does not mutate the list it was given', () => {
    const before = ALL_PHASES.map((p) => p.cif_filename);
    rankCandidates({ mean_at_pct: { Si: 90 } }, ALL_PHASES);
    expect(ALL_PHASES.map((p) => p.cif_filename)).toEqual(before);
  });
});

describe('the panel', () => {
  it('lists every structure with its composition, not its phase name', () => {
    render(<StructurePanel handle={makeHandle()} />);
    expect(screen.getByText('Al 97.0 · Si 1.2')).toBeTruthy();
    expect(screen.getByText('Al 60.3 · Si 38.0')).toBeTruthy();
    expect(screen.getByText('Si 56.0 · Al 42.6')).toBeTruthy();
  });

  it('selects a structure when its row is clicked', () => {
    const h = makeHandle();
    render(<StructurePanel handle={h} />);
    fireEvent.click(screen.getByText('Al 60.3 · Si 38.0'));
    expect(h.setSelectedStructureId).toHaveBeenCalledWith(1);
  });

  it('names the selected structure in one click', () => {
    const h = makeHandle({ selectedStructureId: 1 });
    render(<StructurePanel handle={h} />);
    fireEvent.click(screen.getByTitle('structures.nameTooltip:{"name":"AlSi.cif"}'));
    expect(h.handleAssignStructure).toHaveBeenCalledWith(1, 2);
  });

  it('can take a name off again', () => {
    const h = makeHandle({ selectedStructureId: 1 });
    render(<StructurePanel handle={h} />);
    fireEvent.click(screen.getByText('structures.clearName'));
    expect(h.handleAssignStructure).toHaveBeenCalledWith(1, -1);
  });

  it('merges the selected structure with another', () => {
    const h = makeHandle({ selectedStructureId: 1 });
    render(<StructurePanel handle={h} />);
    fireEvent.change(screen.getByLabelText('structures.merge'), {
      target: { value: '2' },
    });
    fireEvent.click(screen.getByText('structures.mergeGo'));
    expect(h.handleMergeStructures).toHaveBeenCalledWith(1, 2);
  });

  it('never offers a structure as a merge target with itself', () => {
    const h = makeHandle({ selectedStructureId: 1 });
    render(<StructurePanel handle={h} />);
    const opts = [...screen.getByLabelText('structures.merge').options]
      .map((o) => o.value).filter(Boolean);
    expect(opts).not.toContain('1');
    expect(opts).toEqual(['0', '2']);
  });

  it('splits the selected structure locally', () => {
    const h = makeHandle({ selectedStructureId: 0 });
    render(<StructurePanel handle={h} />);
    fireEvent.click(screen.getByText('structures.splitInto:{"count":3}'));
    expect(h.handleSplitStructure).toHaveBeenCalledWith(0, 3);
  });

  it('moves one boundary in and out', () => {
    const h = makeHandle({ selectedStructureId: 2 });
    render(<StructurePanel handle={h} />);
    fireEvent.click(screen.getByText('+1 px'));
    expect(h.handleGrowStructure).toHaveBeenCalledWith(2, 1);
    fireEvent.click(screen.getByText('−1 px'));
    expect(h.handleGrowStructure).toHaveBeenCalledWith(2, -1);
  });

  it('snaps every boundary onto the chemistry, map-wide', () => {
    const h = makeHandle();
    render(<StructurePanel handle={h} />);
    fireEvent.change(screen.getByLabelText('structures.snap'), {
      target: { value: '4' },
    });
    fireEvent.click(screen.getByText('structures.snapGo'));
    expect(h.handleSnapEdges).toHaveBeenCalledWith(4);
  });

  it('shows the per-structure tools only once one is selected', () => {
    render(<StructurePanel handle={makeHandle()} />);
    expect(screen.queryByText('structures.mergeGo')).toBeNull();
    cleanup();
    render(<StructurePanel handle={makeHandle({ selectedStructureId: 0 })} />);
    expect(screen.getByText('structures.mergeGo')).toBeTruthy();
  });

  it('says so plainly when the map has no structures', () => {
    render(<StructurePanel handle={makeHandle({
      phaseMap: { loaded: true, structures: [], all_phases: ALL_PHASES },
    })} />);
    expect(screen.getByText('structures.none')).toBeTruthy();
  });

  it('renders nothing at all without a map', () => {
    const { container } = render(
      <StructurePanel handle={makeHandle({ phaseMap: { loaded: false } })} />);
    expect(container.firstChild).toBeNull();
  });

  it('disables every action while one is in flight', () => {
    const h = makeHandle({ selectedStructureId: 0, structureBusy: true });
    render(<StructurePanel handle={h} />);
    expect(screen.getByText('structures.snapGo').disabled).toBe(true);
    expect(screen.getByText('+1 px').disabled).toBe(true);
  });
});
