// @vitest-environment jsdom
/**
 * Declaring regions by hand.
 *
 * The two behaviours worth pinning are the ones that can destroy a map
 * silently: an empty definition must claim nothing (it is a half-written
 * edit, not "the whole map"), and a `<select>` value must be read before the
 * state updater runs — the matrix dropdown in PhaseRules shipped broken for
 * exactly that reason and nothing on screen said so.
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

import RegionDefs, { isEmptyDef, seedDefFromRegion } from './RegionDefs';

const ELEMENTS = ['Al', 'Fe', 'Mn', 'Si'];

function setup(over = {}) {
  const setDefs = vi.fn();
  const setElementWeights = vi.fn();
  const setClusterRemainder = vi.fn();
  const props = {
    defs: [], setDefs, elements: ELEMENTS, allPhases: [],
    clusterRemainder: true, setClusterRemainder,
    elementWeights: {}, setElementWeights,
    onReclassify: vi.fn(), busy: false, ...over,
  };
  render(<RegionDefs {...props} />);
  return { ...props };
}

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

describe('isEmptyDef', () => {
  it('calls a definition with no clause empty', () => {
    expect(isEmptyDef({ name: 'x', elements: [], ratios: [] })).toBe(true);
    expect(isEmptyDef(undefined)).toBe(true);
  });

  it('one clause that BITES is enough to be non-empty', () => {
    expect(isEmptyDef({ elements: [{ element: 'Si', min_at_pct: 20 }] }))
      .toBe(false);
    expect(isEmptyDef({ ratios: [{ numerator: 'Si', denominator: 'Al',
                                   max_ratio: 3 }] })).toBe(false);
    expect(isEmptyDef({ enrichment: [{ element: 'Si', min_factor: 2 }] }))
      .toBe(false);
  });

  it('a clause with no bound set does not count', () => {
    // This is exactly what the "Add" button creates before you type a
    // number. Counting it made the definition look non-empty while
    // constraining nothing, so it matched every pixel and one definition
    // ate the whole map - with the warning chip suppressed.
    expect(isEmptyDef({ elements: [{ element: 'Si' }] })).toBe(true);
    expect(isEmptyDef({ ratios: [{ numerator: 'Si', denominator: 'Al' }] }))
      .toBe(true);
  });

  it('one bound is enough - a one-sided window is a real window', () => {
    expect(isEmptyDef({ elements: [{ element: 'Si', max_at_pct: 5 }] }))
      .toBe(false);
  });
});

describe('seedDefFromRegion', () => {
  const detail = {
    region_id: 2,
    cif_filename: 'Si.cif',
    composition: [
      { element: 'Si', at_pct: 40, spread_at_pct: 5, enrichment: 7.7 },
      { element: 'Al', at_pct: 58, spread_at_pct: 4, enrichment: 0.5 },
    ],
  };

  it('writes a window around the region it came from', () => {
    const d = seedDefFromRegion(detail);
    const si = d.elements.find((e) => e.element === 'Si');
    expect(si.min_at_pct).toBe(30);
    expect(si.max_at_pct).toBe(50);
  });

  it('leaves out elements that are not enriched there', () => {
    // Seeding every element writes clauses that say "and the matrix is still
    // the matrix" - true, useless, and fragile the moment a neighbour shifts.
    const d = seedDefFromRegion(detail);
    expect(d.elements.map((e) => e.element)).toEqual(['Si']);
  });

  it('never writes a negative lower bound', () => {
    const d = seedDefFromRegion({
      region_id: 0,
      composition: [{ element: 'Fe', at_pct: 1, spread_at_pct: 3, enrichment: 9 }],
    });
    expect(d.elements[0].min_at_pct).toBe(0);
  });

  it('returns nothing without a region', () => {
    expect(seedDefFromRegion(null)).toBeNull();
  });
});

describe('the editor', () => {
  it('says the grouping is automatic while there are no definitions', () => {
    setup();
    expect(screen.getByText('defs.none')).toBeTruthy();
  });

  it('adds an empty definition on request', () => {
    const { setDefs } = setup();
    fireEvent.click(screen.getByText('defs.add'));
    expect(setDefs).toHaveBeenCalledWith([expect.objectContaining({
      elements: [], ratios: [], phase_key: '',
    })]);
  });

  it('warns that an empty definition claims nothing', () => {
    setup({ defs: [{ name: 'wip', elements: [] }] });
    expect(screen.getByText('defs.emptyWarning')).toBeTruthy();
  });

  it('cannot seed from a region while none is picked', () => {
    setup();
    expect(screen.getByText('defs.seed').disabled).toBe(true);
  });

  it('seeds from the picked region when there is one', () => {
    const { setDefs } = setup({
      inspectorDetail: {
        region_id: 1, cif_filename: 'Si.cif',
        composition: [{ element: 'Si', at_pct: 40, spread_at_pct: 5,
                        enrichment: 7.7 }],
      },
    });
    fireEvent.click(screen.getByText('defs.seed'));
    expect(setDefs).toHaveBeenCalledWith([expect.objectContaining({
      name: 'Si.cif',
    })]);
  });

  it('reads the element pick before the updater runs', () => {
    // React resets a controlled <select> before a functional state update
    // lands, so reading e.target.value inside the updater yields ''. That
    // shipped once already and the clause arrived with no element in it.
    const defs = [{ name: 'a', elements: [], ratios: [], enrichment: [] }];
    const { setDefs } = setup({ defs });
    // Two selects in an open definition: the element picker and the phase
    // picker. The first is the one under test.
    fireEvent.change(screen.getAllByRole('combobox')[0],
                     { target: { value: 'Fe' } });
    fireEvent.click(screen.getByText('defs.addClause'));
    expect(setDefs).toHaveBeenCalledWith([expect.objectContaining({
      elements: [{ element: 'Fe', min_at_pct: null, max_at_pct: null }],
    })]);
  });

  it('moves a definition, because order is the whole conflict rule', () => {
    const defs = [
      { name: 'a', elements: [{ element: 'Si' }] },
      { name: 'b', elements: [{ element: 'Fe' }] },
    ];
    const { setDefs } = setup({ defs });
    fireEvent.click(screen.getAllByTitle('defs.moveDownTooltip')[0]);
    expect(setDefs).toHaveBeenCalledWith([defs[1], defs[0]]);
  });

  it('cannot move the first one up or the last one down', () => {
    setup({ defs: [{ name: 'a', elements: [{ element: 'Si' }] }] });
    expect(screen.getByTitle('defs.moveUpTooltip').disabled).toBe(true);
    expect(screen.getByTitle('defs.moveDownTooltip').disabled).toBe(true);
  });

  it('removes a definition', () => {
    const defs = [
      { name: 'a', elements: [{ element: 'Si' }] },
      { name: 'b', elements: [{ element: 'Fe' }] },
    ];
    const { setDefs } = setup({ defs });
    fireEvent.click(screen.getAllByText('×')[0]);
    expect(setDefs).toHaveBeenCalledWith([defs[1]]);
  });

  it('shows what each window really claimed, from the backend', async () => {
    previewRegionDefs.mockResolvedValue({ data: {
      definitions: [{ name: 'a', n_pixels: 202, percentage: 1.87,
                      overlap_pixels: 0, reason: null }],
      unclaimed_pixels: 10, unclaimed_percentage: 0.1, total_pixels: 10800,
    } });
    setup({ defs: [{ name: 'a', elements: [{ element: 'Si', min_at_pct: 20 }] }] });
    fireEvent.click(screen.getByText('defs.preview'));
    await waitFor(() => expect(
      screen.getByText('defs.previewHit:{"px":"202","pct":1.87}')).toBeTruthy());
  });

  it('shows the backend reason when a window claims nothing', async () => {
    previewRegionDefs.mockResolvedValue({ data: {
      definitions: [{ name: 'a', n_pixels: 0, percentage: 0,
                      overlap_pixels: 0, reason: 'Mg not measured here' }],
      unclaimed_pixels: 0, unclaimed_percentage: 0, total_pixels: 10,
    } });
    setup({ defs: [{ name: 'a', elements: [{ element: 'Mg', min_at_pct: 5 }] }] });
    fireEvent.click(screen.getByText('defs.preview'));
    await waitFor(() => expect(
      screen.getByText('Mg not measured here')).toBeTruthy());
  });

  it('cannot preview with nothing to preview', () => {
    setup();
    expect(screen.getByText('defs.preview').disabled).toBe(true);
  });

  it('offers the leftover switch only once something is declared', () => {
    setup();
    expect(screen.queryByText('defs.remainder')).toBeNull();
    cleanup();
    setup({ defs: [{ name: 'a', elements: [{ element: 'Si' }] }] });
    expect(screen.getByText('defs.remainder')).toBeTruthy();
  });

  it('reports how many weights are actually doing something', () => {
    setup({ elementWeights: { Fe: 3, Al: 1 } });
    expect(screen.getByText('defs.weightsActive:{"count":1}')).toBeTruthy();
  });

  it('says nothing about weights while every slider sits at one', () => {
    setup({ elementWeights: { Fe: 1, Al: 1 } });
    expect(screen.queryByText(/defs\.weightsActive/)).toBeNull();
  });

  it('sets a weight without disturbing the others', () => {
    const { setElementWeights } = setup({ elementWeights: { Al: 2 } });
    const sliders = screen.getAllByRole('slider');
    fireEvent.change(sliders[1], { target: { value: '4' } });   // Fe
    expect(setElementWeights).toHaveBeenCalledWith({ Al: 2, Fe: 4 });
  });

  it('runs the classification on request', () => {
    const { onReclassify } = setup({ defs: [{ name: 'a', elements: [] }] });
    fireEvent.click(screen.getByText('defs.apply'));
    expect(onReclassify).toHaveBeenCalled();
  });
});
