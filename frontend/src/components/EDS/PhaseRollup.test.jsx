// @vitest-environment jsdom
/**
 * "11 regions → 4 phases".
 *
 * The user read the repeated phase names in the region list as a fault:
 * "die tauchen dann ca 100 mal in der Liste auf". They are not — a phase
 * covering four regions is written four times because the list is per
 * region. This box counts each phase once and points at the view where
 * neighbouring regions of one phase become one area.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

import PhaseRollup, { rollUpPhases, countUnnamed } from './PhaseRollup';

const STRUCTURES = [
  { region_id: 0, n_pixels: 3000, phase_index: 0, cif_filename: 'Al.cif' },
  { region_id: 1, n_pixels: 2000, phase_index: 0, cif_filename: 'Al.cif' },
  { region_id: 2, n_pixels: 500, phase_index: 1, cif_filename: 'Si.cif' },
  { region_id: 3, n_pixels: 100, phase_index: -1, cif_filename: null },
];

const ALL_PHASES = [
  { phase_index: 0, cif_filename: 'Al.cif', color: '#c792ea' },
  { phase_index: 1, cif_filename: 'Si.cif', color: '#ff526f' },
  { phase_index: 2, cif_filename: 'Cu.cif', color: '#5fe47a' },
];

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

describe('rollUpPhases', () => {
  it('counts a phase once however many regions carry it', () => {
    const rolled = rollUpPhases(STRUCTURES, ALL_PHASES);
    expect(rolled).toHaveLength(2);
    const al = rolled.find((p) => p.cif_filename === 'Al.cif');
    expect(al.n_regions).toBe(2);
    expect(al.n_pixels).toBe(5000);
  });

  it('leaves out phases no region carries', () => {
    const rolled = rollUpPhases(STRUCTURES, ALL_PHASES);
    expect(rolled.map((p) => p.cif_filename)).not.toContain('Cu.cif');
  });

  it('ignores regions that have no phase yet', () => {
    const rolled = rollUpPhases(STRUCTURES, ALL_PHASES);
    expect(rolled.reduce((n, p) => n + p.n_regions, 0)).toBe(3);
  });

  it('takes the colour from the phase list, not from a region', () => {
    // The region palette is a different scheme on purpose; borrowing it
    // here would show a phase in a colour the phase map never paints.
    const rolled = rollUpPhases(STRUCTURES, ALL_PHASES);
    expect(rolled.find((p) => p.cif_filename === 'Al.cif').color).toBe('#c792ea');
  });

  it('keys on the phase index, so two entries sharing a name stay apart', () => {
    const regions = [
      { region_id: 0, n_pixels: 10, phase_index: 0, cif_filename: 'Al.cif' },
      { region_id: 1, n_pixels: 10, phase_index: 1, cif_filename: 'Al.cif' },
    ];
    const phases = [
      { phase_index: 0, cif_filename: 'Al.cif', color: '#111111' },
      { phase_index: 1, cif_filename: 'Al.cif', color: '#222222' },
    ];
    expect(rollUpPhases(regions, phases)).toHaveLength(2);
  });

  it('puts the biggest phase first', () => {
    const rolled = rollUpPhases(STRUCTURES, ALL_PHASES);
    expect(rolled[0].cif_filename).toBe('Al.cif');
  });

  it('survives empty input', () => {
    expect(rollUpPhases([], [])).toEqual([]);
    expect(rollUpPhases(undefined, undefined)).toEqual([]);
  });
});

describe('countUnnamed', () => {
  it('counts the regions with no phase', () => {
    expect(countUnnamed(STRUCTURES)).toBe(1);
  });

  it('treats a missing phase_index as unnamed rather than as phase 0', () => {
    expect(countUnnamed([{ region_id: 0, n_pixels: 1 }])).toBe(1);
  });
});

describe('the roll-up box', () => {
  it('states the collapse in one line', () => {
    render(<PhaseRollup regions={STRUCTURES} allPhases={ALL_PHASES} />);
    expect(screen.getByText('rollup.headline:{"regions":4,"phases":2}'))
      .toBeTruthy();
  });

  it('warns while regions are still unnamed', () => {
    render(<PhaseRollup regions={STRUCTURES} allPhases={ALL_PHASES} />);
    expect(screen.getByText('rollup.unnamed:{"count":1}')).toBeTruthy();
  });

  it('says nothing about unnamed regions once there are none', () => {
    const named = STRUCTURES.filter((s) => s.phase_index >= 0);
    render(<PhaseRollup regions={named} allPhases={ALL_PHASES} />);
    expect(screen.queryByText(/rollup\.unnamed/)).toBeNull();
  });

  it('shows how many regions each phase came from', () => {
    render(<PhaseRollup regions={STRUCTURES} allPhases={ALL_PHASES} />);
    expect(screen.getByText('rollup.fromRegions:{"count":2}')).toBeTruthy();
  });

  it('switches to the phase view when asked', () => {
    const onShow = vi.fn();
    render(<PhaseRollup regions={STRUCTURES} allPhases={ALL_PHASES}
                        onShowPhases={onShow} />);
    fireEvent.click(screen.getByText('rollup.showPhases'));
    expect(onShow).toHaveBeenCalled();
  });

  it('renders nothing without regions', () => {
    const { container } = render(
      <PhaseRollup regions={[]} allPhases={ALL_PHASES} />);
    expect(container.firstChild).toBeNull();
  });

  it('shows area shares against the whole map, not against the named part', () => {
    // 3000+2000 of 10000 px is 50%, not 5000/5600.
    render(<PhaseRollup regions={STRUCTURES} allPhases={ALL_PHASES}
                        totalPixels={10000} />);
    expect(screen.getByText('50.0%')).toBeTruthy();
  });
});
