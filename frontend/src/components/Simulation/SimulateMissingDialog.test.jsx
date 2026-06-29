// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup, render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import SimulateMissingDialog from './SimulateMissingDialog';

afterEach(() => cleanup());

const MISSING = [
  { stem: 'Al', xtal_path: 'X/Al.xtal', missing: ['sht'],
    a_max_nm: 0.405, recommended_dmin: 0.05, dmin_level: 'ok',
    min_occupancy: 1.0, occupancy_level: 'ok' },
  { stem: 'Big', xtal_path: 'X/Big.xtal', missing: ['h5', 'sht'],
    a_max_nm: 1.232, recommended_dmin: 0.09, dmin_level: 'large',
    min_occupancy: 1.0, occupancy_level: 'ok' },
  { stem: 'Dis', xtal_path: 'X/Dis.xtal', missing: ['sht'],
    a_max_nm: 1.0, recommended_dmin: 0.07, dmin_level: 'large',
    min_occupancy: 0.14, occupancy_level: 'severe' },
];

function setup(overrides = {}) {
  const onLaunch = vi.fn();
  const onClose = vi.fn();
  render(<SimulateMissingDialog missing={MISSING} defaultDmin={0.05}
    onLaunch={onLaunch} onClose={onClose} {...overrides} />);
  return { onLaunch, onClose };
}

describe('SimulateMissingDialog', () => {
  it('lists phases, flags large + disordered', () => {
    setup();
    expect(screen.getByText('Al')).toBeInTheDocument();
    expect(screen.getByText('Dis')).toBeInTheDocument();
    expect(screen.getAllByText(/LARGE/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/disordered \(occ 14%\)/)).toBeInTheDocument();
    expect(screen.getByText(/1 disordered/)).toBeInTheDocument();
  });

  it('pre-fills recommended dmin', () => {
    setup();
    expect(screen.getByDisplayValue('0.05')).toBeInTheDocument();
    expect(screen.getByDisplayValue('0.09')).toBeInTheDocument();
    expect(screen.getByDisplayValue('0.07')).toBeInTheDocument();
  });

  it('excludes severe-disordered phases by default and launches a selection list', () => {
    const { onLaunch } = setup();
    // default included = Al + Big (Dis severe -> excluded) -> "Launch 2 jobs"
    fireEvent.click(screen.getByRole('button', { name: /Launch 2 job/ }));
    expect(onLaunch).toHaveBeenCalledTimes(1);
    const sel = onLaunch.mock.calls[0][0];
    const stems = sel.map(s => s.stem);
    expect(stems).toEqual(['Al', 'Big']);
    expect(stems).not.toContain('Dis');
    expect(sel).toContainEqual({ stem: 'Al', xtal_path: 'X/Al.xtal', dmin: 0.05 });
    expect(sel).toContainEqual({ stem: 'Big', xtal_path: 'X/Big.xtal', dmin: 0.09 });
  });

  it('can include a disordered phase by ticking it', () => {
    const { onLaunch } = setup();
    const boxes = screen.getAllByRole('checkbox');
    // order: Al, Big, Dis -> Dis is index 2 (unchecked by default)
    expect(boxes[2].checked).toBe(false);
    fireEvent.click(boxes[2]);
    fireEvent.click(screen.getByRole('button', { name: /Launch 3 job/ }));
    const stems = onLaunch.mock.calls[0][0].map(s => s.stem);
    expect(stems).toContain('Dis');
  });

  it('can exclude an ordered phase', () => {
    const { onLaunch } = setup();
    const boxes = screen.getAllByRole('checkbox');
    fireEvent.click(boxes[1]); // uncheck Big
    fireEvent.click(screen.getByRole('button', { name: /Launch 1 job/ }));
    const stems = onLaunch.mock.calls[0][0].map(s => s.stem);
    expect(stems).toEqual(['Al']);
  });

  it('"set all to" applies one dmin to every phase', () => {
    const { onLaunch } = setup();
    fireEvent.change(screen.getByPlaceholderText('0.05'), { target: { value: '0.11' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
    fireEvent.click(screen.getByRole('button', { name: /Launch 2 job/ }));
    const sel = onLaunch.mock.calls[0][0];
    expect(sel.every(s => s.dmin === 0.11)).toBe(true);
  });

  it('calls onClose from Cancel', () => {
    const { onClose } = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // Two-way engine switch: single header pipeline badge, no per-phase column.
  // -------------------------------------------------------------------------
  it('shows the "GPU pipeline" header badge by default (engine=ours)', () => {
    setup({ engine: 'ours' });
    expect(screen.getByText(/GPU pipeline/)).toBeInTheDocument();
    expect(screen.queryByText(/EMsoft pipeline/)).toBeNull();
  });

  it('shows the "EMsoft pipeline" header badge when engine=emsoft', () => {
    setup({ engine: 'emsoft' });
    expect(screen.getByText(/EMsoft pipeline/)).toBeInTheDocument();
    expect(screen.queryByText(/GPU pipeline/)).toBeNull();
  });

  it('never renders a per-phase engine column / pills', () => {
    setup({ engine: 'ours' });
    // No per-phase routing pills (those were the old 3-way auto mode).
    expect(screen.queryByText('⚡ GPU')).toBeNull();
    expect(screen.queryByText('⚙ EMsoft')).toBeNull();
    // No "engine" column header.
    expect(screen.queryByRole('columnheader', { name: 'engine' })).toBeNull();
  });

  it('launch payload carries only stem/xtal_path/dmin (no recommended_engine)', () => {
    const { onLaunch } = setup({ engine: 'ours' });
    fireEvent.click(screen.getByRole('button', { name: /Launch 2 job/ }));
    const sel = onLaunch.mock.calls[0][0];
    expect(sel).toContainEqual({ stem: 'Al',  xtal_path: 'X/Al.xtal',  dmin: 0.05 });
    expect(sel).toContainEqual({ stem: 'Big', xtal_path: 'X/Big.xtal', dmin: 0.09 });
    expect(sel.every(s => !('recommended_engine' in s))).toBe(true);
  });
});
