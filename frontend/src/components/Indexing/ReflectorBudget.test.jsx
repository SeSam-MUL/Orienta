// @vitest-environment jsdom
//
// The reflector control exists because trimming reflector families changes the
// ANSWER, not just the memory: measured on Ni (58 families, 800 real patterns)
// 40 and 32 reproduce the full set exactly, 24 indexes 192/800 at 119.7 deg,
// and 16 fails outright — silently, with orientations that still look like
// data. So the UI has two jobs: show what each choice costs, and never make
// the choice itself.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

vi.mock('../../services/api', () => ({
  indexApi: { houghReflectorCost: vi.fn() },
}));

import { indexApi } from '../../services/api';
import ReflectorBudget from './ReflectorBudget';

const GiB = 1024 ** 3;

/** The measured shape of Al7FeCu2.cif — a CIF stored as P 1. */
const NO_SYMMETRY = {
  cif_path: 'C:/lib/Al7FeCu2.cif',
  phase_name: 'Al7FeCu2',
  point_group: '1',
  space_group: 'P1',
  reflectors_available: 70,
  budget_bytes: 4.3 * GiB,
  available_bytes: 8.6 * GiB,
  no_symmetry: true,
  table: [
    { reflectors: 70, rows: 1000494880, bytes: 44.726 * GiB, fits: false },
    { reflectors: 50, rows: 112498750, bytes: 5.029 * GiB, fits: false },
    { reflectors: 40, rows: 32412765, bytes: 1.449 * GiB, fits: true },
    { reflectors: 32, rows: 7126740, bytes: 0.319 * GiB, fits: true },
    { reflectors: 24, rows: 0, bytes: 0, fits: true },
  ],
};

const CUBIC = {
  ...NO_SYMMETRY,
  cif_path: 'C:/lib/Al.cif',
  phase_name: 'Al',
  point_group: 'm-3m',
  space_group: 'Fm-3m',
  reflectors_available: 64,
  no_symmetry: false,
  table: [
    { reflectors: 64, rows: 0, bytes: 0, fits: true },
    { reflectors: 40, rows: 0, bytes: 0, fits: true },
    { reflectors: 32, rows: 0, bytes: 0, fits: true },
  ],
};

beforeEach(() => indexApi.houghReflectorCost.mockReset());
afterEach(() => cleanup());

describe('ReflectorBudget', () => {
  it('shows the price of every option, including the ones that do not fit', async () => {
    indexApi.houghReflectorCost.mockResolvedValue({ data: NO_SYMMETRY });
    render(<ReflectorBudget cifPath={NO_SYMMETRY.cif_path} onChange={() => {}} />);

    const select = await screen.findByRole('combobox');
    const options = [...select.querySelectorAll('option')].map((o) => o.textContent);
    expect(options).toHaveLength(5);
    expect(options[0]).toMatch(/44\.73 GiB/);   // the honest full-set price
    expect(options[3]).toMatch(/0\.32 GiB/);
    // An unaffordable option stays SELECTABLE and is marked, rather than being
    // hidden: a missing option reads as "this was never possible", and the run
    // would then fail with no way to see why.
    expect(options[0]).toMatch(/too big/i);
    expect([...select.querySelectorAll('option')][0].disabled).toBe(false);
  });

  it('defaults to the full set and reports "all" rather than a bare number', async () => {
    indexApi.houghReflectorCost.mockResolvedValue({ data: NO_SYMMETRY });
    render(<ReflectorBudget cifPath={NO_SYMMETRY.cif_path} value={null} onChange={() => {}} />);
    const select = await screen.findByRole('combobox');
    expect(select.value).toBe('70');
    expect(select.querySelector('option').textContent).toMatch(/all 70/);
  });

  it('reports "no limit" (null) when the full set is chosen again', async () => {
    indexApi.houghReflectorCost.mockResolvedValue({ data: NO_SYMMETRY });
    const onChange = vi.fn();
    render(<ReflectorBudget cifPath={NO_SYMMETRY.cif_path} value={32} onChange={onChange} />);
    const select = await screen.findByRole('combobox');

    fireEvent.change(select, { target: { value: '40' } });
    expect(onChange).toHaveBeenLastCalledWith(40);

    // Back to the full set must clear the limit, not pin it to 70: pinning
    // would freeze this phase at today's family count and silently ignore a
    // later CIF that has more.
    fireEvent.change(select, { target: { value: '70' } });
    expect(onChange).toHaveBeenLastCalledWith(null);
  });

  it('names the P 1 CIF, which is the fixable cause', async () => {
    indexApi.houghReflectorCost.mockResolvedValue({ data: NO_SYMMETRY });
    render(<ReflectorBudget cifPath={NO_SYMMETRY.cif_path} onChange={() => {}} />);
    expect(await screen.findByText(/no symmetry \(P1\)/i)).toBeInTheDocument();
  });

  it('says nothing about symmetry for a phase that has some', async () => {
    indexApi.houghReflectorCost.mockResolvedValue({ data: CUBIC });
    render(<ReflectorBudget cifPath={CUBIC.cif_path} onChange={() => {}} />);
    await screen.findByRole('combobox');
    expect(screen.queryByText(/no symmetry/i)).toBeNull();
    // Nothing is unaffordable here, so no budget scolding either.
    expect(screen.queryByText(/does not fit/i)).toBeNull();
  });

  it('warns to check the result only once a limit is actually set', async () => {
    indexApi.houghReflectorCost.mockResolvedValue({ data: NO_SYMMETRY });
    const { rerender } = render(
      <ReflectorBudget cifPath={NO_SYMMETRY.cif_path} value={null} onChange={() => {}} />,
    );
    await screen.findByRole('combobox');
    expect(screen.queryByText(/change the result/i)).toBeNull();

    rerender(<ReflectorBudget cifPath={NO_SYMMETRY.cif_path} value={32} onChange={() => {}} />);
    expect(await screen.findByText(/change the result/i)).toBeInTheDocument();
  });

  it('degrades to a note instead of breaking the phase list', async () => {
    // Rejected with the shape axios actually produces, not a bare `Error` —
    // which is also what keeps the runner from reporting it as an unhandled
    // error before the component's catch has had a chance to say otherwise.
    indexApi.houghReflectorCost.mockRejectedValueOnce({
      response: { data: { detail: 'Could not read broken.cif' } },
    });
    render(<ReflectorBudget cifPath="C:/lib/broken.cif" onChange={() => {}} />);
    expect(await screen.findByText(/unavailable/i)).toBeInTheDocument();
  });

  it('re-reads when the phase changes, so one card cannot show another\'s cost', async () => {
    indexApi.houghReflectorCost.mockResolvedValue({ data: CUBIC });
    const { rerender } = render(<ReflectorBudget cifPath={CUBIC.cif_path} onChange={() => {}} />);
    await screen.findByRole('combobox');

    indexApi.houghReflectorCost.mockResolvedValue({ data: NO_SYMMETRY });
    rerender(<ReflectorBudget cifPath={NO_SYMMETRY.cif_path} onChange={() => {}} />);
    await waitFor(() => expect(screen.getByRole('combobox').value).toBe('70'));
    expect(indexApi.houghReflectorCost).toHaveBeenCalledTimes(2);
  });
});
