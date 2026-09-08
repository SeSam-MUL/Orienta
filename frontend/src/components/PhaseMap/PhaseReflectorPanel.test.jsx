// @vitest-environment jsdom
//
// The reflector control has to exist where the failure appears. Phase
// Verification and the pseudo-symmetry step after a spherical run build their
// own Hough indexers, so a phase they cannot afford is unusable on this page
// even when the Indexing page was never opened for this result.
//
// The other half of the contract is that it costs nothing until asked:
// measuring one phase's cost curve probes a build per reflector count and takes
// 3-5 s (measured), so three phases fetched on page load would be ten seconds
// of work for a question nobody asked.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';

vi.mock('../../services/api', () => ({
  indexApi: {
    phaseCheckPhases: vi.fn(),
    setHoughReflectorLimit: vi.fn(() => Promise.resolve({ data: {} })),
    houghReflectorCost: vi.fn(),
  },
}));

import { indexApi } from '../../services/api';
import PhaseReflectorPanel from './PhaseReflectorPanel';

const GiB = 1024 ** 3;

const PHASES = {
  phases: [
    { phase_id: 1, name: 'Al7FeCu2', cif_path: 'C:/lib/Al7FeCu2.cif', point_group: '1', max_reflectors: 32 },
    { phase_id: 2, name: 'Al', cif_path: 'C:/lib/Al.cif', point_group: 'm-3m', max_reflectors: null },
  ],
};

const COST = {
  reflectors_available: 70,
  budget_bytes: 4.3 * GiB,
  no_symmetry: true,
  space_group: 'P1',
  table: [
    { reflectors: 70, rows: 1000494880, bytes: 44.726 * GiB, fits: false },
    { reflectors: 32, rows: 7126740, bytes: 0.319 * GiB, fits: true },
  ],
};

beforeEach(() => {
  indexApi.phaseCheckPhases.mockReset();
  indexApi.setHoughReflectorLimit.mockReset();
  indexApi.setHoughReflectorLimit.mockResolvedValue({ data: {} });
  indexApi.houghReflectorCost.mockReset();
  indexApi.houghReflectorCost.mockResolvedValue({ data: COST });
});
afterEach(() => cleanup());

describe('PhaseReflectorPanel', () => {
  it('fetches nothing until it is opened', async () => {
    indexApi.phaseCheckPhases.mockResolvedValue({ data: PHASES });
    render(<PhaseReflectorPanel resultId="r1" />);
    expect(indexApi.phaseCheckPhases).not.toHaveBeenCalled();
    expect(indexApi.houghReflectorCost).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button'));
    await waitFor(() => expect(indexApi.phaseCheckPhases).toHaveBeenCalledTimes(1));
  });

  it('lists each phase with its symmetry and its own control', async () => {
    indexApi.phaseCheckPhases.mockResolvedValue({ data: PHASES });
    render(<PhaseReflectorPanel resultId="r1" />);
    fireEvent.click(screen.getByRole('button'));

    expect(await screen.findByText(/Al7FeCu2/)).toBeInTheDocument();
    expect(screen.getByText(/\(1\)/)).toBeInTheDocument();       // point group
    expect(screen.getByText(/\(m-3m\)/)).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByRole('combobox')).toHaveLength(2));
  });

  it('shows the limit the backend already remembers, not a fresh "all"', async () => {
    // The value may have been set on the Indexing page, or in an earlier
    // session — showing "all 70" here would misreport the state of the run.
    indexApi.phaseCheckPhases.mockResolvedValue({ data: PHASES });
    render(<PhaseReflectorPanel resultId="r1" />);
    fireEvent.click(screen.getByRole('button'));
    const boxes = await screen.findAllByRole('combobox');
    expect(boxes[0].value).toBe('32');
  });

  it('registers a change immediately — there is no run here to carry it', async () => {
    indexApi.phaseCheckPhases.mockResolvedValue({ data: PHASES });
    render(<PhaseReflectorPanel resultId="r1" />);
    fireEvent.click(screen.getByRole('button'));
    const boxes = await screen.findAllByRole('combobox');

    fireEvent.change(boxes[0], { target: { value: '70' } });
    // 70 is the full set, so the limit is CLEARED rather than pinned.
    expect(indexApi.setHoughReflectorLimit).toHaveBeenLastCalledWith('C:/lib/Al7FeCu2.cif', null);

    fireEvent.change(boxes[0], { target: { value: '32' } });
    expect(indexApi.setHoughReflectorLimit).toHaveBeenLastCalledWith('C:/lib/Al7FeCu2.cif', 32);
  });

  it('says so plainly when no phase has a CIF', async () => {
    indexApi.phaseCheckPhases.mockResolvedValue({ data: { phases: [] } });
    render(<PhaseReflectorPanel resultId="r1" />);
    fireEvent.click(screen.getByRole('button'));
    expect(await screen.findByText(/no phase in this result has a cif/i)).toBeInTheDocument();
  });

  it('survives a backend that cannot answer', async () => {
    indexApi.phaseCheckPhases.mockRejectedValueOnce({
      response: { data: { detail: 'No indexing result available' } },
    });
    render(<PhaseReflectorPanel resultId={null} />);
    fireEvent.click(screen.getByRole('button'));
    expect(await screen.findByText(/unavailable/i)).toBeInTheDocument();
  });
});
