// @vitest-environment jsdom
/**
 * The particle comparison inside the Phase Test dialog: add particles, run, read the two tables.
 */
import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, cleanup, act, fireEvent } from '@testing-library/react';

const RESULT = {
  top_n: 10,
  particles: [
    { name: 'P1', assigned: 'al6fe', next_best: 'mnal6', delta: 0.003, indistinguishable: true,
      phases: [{ phase_key: 'al6fe', mean: 0.565, std: 0.037, max: 0.627, n: 10, n_total: 13 },
        { phase_key: 'mnal6', mean: 0.562, std: 0.035, max: 0.62, n: 10, n_total: 13 }] },
  ],
  phase_order: ['al6fe', 'mnal6'],
  phase_info: { al6fe: { label: 'Al6Fe', space_group: 'Cmcm' }, mnal6: { label: 'MnAl6', space_group: 'Cmcm' } },
  matrix: { al6fe: { P1: 0.565 }, mnal6: { P1: 0.562 } },
  best_per_particle: { P1: 'al6fe' },
  latex: { zuordnung: '\\begin{table}A', kandidaten: '\\begin{table}B' },
  csv: 'Partikel;Phase\n',
};
const start = vi.fn(() => Promise.resolve({ data: { job_id: 'j1', total: 13 } }));
const progress = vi.fn(() => Promise.resolve({ data: { status: 'done', done: 13, total: 13, result: RESULT } }));
vi.mock('../../services/api', () => ({
  indexApi: {
    particleNccStart: (b) => start(b),
    particleNccProgress: (j) => progress(j),
    particleNccCancel: () => Promise.resolve({}),
  },
}));

import i18n from '../../i18n';
import ParticleNccPanel from './ParticleNccPanel';

afterEach(async () => {
  cleanup();
  vi.useRealTimers();
  start.mockClear();
  await act(async () => { await i18n.changeLanguage('en'); });
});

const props = {
  pixelIndex: 7, nRows: 4, nCols: 5,
  phases: [{ key: 'al6fe' }, { key: 'mnal6' }, { key: 'al' }],
  selectedKeys: new Set(['al6fe', 'mnal6']),
  settings: { bandwidth: 128, bgRemove: true, usePixelPc: false, aperture: 'circular', apertureRadius: 1 },
  blocked: false,
};

describe('ParticleNccPanel', () => {
  it('runs the comparison and shows both tables', async () => {
    vi.useFakeTimers();
    await act(async () => { await i18n.changeLanguage('de'); });
    const { getByText, getByTestId, queryByTestId } = render(<ParticleNccPanel {...props} />);
    fireEvent.click(getByText(/als Partikel hinzufügen/));
    expect(getByText(/P1: \(1, 2\), r 2/)).toBeTruthy();
    await act(async () => { fireEvent.click(getByText('1 Partikel mit 2 Phasen vergleichen')); });
    expect(start).toHaveBeenCalledWith(expect.objectContaining({
      phase_keys: ['al6fe', 'mnal6'], top_n: 10,
      particles: [{ name: 'P1', center_index: 7, radius_px: 2 }],
    }));
    await act(async () => { await vi.advanceTimersByTimeAsync(800); });
    const res = getByTestId('particle-ncc-result');
    expect(res.textContent).toContain('0,565 ± 0,037');
    expect(res.textContent).toContain('≈ nicht unterscheidbar');
    expect(res.textContent).toContain('MnAl6 (0,562)');
    expect(queryByTestId('particle-ncc-result')).toBeTruthy();
  });

  it('cannot run without particles or while the phase list is blocked', () => {
    const { getByText, rerender } = render(<ParticleNccPanel {...props} />);
    expect(getByText('Compare 0 particles with 2 phases').disabled).toBe(true);
    rerender(<ParticleNccPanel {...props} blocked />);
    fireEvent.click(getByText(/as a particle/));
    expect(getByText('Compare 1 particle with 2 phases').disabled).toBe(true);
  });
});
