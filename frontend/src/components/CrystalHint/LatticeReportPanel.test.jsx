// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import LatticeReportPanel from './LatticeReportPanel';

afterEach(() => cleanup());

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#000', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
    green: '#0f0', red: '#f00', orange: '#fa3',
  },
}));

describe('LatticeReportPanel', () => {
  it('returns null when lattice is null', () => {
    const { container } = render(<LatticeReportPanel lattice={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the estimated a + range + system + hkls', () => {
    const lattice = {
      d_spacings_A: [2.34, 2.02, 1.43],
      a_estimate_A: 4.05,
      a_range_A: [3.24, 4.86],
      confidence: 'high',
      crystal_system_best: 'cubic — FCC (Fm-3m / Fd-3m)',
      candidate_hkls: [[1, 1, 1], [2, 0, 0], [2, 2, 0]],
      warnings: [],
    };
    const { container } = render(<LatticeReportPanel lattice={lattice} />);
    expect(container.textContent).toMatch(/4\.05/);
    expect(container.textContent).toMatch(/cubic/);
    expect(container.textContent).toMatch(/FCC/);
    expect(container.textContent).toMatch(/\(111\)/);
    expect(container.textContent).toMatch(/2\.34/);
  });

  it('shows fallback for missing a_estimate', () => {
    const lattice = {
      d_spacings_A: [],
      a_estimate_A: null,
      a_range_A: null,
      confidence: 'none',
      crystal_system_best: null,
      candidate_hkls: [],
      warnings: ['no bands'],
    };
    const { container } = render(<LatticeReportPanel lattice={lattice} />);
    expect(container.textContent).toMatch(/—/);
    expect(container.textContent).toMatch(/no bands/);
  });

  it('renders warnings list when present', () => {
    const lattice = {
      d_spacings_A: [2.0],
      a_estimate_A: 4.0,
      a_range_A: [3.2, 4.8],
      confidence: 'low',
      crystal_system_best: 'cubic',
      candidate_hkls: [[1, 1, 1]],
      warnings: ['rough heuristic', 'voltage corrupt'],
    };
    const { container } = render(<LatticeReportPanel lattice={lattice} />);
    expect(container.textContent).toMatch(/rough heuristic/);
    expect(container.textContent).toMatch(/voltage corrupt/);
  });
});
