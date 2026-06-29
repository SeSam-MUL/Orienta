// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import MissingPhaseClusters from './MissingPhaseClusters';

afterEach(() => cleanup());

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#000', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
  },
}));

describe('MissingPhaseClusters', () => {
  it('renders empty state when no clusters', () => {
    const { getByText } = render(
      <MissingPhaseClusters clusters={[]} presetEntry={null} />
    );
    expect(getByText(/No clusters of unmatched pixels/i)).toBeTruthy();
  });

  it('renders one row per cluster with the structural signature', () => {
    const clusters = [
      { crystal_system: 'cubic', lattice_a_A: 6.4, n_fold: 4,
        pixel_count: 30, fraction_of_analyzed: 0.15 },
      { crystal_system: 'hexagonal', lattice_a_A: 10.2, n_fold: 6,
        pixel_count: 12, fraction_of_analyzed: 0.06 },
    ];
    const { container } = render(
      <MissingPhaseClusters clusters={clusters} presetEntry={null} />
    );
    expect(container.textContent).toMatch(/cubic/);
    expect(container.textContent).toMatch(/6\.4/);
    expect(container.textContent).toMatch(/4-fold/);
    expect(container.textContent).toMatch(/hexagonal/);
    expect(container.textContent).toMatch(/10\.2/);
  });

  it('shows preset-aware hint when lattice matches expected phase', () => {
    const clusters = [
      { crystal_system: 'cubic', lattice_a_A: 6.4, n_fold: 4,
        pixel_count: 30, fraction_of_analyzed: 0.15 },
    ];
    const presetEntry = {
      label: 'AA226',
      expected_phases: [
        { name: 'Mg2Si', a_A: 6.39 },
        { name: 'Al', a_A: 4.05 },
      ],
    };
    const { container } = render(
      <MissingPhaseClusters clusters={clusters} presetEntry={presetEntry} />
    );
    expect(container.textContent).toMatch(/Mg2Si/);
    // Al is NOT within 20% of 6.4Å so shouldn't be listed
    expect(container.textContent).not.toMatch(/^Al$/m);
  });

  it('shows pixel count + percentage', () => {
    const clusters = [
      { crystal_system: 'cubic', lattice_a_A: 5.0, n_fold: 3,
        pixel_count: 47, fraction_of_analyzed: 0.235 },
    ];
    const { container } = render(
      <MissingPhaseClusters clusters={clusters} presetEntry={null} />
    );
    expect(container.textContent).toMatch(/47 pixels/);
    expect(container.textContent).toMatch(/24%/);  // 23.5% rounded
  });
});
