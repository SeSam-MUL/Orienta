// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent } from '@testing-library/react';
import DegeneracyWarningBanner from './DegeneracyWarningBanner';

afterEach(() => cleanup());

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#000', border: '#444',
    cyan: '#0ff', red: '#f00', green: '#0f0',
    text: '#fff', textSecondary: '#aaa', yellow: '#fd0',
  },
}));

const hardCluster = { tier: 'hard', members: [0, 2], reason: 'Al6Fe ≈ MnAl6' };
const softCluster = { tier: 'soft', members: [1, 3], reason: 'aIm3 / aPm3' };

describe('DegeneracyWarningBanner', () => {
  it('renders null when there are no clusters', () => {
    const { container } = render(
      <DegeneracyWarningBanner degeneracy={{ clusters: [], byPhaseIndex: {} }}
        onReducePhases={vi.fn()} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders a hard row with a reduce button and a soft row without one', () => {
    const { getByText, getAllByRole } = render(
      <DegeneracyWarningBanner
        degeneracy={{ clusters: [hardCluster, softCluster], byPhaseIndex: {} }}
        onReducePhases={vi.fn()} />
    );
    expect(getByText(/Al6Fe ≈ MnAl6/)).toBeTruthy();
    expect(getByText(/aIm3 \/ aPm3/)).toBeTruthy();
    // Exactly one action button (hard cluster only).
    expect(getAllByRole('button')).toHaveLength(1);
  });

  it('clicking "reduce" calls onReducePhases with the planned indices', () => {
    const onReduce = vi.fn();
    const { getByRole } = render(
      <DegeneracyWarningBanner
        degeneracy={{ clusters: [hardCluster], byPhaseIndex: {} }}
        onReducePhases={onReduce} />
    );
    fireEvent.click(getByRole('button'));
    expect(onReduce).toHaveBeenCalledWith([2]); // keep index 0, remove 2
  });
});
