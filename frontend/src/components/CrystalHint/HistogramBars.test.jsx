// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import HistogramBars from './HistogramBars';

afterEach(() => cleanup());

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#000', border: '#444',
    cyan: '#0ff', red: '#f00', green: '#0f0', orange: '#fa3',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
  },
}));

describe('HistogramBars', () => {
  it('renders empty state when data is empty', () => {
    const { getByText } = render(
      <HistogramBars title="My histo" data={{}} total={0} />
    );
    expect(getByText('No data')).toBeTruthy();
  });

  it('renders one bar per entry with the count visible', () => {
    const { getByText, container } = render(
      <HistogramBars
        title="Symmetry distribution"
        data={{ 2: 10, 4: 30, 6: 5 }}
        total={45}
        labelKey={k => `${k}-fold`}
      />
    );
    expect(getByText('2-fold')).toBeTruthy();
    expect(getByText('4-fold')).toBeTruthy();
    expect(getByText('6-fold')).toBeTruthy();
    // Each row contributes one <span> as the bar track wrapper. We look for
    // the three count cells (right-aligned monospace), which are easier to
    // match across React's style serialisation.
    const counts = [...container.querySelectorAll('span')]
      .filter(s => /^(\d+)/.test(s.textContent || ''));
    expect(counts.length).toBeGreaterThanOrEqual(3);
  });

  it('sorts bars by count desc when no orderKeys given', () => {
    const { container } = render(
      <HistogramBars
        title="t" data={{ a: 5, b: 30, c: 10 }} total={45}
      />
    );
    const rows = container.querySelectorAll('span[style*="font-family: monospace"]');
    // First label cell should be 'b' (count=30)
    expect(rows[0].textContent).toBe('b');
  });

  it('respects orderKeys', () => {
    const { container } = render(
      <HistogramBars
        title="t"
        data={{ '3.5-5.0': 12, '<3.5': 5, '5.0-7.0': 7 }}
        total={24}
        orderKeys={['<3.5', '3.5-5.0', '5.0-7.0', '7.0-10.0']}
      />
    );
    const rows = container.querySelectorAll('span[style*="font-family: monospace"]');
    expect(rows[0].textContent).toBe('<3.5');
  });

  it('shows percentage when total is given', () => {
    const { container } = render(
      <HistogramBars title="t" data={{ a: 10 }} total={100} />
    );
    // 10/100 = 10%
    expect(container.textContent).toMatch(/10%/);
  });
});
