// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import RegionStatsPanel from './RegionStatsPanel';

afterEach(() => cleanup());

vi.mock('../../theme/components', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#000', border: '#444', accent: '#a0f',
    cyan: '#0ff', red: '#f00', green: '#0f0', purple: '#a0f',
    text: '#fff', textSecondary: '#aaa',
  },
  spacing: { groupMargin: 8, groupSpacing: 6, outerSpacing: 8 },
  GroupBox: ({ title, children, style }) => (
    <div data-groupbox style={style}>
      <div data-groupbox-title>{title}</div>
      <div data-groupbox-body>{children}</div>
    </div>
  ),
}));

const sampleData = {
  row_start: 5, row_end: 15,
  col_start: 8, col_end: 20,
  n_pixels: 143,
  phases: [
    { id: 1, name: 'Al7Cu2Fe', count: 100, fraction: 0.6993 },
    { id: 2, name: 'Al',       count:  30, fraction: 0.2098 },
    { id: -1, name: 'unindexed', count: 13, fraction: 0.0909 },
  ],
  scalars: {
    ci: { mean: 0.834, std: 0.123 },
    bc: { mean: 187.6, std: 22.4 },
    kam: null,
    gos: { mean: 1.45, std: 0.31 },
    uncertainty: { mean: 0.05, std: 0.02 },
  },
};

describe('RegionStatsPanel', () => {
  it('renders nothing tabley when no data + no loading + no error', () => {
    const { container } = render(
      <RegionStatsPanel data={null} error={null} loading={false} />
    );
    // Returns null entirely.
    expect(container.firstChild).toBeNull();
  });

  it('shows a Computing… placeholder when loading', () => {
    const { getByText } = render(
      <RegionStatsPanel data={null} error={null} loading={true} />
    );
    expect(getByText(/Computing/i)).toBeTruthy();
  });

  it('shows a red error message on error', () => {
    const { getByText, container } = render(
      <RegionStatsPanel data={null} error="boom" loading={false} />
    );
    expect(getByText(/boom/i)).toBeTruthy();
    const errorNode = container.querySelector('[data-region-stats-error]');
    expect(errorNode).toBeTruthy();
    expect(errorNode.style.color).toMatch(/#f00|red|rgb\(255,\s*0,\s*0\)/);
  });

  it('renders the rectangle header with bounds and pixel count', () => {
    const { getByText } = render(
      <RegionStatsPanel data={sampleData} error={null} loading={false} />
    );
    // exact format: "Region (r=5..15, c=8..20) · 143 px"
    expect(getByText(/r=5\.\.15/)).toBeTruthy();
    expect(getByText(/c=8\.\.20/)).toBeTruthy();
    expect(getByText(/143 px/)).toBeTruthy();
  });

  it('renders phase table rows sorted by fraction desc with unindexed last', () => {
    const { container, getByText } = render(
      <RegionStatsPanel data={sampleData} error={null} loading={false} />
    );
    expect(getByText('Al7Cu2Fe')).toBeTruthy();
    expect(getByText('Al')).toBeTruthy();
    expect(getByText('unindexed')).toBeTruthy();
    expect(getByText('69.9%')).toBeTruthy();   // 0.6993 → 69.9%
    expect(getByText('21.0%')).toBeTruthy();   // 0.2098 → 21.0%
    expect(getByText('9.1%')).toBeTruthy();    // 0.0909 → 9.1%

    const phaseRows = Array.from(
      container.querySelectorAll('[data-region-phase-row]')
    ).map((el) => el.getAttribute('data-name'));
    // unindexed forced to the very end regardless of fraction order
    expect(phaseRows[phaseRows.length - 1]).toBe('unindexed');
  });

  it('renders scalar table with mean ± std and skips null scalars', () => {
    const { container, getByText, queryByText } = render(
      <RegionStatsPanel data={sampleData} error={null} loading={false} />
    );
    // ci: 0.83 ± 0.12
    expect(getByText(/0\.83/)).toBeTruthy();
    expect(getByText(/0\.12/)).toBeTruthy();
    // bc: 187.60 ± 22.40
    expect(getByText(/187\.60/)).toBeTruthy();
    expect(getByText(/22\.40/)).toBeTruthy();

    const scalarRows = Array.from(
      container.querySelectorAll('[data-region-scalar-row]')
    ).map((el) => el.getAttribute('data-key'));
    expect(scalarRows).not.toContain('kam');  // null → skipped
    expect(queryByText('kam')).toBeNull();
  });
});
