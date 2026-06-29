// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import PhaseMapProbeOverlay from './PhaseMapProbeOverlay';

afterEach(() => cleanup());

vi.mock('../../theme/components', () => ({
  colors: {
    bgSecondary: '#000', border: '#444', cyan: '#0ff',
    red: '#f00', green: '#0f0', purple: '#a0f',
    text: '#fff', textSecondary: '#aaa',
  },
}));

const probe = {
  row: 5, col: 7,
  phase: { id: 1, name: 'Al7Cu2Fe' },
  scalars: {
    ci: 0.876543,
    bc: 178.4,
    kam: 1.234,
    gos: 2.5,
    uncertainty: 0.123456,
  },
};

describe('PhaseMapProbeOverlay', () => {
  it('renders nothing when visible=false', () => {
    const { container } = render(
      <PhaseMapProbeOverlay probe={probe} visible={false} pos={{ x: 100, y: 100 }} />
    );
    expect(container.querySelector('[data-phasemap-probe-overlay]')).toBeNull();
  });

  it('renders nothing when pos is null', () => {
    const { container } = render(
      <PhaseMapProbeOverlay probe={probe} visible={true} pos={null} />
    );
    expect(container.querySelector('[data-phasemap-probe-overlay]')).toBeNull();
  });

  it('renders header (row, col) + phase + scalars when probe is present', () => {
    const { container, getByText } = render(
      <PhaseMapProbeOverlay probe={probe} visible={true} pos={{ x: 100, y: 100 }} />
    );
    expect(container.querySelector('[data-phasemap-probe-overlay]')).toBeTruthy();
    expect(getByText('(5, 7)')).toBeTruthy();
    expect(getByText('Phase')).toBeTruthy();
    expect(getByText('Al7Cu2Fe')).toBeTruthy();
    // ci is 3 decimals
    expect(getByText('0.877')).toBeTruthy();
    // bc rounded
    expect(getByText('178')).toBeTruthy();
    // kam → 2 decimals + deg
    expect(getByText('1.23°')).toBeTruthy();
    // gos → 2 decimals + deg
    expect(getByText('2.50°')).toBeTruthy();
    // uncertainty 3 decimals
    expect(getByText('0.123')).toBeTruthy();
  });

  it('positions itself at pos.x + 16, pos.y + 16 (cursor offset)', () => {
    const { container } = render(
      <PhaseMapProbeOverlay probe={probe} visible={true} pos={{ x: 200, y: 300 }} />
    );
    const tip = container.querySelector('[data-phasemap-probe-overlay]');
    expect(tip.style.left).toBe('216px');
    expect(tip.style.top).toBe('316px');
  });

  it('omits the Phase row when probe.phase is null', () => {
    const noPhase = { ...probe, phase: null };
    const { queryByText } = render(
      <PhaseMapProbeOverlay probe={noPhase} visible={true} pos={{ x: 0, y: 0 }} />
    );
    expect(queryByText('Al7Cu2Fe')).toBeNull();
    expect(queryByText('Phase')).toBeNull();
  });

  it('omits scalar rows whose value is null', () => {
    const partial = {
      row: 1, col: 2,
      phase: null,
      scalars: { ci: 0.5, bc: null, kam: null, gos: null, uncertainty: null },
    };
    const { queryByText, getByText } = render(
      <PhaseMapProbeOverlay probe={partial} visible={true} pos={{ x: 0, y: 0 }} />
    );
    expect(getByText('ci')).toBeTruthy();
    expect(getByText('0.500')).toBeTruthy();
    expect(queryByText('bc')).toBeNull();
    expect(queryByText('kam')).toBeNull();
    expect(queryByText('gos')).toBeNull();
    expect(queryByText('uncertainty')).toBeNull();
  });

  it('sorts scalar rows alphabetically by key', () => {
    const p = {
      row: 0, col: 0,
      phase: null,
      scalars: { gos: 1.0, bc: 100, ci: 0.5 },
    };
    const { container } = render(
      <PhaseMapProbeOverlay probe={p} visible={true} pos={{ x: 0, y: 0 }} />
    );
    const labels = Array.from(
      container.querySelectorAll('[data-phasemap-probe-row]')
    ).map((el) => el.getAttribute('data-key'));
    // Header is not a scalar row → just the three scalars in alpha order
    expect(labels).toEqual(['bc', 'ci', 'gos']);
  });

  it('shows a Loading… placeholder when visible but probe is null', () => {
    const { getByText, container } = render(
      <PhaseMapProbeOverlay probe={null} visible={true} pos={{ x: 0, y: 0 }} />
    );
    expect(container.querySelector('[data-phasemap-probe-overlay]')).toBeTruthy();
    expect(getByText(/Loading/i)).toBeTruthy();
  });

  it('shows the error message and a red border when error is set', () => {
    const { container, getByText } = render(
      <PhaseMapProbeOverlay probe={null} error="probe failed" visible={true} pos={{ x: 0, y: 0 }} />
    );
    expect(getByText(/probe failed/i)).toBeTruthy();
    const tooltip = container.querySelector('[data-phasemap-probe-overlay]');
    expect(tooltip.style.border).toMatch(/#f00|red|rgb\(255,\s*0,\s*0\)/);
  });

  it('formats per-phase ci_<id> keys with 2 decimals (not 3, since key !== "ci")', () => {
    // The spec says only "ci" and "uncertainty" are 3-dec — everything else
    // (incl. ci_<id>) is 2-dec.
    const p = {
      row: 0, col: 0,
      phase: null,
      scalars: { ci_0: 0.876543, ci_1: 0.123 },
    };
    const { getByText } = render(
      <PhaseMapProbeOverlay probe={p} visible={true} pos={{ x: 0, y: 0 }} />
    );
    expect(getByText('0.88')).toBeTruthy();
    expect(getByText('0.12')).toBeTruthy();
  });
});
