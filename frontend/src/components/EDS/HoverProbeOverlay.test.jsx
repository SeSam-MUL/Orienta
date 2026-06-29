// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import HoverProbeOverlay from './HoverProbeOverlay';

afterEach(() => cleanup());

vi.mock('../../theme/components', () => ({
  colors: { bgSecondary:'#000', border:'#444', cyan:'#0ff', red:'#f00', green:'#0f0', purple:'#a0f', text:'#fff', textSecondary:'#aaa' },
}));

const probe = {
  row: 5, col: 7,
  bc: 178.4,
  elements: {
    'Al': { counts: 4520, wt_pct: 42.1, at_pct: 65.3 },
    'Fe': { counts:  812, wt_pct:  8.3, at_pct: 12.0 },
  },
  phase: { id: 1, name: 'Al7Cu2Fe' },
  display_mode: 'at_pct',
};

describe('HoverProbeOverlay', () => {
  it('renders nothing when visible=false', () => {
    const { container } = render(<HoverProbeOverlay probe={probe} displayMode="at_pct" visible={false} pos={{x:100,y:100}} />);
    expect(container.querySelector('[data-hover-probe-overlay]')).toBeNull();
  });

  it('renders nothing when pos is null', () => {
    const { container } = render(<HoverProbeOverlay probe={probe} displayMode="at_pct" visible={true} pos={null} />);
    expect(container.querySelector('[data-hover-probe-overlay]')).toBeNull();
  });

  it('renders the row/col header + bc + elements + phase when probe is present', () => {
    const { container, getByText } = render(<HoverProbeOverlay probe={probe} displayMode="at_pct" visible={true} pos={{x:100,y:100}} />);
    expect(container.querySelector('[data-hover-probe-overlay]')).toBeTruthy();
    expect(getByText('(5, 7)')).toBeTruthy();
    expect(getByText('178')).toBeTruthy();           // bc rounded
    expect(getByText('Al')).toBeTruthy();
    expect(getByText('65.3%')).toBeTruthy();         // Al at_pct
    expect(getByText('Al7Cu2Fe')).toBeTruthy();      // phase name
  });

  it('reflects displayMode change in element value formatting', () => {
    const { getByText } = render(<HoverProbeOverlay probe={probe} displayMode="counts" visible={true} pos={{x:100,y:100}} />);
    expect(getByText('4,520')).toBeTruthy();         // counts comma-grouped
  });

  it('shows a Loading… placeholder when visible but probe is null', () => {
    const { getByText, container } = render(<HoverProbeOverlay probe={null} displayMode="at_pct" visible={true} pos={{x:100,y:100}} />);
    expect(container.querySelector('[data-hover-probe-overlay]')).toBeTruthy();
    expect(getByText(/Loading/i)).toBeTruthy();
  });

  it('shows the error message and a red border when error is set', () => {
    const { container, getByText } = render(<HoverProbeOverlay probe={null} error="probe failed" displayMode="at_pct" visible={true} pos={{x:100,y:100}} />);
    expect(getByText(/probe failed/i)).toBeTruthy();
    const tooltip = container.querySelector('[data-hover-probe-overlay]');
    expect(tooltip.style.border).toMatch(/#f00|red|rgb\(255,\s*0,\s*0\)/);
  });

  it('positions itself at pos.x + 16, pos.y + 16 (cursor offset)', () => {
    const { container } = render(<HoverProbeOverlay probe={probe} displayMode="at_pct" visible={true} pos={{x:200,y:300}} />);
    const tip = container.querySelector('[data-hover-probe-overlay]');
    expect(tip.style.left).toBe('216px');
    expect(tip.style.top).toBe('316px');
  });

  it('omits the phase row when probe.phase is null', () => {
    const noPhase = { ...probe, phase: null };
    const { queryByText } = render(<HoverProbeOverlay probe={noPhase} displayMode="at_pct" visible={true} pos={{x:0,y:0}} />);
    expect(queryByText('Al7Cu2Fe')).toBeNull();
    expect(queryByText('Phase')).toBeNull();
  });

  it('omits BC row when probe.bc is null', () => {
    const noBC = { ...probe, bc: null };
    const { queryByText } = render(<HoverProbeOverlay probe={noBC} displayMode="at_pct" visible={true} pos={{x:0,y:0}} />);
    expect(queryByText('BC')).toBeNull();
  });
});
