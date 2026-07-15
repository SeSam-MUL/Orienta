// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent } from '@testing-library/react';
import NeighbourhoodZoom from './NeighbourhoodZoom';

vi.mock('../../theme/tokens', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#111', border: '#444',
    text: '#fff', textSecondary: '#aaa', accent: '#0ff',
  },
}));

afterEach(() => cleanup());

const KINDS = [
  { id: 'ipf-z', label: 'Z' },
  { id: 'ipf-y', label: 'Y' },
  { id: 'ipf-x', label: 'X' },
  { id: 'phase', label: 'Phase', sep: true },
];

const base = (over = {}) => ({
  imageB64: null,               // no canvas needed for these tests
  shape: [10, 12],
  pixel: { row: 4, col: 5 },
  onNudge: vi.fn(),
  caption: 'Neighbourhood (±7 px)',
  labels: { layerGroup: 'Zoom map layer', layerLead: 'IPF' },
  ...over,
});

describe('NeighbourhoodZoom', () => {
  it('renders nothing without a selected pixel', () => {
    const { container } = render(
      <NeighbourhoodZoom {...base({ pixel: null })} />);
    expect(container.firstChild).toBeNull();
  });

  it('nudges via the arrow buttons and disables at the grid edge', () => {
    const onNudge = vi.fn();
    const { getByLabelText } = render(<NeighbourhoodZoom {...base({
      onNudge, pixel: { row: 0, col: 5 },
      labels: { up: 'up', down: 'down', left: 'left', right: 'right' },
    })} />);
    expect(getByLabelText('up').disabled).toBe(true);   // row 0 → can't go up
    fireEvent.click(getByLabelText('down'));
    expect(onNudge).toHaveBeenCalledWith(1, 0);
  });

  it('nudges via window arrow keys but NOT while a form control is focused', () => {
    const onNudge = vi.fn();
    render(
      <div>
        <input aria-label="ref" />
        <NeighbourhoodZoom {...base({ onNudge })} />
      </div>
    );
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(onNudge).toHaveBeenCalledWith(0, -1);
    document.querySelector('input').focus();
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(onNudge).toHaveBeenCalledTimes(1); // still just the first call
  });

  it('renders no layer switch when kinds are absent', () => {
    const { queryByRole } = render(<NeighbourhoodZoom {...base()} />);
    expect(queryByRole('radiogroup')).toBeNull();
  });

  it('renders the layer radiogroup, marks the active kind, switches on click', () => {
    const onKindChange = vi.fn();
    const { getByRole, getAllByRole } = render(<NeighbourhoodZoom {...base({
      kind: 'ipf-z', kinds: KINDS, onKindChange,
    })} />);
    const group = getByRole('radiogroup');
    expect(group.getAttribute('aria-label')).toBe('Zoom map layer');
    const radios = getAllByRole('radio');
    expect(radios).toHaveLength(4);
    expect(radios[0].getAttribute('aria-checked')).toBe('true');   // Z active
    expect(radios[3].getAttribute('aria-checked')).toBe('false');
    fireEvent.click(radios[3]);
    expect(onKindChange).toHaveBeenCalledWith('phase');
  });

  it('does not fire the window-key nudge while a layer radio is focused', () => {
    const onNudge = vi.fn();
    const { getAllByRole } = render(<NeighbourhoodZoom {...base({
      onNudge, kind: 'ipf-z', kinds: KINDS, onKindChange: vi.fn(),
    })} />);
    getAllByRole('radio')[1].focus();
    fireEvent.keyDown(window, { key: 'ArrowUp' });
    expect(onNudge).not.toHaveBeenCalled();
  });
});
