// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import ExportScalePanel from './ExportScalePanel';

afterEach(() => cleanup());

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o?.count != null ? `${k}:${o.count}` : k) }),
}));

vi.mock('../../theme/components', () => ({
  colors: { text: '#fff', textSecondary: '#aaa' },
  spacing: { innerSpacing: 4 },
  CollapsibleGroup: ({ title, children }) => <div><span>{title}</span>{children}</div>,
}));

const ENTRIES = [
  { type: 'colorkey', label: 'IPF colour key' },
  { type: 'valuescale', layerId: 'bc', label: 'Band Contrast' },
  { type: 'valuescale', layerId: 'ci', label: 'CI (Best)' },
];

function setup(annotations = []) {
  const onChange = vi.fn();
  const onSelect = vi.fn();
  render(
    <ExportScalePanel
      entries={ENTRIES}
      annotations={annotations}
      onChange={onChange}
      onSelect={onSelect}
    />,
  );
  return { onChange, onSelect };
}

describe('ExportScalePanel', () => {
  it('lists every scale the figure could carry, all unticked at first', () => {
    setup();
    expect(screen.getByText('IPF colour key')).toBeTruthy();
    expect(screen.getByText('Band Contrast')).toBeTruthy();
    expect(screen.getByText('CI (Best)')).toBeTruthy();
    screen.getAllByRole('checkbox').forEach((cb) => expect(cb.checked).toBe(false));
  });

  it('ticking one puts exactly that body into the figure and selects it', () => {
    const { onChange, onSelect } = setup();
    fireEvent.click(screen.getByText('Band Contrast').closest('label').querySelector('input'));
    const next = onChange.mock.calls[0][0];
    expect(next).toHaveLength(1);
    expect(next[0]).toMatchObject({ type: 'valuescale', props: expect.objectContaining({ layerId: 'bc' }) });
    expect(onSelect).toHaveBeenCalledWith(next[0].id);
  });

  it('the tick reads the body, so one removed elsewhere shows as unticked', () => {
    const body = { id: 'x', type: 'valuescale', x: 0, y: 0, w: 0.1, h: 0.3, props: { layerId: 'ci' } };
    setup([body]);
    const boxes = screen.getAllByRole('checkbox');
    expect(boxes.map((b) => b.checked)).toEqual([false, false, true]);
  });

  it('unticking takes the body out again', () => {
    const body = { id: 'x', type: 'colorkey', x: 0, y: 0, w: 0.3, h: 0.3, props: {} };
    const { onChange } = setup([body]);
    fireEvent.click(screen.getAllByRole('checkbox')[0]);
    expect(onChange.mock.calls[0][0]).toEqual([]);
  });

  it('says so plainly when the map has no scale at all', () => {
    render(<ExportScalePanel entries={[]} annotations={[]} onChange={() => {}} />);
    expect(screen.getByText('phasemap:exportScales.none')).toBeTruthy();
  });
});
