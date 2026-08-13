// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import AddLayerPicker, { matchesQuery, partitionOptions } from './AddLayerPicker';

afterEach(() => cleanup());

vi.mock('../../theme/components', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#111', bgTertiary: '#222', border: '#444',
    text: '#fff', textSecondary: '#aaa',
  },
}));

const OPTIONS = [
  { value: 'phase', label: 'Indexing Result: Phase Map', group: 'Indexing Result', name: 'Phase Map', disabled: false },
  { value: 'ci', label: 'Indexing Result: CI (Best)', group: 'Indexing Result', name: 'CI (Best)', disabled: false },
  { value: 'eds:Fe', label: 'H5OINA Source: EDS: Fe', group: 'H5OINA Source', name: 'EDS: Fe', disabled: false },
  {
    value: 'forward-ncc', label: 'Forward Diagnostics: Forward NCC',
    group: 'Forward Diagnostics', name: 'Forward NCC',
    disabled: true, tip: "Run 'Compute Diagnostics' first",
  },
];

function open(props = {}) {
  const onAdd = vi.fn();
  render(<AddLayerPicker options={OPTIONS} onAdd={onAdd} {...props} />);
  fireEvent.click(screen.getByText('+ Add Layer'));
  return onAdd;
}

const box = () => screen.getByRole('checkbox');
const search = () => screen.getByPlaceholderText('Search layers…');

describe('AddLayerPicker', () => {
  it('offers only what can actually be drawn, and says how many it is holding back', () => {
    open();
    expect(screen.getByText('Phase Map')).toBeTruthy();
    expect(screen.queryByText(/Forward NCC/)).toBeNull();
    // The count is the point: "where did Forward NCC go?" answers itself.
    expect(screen.getByText(/Show unavailable \(1\)/)).toBeTruthy();
  });

  it('reveals the gated layers with the reason when asked', () => {
    open();
    fireEvent.click(box());
    const gated = screen.getByText(/Forward NCC/);
    expect(gated).toBeTruthy();
    expect(gated.textContent).toContain("Run 'Compute Diagnostics' first");
  });

  it('still lets a revealed layer be added — add first, compute after', () => {
    const onAdd = open();
    fireEvent.click(box());
    fireEvent.click(screen.getByText(/Forward NCC/));
    expect(onAdd).toHaveBeenCalledWith('forward-ncc');
  });

  it('searches by layer name and by group', () => {
    open();
    fireEvent.change(search(), { target: { value: 'ci' } });
    expect(screen.getByText('CI (Best)')).toBeTruthy();
    expect(screen.queryByText('Phase Map')).toBeNull();

    fireEvent.change(search(), { target: { value: 'h5oina' } });
    expect(screen.getByText('EDS: Fe')).toBeTruthy();
    expect(screen.queryByText('CI (Best)')).toBeNull();
  });

  it('says so when nothing matches instead of showing an empty box', () => {
    open();
    fireEvent.change(search(), { target: { value: 'zzz' } });
    expect(screen.getByText('No layer matches')).toBeTruthy();
  });

  it('adds the first match on Enter and closes', () => {
    const onAdd = open();
    fireEvent.change(search(), { target: { value: 'eds fe' } });
    fireEvent.keyDown(search(), { key: 'Enter' });
    expect(onAdd).toHaveBeenCalledWith('eds:Fe');
    expect(screen.queryByPlaceholderText('Search layers…')).toBeNull();
  });

  it('closes on Escape without adding anything', () => {
    const onAdd = open();
    fireEvent.keyDown(search(), { key: 'Escape' });
    expect(onAdd).not.toHaveBeenCalled();
    expect(screen.queryByPlaceholderText('Search layers…')).toBeNull();
  });

  it('shows the limit instead of opening once the stack is full', () => {
    render(<AddLayerPicker options={OPTIONS} onAdd={vi.fn()} disabled disabledLabel="Limit 8" />);
    const btn = screen.getByText('Limit 8');
    fireEvent.click(btn);
    expect(screen.queryByPlaceholderText('Search layers…')).toBeNull();
  });
});

describe('matchesQuery', () => {
  const o = OPTIONS[2];
  it('is case-insensitive and word-order-free', () => {
    expect(matchesQuery(o, 'fe')).toBe(true);
    expect(matchesQuery(o, 'FE')).toBe(true);
    expect(matchesQuery(o, 'fe h5oina')).toBe(true);   // reversed order
    expect(matchesQuery(o, 'fe kam')).toBe(false);      // every word must hit
  });
  it('matches everything on an empty query', () => {
    expect(matchesQuery(o, '')).toBe(true);
    expect(matchesQuery(o, '   ')).toBe(true);
  });
});

describe('partitionOptions', () => {
  it('counts what it hides so the checkbox can be honest', () => {
    const hidden = partitionOptions(OPTIONS, '', false);
    expect(hidden.visible.map((o) => o.value)).toEqual(['phase', 'ci', 'eds:Fe']);
    expect(hidden.hiddenCount).toBe(1);

    const shown = partitionOptions(OPTIONS, '', true);
    expect(shown.visible).toHaveLength(4);
    expect(shown.hiddenCount).toBe(0);
  });

  it('keeps unavailable ones last so the usable ones are on top', () => {
    const { visible } = partitionOptions(OPTIONS, '', true);
    expect(visible[visible.length - 1].value).toBe('forward-ncc');
  });

  it('counts gated entries across the whole list, not just the current search', () => {
    const { totalGated, visible } = partitionOptions(OPTIONS, 'phase', false);
    expect(visible.map((o) => o.value)).toEqual(['phase']);
    expect(totalGated).toBe(1);
  });
});
