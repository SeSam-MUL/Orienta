// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent } from '@testing-library/react';
import ToolToolbar from './ToolToolbar';

afterEach(() => cleanup());

vi.mock('../../theme/components', () => ({
  colors: {
    bg: '#0b0b0b', bgSecondary: '#000', border: '#444', accent: '#a0f',
    cyan: '#0ff', red: '#f00', green: '#0f0', purple: '#a0f',
    text: '#fff', textSecondary: '#aaa',
  },
}));

vi.mock('../EDS/SwipeCompareController', () => ({
  default: ({ value, onChange, layers }) => (
    <div data-swipe-mock data-layers-count={layers?.length ?? 0}>
      mock-swipe-a={value?.a ?? ''}
      <button onClick={() => onChange?.({ a: 'foo', b: 'bar' })}>
        do-swipe
      </button>
    </div>
  ),
}));

const baseProps = {
  view: 'stack',
  setView: vi.fn(),
  tileMinWidth: 240,
  setTileMinWidth: vi.fn(),
  linescanMode: false,
  setLinescanMode: vi.fn(),
  magnifierEnabled: false,
  setMagnifierEnabled: vi.fn(),
  onExport: vi.fn(),
  swipe: { a: null, b: null, splitX: 0.5 },
  setSwipe: vi.fn(),
  layers: [{ id: 'phase', label: 'Phase', visible: true }],
};

describe('ToolToolbar', () => {
  it('renders all controls when view=stack (no tile-min slider)', () => {
    const { getByText, getByRole, queryByLabelText, container } = render(
      <ToolToolbar {...baseProps} />
    );
    // View toggle: both Stack and Grid options visible.
    expect(getByText('Stack')).toBeTruthy();
    expect(getByText('Grid')).toBeTruthy();
    // Lens + Linescan + Export buttons.
    expect(getByRole('button', { name: /Lens OFF/i })).toBeTruthy();
    expect(getByRole('button', { name: /Linescan OFF/i })).toBeTruthy();
    expect(getByRole('button', { name: /Export PNG/i })).toBeTruthy();
    // Tile-min slider hidden in stack view.
    expect(queryByLabelText(/Minimum tile width/i)).toBeNull();
    // Swipe mock present.
    expect(container.querySelector('[data-swipe-mock]')).toBeTruthy();
  });

  it('shows the tile-min slider only in grid view', () => {
    const { getByLabelText } = render(
      <ToolToolbar {...baseProps} view="grid" />
    );
    const slider = getByLabelText(/Minimum tile width/i);
    expect(slider).toBeTruthy();
    expect(slider.value).toBe('240');
  });

  it('clicking Stack/Grid calls setView with the new value', () => {
    const setView = vi.fn();
    const { getByText } = render(
      <ToolToolbar {...baseProps} setView={setView} view="stack" />
    );
    fireEvent.click(getByText('Grid'));
    expect(setView).toHaveBeenCalledWith('grid');
  });

  it('clicking Linescan/Lens toggles via setters with !current', () => {
    const setLinescan = vi.fn();
    const setLens = vi.fn();
    const { getByRole } = render(
      <ToolToolbar
        {...baseProps}
        setLinescanMode={setLinescan}
        setMagnifierEnabled={setLens}
        linescanMode={false}
        magnifierEnabled={false}
      />
    );
    fireEvent.click(getByRole('button', { name: /Linescan OFF/i }));
    expect(setLinescan).toHaveBeenCalledWith(true);
    fireEvent.click(getByRole('button', { name: /Lens OFF/i }));
    expect(setLens).toHaveBeenCalledWith(true);
  });

  it('clicking Export PNG calls onExport', () => {
    const onExport = vi.fn();
    const { getByRole } = render(
      <ToolToolbar {...baseProps} onExport={onExport} />
    );
    fireEvent.click(getByRole('button', { name: /Export PNG/i }));
    expect(onExport).toHaveBeenCalledTimes(1);
  });

  it('tile slider change calls setTileMinWidth with parsed number', () => {
    const setTile = vi.fn();
    const { getByLabelText } = render(
      <ToolToolbar
        {...baseProps}
        view="grid"
        setTileMinWidth={setTile}
        tileMinWidth={240}
      />
    );
    const slider = getByLabelText(/Minimum tile width/i);
    fireEvent.change(slider, { target: { value: '300' } });
    expect(setTile).toHaveBeenCalledWith(300);
  });

  it('forwards swipe onChange via the mocked SwipeCompareController', () => {
    const setSwipe = vi.fn();
    const { getByText } = render(
      <ToolToolbar {...baseProps} setSwipe={setSwipe} />
    );
    fireEvent.click(getByText('do-swipe'));
    expect(setSwipe).toHaveBeenCalledWith({ a: 'foo', b: 'bar' });
  });
});
