// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { CursorSyncProvider, useCursorSync } from './CursorSyncContext';
import TileGrid from './TileGrid';

vi.mock('../../theme/components', () => ({
  colors: { bgSecondary: '#000', border: '#444', cyan: '#0ff', red: '#f00', text: '#fff', textSecondary: '#aaa' },
}));

function CursorSpy({ onPos }) { useCursorSync(onPos); return null; }
function mockRect(node, rect) {
  vi.spyOn(node, 'getBoundingClientRect').mockReturnValue({
    left: 0, top: 0, right: rect.width, bottom: rect.height,
    width: rect.width, height: rect.height, x: 0, y: 0, toJSON: () => ({}),
  });
}

const sampleLayers = [
  { id: 'bc',         kind: 'bc',         label: 'BC',  visible: true,  opacity: 0.5, blend: 'multiply' },
  { id: 'eds-Al Kα1', kind: 'eds-element', element: 'Al Kα1', label: 'Al Kα1', visible: true, opacity: 0.7, blend: 'screen' },
  { id: 'eds-Fe Kα1', kind: 'eds-element', element: 'Fe Kα1', label: 'Fe Kα1', visible: false, opacity: 0.7, blend: 'screen' },
];

beforeEach(() => { vi.clearAllMocks(); });

describe('TileGrid', () => {
  it('renders one tile per visible layer; hidden layers are skipped', () => {
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={sampleLayers} bitmaps={new Map()} errors={new Map()} shape={[10, 10]} />
      </CursorSyncProvider>
    );
    expect(container.querySelectorAll('[data-tile]')).toHaveLength(2);
  });

  it('shows an empty-state hint when no layers are visible', () => {
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={[]} bitmaps={new Map()} errors={new Map()} shape={[10, 10]} />
      </CursorSyncProvider>
    );
    expect(container.querySelectorAll('[data-tile]')).toHaveLength(0);
    expect(container.textContent).toMatch(/Tick|preset/i);
  });

  it('uses CSS Grid with auto-fit minmax(240px, 1fr)', () => {
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={sampleLayers} bitmaps={new Map()} errors={new Map()} shape={[10, 10]} />
      </CursorSyncProvider>
    );
    const grid = container.querySelector('[data-tile-grid]');
    expect(grid).toBeTruthy();
    // inline style or matchMedia? Inline-style minmax(240px, 1fr) is enough for the assertion.
    expect(grid.style.gridTemplateColumns).toMatch(/auto-fit/);
    expect(grid.style.gridTemplateColumns).toMatch(/240px/);
  });

  it('publishes cursor sync from a tile on mouse move (row/col + screen coords)', () => {
    const cb = vi.fn();
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={sampleLayers.filter(l => l.id === 'bc')} bitmaps={new Map()} errors={new Map()} shape={[100, 100]} />
        <CursorSpy onPos={cb} />
      </CursorSyncProvider>
    );
    const tile = container.querySelector('[data-tile]');
    // Tile owns an inner hostRef (the image area). Find it.
    const host = tile.querySelector('[data-tile-host]');
    mockRect(host, { width: 200, height: 200 });
    cb.mockClear();
    fireEvent.mouseMove(host, { clientX: 100, clientY: 100 });
    expect(cb).toHaveBeenCalledWith(expect.objectContaining({
      row: 50, col: 50, hovering: true, screenX: 100, screenY: 100,
    }));
  });

  it('calls onPixelClick with the underlying image pixel', () => {
    const onPixelClick = vi.fn();
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={sampleLayers.filter(l => l.id === 'bc')} bitmaps={new Map()} errors={new Map()} shape={[100, 100]} onPixelClick={onPixelClick} />
      </CursorSyncProvider>
    );
    const host = container.querySelector('[data-tile-host]');
    mockRect(host, { width: 200, height: 200 });
    fireEvent.click(host, { clientX: 100, clientY: 100 });
    expect(onPixelClick).toHaveBeenCalledWith(50, 50);
  });

  it('renders a crosshair when a sibling publishes a hovering position', () => {
    const visible = sampleLayers.filter(l => l.visible);
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={visible} bitmaps={new Map()} errors={new Map()} shape={[100, 100]} />
      </CursorSyncProvider>
    );
    const host = container.querySelector('[data-tile-host]');
    mockRect(host, { width: 200, height: 200 });
    fireEvent.mouseMove(host, { clientX: 100, clientY: 100 });
    // Every visible tile should now show a crosshair (publish hits all subscribers).
    const crosshairs = container.querySelectorAll('[data-tile-crosshair]');
    expect(crosshairs.length).toBe(2);
  });

  it('shows a red border on the tile when an error is present for that layer', () => {
    const errors = new Map();
    errors.set('bc', 'fetch failed');
    const visible = sampleLayers.filter(l => l.visible);
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={visible} bitmaps={new Map()} errors={errors} shape={[10, 10]} />
      </CursorSyncProvider>
    );
    const bcTile = container.querySelector('[data-tile][data-layer-id="bc"]');
    expect(bcTile).toBeTruthy();
    // Inline style border colour must include the red colour token.
    expect(bcTile.style.border).toMatch(/#f00|red/);
  });
});
