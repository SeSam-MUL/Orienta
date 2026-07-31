// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { CursorSyncProvider } from './CursorSyncContext';
import TileGrid from './TileGrid';
import { useZoomViews, SYNC_ALL, SYNC_SINGLE } from './hooks/useZoomViews';

vi.mock('../../theme/components', () => ({
  colors: {
    bgSecondary: '#000', border: '#444', cyan: '#0ff', red: '#f00',
    text: '#fff', textSecondary: '#aaa', accent: '#8be9fd',
  },
}));

const layers = [
  { id: 'bc', kind: 'bc', label: 'BC', visible: true, opacity: 1, blend: 'normal' },
  { id: 'eds-Al', kind: 'eds-element', element: 'Al', label: 'Al', visible: true, opacity: 1, blend: 'screen' },
];

function mockRect(node, { width, height, left = 0, top = 0 }) {
  vi.spyOn(node, 'getBoundingClientRect').mockReturnValue({
    left, top, right: left + width, bottom: top + height,
    width, height, x: left, y: top, toJSON: () => ({}),
  });
}

/** Real hook + real TileGrid — the wiring is what these tests are about. */
function Harness({ mode = SYNC_ALL, onPixelClick }) {
  const zoom = useZoomViews(mode);
  return (
    <TileGrid
      layers={layers}
      bitmaps={new Map()}
      errors={new Map()}
      shape={[100, 100]}
      zoom={zoom}
      onPixelClick={onPixelClick}
    />
  );
}

function setup(props = {}) {
  const utils = render(
    <CursorSyncProvider><Harness {...props} /></CursorSyncProvider>,
  );
  const hosts = [...utils.container.querySelectorAll('[data-tile-host]')];
  hosts.forEach((h) => mockRect(h, { width: 200, height: 200 }));
  const zoomLayers = () => [...utils.container.querySelectorAll('[data-tile-zoom-layer]')]
    .map((n) => n.style.transform);
  return { ...utils, hosts, zoomLayers };
}

beforeEach(() => { vi.clearAllMocks(); });

describe('TileGrid zoom', () => {
  it('renders unzoomed until asked', () => {
    const { zoomLayers } = setup();
    expect(zoomLayers()).toEqual(['none', 'none']);
  });

  it('ignores a plain wheel so the grid keeps scrolling', () => {
    const { hosts, zoomLayers } = setup();
    fireEvent.wheel(hosts[0], { deltaY: -120, clientX: 100, clientY: 100 });
    expect(zoomLayers()).toEqual(['none', 'none']);
  });

  it('Ctrl+wheel zooms in, and in sync mode every tile follows', () => {
    const { hosts, zoomLayers } = setup({ mode: SYNC_ALL });
    fireEvent.wheel(hosts[0], { deltaY: -120, ctrlKey: true, clientX: 100, clientY: 100 });
    const [a, b] = zoomLayers();
    expect(a).toMatch(/scale\(1\.25\)/);
    expect(b).toBe(a);
  });

  it('single mode zooms only the tile under the cursor', () => {
    const { hosts, zoomLayers } = setup({ mode: SYNC_SINGLE });
    fireEvent.wheel(hosts[1], { deltaY: -120, ctrlKey: true, clientX: 100, clientY: 100 });
    const [a, b] = zoomLayers();
    expect(a).toBe('none');
    expect(b).toMatch(/scale\(1\.25\)/);
  });

  it('Ctrl+wheel down zooms back out and stops at 1×', () => {
    const { hosts, zoomLayers } = setup();
    fireEvent.wheel(hosts[0], { deltaY: -120, ctrlKey: true, clientX: 100, clientY: 100 });
    fireEvent.wheel(hosts[0], { deltaY: 120, ctrlKey: true, clientX: 100, clientY: 100 });
    fireEvent.wheel(hosts[0], { deltaY: 120, ctrlKey: true, clientX: 100, clientY: 100 });
    expect(zoomLayers()).toEqual(['none', 'none']);
  });

  it('shows a zoom badge on the tile once zoomed', () => {
    const { container, hosts } = setup();
    expect(container.querySelector('[data-tile-zoom-badge]')).toBeNull();
    fireEvent.wheel(hosts[0], { deltaY: -120, ctrlKey: true, clientX: 100, clientY: 100 });
    expect(container.querySelector('[data-tile-zoom-badge]').textContent).toMatch(/1\.3×/);
  });

  it('double-click resets the tile', () => {
    const { hosts, zoomLayers } = setup({ mode: SYNC_SINGLE });
    fireEvent.wheel(hosts[0], { deltaY: -120, ctrlKey: true, clientX: 100, clientY: 100 });
    expect(zoomLayers()[0]).toMatch(/scale/);
    fireEvent.doubleClick(hosts[0]);
    expect(zoomLayers()[0]).toBe('none');
  });

  it('click-to-quantify follows the zoom instead of the raw box', () => {
    const onPixelClick = vi.fn();
    const { hosts } = setup({ onPixelClick });
    // Unzoomed: the box centre is the map centre.
    fireEvent.click(hosts[0], { clientX: 100, clientY: 100 });
    expect(onPixelClick).toHaveBeenLastCalledWith(50, 50);

    // Zoom towards the top-left corner, then click the box centre again: it
    // must now report a pixel from that corner region, not (50, 50).
    fireEvent.wheel(hosts[0], { deltaY: -120, ctrlKey: true, clientX: 0, clientY: 0 });
    onPixelClick.mockClear();
    fireEvent.click(hosts[0], { clientX: 100, clientY: 100 });
    const [row, col] = onPixelClick.mock.calls[0];
    expect(row).toBeLessThan(50);
    expect(col).toBeLessThan(50);
  });

  it('panning a zoomed tile moves the view and swallows the trailing click', () => {
    const onPixelClick = vi.fn();
    const { hosts, zoomLayers } = setup({ onPixelClick });
    fireEvent.wheel(hosts[0], { deltaY: -120, ctrlKey: true, clientX: 100, clientY: 100 });
    fireEvent.wheel(hosts[0], { deltaY: -120, ctrlKey: true, clientX: 100, clientY: 100 });
    const before = zoomLayers()[0];

    fireEvent.mouseDown(hosts[0], { clientX: 100, clientY: 100, button: 0, buttons: 1 });
    fireEvent.mouseMove(hosts[0], { clientX: 140, clientY: 100, buttons: 1 });
    fireEvent.mouseUp(hosts[0], { clientX: 140, clientY: 100 });
    fireEvent.click(hosts[0], { clientX: 140, clientY: 100 });

    expect(zoomLayers()[0]).not.toBe(before);      // the view moved
    expect(onPixelClick).not.toHaveBeenCalled();   // ...and it was not a click
  });

  it('a plain click at 1× still quantifies (pan never engages)', () => {
    const onPixelClick = vi.fn();
    const { hosts } = setup({ onPixelClick });
    fireEvent.mouseDown(hosts[0], { clientX: 100, clientY: 100, button: 0, buttons: 1 });
    fireEvent.mouseMove(hosts[0], { clientX: 140, clientY: 100, buttons: 1 });
    fireEvent.mouseUp(hosts[0], { clientX: 140, clientY: 100 });
    fireEvent.click(hosts[0], { clientX: 140, clientY: 100 });
    expect(onPixelClick).toHaveBeenCalledWith(50, 70);
  });
});
