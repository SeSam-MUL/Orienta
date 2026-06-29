// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { CursorSyncProvider, useCursorSync, useCursorPublisher } from './CursorSyncContext';
import OverlayCard from './OverlayCard';

// Stub LayeredCanvas — we don't want PhaseMap's compositor pulling in
// canvas-context code during a JSDOM unit test. We just need to verify
// OverlayCard's wrapper behaviour (pointer events, crosshair, onPixelClick).
vi.mock('../PhaseMap/LayeredCanvas', () => ({
  default: ({ layers, bitmapVersion }) => (
    <div data-testid="composite" data-bitmap-version={String(bitmapVersion)}>
      layers:{layers.length}
    </div>
  ),
}));

// Stub theme module — OverlayCard pulls colors, the test environment
// doesn't need to load the real CSS-in-JS stack.
vi.mock('../../theme/components', () => ({
  colors: { bgSecondary: '#000', border: '#444', cyan: '#0ff' },
  spacing: { outerSpacing: 8 },
}));

function CursorSpy({ onPos }) {
  useCursorSync(onPos);
  return null;
}

function mockRect(node, rect) {
  vi.spyOn(node, 'getBoundingClientRect').mockReturnValue({
    left: 0, top: 0, right: rect.width, bottom: rect.height,
    width: rect.width, height: rect.height,
    x: 0, y: 0, toJSON: () => ({}),
  });
}

beforeEach(() => { vi.clearAllMocks(); });

describe('OverlayCard', () => {
  it('renders the LayeredCanvas stub with passed layers', () => {
    const { getByTestId } = render(
      <CursorSyncProvider>
        <OverlayCard layers={[{ id: 'bc', visible: true, opacity: 0.5, blend: 'multiply' }]} bitmaps={new Map()} errors={new Map()} shape={[100, 100]} />
      </CursorSyncProvider>
    );
    expect(getByTestId('composite').textContent).toBe('layers:1');
  });

  it('forwards bitmapVersion to LayeredCanvas so the composite repaints when bitmaps arrive', () => {
    // Regression: OverlayCard previously rendered <LayeredCanvas> without
    // bitmapVersion. LayeredCanvas memoises nativeSize on [layers, bitmapVersion];
    // with the version frozen at its default it never recomputed once bitmaps
    // landed → the composite was stuck on "No layers loaded".
    const { container, rerender } = render(
      <CursorSyncProvider>
        <OverlayCard layers={[{ id: 'bc', visible: true }]} bitmaps={new Map()} bitmapVersion={0} errors={new Map()} shape={[100, 100]} />
      </CursorSyncProvider>
    );
    expect(container.querySelector('[data-testid="composite"]').dataset.bitmapVersion).toBe('0');
    rerender(
      <CursorSyncProvider>
        <OverlayCard layers={[{ id: 'bc', visible: true }]} bitmaps={new Map()} bitmapVersion={3} errors={new Map()} shape={[100, 100]} />
      </CursorSyncProvider>
    );
    expect(container.querySelector('[data-testid="composite"]').dataset.bitmapVersion).toBe('3');
  });

  it('publishes (row, col, hovering, screenX, screenY) on mouse move', () => {
    const cb = vi.fn();
    const { container } = render(
      <CursorSyncProvider>
        <OverlayCard layers={[]} bitmaps={new Map()} errors={new Map()} shape={[100, 100]} />
        <CursorSpy onPos={cb} />
      </CursorSyncProvider>
    );
    const host = container.querySelector('[data-overlay-card-host]');
    mockRect(host, { width: 200, height: 200 });
    cb.mockClear();
    fireEvent.mouseMove(host, { clientX: 100, clientY: 100 });
    expect(cb).toHaveBeenCalledWith(expect.objectContaining({
      row: 50, col: 50, hovering: true, screenX: 100, screenY: 100,
    }));
  });

  it('publishes hovering=false on mouse leave', () => {
    const cb = vi.fn();
    const { container } = render(
      <CursorSyncProvider>
        <OverlayCard layers={[]} bitmaps={new Map()} errors={new Map()} shape={[100, 100]} />
        <CursorSpy onPos={cb} />
      </CursorSyncProvider>
    );
    const host = container.querySelector('[data-overlay-card-host]');
    mockRect(host, { width: 200, height: 200 });
    cb.mockClear();
    fireEvent.mouseLeave(host);
    expect(cb).toHaveBeenCalledWith(expect.objectContaining({ hovering: false }));
  });

  it('calls onPixelClick with the underlying image pixel', () => {
    const onPixelClick = vi.fn();
    const { container } = render(
      <CursorSyncProvider>
        <OverlayCard layers={[]} bitmaps={new Map()} errors={new Map()} shape={[100, 100]} onPixelClick={onPixelClick} />
      </CursorSyncProvider>
    );
    const host = container.querySelector('[data-overlay-card-host]');
    mockRect(host, { width: 200, height: 200 });
    fireEvent.click(host, { clientX: 100, clientY: 100 });
    expect(onPixelClick).toHaveBeenCalledWith(50, 50);
  });

  it('does not crash when shape is null (pre-load)', () => {
    const cb = vi.fn();
    const { container } = render(
      <CursorSyncProvider>
        <OverlayCard layers={[]} bitmaps={new Map()} errors={new Map()} shape={null} />
        <CursorSpy onPos={cb} />
      </CursorSyncProvider>
    );
    const host = container.querySelector('[data-overlay-card-host]');
    mockRect(host, { width: 200, height: 200 });
    fireEvent.mouseMove(host, { clientX: 50, clientY: 50 });
    fireEvent.mouseLeave(host);
    // shape=null → BOTH publishing paths are no-ops. Initial subscribe replay
    // fired ONE call with the initial state; no further calls.
    expect(cb).toHaveBeenCalledTimes(1);
    expect(cb).toHaveBeenLastCalledWith(expect.objectContaining({ hovering: false }));
  });

  it('renders a crosshair when a sibling publishes a hovering position', () => {
    function Publisher() {
      const pub = useCursorPublisher();
      // imperatively publish a position once on mount
      pub({ row: 50, col: 50, hovering: true, screenX: 0, screenY: 0 });
      return null;
    }
    const { container } = render(
      <CursorSyncProvider>
        <OverlayCard layers={[]} bitmaps={new Map()} errors={new Map()} shape={[100, 100]} />
        <Publisher />
      </CursorSyncProvider>
    );
    const host = container.querySelector('[data-overlay-card-host]');
    mockRect(host, { width: 200, height: 200 });
    // Crosshair is rendered after a render pass following the publish.
    // Trigger a re-render of OverlayCard by simulating a mouse move on it.
    fireEvent.mouseMove(host, { clientX: 0, clientY: 0 });
    const ch = host.querySelector('[data-overlay-crosshair]');
    expect(ch).toBeTruthy();
  });
});
