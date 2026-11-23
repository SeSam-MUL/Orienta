// @vitest-environment jsdom
/**
 * Tilt correction of the EDS display: a tile takes the sample's true aspect
 * ratio, and a click still lands on the same map pixel.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { CursorSyncProvider } from './CursorSyncContext';
import TileGrid from './TileGrid';
import { stretchFor } from './tiltStretch';

vi.mock('../../theme/components', () => ({
  colors: { bgSecondary: '#000', border: '#444', cyan: '#0ff', red: '#f00', text: '#fff', textSecondary: '#aaa' },
}));

const layers = [
  { id: 'eds-Al', kind: 'eds-element', element: 'Al', label: 'Al', visible: true, opacity: 1 },
];
const AB5 = { befund: 'X gestaucht', faktor_korrektur: 2.92, karten: [{ typ: 'EDS', erbt_bildverzerrung: true }] };

function aspectOf(container) {
  const tile = container.querySelector('[data-tile-host]');
  // jsdom may drop the property it does not know; the attribute keeps what React wrote.
  return tile.style.aspectRatio || (tile.getAttribute('style') || '').match(/aspect-ratio:\s*([^;]+)/)?.[1]?.trim();
}

describe('TileGrid with the tilt correction', () => {
  it('draws a 12 × 9 map 2.92× wider when X is foreshortened', () => {
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={layers} bitmaps={new Map()} errors={new Map()} shape={[9, 12]}
          stretchFor={(l) => stretchFor(AB5, l.kind)} />
      </CursorSyncProvider>,
    );
    expect(aspectOf(container)).toBe('35.04/9');
  });

  it('keeps the stored aspect without a finding or with the switch off', () => {
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={layers} bitmaps={new Map()} errors={new Map()} shape={[9, 12]}
          stretchFor={(l) => stretchFor(AB5, l.kind, false)} />
      </CursorSyncProvider>,
    );
    expect(aspectOf(container)).toBe('12/9');
  });

  it('maps a click on the stretched tile to the same pixel', () => {
    const onPixelClick = vi.fn();
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid layers={layers} bitmaps={new Map()} errors={new Map()} shape={[9, 12]}
          onPixelClick={onPixelClick} stretchFor={(l) => stretchFor(AB5, l.kind)} />
      </CursorSyncProvider>,
    );
    const tile = container.querySelector('[data-tile-host]');
    // The stretched box: 350.4 px wide for 90 px high (12 · 2.92 : 9).
    vi.spyOn(tile, 'getBoundingClientRect').mockReturnValue({
      left: 0, top: 0, right: 350.4, bottom: 90, width: 350.4, height: 90, x: 0, y: 0, toJSON: () => ({}),
    });
    // Three quarters across, a third down -> column 9, row 3 of the 12 × 9 map.
    fireEvent.click(tile, { clientX: 350.4 * 0.79, clientY: 90 * 0.37 });
    expect(onPixelClick).toHaveBeenCalledWith(3, 9);
  });
});
