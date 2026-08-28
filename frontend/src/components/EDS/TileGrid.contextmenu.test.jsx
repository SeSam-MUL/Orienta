// @vitest-environment jsdom
/**
 * A right-click on a tile belongs to that tile.
 *
 * The phase map's grid sits inside a host that also handles `contextmenu`, and
 * reports `layer: null` — "the composite was hit", which is the right answer in
 * the STACKED view where there are no tiles. React events bubble, so on the
 * grid the tile's handler ran with its layer and the host's then overwrote it
 * with null. Measured in the running app on 2026-08-27: right-clicking the
 * `se:SE/Elektronenbild 1` tile offered "export the composed map", and the
 * per-layer export was unreachable — which is exactly the export whose scale
 * this work had just fixed.
 */
import { describe, it, expect, vi, beforeAll, afterAll } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { CursorSyncProvider } from './CursorSyncContext';
import TileGrid from './TileGrid';

vi.mock('../../theme/components', () => ({
  colors: { bgSecondary: '#000', border: '#444', cyan: '#0ff', red: '#f00', text: '#fff', textSecondary: '#aaa' },
}));

// jsdom has no 2d context; the tiles only need the drawing calls swallowed.
let realGetContext;
beforeAll(() => {
  realGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function fake(type) {
    if (type !== '2d') return null;
    return {
      canvas: this, globalAlpha: 1, globalCompositeOperation: 'source-over',
      imageSmoothingEnabled: true, fillStyle: '#000',
      clearRect() {}, fillRect() {}, drawImage() {}, save() {}, restore() {},
      getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
      putImageData() {},
    };
  };
});
afterAll(() => { HTMLCanvasElement.prototype.getContext = realGetContext; });

const layers = [
  { id: 'phase', kind: 'phase', label: 'Phase Map', visible: true, opacity: 1, blend: 'normal' },
  { id: 'se:SE/Elektronenbild 1', kind: 'electron', label: 'SE', visible: true, opacity: 1, blend: 'normal' },
];
const bitmaps = new Map([
  ['phase', { width: 120, height: 90 }],
  ['se:SE/Elektronenbild 1', { width: 1024, height: 768 }],
]);

describe('a tile owns its own context menu', () => {
  it('does not let an enclosing host overwrite the hit with "no layer"', () => {
    const onTile = vi.fn();
    const onHost = vi.fn();
    const { container } = render(
      <CursorSyncProvider>
        <div onContextMenu={onHost}>
          <TileGrid
            layers={layers}
            bitmaps={bitmaps}
            errors={new Map()}
            shape={[90, 120]}
            onTileContextMenu={(layer, x, y) => onTile(layer.id, x, y)}
          />
        </div>
      </CursorSyncProvider>,
    );
    const tiles = container.querySelectorAll('canvas');
    expect(tiles.length).toBeGreaterThanOrEqual(2);

    fireEvent.contextMenu(tiles[1], { clientX: 40, clientY: 50 });
    expect(onTile).toHaveBeenCalledWith('se:SE/Elektronenbild 1', 40, 50);
    // The host must not hear it: the tile is the more specific target, and the
    // host's answer ("the composite") would land second and win.
    expect(onHost).not.toHaveBeenCalled();
  });

  it('still reports the first tile for a click on the first tile', () => {
    const onTile = vi.fn();
    const { container } = render(
      <CursorSyncProvider>
        <TileGrid
          layers={layers}
          bitmaps={bitmaps}
          errors={new Map()}
          shape={[90, 120]}
          onTileContextMenu={(layer) => onTile(layer.id)}
        />
      </CursorSyncProvider>,
    );
    fireEvent.contextMenu(container.querySelectorAll('canvas')[0], { clientX: 1, clientY: 1 });
    expect(onTile).toHaveBeenCalledWith('phase');
  });
});
