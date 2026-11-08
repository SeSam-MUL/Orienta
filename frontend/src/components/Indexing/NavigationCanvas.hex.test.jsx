// @vitest-environment jsdom
/**
 * The Indexing page's navigation map on a hexagonal scan, measured through the real component: a drag on the
 * display raster becomes a region of index cells, the region is framed where its hexagons are, and a square scan
 * keeps the arithmetic it always had. The canvas context is a recorder, the image a stub of the picture's size.
 *
 * Expectations are worked by hand below or come from tests/fixtures/hex_raster_cases.json (an independent oracle).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, fireEvent, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { cwd } from 'node:process';
import NavigationCanvas from './NavigationCanvas';
import { makeHexMap } from '../../services/hexMap';
import { parseScanGeometry } from '../../services/scanGeometry';

// vitest runs from frontend/ (import.meta.url is not a file: URL in the jsdom environment)
const FIXTURE = JSON.parse(
  readFileSync(resolve(cwd(), '..', 'tests', 'fixtures', 'hex_raster_cases.json'), 'utf8'));
const CASE = FIXTURE.geometries.find((c) => c.name === 'a_3x3_right');       // a 12 x 14 raster
const hexMap = makeHexMap(parseScanGeometry(CASE.geometry));

const calls = [];                       // every canvas call: [name, ...args]
let picture = { width: 14, height: 12 };

beforeEach(() => {
  calls.length = 0;
  picture = { width: 14, height: 12 };
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  vi.stubGlobal('Image', class {
    set src(_v) { setTimeout(() => { this.width = picture.width; this.height = picture.height; this.onload?.(); }, 0); }
  });
  HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
    get: (_t, name) => {
      if (name === 'measureText') return () => ({ width: 10 });
      if (name === 'createImageData') return (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), w, h });
      return (...args) => { calls.push([name, ...args]); };
    },
    set: () => true,
  });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

// The canvas is 400 x 300 (the component's default). A 14 x 12 picture is drawn at scale 25 with its left edge at
// x = 25, a 3 x 3 one at scale 100 with its left edge at x = 50.
const mount = async (props) => {
  const utils = render(
    <NavigationCanvas
      navImage="Zm9v" nRows={3} nCols={3} selectionMode="region"
      region={{ rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0 }} onRegionChange={vi.fn()} {...props}
    />);
  const canvas = utils.container.querySelector('canvas');
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 400, height: 300, right: 400, bottom: 300 });
  await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
  return { ...utils, canvas };
};
const hexAt = (i, j) => ({ clientX: 25 + (j + 0.5) * 25, clientY: (i + 0.5) * 25 });

describe('a drag on a hexagonal navigation map', () => {
  it('selects the box of the cells whose centres it contains', async () => {
    const onRegionChange = vi.fn();
    const { canvas } = await mount({ hexMap, onRegionChange });
    fireEvent.mouseDown(canvas, hexAt(3, 3));
    fireEvent.mouseUp(canvas, hexAt(9, 9));
    // raster pixels (3, 3)..(9, 9): x in [0.5, 4.0), y in [0.345, 3.845) um -> cells (1,0), (1,1), (2,1)
    expect(onRegionChange).toHaveBeenCalledWith({ rowStart: 1, rowEnd: 3, colStart: 0, colEnd: 2 });
  });

  it('selects nothing for a drag that ends outside the picture', async () => {
    const onRegionChange = vi.fn();
    const { canvas } = await mount({ hexMap, onRegionChange });
    fireEvent.mouseDown(canvas, hexAt(3, 3));
    fireEvent.mouseUp(canvas, { clientX: 5, clientY: 100 });                // left of the picture
    expect(onRegionChange).not.toHaveBeenCalled();
  });

  it('selects nothing for a click on a corner no hexagon covers', async () => {
    const corner = CASE.pixels.find((p) => p.cell === null);
    const onRegionChange = vi.fn();
    const { canvas } = await mount({ hexMap, onRegionChange });
    fireEvent.mouseDown(canvas, hexAt(corner.i, corner.j));
    fireEvent.mouseUp(canvas, hexAt(corner.i, corner.j));
    expect(onRegionChange).not.toHaveBeenCalled();
  });

  it('frames the typed region where its hexagons are', async () => {
    await mount({
      hexMap, region: { rowStart: 1, rowEnd: 3, colStart: 0, colEnd: 2 },
    });
    // cells (1,0)..(2,1) are the raster box x 0..10, y 3..12 of 14 x 12: canvas x 25..275, y 75..300
    const frame = calls.filter((c) => c[0] === 'strokeRect').pop();
    expect(frame.slice(1)).toEqual([25, 75, 250, 225]);
  });

  it('frames a saved region where its hexagons are', async () => {
    await mount({ hexMap, savedRegions: [{ rowStart: 0, rowEnd: 1, colStart: 0, colEnd: 2 }] });
    // cells (0,0)..(0,1) are the raster box x 0..8, y 0..5 of 14 x 12: canvas x 25..225, y 0..125
    const frame = calls.filter((c) => c[0] === 'strokeRect').pop();
    expect(frame.slice(1)).toEqual([25, 0, 200, 125]);
  });

  it('draws no frame for an empty saved region, and still frames the others', async () => {
    const empty = { rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0 };
    await mount({ hexMap, savedRegions: [empty] });
    expect(calls.filter((c) => c[0] === 'strokeRect')).toHaveLength(0);
    calls.length = 0;
    await mount({ hexMap, savedRegions: [empty, { rowStart: 0, rowEnd: 1, colStart: 0, colEnd: 2 }] });
    const frames = calls.filter((c) => c[0] === 'strokeRect');
    expect(frames.length).toBeGreaterThan(0);
    expect(frames.every((c) => c.slice(1).join() === [25, 0, 200, 125].join())).toBe(true);   // only the real one
  });

  it('draws no EDS layer over a picture that is not the index grid', async () => {
    const layer = { image: { width: 3, height: 3 }, opacity: 0.5 };
    await mount({ hexMap, edsLayers: [layer] });
    const drawn = calls.filter((c) => c[0] === 'drawImage');
    expect(drawn).toHaveLength(1);                                          // the picture itself
  });

  it('draws no EDS layer inside the dimmed region either', async () => {
    const layer = { image: { width: 3, height: 3 }, opacity: 0.5 };
    await mount({ hexMap, edsLayers: [layer], region: { rowStart: 1, rowEnd: 3, colStart: 0, colEnd: 2 } });
    const drawn = calls.filter((c) => c[0] === 'drawImage');
    expect(drawn.length).toBeGreaterThan(1);                                // the picture, and again inside the region
    expect(drawn.some((c) => c[1] === layer.image)).toBe(false);
  });
});

describe('the chemistry mask on a hexagonal navigation map', () => {
  it('hatches the excluded cells, leaves the selected ones clear and does not touch the corners', async () => {
    const maskData = { mask2d: [1, 0, 0, 0, 1, 0, 0, 0, 1], shape: [3, 3] };        // cells (0,0), (1,1), (2,2)
    await mount({ hexMap, selectionMode: 'mask', maskData });
    // the first layer of the overlay marks the EXCLUDED pixels; its alpha is zero everywhere else
    const hatch = calls.find((c) => c[0] === 'putImageData')[1].data;
    expect(hatch.length).toBe(12 * 14 * 4);                                          // the raster, not the 3 x 3 grid
    let excluded = 0;
    let corners = 0;
    for (const p of CASE.pixels) {
      const alpha = hatch[(p.i * 14 + p.j) * 4 + 3];
      if (p.cell === null) {
        expect(alpha).toBe(0);                                                        // no hexagon: not part of the scan
        corners += 1;
      } else if (maskData.mask2d[p.cell[0] * 3 + p.cell[1]]) {
        expect(alpha).toBe(0);                                                        // selected
      } else {
        expect(alpha).toBeGreaterThan(0);                                             // excluded
        excluded += 1;
      }
    }
    expect(excluded).toBeGreaterThan(0);                                              // both kinds were checked
    expect(corners).toBeGreaterThan(0);
  });

  // the third layer of the overlay is the cyan border at the edge of the selection
  const border = () => {
    const puts = calls.filter((c) => c[0] === 'putImageData');
    return puts[puts.length - 1][1].data;
  };

  it('draws no border along the zigzag edge of a fully selected scan', async () => {
    await mount({ hexMap, selectionMode: 'mask', maskData: { mask2d: [1, 1, 1, 1, 1, 1, 1, 1, 1], shape: [3, 3] } });
    // every excluded neighbour of a selected pixel would be a corner no hexagon covers, which is not excluded
    expect(border().length).toBe(12 * 14 * 4);
    expect(Array.from(border()).filter((v, k) => k % 4 === 3 && v > 0)).toHaveLength(0);
  });

  it('draws the border where a selected cell meets an excluded one, and only on selected pixels', async () => {
    const maskData = { mask2d: [0, 0, 0, 0, 1, 0, 0, 0, 0], shape: [3, 3] };         // the middle cell alone
    await mount({ hexMap, selectionMode: 'mask', maskData });
    const data = border();
    let bordered = 0;
    for (const p of CASE.pixels) {
      const alpha = data[(p.i * 14 + p.j) * 4 + 3];
      if (alpha > 0) {
        bordered += 1;
        expect(p.cell).toEqual([1, 1]);                                               // never outside the selection
      }
    }
    expect(bordered).toBeGreaterThan(0);
  });
});

describe('a drag on a square navigation map', () => {
  it('is the arithmetic it always was', async () => {
    picture = { width: 3, height: 3 };
    const onRegionChange = vi.fn();
    const { canvas } = await mount({ hexMap: null, onRegionChange });
    fireEvent.mouseDown(canvas, { clientX: 60, clientY: 10 });              // cell (0, 0)
    fireEvent.mouseUp(canvas, { clientX: 340, clientY: 290 });              // cell (2, 2)
    expect(onRegionChange).toHaveBeenCalledWith({ rowStart: 0, rowEnd: 3, colStart: 0, colEnd: 3 });
  });

  it('frames an empty saved region as it always did', async () => {
    picture = { width: 3, height: 3 };
    await mount({ hexMap: null, savedRegions: [{ rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0 }] });
    expect(calls.filter((c) => c[0] === 'strokeRect').pop().slice(1)).toEqual([50, 0, 0, 0]);
  });

  it('draws the border at the edge of a selection on a square scan', async () => {
    picture = { width: 3, height: 3 };
    await mount({ hexMap: null, selectionMode: 'mask', maskData: { mask2d: [1, 1, 0, 1, 1, 0, 1, 1, 0], shape: [3, 3] } });
    const puts = calls.filter((c) => c[0] === 'putImageData');
    const data = puts[puts.length - 1][1].data;
    // the two selected columns border the excluded one: column 1 (alpha set), column 0 (none)
    expect([0, 3, 6].map((i) => data[(i * 4) + 3])).toEqual([0, 0, 0]);
    expect([1, 4, 7].map((i) => data[(i * 4) + 3] > 0)).toEqual([true, true, true]);
  });

  it('frames the typed region between its grid lines', async () => {
    picture = { width: 3, height: 3 };
    await mount({ hexMap: null, region: { rowStart: 1, rowEnd: 3, colStart: 0, colEnd: 2 } });
    const frame = calls.filter((c) => c[0] === 'strokeRect').pop();
    expect(frame.slice(1)).toEqual([50, 100, 200, 200]);
  });

  it('still draws an EDS layer over the grid picture', async () => {
    picture = { width: 3, height: 3 };
    const layer = { image: { width: 3, height: 3 }, opacity: 0.5 };
    await mount({ hexMap: null, edsLayers: [layer] });
    expect(calls.filter((c) => c[0] === 'drawImage')).toHaveLength(2);      // the picture and the layer
  });

  it('still draws an EDS layer inside the dimmed region', async () => {
    picture = { width: 3, height: 3 };
    const layer = { image: { width: 3, height: 3 }, opacity: 0.5 };
    await mount({ hexMap: null, edsLayers: [layer], region: { rowStart: 1, rowEnd: 3, colStart: 0, colEnd: 2 } });
    // twice: over the picture, and again inside the clipped region
    expect(calls.filter((c) => c[0] === 'drawImage' && c[1] === layer.image)).toHaveLength(2);
  });
});
