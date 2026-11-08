// @vitest-environment jsdom
/**
 * The pointer, the hover ring and the indexed-region frame on a hexagonal phase map, mounted.
 *
 *  - the pointer in a host whose aspect differs from the map's (a letterbox), zoomed in and panned;
 *  - the hover ring of a cell sits on the centre of that cell, also at the coarse factors of a large scan,
 *    where the pixel that holds the centre can lie half a step away;
 *  - the frame of an indexed region is drawn in the pixels of the raster, not in cells of the index grid.
 *
 * Every expectation is worked by hand in the comments (or comes from tests/fixtures/hex_raster_cases.json,
 * written by an independent oracle); none is computed with the code under test.
 */
/* global process */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const listResults = vi.fn();
const canvasProps = vi.hoisted(() => ({ last: null }));
const drawerProps = vi.hoisted(() => ({ last: null }));
// the size of the picture every layer answer is decoded to
const bitmap = vi.hoisted(() => ({ width: 14, height: 12 }));

vi.mock('../../services/api', () => {
  const ok = (data) => Promise.resolve({ data });
  const anyApi = (overrides = {}) => new Proxy({ ...overrides }, {
    get: (target, prop) => {
      if (prop in target) return target[prop];
      if (typeof prop !== 'string') return undefined;
      target[prop] = vi.fn(() => ok({}));
      return target[prop];
    },
  });
  return {
    default: anyApi(),
    indexApi: anyApi({ listResults: (...a) => listResults(...a) }),
    phaseMapApi: anyApi({
      layer: vi.fn(() => ok({ image: 'Zm9v', scale: null })),
      render: vi.fn(() => ok({ image: 'Zm9v' })),
    }),
    analysisApi: anyApi(), ebsdApi: anyApi(), h5Api: anyApi(), pcApi: anyApi(),
    forwardDiagApi: anyApi({ summary: vi.fn(() => Promise.reject(new Error('404'))) }),
    refinementApi: anyApi({ summary: vi.fn(() => Promise.reject(new Error('404'))) }),
    citationsApi: anyApi({ forResult: vi.fn(() => ok({ steps: [], bibtex: '' })) }),
  };
});
vi.mock('../../services/addonsApi', () => ({
  addonsApi: { mapImage: vi.fn(() => Promise.resolve({ data: { image: '', scale: null } })) },
}));
vi.mock('./LayeredCanvas', () => ({
  default: (props) => { canvasProps.last = props; return <div data-testid="canvas" />; },
}));
vi.mock('../EDS/LinescanProfilePlot', () => ({ default: () => null }));
vi.mock('../EDS/MagnifierLens', () => ({ default: () => null }));
vi.mock('../PatternMatch/LinkedPatternImage', () => ({ default: () => null }));
vi.mock('../PoleFigure/openPoleFigureWindow', () => ({ openPoleFigureWindow: vi.fn() }));
vi.mock('./AnomalyBrowser/AnomalyBrowserDrawer', () => ({
  default: (props) => { drawerProps.last = props; return null; },
}));

vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ blob: () => Promise.resolve({}) })));
vi.stubGlobal('createImageBitmap', vi.fn(() => Promise.resolve({
  width: bitmap.width, height: bitmap.height, close: () => {},
})));

import PhaseMapPage, { CanvasInteractionLayer, HoverProbeLayer } from './PhaseMapPage';
import { CursorSyncProvider } from '../EDS/CursorSyncContext';
import { makeHexMap } from '../../services/hexMap';
import { parseScanGeometry } from '../../services/scanGeometry';
import { phaseMapApi } from '../../services/api';

// vitest runs from frontend/ (import.meta.url is not a file: URL in the jsdom environment)
const FIXTURE = JSON.parse(
  readFileSync(resolve(process.cwd(), '..', 'tests', 'fixtures', 'hex_raster_cases.json'), 'utf8'));
const CASE = FIXTURE.geometries.find((c) => c.name === 'a_3x3_right');     // 3 x 3 cells, a = 2 um: a 12 x 14 raster
const wait = (ms = 30) => act(async () => { await new Promise((r) => setTimeout(r, ms)); });

beforeEach(() => {
  vi.clearAllMocks();
  phaseMapApi.availableMaps.mockReset();
  canvasProps.last = null;
  drawerProps.last = null;
  bitmap.width = 14;
  bitmap.height = 12;
});
afterEach(cleanup);

// ---------------------------------------------------------------------------------------------------------
// The pointer in a letterboxed, zoomed and panned host
// ---------------------------------------------------------------------------------------------------------
describe('the pointer in a host of another aspect than the map, zoomed in and panned', () => {
  const hexMap = makeHexMap(parseScanGeometry(CASE.geometry));
  // pixels of the oracle's raster that hold a hexagon, one per cell
  const INSIDE = [];
  for (const p of CASE.pixels) {
    if (p.cell && !INSIDE.some((q) => q.cell[0] === p.cell[0] && q.cell[1] === p.cell[1])) INSIDE.push(p);
  }
  const CORNER = CASE.pixels.find((p) => p.cell === null);
  // Host 200 x 120 px at the origin; the raster is 14 x 12 pixels (aspect 7/6 = 1.167, the host 1.667).
  // View: scale 2, centre at (0.6, 0.4) of the map.
  //   zoomed rect: width 2 * 200 = 400, height 2 * 120 = 240,
  //                left = 0 + 200 / 2 - 0.6 * 400 = -140, top = 0 + 120 / 2 - 0.4 * 240 = -36
  //   the picture is letterboxed inside it: the rect is wider than the picture (1.667 > 1.167), so it fills the
  //                height: 240 high, 240 * 7/6 = 280 wide, with a margin of (400 - 280) / 2 = 60 on the left
  //   so one raster pixel is 280 / 14 = 20 px (240 / 12 = 20), and the centre of pixel (i, j) is at
  //                clientX = -140 + 60 + 20 * (j + 0.5) = 20 j - 70,   clientY = -36 + 20 * (i + 0.5) = 20 i - 26
  const HOST = { left: 0, top: 0, width: 200, height: 120, right: 200, bottom: 120, x: 0, y: 0, toJSON: () => {} };
  const VIEW = { scale: 2, cx: 0.6, cy: 0.4 };
  const zoom = { viewFor: () => VIEW, pan: vi.fn(), zoomAtPointer: vi.fn(), resetOne: vi.fn() };
  const at = (p) => ({ clientX: 20 * p.j - 70, clientY: 20 * p.i - 26 });

  const layer = (extra = {}) => (
    <CanvasInteractionLayer
      view="stack" layers={[]} bitmaps={new Map()} errors={new Map()} shape={hexMap.shape} tileMinWidth={200}
      bitmapVersion={0} scalebar={{ enabled: false }} title={null} stepX={null} ipfKeyImage={null}
      showIpfKey={false} hoverPixel={null} onPixelClick={vi.fn()} onRegionSelected={vi.fn()}
      onLineComplete={vi.fn()} linescanMode={false} magnifierEnabled={false} onContextMenu={vi.fn()}
      zoom={zoom} onContentBbox={vi.fn()} wheelHostRef={{ current: null }} roiBbox={null}
      hexMap={hexMap} {...extra}
    />
  );
  const mountLayer = (extra = {}, probe = null) => {
    const utils = render(
      <CursorSyncProvider>
        {layer(extra)}
        {probe && <HoverProbeLayer probe={null} error={null} requestProbe={probe} clearProbe={vi.fn()} hexMap={hexMap} />}
      </CursorSyncProvider>);
    const host = utils.container.querySelector('[data-phasemap-canvas-host]');
    host.getBoundingClientRect = () => HOST;
    return host;
  };

  it('has pixels the click positions below can tell apart (so the tests can fail)', () => {
    expect(INSIDE.length).toBeGreaterThanOrEqual(4);
    expect(CORNER).toBeTruthy();
    // the oracle's cell of at least one pixel is not the pixel itself
    expect(INSIDE.some((p) => p.i !== p.cell[0] || p.j !== p.cell[1])).toBe(true);
  });

  it('reads the cell of the pixel under the pointer', () => {
    const onPixelClick = vi.fn();
    const host = mountLayer({ onPixelClick });
    for (const p of INSIDE) fireEvent.click(host, at(p));
    expect(onPixelClick.mock.calls).toEqual(INSIDE.map((p) => p.cell));
  });

  it('reads nothing in a corner no hexagon covers', () => {
    const onPixelClick = vi.fn();
    const host = mountLayer({ onPixelClick });
    fireEvent.click(host, at(CORNER));
    expect(onPixelClick).not.toHaveBeenCalled();
  });

  it('asks the probe about the same cells on hover', () => {
    const requestProbe = vi.fn();
    const host = mountLayer({}, requestProbe);
    for (const p of INSIDE) fireEvent.mouseMove(host, at(p));
    expect(requestProbe.mock.calls).toEqual(INSIDE.map((p) => p.cell));
  });

  it('would read other cells without the zoom and the letterbox (the positions are not an accident)', () => {
    const onPixelClick = vi.fn();
    const host = mountLayer({ onPixelClick, zoom: null });
    for (const p of INSIDE) fireEvent.click(host, at(p));
    expect(onPixelClick.mock.calls).not.toEqual(INSIDE.map((p) => p.cell));
  });
});

// ---------------------------------------------------------------------------------------------------------
// The page: hover ring and indexed-region frame
// ---------------------------------------------------------------------------------------------------------
const entry = (scan_geometry, extra = {}) => ({
  id: 'r1', result_id: 'r1', label: 'spherical - Al r1', is_active: true, source_file: 'x.up1',
  phases: ['Al'], shape: [3, 3], n_pixels: 9, ci_mean: 0.3, scan_geometry, ...extra,
});
async function mountPage(geometry, extra = {}) {
  listResults.mockResolvedValue({ data: { results: [entry(geometry, extra)] } });
  render(<PhaseMapPage isActive />);
  await waitFor(() => expect(screen.getByTestId('canvas')).toBeTruthy());
  await wait(80);                                       // the layer's bitmap is in
}
const hover = (row, col) => act(() => drawerProps.last.onHoverPixel(row, col));

// 1100 x 1100 cells, a = 2 um, b = sqrt(3) um, odd rows shifted by half a step. The raster, by hand:
//   x from -a/2 = -1 (the even rows start at 0) to 1099 * 2 + 1 (odd rows) + 1 = 2200: 2201 um wide
//   y from -rho = -2b/3 = -1.1547 to 1099 * 1.7320508 + 1.1547 = 1904.678: 1905.833 um high
//   factor 4 (pixel 0.5 um) would need 4402 x 3812 = 16.8 million pixels, more than the 16 million allowed;
//   factor 2 (pixel 1 um): 2201 x 1906 pixels.
// With 2200 x 2200 cells every extent doubles in rows and columns: factor 2 is 4401 x 3812 = 16.8 million, too
// many again, so factor 1 (pixel 2 um): 2201 x 1906 pixels.
const BIG = { kind: 'hex', n_rows: 1100, n_cols: 1100, step_x: 2, step_y: 1.7320508075688772, odd_row_shift: 0.5 };
const BIGGER = { ...BIG, n_rows: 2200, n_cols: 2200 };

describe('the hover ring of a cell on the page', () => {
  // The ring is placed at (row + 0.5, col + 0.5) of a possibly fractional position, in pixels of the raster,
  // counted from the first pixel (x0 = -1 um, y0 = -1.1547 um).
  it('sits on the centre of the cell at factor 2, not on the pixel that holds it', async () => {
    bitmap.width = 2201; bitmap.height = 1906;
    await mountPage(BIG);
    expect(canvasProps.last.stepX).toBe(1);               // factor 2: a pixel is 2 / 2 um
    // cell (4, 10), an even row: x = 10 * 2 = 20 um, y = 4 * 1.7320508 = 6.9282 um
    //   col position = (20 + 1) / 1 - 0.5 = 20.5, row position = (6.9282 + 1.1547) / 1 - 0.5 = 7.5829
    //   (the pixel that holds the centre is (8, 21): a ring there is 0.5 pixels right and 0.42 below)
    hover(4, 10);
    expect(canvasProps.last.hoverPixel.col).toBeCloseTo(20.5, 6);
    expect(canvasProps.last.hoverPixel.row).toBeCloseTo(7.5829, 4);
    // cell (5, 10), an odd row: x = 20 + 1 = 21 um, y = 5 * 1.7320508 = 8.6603 um
    //   col position = 22 - 0.5 = 21.5, row position = 8.6603 + 1.1547 - 0.5 = 9.3150
    hover(5, 10);
    expect(canvasProps.last.hoverPixel.col).toBeCloseTo(21.5, 6);
    expect(canvasProps.last.hoverPixel.row).toBeCloseTo(9.315, 4);
  });

  it('sits on the centre of the cell at factor 1', async () => {
    bitmap.width = 2201; bitmap.height = 1906;
    await mountPage(BIGGER);
    expect(canvasProps.last.stepX).toBe(2);               // factor 1: a pixel is one step
    // cell (4, 10): col position = (20 + 1) / 2 - 0.5 = 10, row position = (6.9282 + 1.1547) / 2 - 0.5 = 3.5415
    hover(4, 10);
    expect(canvasProps.last.hoverPixel.col).toBeCloseTo(10, 6);
    expect(canvasProps.last.hoverPixel.row).toBeCloseTo(3.5415, 4);
    // cell (5, 10): col position = (21 + 1) / 2 - 0.5 = 10.5, row position = (8.6603 + 1.1547) / 2 - 0.5 = 4.4075
    hover(5, 10);
    expect(canvasProps.last.hoverPixel.col).toBeCloseTo(10.5, 6);
    expect(canvasProps.last.hoverPixel.row).toBeCloseTo(4.4075, 4);
  });

  it('is the cell itself on a square scan', async () => {
    await mountPage({ kind: 'square', n_rows: 12, n_cols: 14, step_x: 2, step_y: 2, odd_row_shift: 0 });
    hover(4, 10);
    expect(canvasProps.last.hoverPixel).toEqual({ row: 4, col: 10 });
  });

  it('is gone when the cursor leaves', async () => {
    await mountPage(CASE.geometry);
    hover(1, 1);
    expect(canvasProps.last.hoverPixel).not.toBeNull();
    act(() => drawerProps.last.onLeavePixel());
    expect(canvasProps.last.hoverPixel).toBeNull();
  });
});

describe('the frame of the indexed region on the page', () => {
  // The region covers the cells in rows 1-2 and columns 0-1 of the 3 x 3 oracle scan (a = 2, b = 1.7320508,
  // odd rows shifted by one um; raster 14 x 12 pixels of 0.5 um, x0 = -1, y0 = -1.1547):
  //   x: the left edge of the cells is the leftmost centre minus a/2. The centres: row 1 (odd) at 1 and 3, row 2
  //      at 0 and 2, so from 0 - 1 = -1 to 3 + 1 = 4 um -> pixels (-1 + 1) / 0.5 = 0 to (4 + 1) / 0.5 = 10: x 0, w 10
  //   y: from row 1 centre 1.7321 - rho 1.1547 = 0.5774 to row 2 centre 3.4641 + 1.1547 = 4.6188 um
  //      -> pixels floor((0.5774 + 1.1547) / 0.5) = floor(3.46) = 3 to ceil((4.6188 + 1.1547) / 0.5) = ceil(11.55) = 12:
  //      y 3, h 9
  it('is given in the pixels of the raster', async () => {
    await mountPage(CASE.geometry, { roi_bbox: { x: 0, y: 1, w: 2, h: 2 } });
    expect(canvasProps.last.roiBbox).toEqual({ x: 0, y: 3, w: 10, h: 9 });
  });

  it('is the box as the backend gave it on a square scan', async () => {
    await mountPage({ kind: 'square', n_rows: 12, n_cols: 14, step_x: 2, step_y: 2, odd_row_shift: 0 },
      { roi_bbox: { x: 3, y: 4, w: 5, h: 6 } });
    expect(canvasProps.last.roiBbox).toEqual({ x: 3, y: 4, w: 5, h: 6 });
  });
});
