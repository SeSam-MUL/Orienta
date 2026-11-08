// @vitest-environment jsdom
/**
 * A rectangle, a line and the scale bar on a hexagonal map, measured through the real page: what is drawn
 * on the display raster reaches the backend as index-space selections (a bounding box and a mask, or a list
 * of points), and the scale bar measures display pixels.
 *
 * Expectations are worked by hand below or come from tests/fixtures/hex_raster_cases.json (an independent
 * oracle); none comes from hexGeometry.js or hexSelection.js.
 */
/* global process */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const listResults = vi.fn();
const regionStats = vi.fn();
const linescan = vi.fn();
const canvasProps = vi.hoisted(() => ({ last: null }));
const tileProps = vi.hoisted(() => ({ last: null }));

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
      render: vi.fn(() => ok({ image: 'Zm9v' })),          // the page's preview call: an error here replaces the canvas
      regionStats: (...a) => regionStats(...a),
      linescan: (...a) => linescan(...a),
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
vi.mock('./AnomalyBrowser/AnomalyBrowserDrawer', () => ({ default: () => null }));
// The grid view draws a canvas per layer; only the handler it is given matters here.
vi.mock('../EDS/TileGrid', () => ({
  default: (props) => { tileProps.last = props; return null; },
}));

vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ blob: () => Promise.resolve({}) })));
vi.stubGlobal('createImageBitmap', vi.fn(() => Promise.resolve({ width: 14, height: 12, close: () => {} })));

import PhaseMapPage, { CanvasInteractionLayer } from './PhaseMapPage';
import { CursorSyncProvider } from '../EDS/CursorSyncContext';
import { makeHexMap } from '../../services/hexMap';
import { parseScanGeometry } from '../../services/scanGeometry';
import { phaseMapApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';

// vitest runs from frontend/ (import.meta.url is not a file: URL in the jsdom environment)
const FIXTURE = JSON.parse(
  readFileSync(resolve(process.cwd(), '..', 'tests', 'fixtures', 'hex_raster_cases.json'), 'utf8'));
const CASE = FIXTURE.geometries.find((c) => c.name === 'a_3x3_right');     // 3 x 3 cells, a = 2 um: a 12 x 14 raster
const CORNER = CASE.pixels.find((p) => p.cell === null);                   // a pixel no hexagon covers
const SQUARE = { kind: 'square', n_rows: 12, n_cols: 14, step_x: 2, step_y: 2, odd_row_shift: 0 };
const HOST = { left: 0, top: 0, width: 140, height: 120, right: 140, bottom: 120, x: 0, y: 0, toJSON: () => {} };
const at = (i, j) => ({ clientX: j * 10 + 5, clientY: i * 10 + 5 });
const entry = (scan_geometry, id = 'r1') => ({
  id, result_id: id, label: `spherical - Al ${id}`, is_active: true, source_file: 'x.up1',
  phases: ['Al'], shape: [3, 3], n_pixels: 9, ci_mean: 0.3, scan_geometry,
});
const wait = (ms = 30) => act(async () => { await new Promise((r) => setTimeout(r, ms)); });
/** An answer the test releases itself. */
function held(data) {
  let release;
  const promise = new Promise((resolve) => { release = () => resolve({ data }); });
  return { promise, release };
}

const mount = async (geometry) => {
  listResults.mockResolvedValue({ data: { results: [entry(geometry)] } });
  const utils = render(<PhaseMapPage isActive />);
  const host = await waitFor(() => {
    const h = utils.container.querySelector('[data-phasemap-canvas-host]');
    expect(h).toBeTruthy();
    return h;
  });
  host.getBoundingClientRect = () => HOST;
  // wait for the layer's bitmap: only then does the canvas know the raster shape
  await wait(60);
  return { ...utils, host };
};

/** Show another result (the page learns of it when its Indexing tab is clicked), and return the host again. */
async function showResult(utils, geometry, id) {
  const before = listResults.mock.calls.length;
  listResults.mockResolvedValue({ data: { results: [entry(geometry, id)] } });
  fireEvent.click(screen.getByRole('tab', { name: /Indexing/ }));
  await waitFor(() => expect(listResults.mock.calls.length).toBeGreaterThan(before));
  await wait(60);
  const host = utils.container.querySelector('[data-phasemap-canvas-host]');
  host.getBoundingClientRect = () => HOST;
  return host;
}

beforeEach(() => {
  vi.clearAllMocks();
  phaseMapApi.availableMaps.mockReset();               // back to "no dataset geometry"
  canvasProps.last = null;
  regionStats.mockImplementation(() => Promise.resolve({ data: { n_pixels: 3, phases: [], scalars: {} } }));
  linescan.mockImplementation(() => Promise.resolve({ data: { samples: 2, series: {} } }));
  useDataStore.setState({ stepSize: { x: 2, y: 2, units: 'µm' } });
});
afterEach(cleanup);

describe('the oracle case the tests below rest on', () => {
  it('has a pixel no hexagon covers, and a raster that is not the index grid', () => {
    expect(CORNER).toBeTruthy();
    expect([CASE.spec.height, CASE.spec.width]).toEqual([12, 14]);
  });
});

describe('a rectangle drawn on a hexagonal map', () => {
  it('asks for the cells whose centres it contains, as a box and a mask', async () => {
    const { host } = await mount(CASE.geometry);
    // pixels (3,3)..(9,9): x in [0.5, 4.0) um, y in [0.345, 3.845) um. Cell centres: row 1 (y 1.73) at
    // x = 1, 3; row 2 (y 3.46) at x = 0, 2, 4 -> only x = 2 (0 < 0.5, 4 is not < 4.0). Row 0 (y 0) is above.
    // Cells (1,0), (1,1), (2,1): box rows 1-2, columns 0-1, mask [1, 1, 0, 1].
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    fireEvent.mouseUp(host, at(9, 9));
    await waitFor(() => expect(regionStats).toHaveBeenCalledTimes(1));
    expect(regionStats).toHaveBeenCalledWith({
      row_start: 1, col_start: 0, row_end: 2, col_end: 1, mask: [true, true, false, true],
    });
  });

  it('asks for the same cells when it is dragged from the other corner', async () => {
    const { host } = await mount(CASE.geometry);
    fireEvent.mouseDown(host, { shiftKey: true, ...at(9, 9) });
    fireEvent.mouseUp(host, at(3, 3));
    await waitFor(() => expect(regionStats).toHaveBeenCalledTimes(1));
    expect(regionStats.mock.calls[0][0]).toEqual({
      row_start: 1, col_start: 0, row_end: 2, col_end: 1, mask: [true, true, false, true],
    });
  });

  it('asks for nothing when it covers no hexagon at all', async () => {
    const { host } = await mount(CASE.geometry);
    fireEvent.mouseDown(host, { shiftKey: true, ...at(CORNER.i, CORNER.j) });   // a transparent corner
    fireEvent.mouseUp(host, at(CORNER.i, CORNER.j));
    await wait();
    expect(regionStats).not.toHaveBeenCalled();
    expect(screen.queryByText('Region Statistics')).toBeNull();
  });

  it('is the old request, without a mask, on a square scan', async () => {
    const { host } = await mount(SQUARE);
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    fireEvent.mouseUp(host, at(9, 9));
    await waitFor(() => expect(regionStats).toHaveBeenCalledTimes(1));
    expect(regionStats).toHaveBeenCalledWith({ row_start: 3, col_start: 3, row_end: 9, col_end: 9 });
  });
});

/** Nearest of the nine points by plain distance, a tie to the lowest index: written from the lattice only. */
function bruteNearest(x, y) {
  let best = -1;
  let bestD = Infinity;
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      const px = c * 2 + (r % 2 === 1 ? 1 : 0);
      const py = r * Math.sqrt(3);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < bestD) { bestD = d; best = r * 3 + c; }
    }
  }
  return [Math.floor(best / 3), best % 3];
}

describe('a line drawn on a hexagonal map', () => {
  it('samples the scan points under the physical segment, not the pixels of the index grid', async () => {
    const { host } = await mount(CASE.geometry);
    fireEvent.click(screen.getByText('Linescan OFF'));
    fireEvent.mouseDown(host, at(3, 3));
    fireEvent.mouseUp(host, at(3, 11));
    await waitFor(() => expect(linescan).toHaveBeenCalledTimes(1));
    const body = linescan.mock.calls[0][0];
    expect(body.layers.length).toBeGreaterThan(0);
    expect(body.points).toHaveLength(128);
    // pixel centres (0.75, 0.595) -> (4.75, 0.595) um (x0 = -1, y0 = -1.155, 0.5 um pixels)
    const want = Array.from({ length: 128 }, (_, k) => bruteNearest(0.75 + (4 * k) / 127, 1.75 - (2 * Math.sqrt(3)) / 3));
    expect(body.points).toEqual(want);
    // the route ignores the corners and n_samples when it is given points, so none are sent
    expect(body.start_row).toBeUndefined();
    expect(body.n_samples).toBeUndefined();
  });

  it('is the old request, between two pixels, on a square scan', async () => {
    const { host } = await mount(SQUARE);
    fireEvent.click(screen.getByText('Linescan OFF'));
    fireEvent.mouseDown(host, at(3, 3));
    fireEvent.mouseUp(host, at(3, 11));
    await waitFor(() => expect(linescan).toHaveBeenCalledTimes(1));
    expect(linescan.mock.calls[0][0]).toMatchObject({ start_row: 3, start_col: 3, end_row: 3, end_col: 11 });
    expect(linescan.mock.calls[0][0].points).toBeUndefined();
  });
});

describe('the scale bar on a hexagonal map', () => {
  it('measures display pixels: the scan step divided by the factor', async () => {
    await mount(CASE.geometry);
    await waitFor(() => expect(canvasProps.last).not.toBeNull());
    expect(canvasProps.last.stepX).toBe(0.5);                  // the map's 2 um step, over the factor 4
  });

  it('takes the step from the map, not from the file loaded in the viewer', async () => {
    useDataStore.setState({ stepSize: { x: 7, y: 7, units: 'µm' } });
    await mount(CASE.geometry);
    await waitFor(() => expect(canvasProps.last).not.toBeNull());
    expect(canvasProps.last.stepX).toBe(0.5);
  });

  it('is the scan step itself on a square scan', async () => {
    await mount(SQUARE);
    await waitFor(() => expect(canvasProps.last).not.toBeNull());
    expect(canvasProps.last.stepX).toBe(2);
  });

  it('quotes no length when the file did not state the step: the placeholder is not a measurement', async () => {
    // the store holds a confirmed step of 2 um, which the square branch would quote; on a hexagonal map the
    // step of the geometry is what counts, and here it is a placeholder (0.5 um would be an invented number)
    await mount({ ...CASE.geometry, steps_known: false });
    await waitFor(() => expect(canvasProps.last).not.toBeNull());
    expect(canvasProps.last.stepX).toBeNull();
  });
});

describe('the step fields of the annotation panel', () => {
  const fieldOf = (label) => screen.getByText(label).parentElement.querySelector('input');
  // the panel is folded away until it is opened
  const open = () => fireEvent.click(screen.getByRole('button', { name: /Annotation & calibration/ }));

  it('are locked on a hexagonal map and show the step of the map', async () => {
    await mount(CASE.geometry);                              // a = 2 um, b = 1.7320508 um (the oracle case)
    open();
    expect(fieldOf('Step X:').disabled).toBe(true);
    expect(fieldOf('Step Y:').disabled).toBe(true);
    expect(Number(fieldOf('Step X:').value)).toBe(2);
    expect(Number(fieldOf('Step Y:').value)).toBeCloseTo(1.7320508, 6);
    expect(screen.queryByText(/Drives the scalebar length/)).toBeNull();
    expect(screen.queryByText(/comes from the scan geometry of the map/)).not.toBeNull();
  });

  it('show nothing, locked, when the file did not state the step', async () => {
    await mount({ ...CASE.geometry, steps_known: false });
    open();
    expect(fieldOf('Step X:').disabled).toBe(true);
    expect(fieldOf('Step X:').value).toBe('');
    expect(fieldOf('Step Y:').value).toBe('');
  });

  it('are unchanged on a square scan: editable, with the step of the file and the old hint', async () => {
    await mount(SQUARE);
    open();
    expect(fieldOf('Step X:').disabled).toBe(false);
    expect(fieldOf('Step Y:').disabled).toBe(false);
    expect(Number(fieldOf('Step X:').value)).toBe(2);        // the store's step, set in beforeEach
    expect(screen.queryByText(/Drives the scalebar length/)).not.toBeNull();
    expect(screen.queryByText(/comes from the scan geometry of the map/)).toBeNull();
  });
});

describe('a selection belongs to the result and the raster it was made on', () => {
  it('goes when another result is shown, and so does an answer still on its way', async () => {
    const pending = held({ n_pixels: 3, phases: [], scalars: {} });
    regionStats.mockImplementation(() => pending.promise);
    const utils = await mount(CASE.geometry);
    fireEvent.mouseDown(utils.host, { shiftKey: true, ...at(3, 3) });
    fireEvent.mouseUp(utils.host, at(9, 9));
    await waitFor(() => expect(regionStats).toHaveBeenCalledTimes(1));
    expect(screen.queryByText('Region Statistics')).not.toBeNull();     // asked, not answered: the panel waits
    await showResult(utils, CASE.geometry, 'r2');
    expect(screen.queryByText('Region Statistics')).toBeNull();         // the wait is over with the selection
    await act(async () => { pending.release(); await Promise.resolve(); });
    await wait();
    expect(screen.queryByText('Region Statistics')).toBeNull();         // and the late answer is not shown
  });

  it('shows the statistics of a selection made on the result that stays', async () => {
    const { host } = await mount(CASE.geometry);
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    fireEvent.mouseUp(host, at(9, 9));
    await waitFor(() => expect(screen.queryByText('Region Statistics')).not.toBeNull());
    await wait();
    expect(screen.queryByText('Region Statistics')).not.toBeNull();
  });

  it('removes the profile of a line when another result is shown', async () => {
    const utils = await mount(CASE.geometry);
    fireEvent.click(screen.getByText('Linescan OFF'));
    fireEvent.mouseDown(utils.host, at(3, 3));
    fireEvent.mouseUp(utils.host, at(3, 11));
    await waitFor(() => expect(screen.queryByText('Linescan Profile')).not.toBeNull());
    await showResult(utils, CASE.geometry, 'r2');
    expect(screen.queryByText('Linescan Profile')).toBeNull();
  });

  it('drops a rectangle that is under way when the scan geometry arrives under the map that is already shown', async () => {
    // A dataset without a stamped result: its geometry comes from available-maps, after the map is on screen.
    // The rectangle starts on the picture while the page still takes the scan for a square one and ends after
    // the answer: its two pixels would be read with two different maps.
    const early = held({ maps: [], dataset_scan_geometry: CASE.geometry });
    phaseMapApi.availableMaps.mockImplementation(() => early.promise);
    const utils = await mount(undefined);
    const host = utils.host;
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    await act(async () => { early.release(); await Promise.resolve(); });
    await wait(80);
    expect(utils.container.querySelector('[data-phasemap-canvas-host]')).toBe(host);   // the same map, not a new one
    host.getBoundingClientRect = () => HOST;
    expect(canvasProps.last.stepX).toBe(0.5);                           // ... and it now knows the scan is hexagonal
    fireEvent.mouseUp(host, at(9, 9));
    await wait();
    expect(regionStats).not.toHaveBeenCalled();
    // a rectangle made after the answer is read on the hexagonal map (the case of the first test)
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    fireEvent.mouseUp(host, at(9, 9));
    await waitFor(() => expect(regionStats).toHaveBeenCalledTimes(1));
    expect(regionStats).toHaveBeenCalledWith({
      row_start: 1, col_start: 0, row_end: 2, col_end: 1, mask: [true, true, false, true],
    });
  });

  it('is untouched on a square scan: its statistics stay when another result is shown', async () => {
    const utils = await mount(SQUARE);
    fireEvent.mouseDown(utils.host, { shiftKey: true, ...at(3, 3) });
    fireEvent.mouseUp(utils.host, at(9, 9));
    await waitFor(() => expect(screen.queryByText('Region Statistics')).not.toBeNull());
    await showResult(utils, SQUARE, 'r2');
    expect(screen.queryByText('Region Statistics')).not.toBeNull();
  });
});

// The page replaces the map by a loading overlay while it refreshes, which also ends a gesture; but when the
// geometry changes under a map that stays mounted, the pixels of a half-made rectangle or line would be read on
// the new raster. The layer itself, rendered again with another selection key (the result or raster changed):
describe('a rectangle or a line that is under way when the result or the raster changes', () => {
  const hexMap = makeHexMap(parseScanGeometry(CASE.geometry));
  const layerFor = ({ selectionKey, shape = hexMap.shape, map = hexMap, linescanMode = false,
    onRegionSelected, onLineComplete, view = 'stack' }) => (
    <CursorSyncProvider>
      <CanvasInteractionLayer
        view={view} layers={[]} bitmaps={new Map()} errors={new Map()} shape={shape} tileMinWidth={200}
        bitmapVersion={0} scalebar={{ enabled: false }} title={null} stepX={null} ipfKeyImage={null}
        showIpfKey={false} hoverPixel={null} onPixelClick={vi.fn()} onRegionSelected={onRegionSelected}
        onLineComplete={onLineComplete} linescanMode={linescanMode} magnifierEnabled={false}
        onContextMenu={vi.fn()} zoom={null} onContentBbox={vi.fn()} wheelHostRef={{ current: null }}
        roiBbox={null} hexMap={map} selectionKey={selectionKey}
      />
    </CursorSyncProvider>);
  const mountLayer = (props) => {
    const utils = render(layerFor(props));
    const host = utils.container.querySelector('[data-phasemap-canvas-host]');
    host.getBoundingClientRect = () => HOST;
    return { host, again: (next) => utils.rerender(layerFor({ ...props, ...next })) };
  };

  it('delivers a rectangle whose key did not change (so the cases below can fail)', () => {
    const onRegionSelected = vi.fn();
    const { host } = mountLayer({ selectionKey: 'r1|a', onRegionSelected });
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    fireEvent.mouseUp(host, at(9, 9));
    expect(onRegionSelected).toHaveBeenCalledWith({ rowStart: 3, colStart: 3, rowEnd: 9, colEnd: 9 });
  });

  it('drops a rectangle when another result is shown before it ends', () => {
    const onRegionSelected = vi.fn();
    const { host, again } = mountLayer({ selectionKey: 'r1|a', onRegionSelected });
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    again({ selectionKey: 'r2|a' });
    fireEvent.mouseUp(host, at(9, 9));
    expect(onRegionSelected).not.toHaveBeenCalled();
  });

  it('drops a rectangle when the raster changes before it ends', () => {
    const onRegionSelected = vi.fn();
    const { host, again } = mountLayer({ selectionKey: 'r1|a', onRegionSelected });
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    again({ selectionKey: 'r1|b' });
    fireEvent.mouseUp(host, at(9, 9));
    expect(onRegionSelected).not.toHaveBeenCalled();
  });

  it('drops a rectangle when the picture shown stops being the raster before it ends', () => {
    const onRegionSelected = vi.fn();
    const { host, again } = mountLayer({ selectionKey: 'r1|a', onRegionSelected });
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    again({ shape: [6, 5] });
    fireEvent.mouseUp(host, at(9, 9));
    expect(onRegionSelected).not.toHaveBeenCalled();
  });

  it('starts no rectangle on a picture that is not the raster', () => {
    const onRegionSelected = vi.fn();
    const { host, again } = mountLayer({ selectionKey: 'r1|a', shape: [6, 5], onRegionSelected });
    fireEvent.mouseDown(host, { shiftKey: true, ...at(3, 3) });
    again({ shape: hexMap.shape });                    // the raster arrives while the button is down
    fireEvent.mouseUp(host, at(9, 9));
    expect(onRegionSelected).not.toHaveBeenCalled();
  });

  it('delivers a line whose key did not change, and drops one whose key changed', () => {
    const onLineComplete = vi.fn();
    const { host, again } = mountLayer({ selectionKey: 'r1|a', linescanMode: true, onLineComplete });
    fireEvent.mouseDown(host, at(3, 3));
    fireEvent.mouseUp(host, at(3, 11));
    expect(onLineComplete).toHaveBeenCalledTimes(1);
    expect(onLineComplete).toHaveBeenCalledWith({ start: { row: 3, col: 3 }, end: expect.objectContaining({ row: 3, col: 11 }) });

    fireEvent.mouseDown(host, at(3, 3));
    again({ selectionKey: 'r2|a' });
    fireEvent.mouseUp(host, at(3, 11));
    expect(onLineComplete).toHaveBeenCalledTimes(1);   // still the first one
  });

  it('drops a line when the picture shown stops being the raster before it ends', () => {
    const onLineComplete = vi.fn();
    const { host, again } = mountLayer({ selectionKey: 'r1|a', linescanMode: true, onLineComplete });
    fireEvent.mouseDown(host, at(3, 3));
    again({ shape: [6, 5] });
    fireEvent.mouseUp(host, at(3, 11));
    expect(onLineComplete).not.toHaveBeenCalled();
  });

  it('starts no line on a picture that is not the raster, even if the raster arrives before it ends', () => {
    const onLineComplete = vi.fn();
    const { host, again } = mountLayer({ selectionKey: 'r1|a', shape: [6, 5], linescanMode: true, onLineComplete });
    fireEvent.mouseDown(host, at(3, 3));
    again({ shape: hexMap.shape });
    fireEvent.mouseUp(host, at(3, 11));
    expect(onLineComplete).not.toHaveBeenCalled();
  });

  it('gives the grid view the same check: a rectangle it reports is dropped after the result changed', () => {
    const onRegionSelected = vi.fn();
    const region = { rowStart: 3, colStart: 3, rowEnd: 9, colEnd: 9 };
    const { host, again } = mountLayer({ selectionKey: 'r1|a', view: 'grid', onRegionSelected });
    fireEvent.mouseDown(host, at(3, 3));
    act(() => tileProps.last.onRegionSelected(region));
    expect(onRegionSelected).toHaveBeenCalledTimes(1);                 // the same result: delivered
    fireEvent.mouseDown(host, at(3, 3));
    again({ selectionKey: 'r2|a' });
    act(() => tileProps.last.onRegionSelected(region));
    expect(onRegionSelected).toHaveBeenCalledTimes(1);                 // another result: dropped
  });

  it('gives the grid view of a square scan the handler it was given', () => {
    const onRegionSelected = vi.fn();
    mountLayer({ selectionKey: null, map: null, view: 'grid', onRegionSelected });
    expect(tileProps.last.onRegionSelected).toBe(onRegionSelected);
  });

  it('starts no line on a picture that is not the raster', () => {
    const onLineComplete = vi.fn();
    const { host } = mountLayer({ selectionKey: 'r1|a', shape: [6, 5], linescanMode: true, onLineComplete });
    fireEvent.mouseDown(host, at(3, 3));
    fireEvent.mouseUp(host, at(3, 4));
    expect(onLineComplete).not.toHaveBeenCalled();
  });

  it('leaves a square scan alone: no key, no check', () => {
    const onRegionSelected = vi.fn();
    const onLineComplete = vi.fn();
    const { host, again } = mountLayer({
      selectionKey: null, map: null, shape: [12, 14], onRegionSelected, onLineComplete });
    fireEvent.mouseDown(host, { shiftKey: true, ...at(1, 1) });
    again({ shape: [8, 9] });                          // a picture of another size is no reason to drop it
    fireEvent.mouseUp(host, at(4, 4));
    expect(onRegionSelected).toHaveBeenCalledTimes(1);
  });
});
