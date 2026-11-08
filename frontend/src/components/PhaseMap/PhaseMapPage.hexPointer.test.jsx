// @vitest-environment jsdom
/**
 * The pointer on a hexagonal map: a pixel of the display raster is turned into the measurement point it
 * shows before anything asks the backend about it, and a corner no hexagon covers is nothing.
 *
 * The expectations come from tests/fixtures/hex_raster_cases.json (written by an independent oracle), not
 * from hexGeometry.js.
 */
/* global process */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

vi.mock('../../services/api', () => {
  const ok = (data) => Promise.resolve({ data });
  const anyApi = () => new Proxy({}, {
    get: (target, prop) => {
      if (typeof prop !== 'string') return undefined;
      if (!(prop in target)) target[prop] = vi.fn(() => ok({}));
      return target[prop];
    },
  });
  return {
    default: anyApi(), indexApi: anyApi(), phaseMapApi: anyApi(), analysisApi: anyApi(), ebsdApi: anyApi(),
    h5Api: anyApi(), pcApi: anyApi(), forwardDiagApi: anyApi(), refinementApi: anyApi(), citationsApi: anyApi(),
  };
});
vi.mock('../../services/addonsApi', () => ({ addonsApi: { mapImage: vi.fn() } }));
vi.mock('./LayeredCanvas', () => ({ default: () => null }));
vi.mock('../EDS/LinescanProfilePlot', () => ({ default: () => null }));
vi.mock('../EDS/MagnifierLens', () => ({ default: () => null }));
vi.mock('../PatternMatch/LinkedPatternImage', () => ({ default: () => null }));
vi.mock('../PoleFigure/openPoleFigureWindow', () => ({ openPoleFigureWindow: vi.fn() }));
vi.mock('./AnomalyBrowser/AnomalyBrowserDrawer', () => ({ default: () => null }));

import { CanvasInteractionLayer, HoverProbeLayer } from './PhaseMapPage';
import { CursorSyncProvider, useCursorPublisher } from '../EDS/CursorSyncContext';
import { usePhaseMapProbe } from './hooks/usePhaseMapProbe';
import { phaseMapApi } from '../../services/api';
import { makeHexMap } from '../../services/hexMap';
import { parseScanGeometry } from '../../services/scanGeometry';

// vitest runs from frontend/ (import.meta.url is not a file: URL in the jsdom environment)
const FIXTURE = JSON.parse(
  readFileSync(resolve(process.cwd(), '..', 'tests', 'fixtures', 'hex_raster_cases.json'), 'utf8'));
const CASE = FIXTURE.geometries.find((c) => c.name === 'a_3x3_right');       // 3 x 3, a = 2 um: a 12 x 14 raster
const hexMap = makeHexMap(parseScanGeometry(CASE.geometry));
const SHAPE = hexMap.shape;
// one pixel of every cell of the scan, as the oracle assigns them, and a pixel no hexagon covers
const ONE_PER_CELL = [];
for (const p of CASE.pixels) {
  if (p.cell && !ONE_PER_CELL.some((q) => q.cell[0] === p.cell[0] && q.cell[1] === p.cell[1])) ONE_PER_CELL.push(p);
}
const INSIDE = ONE_PER_CELL;
const CORNER = CASE.pixels.find((p) => p.cell === null);
// 140 x 120 CSS px for 14 x 12 raster pixels: 10 px per pixel, and the same aspect, so no letterbox
const HOST = { left: 0, top: 0, width: 140, height: 120, right: 140, bottom: 120, x: 0, y: 0, toJSON: () => {} };
const at = (p) => ({ clientX: p.j * 10 + 5, clientY: p.i * 10 + 5 });

afterEach(cleanup);

const layer = (extra = {}) => (
  <CanvasInteractionLayer
    view="stack" layers={[]} bitmaps={new Map()} errors={new Map()} shape={SHAPE} tileMinWidth={200}
    bitmapVersion={0} scalebar={{ enabled: false }} title={null} stepX={null} ipfKeyImage={null}
    showIpfKey={false} hoverPixel={null} onPixelClick={vi.fn()} onRegionSelected={vi.fn()}
    onLineComplete={vi.fn()} linescanMode={false} magnifierEnabled={false} onContextMenu={vi.fn()}
    zoom={null} onContentBbox={vi.fn()} wheelHostRef={{ current: null }} roiBbox={null}
    hexMap={hexMap} {...extra}
  />
);

function mountLayer(extra = {}) {
  const onPixelClick = vi.fn();
  const utils = render(<CursorSyncProvider>{layer({ onPixelClick, ...extra })}</CursorSyncProvider>);
  const host = utils.container.querySelector('[data-phasemap-canvas-host]');
  host.getBoundingClientRect = () => HOST;
  return { ...utils, host, onPixelClick };
}

describe('the oracle case', () => {
  it('has a cell on both kinds of rows and a corner, so the tests below can fail', () => {
    expect(INSIDE.length).toBeGreaterThanOrEqual(4);                // the fixture samples some pixels per cell
    expect(new Set(INSIDE.map((p) => p.cell[0])).size).toBe(3);     // rows 0, 1 and 2: both kinds of rows
    expect(CORNER).toBeTruthy();
    expect(CORNER.cell).toBeNull();
  });
});

describe('a click on a hexagonal map', () => {
  it('reports the measurement point under the pointer, not the raster pixel', () => {
    const { host, onPixelClick } = mountLayer();
    for (const p of INSIDE) fireEvent.click(host, at(p));
    expect(onPixelClick.mock.calls).toEqual(INSIDE.map((p) => p.cell));
    // the raster pixel itself would have been a different pair for at least one cell
    expect(INSIDE.some((p) => p.i !== p.cell[0] || p.j !== p.cell[1])).toBe(true);
  });

  it('opens nothing on a corner no hexagon covers', () => {
    const { host, onPixelClick } = mountLayer();
    fireEvent.click(host, at(CORNER));
    expect(onPixelClick).not.toHaveBeenCalled();
  });

  it('reads nothing while the picture shown is not on that raster', () => {
    const { host, onPixelClick } = mountLayer({ shape: [6, 5] });   // the index grid, not the raster
    fireEvent.click(host, { clientX: 20, clientY: 20 });
    expect(onPixelClick).not.toHaveBeenCalled();
  });

  it('is the old behaviour for a square scan: the pixel is the point', () => {
    const { host, onPixelClick } = mountLayer({ hexMap: null });
    fireEvent.click(host, at(INSIDE[0]));
    expect(onPixelClick).toHaveBeenCalledWith(INSIDE[0].i, INSIDE[0].j);
  });

  it('is the old behaviour for a square scan also when the picture has another shape', () => {
    const { host, onPixelClick } = mountLayer({ hexMap: null, shape: [6, 5] });
    fireEvent.click(host, { clientX: 20, clientY: 20 });
    expect(onPixelClick).toHaveBeenCalledTimes(1);
  });
});

function Publisher({ onReady }) {
  onReady(useCursorPublisher());
  return null;
}

describe('the hover probe on a hexagonal map', () => {
  const mountProbe = (map) => {
    const requestProbe = vi.fn();
    const clearProbe = vi.fn();
    let publish;
    render(
      <CursorSyncProvider>
        <Publisher onReady={(p) => { publish = p; }} />
        <HoverProbeLayer probe={null} error={null} requestProbe={requestProbe} clearProbe={clearProbe} hexMap={map} />
      </CursorSyncProvider>);
    clearProbe.mockClear();                  // subscribing replays "not hovering"; that is not what is tested
    return {
      requestProbe, clearProbe,
      publish: (pos) => act(() => publish({ hovering: true, screenX: 1, screenY: 1, ...pos })),
    };
  };

  it('asks about the measurement point under the cursor', () => {
    const { requestProbe, publish } = mountProbe(hexMap);
    for (const p of INSIDE) publish({ row: p.i, col: p.j });
    expect(requestProbe.mock.calls).toEqual(INSIDE.map((p) => p.cell));
  });

  it('asks about nothing and hides the tooltip in a corner', () => {
    const { requestProbe, clearProbe, publish } = mountProbe(hexMap);
    publish({ row: CORNER.i, col: CORNER.j });
    expect(requestProbe).not.toHaveBeenCalled();
    expect(clearProbe).toHaveBeenCalledTimes(1);
  });

  it('is the old behaviour for a square scan', () => {
    const { requestProbe, publish } = mountProbe(null);
    publish({ row: 3, col: 4 });
    expect(requestProbe).toHaveBeenCalledWith(3, 4);
  });

  it('works end to end from the pointer over the map', () => {
    const requestProbe = vi.fn();
    const utils = render(
      <CursorSyncProvider>
        {layer()}
        <HoverProbeLayer probe={null} error={null} requestProbe={requestProbe} clearProbe={vi.fn()} hexMap={hexMap} />
      </CursorSyncProvider>);
    const host = utils.container.querySelector('[data-phasemap-canvas-host]');
    host.getBoundingClientRect = () => HOST;
    fireEvent.mouseMove(host, at(INSIDE[1]));
    expect(requestProbe).toHaveBeenCalledWith(INSIDE[1].cell[0], INSIDE[1].cell[1]);
  });

  it('does not ask while the picture shown is not on that raster', () => {
    const requestProbe = vi.fn();
    const utils = render(
      <CursorSyncProvider>
        {layer({ shape: [6, 5] })}
        <HoverProbeLayer probe={null} error={null} requestProbe={requestProbe} clearProbe={vi.fn()} hexMap={hexMap} />
      </CursorSyncProvider>);
    const host = utils.container.querySelector('[data-phasemap-canvas-host]');
    host.getBoundingClientRect = () => HOST;
    fireEvent.mouseMove(host, { clientX: 20, clientY: 20 });
    expect(requestProbe).not.toHaveBeenCalled();
  });

  it('reaches the backend with exactly the (row, col) of the cell, through the real probe hook', async () => {
    function Probing() {
      const { probe, error, requestProbe, clear } = usePhaseMapProbe({ debounceMs: 0 });
      return <HoverProbeLayer probe={probe} error={error} requestProbe={requestProbe} clearProbe={clear} hexMap={hexMap} />;
    }
    phaseMapApi.probe.mockClear();
    const utils = render(<CursorSyncProvider>{layer()}<Probing /></CursorSyncProvider>);
    const host = utils.container.querySelector('[data-phasemap-canvas-host]');
    host.getBoundingClientRect = () => HOST;

    fireEvent.mouseMove(host, at(CORNER));
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(phaseMapApi.probe).not.toHaveBeenCalled();

    const last = INSIDE[INSIDE.length - 1];
    fireEvent.mouseMove(host, at(last));
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(phaseMapApi.probe).toHaveBeenCalledTimes(1);
    expect(phaseMapApi.probe).toHaveBeenCalledWith(last.cell[0], last.cell[1]);
  });
});
