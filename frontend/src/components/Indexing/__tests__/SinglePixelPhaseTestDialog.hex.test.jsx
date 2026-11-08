// @vitest-environment jsdom
/**
 * The single-pixel phase test's scan picker on a hexagonal scan, through the real dialog and the real data store:
 * a click names the measurement point under the display raster (none in a corner no hexagon covers), the crosshair
 * sits on its hexagon, and a picture that is not the expected raster, or that was fetched for another measurement,
 * is not shown.
 *
 * The scan is a made-up 3 x 3 one with 2 um between points in a row, a row pitch of 1.732 um and odd rows shifted
 * by +0.5 of a step; its raster is 12 x 14 pixels of 0.5 um starting at (-1, -1.155) um
 * (tests/fixtures/hex_raster_cases.json, 'a_3x3_right'). Expectations are worked by hand below.
 */
import '@testing-library/jest-dom/vitest';
import { render, fireEvent, waitFor, cleanup, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { cwd } from 'node:process';

afterEach(() => cleanup());

vi.mock('../../../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    indexApi: {
      phaseTestStart: vi.fn(), phaseTestProgress: vi.fn(),
      phaseTestCancel: vi.fn(), phaseTestPhases: vi.fn(),
      phaseTestRemask: vi.fn(), singlePixelPhaseTest: vi.fn(),
      phaseTestPixelChemistry: vi.fn(),
    },
    ebsdApi: { overview: vi.fn(), getPattern: vi.fn() },
    edsApi: { elements: vi.fn(), getMap: vi.fn() },
  };
});

import { indexApi, ebsdApi, edsApi } from '../../../services/api';
import useDataStore from '../../../stores/useDataStore';
import { parseScanGeometry } from '../../../services/scanGeometry';
import { makeHexMap } from '../../../services/hexMap';
import enIndexing from '../../../locales/en/indexing.json';
import enEbsdViewer from '../../../locales/en/ebsdviewer.json';
import SinglePixelPhaseTestDialog, { ScanPicker } from '../SinglePixelPhaseTestDialog';

// vitest runs from frontend/ (import.meta.url is not a file: URL in the jsdom environment)
const FIXTURE = JSON.parse(
  readFileSync(resolve(cwd(), '..', 'tests', 'fixtures', 'hex_raster_cases.json'), 'utf8'));
const HEX_CASE = FIXTURE.geometries.find((c) => c.name === 'a_3x3_right');
const HEX = HEX_CASE.geometry;
// What the backend stamps on the picture of this scan, in its own names
const STAMP = {
  kind: 'hex_raster', factor: HEX_CASE.spec.factor, shape: [HEX_CASE.spec.height, HEX_CASE.spec.width],
  step_um: HEX_CASE.spec.step_um, x0: HEX_CASE.spec.x0, y0: HEX_CASE.spec.y0,
};
const SQUARE = { kind: 'square', n_rows: 3, n_cols: 4, grid_shape: [3, 4], step_x: 1, step_y: 1 };

// The picker is 280 x 240 (an aspect of 14 / 12 in a 280 x 240 box).
const W = 280;
const H = 240;

/** The raster pixel (i, j) that holds the centre of point (row, col): the lattice by hand, odd rows by +0.5 step. */
const centrePixel = (row, col) => {
  const x = 2 * col + (row % 2 === 1 ? 1 : 0);          // um along a row
  const y = 1.7320508 * row;                              // um across rows
  return { i: Math.floor((y + 1.1547005) / 0.5), j: Math.floor((x + 1) / 0.5) };
};

const deferred = () => {
  let resolveIt;
  const promise = new Promise((r) => { resolveIt = r; });
  return { promise, resolve: resolveIt };
};

const holdStore = (geometry, gridShape) => useDataStore.setState({
  ebsdLoaded: true, isFileOpen: true, filePath: '/data/a/scan.up1', gridShape,
  patternCount: gridShape[0] * gridShape[1],
  currentIndex: 7, currentRow: 2, currentCol: 1, scanGeometry: parseScanGeometry(geometry),
});
const position = () => {
  const st = useDataStore.getState();
  return [st.currentRow, st.currentCol, st.currentIndex];
};

/**
 * What the browser reports for the picker box: `getBoundingClientRect` is the outer box with its frame, and
 * `clientLeft`/`clientTop`/`clientWidth`/`clientHeight` are the area inside the frame (`border` px wide).
 * jsdom lays nothing out, so the test says these numbers itself.
 */
const stubBox = (el, { left = 0, top = 0, border = 0, w = W, h = H } = {}) => {
  const rect = { left, top, width: w, height: h, right: left + w, bottom: top + h };
  Object.defineProperty(el, 'getBoundingClientRect', { value: () => rect, configurable: true, writable: true });
  Object.defineProperty(el, 'clientLeft', { value: border, configurable: true });
  Object.defineProperty(el, 'clientTop', { value: border, configurable: true });
  Object.defineProperty(el, 'clientWidth', { value: w - 2 * border, configurable: true });
  Object.defineProperty(el, 'clientHeight', { value: h - 2 * border, configurable: true });
};

const mount = async () => {
  const utils = render(<SinglePixelPhaseTestDialog open onClose={() => {}} />);
  const picker = await waitFor(() => {
    const p = document.querySelector('[data-scan-picker]');
    expect(p).toBeTruthy();
    return p;
  });
  // a box without a frame: the picture fills the whole rectangle
  stubBox(picker);
  await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
  return { ...utils, picker };
};
const clickAtPixel = (picker, i, j) => fireEvent.click(picker, {
  clientX: ((j + 0.5) / 14) * W, clientY: ((i + 0.5) / 12) * H,
});

beforeEach(() => {
  vi.clearAllMocks();
  ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64', display: STAMP } });
  ebsdApi.getPattern.mockResolvedValue({ data: { image: 'PATB64' } });
  indexApi.phaseTestPhases.mockResolvedValue({ data: [{ key: 'Al', label: 'Al', space_group: 'Fm-3m' }] });
  indexApi.phaseTestPixelChemistry.mockResolvedValue({ data: { pixel_index: 7, pixel_at_pct: null } });
  edsApi.elements.mockResolvedValue({ data: { elements: [] } });
  edsApi.getMap.mockResolvedValue({ data: { image: 'EDSB64' } });
  holdStore(HEX, [3, 3]);
});

describe('the phase test scan picker on a hexagonal scan', () => {
  it('has the aspect of the raster', async () => {
    const { picker } = await mount();
    expect(picker.style.width).toBe(`${W}px`);
    expect(picker.style.height).toBe(`${H}px`);
  });

  it('asks the backend for the raster', async () => {
    await mount();
    expect(ebsdApi.overview).toHaveBeenCalledWith('bc', { display: 'raster' });
  });

  it('picks the point under the click: raster pixel (5, 8) holds the centre of point (1, 1) = index 4', async () => {
    const { picker } = await mount();
    fireEvent.click(picker, { clientX: (8.5 / 14) * W, clientY: (5.5 / 12) * H });
    expect(position()).toEqual([1, 1, 4]);
  });

  it('picks exactly the (row, col) of the cell clicked, for every cell of the scan', async () => {
    const { picker } = await mount();
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const { i, j } = centrePixel(row, col);
        clickAtPixel(picker, i, j);
        expect(position()).toEqual([row, col, row * 3 + col]);
      }
    }
  });

  it('picks nothing in a corner no hexagon covers', async () => {
    const { picker } = await mount();
    fireEvent.click(picker, { clientX: (0.5 / 14) * W, clientY: (0.5 / 12) * H });
    expect(position()).toEqual([2, 1, 7]);
  });

  it('draws its box border-box, so the frame is inside the 280 x 240 it reports', async () => {
    const { picker } = await mount();
    expect(picker.style.boxSizing).toBe('border-box');
  });

  it('puts the crosshair on the hexagon of the active point', async () => {
    // index 7 = point (2, 1) at (2, 3.464) um; the raster is 14 x 12 pixels of 0.5 um from (-1, -1.155) um:
    // (2 + 1) um of 7 across, (3.464 + 1.155) um of 6 down -- the exact centre, not the middle of its pixel (9, 6)
    // The picture lies inside the 1 px frame of the 280 x 240 box, so the crosshair is drawn over 278 x 238.
    const { picker } = await mount();
    const circle = picker.querySelector('svg circle');
    expect(picker.querySelector('svg').getAttribute('width')).toBe('278');
    expect(picker.querySelector('svg').getAttribute('height')).toBe('238');
    expect(Number(circle.getAttribute('cx'))).toBeCloseTo((3 / 7) * (W - 2), 6);
    expect(Number(circle.getAttribute('cy'))).toBeCloseTo(((4 * Math.sqrt(3)) / 9) * (H - 2), 6);
  });

  it('measures a click against the area inside the 1 px frame, where the picture lies', async () => {
    const { picker } = await mount();
    // the browser's numbers for a box at (100.8, 50.3) with a 1 px frame: outer 280 x 240, inside 278 x 238 from (101.8, 51.3)
    stubBox(picker, { left: 100.8, top: 50.3, border: 1 });
    // The cell boundary between points (0, 0) and (0, 1) is x = 1 um: the raster pixels 3 and 4 meet at 4 / 14 of the
    // width (x0 = -1 um, 0.5 um per pixel: -1 + 4 * 0.5 = 1). Row 0 at y = 0 um is raster row 2 (-1.155 + 2 * 0.5 = -0.155 to 0.345),
    // whose pixels 3 and 4 have their centres at x = 0.75 and 1.25 um: nearest points (0, 0) and (0, 1), both inside.
    // The click is at clientX 181, clientY 101.
    //   inside the frame: (181 - 101.8) / 278 = 0.28489, * 14 = 3.9885 -> pixel 3 -> point (0, 0)
    //                     (101 - 51.3) / 238 = 0.20882, * 12 = 2.506 -> pixel row 2
    //   against the outer box (without the correction): (181 - 100.8) / 280 = 0.28643, * 14 = 4.010 -> pixel 4 -> point (0, 1)
    fireEvent.click(picker, { clientX: 181, clientY: 101 });
    expect(position()).toEqual([0, 0, 0]);
  });

  it('measures a click on the zoomed picture against the area inside the frame, too', () => {
    const picks = [];
    const hexMap = makeHexMap(parseScanGeometry(HEX));
    const { container } = render(
      <ScanPicker gridShape={[3, 3]} hexMap={hexMap} pixelIndex={7} overviewB64={null}
        view={{ scale: 2, cx: 0.5, cy: 0.5 }} onPick={(r, c) => picks.push([r, c])} t={(k) => k} />);
    const picker = container.querySelector('[data-scan-picker]');
    // a box at (100.5, 150) with a 1 px frame: outer 280 x 240, inside 278 x 238 from (101.5, 151)
    stubBox(picker, { left: 100.5, top: 150, border: 1 });
    // Zoom 2 about the middle: the picture inside the frame is 556 x 476 and lies from
    //   x = 101.5 + 278 / 2 - 0.5 * 556 = -37.5   and   y = 151 + 238 / 2 - 0.5 * 476 = 32.
    // The click is at clientX 121, clientY 131 (the pixels 3 and 4 meet at 4 / 14 * 556 = 158.857 from the left edge).
    //   inside the frame: (121 + 37.5) / 556 = 0.28507, * 14 = 3.991 -> pixel 3 -> point (0, 0)
    //                     (131 - 32) / 476 = 0.20798, * 12 = 2.496 -> pixel row 2
    //   against the outer box, zoomed (without the correction): left = 100.5 + 140 - 280 = -39.5, top = 150 + 120 - 240 = 30,
    //                     (121 + 39.5) / 560 = 0.28661, * 14 = 4.013 -> pixel 4 -> point (0, 1)
    fireEvent.click(picker, { clientX: 121, clientY: 131 });
    expect(picks).toEqual([[0, 0]]);
  });

  it('offers no EDS element buttons and fetches no element list or map for a picture that is not the index grid', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: ['Al'] } });
    const { picker, queryByRole, queryByText } = await mount();
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(queryByRole('button', { name: 'Al' })).toBeNull();
    expect(queryByText(enIndexing.phaseTest.edsOverlay)).toBeNull();
    expect(edsApi.elements).not.toHaveBeenCalled();
    expect(edsApi.getMap).not.toHaveBeenCalled();
    expect(picker.querySelector('img[src$="EDSB64"]')).toBeNull();
  });

  it('shows the picture the backend drew on the raster', async () => {
    const { picker } = await mount();
    expect(picker.querySelector('img[src$="OVB64"]')).toBeTruthy();
  });

  it('shows no picture of another shape', async () => {
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64', display: { ...STAMP, shape: [3, 3], factor: 1 } } });
    const { picker } = await mount();
    expect(ebsdApi.overview).toHaveBeenCalled();
    expect(picker.querySelector('img[src$="OVB64"]')).toBeNull();
  });

  it('shows no picture that has no raster at all for a hexagonal scan', async () => {
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64' } });
    const { picker } = await mount();
    expect(picker.querySelector('img[src$="OVB64"]')).toBeNull();
  });
});

describe('the phase test scan picker when the picture cannot be had on a hexagonal scan', () => {
  const gone = (picker, text) => {
    expect(picker.querySelector('img[src$="OVB64"]')).toBeNull();
    expect(picker.textContent).toContain(text);
    expect(picker.textContent).not.toContain(enIndexing.phaseTest.loadingOverview);
  };

  it('says the overview failed when the request is rejected, instead of loading for ever', async () => {
    ebsdApi.overview.mockRejectedValue(new Error('500'));
    const { picker } = await mount();
    gone(picker, enEbsdViewer.overview.failed);
  });

  it('says the overview failed when the answer carries no picture', async () => {
    ebsdApi.overview.mockResolvedValue({ data: { display: STAMP } });
    const { picker } = await mount();
    gone(picker, enEbsdViewer.overview.failed);
  });

  it('says the picture does not match the layout when it is not this scan\'s raster', async () => {
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64', display: { ...STAMP, shape: [3, 3], factor: 1 } } });
    const { picker } = await mount();
    gone(picker, enEbsdViewer.overview.rasterMismatch);
  });

  it('shows the loading text, not an error, while the request is still out', async () => {
    ebsdApi.overview.mockReturnValue(new Promise(() => {}));
    const { picker } = await mount();
    expect(picker.textContent).toContain(enIndexing.phaseTest.loadingOverview);
    expect(picker.textContent).not.toContain(enEbsdViewer.overview.failed);
  });

  it('drops the error once the measurement changes, and shows a good answer', async () => {
    ebsdApi.overview.mockRejectedValueOnce(new Error('500'));
    const { picker } = await mount();
    expect(picker.textContent).toContain(enEbsdViewer.overview.failed);
    act(() => { holdStore(HEX, [3, 3]); });
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(picker.querySelector('img[src$="OVB64"]')).toBeTruthy();
    expect(picker.textContent).not.toContain(enEbsdViewer.overview.failed);
  });
});

describe('the phase test scan picker on a hexagonal scan while zoomed', () => {
  // Four wheel notches in (factor 1.25 each, so scale 1.25^4 = 2.44140625) with the pointer at screen (210, 60) =
  // (0.75, 0.25) of the 280 x 240 box. The picture point under the pointer stays under it, so the picture's left
  // and top edge sit at 210 * (1 - 2.44140625) = -302.6953125 and 60 * (1 - 2.44140625) = -86.484375, and one
  // raster pixel (20 px square unzoomed) is 20 * 2.44140625 = 48.828125 px square. Pixels (i, j) = (2..6, 6..11)
  // are the ones inside the box.
  const LEFT = -302.6953125;
  const TOP = -86.484375;
  const PX = 48.828125;
  const zoomIn = (picker) => {
    for (let k = 0; k < 4; k++) {
      fireEvent.wheel(picker, { deltaY: -100, clientX: 0.75 * W, clientY: 0.25 * H });
    }
  };
  const clickZoomedPixel = (picker, i, j) => fireEvent.click(picker, {
    clientX: LEFT + (j + 0.5) * PX, clientY: TOP + (i + 0.5) * PX,
  });

  it('names the point under the click in the picture, not in the box: pixel (5, 8) is point (1, 1) = index 4', async () => {
    const { picker } = await mount();
    zoomIn(picker);
    // screen (112.34, 182.07); read against the box that would be raster pixel (9, 5), which lies in point (2, 1)
    clickZoomedPixel(picker, 5, 8);
    expect(position()).toEqual([1, 1, 4]);
  });

  it('names the right point for other visible cells: pixel (2, 6) is point (0, 1), pixel (2, 10) is point (0, 2)', async () => {
    const { picker } = await mount();
    zoomIn(picker);
    clickZoomedPixel(picker, 2, 6);    // screen (14.65, 35.64); read against the box: raster pixel (1, 0)
    expect(position()).toEqual([0, 1, 1]);
    clickZoomedPixel(picker, 2, 10);   // screen (209.96, 35.64)
    expect(position()).toEqual([0, 2, 2]);
  });

  it('carries the crosshair with the picture: it stays at the point\'s place in the picture inside the zoomed layer', async () => {
    const { picker } = await mount();
    zoomIn(picker);
    const circle = picker.querySelector('svg circle');
    // index 7 = point (2, 1), at 3 / 7 across and 4 sqrt(3) / 9 down the picture: unchanged by the zoom ...
    // (drawn over the 278 x 238 area inside the 1 px frame)
    expect(Number(circle.getAttribute('cx'))).toBeCloseTo((3 / 7) * (W - 2), 6);
    expect(Number(circle.getAttribute('cy'))).toBeCloseTo(((4 * Math.sqrt(3)) / 9) * (H - 2), 6);
    // ... and the layer that holds picture and crosshair together is the one that is scaled
    const layer = circle.closest('div[style*="scale("]');
    expect(layer).toBeTruthy();
    expect(layer.style.transform).toContain('scale(2.44140625)');
    expect(layer.contains(picker.querySelector('img[src$="OVB64"]'))).toBe(true);
  });
});

describe('the phase test scan picker when the measurement changes under it', () => {
  it('withholds the old picture until the new one is in, and a late answer does not replace the new one', async () => {
    const first = deferred();
    const second = deferred();
    ebsdApi.overview.mockReset();
    ebsdApi.overview.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const { picker } = await mount();
    // the first measurement's picture is still on its way when the store moves to another measurement of equal size
    act(() => { holdStore(HEX, [3, 3]); });
    expect(ebsdApi.overview).toHaveBeenCalledTimes(2);
    await act(async () => { second.resolve({ data: { image: 'NEWB64', display: STAMP } }); });
    expect(picker.querySelector('img[src$="NEWB64"]')).toBeTruthy();
    await act(async () => { first.resolve({ data: { image: 'OLDB64', display: STAMP } }); });
    expect(picker.querySelector('img[src$="OLDB64"]')).toBeNull();
    expect(picker.querySelector('img[src$="NEWB64"]')).toBeTruthy();
  });

  it('shows the loading text, and no picture, while the answer for the new measurement is pending', async () => {
    const { picker } = await mount();
    expect(picker.querySelector('img[src$="OVB64"]')).toBeTruthy();
    ebsdApi.overview.mockReturnValue(new Promise(() => {}));
    act(() => { holdStore(HEX, [3, 3]); });
    expect(picker.querySelector('img[src$="OVB64"]')).toBeNull();
    expect(picker.textContent).toContain(enIndexing.phaseTest.loadingOverview);
  });

  it('shows no picture of the previous file while the new file is answered, whatever its geometry', async () => {
    const { picker } = await mount();
    expect(picker.querySelector('img[src$="OVB64"]')).toBeTruthy();
    ebsdApi.overview.mockReturnValue(new Promise(() => {}));
    // another file path, the very same geometry object
    act(() => { useDataStore.setState({ filePath: '/data/b/other.up1' }); });
    expect(picker.querySelector('img[src$="OVB64"]')).toBeNull();
  });

  it('shows no raster picture once the store holds a square scan', async () => {
    const { picker } = await mount();
    expect(picker.querySelector('img[src$="OVB64"]')).toBeTruthy();
    ebsdApi.overview.mockReturnValue(new Promise(() => {}));
    act(() => { holdStore(SQUARE, [3, 4]); });
    expect(picker.querySelector('img[src$="OVB64"]')).toBeNull();
  });
});

describe('the phase test scan picker when a hexagonal scan gives way to a square one', () => {
  const textBox = (picker, text) => [...picker.querySelectorAll('div')]
    .find((d) => d.children.length === 0 && d.textContent === text);
  const toSquare = async () => {
    act(() => { holdStore(SQUARE, [3, 4]); });
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
  };

  it('says the overview failed, at full opacity, when the new request is rejected', async () => {
    const { picker } = await mount();
    expect(picker.querySelector('img[src$="OVB64"]')).toBeTruthy();
    ebsdApi.overview.mockRejectedValue(new Error('500'));
    await toSquare();
    expect(picker.textContent).toContain(enEbsdViewer.overview.failed);
    expect(picker.textContent).not.toContain(enIndexing.phaseTest.loadingOverview);
    expect(textBox(picker, enEbsdViewer.overview.failed).style.opacity).toBe('1');
  });

  it('says the overview failed when the new answer carries no picture', async () => {
    const { picker } = await mount();
    ebsdApi.overview.mockResolvedValue({ data: {} });
    await toSquare();
    expect(picker.textContent).toContain(enEbsdViewer.overview.failed);
    expect(picker.textContent).not.toContain(enIndexing.phaseTest.loadingOverview);
  });

  it('shows the loading text, dimmed, while the new request is out', async () => {
    const { picker } = await mount();
    ebsdApi.overview.mockReturnValue(new Promise(() => {}));
    await toSquare();
    expect(picker.textContent).not.toContain(enEbsdViewer.overview.failed);
    expect(textBox(picker, enIndexing.phaseTest.loadingOverview).style.opacity).toBe('0.5');
  });

  it('shows a good answer for the square scan, and no error after it', async () => {
    const { picker } = await mount();
    ebsdApi.overview.mockResolvedValue({ data: { image: 'SQB64' } });
    await toSquare();
    expect(picker.querySelector('img[src$="SQB64"]')).toBeTruthy();
    expect(picker.textContent).not.toContain(enEbsdViewer.overview.failed);
  });
});

describe('the phase test scan picker on a square scan', () => {
  beforeEach(() => {
    holdStore(SQUARE, [3, 4]);
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64' } });
  });

  it('keeps the arithmetic of the grid: the picture is shown and a click names the cell under it', async () => {
    const { picker } = await mount();
    expect(picker.querySelector('img[src$="OVB64"]')).toBeTruthy();
    // 4 x 3 cells: the box is 280 x 210 (an aspect of 4 / 3); the cell (2, 3) is the bottom right one
    expect(picker.style.height).toBe('210px');
    picker.getBoundingClientRect = () => ({ left: 0, top: 0, width: 280, height: 210, right: 280, bottom: 210 });
    fireEvent.click(picker, { clientX: 279, clientY: 209 });
    expect(position()).toEqual([2, 3, 11]);
    fireEvent.click(picker, { clientX: 1, clientY: 1 });
    expect(position()).toEqual([0, 0, 0]);
  });

  it('is drawn as before the frame fix: content-box, so the picture and the crosshair keep the full 280 x 210', async () => {
    const { picker } = await mount();
    // border-box would shrink the box by the frame (to 278 x 208) while the crosshair kept 280 x 210
    expect(picker.style.boxSizing).not.toBe('border-box');
    expect(picker.style.width).toBe('280px');
    expect(picker.style.height).toBe('210px');
    expect(picker.querySelector('svg').getAttribute('width')).toBe('280');
    expect(picker.querySelector('svg').getAttribute('height')).toBe('210');
  });

  it('measures a click against the outer box, not the area inside the frame', async () => {
    const { picker } = await mount();
    // a content-box 280 x 210 box at (100, 50) with a 1 px frame: getBoundingClientRect is the outer box,
    // clientWidth/Height the inside. 4 x 3 cells in 280 x 210 are 70 x 70 each. Click at 70.2 and 105 from the corner:
    // 70.2 / 70 = 1.003 -> column 1; 105 / 70 = 1.5 -> row 1; index 1 * 4 + 1 = 5.
    // Against the inside (from 1, 278 wide) it would be (70.2 - 1) / 69.5 = 0.9957 -> column 0 -> index 4.
    stubBox(picker, { left: 100, top: 50, border: 1, w: 280, h: 210 });
    fireEvent.click(picker, { clientX: 170.2, clientY: 155 });
    expect(position()).toEqual([1, 1, 5]);
  });

  it('keeps the EDS element buttons, and draws the element map over the grid picture', async () => {
    edsApi.elements.mockResolvedValue({ data: { elements: ['Al'] } });
    const { picker, findByRole } = await mount();
    fireEvent.click(await findByRole('button', { name: 'Al' }));
    await waitFor(() => expect(edsApi.getMap).toHaveBeenCalled());
    await waitFor(() => expect(picker.querySelector('img[src$="EDSB64"]')).toBeTruthy());
  });

  it('keeps the loading text, with no error text, when the request is rejected', async () => {
    ebsdApi.overview.mockRejectedValue(new Error('500'));
    const { picker } = await mount();
    expect(picker.textContent).toContain(enIndexing.phaseTest.loadingOverview);
    expect(picker.textContent).not.toContain(enEbsdViewer.overview.failed);
  });

  it('keeps the loading text, with no error text, when the answer carries no picture', async () => {
    ebsdApi.overview.mockResolvedValue({ data: {} });
    const { picker } = await mount();
    expect(picker.textContent).toContain(enIndexing.phaseTest.loadingOverview);
    expect(picker.textContent).not.toContain(enEbsdViewer.overview.failed);
  });

  it('is a plain grid picker for a hexagonal file the loader resampled to a square map', async () => {
    // the loader holds such a file as a square scan (its geometry says "square", the answer carries no
    // display raster); the flag only tells the user, it never switches the picker to the hexagonal path
    useDataStore.setState({ hexResampled: true });
    try {
      const { picker } = await mount();
      expect(picker.querySelector('img[src$="OVB64"]')).toBeTruthy();
      expect(picker.style.height).toBe('210px');
      picker.getBoundingClientRect = () => ({ left: 0, top: 0, width: 280, height: 210, right: 280, bottom: 210 });
      fireEvent.click(picker, { clientX: 279, clientY: 209 });
      expect(position()).toEqual([2, 3, 11]);
    } finally {
      useDataStore.setState({ hexResampled: null });
    }
  });
});

describe('ScanPicker alone', () => {
  it('is exported and takes the map as a prop: without one it is the grid picker', () => {
    const picks = [];
    const { container } = render(
      <ScanPicker gridShape={[3, 4]} pixelIndex={5} overviewB64={null} onPick={(r, c) => picks.push([r, c])}
        t={(k) => k} />);
    const picker = container.querySelector('[data-scan-picker]');
    picker.getBoundingClientRect = () => ({ left: 0, top: 0, width: 280, height: 210, right: 280, bottom: 210 });
    fireEvent.click(picker, { clientX: 140, clientY: 105 });
    expect(picks).toEqual([[1, 2]]);
  });
});
