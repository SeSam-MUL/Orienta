// @vitest-environment jsdom
/**
 * The Crystal Hint single-pixel picker on a hexagonal scan, through the real component and the real data store:
 * the picture is the display raster, a click names the measurement point under it (none in a corner no hexagon
 * covers), the crosshair sits on its hexagon, and a picture that is not the expected raster, or that was fetched
 * for another measurement, is not shown.
 *
 * The scan is a made-up 3 x 3 one with 2 um between points in a row, a row pitch of 1.732 um and odd rows shifted
 * by +0.5 of a step; its raster is 12 x 14 pixels of 0.5 um starting at (-1, -1.155) um
 * (tests/fixtures/hex_raster_cases.json, 'a_3x3_right'). Expectations are worked by hand below.
 */
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { render, cleanup, fireEvent, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { cwd } from 'node:process';

vi.mock('../../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    ebsdApi: { overview: vi.fn() },
    crystalHintApi: { analyzePixel: vi.fn(), externalSearch: vi.fn() },
  };
});
vi.mock('./SymmetryReportPanel', () => ({ default: () => null }));
vi.mock('./LatticeReportPanel', () => ({ default: () => null }));
vi.mock('./CandidateList', () => ({ default: () => null }));
vi.mock('./PatternPreview', () => ({ default: () => null }));

import SinglePixelMode from './SinglePixelMode';
import { ebsdApi } from '../../services/api';
import useDataStore from '../../stores/useDataStore';
import { parseScanGeometry } from '../../services/scanGeometry';
import enCrystalHint from '../../locales/en/crystalhint.json';
import enEbsdViewer from '../../locales/en/ebsdviewer.json';

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
const SQUARE = { kind: 'square', n_rows: 10, n_cols: 12, grid_shape: [10, 12], step_x: 1, step_y: 1 };

// The picker is 320 x 274 (an aspect of 14 / 12 in a 320 x 280 box).
const W = 320;
const H = 274;

/** The raster pixel (i, j) that holds the centre of point (row, col): the lattice by hand, odd rows by +0.5 step. */
const centrePixel = (row, col) => {
  const x = 2 * col + (row % 2 === 1 ? 1 : 0);          // um along a row
  const y = 1.7320508 * row;                              // um across rows
  return { i: Math.floor((y + 1.1547005) / 0.5), j: Math.floor((x + 1) / 0.5) };
};
const clickAtPixel = (picker, i, j) => fireEvent.click(picker, {
  clientX: ((j + 0.5) / 14) * W, clientY: ((i + 0.5) / 12) * H,
});

const deferred = () => {
  let resolveIt;
  const promise = new Promise((r) => { resolveIt = r; });
  return { promise, resolve: resolveIt };
};

const holdStore = (geometry, gridShape) => useDataStore.setState({
  ebsdLoaded: true, isFileOpen: false, gridShape, patternCount: gridShape[0] * gridShape[1],
  currentIndex: 7, currentRow: 2, currentCol: 1, scanGeometry: parseScanGeometry(geometry),
});

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

const mount = async (element = <SinglePixelMode elements={[]} isActive />) => {
  const utils = render(element);
  const picker = [...utils.container.querySelectorAll('div')].find((d) => d.style.cursor === 'crosshair');
  // a box without a frame: the picture fills the whole rectangle
  stubBox(picker);
  await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
  return { ...utils, picker, input: utils.container.querySelector('input[type="number"]') };
};

beforeEach(() => {
  vi.clearAllMocks();
  ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64', display: STAMP } });
  holdStore(HEX, [3, 3]);
});
afterEach(() => cleanup());

describe('the Crystal Hint scan picker on a hexagonal scan', () => {
  it('has the aspect of the raster, not of the 3 x 3 index grid', async () => {
    const { picker } = await mount();
    expect(picker.style.width).toBe(`${W}px`);
    expect(picker.style.height).toBe(`${H}px`);
  });

  it('asks the backend for the raster', async () => {
    await mount();
    expect(ebsdApi.overview).toHaveBeenCalledWith('bc', { display: 'raster' });
  });

  it('picks the point under the click: raster pixel (5, 8) holds the centre of point (1, 1) = index 4', async () => {
    const { picker, input } = await mount();
    fireEvent.click(picker, { clientX: (8.5 / 14) * W, clientY: (5.5 / 12) * H });
    expect(input.value).toBe('4');
    const st = useDataStore.getState();
    expect([st.currentRow, st.currentCol, st.currentIndex]).toEqual([1, 1, 4]);
  });

  it('picks exactly the (row, col) of the cell clicked, for every cell of the scan', async () => {
    const { picker, input } = await mount();
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const { i, j } = centrePixel(row, col);
        clickAtPixel(picker, i, j);
        const st = useDataStore.getState();
        expect([st.currentRow, st.currentCol, st.currentIndex]).toEqual([row, col, row * 3 + col]);
        expect(input.value).toBe(String(row * 3 + col));
      }
    }
  });

  it('picks nothing in a corner no hexagon covers', async () => {
    const { picker, input } = await mount();
    fireEvent.click(picker, { clientX: (0.5 / 14) * W, clientY: (0.5 / 12) * H });
    expect(input.value).toBe('7');
    const st = useDataStore.getState();
    expect([st.currentRow, st.currentCol, st.currentIndex]).toEqual([2, 1, 7]);
  });

  it('draws its box border-box, so the frame is inside the 320 x 274 it reports', async () => {
    const { picker } = await mount();
    expect(picker.style.boxSizing).toBe('border-box');
  });

  it('puts the crosshair on the hexagon of the active point', async () => {
    // index 7 = point (2, 1) at (2, 3.464) um; the raster is 14 x 12 pixels of 0.5 um from (-1, -1.155) um:
    // (2 + 1) um of 7 across, (3.464 + 1.155) um of 6 down -- the exact centre, not the middle of its pixel (9, 6).
    // The picture lies inside the 1 px frame of the 320 x 274 box, so the crosshair is drawn over 318 x 272.
    const { container } = await mount();
    const circle = container.querySelector('svg circle');
    expect(container.querySelector('svg').getAttribute('width')).toBe('318');
    expect(container.querySelector('svg').getAttribute('height')).toBe('272');
    expect(Number(circle.getAttribute('cx'))).toBeCloseTo((3 / 7) * (W - 2), 6);
    expect(Number(circle.getAttribute('cy'))).toBeCloseTo(((4 * Math.sqrt(3)) / 9) * (H - 2), 6);
  });

  it('measures a click against the area inside the 1 px frame, where the picture lies', async () => {
    const { picker } = await mount();
    // the browser's numbers for a box at (100.4, 50.3) with a 1 px frame: outer 320 x 274, inside 318 x 272 from (101.4, 51.3)
    stubBox(picker, { left: 100.4, top: 50.3, border: 1 });
    // The cell boundary between points (0, 0) and (0, 1) is x = 1 um: the raster pixels 3 and 4 meet at 4 / 14 of the
    // width (x0 = -1 um, 0.5 um per pixel: -1 + 4 * 0.5 = 1). Row 0 at y = 0 um is raster row 2 (-1.155 + 2 * 0.5 = -0.155 to 0.345),
    // whose pixels 3 and 4 have their centres at x = 0.75 and 1.25 um: nearest points (0, 0) and (0, 1), both inside.
    // The click is at clientX 192, clientY 108.
    //   inside the frame: (192 - 101.4) / 318 = 0.28491, * 14 = 3.989 -> pixel 3 -> point (0, 0)
    //                     (108 - 51.3) / 272 = 0.20846, * 12 = 2.502 -> pixel row 2
    //   against the outer box (without the correction): (192 - 100.4) / 320 = 0.28625, * 14 = 4.0075 -> pixel 4 -> point (0, 1)
    fireEvent.click(picker, { clientX: 192, clientY: 108 });
    const st = useDataStore.getState();
    expect([st.currentRow, st.currentCol, st.currentIndex]).toEqual([0, 0, 0]);
  });

  it('shows the picture the backend drew on the raster', async () => {
    const { container } = await mount();
    expect(container.querySelector('img[src$="OVB64"]')).toBeTruthy();
  });

  it('shows no picture that is not the raster of this scan', async () => {
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64', display: { ...STAMP, shape: [3, 3], factor: 1 } } });
    const { container } = await mount();
    expect(ebsdApi.overview).toHaveBeenCalled();
    expect(container.querySelector('img[src$="OVB64"]')).toBeNull();
  });

  it('shows no picture that has no raster at all for a hexagonal scan', async () => {
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64' } });
    const { container } = await mount();
    expect(ebsdApi.overview).toHaveBeenCalled();
    expect(container.querySelector('img[src$="OVB64"]')).toBeNull();
  });
});

describe('the Crystal Hint scan picker when the picture cannot be had on a hexagonal scan', () => {
  const gone = (container, text) => {
    expect(container.querySelector('img[src$="OVB64"]')).toBeNull();
    expect(container.textContent).toContain(text);
    expect(container.textContent).not.toContain(enCrystalHint.single.loadingOverview);
  };

  it('says the overview failed when the request is rejected, instead of loading for ever', async () => {
    ebsdApi.overview.mockRejectedValue(new Error('500'));
    const { container } = await mount();
    gone(container, enEbsdViewer.overview.failed);
  });

  it('says the overview failed when the answer carries no picture', async () => {
    ebsdApi.overview.mockResolvedValue({ data: { display: STAMP } });
    const { container } = await mount();
    gone(container, enEbsdViewer.overview.failed);
  });

  it('says the picture does not match the layout when it is not this scan\'s raster', async () => {
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64', display: { ...STAMP, shape: [3, 3], factor: 1 } } });
    const { container } = await mount();
    gone(container, enEbsdViewer.overview.rasterMismatch);
  });

  it('shows the loading text, not an error, while the request is still out', async () => {
    ebsdApi.overview.mockReturnValue(new Promise(() => {}));
    const { container } = await mount();
    expect(container.textContent).toContain(enCrystalHint.single.loadingOverview);
    expect(container.textContent).not.toContain(enEbsdViewer.overview.failed);
  });

  it('drops the error once the measurement changes and its answer is pending, and shows a good answer', async () => {
    ebsdApi.overview.mockRejectedValueOnce(new Error('500'));
    const { container } = await mount();
    expect(container.textContent).toContain(enEbsdViewer.overview.failed);
    act(() => { holdStore(HEX, [3, 3]); });
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(container.querySelector('img[src$="OVB64"]')).toBeTruthy();
    expect(container.textContent).not.toContain(enEbsdViewer.overview.failed);
  });
});

describe('the Crystal Hint scan picker when the measurement changes under it', () => {
  it('withholds the old picture until the new one is in, and a late answer does not replace the new one', async () => {
    const first = deferred();
    const second = deferred();
    ebsdApi.overview.mockReset();
    ebsdApi.overview.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const { container } = await mount();
    // the first measurement's picture is still on its way when the store moves to another measurement of equal size
    act(() => { holdStore(HEX, [3, 3]); });
    expect(ebsdApi.overview).toHaveBeenCalledTimes(2);
    await act(async () => { second.resolve({ data: { image: 'NEWB64', display: STAMP } }); });
    expect(container.querySelector('img[src$="NEWB64"]')).toBeTruthy();
    await act(async () => { first.resolve({ data: { image: 'OLDB64', display: STAMP } }); });
    expect(container.querySelector('img[src$="OLDB64"]')).toBeNull();
    expect(container.querySelector('img[src$="NEWB64"]')).toBeTruthy();
  });

  it('shows the loading text, and no picture, while the answer for the new measurement is pending', async () => {
    const { container } = await mount();
    expect(container.querySelector('img[src$="OVB64"]')).toBeTruthy();
    ebsdApi.overview.mockReturnValue(new Promise(() => {}));
    act(() => { holdStore(HEX, [3, 3]); });
    expect(container.querySelector('img[src$="OVB64"]')).toBeNull();
    expect(container.textContent).toContain(enCrystalHint.single.loadingOverview);
  });

  it('shows no raster picture once the store holds a square scan', async () => {
    const { container } = await mount();
    expect(container.querySelector('img[src$="OVB64"]')).toBeTruthy();
    ebsdApi.overview.mockReturnValue(new Promise(() => {}));
    act(() => { holdStore(SQUARE, [10, 12]); });
    expect(container.querySelector('img[src$="OVB64"]')).toBeNull();
  });
});

describe('the Crystal Hint scan picker when a hexagonal scan gives way to a square one', () => {
  const textBox = (container, text) => [...container.querySelectorAll('div')]
    .find((d) => d.children.length === 0 && d.textContent === text);
  const toSquare = async () => {
    act(() => { holdStore(SQUARE, [10, 12]); });
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
  };

  it('says the overview failed, at full opacity, when the new request is rejected', async () => {
    const { container } = await mount();
    expect(container.querySelector('img[src$="OVB64"]')).toBeTruthy();
    ebsdApi.overview.mockRejectedValue(new Error('500'));
    await toSquare();
    expect(container.textContent).toContain(enEbsdViewer.overview.failed);
    expect(container.textContent).not.toContain(enCrystalHint.single.loadingOverview);
    expect(textBox(container, enEbsdViewer.overview.failed).style.opacity).toBe('1');
  });

  it('says the overview failed when the new answer carries no picture', async () => {
    const { container } = await mount();
    ebsdApi.overview.mockResolvedValue({ data: {} });
    await toSquare();
    expect(container.textContent).toContain(enEbsdViewer.overview.failed);
    expect(container.textContent).not.toContain(enCrystalHint.single.loadingOverview);
  });

  it('shows the loading text, dimmed, while the new request is out', async () => {
    const { container } = await mount();
    ebsdApi.overview.mockReturnValue(new Promise(() => {}));
    await toSquare();
    expect(container.textContent).not.toContain(enEbsdViewer.overview.failed);
    expect(textBox(container, enCrystalHint.single.loadingOverview).style.opacity).toBe('0.5');
  });

  it('shows a good answer for the square scan, and no error after it', async () => {
    const { container } = await mount();
    ebsdApi.overview.mockResolvedValue({ data: { image: 'SQB64' } });
    await toSquare();
    expect(container.querySelector('img[src$="SQB64"]')).toBeTruthy();
    expect(container.textContent).not.toContain(enEbsdViewer.overview.failed);
  });
});

describe('the Crystal Hint scan picker on a square scan', () => {
  beforeEach(() => {
    holdStore(SQUARE, [10, 12]);
    ebsdApi.overview.mockResolvedValue({ data: { image: 'OVB64' } });
  });
  afterEach(() => { useDataStore.setState({ hexResampled: null }); });

  it('keeps the arithmetic of the grid: the picture is shown and a click names the cell under it', async () => {
    const { picker, input, container } = await mount();
    expect(container.querySelector('img[src$="OVB64"]')).toBeTruthy();
    // 12 x 10 cells, an aspect of 1.2: the box is 320 x 267, and its middle is cell (5, 6)
    expect(picker.style.height).toBe('267px');
    fireEvent.click(picker, { clientX: W / 2, clientY: H / 2 });
    expect(input.value).toBe('66');
    // the bottom right corner is the last cell
    fireEvent.click(picker, { clientX: W - 1, clientY: H - 1 });
    expect(input.value).toBe('119');
  });

  it('is drawn as before the frame fix: content-box, so the picture and the crosshair keep the full 320 x 267', async () => {
    const { picker, container } = await mount();
    // border-box would shrink the box by the frame (to 318 x 265) while the crosshair kept 320 x 267
    expect(picker.style.boxSizing).not.toBe('border-box');
    expect(picker.style.width).toBe('320px');
    expect(picker.style.height).toBe('267px');
    expect(container.querySelector('svg').getAttribute('width')).toBe('320');
    expect(container.querySelector('svg').getAttribute('height')).toBe('267');
  });

  it('measures a click against the outer box, not the area inside the frame', async () => {
    const { picker, input } = await mount();
    // the browser's numbers for a content-box 320 x 267 box at (100, 50) with a 1 px frame: getBoundingClientRect is
    // the outer box, clientWidth/Height the inside; a square scan keeps measuring against the outer box.
    // 12 x 10 cells in 320 x 267: a cell is 26.667 wide and 26.7 high. Click at x = 27 and y = 140 from the corner:
    // 27 / 26.667 = 1.0125 -> column 1; 140 / 26.7 = 5.24 -> row 5; index 5 * 12 + 1 = 61.
    // Against the inside (from 1, 318 wide) it would be (27 - 1) / 26.5 = 0.98 -> column 0 -> index 60.
    stubBox(picker, { left: 100, top: 50, border: 1, w: 320, h: 267 });
    fireEvent.click(picker, { clientX: 127, clientY: 190 });
    expect(input.value).toBe('61');
  });

  it('keeps the loading text, with no error text, when the request is rejected', async () => {
    ebsdApi.overview.mockRejectedValue(new Error('500'));
    const { container } = await mount();
    expect(container.textContent).toContain(enCrystalHint.single.loadingOverview);
    expect(container.textContent).not.toContain(enEbsdViewer.overview.failed);
  });

  it('keeps the loading text, with no error text, when the answer carries no picture', async () => {
    ebsdApi.overview.mockResolvedValue({ data: {} });
    const { container } = await mount();
    expect(container.textContent).toContain(enCrystalHint.single.loadingOverview);
    expect(container.textContent).not.toContain(enEbsdViewer.overview.failed);
  });

  it('is a plain grid picker for a hexagonal file the loader resampled to a square map', async () => {
    // the loader holds such a file as a square scan (its geometry says "square", the answer carries no
    // display raster); the flag only tells the user, it never switches the picker to the hexagonal path
    act(() => { useDataStore.setState({ hexResampled: true }); });
    const { picker, input, container } = await mount();
    expect(ebsdApi.overview).toHaveBeenCalledWith('bc', { display: 'raster' });
    expect(container.querySelector('img[src$="OVB64"]')).toBeTruthy();
    expect(picker.style.height).toBe('267px');
    fireEvent.click(picker, { clientX: W / 2, clientY: H / 2 });
    expect(input.value).toBe('66');
  });
});
