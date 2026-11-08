// @vitest-environment jsdom
//
// The Indexing page's navigation map, mounted with the backend mocked: the page asks for the display raster,
// hands the map it builds from the store's scan geometry to the canvas, and withholds a picture that is not
// the raster it expects. The canvas context is a recorder; the image a stub of the picture's size.
//
// Hexagonal case: a made-up 3 x 3 scan with 2 um between points in a row, row pitch 1.732 um, odd rows shifted
// by +0.5 of a step; its raster is 12 x 14 pixels of 0.5 um (tests/fixtures/hex_raster_cases.json, 'a_3x3_right').
import { describe, it, expect, vi, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { render, cleanup, fireEvent, waitFor, act } from '@testing-library/react';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { cwd } from 'node:process';

vi.mock('../../../services/api', async (importOriginal) => {
  const actual = await importOriginal();
  const fake = (base) => new Proxy(base, {
    get: (target, prop) => {
      if (prop === 'then' || typeof prop === 'symbol') return undefined;
      if (!(prop in target)) target[prop] = vi.fn(async () => ({ data: {} }));
      return target[prop];
    },
  });
  return {
    ...actual,
    indexApi: fake({}),
    ebsdApi: fake({}),
    pcApi: fake({}),
    edsApi: fake({}),
    dictionaryGpuApi: fake({}),
    phaseMapApi: fake({}),
    collectionsApi: fake({}),
    getGpuStatus: vi.fn(async () => ({ available: false, name: '', vram_total_gb: 0, vram_free_gb: 0 })),
  };
});

import IndexingPage from '../IndexingPage';
import { ebsdApi } from '../../../services/api';
import useDataStore from '../../../stores/useDataStore';
import useLoadedFilesStore from '../../../stores/useLoadedFilesStore';
import { parseScanGeometry } from '../../../services/scanGeometry';
import { makeHexMap } from '../../../services/hexMap';
import enIndexing from '../../../locales/en/indexing.json';
import enViewer from '../../../locales/en/ebsdviewer.json';

// vitest runs from frontend/ (import.meta.url is not a file: URL in the jsdom environment)
const FIXTURE = JSON.parse(
  readFileSync(resolve(cwd(), '..', 'tests', 'fixtures', 'hex_raster_cases.json'), 'utf8'));
const HEX_CASE = FIXTURE.geometries.find((c) => c.name === 'a_3x3_right');
const HEX = HEX_CASE.geometry;
// What the backend stamps on the picture of this scan: the oracle's own raster description, in the backend's names
const STAMP = {
  kind: 'hex_raster', factor: HEX_CASE.spec.factor, shape: [HEX_CASE.spec.height, HEX_CASE.spec.width],
  step_um: HEX_CASE.spec.step_um, x0: HEX_CASE.spec.x0, y0: HEX_CASE.spec.y0,
};
const SQUARE = { kind: 'square', n_rows: 3, n_cols: 3, grid_shape: [3, 3], step_x: 1, step_y: 1 };
const FILE = '/data/a/scan.up1';
const datasets = (geometry) => ({
  active: 'scan', file_path: FILE,
  datasets: [{ name: 'scan', active: true, navigation_shape: [3, 3], signal_shape: [8, 8], n_patterns: 9,
    scan_geometry: geometry }],
});

const calls = [];                       // every canvas call: [name, ...args]
let picture = { width: 14, height: 12 };
const realGetContext = HTMLCanvasElement.prototype.getContext;
const realRect = HTMLCanvasElement.prototype.getBoundingClientRect;
beforeAll(() => {
  HTMLCanvasElement.prototype.getContext = () => new Proxy({}, {
    get: (_t, name) => {
      if (name === 'measureText') return () => ({ width: 10 });
      if (name === 'createImageData') return (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), w, h });
      return (...args) => { calls.push([name, ...args]); };
    },
    set: () => true,
  });
  HTMLCanvasElement.prototype.getBoundingClientRect = () => ({ left: 0, top: 0, width: 400, height: 300, right: 400, bottom: 300 });
});
afterAll(() => {
  HTMLCanvasElement.prototype.getContext = realGetContext;
  HTMLCanvasElement.prototype.getBoundingClientRect = realRect;
});

beforeEach(() => {
  calls.length = 0;
  picture = { width: 14, height: 12 };
  localStorage.clear();
  sessionStorage.clear();
  vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
  vi.stubGlobal('Image', class {
    set src(_v) { setTimeout(() => { this.width = picture.width; this.height = picture.height; this.onload?.(); }, 0); }
  });
  ebsdApi.loadedFiles.mockReset();
  ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [{ path: FILE, name: 'scan', active: true }] } });
  ebsdApi.overview.mockReset();
  useLoadedFilesStore.setState({ files: [] });
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function prepare(geometry, overviewAnswer) {
  ebsdApi.datasets.mockReset();
  ebsdApi.datasets.mockResolvedValue({ data: datasets(geometry) });
  ebsdApi.overview.mockResolvedValue({ data: overviewAnswer });
  useDataStore.setState({
    ebsdLoaded: true, ebsdInfo: null, isFileOpen: true, filePath: FILE, gridShape: [3, 3], patternCount: 9,
    scanGeometry: parseScanGeometry(geometry),
  });
}

async function mountPage() {
  const utils = render(<IndexingPage isActive />);
  await waitFor(() => expect(ebsdApi.overview).toHaveBeenCalled());
  await act(async () => { await new Promise((r) => setTimeout(r, 30)); });
  const canvas = utils.container.querySelector(`canvas[aria-label="${enIndexing.navMap.dragToSelect}"]`);
  expect(canvas).toBeTruthy();
  return { ...utils, canvas };
}

// What the canvas shows now: the calls after its last clear.
const lastFrame = () => calls.slice(calls.map((c) => c[0]).lastIndexOf('clearRect'));
const lastFrameTexts = () => lastFrame().filter((c) => c[0] === 'fillText').map((c) => c[1]);

async function chooseRegionMode(container) {
  const radio = [...container.querySelectorAll('label')]
    .find((l) => l.textContent.includes(enIndexing.selection.region)).querySelector('input');
  await act(async () => { fireEvent.click(radio); });
}

describe('the navigation map of a hexagonal scan, mounted', () => {
  it('asks for the display raster, draws it, and turns a drag into the cells whose centres it holds', async () => {
    prepare(HEX, { image: 'Zm9v', display: STAMP });
    const { container, canvas } = await mountPage();
    expect(ebsdApi.overview).toHaveBeenCalledWith('bc', { display: 'raster' });
    expect(calls.filter((c) => c[0] === 'drawImage').length).toBeGreaterThan(0);        // the picture is shown
    await chooseRegionMode(container);
    // raster pixels (3, 3)..(9, 9) are cells (1,0), (1,1), (2,1): rows 1-2, columns 0-1, which the frame
    // puts around their hexagons: raster box x 0..10, y 3..12 of 14 x 12, canvas (25, 75) 250 x 225
    calls.length = 0;
    await act(async () => { fireEvent.mouseDown(canvas, { clientX: 25 + 3.5 * 25, clientY: 3.5 * 25 }); });
    await act(async () => { fireEvent.mouseUp(canvas, { clientX: 25 + 9.5 * 25, clientY: 9.5 * 25 }); });
    const frame = calls.filter((c) => c[0] === 'strokeRect').pop();
    expect(frame.slice(1)).toEqual([25, 75, 250, 225]);
  });

  it('shows no picture when the backend drew something other than the raster this page expects', async () => {
    // an index-grid picture (no `display`), as an older backend or an unconverted route would send
    prepare(HEX, { image: 'Zm9v' });
    picture = { width: 3, height: 3 };
    await mountPage();
    expect(calls.filter((c) => c[0] === 'drawImage')).toHaveLength(0);
    // the data is loaded: the finished answer is wrong, which the map says, not "load data first"
    expect(lastFrameTexts()).toContain(enViewer.overview.rasterMismatch);
    expect(lastFrameTexts()).not.toContain(enIndexing.navMap.noImage);
  });

  it('shows no picture of a raster with the same shape but another origin on the sample', async () => {
    // e.g. the picture of a crop from an odd row, which arrives after the page has moved to the other dataset
    prepare(HEX, { image: 'Zm9v', display: { ...STAMP, x0: STAMP.x0 + 0.25 } });
    await mountPage();
    expect(calls.filter((c) => c[0] === 'drawImage')).toHaveLength(0);
  });
});

describe('the navigation map of a square scan, mounted', () => {
  it('keeps the arithmetic of the grid: the picture is shown and a drag is the box of the cells under it', async () => {
    prepare(SQUARE, { image: 'Zm9v' });
    picture = { width: 3, height: 3 };
    const { container, canvas } = await mountPage();
    expect(calls.filter((c) => c[0] === 'drawImage').length).toBeGreaterThan(0);
    await chooseRegionMode(container);
    calls.length = 0;
    // cell (0, 0) of a 3 x 3 picture at scale 100, then cell (1, 2)
    await act(async () => { fireEvent.mouseDown(canvas, { clientX: 60, clientY: 10 }); });
    await act(async () => { fireEvent.mouseUp(canvas, { clientX: 250, clientY: 150 }); });
    // cells (0,0)..(1,2): grid lines x 50..350, y 0..200
    const frame = calls.filter((c) => c[0] === 'strokeRect').pop();
    expect(frame.slice(1)).toEqual([50, 0, 300, 200]);
  });

  it('says that nothing is loaded when nothing is', async () => {
    prepare(SQUARE, {});
    await mountPage();
    expect(lastFrameTexts()).toContain(enIndexing.navMap.noImage);
    expect(lastFrameTexts()).toContain(enIndexing.navMap.loadFirst);
    expect(lastFrameTexts()).not.toContain(enViewer.overview.computing);
  });
});

// --- The geometry under the page changes ----------------------------------------------------------------------
const byName = (name) => FIXTURE.geometries.find((c) => c.name === name);
// the same 3 x 3 grid with the odd rows shifted the other way: another raster of the same shape
const specOf = (geometry) => {
  const s = makeHexMap(parseScanGeometry(geometry)).spec;       // only the mock's answer is taken from it
  return { factor: s.factor, height: s.height, width: s.width, step_um: s.stepUm, x0: s.x0, y0: s.y0 };
};
const SHIFTED = { geometry: { ...HEX, odd_row_shift: -0.5 }, spec: specOf({ ...HEX, odd_row_shift: -0.5 }) };
const OTHER_GRID = { geometry: byName('b_4x5_left').geometry, spec: specOf(byName('b_4x5_left').geometry) };
const stampOf = (c) => ({
  kind: 'hex_raster', factor: c.spec.factor, shape: [c.spec.height, c.spec.width],
  step_um: c.spec.step_um, x0: c.spec.x0, y0: c.spec.y0,
});
const tick = (ms) => new Promise((r) => setTimeout(r, ms));
// effects run after an act's callback, so a picture that loads from an effect needs a second wait
const settle = () => act(async () => { await tick(30); });
const deferred = () => {
  let resolve; let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
};
const switchGeometry = async (geometry) => {
  await act(async () => {
    useDataStore.setState({ scanGeometry: parseScanGeometry(geometry) });
    await tick(30);
  });
  await settle();
};
const shownImages = () => lastFrame().filter((c) => c[0] === 'drawImage');
const regionEnd = (container) => container.querySelector(
  `input[title="${enIndexing.hoverTips.selRowEnd}"]`).value;
const addRegionButton = (container) => [...container.querySelectorAll('button')]
  .find((b) => b.textContent === enIndexing.selection.addRegion);
const savedCount = (container) => [...container.querySelectorAll('span')]
  .map((s) => s.textContent).find((s) => /saved region/.test(s));
const dragRegion = async (canvas) => {          // raster pixels (3, 3)..(9, 9): rows 1-2, columns 0-1
  await act(async () => { fireEvent.mouseDown(canvas, { clientX: 25 + 3.5 * 25, clientY: 3.5 * 25 }); });
  await act(async () => { fireEvent.mouseUp(canvas, { clientX: 25 + 9.5 * 25, clientY: 9.5 * 25 }); });
};

describe('the navigation map when the geometry under the page changes, mounted', () => {
  it('is loading, not empty, until the answer for the new geometry arrives', async () => {
    prepare(HEX, { image: 'Zm9v', display: STAMP });
    await mountPage();
    expect(shownImages().length).toBeGreaterThan(0);
    const pending = deferred();
    ebsdApi.overview.mockReturnValueOnce(pending.promise);
    await switchGeometry(SHIFTED.geometry);
    // the old picture is another raster: it is gone, and the map does not claim that nothing is loaded
    expect(shownImages()).toHaveLength(0);
    expect(lastFrameTexts()).toContain(enViewer.overview.computing);
    expect(lastFrameTexts()).not.toContain(enIndexing.navMap.noImage);
    expect(lastFrameTexts()).not.toContain(enIndexing.navMap.loadFirst);
    picture = { width: SHIFTED.spec.width, height: SHIFTED.spec.height };
    await act(async () => { pending.resolve({ data: { image: 'Zm9v', display: stampOf(SHIFTED) } }); await tick(30); });
    await settle();
    expect(shownImages().length).toBeGreaterThan(0);
    expect(lastFrameTexts()).not.toContain(enViewer.overview.computing);
  });

  it('drops a late answer of the geometry the page has left', async () => {
    const srcs = [];
    vi.stubGlobal('Image', class {
      set src(v) { srcs.push(v); setTimeout(() => { this.width = picture.width; this.height = picture.height; this.onload?.(); }, 0); }
    });
    prepare(HEX, { image: 'Zm9v', display: STAMP });
    const early = deferred();
    ebsdApi.overview.mockReturnValueOnce(early.promise);                 // asked under HEX, answered last
    await mountPage();
    const second = deferred();
    ebsdApi.overview.mockReturnValueOnce(second.promise);                // asked under the new geometry
    await switchGeometry(SHIFTED.geometry);
    picture = { width: SHIFTED.spec.width, height: SHIFTED.spec.height };
    await act(async () => { second.resolve({ data: { image: 'Zm9v', display: stampOf(SHIFTED) } }); await tick(30); });
    await settle();
    expect(shownImages().length).toBeGreaterThan(0);
    await act(async () => { early.resolve({ data: { image: 'bGF0ZQ==', display: STAMP } }); await tick(30); });
    await settle();
    expect(srcs.some((s) => s.includes('bGF0ZQ=='))).toBe(false);         // the late picture was never loaded
    expect(shownImages().length).toBeGreaterThan(0);                      // and the right one is still there
    expect(lastFrameTexts()).not.toContain(enViewer.overview.rasterMismatch);
  });

  it('drops the answer to an earlier question of the same geometry', async () => {
    const srcs = [];
    vi.stubGlobal('Image', class {
      set src(v) { srcs.push(v); setTimeout(() => { this.width = picture.width; this.height = picture.height; this.onload?.(); }, 0); }
    });
    prepare(HEX, { image: 'Zm9v', display: STAMP });
    const early = deferred();
    ebsdApi.overview.mockReturnValueOnce(early.promise);
    await mountPage();
    const second = deferred();
    ebsdApi.overview.mockReturnValueOnce(second.promise);
    // another file of the same geometry is loaded: the page asks again
    await act(async () => { useDataStore.setState({ filePath: '/data/b/other.up1' }); await tick(30); });
    await settle();
    expect(ebsdApi.overview).toHaveBeenCalledTimes(2);
    await act(async () => { second.resolve({ data: { image: 'Zm9v', display: STAMP } }); await tick(30); });
    await settle();
    await act(async () => { early.resolve({ data: { image: 'bGF0ZQ==', display: STAMP } }); await tick(30); });
    await settle();
    expect(srcs.some((s) => s.includes('bGF0ZQ=='))).toBe(false);
    expect(shownImages().length).toBeGreaterThan(0);
  });

  it('leaves no old picture when the question for the new geometry fails', async () => {
    prepare(HEX, { image: 'Zm9v', display: STAMP });
    await mountPage();
    ebsdApi.overview.mockRejectedValueOnce(new Error('boom'));
    await switchGeometry(SHIFTED.geometry);
    expect(shownImages()).toHaveLength(0);
    expect(lastFrameTexts()).toContain(enViewer.overview.failed);          // an answer, not an endless wait
    expect(lastFrameTexts()).not.toContain(enViewer.overview.computing);
    expect(lastFrameTexts()).not.toContain(enIndexing.navMap.noImage);
    // one question for the geometry change (and the mount's), not a retry loop
    await settle();
    expect(ebsdApi.overview).toHaveBeenCalledTimes(2);
  });

  it('leaves no old picture when a new question of the same geometry fails', async () => {
    prepare(HEX, { image: 'Zm9v', display: STAMP });
    await mountPage();
    expect(shownImages().length).toBeGreaterThan(0);
    ebsdApi.overview.mockRejectedValueOnce(new Error('boom'));
    await act(async () => { useDataStore.setState({ filePath: '/data/b/other.up1' }); await tick(30); });
    await settle();
    expect(shownImages()).toHaveLength(0);
    expect(lastFrameTexts()).toContain(enViewer.overview.failed);
  });

  it('says the overview failed, once, when the page moves from a hexagonal to a square scan and the question fails', async () => {
    prepare(HEX, { image: 'Zm9v', display: STAMP });
    await mountPage();
    expect(shownImages().length).toBeGreaterThan(0);
    ebsdApi.overview.mockRejectedValueOnce(new Error('boom'));
    await switchGeometry(SQUARE);
    expect(shownImages()).toHaveLength(0);
    expect(lastFrameTexts()).toContain(enViewer.overview.failed);
    expect(lastFrameTexts()).not.toContain(enViewer.overview.computing);
    await settle();
    expect(ebsdApi.overview).toHaveBeenCalledTimes(2);                     // no retry loop
  });

  it('says the overview failed when the page moves from a hexagonal to a square scan and the answer has no picture', async () => {
    prepare(HEX, { image: 'Zm9v', display: STAMP });
    await mountPage();
    expect(shownImages().length).toBeGreaterThan(0);
    ebsdApi.overview.mockResolvedValueOnce({ data: {} });
    await switchGeometry(SQUARE);
    expect(shownImages()).toHaveLength(0);
    expect(lastFrameTexts()).toContain(enViewer.overview.failed);
    expect(lastFrameTexts()).not.toContain(enViewer.overview.computing);
    await settle();
    expect(ebsdApi.overview).toHaveBeenCalledTimes(2);
  });

  it('keeps the old picture of a square scan when a new question of a square scan fails', async () => {
    prepare(SQUARE, { image: 'Zm9v' });
    picture = { width: 3, height: 3 };
    await mountPage();
    expect(shownImages().length).toBeGreaterThan(0);
    ebsdApi.overview.mockRejectedValueOnce(new Error('boom'));
    await act(async () => { useDataStore.setState({ filePath: '/data/b/other.up1' }); await tick(30); });
    await settle();
    expect(ebsdApi.overview).toHaveBeenCalledTimes(2);
    expect(shownImages().length).toBeGreaterThan(0);
    expect(lastFrameTexts()).not.toContain(enViewer.overview.failed);
  });

  it('empties the typed region with the geometry, and the saved regions with the grid', async () => {
    prepare(HEX, { image: 'Zm9v', display: STAMP });
    const { container, canvas } = await mountPage();
    await chooseRegionMode(container);
    await dragRegion(canvas);
    expect(regionEnd(container)).toBe('3');                                // rows 1-2: the end is exclusive
    await act(async () => { fireEvent.click(addRegionButton(container)); });
    expect(savedCount(container)).toBe('1 saved region');

    // the same 3 x 3 grid with another raster: the cells of the saved region are still cells of this scan
    ebsdApi.overview.mockResolvedValue({ data: { image: 'Zm9v', display: stampOf(SHIFTED) } });
    await switchGeometry(SHIFTED.geometry);
    expect(regionEnd(container)).toBe('0');
    expect(savedCount(container)).toBe('1 saved region');

    // another grid: they are cells of something else
    ebsdApi.overview.mockResolvedValue({ data: { image: 'Zm9v', display: stampOf(OTHER_GRID) } });
    await switchGeometry(OTHER_GRID.geometry);
    expect(savedCount(container)).toMatch(/^0 saved region/);
  });

  describe('how many times the server is asked', () => {
    const OTHER = '/data/b/other.up1';
    const others = () => ebsdApi.overview.mock.calls.length;

    it('once for a file switch to another hexagonal geometry, not once per place that wants a picture', async () => {
      prepare(HEX, { image: 'Zm9v', display: STAMP });
      ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
        { path: FILE, name: 'scan', active: true }, { path: OTHER, name: 'other', active: false }] } });
      ebsdApi.switchFile.mockReset();
      ebsdApi.switchFile.mockResolvedValue({ data: { success: true, file_path: OTHER } });
      ebsdApi.info.mockReset();
      ebsdApi.info.mockResolvedValue({ data: {
        loaded: true, file_path: OTHER, data_shape: [3, 3, 8, 8], scan_geometry: SHIFTED.geometry } });
      const { container } = await mountPage();
      expect(others()).toBe(1);                                    // the mount's own question
      const pending = deferred();
      ebsdApi.overview.mockReturnValueOnce(pending.promise);       // the answer for the new file is slow
      const option = await waitFor(() => {
        const o = container.querySelector(`option[value="file:${OTHER}"]`);
        expect(o).toBeTruthy();
        return o;
      });
      await act(async () => { fireEvent.change(option.parentElement, { target: { value: `file:${OTHER}` } }); });
      await waitFor(() => expect(useDataStore.getState().filePath).toBe(OTHER));
      await settle();
      // the geometry effect, fetchDataState and the call after syncFromBackend all wanted a picture: one question
      expect(others()).toBe(2);
      picture = { width: SHIFTED.spec.width, height: SHIFTED.spec.height };
      await act(async () => { pending.resolve({ data: { image: 'Zm9v', display: stampOf(SHIFTED) } }); await tick(30); });
      await settle();
      expect(shownImages().length).toBeGreaterThan(0);
      expect(others()).toBe(2);
    });

    it('once when the page becomes visible after another page changed the file and the geometry', async () => {
      prepare(HEX, { image: 'Zm9v', display: STAMP });
      const { rerender } = await mountPage();
      rerender(<IndexingPage isActive={false} />);
      await settle();
      expect(others()).toBe(1);
      const pending = deferred();
      ebsdApi.overview.mockReturnValueOnce(pending.promise);
      await act(async () => {
        useDataStore.setState({ filePath: OTHER, scanGeometry: parseScanGeometry(SHIFTED.geometry) });
        await tick(30);
      });
      expect(others()).toBe(1);                                    // hidden: nothing is asked
      rerender(<IndexingPage isActive />);
      await settle();
      expect(others()).toBe(2);                                    // the geometry effect and fetchDataState: one question
      await act(async () => { pending.resolve({ data: { image: 'Zm9v', display: stampOf(SHIFTED) } }); await tick(30); });
      await settle();
      expect(others()).toBe(2);
    });

    it('asks again for the same geometry when the file changes while the question for the old file is out', async () => {
      prepare(HEX, { image: 'Zm9v', display: STAMP });
      await mountPage();
      const pending = deferred();
      ebsdApi.overview.mockReturnValueOnce(pending.promise);
      await act(async () => { useDataStore.setState({ filePath: OTHER }); await tick(30); });
      await settle();
      expect(others()).toBe(2);
      await act(async () => { useDataStore.setState({ filePath: '/data/c/third.up1' }); await tick(30); });
      await settle();
      expect(others()).toBe(3);                                    // the first of these two answers is for another file
    });

    // The page hidden and shown again is a place that asks for the picture of the same file and geometry again.
    const hideAndShow = async (rerender) => {
      rerender(<IndexingPage isActive={false} />);
      await settle();
      rerender(<IndexingPage isActive />);
      await settle();
    };

    it('asks again for the same file and geometry once the question has failed', async () => {
      prepare(HEX, { image: 'Zm9v', display: STAMP });
      ebsdApi.overview.mockRejectedValueOnce(new Error('500'));
      const { rerender } = await mountPage();
      expect(others()).toBe(1);                                    // the mount's own question, which failed
      await hideAndShow(rerender);
      expect(others()).toBe(2);                                    // a failed question is not "still out"
    });

    it('asks again for the same file and geometry once the question has been answered', async () => {
      prepare(HEX, { image: 'Zm9v', display: STAMP });
      const { rerender } = await mountPage();
      expect(others()).toBe(1);
      await hideAndShow(rerender);
      expect(others()).toBe(2);                                    // an answered question is not "still out" either
    });

    it('asks only once while the question for the same file, geometry and measurement is out', async () => {
      prepare(HEX, { image: 'Zm9v', display: STAMP });
      useDataStore.setState({ ebsdInfo: { variant: { measurement: '1', pattern_source: 'processed' } } });
      ebsdApi.overview.mockReturnValueOnce(deferred().promise);    // never answered
      const { rerender } = await mountPage();
      expect(others()).toBe(1);
      await hideAndShow(rerender);
      expect(others()).toBe(1);
    });

    it('asks again when the measurement of the same file changes while the question is out', async () => {
      prepare(HEX, { image: 'Zm9v', display: STAMP });
      useDataStore.setState({ ebsdInfo: { variant: { measurement: '1', pattern_source: 'processed' } } });
      ebsdApi.overview.mockReturnValueOnce(deferred().promise);    // never answered
      const { rerender } = await mountPage();
      expect(others()).toBe(1);
      rerender(<IndexingPage isActive={false} />);
      await settle();
      // the viewer loads measurement 2 of the same file: same path, same 3 x 3 geometry
      await act(async () => {
        useDataStore.setState({ ebsdInfo: { variant: { measurement: '2', pattern_source: 'processed' } } });
        await tick(30);
      });
      rerender(<IndexingPage isActive />);
      await settle();
      expect(others()).toBe(2);
    });

    it('asks as often as before on a square scan', async () => {
      prepare(SQUARE, { image: 'Zm9v' });
      picture = { width: 3, height: 3 };
      await mountPage();
      const before = others();
      const pending = deferred();
      ebsdApi.overview.mockReturnValue(pending.promise);
      await act(async () => { useDataStore.setState({ filePath: OTHER }); await tick(30); });
      await settle();
      const afterSwitch = others();
      expect(afterSwitch).toBeGreaterThan(before);
      // a second place asking while the first question is out is not held back
      await act(async () => { useDataStore.setState({ filePath: '/data/c/third.up1' }); await tick(30); });
      await settle();
      expect(others()).toBeGreaterThan(afterSwitch);
    });
  });

  it('keeps the typed region of a square scan when its grid changes', async () => {
    prepare(SQUARE, { image: 'Zm9v' });
    picture = { width: 3, height: 3 };
    const { container, canvas } = await mountPage();
    await chooseRegionMode(container);
    await act(async () => { fireEvent.mouseDown(canvas, { clientX: 60, clientY: 10 }); });
    await act(async () => { fireEvent.mouseUp(canvas, { clientX: 250, clientY: 150 }); });
    expect(regionEnd(container)).toBe('2');
    await switchGeometry({ ...SQUARE, n_rows: 4, n_cols: 4 });
    expect(regionEnd(container)).toBe('2');
  });
});
