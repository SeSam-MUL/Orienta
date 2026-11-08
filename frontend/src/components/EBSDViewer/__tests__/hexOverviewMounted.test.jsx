// @vitest-environment jsdom
//
// The viewer, mounted, with the backend mocked: what the overview does with a hexagonal scan when the
// pointer clicks it, when the page moves from one dataset to another, and when answers arrive late.
// (hexOverviewWiring.test.js reads the source for the same statements; this one holds the BEHAVIOUR.)
//
// The scan is a made-up hexagonal one: 3 rows x 3 columns, a step of 2 um along the rows and sqrt(3) um
// between them, odd rows shifted by +0.5 of a step. Its display raster is 12 x 14 pixels of 0.5 um, starting
// at x0 = -1 um and y0 = -2 sqrt(3) / 3 = -1.1547 um (factor 4). Worked by hand:
//   - pixel (0, 0) has its centre at (-0.75, -0.90) um, outside every hexagon: no measurement point.
//   - pixel (5, 8) has its centre at (3.25, 1.595) um; the nearest point is row 1, column 1, at
//     (1 * 2 + 0.5 * 2, 1 * sqrt(3)) = (3, 1.732), and the pixel lies inside its hexagon.
// The same window cut from the other parity of rows has odd rows shifted by -0.5: the same 12 x 14 raster,
// starting at x0 = -2 um. A crop of two rows from row 1 has that offset and a raster of 9 x 14 pixels.
import { describe, it, expect, vi, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

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
    ebsdApi: fake({}),
    edsApi: fake({}),
    pcApi: fake({}),
    scanGeometryApi: fake({}),
  };
});
vi.mock('../../common/ImageExportDialog', () => ({ default: () => null }));

import EBSDViewer from '../EBSDViewer';
import { ebsdApi, edsApi, scanGeometryApi } from '../../../services/api';
import useLoadedFilesStore from '../../../stores/useLoadedFilesStore';
import useDataStore from '../../../stores/useDataStore';
import HEX_LIMITS from '../__fixtures__/hexLimits.json';
import EN from '../../../locales/en/ebsdviewer.json';

const hex = (rows, cols, shift) => ({
  kind: 'hex', n_rows: rows, n_cols: cols, grid_shape: [rows, cols], step_x: 2, step_y: Math.sqrt(3),
  odd_row_shift: shift, shift_confirmed: true, steps_known: true,
});
const SQUARE = { kind: 'square', n_rows: 4, n_cols: 4, grid_shape: [4, 4], step_x: 1, step_y: 1 };
const HEX = hex(3, 3, 0.5);
const OTHER_PARITY = hex(3, 3, -0.5);
const CROP = hex(2, 3, -0.5);

const STAMP_HEX = { kind: 'hex_raster', factor: 4, shape: [12, 14], step_um: 0.5, x0: -1, y0: -1.1547005383792515 };
const STAMP_OTHER_PARITY = { ...STAMP_HEX, x0: -2 };
const STAMP_CROP = { ...STAMP_HEX, shape: [9, 14], x0: -2 };

const entry = (name, geometry, active = false) => ({
  name, active, navigation_shape: [geometry.n_cols, geometry.n_rows], signal_shape: [8, 8],
  n_patterns: geometry.n_rows * geometry.n_cols, dtype: 'uint8', scan_geometry: geometry,
});
const HEX_DATASETS = [
  entry('hexscan', HEX, true), entry('copy', HEX), entry('twin', OTHER_PARITY), entry('crop_rows', CROP),
];
const answerOf = (dataset, image, display) => ({
  image, display, dataset, mode: 'bc', shape: [3, 3], min: 0, max: 1, sampled: false, source: 'native',
});

// jsdom lays nothing out: the picture gets the size of its pixels times ten, which makes the arithmetic by hand.
const natural = { w: 14, h: 12 };
const STUBBED = [
  ['naturalWidth', { get: () => natural.w }],
  ['naturalHeight', { get: () => natural.h }],
  ['getBoundingClientRect', {
    value: () => ({
      left: 0, top: 0, x: 0, y: 0, width: natural.w * 10, height: natural.h * 10,
      right: natural.w * 10, bottom: natural.h * 10,
    }),
  }],
];
const saved = new Map();
const hadResizeObserver = 'ResizeObserver' in globalThis;
beforeAll(() => {
  for (const [name, descriptor] of STUBBED) {
    saved.set(name, Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, name));
    Object.defineProperty(HTMLImageElement.prototype, name, { configurable: true, ...descriptor });
  }
  if (!hadResizeObserver) {
    globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  }
});
afterAll(() => {
  for (const [name] of STUBBED) {
    if (saved.get(name)) Object.defineProperty(HTMLImageElement.prototype, name, saved.get(name));
    else delete HTMLImageElement.prototype[name];
  }
  if (!hadResizeObserver) delete globalThis.ResizeObserver;
});

const deferred = () => {
  let resolve; let reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
};
/** One entry per overview request the viewer has made, in order; the test answers them when it likes. */
let asks;
let datasetList;

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  useLoadedFilesStore.setState({ files: [] });
  useDataStore.getState().clearFile();
  useDataStore.setState({ ebsdLoaded: false, ebsdInfo: null });
  natural.w = 14; natural.h = 12;
  asks = [];
  datasetList = HEX_DATASETS;
  ebsdApi.datasets.mockReset();
  ebsdApi.datasets.mockImplementation(async () => ({ data: {
    active: datasetList.find((d) => d.active).name, file_path: '/scans/scan.h5oina', datasets: datasetList } }));
  ebsdApi.overview.mockReset();
  ebsdApi.overview.mockImplementation(() => { const d = deferred(); asks.push(d); return d.promise; });
  ebsdApi.getPattern.mockReset();
  ebsdApi.getPattern.mockResolvedValue({ data: {} });
  ebsdApi.selectDataset.mockReset();
  ebsdApi.selectDataset.mockResolvedValue({ data: {} });
  ebsdApi.crop.mockReset();
  ebsdApi.loadWithProgress.mockReset();
  ebsdApi.inspect.mockReset();
  ebsdApi.inspect.mockResolvedValue({ data: { kind: 'plain' } });
  ebsdApi.loadedFiles.mockReset();
  ebsdApi.loadedFiles.mockResolvedValue({ data: {} });
  edsApi.quantifyPixel.mockReset();
  edsApi.quantifyPixel.mockResolvedValue({ data: {} });
  edsApi.suggestPhases.mockReset();
  edsApi.suggestPhases.mockResolvedValue({ data: {} });
  scanGeometryApi.hexLimits.mockReset();
  scanGeometryApi.hexLimits.mockResolvedValue({ data: HEX_LIMITS });
});
afterEach(() => cleanup());

const sleep = (ms) => act(async () => { await new Promise((r) => setTimeout(r, ms)); });
const picture = () => document.querySelector('[data-overview-zoom-layer] img');
const answer = (ask, data) => act(async () => { ask.resolve({ data }); });
const errorBox = () => screen.queryByText(EN.overview.failed);
const mismatchText = () => screen.queryByText(EN.overview.rasterMismatch);
const spinner = () => screen.queryByText(EN.overview.computing);

/** The page entered with the backend holding `datasetList`; returns once the first overview was asked for. */
async function enter() {
  const view = render(<EBSDViewer isActive onNavigate={vi.fn()} />);
  await waitFor(() => expect(asks).toHaveLength(1));
  return view;
}
/** ... and the first overview answered with the raster of the scan. */
async function enterWithPicture() {
  const view = await enter();
  await answer(asks[0], answerOf('hexscan', 'IMGHEX', STAMP_HEX));
  await waitFor(() => expect(picture()).not.toBeNull());
  await waitFor(() => expect(ebsdApi.getPattern).toHaveBeenCalled());   // the load of the first pattern
  await sleep(300);          // the viewer's own 120 ms debounced reload after a load has run by now
  ebsdApi.getPattern.mockClear();
  edsApi.quantifyPixel.mockClear();
  return view;
}
/** What another page does: it puts a different geometry into the data store and does not tell the viewer. */
const outside = (geometry) => act(async () => {
  const info = useDataStore.getState().ebsdInfo;
  useDataStore.getState().setEBSDLoaded({ ...info, grid_shape: geometry.grid_shape, scan_geometry: geometry });
});
const viewer = (isActive) => <EBSDViewer isActive={isActive} onNavigate={vi.fn()} />;
const listItem = (name) => screen.getAllByText(name).find((el) => el.closest('.list-item-interactive'));
const clickDataset = (name) => act(async () => { fireEvent.click(listItem(name)); });
/** The pointer at the middle of raster pixel (i, j); a pixel is ten screen pixels wide here. */
const at = (i, j) => ({ clientX: (j + 0.5) * 10, clientY: (i + 0.5) * 10 });
const press = (target, point, extra = {}) => {
  fireEvent.mouseDown(target, { button: 0, ...point, ...extra });
  fireEvent.mouseUp(target, { button: 0, ...point, ...extra });
  fireEvent.click(target, { button: 0, ...point, ...extra });
};

describe('a click on the hexagonal overview', () => {
  it('in a corner no hexagon covers asks for no pattern and no composition', async () => {
    await enterWithPicture();
    press(picture(), at(0, 0));
    await sleep(150);                                  // past the viewer's 50 ms navigation delay
    expect(ebsdApi.getPattern).not.toHaveBeenCalled();
    expect(edsApi.quantifyPixel).not.toHaveBeenCalled();
  });

  it('on a cell asks for exactly the (row, col) of that cell', async () => {
    await enterWithPicture();
    press(picture(), at(5, 8));
    await waitFor(() => expect(ebsdApi.getPattern).toHaveBeenCalled());
    await sleep(150);
    // pixel (5, 8) is the cell (1, 1); read as a 3 x 3 grid it would be (1, 2), and the start is (0, 0)
    expect(ebsdApi.getPattern.mock.calls.map((c) => [c[0], c[1]])).not.toEqual([]);
    for (const call of ebsdApi.getPattern.mock.calls) expect([call[0], call[1]]).toEqual([1, 1]);
    expect(edsApi.quantifyPixel).toHaveBeenCalledWith(1, 1, 'at_pct');
  });
});

describe('moving to another dataset', () => {
  it('shows no error box while the old picture waits for the geometry of the new one, then the new picture', async () => {
    await enterWithPicture();
    const hold = deferred();
    ebsdApi.getPattern.mockImplementation(() => hold.promise);   // the switch waits here, before it asks for the overview
    await clickDataset('crop_rows');
    expect(asks).toHaveLength(1);                                 // nothing has been asked yet
    expect(errorBox()).toBeNull();
    expect(mismatchText()).toBeNull();
    expect(picture()).toBeNull();                                 // the picture of the old raster is not shown
    expect(spinner()).not.toBeNull();

    await act(async () => { hold.resolve({ data: {} }); });
    await waitFor(() => expect(asks).toHaveLength(2));
    expect(errorBox()).toBeNull();
    expect(mismatchText()).toBeNull();
    expect(spinner()).not.toBeNull();

    await answer(asks[1], answerOf('crop_rows', 'IMGCROP', STAMP_CROP));
    await waitFor(() => expect(picture()).not.toBeNull());
    expect(picture().getAttribute('src')).toContain('IMGCROP');
    expect(errorBox()).toBeNull();
  });

  it('still reports an answer that disagrees with the geometry it was asked for', async () => {
    await enterWithPicture();
    await clickDataset('crop_rows');
    await waitFor(() => expect(asks).toHaveLength(2));
    await answer(asks[1], answerOf('crop_rows', 'IMGHEX', STAMP_HEX));   // the raster of the whole scan
    expect(mismatchText()).not.toBeNull();
    expect(picture()).toBeNull();
  });

  it('takes the raster of a window cut from the other parity of rows for what it is', async () => {
    await enterWithPicture();
    await clickDataset('twin');
    await waitFor(() => expect(asks).toHaveLength(2));
    await answer(asks[1], answerOf('twin', 'IMGTWIN', STAMP_OTHER_PARITY));
    await waitFor(() => expect(picture()).not.toBeNull());
    expect(picture().getAttribute('src')).toContain('IMGTWIN');
    expect(mismatchText()).toBeNull();
  });

  it('refuses a picture of the same shape whose odd rows lie the other way', async () => {
    await enterWithPicture();
    await clickDataset('twin');
    await waitFor(() => expect(asks).toHaveLength(2));
    await answer(asks[1], answerOf('twin', 'IMGHEX', STAMP_HEX));        // 12 x 14, factor 4, but from x0 = -1
    expect(mismatchText()).not.toBeNull();
    expect(picture()).toBeNull();
  });
});

describe('answers that arrive late', () => {
  it('an answer to an older question does not replace the answer to the last one', async () => {
    await enter();                                       // asks[0]: Band Contrast, never answered in time
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'σ' })); });
    await waitFor(() => expect(asks).toHaveLength(2));   // asks[1]: the same scan, another mode
    await answer(asks[1], answerOf('hexscan', 'IMGNEW', STAMP_HEX));
    await waitFor(() => expect(picture()).not.toBeNull());
    await answer(asks[0], answerOf('hexscan', 'IMGOLD', STAMP_HEX));
    expect(picture().getAttribute('src')).toContain('IMGNEW');
    expect(picture().getAttribute('src')).not.toContain('IMGOLD');
  });

  it('a picture is shown for the geometry it was fetched for, whatever dataset the answer names', async () => {
    await enter();
    await answer(asks[0], answerOf('some_other_name', 'IMGANY', STAMP_HEX));
    await waitFor(() => expect(picture()).not.toBeNull());
    expect(picture().getAttribute('src')).toContain('IMGANY');
  });
});

describe('a geometry that another page puts into the data store', () => {
  it('is followed: the viewer fetches the overview and the dataset list anew and shows the new picture', async () => {
    await enterWithPicture();
    const listCalls = ebsdApi.datasets.mock.calls.length;
    await outside(OTHER_PARITY);                         // the old picture (x0 = -1) does not fit this one
    await waitFor(() => expect(asks).toHaveLength(2));
    expect(picture()).toBeNull();
    expect(spinner()).not.toBeNull();
    expect(errorBox()).toBeNull();
    await waitFor(() => expect(ebsdApi.datasets.mock.calls.length).toBe(listCalls + 1));
    await answer(asks[1], answerOf('twin', 'IMGTWIN', STAMP_OTHER_PARITY));
    await waitFor(() => expect(picture()).not.toBeNull());
    expect(picture().getAttribute('src')).toContain('IMGTWIN');
    expect(spinner()).toBeNull();
    expect(mismatchText()).toBeNull();
    await sleep(100);
    expect(asks).toHaveLength(2);                        // and that was all
  });

  it('waits for the page to be visible, and asks once when it is', async () => {
    const view = await enterWithPicture();
    view.rerender(viewer(false));                        // the page is mounted but hidden
    await outside(OTHER_PARITY);
    await sleep(100);
    expect(asks).toHaveLength(1);
    view.rerender(viewer(true));
    await waitFor(() => expect(asks).toHaveLength(2));
    await answer(asks[1], answerOf('twin', 'IMGTWIN', STAMP_OTHER_PARITY));
    await waitFor(() => expect(picture()).not.toBeNull());
    await sleep(100);
    expect(asks).toHaveLength(2);
  });

  it('that cannot be fetched shows the error box and asks once, not again and again', async () => {
    await enterWithPicture();
    await outside(OTHER_PARITY);
    await waitFor(() => expect(asks).toHaveLength(2));
    await act(async () => { asks[1].reject(new Error('backend gone')); });
    await waitFor(() => expect(errorBox()).not.toBeNull());
    expect(screen.getByText('backend gone')).not.toBeNull();
    await sleep(200);
    expect(asks).toHaveLength(2);
    expect(picture()).toBeNull();
    // the next change of geometry is a new reason to ask
    await outside(CROP);
    await waitFor(() => expect(asks).toHaveLength(3));
  });

  it('lets a change of mode show the picture, after a fetch that failed', async () => {
    await enterWithPicture();
    await outside(OTHER_PARITY);
    await waitFor(() => expect(asks).toHaveLength(2));
    await act(async () => { asks[1].reject(new Error('backend gone')); });
    await waitFor(() => expect(errorBox()).not.toBeNull());
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'σ' })); });
    await waitFor(() => expect(asks).toHaveLength(3));
    await answer(asks[2], answerOf('twin', 'IMGSIGMA', STAMP_OTHER_PARITY));
    await waitFor(() => expect(picture()).not.toBeNull());
    expect(picture().getAttribute('src')).toContain('IMGSIGMA');
    expect(errorBox()).toBeNull();
  });

  it('lets a change of mode, while the viewer still waits, show the picture of the last question', async () => {
    await enterWithPicture();
    await outside(OTHER_PARITY);
    await waitFor(() => expect(asks).toHaveLength(2));
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'σ' })); });
    await waitFor(() => expect(asks).toHaveLength(3));
    await answer(asks[2], answerOf('twin', 'IMGSIGMA', STAMP_OTHER_PARITY));
    await waitFor(() => expect(picture()).not.toBeNull());
    await answer(asks[1], answerOf('twin', 'IMGLATE', STAMP_OTHER_PARITY));
    expect(picture().getAttribute('src')).toContain('IMGSIGMA');
  });

  it('that is a square one: the hexagonal picture goes and the square one comes', async () => {
    await enterWithPicture();
    await outside(SQUARE);
    await waitFor(() => expect(asks).toHaveLength(2));
    expect(picture()).toBeNull();
    await answer(asks[1], answerOf('sq', 'IMGSQ', undefined));
    await waitFor(() => expect(picture()).not.toBeNull());
    expect(picture().getAttribute('src')).toContain('IMGSQ');
    expect(mismatchText()).toBeNull();
    await sleep(100);
    expect(asks).toHaveLength(2);
  });

  it('is not followed twice when the viewer itself moved to the dataset', async () => {
    await enterWithPicture();
    const hold = deferred();
    ebsdApi.getPattern.mockImplementation(() => hold.promise);
    await clickDataset('twin');
    await sleep(100);
    expect(asks).toHaveLength(1);                        // the switch has not asked yet, and nobody asks for it
    await act(async () => { hold.resolve({ data: {} }); });
    await waitFor(() => expect(asks).toHaveLength(2));
    await answer(asks[1], answerOf('twin', 'IMGTWIN', STAMP_OTHER_PARITY));
    await waitFor(() => expect(picture()).not.toBeNull());
    await sleep(100);
    expect(asks).toHaveLength(2);
  });

  it('between two square scans asks for nothing', async () => {
    datasetList = [entry('sq', SQUARE, true)];
    natural.w = 4; natural.h = 4;
    await enter();
    await answer(asks[0], answerOf('sq', 'IMGSQ', undefined));
    await waitFor(() => expect(picture()).not.toBeNull());
    const listCalls = ebsdApi.datasets.mock.calls.length;
    await outside({ ...SQUARE, n_rows: 5, n_cols: 5, grid_shape: [5, 5] });
    await sleep(200);
    expect(asks).toHaveLength(1);
    expect(ebsdApi.datasets.mock.calls.length).toBe(listCalls);
    expect(picture().getAttribute('src')).toContain('IMGSQ');
  });
});

describe('a selection and the file loaded after it', () => {
  const squareAnswer = {
    success: true, dataset_name: 'sq', file_path: '/scans/square.h5oina', format_type: 'EDAX',
    data_shape: [4, 4, 8, 8], navigation_shape: [4, 4], signal_shape: [8, 8], grid_shape: [4, 4],
    scan_geometry: SQUARE, pattern_shape: [8, 8], pattern_count: 16, n_patterns: 16,
    has_patterns: true, has_raw_patterns: true,
  };

  it('does not outlive a change of geometry: the same pixels would name other cells', async () => {
    await enterWithPicture();
    // Shift+drag from pixel (2, 2) to pixel (5, 8): a rectangle that holds cells of the scan
    fireEvent.mouseDown(picture(), { button: 0, shiftKey: true, ...at(2, 2) });
    fireEvent.mouseMove(picture(), { buttons: 1, shiftKey: true, ...at(5, 8) });
    fireEvent.mouseUp(picture(), { button: 0, shiftKey: true, ...at(5, 8) });
    await waitFor(() => expect(screen.getByTestId('crop-apply').disabled).toBe(false));

    ebsdApi.loadWithProgress.mockResolvedValue({ data: squareAnswer });
    const box = screen.getByPlaceholderText(/.+/, { selector: 'input[type="text"]' });
    fireEvent.change(box, { target: { value: '/scans/square.h5oina' } });
    await act(async () => { fireEvent.keyDown(box, { key: 'Enter' }); });
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalled());
    await waitFor(() => expect(useDataStore.getState().ebsdInfo.scan_geometry.kind).toBe('square'));
    await waitFor(() => expect(screen.getByTestId('crop-apply').disabled).toBe(true));
  });
});

describe('Shift+drag with no selection tool armed', () => {
  it('crops a rectangle: the request names the shape, it does not send null', async () => {
    datasetList = [entry('sq', SQUARE, true)];
    natural.w = 4; natural.h = 4;                                  // a square scan: one picture pixel per cell
    ebsdApi.crop.mockResolvedValue({ data: {
      name: 'sq_crop', n_selected: 6, materialised: true, n_patterns: 6,
      window: { row0: 1, col0: 1, rows: 2, cols: 3 } } });
    await enter();
    await answer(asks[0], answerOf('sq', 'IMGSQ', undefined));
    await waitFor(() => expect(picture()).not.toBeNull());
    // the pointer maps to the grid by rounding: (10, 10) of 40 x 40 is (1, 1), (x 30, y 20) is (row 2, col 3)
    fireEvent.mouseDown(picture(), { button: 0, shiftKey: true, clientX: 10, clientY: 10 });
    fireEvent.mouseMove(picture(), { buttons: 1, shiftKey: true, clientX: 30, clientY: 20 });
    fireEvent.mouseUp(picture(), { button: 0, shiftKey: true, clientX: 30, clientY: 20 });
    await waitFor(() => expect(screen.getByTestId('crop-apply').disabled).toBe(false));
    await act(async () => { fireEvent.click(screen.getByTestId('crop-apply')); });
    await waitFor(() => expect(ebsdApi.crop).toHaveBeenCalledTimes(1));
    expect(ebsdApi.crop.mock.calls[0][0]).toMatchObject({ shape: 'rect', row0: 1, col0: 1, rows: 2, cols: 3 });
  });
});
