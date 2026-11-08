// @vitest-environment jsdom
//
// The window-size field of the frame average, mounted with the real i18n (en) and the backend mocked.
// On a hexagonal scan the backend takes odd sizes up to 49 and answers anything else with a 400 that only
// the log would show: the field says so beside itself, lets nothing else be sent (button and Alt+A), and
// a square scan keeps 1 to 50.
// Expected values are the backend's rule (routes/ebsd_viewer.py frame_average): 1 <= size <= 49 and odd.
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
import DE from '../../../locales/de/ebsdviewer.json';
import JA from '../../../locales/ja/ebsdviewer.json';
import ZH from '../../../locales/zh/ebsdviewer.json';

const hex = (rows, cols, shift) => ({
  kind: 'hex', n_rows: rows, n_cols: cols, grid_shape: [rows, cols], step_x: 2, step_y: Math.sqrt(3),
  odd_row_shift: shift, shift_confirmed: true, steps_known: true,
});
const SQUARE = { kind: 'square', n_rows: 4, n_cols: 4, grid_shape: [4, 4], step_x: 1, step_y: 1 };
const HEX = hex(3, 3, 0.5);
const STAMP_HEX = { kind: 'hex_raster', factor: 4, shape: [12, 14], step_um: 0.5, x0: -1, y0: -1.1547005383792515 };
const DATASETS = [{
  name: 'hexscan', active: true, navigation_shape: [3, 3], signal_shape: [8, 8], n_patterns: 9,
  dtype: 'uint8', scan_geometry: HEX,
}];

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

let asks;
beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  useLoadedFilesStore.setState({ files: [] });
  useDataStore.getState().clearFile();
  useDataStore.setState({ ebsdLoaded: false, ebsdInfo: null });
  asks = [];
  ebsdApi.datasets.mockReset();
  ebsdApi.datasets.mockImplementation(async () => ({ data: {
    active: 'hexscan', file_path: '/scans/scan.h5oina', datasets: DATASETS } }));
  ebsdApi.overview.mockReset();
  ebsdApi.overview.mockImplementation(() => new Promise((resolve) => { asks.push(resolve); }));
  ebsdApi.getPattern.mockReset();
  ebsdApi.getPattern.mockResolvedValue({ data: {} });
  ebsdApi.selectDataset.mockReset();
  ebsdApi.selectDataset.mockResolvedValue({ data: {} });
  ebsdApi.frameAverage.mockReset();
  ebsdApi.frameAverage.mockResolvedValue({ data: {} });
  ebsdApi.deepcopy.mockReset();
  ebsdApi.deepcopy.mockResolvedValue({ data: {} });
  ebsdApi.loadedFiles.mockReset();
  ebsdApi.loadedFiles.mockResolvedValue({ data: {} });
  ebsdApi.inspect.mockReset();
  ebsdApi.inspect.mockResolvedValue({ data: { kind: 'plain' } });
  edsApi.quantifyPixel.mockReset();
  edsApi.quantifyPixel.mockResolvedValue({ data: {} });
  edsApi.suggestPhases.mockReset();
  edsApi.suggestPhases.mockResolvedValue({ data: {} });
  scanGeometryApi.hexLimits.mockReset();
  scanGeometryApi.hexLimits.mockResolvedValue({ data: HEX_LIMITS });
});
afterEach(() => cleanup());

const field = () => screen.getByText(EN.processing.windowSizeLabel).parentElement.querySelector('input');
const runBtn = () => screen.getByRole('button', { name: EN.processing.frameAverage });
const note = () => screen.queryByTestId('window-size-hex-note');
const typeSize = (v) => act(async () => { fireEvent.change(field(), { target: { value: String(v) } }); });
const sleep = (ms) => act(async () => { await new Promise((r) => setTimeout(r, ms)); });

async function enterHex() {
  const view = render(<EBSDViewer isActive onNavigate={vi.fn()} />);
  await waitFor(() => expect(asks).toHaveLength(1));
  await act(async () => { asks[0]({ data: {
    image: 'IMG', display: STAMP_HEX, dataset: 'hexscan', mode: 'bc', shape: [3, 3], min: 0, max: 1,
    sampled: false, source: 'native' } }); });
  await waitFor(() => expect(document.querySelector('[data-overview-zoom-layer] img')).not.toBeNull());
  await sleep(300);
  return view;
}
const enterSquare = async () => {
  await enterHex();
  await act(async () => {
    const info = useDataStore.getState().ebsdInfo;
    useDataStore.getState().setEBSDLoaded({ ...info, grid_shape: SQUARE.grid_shape, scan_geometry: SQUARE });
  });
  await waitFor(() => expect(note()).toBeNull());
};

describe('the window size of the frame average on a hexagonal scan', () => {
  it('allows only odd sizes up to 49 in the field itself, and says so', async () => {
    await enterHex();
    expect(field().max).toBe('49');
    expect(field().step).toBe('2');
    expect(field().min).toBe('1');
    expect(note().textContent).toBe(EN.processing.windowSizeHexHint);
    expect(runBtn().disabled).toBe(false);          // the default, 3, is a ring
    expect(note().getAttribute('role')).toBeNull();
  });

  it.each([[2], [4], [50], [0], [48]])('shows the reason for %s and sends nothing', async (size) => {
    await enterHex();
    await typeSize(size);
    expect(note().textContent).toBe(EN.processing.windowSizeHexInvalid);
    expect(note().getAttribute('role')).toBe('alert');
    expect(field().getAttribute('aria-invalid')).toBe('true');
    expect(runBtn().disabled).toBe(true);
    fireEvent.click(runBtn());
    fireEvent.keyDown(document.body, { key: 'a', altKey: true });          // the shortcut of the same button
    await sleep(100);
    expect(ebsdApi.frameAverage).not.toHaveBeenCalled();
    expect(ebsdApi.deepcopy).not.toHaveBeenCalled();
  });

  it.each([[1], [3], [5], [49]])('sends %s', async (size) => {
    await enterHex();
    await typeSize(size);
    expect(note().getAttribute('role')).toBeNull();
    expect(runBtn().disabled).toBe(false);
    fireEvent.click(runBtn());
    await waitFor(() => expect(ebsdApi.frameAverage).toHaveBeenCalledWith(size));
  });

  it('the shortcut sends a valid size too, and refuses an invalid one before it makes a copy', async () => {
    await enterHex();
    await typeSize(4);
    fireEvent.keyDown(document.body, { key: 'a', altKey: true });
    await sleep(100);
    expect(ebsdApi.frameAverage).not.toHaveBeenCalled();
    await typeSize(5);
    fireEvent.keyDown(document.body, { key: 'a', altKey: true });
    await waitFor(() => expect(ebsdApi.frameAverage).toHaveBeenCalledWith(5));
  });
});

describe('the window size of the frame average on a square scan', () => {
  it('is unchanged: 1 to 50 in steps of one, no note, an even size is sent', async () => {
    await enterSquare();
    expect(field().max).toBe('50');
    expect(field().step).toBe('1');
    expect(note()).toBeNull();
    await typeSize(4);
    expect(runBtn().disabled).toBe(false);
    fireEvent.click(runBtn());
    await waitFor(() => expect(ebsdApi.frameAverage).toHaveBeenCalledWith(4));
  });
});

describe('the texts of the window-size note', () => {
  it.each([['en', EN], ['de', DE], ['ja', JA], ['zh', ZH]])('%s has both, and both name 49', (lang, loc) => {
    for (const key of ['windowSizeHexHint', 'windowSizeHexInvalid']) {
      expect(loc.processing[key], `${lang} ${key}`).toMatch(/49/);
    }
  });
  it('German addresses the reader with "Sie"', () => {
    expect(DE.processing.windowSizeHexInvalid).toMatch(/Wählen Sie/);
    expect(DE.processing.windowSizeHexInvalid).not.toMatch(/\bWähle\b|\bdu\b/i);
  });
});
