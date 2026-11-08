// @vitest-environment jsdom
//
// The Indexing page's dataset dropdown, mounted, with the backend mocked: picking another dataset of
// the open file (a crop, a copy) moves the grid the whole application works on. The data store's
// `gridShape` / `scanGeometry` (status bar, PC refinement, which turns a row and a column into the
// flat index `row * gridShape[1] + col`) and the viewer's `ebsdInfo` have to follow, or the pages
// work on the previous dataset's grid.
//
// Made-up hexagonal scan of 6 rows x 5 columns (odd rows shifted by +0.5 of a step); the crop of
// three rows from row 1 on has a grid of 3 x 4 and the opposite offset, -0.5.
import { describe, it, expect, vi, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { render, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

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
    isSupersededLoad: vi.fn(async () => false),
  };
});

import IndexingPage from '../IndexingPage';
import { ebsdApi, isSupersededLoad } from '../../../services/api';
import useDataStore from '../../../stores/useDataStore';
import useLoadedFilesStore from '../../../stores/useLoadedFilesStore';
import commonEn from '../../../locales/en/common.json';
import viewerEn from '../../../locales/en/ebsdviewer.json';

const FILE = '/data/a/raw_scan.up1';
const geometry = (rows, cols, shift) => ({
  kind: 'hex', n_rows: rows, n_cols: cols, grid_shape: [rows, cols], step_x: 0.5, step_y: 0.433,
  odd_row_shift: shift, shift_confirmed: true, steps_known: true,
});
const DATASETS = {
  active: 'raw_scan', file_path: FILE,
  datasets: [
    { name: 'raw_scan', active: true, navigation_shape: [5, 6], signal_shape: [8, 8], n_patterns: 30,
      scan_geometry: geometry(6, 5, 0.5) },
    { name: 'odd', active: false, navigation_shape: [4, 3], signal_shape: [8, 8], n_patterns: 12,
      scan_geometry: geometry(3, 4, -0.5) },
  ],
};

// jsdom has no canvas; the navigation map only needs a context whose drawing calls do nothing.
const realGetContext = HTMLCanvasElement.prototype.getContext;
const noContext = () => new Proxy({}, {
  get: (target, prop) => (prop in target ? target[prop] : () => ({ width: 0 })),
  set: (target, prop, value) => { target[prop] = value; return true; },
});
const hadResizeObserver = 'ResizeObserver' in globalThis;
beforeAll(() => {
  HTMLCanvasElement.prototype.getContext = noContext;
  if (!hadResizeObserver) {
    globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  }
});
afterAll(() => {
  HTMLCanvasElement.prototype.getContext = realGetContext;
  if (!hadResizeObserver) delete globalThis.ResizeObserver;
});

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  isSupersededLoad.mockReset();
  isSupersededLoad.mockResolvedValue(false);
  ebsdApi.datasets.mockReset();
  ebsdApi.datasets.mockResolvedValue({ data: DATASETS });
  ebsdApi.selectDataset.mockReset();
  ebsdApi.selectDataset.mockResolvedValue({ data: { success: true } });
  ebsdApi.loadedFiles.mockReset();
  ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [{ path: FILE, name: 'raw_scan', active: true }] } });
  useLoadedFilesStore.setState({ files: [] });
  useDataStore.setState({
    ebsdLoaded: true,
    ebsdInfo: { grid_shape: [6, 5], navigation_shape: [5, 6], pattern_shape: [8, 8], n_patterns: 30,
      pattern_count: 30, scan_geometry: geometry(6, 5, 0.5), format_type: 'EDAX' },
    isFileOpen: true, filePath: FILE, gridShape: [6, 5], patternCount: 30,
    scanGeometry: { kind: 'hex', nRows: 6, nCols: 5, gridShape: [6, 5], oddRowShift: 0.5 },
  });
});
afterEach(() => cleanup());

async function pickTheCrop() {
  const { container } = render(<IndexingPage isActive />);
  const option = await waitFor(() => {
    const o = container.querySelector('option[value="dataset:odd"]');
    expect(o).toBeTruthy();
    return o;
  });
  await act(async () => { fireEvent.change(option.parentElement, { target: { value: 'dataset:odd' } }); });
  await waitFor(() => expect(ebsdApi.selectDataset).toHaveBeenCalledWith('odd'));
}

describe('picking another dataset of the open file in the Indexing dropdown', () => {
  it('moves the data store to that dataset\'s grid and scan geometry', async () => {
    await pickTheCrop();
    await waitFor(() => expect(useDataStore.getState().gridShape).toEqual([3, 4]));
    const st = useDataStore.getState();
    expect(st.scanGeometry.oddRowShift).toBe(-0.5);
    expect(st.scanGeometry.gridShape).toEqual([3, 4]);
    expect(st.patternCount).toBe(12);
  });

  it('and the viewer\'s copy of the load answer with it', async () => {
    await pickTheCrop();
    await waitFor(() => expect(useDataStore.getState().ebsdInfo.grid_shape).toEqual([3, 4]));
    const info = useDataStore.getState().ebsdInfo;
    expect(info.scan_geometry.odd_row_shift).toBe(-0.5);
    expect(info.navigation_shape).toEqual([4, 3]);
    expect(info.n_patterns).toBe(12);
    expect(info.format_type).toBe('EDAX');          // the rest of the answer is untouched
  });

  it('leaves the store alone when no viewer answer exists to keep in step', async () => {
    useDataStore.setState({ ebsdLoaded: false, ebsdInfo: null });
    await pickTheCrop();
    await waitFor(() => expect(useDataStore.getState().gridShape).toEqual([3, 4]));
    expect(useDataStore.getState().ebsdLoaded).toBe(false);
    expect(useDataStore.getState().ebsdInfo).toBeNull();
  });
});

describe('picking another LOADED FILE in the Indexing dropdown', () => {
  const OTHER = '/data/b/other_scan.up1';
  // the answer of POST /switch-file: the load answer of the file now active, a 3 x 4 square map
  const SWITCH_ANSWER = {
    success: true, file_path: OTHER, format_type: 'EDAX', grid_shape: [3, 4], navigation_shape: [4, 3],
    pattern_shape: [8, 8], n_patterns: 12, pattern_count: 12,
    scan_geometry: { kind: 'square', n_rows: 3, n_cols: 4, grid_shape: [3, 4], step_x: 1, step_y: 1 },
  };

  async function pickTheOtherFile() {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: FILE, name: 'raw_scan', active: true },
      { path: OTHER, name: 'other_scan', active: false },
    ] } });
    ebsdApi.switchFile.mockReset();
    ebsdApi.switchFile.mockResolvedValue({ data: SWITCH_ANSWER });
    ebsdApi.info.mockReset();
    ebsdApi.info.mockResolvedValue({ data: { loaded: true, file_path: OTHER, data_shape: [3, 4, 8, 8] } });
    const { container } = render(<IndexingPage isActive />);
    await waitFor(() => expect(container.querySelector(`option[value="file:${OTHER}"]`)).toBeTruthy());
    const select = container.querySelector(`option[value="file:${OTHER}"]`).parentElement;
    await act(async () => { fireEvent.change(select, { target: { value: `file:${OTHER}` } }); });
    await waitFor(() => expect(ebsdApi.switchFile).toHaveBeenCalledWith(OTHER));
    await waitFor(() => expect(useDataStore.getState().filePath).toBe(OTHER));
  }

  it('replaces the viewer\'s copy of the load answer with the new file\'s, like the header switcher', async () => {
    await pickTheOtherFile();
    await waitFor(() => expect(useDataStore.getState().ebsdInfo.file_path).toBe(OTHER));
    const info = useDataStore.getState().ebsdInfo;
    expect(info.grid_shape).toEqual([3, 4]);
    expect(info.scan_geometry.kind).toBe('square');
    expect(useDataStore.getState().gridShape).toEqual([3, 4]);
  });

  it('does not write its answer over the viewer\'s copy when a later load or switch has succeeded', async () => {
    // another page (the header switcher, the viewer) loaded a file after this switch began: its answer
    // describes the file the backend ends on, the one here is the older
    isSupersededLoad.mockResolvedValue(true);
    await pickTheOtherFile();
    await waitFor(() => expect(isSupersededLoad).toHaveBeenCalledWith(expect.objectContaining({ data: SWITCH_ANSWER })));
    await waitFor(() => expect(ebsdApi.overview).toHaveBeenCalled());
    const info = useDataStore.getState().ebsdInfo;
    expect(info.file_path).not.toBe(OTHER);
    expect(info.grid_shape).toEqual([6, 5]);
  });

  it('writes nothing where the viewer has not been entered (a reload left no copy and ebsdLoaded false)', async () => {
    useDataStore.setState({ ebsdLoaded: false, ebsdInfo: null });
    await pickTheOtherFile();
    expect(useDataStore.getState().ebsdLoaded).toBe(false);
    expect(useDataStore.getState().ebsdInfo).toBeNull();
  });

  // a file that no longer holds the pattern source it was last loaded as (a pattern variant), and
  // an APEX pairing that conflicts with the pattern file (its texts live with the viewer)
  it.each([
    ['loadVariantUnknown', commonEn.errors.loadVariantUnknown],
    ['apexGuidMismatch', viewerEn.errors.apexGuidMismatch],
  ])('words a refusal that carries the code %s in the interface\'s language, not in the backend\'s English', async (code, text) => {
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: [
      { path: FILE, name: 'raw_scan', active: true },
      { path: OTHER, name: 'other_scan', active: false },
    ] } });
    ebsdApi.switchFile.mockReset();
    ebsdApi.switchFile.mockRejectedValue(Object.assign(new Error('Request failed with status code 400'), {
      response: { status: 400, headers: { 'x-orienta-code': code },
                  data: { detail: 'BACKEND ENGLISH SENTENCE' } },
    }));
    const { container } = render(<IndexingPage isActive />);
    await waitFor(() => expect(container.querySelector(`option[value="file:${OTHER}"]`)).toBeTruthy());
    const select = container.querySelector(`option[value="file:${OTHER}"]`).parentElement;
    await act(async () => { fireEvent.change(select, { target: { value: `file:${OTHER}` } }); });
    await waitFor(() => expect(container.textContent).toContain(text));
    expect(container.textContent).not.toContain('BACKEND ENGLISH SENTENCE');
  });
});
