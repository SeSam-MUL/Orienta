// @vitest-environment jsdom
//
// The viewer, mounted, with the backend mocked: after a load or a file switch, every holder of the
// navigation grid and the scan geometry (`ebsdInfo`, which the overview clicks and the crop windows
// read, and the data store's `gridShape` / `scanGeometry`, which the status bar and the PC page read)
// describes the dataset the backend ends up with.
//
// The scan is a made-up hexagonal one: 6 rows x 5 columns, odd rows shifted by +0.5 of a step. A crop
// of three rows from row 1 on has the opposite offset (-0.5) and a grid of 3 x 4; it is what the
// backend answers with when the file is entered again with that crop active.
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
import { ebsdApi, scanGeometryApi } from '../../../services/api';
import useLoadedFilesStore from '../../../stores/useLoadedFilesStore';
import useDataStore from '../../../stores/useDataStore';
import HEX_LIMITS from '../__fixtures__/hexLimits.json';

const FILE_A = '/data/a/raw_scan.up1';
const FILE_B = '/data/b/other_scan.up1';

const geometry = (rows, cols, shift) => ({
  kind: 'hex', n_rows: rows, n_cols: cols, grid_shape: [rows, cols], step_x: 0.5, step_y: 0.433,
  odd_row_shift: shift, shift_confirmed: true, steps_known: true,
});
const RAW_GEOMETRY = geometry(6, 5, 0.5);
const CROP_GEOMETRY = geometry(3, 4, -0.5);

/** The answer of /load and /switch-file for the file entered with its crop active. */
const CROP_ANSWER = {
  success: true, dataset_name: 'odd', file_path: FILE_B, format_type: 'EDAX',
  data_shape: [3, 4, 8, 8], navigation_shape: [4, 3], signal_shape: [8, 8], grid_shape: [3, 4],
  scan_geometry: CROP_GEOMETRY, pattern_shape: [8, 8], pattern_count: 12, n_patterns: 12,
  has_patterns: true, has_raw_patterns: true,
};
const INFO_OF_THE_CROP = {
  loaded: true, file_path: FILE_B, active_dataset: 'odd', data_shape: [3, 4, 8, 8],
  dtype: 'uint8', scan_geometry: CROP_GEOMETRY,
};

const hadResizeObserver = 'ResizeObserver' in globalThis;
beforeAll(() => {
  if (!hadResizeObserver) {
    globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  }
});
afterAll(() => { if (!hadResizeObserver) delete globalThis.ResizeObserver; });

/** The viewer as it stands with the raw hexagonal scan of file A open. */
function fileAIsOpen() {
  useDataStore.setState({
    ebsdLoaded: true,
    ebsdInfo: { grid_shape: [6, 5], navigation_shape: [5, 6], pattern_shape: [8, 8], n_patterns: 30,
      scan_geometry: RAW_GEOMETRY, format_type: 'EDAX', file_path: FILE_A },
    isFileOpen: true, filePath: FILE_A, formatType: 'EDAX', gridShape: [6, 5],
    scanGeometry: { kind: 'hex', nRows: 6, nCols: 5, gridShape: [6, 5], oddRowShift: 0.5 },
  });
}

const files = (activeB) => [
  { path: FILE_A, name: 'raw_scan', active: !activeB },
  { path: FILE_B, name: 'other_scan', active: activeB },
];

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  useLoadedFilesStore.setState({ files: [] });
  for (const name of ['loadWithProgress', 'switchFile', 'info', 'inspect', 'loadedFiles', 'datasets']) {
    ebsdApi[name].mockReset();
  }
  ebsdApi.getPattern.mockReset();
  ebsdApi.getPattern.mockResolvedValue({ data: {} });
  ebsdApi.inspect.mockResolvedValue({ data: { kind: 'plain' } });
  ebsdApi.loadedFiles.mockResolvedValue({ data: { files: files(false) } });
  ebsdApi.datasets.mockResolvedValue({ data: { active: 'odd', file_path: FILE_B, datasets: [{
    name: 'odd', active: true, navigation_shape: [4, 3], signal_shape: [8, 8], n_patterns: 12,
    scan_geometry: CROP_GEOMETRY }] } });
  scanGeometryApi.hexLimits.mockReset();
  scanGeometryApi.hexLimits.mockResolvedValue({ data: HEX_LIMITS });
  fileAIsOpen();
});
afterEach(() => cleanup());

/** What the viewer's holders say now. */
const holders = () => {
  const st = useDataStore.getState();
  return {
    infoGrid: st.ebsdInfo?.grid_shape,
    infoShift: st.ebsdInfo?.scan_geometry?.odd_row_shift,
    storeGrid: st.gridShape,
    storeShift: st.scanGeometry?.oddRowShift,
  };
};
const THE_CROP = { infoGrid: [3, 4], infoShift: -0.5, storeGrid: [3, 4], storeShift: -0.5 };

describe('after a load that restores a crop, the viewer holds the crop\'s grid and geometry', () => {
  it('typing a path', async () => {
    ebsdApi.loadWithProgress.mockResolvedValue({ data: CROP_ANSWER });
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: files(true) } });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    const box = screen.getByPlaceholderText(/.+/, { selector: 'input[type="text"]' });
    fireEvent.change(box, { target: { value: FILE_B } });
    await act(async () => { fireEvent.keyDown(box, { key: 'Enter' }); });
    await waitFor(() => expect(ebsdApi.loadWithProgress).toHaveBeenCalled());
    await waitFor(() => expect(holders()).toEqual(THE_CROP));
  });

  it('the viewer\'s own file list', async () => {
    ebsdApi.switchFile.mockResolvedValue({ data: { ...CROP_ANSWER, apex_pairing_dropped: false } });
    render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await act(async () => { await useLoadedFilesStore.getState().refresh(); });
    const entry = screen.getAllByText(/other_scan/).find((el) => el.tagName !== 'OPTION');
    await act(async () => { fireEvent.click(entry); });
    await waitFor(() => expect(ebsdApi.switchFile).toHaveBeenCalledWith(FILE_B));
    await waitFor(() => expect(holders()).toEqual(THE_CROP));
  });
});

describe('the header file switcher refreshes every holder of the grid', () => {
  it('the viewer keeps no trace of the previous file\'s grid and row offset', async () => {
    ebsdApi.switchFile.mockResolvedValue({ data: { ...CROP_ANSWER, apex_pairing_dropped: false } });
    ebsdApi.info.mockResolvedValue({ data: INFO_OF_THE_CROP });
    const { container } = render(<EBSDViewer isActive onNavigate={vi.fn()} />);
    await act(async () => { await useLoadedFilesStore.getState().refresh(); });
    const select = container.querySelector('[data-file-switcher] select');
    expect(select).toBeTruthy();
    ebsdApi.loadedFiles.mockResolvedValue({ data: { files: files(true) } });
    await act(async () => { fireEvent.change(select, { target: { value: FILE_B } }); });
    await waitFor(() => expect(ebsdApi.switchFile).toHaveBeenCalledWith(FILE_B));
    await waitFor(() => expect(holders()).toEqual(THE_CROP));
  });
});
