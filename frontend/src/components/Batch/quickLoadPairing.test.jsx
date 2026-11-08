// @vitest-environment jsdom
// The batch page says, under each file it adds, which grid the file is and which APEX measurement it
// was paired with (a pattern file is placed by the metadata of its measurement, found by the GUID in
// its header). A hexagonal scan with no metadata is refused when it is added, by code, in the
// page's language. Real BatchWorkflow, real stores, real i18n; the only seam is the API.
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';

const HEX = 'D:/scans/a.up1';
const SQUARE_UP = 'D:/scans/sq.up1';
const SQUARE_H5OINA = 'D:/scans/map.h5oina';
const LONELY = 'D:/scans/lonely.up1';
const ALL = [HEX, SQUARE_UP, SQUARE_H5OINA, LONELY];

const refusal = (code, prose) => Object.assign(new Error('Request failed with status code 400'), {
  response: { status: 400, headers: { 'x-orienta-code': code }, data: { detail: prose } },
});

vi.mock('../../services/api', () => ({
  batchV2Api: {
    scanFolder: vi.fn(),
    quickLoad: vi.fn(),
    pcStatus: vi.fn(() => Promise.resolve({ data: [] })),
  },
  indexApi: {
    gpuStatus: vi.fn(() => Promise.resolve({ data: { device: 'cuda' } })),
    discoverFiles: vi.fn(() => Promise.resolve({ data: { files: [] } })),
  },
}));

vi.mock('./BatchFileTree', () => ({ default: () => null }));
vi.mock('./BatchDashboard', () => ({ default: () => null }));

import i18n from '../../i18n';
import { batchV2Api } from '../../services/api';
import BatchWorkflow from './BatchWorkflow';
import useBatchConfigStore from '../../stores/useBatchConfigStore';

const ANSWER = {
  has_eds: false, eds_elements: [], pc: [0.5, 0.5, 0.5], pc_source: 'header',
};

// what each file says, per language: written out here, not read from the locale files
const TEXT = {
  en: { hex: 'hexagonal, 3 x 5', square: 'square, 4 x 4', paired: 'Metadata: session.h5, measurement Area 1 / OIM Map 2', none: 'No metadata found' },
  de: { hex: 'hexagonal, 3 x 5', square: 'quadratisch, 4 x 4', paired: 'Metadaten: session.h5, Messung Area 1 / OIM Map 2', none: 'Keine Metadaten gefunden' },
  ja: { hex: '六方格子、3 x 5', square: '正方格子、4 x 4', paired: 'メタデータ: session.h5、測定 Area 1 / OIM Map 2', none: 'メタデータが見つかりません' },
  zh: { hex: '六方网格，3 x 5', square: '方形网格，4 x 4', paired: '元数据：session.h5，测量 Area 1 / OIM Map 2', none: '未找到元数据' },
};

async function addFolder() {
  window.electronAPI = { openFolder: vi.fn(() => Promise.resolve('D:/scans')) };
  render(<BatchWorkflow isActive />);
  fireEvent.click(screen.getByRole('button', { name: /add folder|ordner|フォルダ|文件夹/i }));
  await waitFor(() => expect(batchV2Api.quickLoad).toHaveBeenCalledTimes(ALL.length));
}

beforeEach(() => {
  vi.clearAllMocks();
  useBatchConfigStore.setState({ files: [], activeFileIndex: -1, currentStep: 1, method: 'hough', globalPhases: [] });
  batchV2Api.scanFolder.mockResolvedValue({ data: { files: ALL, details: [] } });
  batchV2Api.quickLoad.mockImplementation((fp) => {
    if (fp === HEX) {
      return Promise.resolve({ data: {
        ...ANSWER, grid_shape: [3, 5], grid_kind: 'hex',
        scan_geometry: { kind: 'hex', grid_shape: [3, 5] },
        apex: { h5_path: 'D:/scans/session.h5', measurement_id: 'GUID-B', measurement_name: 'Area 1 / OIM Map 2', pairing: 'guid' },
      } });
    }
    if (fp === SQUARE_UP) {
      return Promise.resolve({ data: { ...ANSWER, grid_shape: [4, 4], grid_kind: 'square', scan_geometry: { kind: 'square', grid_shape: [4, 4] }, apex: null } });
    }
    if (fp === SQUARE_H5OINA) {
      return Promise.resolve({ data: { ...ANSWER, grid_shape: [4, 4], grid_kind: 'square', scan_geometry: { kind: 'square', grid_shape: [4, 4] }, apex: null } });
    }
    return Promise.reject(refusal('edaxHexUpUnsupported', 'ENGLISH PROSE OF THE BACKEND'));
  });
});

afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

const lineOf = (name) => screen.getByTestId(`batch-file-${name}`);

describe('what a file says about its grid and its metadata', () => {
  it.each(Object.keys(TEXT))('names grid and pairing of a hexagonal file with metadata, in %s', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    await addFolder();
    const hex = await screen.findByTestId('batch-file-a.up1');
    expect(hex.textContent).toContain(TEXT[lang].hex);
    expect(hex.textContent).toContain(TEXT[lang].paired);
    expect(hex.textContent).not.toContain('GUID-B');
  });

  it('says square for a square file and states no metadata for a square .up1 only', async () => {
    await addFolder();
    await screen.findByTestId('batch-file-a.up1');
    expect(lineOf('sq.up1').textContent).toContain(TEXT.en.square);
    expect(lineOf('sq.up1').textContent).toContain(TEXT.en.none);
    // an .h5oina carries its own grid: "no metadata found" would be noise there
    expect(lineOf('map.h5oina').textContent).toContain(TEXT.en.square);
    expect(lineOf('map.h5oina').textContent).not.toContain(TEXT.en.none);
    expect(lineOf('map.h5oina').textContent).not.toContain('Metadata:');
  });

  it('keeps the grid kind, the shape and the pairing in the store', async () => {
    await addFolder();
    await screen.findByTestId('batch-file-a.up1');
    const hex = useBatchConfigStore.getState().files.find((f) => f.file_path === HEX);
    expect(hex.grid_kind).toBe('hex');
    expect(hex.grid_dims).toEqual([3, 5]);
    expect(hex.apex.measurement_id).toBe('GUID-B');
    const square = useBatchConfigStore.getState().files.find((f) => f.file_path === SQUARE_UP);
    expect(square.grid_kind).toBe('square');
    expect(square.apex).toBeNull();
  });

  it('refuses a hexagonal file with no metadata when it is added, in the page language, and does not list it', async () => {
    await addFolder();
    const shown = await screen.findByTestId('batch-refusal');
    expect(shown.textContent).toContain('lonely.up1');
    expect(shown.textContent).toContain('Hexagonal scan without metadata: put its APEX .h5 in the same folder.');
    expect(document.body.textContent).not.toContain('ENGLISH PROSE');
    expect(screen.queryByTestId('batch-file-lonely.up1')).toBeNull();
    expect(useBatchConfigStore.getState().files.map((f) => f.file_path)).toEqual([HEX, SQUARE_UP, SQUARE_H5OINA]);
  });
});
