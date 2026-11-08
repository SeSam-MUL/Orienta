// @vitest-environment jsdom
// A hexagonal pattern file with no APEX metadata beside it, and an APEX metadata file added as a scan,
// cannot run in a batch. The backend refuses them when the file is ADDED (quick-load answers a coded
// 400); the page must say so then, in its own language, and not take the file into the batch.
// Real BatchWorkflow, real stores, real i18n; the only seam is the API.
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup, act } from '@testing-library/react';

const HEX = 'D:/scans/hex_scan.up1';
const SQUARE = 'D:/scans/square_scan.h5oina';
const BROKEN = 'D:/scans/broken_scan.h5oina';

const refusal = (code, prose) => Object.assign(new Error('Request failed with status code 400'), {
  response: { status: 400, headers: code ? { 'x-orienta-code': code } : {}, data: { detail: prose } },
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

const ENGLISH_PROSE = 'lonely.up1 was acquired on a hexagonal grid: BACKEND ENGLISH SENTENCE';

// the text of the code `edaxHexUpUnsupported`, per language: the part that says what is wrong
const TEXT = {
  en: 'Hexagonal scan without metadata: put its APEX .h5 in the same folder.',
  de: 'Hexagonaler Scan ohne Metadaten: Legen Sie die APEX-.h5 in denselben Ordner.',
  ja: 'メタデータのない六方格子スキャンです。APEX の .h5 を同じフォルダーに置いてください。',
  zh: '没有元数据的六方网格扫描：请将其 APEX .h5 放在同一文件夹中。',
};

// the text of `apexNeedsMeasurement` (an APEX metadata file added as a scan)
const NEEDS_MEASUREMENT = {
  en: 'This is a metadata file; add its .up1/.up2 files.',
  de: 'Das ist eine Metadatendatei; fügen Sie ihre .up1/.up2-Dateien hinzu.',
  ja: 'これはメタデータファイルです。対応する .up1/.up2 ファイルを追加してください。',
  zh: '这是元数据文件；请添加它对应的 .up1/.up2 文件。',
};

async function addFolder() {
  window.electronAPI = { openFolder: vi.fn(() => Promise.resolve('D:/scans')) };
  render(<BatchWorkflow isActive />);
  fireEvent.click(screen.getByRole('button', { name: /add folder|ordner|フォルダ|文件夹/i }));
  await waitFor(() => expect(batchV2Api.quickLoad).toHaveBeenCalledTimes(3));
}

beforeEach(() => {
  vi.clearAllMocks();
  useBatchConfigStore.setState({ files: [], activeFileIndex: -1, currentStep: 1, method: 'hough', globalPhases: [] });
  batchV2Api.scanFolder.mockResolvedValue({
    data: { files: [HEX, SQUARE, BROKEN], details: [] },
  });
  batchV2Api.quickLoad.mockImplementation((fp) => {
    if (fp === HEX) return Promise.reject(refusal('edaxHexUpUnsupported', ENGLISH_PROSE));
    if (fp === BROKEN) return Promise.reject(refusal(null, 'Failed to load: boom'));
    return Promise.resolve({ data: { has_eds: false, eds_elements: [], pc: [0.5, 0.5, 0.5], pc_source: 'header' } });
  });
});

afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

describe('adding a folder that holds a hexagonal scan', () => {
  it.each(Object.keys(TEXT))('says why the scan cannot be used, in %s, and not in the backend English', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    await addFolder();
    const shown = await screen.findByTestId('batch-refusal');
    expect(shown.textContent).toContain('hex_scan.up1');
    expect(shown.textContent).toContain(TEXT[lang]);
    expect(document.body.textContent).not.toContain('BACKEND ENGLISH SENTENCE');
  });

  it.each(Object.keys(NEEDS_MEASUREMENT))('says an added metadata file holds no scan, in %s', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    batchV2Api.quickLoad.mockImplementation((fp) => (fp === HEX
      ? Promise.reject(refusal('apexNeedsMeasurement', 'session.h5 is an EDAX APEX metadata file: ENGLISH'))
      : Promise.resolve({ data: { has_eds: false, eds_elements: [], pc: [0.5, 0.5, 0.5], pc_source: 'header' } })));
    await addFolder();
    const shown = await screen.findByTestId('batch-refusal');
    expect(shown.textContent).toContain(NEEDS_MEASUREMENT[lang]);
    expect(document.body.textContent).not.toContain('ENGLISH');
  });

  it('keeps the refused file out of the batch and the others in', async () => {
    await addFolder();
    await screen.findByTestId('batch-refusal');
    const paths = useBatchConfigStore.getState().files.map((f) => f.file_path);
    expect(paths).toEqual([SQUARE, BROKEN]);
    expect(screen.getAllByTestId('batch-refusal')).toHaveLength(1);    // the broken file is not a refusal
    expect(useBatchConfigStore.getState().files[0].loaded).toBe(true);
    expect(useBatchConfigStore.getState().files[1].loaded).toBeFalsy();    // any other failure: skipped as before
  });

  it('clears the notice when the next folder is scanned', async () => {
    await addFolder();
    await screen.findByTestId('batch-refusal');
    batchV2Api.scanFolder.mockResolvedValue({ data: { files: [SQUARE], details: [] } });
    fireEvent.click(screen.getByRole('button', { name: /add folder/i }));
    await waitFor(() => expect(screen.queryByTestId('batch-refusal')).toBeNull());
  });
});
