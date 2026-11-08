// @vitest-environment jsdom
// A batch job that failed with a code shows that code's text in the page's language (JobTable ->
// failureText -> errors.<code> of the batch namespace), not the English sentence the backend stored.
// The codes the hexagonal batch can fail a file with are held here in all four languages. Real JobTable,
// real i18n; the only seam is the API module JobTable imports.
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';

vi.mock('../../services/api', () => ({ indexApi: { fitHoughReflectors: vi.fn() } }));

import i18n from '../../i18n';
import JobTable from './JobTable';

const failed = (code) => ({
  file_name: 'one.up1', phase_name: 'Ni', phase_path: 'D:/lib/Ni.cif', status: 'failed',
  error_msg: 'ENGLISH PROSE FROM THE BACKEND', error_code: code, error_params: null,
});

// written out here per language, not read from the locale files
const TEXT = {
  edaxHexUpUnsupported: {
    en: 'Hexagonal scan without metadata: put its APEX .h5 in the same folder.',
    de: 'Hexagonaler Scan ohne Metadaten: Legen Sie die APEX-.h5 in denselben Ordner.',
    ja: 'メタデータのない六方格子スキャンです。APEX の .h5 を同じフォルダーに置いてください。',
    zh: '没有元数据的六方网格扫描：请将其 APEX .h5 放在同一文件夹中。',
  },
  apexNeedsMeasurement: {
    en: 'This is a metadata file; add its .up1/.up2 files.',
    de: 'Das ist eine Metadatendatei; fügen Sie ihre .up1/.up2-Dateien hinzu.',
    ja: 'これはメタデータファイルです。対応する .up1/.up2 ファイルを追加してください。',
    zh: '这是元数据文件；请添加它对应的 .up1/.up2 文件。',
  },
  emsphinxSourceRefused: {
    en: 'EMSphInx (CPU) cannot index this file; use GPU, Hough or Dictionary.',
    de: 'EMSphInx (CPU) kann diese Datei nicht indizieren; nutzen Sie GPU, Hough oder Dictionary.',
    ja: 'EMSphInx (CPU) ではこのファイルを指数付けできません。GPU、Hough、または Dictionary を使用してください。',
    zh: 'EMSphInx (CPU) 无法标定此文件；请使用 GPU、Hough 或 Dictionary。',
  },
  checkpointPairingChanged: {
    en: 'Checkpoint from other metadata or grid; delete it.',
    de: 'Checkpoint von anderen Metadaten oder Raster; löschen Sie ihn.',
    ja: 'チェックポイントは別のメタデータまたは別のグリッドのものです。削除してください。',
    zh: '检查点来自其他元数据或其他网格；请删除它。',
  },
  hexFrameAverageRefused: {
    en: 'Frame averaging could not run on this hexagonal scan: use an odd window below the limit.',
    de: 'Die Frame-Mittelung lief auf diesem hexagonalen Scan nicht: Nutzen Sie ein ungerades Fenster unter dem Limit.',
    ja: 'この六方格子スキャンではフレーム平均を実行できませんでした。上限未満の奇数のウィンドウを使用してください。',
    zh: '无法对此六方网格扫描进行帧平均：请使用小于上限的奇数窗口。',
  },
  checkpointUnreadable: {
    en: 'The checkpoint of this file cannot be read; delete it.',
    de: 'Der Checkpoint dieser Datei lässt sich nicht lesen; löschen Sie ihn.',
    ja: 'このファイルのチェックポイントを読み取れません。削除してください。',
    zh: '无法读取此文件的检查点；请将其删除。',
  },
};

beforeEach(() => { cleanup(); });
afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

describe('a job that failed with an APEX code the batch namespace has no text for', () => {
  // `apexGuidMismatch` is worded in the shared `ebsdviewer` namespace: a file that cannot be paired fails
  // its jobs with such a code, and the table must find it there (not show the stored English).
  const WORDS = {
    en: 'This pattern file belongs to a different measurement than the one chosen: the ID in its header is another.',
    de: 'Diese Musterdatei gehört zu einer anderen Messung als der gewählten: Die Kennung in ihrem Kopf ist eine andere.',
    ja: 'このパターンファイルは選択した測定ではなく、別の測定のものです（ヘッダーの ID が別の測定のものです）。',
    zh: '此花样文件属于另一次测量，而不是所选的测量：其文件头中的 ID 是另一次测量的。',
  };
  it.each(Object.keys(WORDS))('is worded in %s from the shared namespace', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    render(<JobTable jobs={[failed('apexGuidMismatch')]} />);
    expect(screen.getByTestId('job-failure').textContent).toBe(WORDS[lang]);
  });
});

describe.each(Object.keys(TEXT))('a job that failed with %s', (code) => {
  it.each(['en', 'de', 'ja', 'zh'])('is worded in %s, not in the stored English', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    render(<JobTable jobs={[failed(code)]} />);
    expect(screen.getByTestId('job-failure').textContent).toBe(TEXT[code][lang]);
    expect(document.body.textContent).not.toContain('ENGLISH PROSE');
  });
});
