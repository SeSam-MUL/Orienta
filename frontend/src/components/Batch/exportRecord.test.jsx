// @vitest-environment jsdom
// When a batch has finished, the page shows what its export did, file by file: which grid the files hold,
// which APEX measurement the file ran on, what is missing and why, and what the resampled files did to the
// scan. The backend keeps this in GET /api/batch-v2/{id}/exports; nothing read it before. Real
// BatchWorkflow, real stores, real i18n; the only seam is the API.
//
// The notes and the reasons are fixed English sentences of the backend. Beside each sentence the record carries
// a `code` and `params` (`note_items`, `error_items`), and the page words them in its own language. A code this
// build does not know, and a record from a backend without items, show the English sentence.
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, act, waitFor } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  batchV2Api: {
    scanFolder: vi.fn(),
    quickLoad: vi.fn(),
    pcStatus: vi.fn(() => Promise.resolve({ data: [] })),
    exports: vi.fn(),
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
import useBatchStore from '../../stores/useBatchStore';

// The English sentences as the backend writes them (they are the fallback, and what the files on disk carry).
const RESAMPLE_NOTE = 'Resampled from a hexagonal scan to a square grid by nearest neighbour at step 0.5 um '
  + '(source points 15, points written 12, source points unused 3); nothing was interpolated. '
  + 'The direction of the row offset is an assumption (no file states it): the odd rows are placed half a step '
  + 'to the right of the even rows.';
const CONFIRMED_NOTE = 'Resampled from a hexagonal scan to a square grid by nearest neighbour at step 1 um '
  + '(source points 12, points written 12, source points unused 1); nothing was interpolated.';
const SQUARE_NOTE = 'No resampled square .ang was written: this scan is already on a square grid.';
const NOT_WRITTEN = 'The file was not written; the backend log says why.';
const DETAIL = 'Permission denied: \'x.h5\' <y> & "z"';

const ASSUMED = { step: 0.5, source_points: 15, written: 12, unused: 3, direction_assumed: true, side: 'right' };
const ASSUMED_LEFT = { ...ASSUMED, side: 'left' };
const CONFIRMED = { step: 1, source_points: 12, written: 12, unused: 1, direction_assumed: false, side: null };

const RECORDS = [
  {
    file_path: 'D:/scans/a.up1', dataset: 'a', outputs: { ang: 'D:/out/result_a.ang', ang_square: null },
    errors: { ang_square: NOT_WRITTEN },
    error_items: { ang_square: { code: 'fileNotWritten', params: {}, text: NOT_WRITTEN } },
    grid_kind: 'hex',
    pairing: { h5_name: 'session.h5', measurement_id: 'GUID-B', measurement_name: 'Area 1 / OIM Map 2' },
    notes: [RESAMPLE_NOTE],
    note_items: [{ code: 'resampledFromHex', params: ASSUMED, text: RESAMPLE_NOTE }],
  },
  {
    file_path: 'D:/scans/left.up1', dataset: 'left', outputs: {}, errors: null, error_items: null,
    grid_kind: 'hex', pairing: null,
    notes: [RESAMPLE_NOTE.replace('right', 'left')],
    note_items: [{ code: 'resampledFromHex', params: ASSUMED_LEFT, text: RESAMPLE_NOTE.replace('right', 'left') }],
  },
  {
    file_path: 'D:/scans/conf.up1', dataset: 'conf', outputs: {}, errors: null, error_items: null,
    grid_kind: 'hex', pairing: null,
    notes: [CONFIRMED_NOTE], note_items: [{ code: 'resampledFromHex', params: CONFIRMED, text: CONFIRMED_NOTE }],
  },
  {
    file_path: 'D:/scans/map.h5oina', dataset: 'map', outputs: { ang: 'D:/out/result_map.ang' },
    errors: null, error_items: null, grid_kind: 'square', pairing: null, notes: [], note_items: [],
  },
  {
    file_path: 'D:/scans/sq.h5oina', dataset: 'sq', outputs: {}, errors: null, error_items: null,
    grid_kind: 'square', pairing: null,
    notes: [SQUARE_NOTE], note_items: [{ code: 'noSquareAngForSquareScan', params: {}, text: SQUARE_NOTE }],
  },
  {
    file_path: 'D:/scans/bad.up1', dataset: 'bad', outputs: {}, errors: { export: 'the disk is full' },
    error_items: { export: { code: 'exportFailed', params: { detail: 'the disk is full' }, text: 'the disk is full' } },
    grid_kind: null, pairing: null, notes: [], note_items: [],
  },
  {
    file_path: 'D:/scans/w.up1', dataset: 'w', outputs: {}, errors: { h5_light: DETAIL },
    error_items: { h5_light: { code: 'writerFailed', params: { detail: DETAIL }, text: DETAIL } },
    grid_kind: 'square', pairing: null, notes: [], note_items: [],
  },
  {
    file_path: 'D:/scans/future.up1', dataset: 'future', outputs: {}, errors: { ang: 'An English reason.' },
    error_items: { ang: { code: 'aCodeOfALaterBuild', params: {}, text: 'An English reason.' } },
    grid_kind: 'square', pairing: null, notes: ['An English note.'],
    note_items: [{ code: 'aNoteOfALaterBuild', params: {}, text: 'An English note.' }],
  },
  {
    // a backend that sends no items: the strings of `notes` and `errors` are all there is
    file_path: 'D:/scans/old.up1', dataset: 'old', outputs: {}, errors: { ang: 'An old reason.' },
    grid_kind: 'square', pairing: null, notes: ['An old note.'],
  },
];

// Hand-written per language. The numbers are the params of the fixtures above.
const TEXT = {
  en: {
    grid: 'Grid: hexagonal',
    paired: 'Metadata: session.h5, measurement Area 1 / OIM Map 2',
    assumed: 'Resampled from a hexagonal scan to a square grid by nearest neighbour at step 0.5 µm (source points 15, '
      + 'points written 12, source points unused 3); nothing was interpolated. The direction of the row offset is an '
      + 'assumption (no file states it): the odd rows are placed half a step to the right of the even rows.',
    left: 'the odd rows are placed half a step to the left of the even rows.',
    confirmed: 'Resampled from a hexagonal scan to a square grid by nearest neighbour at step 1 µm (source points 12, '
      + 'points written 12, source points unused 1); nothing was interpolated.',
    assumption: 'assumption',
    square: 'No resampled square .ang was written: this scan is already on a square grid.',
    missing: 'Not written, resampled square .ang: the file was not written; the program log says why.',
    whole: 'Not written, all files of this scan: the export stopped with an error. Technical detail: the disk is full',
    writer: `Not written, light .h5: the file could not be written. Technical detail: ${DETAIL}`,
  },
  de: {
    grid: 'Raster: hexagonal',
    paired: 'Metadaten: session.h5, Messung Area 1 / OIM Map 2',
    assumed: 'Aus einem hexagonalen Scan nach dem nächsten Nachbarn auf ein quadratisches Raster mit der Schrittweite '
      + '0.5 µm umgerechnet (Quellpunkte 15, geschriebene Punkte 12, ungenutzte Quellpunkte 3); es wurde nichts '
      + 'interpoliert. Die Richtung des Zeilenversatzes ist eine Annahme (keine Datei gibt sie an): Die ungeraden '
      + 'Zeilen liegen um einen halben Schritt rechts von den geraden Zeilen.',
    left: 'Die ungeraden Zeilen liegen um einen halben Schritt links von den geraden Zeilen.',
    confirmed: 'Aus einem hexagonalen Scan nach dem nächsten Nachbarn auf ein quadratisches Raster mit der '
      + 'Schrittweite 1 µm umgerechnet (Quellpunkte 12, geschriebene Punkte 12, ungenutzte Quellpunkte 1); es wurde '
      + 'nichts interpoliert.',
    assumption: 'Annahme',
    square: 'Es wurde keine auf ein quadratisches Raster umgerechnete .ang geschrieben: Dieser Scan liegt schon auf '
      + 'einem quadratischen Raster.',
    missing: 'Nicht geschrieben, auf ein quadratisches Raster umgerechnete .ang: Die Datei wurde nicht geschrieben; '
      + 'das Protokoll des Programms nennt den Grund.',
    whole: 'Nicht geschrieben, alle Dateien dieses Scans: Der Export brach mit einem Fehler ab. Technische Angabe: '
      + 'the disk is full',
    writer: `Nicht geschrieben, leichte .h5: Die Datei konnte nicht geschrieben werden. Technische Angabe: ${DETAIL}`,
  },
  ja: {
    grid: 'グリッド: 六方格子',
    paired: 'メタデータ: session.h5、測定 Area 1 / OIM Map 2',
    assumed: '六方格子のスキャンから最近傍法で正方格子に再サンプリングしました（ステップ 0.5 µm、元の点 15、書き出した点 12、'
      + '使われなかった元の点 3）。補間は行っていません。 行のずれの向きは仮定です（どのファイルにも記載がありません）: '
      + '奇数行は偶数行より半ステップ右に置かれています。',
    left: '奇数行は偶数行より半ステップ左に置かれています。',
    confirmed: '六方格子のスキャンから最近傍法で正方格子に再サンプリングしました（ステップ 1 µm、元の点 12、書き出した点 12、'
      + '使われなかった元の点 1）。補間は行っていません。',
    assumption: '仮定',
    square: '正方格子に再サンプリングした .ang は書き出されませんでした: このスキャンはすでに正方格子です。',
    missing: '書き込まれませんでした（正方格子に再サンプリングした .ang）: ファイルは書き込まれませんでした。'
      + '理由はプログラムのログにあります。',
    whole: '書き込まれませんでした（このスキャンのすべてのファイル）: エクスポートがエラーで停止しました。技術的な詳細: the disk is full',
    writer: `書き込まれませんでした（軽量 .h5）: ファイルを書き込めませんでした。技術的な詳細: ${DETAIL}`,
  },
  zh: {
    grid: '网格：六方网格',
    paired: '元数据：session.h5，测量 Area 1 / OIM Map 2',
    assumed: '已从六方扫描按最近邻法重采样到方形网格，步长 0.5 µm（源点 15，写出点 12，未使用的源点 3）；未做任何插值。 '
      + '行偏移的方向是一个假设（没有文件说明）：奇数行放在偶数行右侧半个步长处。',
    left: '奇数行放在偶数行左侧半个步长处。',
    confirmed: '已从六方扫描按最近邻法重采样到方形网格，步长 1 µm（源点 12，写出点 12，未使用的源点 1）；未做任何插值。',
    assumption: '假设',
    square: '未写出重采样到方形网格的 .ang：此扫描本来就是方形网格。',
    missing: '未写出（重采样到方形网格的 .ang）：文件未写出；原因见程序日志。',
    whole: '未写出（此扫描的所有文件）：导出因错误而中止。技术细节：the disk is full',
    writer: `未写出（轻量 .h5）：无法写出该文件。技术细节：${DETAIL}`,
  },
};

function show(status) {
  useBatchConfigStore.setState({ files: [], activeFileIndex: -1, currentStep: 7, method: 'hough', globalPhases: [] });
  useBatchStore.setState({ batchId: 'b1', status });
  render(<BatchWorkflow isActive />);
}

beforeEach(() => {
  vi.clearAllMocks();
  batchV2Api.exports.mockResolvedValue({ data: { batch_id: 'b1', exports: RECORDS } });
});

afterEach(async () => {
  cleanup();
  useBatchStore.setState({ batchId: null, status: null });
  await act(async () => { await i18n.changeLanguage('en'); });
});

describe('the export record of a finished batch', () => {
  it.each(Object.keys(TEXT))('shows grid, pairing, what is missing and why, and the notes, in %s', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    show('completed');
    const hex = await screen.findByTestId('batch-export-a.up1');
    expect(batchV2Api.exports).toHaveBeenCalledWith('b1');
    expect(hex.textContent).toContain(TEXT[lang].grid);
    expect(hex.textContent).toContain(TEXT[lang].paired);
    expect(hex.textContent).toContain(TEXT[lang].missing);
    expect(hex.textContent).toContain(TEXT[lang].assumed);
    expect(hex.textContent).not.toContain('GUID-B');
    expect(screen.getByTestId('batch-export-bad.up1').textContent).toContain(TEXT[lang].whole);
    expect(screen.getByTestId('batch-export-sq.h5oina').textContent).toContain(TEXT[lang].square);
  });

  it.each(Object.keys(TEXT).filter((l) => l !== 'en'))('does not leave an English sentence in the record in %s', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    show('completed');
    await screen.findByTestId('batch-export-a.up1');
    for (const name of ['a.up1', 'left.up1', 'conf.up1', 'sq.h5oina', 'bad.up1', 'w.up1']) {
      const text = screen.getByTestId(`batch-export-${name}`).textContent;
      expect(text).not.toMatch(/Resampled from|nearest neighbour|was not written|No resampled|stopped with an error|could not be written/);
    }
  });

  it.each(Object.keys(TEXT))('names the direction of the row offset as an assumption exactly when the note says so, in %s', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    show('completed');
    await screen.findByTestId('batch-export-a.up1');
    expect(screen.getByTestId('batch-export-a.up1').textContent).toContain(TEXT[lang].assumption);
    expect(screen.getByTestId('batch-export-left.up1').textContent).toContain(TEXT[lang].left);
    expect(screen.getByTestId('batch-export-left.up1').textContent).not.toContain('right');
    const conf = screen.getByTestId('batch-export-conf.up1').textContent;
    expect(conf).toContain(TEXT[lang].confirmed);
    expect(conf).not.toContain(TEXT[lang].assumption);
  });

  it.each(Object.keys(TEXT))('keeps a real exception as technical detail after a translated lead-in, in %s', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    show('completed');
    const w = await screen.findByTestId('batch-export-w.up1');
    // the detail comes through as it is: no HTML escaping, no change of quotes or brackets
    expect(w.textContent).toContain(TEXT[lang].writer);
  });

  it('shows the backend\'s English sentence for a code this build does not know', async () => {
    await act(async () => { await i18n.changeLanguage('de'); });
    show('completed');
    const future = await screen.findByTestId('batch-export-future.up1');
    expect(future.textContent).toContain('Nicht geschrieben, .ang: An English reason.');
    expect(future.textContent).toContain('An English note.');
  });

  it('shows the strings of a record that carries no items (an older backend)', async () => {
    show('completed');
    const old = await screen.findByTestId('batch-export-old.up1');
    expect(old.textContent).toContain('Not written, .ang: An old reason.');
    expect(old.textContent).toContain('An old note.');
  });

  it('says nothing of metadata for a file that had none, and nothing is missing for it', async () => {
    show('completed');
    const square = await screen.findByTestId('batch-export-map.h5oina');
    expect(square.textContent).toContain('Grid: square');
    expect(square.textContent).not.toContain('Metadata');
    expect(square.textContent).not.toContain('Not written');
  });

  it('is not asked for while the batch still runs', async () => {
    show('running');
    await act(async () => { await Promise.resolve(); });
    expect(batchV2Api.exports).not.toHaveBeenCalled();
    expect(screen.queryByTestId('batch-export-a.up1')).toBeNull();
  });

  it('survives an answer that cannot be read: the page stays and shows no record', async () => {
    batchV2Api.exports.mockRejectedValue(new Error('network'));
    show('completed');
    await waitFor(() => expect(batchV2Api.exports).toHaveBeenCalled());
    expect(screen.queryByTestId('batch-export-record')).toBeNull();
  });
});
