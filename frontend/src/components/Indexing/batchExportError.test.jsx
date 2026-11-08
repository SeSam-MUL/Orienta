// @vitest-environment jsdom
// The legacy batch dialog shows why a file that was indexed got no export file (`export_error` of its
// entry), under the file and as a count in the summary. Nothing read that field before: a hexagonal
// scan whose .ctf could not be written looked like a success.
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react';
import i18n from '../../i18n';

const status = vi.hoisted(() => ({
  value: {
    running: false, completed: 3, total: 3, log: [],
    results: [
      { file_path: 'D:/scans/one.up1', error: null, mean_ci: 0.8123, export_error: 'the disk is full' },
      { file_path: 'D:/scans/two.up1', error: null, mean_ci: 0.7, export_error: null },
      { file_path: 'D:/scans/three.up1', error: null, mean_ci: 0.6, export_error: 'permission denied' },
    ],
  },
}));

vi.mock('../../services/api', () => ({
  indexApi: {
    discoverFiles: vi.fn(() => Promise.resolve({ data: { files: [] } })),
    batchStart: vi.fn(() => Promise.resolve({ data: {} })),
    batchStatus: vi.fn(() => Promise.resolve({ data: status.value })),
    batchStop: vi.fn(() => Promise.resolve({ data: {} })),
  },
}));

import BatchIndexingDialog from './BatchIndexingDialog';

// Hand-written per language. {reason} is the backend's own sentence.
const TEXT = {
  en: { line: (r) => `Indexed, but no export file was written: ${r}`, count: '2 without export file' },
  de: { line: (r) => `Indiziert, aber keine Exportdatei geschrieben: ${r}`, count: '2 ohne Exportdatei' },
  ja: { line: (r) => `指数付けは完了しましたが、エクスポートファイルは書き込まれませんでした: ${r}`, count: '2 件はエクスポートファイルなし' },
  zh: { line: (r) => `已完成标定，但未写出导出文件：${r}`, count: '2 个无导出文件' },
};

async function runBatch() {
  render(<BatchIndexingDialog open onClose={() => {}} />);
  await act(async () => {
    fireEvent.change(document.querySelector('textarea'),
      { target: { value: 'D:/scans/one.up1\nD:/scans/two.up1\nD:/scans/three.up1' } });
  });
  const add = screen.getAllByRole('button').find((b) => /^\+ /.test(b.textContent));
  await act(async () => { fireEvent.click(add); });
  const start = screen.getAllByRole('button').find((b) => /[（(]3/.test(b.textContent));
  await act(async () => { fireEvent.click(start); });
}

beforeEach(() => { cleanup(); vi.clearAllMocks(); });
afterEach(async () => { cleanup(); await act(async () => { await i18n.changeLanguage('en'); }); });

describe('the legacy batch dialog and a file without an export file', () => {
  it('shows the reason under that file only, and counts the files without an export', async () => {
    await runBatch();
    const lines = await screen.findAllByTestId('batch-file-export-error');
    expect(lines.map((l) => l.textContent)).toEqual([
      TEXT.en.line('the disk is full'), TEXT.en.line('permission denied')]);
    expect(document.body.textContent).toContain(TEXT.en.count);
    // the file itself still counts as indexed
    expect(document.body.textContent).toContain('3 succeeded');
    expect(document.body.textContent).not.toMatch(/\d failed/);
  });

  it('shows nothing when no file lost its export', async () => {
    status.value = { ...status.value, results: status.value.results.map((r) => ({ ...r, export_error: null })) };
    await runBatch();
    await screen.findByText('3 succeeded');
    expect(screen.queryAllByTestId('batch-file-export-error')).toHaveLength(0);
    expect(document.body.textContent).not.toContain('without export file');
  });
});

describe('in the language of the page', () => {
  it.each(['de', 'ja', 'zh'])('words the line and the count in %s', async (lang) => {
    status.value = { ...status.value, results: [
      { file_path: 'D:/scans/one.up1', error: null, mean_ci: 0.8, export_error: 'the disk is full' },
      { file_path: 'D:/scans/two.up1', error: null, mean_ci: 0.7, export_error: null },
      { file_path: 'D:/scans/three.up1', error: null, mean_ci: 0.6, export_error: 'permission denied' },
    ] };
    await act(async () => { await i18n.changeLanguage(lang); });
    render(<BatchIndexingDialog open onClose={() => {}} />);
    await act(async () => {
      fireEvent.change(document.querySelector('textarea'),
        { target: { value: 'D:/scans/one.up1\nD:/scans/two.up1\nD:/scans/three.up1' } });
    });
    const buttons = screen.getAllByRole('button');
    await act(async () => { fireEvent.click(buttons.find((b) => /^\+ /.test(b.textContent))); });
    const start = screen.getAllByRole('button').find((b) => /[（(]3/.test(b.textContent));
    await act(async () => { fireEvent.click(start); });
    const lines = await screen.findAllByTestId('batch-file-export-error');
    expect(lines.map((l) => l.textContent)).toEqual([
      TEXT[lang].line('the disk is full'), TEXT[lang].line('permission denied')]);
    expect(document.body.textContent).toContain(TEXT[lang].count);
  });
});
