// @vitest-environment jsdom
// Batch v2 always indexes the default patterns of each file (first measurement, processed), whatever the
// loading dialog chose. The first step of the page says so in one sentence, in every language.
// Real BatchWorkflow, real i18n; the only seam is the API.
import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, act } from '@testing-library/react';

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
import BatchWorkflow from './BatchWorkflow';
import useBatchConfigStore from '../../stores/useBatchConfigStore';

// the sentence per language, written by hand from its meaning (not read from the locale files)
const SAYS = {
  en: 'the first measurement, processed where the file has them',
  de: 'die erste Messung, bearbeitet, wo die Datei das hat',
  ja: '最初の測定、処理済みパターンがあればそれ',
  zh: '第一次测量，文件含已处理花样时使用已处理花样',
};

afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

describe('the first step of the batch page', () => {
  it.each(Object.keys(SAYS))('names the patterns the batch indexes, in %s', async (lang) => {
    useBatchConfigStore.setState({ files: [], activeFileIndex: -1, currentStep: 1, method: 'hough', globalPhases: [] });
    await act(async () => { await i18n.changeLanguage(lang); });
    render(<BatchWorkflow isActive />);
    const note = screen.getByTestId('batch-pattern-source');
    expect(note.textContent).toContain(SAYS[lang]);
    expect(note.textContent).not.toMatch(/batch\.step1|patternSource/);   // a missing key shows its name
  });

  it('is still there once files are in the batch', async () => {
    useBatchConfigStore.setState({
      files: [{ file_path: 'D:/scans/a.h5oina', file_name: 'a.h5oina', loaded: true }],
      activeFileIndex: 0, currentStep: 1, method: 'hough', globalPhases: [],
    });
    render(<BatchWorkflow isActive />);
    expect(screen.getByTestId('batch-pattern-source').textContent).toContain(SAYS.en);
  });
});
