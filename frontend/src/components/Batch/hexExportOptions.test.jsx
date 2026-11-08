// @vitest-environment jsdom
// The export step of the batch page offers the square .ang only when a hexagonal file is in the list, and says
// once that the .ctf and the square .ang of hexagonal scans are resampled onto a square grid. A square list
// shows neither. Real BatchWorkflow, real stores, real i18n; the only seam is the API.
import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, act, fireEvent } from '@testing-library/react';

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

// Hand-written per language (not read from the locale files).
const TEXT = {
  en: { box: '.ang (square grid)', line: '.ctf and square .ang of hexagonal scans are resampled onto a square grid.' },
  de: { box: '.ang (quadratisches Raster)', line: '.ctf und quadratisches .ang hexagonaler Scans werden auf ein Quadratraster umgerechnet.' },
  ja: { box: '.ang（正方格子）', line: '六方格子のスキャンの .ctf と正方格子の .ang は、正方格子に再サンプリングされます。' },
  zh: { box: '.ang（方形网格）', line: '六方扫描的 .ctf 和方形 .ang 会被重采样到方形网格。' },
};

const file = (name, kind) => ({
  file_path: `D:/scans/${name}`, file_name: name, loaded: true, grid_kind: kind, grid_dims: [3, 4],
  config: { auto_export: true, export_formats: ['h5_light', 'ang', 'ctf'], phases: [] },
});

function show(files) {
  useBatchConfigStore.setState({ files, activeFileIndex: 0, currentStep: 3, method: 'hough', globalPhases: [] });
  render(<BatchWorkflow isActive />);
}

afterEach(async () => {
  cleanup();
  await act(async () => { await i18n.changeLanguage('en'); });
});

describe('the export step with a hexagonal file in the list', () => {
  it.each(Object.keys(TEXT))('offers the square .ang and says what is resampled, in %s', async (lang) => {
    await act(async () => { await i18n.changeLanguage(lang); });
    show([file('a.up1', 'hex'), file('b.h5oina', 'square')]);
    expect(screen.getByTestId('batch-format-ang_square').textContent).toContain(TEXT[lang].box);
    expect(screen.getByTestId('batch-hex-resampled').textContent).toBe(TEXT[lang].line);
    expect(screen.getByTestId('batch-hex-resampled').textContent).not.toMatch(/formats\.hexResampled|angSquare/);
  });

  it('puts ang_square into the formats of every file when ticked, and takes it out when cleared', () => {
    show([file('a.up1', 'hex'), file('b.h5oina', 'square')]);
    const box = screen.getByTestId('batch-format-ang_square').querySelector('input');
    expect(box.checked).toBe(false);
    fireEvent.click(box);
    for (const f of useBatchConfigStore.getState().files) expect(f.config.export_formats).toContain('ang_square');
    expect(screen.getByTestId('batch-format-ang_square').querySelector('input').checked).toBe(true);
    fireEvent.click(screen.getByTestId('batch-format-ang_square').querySelector('input'));
    for (const f of useBatchConfigStore.getState().files) expect(f.config.export_formats).not.toContain('ang_square');
  });
});

describe('the export step with square files only', () => {
  it('shows neither the square .ang nor the line about resampling', () => {
    show([file('a.h5oina', 'square'), file('b.h5oina', 'square')]);
    expect(screen.queryByTestId('batch-format-ang_square')).toBeNull();
    expect(screen.queryByTestId('batch-hex-resampled')).toBeNull();
    // the formats of every scan stay
    expect(screen.getByText('.ctf')).toBeTruthy();
  });

  it('still shows the box when it is ticked, so it can be cleared', () => {
    const f = file('a.h5oina', 'square');
    f.config.export_formats = ['h5_light', 'ang_square'];
    show([f]);
    expect(screen.getByTestId('batch-format-ang_square').querySelector('input').checked).toBe(true);
  });
});
