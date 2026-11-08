// @vitest-environment jsdom
// Why a batch file failed is shown in the language of the page: the legacy batch result and
// the backend batch job carry an `error_code`; the English prose is only the fallback.
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, cleanup, fireEvent, act } from '@testing-library/react';

const FILE = 'D:/scans/one.up1';

vi.mock('../../services/api', () => ({
  indexApi: {
    discoverFiles: vi.fn(() => Promise.resolve({ data: { files: [] } })),
    batchStart: vi.fn(() => Promise.resolve({ data: {} })),
    batchStatus: vi.fn(() => Promise.resolve({ data: {
      running: false,
      results: [{ file_path: 'D:/scans/one.up1', error: 'ENGLISH PROSE FROM THE BACKEND',
                  error_code: 'pc_missing', mean_ci: null }],
    } })),
    batchStop: vi.fn(() => Promise.resolve({ data: {} })),
  },
}));

import BatchIndexingDialog from './BatchIndexingDialog';
import JobTable from '../Batch/JobTable';
import { failureText } from '../../services/problemText';

beforeEach(() => { cleanup(); vi.clearAllMocks(); });

describe('failureText', () => {
  const t = (key, opts) => (key === 'batch:errors.pc_missing' ? 'TRANSLATED' : opts?.defaultValue);
  it('prefers the translation of the code, falls back to the prose', () => {
    expect(failureText({ error_code: 'pc_missing', error: 'prose' }, t, 'batch')).toBe('TRANSLATED');
    expect(failureText({ error_code: 'unknown_code', error: 'prose' }, t, 'batch')).toBe('prose');
    expect(failureText({ error_msg: 'job prose' }, t, 'batch')).toBe('job prose');
    expect(failureText(null, t, 'batch')).toBe('');
  });
});

describe('the legacy batch dialog', () => {
  it('shows the translated reason under the failed file, not the English prose', async () => {
    render(<BatchIndexingDialog open onClose={() => {}} />);
    await act(async () => {
      fireEvent.change(document.querySelector('textarea'), { target: { value: FILE } });
    });
    await act(async () => { fireEvent.click(screen.getByText(/^\+ Add$/)); });
    await act(async () => { fireEvent.click(screen.getByText(/Start Batch/)); });
    const reason = await screen.findByTestId('batch-file-failure');
    expect(reason.textContent).toMatch(/carries no pattern centre/);
    expect(reason.textContent).not.toMatch(/ENGLISH PROSE/);
  });
});

describe('the batch job table', () => {
  it('shows the translated reason of a failed job', () => {
    render(<JobTable jobs={[
      { file_name: 'one.up1', phase_name: 'Ni', status: 'failed', error_code: 'pc_missing',
        error_msg: 'ENGLISH PROSE FROM THE BACKEND' },
      { file_name: 'two.up1', phase_name: 'Ni', status: 'done', ci_mean: 0.4, duration_sec: 3 },
    ]} />);
    const reasons = screen.getAllByTestId('job-failure');
    expect(reasons).toHaveLength(1);
    expect(reasons[0].textContent).toMatch(/carries no pattern centre/);
  });
});

describe('a job whose calibration belongs to another detector', () => {
  it('is worded in the language of the page (the real batch texts)', async () => {
    const { default: i18n } = await import('i18next');
    const out = failureText({ error_code: 'detector_shape_mismatch', error_msg: 'ENGLISH' },
      (k, o) => i18n.t(k, o), 'batch');
    // the real locale file holds the key, so the code is not shown as the English prose
    expect(out).not.toBe('ENGLISH');
    expect(out).toMatch(/another file/i);
  });
});
