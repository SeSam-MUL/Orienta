// @vitest-environment jsdom
// When a batch has finished, the "open in" buttons hand the first written result over to the next page. The file
// they point at has to be one the batch wrote: with only the square .ang of a hexagonal scan ticked, that file is
// result_<stem>_square.ang, not result_<stem>.ang (which was never written). Real BatchDashboard, real store.
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';

import i18n from '../../i18n';
import BatchDashboard from './BatchDashboard';
import useBatchStore from '../../stores/useBatchStore';

vi.mock('./JobTable', () => ({ default: () => null }));

const JOBS = [{ file_path: 'D:/scans/a.up1', file_name: 'a.up1', status: 'done', phase_name: 'Al' }];

function finished(exportFormats) {
  useBatchStore.setState({
    status: 'completed', jobs: JOBS, totalJobs: 1, completed: 1, failed: 0, batchId: null,
    config: { export_dir: 'D:/out', export_formats: exportFormats },
  });
  render(<BatchDashboard />);
  fireEvent.click(screen.getAllByRole('button').find((b) => /phase map/i.test(b.textContent)));
}

beforeEach(async () => {
  sessionStorage.clear();
  await i18n.changeLanguage('en');
});
afterEach(cleanup);

describe('the hand-off after a batch', () => {
  it('points at result_<stem>_square.ang when only the square .ang was written', () => {
    finished(['ang_square']);
    expect(sessionStorage.getItem('batch_handoff_path')).toBe('D:/out/result_a_square.ang');
  });

  it('still prefers the .ang as measured when it was written too', () => {
    finished(['ang_square', 'ang']);
    expect(sessionStorage.getItem('batch_handoff_path')).toBe('D:/out/result_a.ang');
  });

  it('prefers the light .h5 (which keeps the grid as measured) to the resampled square .ang', () => {
    finished(['ang_square', 'h5_light']);
    expect(sessionStorage.getItem('batch_handoff_path')).toBe('D:/out/result_a_light.h5');
  });

  it('keeps the .ctf and the light .h5 as before', () => {
    finished(['ctf']);
    expect(sessionStorage.getItem('batch_handoff_path')).toBe('D:/out/result_a.ctf');
    cleanup();
    finished(['h5_light']);
    expect(sessionStorage.getItem('batch_handoff_path')).toBe('D:/out/result_a_light.h5');
  });
});
