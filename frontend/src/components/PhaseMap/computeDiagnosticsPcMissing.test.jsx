// @vitest-environment jsdom
// The forward diagnostics of an imported result of a file with no PC are refused with the coded
// 400 (pc_missing); the panel shows the translation, not the English prose or "HTTP 400".
import React from 'react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  forwardDiagApi: {
    compute: vi.fn(() => {
      const err = new Error('ENGLISH PROSE FROM THE BACKEND');
      err.code = 'pc_missing';
      return Promise.reject(err);
    }),
    progress: vi.fn(), summary: vi.fn(() => Promise.resolve(null)), cancel: vi.fn(),
  },
}));

import ComputeDiagnosticsPanel from './ComputeDiagnosticsPanel';
import { forwardDiagApi } from '../../services/api';

afterEach(() => cleanup());

describe('ComputeDiagnosticsPanel', () => {
  it('shows the translated pc_missing refusal', async () => {
    render(<ComputeDiagnosticsPanel resultId="r1" onOpenBrowser={() => {}} />);
    fireEvent.click(screen.getByText('Compute Diagnostics'));
    expect(await screen.findByText(/carries no pattern centre/)).toBeTruthy();
    expect(screen.queryByText(/ENGLISH PROSE/)).toBeNull();
    expect(forwardDiagApi.compute).toHaveBeenCalled();
  });
});
