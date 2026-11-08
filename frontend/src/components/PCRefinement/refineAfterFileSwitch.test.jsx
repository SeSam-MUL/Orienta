// @vitest-environment jsdom
//
// A refine or a grid calibration that finishes after the user switched files: its result is
// stored for ITS dataset, and the page says so instead of showing it as the PC of the file in
// front of the user (or dropping it without a word).
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

const backend = vi.hoisted(() => ({ changed: true, stored: true, fileChanged: false, gridFileChanged: false, optimizedPcCalls: 0, gridDone: 0 }));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
      n_patterns: 1, pc: [0.5, 0.3, 0.8], pc_basis: 'single', pc_per_pixel: false,
      patterns: [{ index: 0, row: 1, col: 2 }], pc_source: 'header',
    }),
    optimize: () => ok({ task_id: 'task-1' }),
    runGridCalibration: () => ok({ task_id: 'grid-1' }),
    getOptimizeStatus: (id) => ok(id === 'grid-1' && backend.gridFileChanged ? {
      status: 'failed', progress: 0.5, code: 'fileChanged', result: null,
      error: 'Another file was loaded as scan_a while the search ran (English fallback).',
    } : id === 'grid-1' ? {
      status: 'completed', progress: 1,
      result: { n_valid_points: 5, n_calibration_points: 5, pc_stats: null,
                dataset: 'scan_a', dataset_changed: backend.changed },
    } : backend.fileChanged ? {
      status: 'failed', progress: 0.5, code: 'fileChanged', result: null,
      error: 'Another file was loaded as scan_a while the search ran (English fallback).',
    } : {
      status: 'completed', progress: 1,
      result: { mean_pc: [0.51, 0.3, 0.8], pc_values: [[0.51, 0.3, 0.8]], ci: 0.4, global_ci: 0.4,
                segments: [], pattern_results: [{ index: 0, ci: 0.4, phase_name: 'Ni' }],
                pc_map_action: 'none', dataset: 'scan_a', dataset_changed: backend.changed,
                pc_stored: backend.stored },
    }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, indexApi: quiet, ebsdApi: quiet, calibrationApi: quiet };
});

import PCRefinement from './PCRefinement';

const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = function getContext() {
    this.__ctx = this.__ctx || fakeContext();
    return this.__ctx;
  };
  backend.changed = true;
  backend.stored = true;
  backend.fileChanged = false;
  backend.gridFileChanged = false;
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => cleanup());

describe('a PC job that finishes after the file was switched', () => {
  it('a global refine says where its PC went, and shows no "Optimization complete"', async () => {
    render(<PCRefinement />);
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    expect(await screen.findByText(/refine finished for 'scan_a' after you switched files/, {}, { timeout: 4000 }))
      .toBeTruthy();
    expect(screen.queryByText(/Optimization complete/i)).toBeNull();
  }, 10000);

  it('a PC the store did not take is not said to be stored', async () => {
    backend.stored = false;
    render(<PCRefinement />);
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    expect(await screen.findByText(/does not know that file, so its PC was not stored/, {}, { timeout: 4000 }))
      .toBeTruthy();
    expect(screen.queryByText(/Its PC was stored/)).toBeNull();
  }, 10000);

  it('a search whose file was replaced says so in the reader\'s words, not the English fallback', async () => {
    backend.fileChanged = true;
    render(<PCRefinement />);
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    expect(await screen.findAllByText(/Another file was opened during the search\. Nothing was written\./, {},
      { timeout: 4000 })).not.toHaveLength(0);
    expect(screen.queryByText(/English fallback/)).toBeNull();
  }, 10000);

  it('a grid calibration whose file was replaced says so in the words of the page too', async () => {
    backend.gridFileChanged = true;
    render(<PCRefinement />);
    const mode = await screen.findByTitle(/PC correction mode/i);
    fireEvent.change(mode, { target: { value: 'Grid Calibration (fit_pc)' } });
    fireEvent.click(await screen.findByText('Run Grid Calibration'));
    expect(await screen.findByText(/Another file was opened during the search\. Nothing was written\./, {},
      { timeout: 5000 })).toBeTruthy();
    expect(screen.queryByText(/English fallback/)).toBeNull();
  }, 10000);

  it('a global refine on the same file is shown as before', async () => {
    backend.changed = false;
    render(<PCRefinement />);
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    await screen.findByText(/Optimization complete|complete/i, {}, { timeout: 4000 });
    expect(screen.queryByText(/after you switched files/)).toBeNull();
  }, 10000);

  it('a grid calibration says the same', async () => {
    render(<PCRefinement />);
    const mode = await screen.findByTitle(/PC correction mode/i);
    fireEvent.change(mode, { target: { value: 'Grid Calibration (fit_pc)' } });
    fireEvent.click(await screen.findByText('Run Grid Calibration'));
    expect(await screen.findByText(/calibration finished for 'scan_a' after you switched files/, {}, { timeout: 5000 }))
      .toBeTruthy();
  }, 10000);
});
