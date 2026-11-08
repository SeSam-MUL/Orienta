// @vitest-environment jsdom
//
// Cancel really cancels, and a refusal during a run is said, not swallowed.
//
// The Cancel button used to stop only the page's polling: the refine went on,
// wrote its PC later unseen, and every new run got a 409. Remove and a typed PC
// failed silently while a refine ran (the PC field even kept the refused value).
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

const backend = vi.hoisted(() => ({ status: 'running', cancels: 0, removeRefused: false }));
const refused = () => Promise.reject(Object.assign(new Error('409'), {
  response: { status: 409, data: { detail: { code: 'pc_job_running', message: 'A PC optimization is running.' } } },
}));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
      n_patterns: 2, pc: [0.5, 0.3, 0.8], pc_basis: 'single', pc_per_pixel: false,
      patterns: [{ index: 0, row: 1, col: 2 }, { index: 1, row: 2, col: 2 }],
    }),
    getPatternImage: () => ok({ image: 'AAAA' }),
    getPatternResult: () => ok({ success: true, segments: [], ci: 0.4 }),
    optimize: () => ok({ task_id: 'task-1', status: 'running' }),
    runGridCalibration: () => ok({ task_id: 'grid-1', status: 'running' }),
    getOptimizeStatus: () => ok({ status: backend.status, progress: 0.3 }),
    cancelOptimize: () => { backend.cancels += 1; return ok({ status: 'cancelling' }); },
    removePattern: () => (backend.removeRefused ? refused() : ok({ success: true })),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, ebsdApi: quiet, indexApi: quiet, calibrationApi: quiet };
});

import PCRefinement from './PCRefinement';
import useToastStore from '../../stores/useToastStore';

const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});

beforeEach(() => {
  backend.status = 'running'; backend.cancels = 0; backend.removeRefused = false;
  useToastStore.setState({ toasts: [] });
  HTMLCanvasElement.prototype.getContext = function getContext() {
    this.__ctx = this.__ctx || fakeContext();
    return this.__ctx;
  };
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => cleanup());

const showPage = (page) => act(() => {
  window.dispatchEvent(new CustomEvent('page-changed', { detail: { page } }));
});
const cancelButton = () => screen.queryByTitle(/Cancel the running PC optimization/);

describe('The progress bar of a Global PC Refine', () => {
  it('shows the fraction the backend reports as percent', async () => {
    render(<PCRefinement />);
    await showPage('pcrefinement');
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(screen.getByText('Global PC Refine'));
    // the fake backend reports progress 0.3 of the run: 30 percent, not 0.3
    await waitFor(() => expect(screen.getByRole('progressbar').getAttribute('aria-valuenow')).toBe('30'), { timeout: 4000 });
    backend.status = 'cancelled';
    await waitFor(() => expect(cancelButton()).toBeNull(), { timeout: 4000 });
  }, 15000);
});

describe('Cancel of a Global PC Refine', () => {
  it('asks the backend to stop and waits until it reports "cancelled"', async () => {
    render(<PCRefinement />);
    await showPage('pcrefinement');
    await screen.findAllByTitle(/^Pattern at row/);
    fireEvent.click(screen.getByText('Global PC Refine'));
    await waitFor(() => expect(cancelButton()).not.toBeNull());

    fireEvent.click(cancelButton());
    await waitFor(() => expect(backend.cancels).toBe(1));
    expect(await screen.findByText(/Cancelling/)).toBeTruthy();
    expect(cancelButton()).not.toBeNull();            // still running until the backend says so
    expect(cancelButton().disabled).toBe(true);

    backend.status = 'cancelled';
    await waitFor(() => expect(cancelButton()).toBeNull(), { timeout: 4000 });
    expect(screen.getByText(/Cancelled\. The PC, the stored values and the patterns are unchanged\./)).toBeTruthy();
  }, 15000);
});

describe('a cancelled grid calibration', () => {
  it('ends the panel\'s wait instead of polling forever', async () => {
    backend.status = 'cancelled';
    render(<PCRefinement />);
    await showPage('pcrefinement');
    const mode = await screen.findByTitle(/PC correction mode/i);
    fireEvent.change(mode, { target: { value: 'Grid Calibration (fit_pc)' } });
    fireEvent.click(await screen.findByText('Run Grid Calibration'));
    expect(await screen.findByText(/Cancelled\. The PC/, {}, { timeout: 5000 })).toBeTruthy();
    await waitFor(() => expect(screen.getByText('Run Grid Calibration').closest('button').disabled).toBe(false));
  }, 15000);
});

describe('a refusal while a PC run is going on', () => {
  it('says why Remove did nothing, in words', async () => {
    render(<PCRefinement />);
    await showPage('pcrefinement');
    await screen.findAllByTitle(/^Pattern at row/);
    backend.removeRefused = true;
    fireEvent.click(screen.getAllByTitle(/remove the selected pattern/i)[0]);
    await waitFor(() => expect(useToastStore.getState().toasts.length).toBe(1));
    const { message, type } = useToastStore.getState().toasts[0];
    expect(type).toBe('warning');
    expect(message).toMatch(/PC refinement or grid calibration is running/);
    expect(message).not.toMatch(/object Object/);
  });
});
