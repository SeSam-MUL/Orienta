// @vitest-environment jsdom
//
// The "PC Drift Analysis" buttons answered a request to a route that never did
// any analysis. They are gone; the page must not offer them (two calibration
// patterns are the condition under which they used to be enabled).
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
      n_patterns: 2, pc: [0.5, 0.3, 0.8], pc_basis: 'single', pc_per_pixel: false,
      patterns: [{ index: 0, row: 1, col: 2 }, { index: 1, row: 3, col: 4 }],
    }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, ebsdApi: quiet, indexApi: quiet, calibrationApi: quiet };
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
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => cleanup());

describe('PC Refinement controls', () => {
  it('offers no drift analysis button', async () => {
    render(<PCRefinement />);
    await screen.findByText(/Global PC Refine/);          // the page is up, patterns are listed
    await waitFor(() => expect(screen.getAllByTitle(/Pattern at row/).length).toBe(2));
    expect(screen.queryByText(/Drift Analysis/i)).toBeNull();
    // the working drift display, which reads the Aztec map, is still there
    expect(screen.getByText(/Show Source PC Drift/)).toBeTruthy();
  });
});
