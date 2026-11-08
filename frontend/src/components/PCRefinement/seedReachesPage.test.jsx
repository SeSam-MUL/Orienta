// @vitest-environment jsdom
//
// "Use as Refinement Seed" writes the per-pixel PC into the store AND into the
// open PC page; the page then asks the backend again, so its fields and the map
// note show the seeded map instead of the PC it had before.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

const backend = vi.hoisted(() => ({ statusCalls: 0, seeded: false }));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => {
      backend.statusCalls += 1;
      return ok({
        has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
        n_patterns: 0, patterns: [],
        pc: backend.seeded ? [0.5026, 0.3273, 0.8458] : [0.5, 0.5, 0.5],
        pc_basis: backend.seeded ? 'map_mean' : 'single',
        pc_per_pixel: backend.seeded,
        pc_map_shape: backend.seeded ? [3, 4] : null,
        // the store's PC, which the head of the page shows (a seed writes the map into the store)
        pc_in_use: { pc: backend.seeded ? [0.5026, 0.3273, 0.8458] : [0.5, 0.5, 0.5], per_pixel: backend.seeded,
                     kind: 'file', calibration: null },
      });
    },
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const analysisApi = {
    ...(orig.analysisApi || {}),
    pcDriftImage: () => ok({ image: 'AAAA', stats: { pc_x: { range: 0.001, mean: 0.5, drift_pct: 0.2 } } }),
    pcDriftToSeed: () => {
      backend.seeded = true;
      return ok({ success: true, target_dataset: 'sampleB', mean_pc: [0.5026, 0.3273, 0.8458],
                  pc_page_updated: true });
    },
  };
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, analysisApi, ebsdApi: quiet, indexApi: quiet, calibrationApi: quiet };
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
  backend.statusCalls = 0;
  backend.seeded = false;
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => cleanup());

describe('Use as Refinement Seed', () => {
  it('makes the open page read the seeded PC back', async () => {
    render(<PCRefinement />);
    fireEvent.click(await screen.findByText('Show Source PC Drift'));
    const seed = await screen.findByText('Use as Refinement Seed');
    await waitFor(() => expect(backend.statusCalls).toBeGreaterThan(0));
    const before = backend.statusCalls;
    fireEvent.click(seed);
    await waitFor(() => expect(backend.statusCalls).toBeGreaterThan(before));
  });

  it('shows the seeded PC in the header and in the "In use" text beside the fields', async () => {
    render(<PCRefinement />);
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent).toMatch(/0\.5000, 0\.5000, 0\.5000/));
    await waitFor(() => expect(screen.getByTestId('pc-in-use-y').textContent).toBe('In use, as Bruker: 0.5000'));
    fireEvent.click(await screen.findByText('Show Source PC Drift'));
    fireEvent.click(await screen.findByText('Use as Refinement Seed'));
    await waitFor(() => expect(screen.getByTestId('pc-file-head').textContent)
      .toMatch(/per-pixel map, mean 0\.5026, 0\.3273, 0\.8458/));
    expect(screen.getByTestId('pc-in-use-x').textContent).toBe('In use, as Bruker: 0.5026');
    expect(screen.getByTestId('pc-in-use-y').textContent).toBe('In use, as Bruker: 0.3273');
    expect(screen.getByTestId('pc-in-use-z').textContent).toBe('In use, as Bruker: 0.8458');
  });
});
