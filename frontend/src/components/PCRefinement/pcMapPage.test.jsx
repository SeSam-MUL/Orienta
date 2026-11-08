// @vitest-environment jsdom
//
// The PC Refinement page and a per-pixel PC map (the result of a grid
// calibration): the page says that its fields show the map's MEAN, and what a PC
// typed into them or a Global PC Refine did to the map. A fake backend stands in
// for /api/pc.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

const backend = vi.hoisted(() => ({
  perPixel: true, shape: [300, 200], updateAction: 'shifted', refineAction: 'replaced', applyAction: 'none', statusCalls: 0, inUseMap: null,
}));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => {
      backend.statusCalls += 1;
      return ok({
        has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
        n_patterns: 1, pc: [0.5, 0.3, 0.8], pc_basis: backend.perPixel ? 'mean' : 'single',
        pc_per_pixel: backend.perPixel, pc_map_shape: backend.perPixel ? backend.shape : null,
        patterns: [{ index: 0, row: 1, col: 2 }],
        // what the file uses (the store's PC); the controller's own detector is `pc_per_pixel` above
        pc_in_use: { pc: [0.5, 0.3, 0.8], per_pixel: backend.inUseMap ?? backend.perPixel, kind: 'file', calibration: null },
      });
    },
    setDetector: () => ok({ success: true, pc_map_action: backend.applyAction }),
    updatePC: () => ok({ success: true, ci: 0.2, segments: [], pc_map_action: backend.updateAction }),
    optimize: () => ok({ task_id: 'task-1' }),
    getOptimizeStatus: () => ok({
      status: 'completed', progress: 1,
      result: {
        mean_pc: [0.51, 0.3, 0.8], pc_values: [[0.51, 0.3, 0.8]], ci: 0.4, global_ci: 0.4,
        segments: [], pattern_results: [{ index: 0, ci: 0.4, phase_name: 'Ni' }],
        pc_map_action: backend.refineAction,
      },
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
  Object.assign(backend, { perPixel: true, shape: [300, 200], updateAction: 'shifted', refineAction: 'replaced', applyAction: 'none', statusCalls: 0, inUseMap: null });
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => cleanup());

describe('PC Refinement page with a per-pixel PC map', () => {
  it('says the fields show the mean of the map, with its size', async () => {
    render(<PCRefinement />);
    const note = await screen.findByTestId('pc-map-note');
    expect(note.textContent).toMatch(/per-pixel PC map is active \(300 × 200 pixels\)/);
    expect(note.textContent).toMatch(/PC in use beside the fields is its mean/);
    // ...and says before anything is used what numbers typed into the fields would do to it
    expect(note.textContent).toMatch(/whole map is moved so that its mean is that PC; the map keeps its shape/);
  });

  it('the head of the page says "per-pixel map" by what the file uses, not by what the page\'s detector holds', async () => {
    backend.perPixel = false;           // the page's own detector holds one PC ...
    backend.inUseMap = true;            // ... and the file uses a map (the store's)
    render(<PCRefinement />);
    const pc = await screen.findByTestId('pc-file-head-pc');
    expect(pc.textContent).toMatch(/^per-pixel map, mean /);
  });

  it('the head of the page names one PC when the file uses one, whatever map the page\'s detector still holds', async () => {
    backend.perPixel = true;
    backend.inUseMap = false;
    render(<PCRefinement />);
    const pc = await screen.findByTestId('pc-file-head-pc');
    expect(pc.textContent).not.toMatch(/per-pixel/);
    expect(pc.textContent).toMatch(/0\.5/);
  });

  it('shows no note for a detector with one PC', async () => {
    backend.perPixel = false;
    render(<PCRefinement />);
    await waitFor(() => expect(backend.statusCalls).toBeGreaterThan(0));
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByTestId('pc-map-note')).toBeNull();
  });

  it('a typed PC says the map was moved, not collapsed', async () => {
    render(<PCRefinement />);
    await screen.findByTestId('pc-map-note');
    for (const [name, v] of [['PC X:', '0.51'], ['PC Y:', '0.3'], ['PC Z:', '0.8']]) {
      fireEvent.change(screen.getByLabelText(name), { target: { value: v } });
    }
    await new Promise((r) => setTimeout(r, 700));      // typing alone moves nothing, however long it takes
    expect(screen.getByTestId('pc-map-note').textContent).toMatch(/per-pixel PC map is active/);
    fireEvent.click(screen.getByRole('button', { name: 'Use these numbers' }));
    await waitFor(
      () => expect(screen.getByTestId('pc-map-note').textContent).toMatch(/moved so that its mean is the PC you entered/),
      { timeout: 4000 },
    );
  }, 10000);

  it('a Global PC Refine says the map was replaced by one PC', async () => {
    render(<PCRefinement />);
    await screen.findByTestId('pc-map-note');
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    await waitFor(
      () => expect(screen.getByTestId('pc-map-note').textContent).toMatch(/replaced by the refined PC/),
      { timeout: 4000 },
    );
  }, 10000);

  it('a refine that left no map to replace says nothing', async () => {
    backend.perPixel = false;
    backend.refineAction = 'none';
    render(<PCRefinement />);
    fireEvent.click(await screen.findByText(/Global PC Refine/));
    await screen.findByText(/Optimization complete|complete/i, {}, { timeout: 4000 });
    expect(screen.queryByTestId('pc-map-note')).toBeNull();
  }, 10000);

  it('Apply that left the map alone keeps saying a map is active', async () => {
    backend.applyAction = 'none';
    render(<PCRefinement />);
    await screen.findByTestId('pc-map-note');
    fireEvent.click(screen.getByText('Apply Detector Settings'));
    await screen.findByText('Detector set');
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.getByTestId('pc-map-note').textContent).toMatch(/per-pixel PC map is active/);
  });

  it('Apply that replaced the map says so and no longer claims a map is active', async () => {
    backend.applyAction = 'replaced';
    render(<PCRefinement />);
    await screen.findByTestId('pc-map-note');
    fireEvent.click(screen.getByText('Apply Detector Settings'));
    await waitFor(
      () => expect(screen.getByTestId('pc-map-note').textContent).toMatch(/replaced by the refined PC/),
    );
  });
});
