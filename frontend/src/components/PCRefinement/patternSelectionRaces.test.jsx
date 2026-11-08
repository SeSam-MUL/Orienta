// @vitest-environment jsdom
//
// The Kikuchi lines on the PC Refinement page belong to ONE calibration pattern
// and ONE PC. Every answer that draws them comes from a request that takes a
// while, so the page must not draw an answer that arrived after the pattern (or
// the PC) it describes was replaced. A fake backend whose answers the test
// releases one by one stands in for /api/pc. The canvas reports how many
// segments it was given (data-segment-count).
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

const backend = vi.hoisted(() => ({ results: {}, updates: [], tilts: [], indexes: [], calls: [], optimizeSegments: {} }));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const deferred = () => {
    let resolve; const promise = new Promise((r) => { resolve = r; });
    return { promise, resolve };
  };
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
      n_patterns: 2, pc: [0.5, 0.3, 0.8], pc_basis: 'single', pc_per_pixel: false,
      patterns: [{ index: 0, row: 0, col: 0 }, { index: 1, row: 0, col: 1 }],
    }),
    getPatternImage: (idx) => { backend.calls.push(['image', idx]); return ok({ image: `IMG${idx}` }); },
    getPatternResult: (idx) => {
      backend.calls.push(['result', idx]);
      const d = deferred();
      (backend.results[idx] = backend.results[idx] || []).push(d);
      return d.promise;
    },
    indexPattern: (row, col) => {
      const d = deferred();
      backend.indexes.push({ row, col, ...d });
      return d.promise;
    },
    updateTilt: (...args) => {
      const d = deferred();
      backend.tilts.push({ args, ...d });
      return d.promise;
    },
    updatePC: (...args) => {
      const d = deferred();
      backend.updates.push({ args, ...d });
      return d.promise;
    },
    optimize: () => ok({ task_id: 'task-1' }),
    getOptimizeStatus: () => ok({
      status: 'completed', progress: 1,
      result: {
        mean_pc: [0.51, 0.3, 0.8], pc_values: [[0.51, 0.3, 0.8], [0.52, 0.3, 0.8]],
        ci: 0.4, global_ci: 0.4,
        segments: segs(backend.optimizeSegments[0]),            // the FIRST pattern's, as ever
        pattern_results: [
          { index: 0, ci: 0.4, phase_name: 'Ni', segments: segs(backend.optimizeSegments[0]) },
          { index: 1, ci: 0.3, phase_name: 'Ni', segments: segs(backend.optimizeSegments[1]) },
        ],
        pc_map_action: 'none',
      },
    }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const ebsdApi = new Proxy({
    getPattern: (row, col) => { backend.calls.push(['ebsdPattern', row, col]); return ok({ image: 'VIEWER' }); },
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, ebsdApi, indexApi: quiet, calibrationApi: quiet };
});

function segs(n = 0) {
  return Array.from({ length: n }, (_, i) => [[i, 0], [i, 10]]);
}

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
  backend.results = {}; backend.updates = []; backend.tilts = []; backend.indexes = []; backend.calls = [];
  backend.optimizeSegments = { 0: 3, 1: 9 };
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => cleanup());

const lines = () => Number(screen.getByTestId('pc-pattern-canvas').getAttribute('data-segment-count'));
const releaseResult = async (idx, n, which = -1) => {
  const list = backend.results[idx];
  const d = which < 0 ? list[list.length + which] : list[which];
  await act(async () => { d.resolve({ data: { success: true, segments: segs(n), ci: 0.5 } }); });
};
// A PC typed into the three Bruker fields and applied with "Use these numbers" (typing alone applies nothing).
const useTyped = (x = '0.51') => {
  for (const [name, v] of [['PC X:', x], ['PC Y:', '0.3'], ['PC Z:', '0.8']]) {
    fireEvent.change(screen.getByLabelText(name), { target: { value: v } });
  }
  fireEvent.click(screen.getByRole('button', { name: 'Use these numbers' }));
};
const pick = async (row, col) => {
  fireEvent.click(await screen.findByTitle(new RegExp(`Pattern at row ${row}, col ${col}`)));
};

describe('Kikuchi lines and the selected pattern', () => {
  it('picking another pattern clears the lines of the previous one at once', async () => {
    render(<PCRefinement />);
    await pick(0, 0);
    await waitFor(() => expect(backend.results[0]).toHaveLength(1));
    await releaseResult(0, 3);
    await waitFor(() => expect(lines()).toBe(3));

    await pick(0, 1);                       // its answer is still on the way
    expect(lines()).toBe(0);                // the lines of pattern 0 do not stay on pattern 1
    await waitFor(() => expect(backend.results[1]).toHaveLength(1));
    await releaseResult(1, 5);
    await waitFor(() => expect(lines()).toBe(5));
  });

  it('a late answer for a pattern that is no longer selected is not drawn', async () => {
    render(<PCRefinement />);
    await pick(0, 0);
    await waitFor(() => expect(backend.results[0]).toHaveLength(1));
    await pick(0, 1);
    await waitFor(() => expect(backend.results[1]).toHaveLength(1));
    await releaseResult(1, 5);
    await waitFor(() => expect(lines()).toBe(5));
    await releaseResult(0, 3);              // pattern 0's answer arrives last
    await act(async () => { await Promise.resolve(); });
    expect(lines()).toBe(5);
  });

  it('coming back to the page clears the old lines and loads the stored image of the last pattern', async () => {
    render(<PCRefinement />);
    await pick(0, 0);
    await waitFor(() => expect(backend.results[0]).toHaveLength(1));
    await releaseResult(0, 3);
    await waitFor(() => expect(lines()).toBe(3));
    backend.calls = [];

    // leave and re-enter: the page re-syncs and selects the last pattern
    await act(async () => {
      window.dispatchEvent(new CustomEvent('page-changed', { detail: { page: 'pcrefinement' } }));
    });
    await waitFor(() => expect(backend.results[1]).toHaveLength(1));
    expect(lines()).toBe(0);                                  // not pattern 0's lines
    // the image is the STORED pattern's, not the viewer's re-render of the signal
    expect(backend.calls).toContainEqual(['image', 1]);
    expect(backend.calls.filter((c) => c[0] === 'ebsdPattern')).toEqual([]);
    await releaseResult(1, 4);
    await waitFor(() => expect(lines()).toBe(4));
  });

  it('the answer to a PC edit of a pattern that is no longer selected is not drawn', async () => {
    render(<PCRefinement />);
    await pick(0, 0);
    await waitFor(() => expect(backend.results[0]).toHaveLength(1));
    await releaseResult(0, 3);
    await waitFor(() => expect(lines()).toBe(3));

    useTyped();
    await waitFor(() => expect(backend.updates).toHaveLength(1), { timeout: 4000 });
    expect(backend.updates[0].args.slice(0, 3)).toEqual([0.51, 0.3, 0.8]);    // exactly the numbers typed
    expect(backend.updates[0].args[3]).toBe(0);               // asked for pattern 0

    await pick(0, 1);
    await waitFor(() => expect(backend.results[1]).toHaveLength(1));
    await releaseResult(1, 5);
    await waitFor(() => expect(lines()).toBe(5));
    await act(async () => {
      backend.updates[0].resolve({ data: { success: true, ci: 0.2, segments: segs(7) } });
    });
    expect(lines()).toBe(5);                                  // 7 lines are pattern 0's
  }, 10000);

  it('the answer to a tilt edit of a pattern that is no longer selected is not drawn', async () => {
    render(<PCRefinement />);
    await pick(0, 0);
    await waitFor(() => expect(backend.results[0]).toHaveLength(1));
    await releaseResult(0, 3);
    await waitFor(() => expect(lines()).toBe(3));

    fireEvent.change(screen.getByTitle(/Sample tilt from horizontal/), { target: { value: '72' } });
    await waitFor(() => expect(backend.tilts).toHaveLength(1), { timeout: 4000 });
    await pick(0, 1);
    await waitFor(() => expect(backend.results[1]).toHaveLength(1));
    await releaseResult(1, 5);
    await waitFor(() => expect(lines()).toBe(5));
    await act(async () => {
      backend.tilts[0].resolve({ data: { success: true, ci: 0.2, segments: segs(7) } });
    });
    expect(lines()).toBe(5);
  }, 10000);

  it('removing the selected pattern drops its lines and voids its pending answers', async () => {
    render(<PCRefinement />);
    await pick(0, 0);
    await waitFor(() => expect(backend.results[0]).toHaveLength(1));
    await releaseResult(0, 3);
    await waitFor(() => expect(lines()).toBe(3));

    useTyped();
    await waitFor(() => expect(backend.updates).toHaveLength(1), { timeout: 4000 });

    fireEvent.click(screen.getByTitle(/Remove the selected pattern/));
    await waitFor(() => expect(lines()).toBe(0));             // its lines go with it
    await act(async () => {
      backend.updates[0].resolve({ data: { success: true, ci: 0.2, segments: segs(7) } });
    });
    expect(lines()).toBe(0);                                  // and its late answer is not drawn
  }, 10000);

  it('of two PC edits only the newest draws, whichever answer comes last', async () => {
    render(<PCRefinement />);
    await pick(0, 0);
    await waitFor(() => expect(backend.results[0]).toHaveLength(1));
    await releaseResult(0, 3);
    await waitFor(() => expect(lines()).toBe(3));

    // two presses of "Use these numbers", the second with other numbers while the first has no answer yet
    useTyped('0.51');
    await waitFor(() => expect(backend.updates).toHaveLength(1), { timeout: 4000 });
    useTyped('0.52');
    await waitFor(() => expect(backend.updates).toHaveLength(2), { timeout: 4000 });

    await act(async () => { backend.updates[1].resolve({ data: { success: true, ci: 0.3, segments: segs(6) } }); });
    await waitFor(() => expect(lines()).toBe(6));
    await act(async () => { backend.updates[0].resolve({ data: { success: true, ci: 0.2, segments: segs(2) } }); });
    expect(lines()).toBe(6);                                  // the older answer does not overwrite it
  }, 15000);

  it('the answer to Index Pattern of a pattern that is no longer selected is not drawn', async () => {
    render(<PCRefinement />);
    await pick(0, 0);
    await waitFor(() => expect(backend.results[0]).toHaveLength(1));
    await releaseResult(0, 3);
    await waitFor(() => expect(lines()).toBe(3));

    fireEvent.click(await screen.findByText(/Index Pattern/));
    await waitFor(() => expect(backend.indexes).toHaveLength(1));
    expect([backend.indexes[0].row, backend.indexes[0].col]).toEqual([0, 0]);

    await pick(0, 1);
    await waitFor(() => expect(backend.results[1]).toHaveLength(1));
    await releaseResult(1, 5);
    await waitFor(() => expect(lines()).toBe(5));
    await act(async () => {
      backend.indexes[0].resolve({ data: { success: true, ci: 0.4, segments: segs(8), n_bands: 8 } });
    });
    expect(lines()).toBe(5);
  });

  it('a refine draws the lines of the pattern that is selected, not the first one', async () => {
    render(<PCRefinement />);
    await pick(0, 1);
    await waitFor(() => expect(backend.results[1]).toHaveLength(1));
    await releaseResult(1, 5);
    await waitFor(() => expect(lines()).toBe(5));

    fireEvent.click(await screen.findByText(/Global PC Refine/));
    // the refine's own top-level segments are pattern 0's (3 lines); pattern 1's are 9
    await waitFor(() => expect(lines()).toBe(9), { timeout: 4000 });
  }, 10000);
});
