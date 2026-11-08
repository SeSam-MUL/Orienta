// @vitest-environment jsdom
//
// The PC Refinement page shows what the backend holds, and nothing that
// belongs to a pattern or a PC it no longer shows.
//
// Regression (v0.4.7): after examining patterns on this page and going back
// to the viewer for processing, the page stopped responding. Processing with deep copy on
// makes a new active dataset whose PC session is empty; the page kept the old
// list and "phase loaded", and every action failed against the empty session.
// The other cases are neighbours of the same problem.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

const backend = vi.hoisted(() => ({
  status: null, removeFails: false, optimize: null, renders: [],
}));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => (backend.statusOverride ? backend.statusOverride() : ok(backend.status)),
    getPatternImage: (idx) => ok({ image: `IMG${idx}` }),
    getPatternResult: (idx) => ok({ success: true, segments: [], ci: 0.4 + idx / 100 }),
    removePattern: () => (backend.removeFails
      ? Promise.reject(Object.assign(new Error('409'), { response: { status: 409, data: { detail: 'refine running' } } }))
      : ok({ success: true })),
    optimize: () => ok(backend.optimize),
    loadPhase: () => ok({ success: true, phase_name: 'Ni', name: 'Ni' }),
    renderPreview: (req) => {
      backend.renders.push(req);
      return ok({ simulated_b64: null, ncc: 0.5, ncc_reason: null,
                  orientation_euler_deg: [1, 2, 3], orientation_source: 'hough' });
    },
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, ebsdApi: quiet, indexApi: quiet, calibrationApi: quiet };
});

import PCRefinement from './PCRefinement';

const NI = { path: '/sht/Ni/Ni (Ni) [cF4] {20kV}.sht', filename: 'Ni (Ni) [cF4] {20kV}.sht', material: 'Ni', display_label: 'Ni', formula: 'Ni' };
const status = (n, { phase = 'Ni' } = {}) => ({
  has_detector: true, has_phase: !!phase, phase_name: phase, phase_names: phase ? [phase] : [],
  n_patterns: n, pc: [0.5, 0.3, 0.8], pc_basis: 'single', pc_per_pixel: false,
  patterns: Array.from({ length: n }, (_, i) => ({ index: i, row: i + 1, col: 2 })),
});

const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});

beforeEach(() => {
  backend.status = status(2);
  backend.removeFails = false;
  backend.optimize = null;
  backend.renders = [];
  HTMLCanvasElement.prototype.getContext = function getContext() {
    this.__ctx = this.__ctx || fakeContext();
    return this.__ctx;
  };
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [NI] }) }));
});
afterEach(() => cleanup());

const showPage = (page) => act(() => {
  window.dispatchEvent(new CustomEvent('page-changed', { detail: { page } }));
});
const items = () => screen.queryAllByTitle(/^Pattern at row \d+, col 2/);
const headerCi = () => screen.getByText(/^CI:/).textContent;
const experimental = () => screen.queryByAltText(/experimental/i);

describe('PC Refinement page follows the backend', () => {
  it('shows an empty session as empty after the active dataset changed', async () => {
    render(<PCRefinement />);
    await showPage('pcrefinement');
    await waitFor(() => expect(items()).toHaveLength(2));
    // With a phase loaded the forward preview offers its masters.
    await waitFor(() => expect(screen.queryByTitle(/forward-simulate the EBSD pattern/)).not.toBeNull());

    backend.status = status(0, { phase: null });   // processed copy: a session of its own
    await showPage('ebsdviewer');
    await showPage('pcrefinement');
    await waitFor(() => expect(items()).toHaveLength(0));
    expect(screen.getByText('No patterns added')).toBeTruthy();
    expect(screen.getAllByText('No phase loaded').length).toBeGreaterThan(0);
    expect(headerCi()).toMatch(/N\/A/);
    // ...and the page treats the phase as gone, not only its label: no masters
    // are offered for a phase the session does not have.
    await waitFor(() => expect(screen.queryByTitle(/forward-simulate the EBSD pattern/)).toBeNull());
  });

  it('drops the old "Phase loaded" message when a visit finds the session without a phase', async () => {
    render(<PCRefinement />);
    await showPage('pcrefinement');
    await waitFor(() => expect(items()).toHaveLength(2));
    const cif = screen.getByPlaceholderText(/type a \.cif path/i);
    fireEvent.change(cif, { target: { value: 'C:/lib/Ni.cif' } });
    fireEvent.keyDown(cif, { key: 'Enter' });
    await screen.findByText(/Phase loaded: Ni/);
    const headerCard = () => [...document.querySelectorAll('textarea')].some((x) => /Ni/.test(x.value));
    expect(headerCard()).toBe(true);

    backend.status = status(0, { phase: null });
    await showPage('ebsdviewer');
    await showPage('pcrefinement');
    await waitFor(() => expect(items()).toHaveLength(0));
    await waitFor(() => expect(screen.queryByText(/Phase loaded: Ni/)).toBeNull());
    expect(headerCard()).toBe(false);          // the header's phase card goes too
  });

  it('takes the PC of the session into the fields and the preview on a visit', async () => {
    render(<PCRefinement />);
    await showPage('pcrefinement');
    await waitFor(() => expect(backend.renders.at(-1)?.pc).toEqual([0.5, 0.3, 0.8]), { timeout: 4000 });

    backend.status = { ...status(2), pc: [0.61, 0.33, 0.72] };   // a processed copy with its own PC
    await showPage('ebsdviewer');
    await showPage('pcrefinement');
    await waitFor(() => expect(backend.renders.at(-1)?.pc).toEqual([0.61, 0.33, 0.72]), { timeout: 4000 });
  }, 15000);

  it('drops a status answer that was asked before a removal', async () => {
    let late;
    render(<PCRefinement />);
    await showPage('pcrefinement');
    await waitFor(() => expect(items()).toHaveLength(2));

    const slow = new Promise((res) => { late = res; });
    const realStatus = backend.statusOverride;
    backend.statusOverride = () => slow;             // the next status hangs ...
    await showPage('pcrefinement');
    backend.statusOverride = realStatus;
    fireEvent.click(screen.getAllByTitle(/remove the selected pattern/i)[0]);
    await waitFor(() => expect(items()).toHaveLength(1));
    await act(async () => { late({ data: status(2) }); });   // ... and answers after the removal
    expect(items()).toHaveLength(1);                 // the removed pattern does not come back
  });

  it('takes the picture and the CI of a removed pattern away', async () => {
    render(<PCRefinement />);
    await showPage('pcrefinement');            // selects the last pattern
    await waitFor(() => expect(experimental()?.getAttribute('src')).toContain('IMG1'));
    await waitFor(() => expect(headerCi()).toMatch(/0\.41/));

    fireEvent.click(screen.getAllByTitle(/remove the selected pattern/i)[0]);
    await waitFor(() => expect(items()).toHaveLength(1));
    expect(experimental()).toBeNull();
    expect(headerCi()).toMatch(/N\/A/);
  });

  it('keeps the backend list when the backend refuses the removal', async () => {
    render(<PCRefinement />);
    await showPage('pcrefinement');
    await waitFor(() => expect(items()).toHaveLength(2));

    backend.removeFails = true;
    fireEvent.click(screen.getAllByTitle(/remove the selected pattern/i)[0]);
    await act(async () => {});
    await waitFor(() => expect(items()).toHaveLength(2));   // not a local list of 1
  });

  it('shows the CI of the selected pattern after a refine, and previews at the refined PC', async () => {
    backend.optimize = { result: {
      ci: 0.111,                                 // pattern 0's
      mean_pc: [0.52, 0.31, 0.79],
      pattern_results: [{ index: 0, ci: 0.111, segments: [] }, { index: 1, ci: 0.777, segments: [] }],
    } };
    render(<PCRefinement />);
    await showPage('pcrefinement');            // pattern 1 selected
    await waitFor(() => expect(headerCi()).toMatch(/0\.41/));

    fireEvent.click(screen.getByText('Global PC Refine'));
    await waitFor(() => expect(headerCi()).toMatch(/0\.7770/));
    await waitFor(() => expect(backend.renders.at(-1)?.pc).toEqual([0.52, 0.31, 0.79]), { timeout: 4000 });
  }, 10000);
});
