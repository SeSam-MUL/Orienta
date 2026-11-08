// @vitest-environment jsdom
//
// The forward preview shows an NCC only for an orientation that was found. When
// the backend says no orientation could be indexed (ncc_reason), the page says so
// instead of showing a number.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

const backend = vi.hoisted(() => ({ reply: {} }));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
      n_patterns: 1, pc: [0.5, 0.3, 0.8], pc_basis: 'single', pc_per_pixel: false,
      patterns: [{ index: 0, row: 1, col: 2 }],
    }),
    getPatternImage: () => ok({ image: 'AAAA' }),
    getPatternResult: () => ok({ success: true, segments: [], ci: 0.4 }),
    renderPreview: () => ok(backend.reply),
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
  globalThis.fetch = vi.fn(() => Promise.resolve({
    json: () => Promise.resolve({ files: [{ path: '/sht/Ni (Ni) [cF4] {20kV}.sht', filename: 'Ni (Ni) [cF4] {20kV}.sht' }] }),
  }));
});
afterEach(() => cleanup());

describe('forward preview NCC', () => {
  it('says that no orientation was found instead of showing a number', async () => {
    backend.reply = { simulated_b64: null, ncc: null, ncc_reason: 'no_orientation',
                      orientation_euler_deg: [0, 0, 0], orientation_source: 'identity' };
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    const note = await screen.findByTestId('pc-ncc-none', {}, { timeout: 4000 });
    expect(note.textContent).toMatch(/no orientation could be indexed/);
    expect(screen.queryByText(/^NCC = /)).toBeNull();
  }, 10000);

  it('shows the number for an orientation that was found, and no note', async () => {
    backend.reply = { simulated_b64: null, ncc: 0.4321, ncc_reason: null,
                      orientation_euler_deg: [1, 2, 3], orientation_source: 'hough' };
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await screen.findByText(/0\.4321/, {}, { timeout: 4000 });
    expect(screen.queryByTestId('pc-ncc-none')).toBeNull();
  }, 10000);
});
