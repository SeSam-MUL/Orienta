// @vitest-environment jsdom
//
// "Apply Pixel Size" on the PC Refinement page: the detector width and binning
// the user entered must reach the backend detector (/api/pc/params/update) AND the
// forward preview, which renders with the same pixel size and binning. The page
// used to post them to a route that dropped them and never told the preview.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

const backend = vi.hoisted(() => ({ params: [], previews: [] }));

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
    updateParams: (p) => { backend.params.push(p); return ok({ success: true }); },
    renderPreview: (args) => {
      backend.previews.push(args);
      return ok({ simulated_b64: null, ncc: 0.5, orientation_euler_deg: [0, 0, 0],
                  orientation_source: 'hough', indexed_phase: 'Ni' });
    },
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const ebsdApi = new Proxy({
    // the file's detector: 60 x 80 px patterns, no pixel size stored
    getDetector: () => ok({
      has_detector: true, shape: [60, 80], pc: [0.5, 0.3, 0.8], sample_tilt: 70,
      camera_tilt: 0, azimuthal: 0, binning: 1, pixel_size: null,
    }),
  }, { get: (t, k) => (k in t ? t[k] : () => ok({})) });
  const quiet = new Proxy({}, { get: () => () => ok({}) });
  return { ...orig, pcApi, ebsdApi, indexApi: quiet, calibrationApi: quiet };
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
  backend.params = []; backend.previews = [];
  globalThis.fetch = vi.fn(() => Promise.resolve({
    json: () => Promise.resolve({ files: [{ path: '/sht/Ni (Ni) [cF4] {20kV}.sht', filename: 'Ni (Ni) [cF4] {20kV}.sht' }] }),
  }));
});
afterEach(() => cleanup());

describe('Apply Pixel Size', () => {
  it('sends binning and detector width to the backend detector', async () => {
    render(<PCRefinement />);
    const binning = await screen.findByTitle(/Camera binning factor/);
    fireEvent.change(binning, { target: { value: '2' } });
    fireEvent.change(screen.getByTitle(/Physical width of the EBSD detector/), { target: { value: '30' } });
    fireEvent.click(screen.getByText('Apply Pixel Size'));
    await waitFor(() => expect(backend.params).toHaveLength(1));
    expect(backend.params[0]).toEqual({ binning: 2, detector_width_mm: 30 });
  });

  it('the forward preview then renders with the pixel size that follows from them', async () => {
    render(<PCRefinement />);
    // select the pattern so the preview runs
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await waitFor(() => expect(backend.previews.length).toBeGreaterThan(0), { timeout: 4000 });
    expect(backend.previews.at(-1).binning).toBe(1);
    expect(backend.previews.at(-1).pixelSize).toBeNull();

    const binning = await screen.findByTitle(/Camera binning factor/);
    fireEvent.change(binning, { target: { value: '2' } });
    fireEvent.change(screen.getByTitle(/Physical width of the EBSD detector/), { target: { value: '30' } });
    fireEvent.click(screen.getByText('Apply Pixel Size'));

    // 30 mm over 80 columns of 2 x 2 binned pixels: 30000 / (80 * 2) um per unbinned pixel
    await waitFor(() => expect(backend.previews.at(-1).pixelSize).toBeCloseTo(187.5, 6), { timeout: 4000 });
    expect(backend.previews.at(-1).binning).toBe(2);
  }, 15000);
});
