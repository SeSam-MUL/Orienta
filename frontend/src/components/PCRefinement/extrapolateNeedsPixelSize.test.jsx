// @vitest-environment jsdom
//
// "Extrapolate from Points" without a pixel size is refused by the backend with a
// coded 400 (pixel_size_unknown); the page shows the sentence in its own language.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react';

const backend = vi.hoisted(() => ({ calls: [] }));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
      n_patterns: 0, pc: [0.5, 0.3, 0.8], pc_basis: 'single', pc_per_pixel: false, patterns: [],
    }),
    runGridCalibration: (step, mode) => {
      backend.calls.push([step, mode]);
      return Promise.reject({
        response: {
          status: 400,
          headers: { 'x-orienta-code': 'pixel_size_unknown' },
          data: { detail: 'Extrapolating the PC ... English prose' },
        },
        message: 'Request failed with status code 400',
      });
    },
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
  backend.calls = [];
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [] }) }));
});
afterEach(() => cleanup());

describe('Extrapolate from Points', () => {
  it('shows the translated refusal, not the English prose', async () => {
    render(<PCRefinement />);
    const mode = await screen.findByTitle(/PC correction mode/i);
    fireEvent.change(mode, { target: { value: 'Extrapolate from Points' } });
    fireEvent.click(await screen.findByText('Run Grid Calibration'));
    await waitFor(() => expect(backend.calls).toHaveLength(1));
    expect(backend.calls[0][1]).toBe('extrapolate');
    const msg = await screen.findByText(/needs the detector's pixel size/);
    expect(msg.textContent).toMatch(/Apply Pixel Size/);
    expect(msg.textContent).not.toMatch(/English prose/);
  });
});
