// @vitest-environment jsdom
//
// A file that carries no PC and none was set: the PC page says so (the fields show the
// placeholder 0.5/0.5/0.5, and indexing is refused). /api/pc/status carries pc_source.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';

const backend = vi.hoisted(() => ({ source: 'missing' }));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: true, phase_name: 'Ni', phase_names: ['Ni'],
      n_patterns: 0, pc: [0.5, 0.5, 0.5], pc_basis: 'single', pc_per_pixel: false, patterns: [],
      pc_source: backend.source,
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

describe('the "no PC calibrated" hint', () => {
  it('is shown when the store says the PC is missing', async () => {
    backend.source = 'missing';
    render(<PCRefinement />);
    const hint = await screen.findByTestId('pc-missing-hint');
    expect(hint.textContent).toMatch(/^No PC yet, indexing is blocked\.This file has no pattern centre/);
    expect(hint.querySelector('b').textContent).toBe('No PC yet, indexing is blocked.');
  });

  it.each(['header', 'manual', 'refined', 'refined_map'])('is not shown for a PC from %s', async (source) => {
    backend.source = source;
    render(<PCRefinement />);
    await screen.findByTestId('pc-file-head');
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByTestId('pc-missing-hint')).toBeNull();
  });
});
