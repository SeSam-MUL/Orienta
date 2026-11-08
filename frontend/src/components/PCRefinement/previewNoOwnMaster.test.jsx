// @vitest-environment jsdom
//
// A phase the library has no master for gets no preview against another
// phase's master. Alpha-/beta-Ti loaded from their own
// CIFs were previewed against the Mg master, NCC 0.016 - read as a bad pattern
// centre. The page now names the missing master, renders nothing on its own,
// and still lets the user pick a master on purpose.
import React from 'react';
import { describe, it, expect, afterEach, beforeEach, vi } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';

const backend = vi.hoisted(() => ({ phase: 'Ni', renders: [] }));

vi.mock('../../services/api', async (importOriginal) => {
  const orig = await importOriginal();
  const ok = (data = {}) => Promise.resolve({ data });
  const pcApi = new Proxy({
    status: () => ok({
      has_detector: true, has_phase: true, phase_name: backend.phase, phase_names: [backend.phase],
      n_patterns: 1, pc: [0.5, 0.3, 0.8], pc_basis: 'single', pc_per_pixel: false,
      patterns: [{ index: 0, row: 1, col: 2 }],
    }),
    getPatternImage: () => ok({ image: 'AAAA' }),
    getPatternResult: () => ok({ success: true, segments: [], ci: 0.4 }),
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

const MG = { path: '/sht/Mg/Z12 (Mg_COD-9008506) [hP6] {20kV}.sht', filename: 'Z12 (Mg_COD-9008506) [hP6] {20kV}.sht', material: 'Mg', display_label: 'Mg', formula: 'Mg' };
const NI = { path: '/sht/Ni/Ni (Ni) [cF4] {20kV}.sht', filename: 'Ni (Ni) [cF4] {20kV}.sht', material: 'Ni', display_label: 'Ni', formula: 'Ni' };

const fakeContext = () => new Proxy({}, {
  get: (t, k) => (k in t ? t[k] : (k === 'measureText' ? () => ({ width: 0 }) : () => ({}))),
  set: (t, k, v) => { t[k] = v; return true; },
});

beforeEach(() => {
  backend.renders = [];
  HTMLCanvasElement.prototype.getContext = function getContext() {
    this.__ctx = this.__ctx || fakeContext();
    return this.__ctx;
  };
  globalThis.fetch = vi.fn(() => Promise.resolve({ json: () => Promise.resolve({ files: [MG, NI] }) }));
});
afterEach(() => cleanup());

const shtSelect = () => screen.getByTitle(/forward-simulate the EBSD pattern/);

describe('forward preview for a phase without a master of its own', () => {
  it('names the missing master, renders nothing, and lets the user pick one', async () => {
    backend.phase = 'titan-alpha-mp-46';
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));

    const placeholder = await screen.findByTestId('pc-sht-no-own-master', {}, { timeout: 4000 });
    expect(placeholder.textContent).toMatch(/no master for titan-alpha-mp-46/);
    expect(shtSelect().value).toBe('');          // not Mg shown as if chosen
    expect(screen.getAllByText(/no master for titan-alpha-mp-46/).length).toBe(2);   // dropdown + pane
    await new Promise((r) => setTimeout(r, 600));   // past the preview debounce
    expect(backend.renders).toEqual([]);

    fireEvent.change(shtSelect(), { target: { value: MG.path } });   // on purpose
    await waitFor(() => expect(backend.renders.at(-1)?.shtPath).toBe(MG.path), { timeout: 4000 });
    expect(screen.queryByTestId('pc-sht-no-own-master')).toBeNull();
  }, 15000);

  it('a visit during a phase change does not keep the old phase\'s master', async () => {
    // The phase was noted as handled when the master list was ASKED;
    // a visit that voided that answer then kept the old master (Mg for Ni).
    backend.phase = 'Mg';
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await waitFor(() => expect(shtSelect().value).toBe(MG.path), { timeout: 4000 });

    const held = [];
    globalThis.fetch = vi.fn(() => new Promise((res) => held.push(
      () => res({ json: () => Promise.resolve({ files: [MG, NI] }) }))));
    const visit = () => act(async () => {
      window.dispatchEvent(new CustomEvent('page-changed', { detail: { page: 'pcrefinement' } }));
    });
    backend.phase = 'Ni';
    await visit();                                    // status says Ni: the list is asked for Ni ...
    await waitFor(() => expect(held.length).toBeGreaterThanOrEqual(1));   // asked for Ni, held
    await visit();                                    // ... and a second visit voids that answer
    await act(async () => { held.splice(0).forEach((go) => go()); });
    await waitFor(() => expect(shtSelect().value).toBe(NI.path), { timeout: 4000 });
  }, 15000);

  it("takes the phase's own master and shows no placeholder", async () => {
    backend.phase = 'Ni';
    render(<PCRefinement />);
    fireEvent.click(await screen.findByTitle(/Pattern at row 1, col 2/));
    await waitFor(() => expect(backend.renders.at(-1)?.shtPath).toBe(NI.path), { timeout: 4000 });
    expect(shtSelect().value).toBe(NI.path);
    expect(screen.queryByTestId('pc-sht-no-own-master')).toBeNull();
  }, 15000);
});
