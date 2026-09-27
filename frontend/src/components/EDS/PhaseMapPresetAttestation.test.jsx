// @vitest-environment jsdom
/**
 * A compatibility report is a statement about ONE scan, and it has to stay
 * one.
 *
 * The defect a service-lab tester found: `appliedPreset` lived in
 * `PhaseMapControls` state and was written only by `PresetBar.onApplied`.
 * Nothing reset it when the loaded file changed, although three effects in
 * the same file already keyed on `filePath` for exactly this reason. So:
 * apply a preset to scan A (it passes), switch to scan B, classify, export —
 * and `provenance.json` carries `compatibility: {ok: true}` for a scan that
 * was never checked, with nothing in the folder to reveal it. A false
 * "compatible" stamp is worse than no stamp.
 *
 * Both halves are pinned here:
 *   (a) the attestation is DROPPED on a file switch, so the export request
 *       carries `compatibility: null` rather than A's clean bill of health;
 *   (b) while it is alive it names the file it was checked against and the
 *       moment it was checked, so one that ever survives by another route
 *       reads as stale rather than as authority.
 *
 * This is the repo's documented dominant defect class — state held twice,
 * only one copy updated — so the test drives the real hook and the real
 * dialog rather than asserting on a stub.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  render, screen, fireEvent, waitFor, cleanup, act,
} from '@testing-library/react';

// The loaded file. Reassigned between renders to simulate the switch.
let FILE = '/data/scanA.h5oina';

vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: FILE }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

const MAP = {
  loaded: true,
  n_rows: 4,
  n_cols: 4,
  image: '',
  summary: [{ phase_index: 0, cif_filename: 'Al.cif', n_pixels: 16, percentage: 100 }],
  all_phases: [],
  regions: [],
  clusters: [],
};

const runExport = vi.fn();
const checkPreset = vi.fn();
const getPreset = vi.fn();
const listPresets = vi.fn();

vi.mock('../../services/api', () => ({
  edsApi: {
    getPhaseMap: vi.fn(() => Promise.resolve({ data: MAP })),
    setPhaseColors: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    cifPhases: vi.fn(() => Promise.resolve({ data: { phases: [] } })),
    elements: vi.fn(() => Promise.resolve({ data: { elements: ['Al', 'Si'] } })),
    autoClassify: vi.fn(() => Promise.resolve({ data: MAP })),
    clearPhaseMap: vi.fn(() => Promise.resolve({ data: {} })),
  },
  edsExportApi: {
    listPresets: (...a) => listPresets(...a),
    getPreset: (...a) => getPreset(...a),
    checkPreset: (...a) => checkPreset(...a),
    runExport: (...a) => runExport(...a),
    exportPreview: vi.fn(() => Promise.resolve({
      data: {
        ok: true, n_phases: 1, n_regions: 1, n_particles: 3,
        n_classified_px: 16, n_total_px: 16, area_available: true,
        step_x_um: 0.5, step_y_um: 0.5, warnings: [],
      },
    })),
    savePreset: vi.fn(),
    deletePreset: vi.fn(),
    importPreset: vi.fn(),
    exportPresetFile: vi.fn(),
  },
}));

import {
  usePhaseMap, PhaseMapControls, stampCompatibility, NO_APPLIED_PRESET,
} from './PhaseMapPanel';

const PRESET = { name: 'Al matrix', author: 'P. R.', created: '2026-08-27',
                 content_hash: 'aaaabbbbcccc', material_class: 'AA6061' };

function Harness() {
  const handle = usePhaseMap({});
  return <PhaseMapControls handle={handle} />;
}

const flush = () => act(async () => {
  await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
});

beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  FILE = '/data/scanA.h5oina';
  listPresets.mockResolvedValue({ data: { presets: [PRESET] } });
  getPreset.mockResolvedValue({
    data: { preset: { ...PRESET, settings: { mode: 'cluster' } } } });
  checkPreset.mockResolvedValue({ data: { ok: true, blockers: [], warnings: [] } });
  runExport.mockResolvedValue({
    data: { ok: true, folder: 'D:/out', files: [], warnings: [], provenance: {} } });
});
afterEach(() => cleanup());

/** Apply the one preset in the list through the real bar. */
async function applyPreset() {
  fireEvent.click(screen.getByLabelText('presets.title'));
  fireEvent.click(document.querySelector('[data-preset-option="Al matrix"]'));
  fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));
  await waitFor(() => expect(getPreset).toHaveBeenCalled());
  await flush();
}

/** Open the export dialog, fill a destination and run it. */
async function exportOnce() {
  fireEvent.click(screen.getByRole('button', { name: 'export.button' }));
  await waitFor(() => expect(document.querySelector('[data-eds-export-dialog]')).toBeTruthy());
  fireEvent.change(screen.getByLabelText('export.dest'), { target: { value: 'D:/out' } });
  fireEvent.click(screen.getByRole('button', { name: 'export.run' }));
  await waitFor(() => expect(runExport).toHaveBeenCalled());
  return runExport.mock.calls[runExport.mock.calls.length - 1][0];
}

describe('stampCompatibility', () => {
  it('names the file the report was checked against, and when', () => {
    const out = stampCompatibility({ ok: true, warnings: [] }, '/data/scanA.h5oina',
                                   new Date('2026-08-27T09:30:00Z'));
    expect(out.ok).toBe(true);
    expect(out.checked_file).toBe('/data/scanA.h5oina');
    expect(out.checked_at).toBe('2026-08-27T09:30:00.000Z');
  });

  it('keeps the refusal it was given rather than rewriting it', () => {
    const report = { ok: false, overridden: true, blockers: [{ code: 'matrix_mismatch' }] };
    const out = stampCompatibility(report, '/x.h5oina');
    expect(out.ok).toBe(false);
    expect(out.overridden).toBe(true);
    expect(out.blockers[0].code).toBe('matrix_mismatch');
  });

  it('stamps nothing onto nothing — "no preset" must stay no preset', () => {
    expect(stampCompatibility(null, '/x.h5oina')).toBe(null);
    // A file the app somehow does not know is recorded as unknown, never as
    // the last file that happened to be open.
    expect(stampCompatibility({ ok: true }, '').checked_file).toBe(null);
  });

  it('NO_APPLIED_PRESET is the empty record, not a half-filled one', () => {
    expect(NO_APPLIED_PRESET).toEqual({ name: '', compatibility: null });
  });
});

describe('the hook says which scan its settings describe', () => {
  it('hands the loaded file out, so the controls can key on it', async () => {
    render(<Harness />);
    await flush();
    // Without this the controls have no way to notice a switch at all.
    expect(document.querySelector('[data-preset-bar]')).toBeTruthy();
  });
});

describe('the attestation does not survive a file switch', () => {
  it('travels into the export request, naming the scan it was checked against', async () => {
    render(<Harness />);
    await flush();
    await applyPreset();

    const sent = await exportOnce();
    expect(sent.presetName).toBe('Al matrix');
    expect(sent.compatibility.ok).toBe(true);
    expect(sent.compatibility.checked_file).toBe('/data/scanA.h5oina');
    // An ISO instant, so a reader can tell a fresh check from a stale one.
    expect(Number.isFinite(Date.parse(sent.compatibility.checked_at))).toBe(true);
  });

  it('is GONE after the loaded file changes — no clean bill of health for scan B',
     async () => {
       const { rerender } = render(<Harness />);
       await flush();
       await applyPreset();

       // The user loads another scan. Everything else about the page carries
       // on; the attestation must not.
       FILE = '/data/scanB.h5oina';
       rerender(<Harness />);
       await flush();

       const sent = await exportOnce();
       expect(sent.compatibility).toBe(null);
       expect(sent.presetName).toBe('');
     });

  it('a refusal that was overridden is dropped by the switch too', async () => {
    checkPreset.mockResolvedValue({
      data: { ok: false, blockers: [{ code: 'matrix_mismatch', message: 'not Al' }],
              warnings: [] },
    });
    const { rerender } = render(<Harness />);
    await flush();

    fireEvent.click(screen.getByLabelText('presets.title'));
    fireEvent.click(document.querySelector('[data-preset-option="Al matrix"]'));
    fireEvent.click(screen.getByRole('button', { name: 'presets.apply' }));
    await waitFor(() => expect(document.querySelector('[data-preset-blockers]')).toBeTruthy());
    fireEvent.click(screen.getByRole('button', { name: 'presets.applyAnyway' }));
    await waitFor(() => expect(getPreset).toHaveBeenCalled());
    await flush();

    // Alive on the scan it was refused for...
    let sent = await exportOnce();
    expect(sent.compatibility.ok).toBe(false);
    expect(sent.compatibility.overridden).toBe(true);
    expect(sent.compatibility.checked_file).toBe('/data/scanA.h5oina');

    fireEvent.click(screen.getByRole('button', { name: 'export.close' }));
    FILE = '/data/scanB.h5oina';
    rerender(<Harness />);
    await flush();

    // ...and gone on the next one. Carrying an override across files would
    // record "applied over a refusal" against a scan nobody refused it for.
    sent = await exportOnce();
    expect(sent.compatibility).toBe(null);
  });

  it('shows on screen which file the report was checked against', async () => {
    render(<Harness />);
    await flush();
    await applyPreset();

    fireEvent.click(screen.getByRole('button', { name: 'export.button' }));
    await waitFor(() => expect(document.querySelector('[data-eds-export-dialog]')).toBeTruthy());
    const line = document.querySelector('[data-export-checked-against]');
    expect(line).toBeTruthy();
    expect(line.textContent).toContain('/data/scanA.h5oina');
  });
});
