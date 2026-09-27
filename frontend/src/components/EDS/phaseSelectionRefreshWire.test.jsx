// @vitest-environment jsdom
/**
 * Opening the phase picker must ASK the backend again.
 *
 * `PhaseSelection.test.jsx` proves `refreshCifPhases` merges correctly, by
 * calling it. That cannot catch the defect this file is for: the function being
 * right and nothing calling it. That is the failure class this repo keeps
 * hitting — two rendered, unwired IPF buttons; a warning the backend builds
 * that nobody reads; an attestation that is never recomputed. Both sides valid,
 * only together wrong.
 *
 * So this one renders the real controls, opens the real <details>, and counts
 * the real requests.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, fireEvent, cleanup, act } from '@testing-library/react';

vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: '/data/scan.h5oina' }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o ? `${k}:${JSON.stringify(o)}` : k) }),
}));

const THREE = {
  data: {
    phases: [
      { key: 'a', cif_filename: 'Al.cif', formula: 'Al' },
      { key: 'b', cif_filename: 'Si.cif', formula: 'Si' },
      { key: 'c', cif_filename: 'sd_0302719.cif', formula: 'AlFeMnSi' },
    ],
  },
};
const FOUR = {
  data: {
    phases: [...THREE.data.phases,
             { key: 'd', cif_filename: 'Mg2Si.cif', formula: 'Mg2Si' }],
  },
};

vi.mock('../../services/api', () => ({
  edsApi: {
    getPhaseMap: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    setPhaseColors: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    cifPhases: vi.fn(),
    elements: vi.fn(() => Promise.resolve({ data: { elements: [] } })),
    autoClassify: vi.fn(),
    clearPhaseMap: vi.fn(),
  },
  edsExportApi: {
    listPresets: vi.fn(() => Promise.resolve({ data: { presets: [] } })),
    getPreset: vi.fn(), checkPreset: vi.fn(), savePreset: vi.fn(),
    deletePreset: vi.fn(), importPreset: vi.fn(), exportPresetFile: vi.fn(),
    exportPreview: vi.fn(), runExport: vi.fn(),
  },
}));

import { edsApi } from '../../services/api';
import { usePhaseMap, PhaseMapControls } from './PhaseMapPanel';

function Harness() {
  const handle = usePhaseMap({});
  return <PhaseMapControls handle={handle} />;
}

const flush = () => act(async () => { await Promise.resolve(); await Promise.resolve(); });

function picker() {
  const el = document.querySelector('[data-testid="phase-selection"]');
  expect(el, 'the phase-selection <details> must be on screen').toBeTruthy();
  return el;
}

/** Set the open state and let jsdom fire the REAL `toggle` event, once.
 *
 * Firing a hand-built `toggle` on top of setting `.open` calls the handler
 * TWICE — jsdom queues its own event for the state change — which made the
 * first version of this file count 3 requests and read like a product bug.
 */
async function setPickerOpen(open) {
  const d = picker();
  await act(async () => {
    d.open = open;
    await new Promise((resolve) => { setTimeout(resolve, 0); });
  });
  expect(d.open).toBe(open);
}

const openPicker = () => setPickerOpen(true);

function shownFilenames() {
  return [...picker().querySelectorAll('label')]
    .map((l) => l.textContent.trim())
    .filter(Boolean);
}

beforeEach(() => {
  cleanup();
  vi.clearAllMocks();
  edsApi.cifPhases.mockResolvedValue(THREE);
});
afterEach(() => cleanup());

describe('the phase picker is wired to the refresh', () => {
  it('fetches again when it is opened, not only on first load', async () => {
    render(<Harness />);
    await flush();
    expect(edsApi.cifPhases).toHaveBeenCalledTimes(1);   // the per-file load

    await openPicker();
    expect(edsApi.cifPhases).toHaveBeenCalledTimes(2);   // and again on open
  });

  it('shows a CIF that arrived after the panel was first drawn', async () => {
    // The user's report: download a CIF mid-session, and it is missing from the
    // checkboxes until the file is reloaded.
    render(<Harness />);
    await flush();
    expect(shownFilenames()).not.toContain('Mg2Si.cif');

    edsApi.cifPhases.mockResolvedValue(FOUR);            // the download
    await openPicker();

    expect(shownFilenames()).toContain('Mg2Si.cif');
  });

  it('does not ask again when the picker is closed', async () => {
    // A `toggle` fires on close too, and a request there is pure waste.
    render(<Harness />);
    await flush();
    await openPicker();
    const afterOpen = edsApi.cifPhases.mock.calls.length;

    await setPickerOpen(false);

    expect(edsApi.cifPhases).toHaveBeenCalledTimes(afterOpen);
  });

  it('keeps the checkboxes the user had ticked across the refresh', async () => {
    render(<Harness />);
    await flush();
    const boxes = () => [...picker().querySelectorAll('input[type="checkbox"]')];

    await openPicker();
    expect(boxes()).toHaveLength(3);
    fireEvent.click(boxes()[1]);                         // untick Si.cif
    await flush();
    expect(boxes().map((b) => b.checked)).toEqual([true, false, true]);

    edsApi.cifPhases.mockResolvedValue(FOUR);
    // Closed and opened again — no `toggle` fires for a state that did not
    // change, so this is also how the user gets a second refresh.
    await setPickerOpen(false);
    await openPicker();

    // Si stays unticked; the phase that appeared is ticked.
    expect(boxes().map((b) => b.checked)).toEqual([true, false, true, true]);
  });
});
