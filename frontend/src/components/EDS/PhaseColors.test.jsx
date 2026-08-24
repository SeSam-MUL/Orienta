// @vitest-environment jsdom
/**
 * The phase colour picker on the EDS map.
 *
 * The user asked for what the EBSD phase map already has. The one thing that
 * needs pinning is the store KEY: the EBSD page names a phase by the file
 * stem, the EDS candidate list carries the full filename, and if those drift
 * apart the same phase gets two colours and neither page honours the other's.
 */
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

vi.mock('../../services/api', () => ({
  edsApi: {
    autoClassify: vi.fn(() => Promise.resolve({ data: { loaded: true, summary: [] } })),
    getPhaseMap: vi.fn(() => Promise.resolve({
      data: {
        loaded: true, n_rows: 4, n_cols: 4, image: '',
        summary: [
          { phase_index: 0, cif_filename: 'Al.cif', formula: 'Al',
            n_pixels: 10, percentage: 62.5, color: '#c792ea' },
          { phase_index: 1, cif_filename: 'Si.cif', formula: 'Si',
            n_pixels: 6, percentage: 37.5, color: '#ff526f' },
        ],
        all_phases: [], n_locked: 0, undo_label: null,
      },
    })),
    clearPhaseMap: vi.fn(() => Promise.resolve({ data: {} })),
    setPhaseColors: vi.fn(() => Promise.resolve({ data: { loaded: false } })),
    cifPhases: vi.fn(() => Promise.resolve({ data: { phases: [] } })),
  },
}));
vi.mock('../../stores/useDataStore', () => ({
  default: (sel) => sel({ filePath: '/tmp/x.h5oina' }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (k, o) => (o?.name ? `${k}:${o.name}` : k) }),
}));

import { edsApi } from '../../services/api';
import usePhaseColorStore from '../../stores/usePhaseColorStore';
import { usePhaseMap, PhaseMapControls, phaseNameKey } from './PhaseMapPanel';

// The panel is a hook plus two components; the legend lives in the controls.
function Harness() {
  const handle = usePhaseMap({});
  return <PhaseMapControls handle={handle} />;
}

describe('phaseNameKey', () => {
  it('drops the .cif so both pages agree on one key', () => {
    expect(phaseNameKey('Al.cif')).toBe('Al');
    expect(phaseNameKey('Al.CIF')).toBe('Al');
    expect(phaseNameKey('  sd_0302719.cif ')).toBe('sd_0302719');
  });

  it('leaves a name that is already a stem alone', () => {
    expect(phaseNameKey('Al')).toBe('Al');
  });

  it('survives a missing name instead of throwing', () => {
    expect(phaseNameKey(undefined)).toBe('');
    expect(phaseNameKey(null)).toBe('');
  });
});

describe('the colour picker in the legend', () => {
  beforeEach(() => {
    // The project does not run vitest with `globals`, so testing-library's
    // auto-cleanup never registers: without this the previous test's DOM is
    // still mounted and every query matches twice.
    cleanup();
    usePhaseColorStore.setState({ overrides: {} });
    vi.clearAllMocks();
  });

  it('writes the user choice under the stem, not the filename', async () => {
    render(<Harness />);
    const input = await screen.findByLabelText('phaseMap.colorAria:Al.cif');
    fireEvent.change(input, { target: { value: '#ff0000' } });
    expect(usePhaseColorStore.getState().overrides).toEqual({ Al: '#ff0000' });
  });

  it('sends the overrides to the backend, which renders the map', async () => {
    render(<Harness />);
    const input = await screen.findByLabelText('phaseMap.colorAria:Si.cif');
    fireEvent.change(input, { target: { value: '#00ff00' } });
    await waitFor(() => {
      expect(edsApi.setPhaseColors).toHaveBeenCalledWith({ Si: '#00ff00' });
    });
  });

  it('right-click clears the override', async () => {
    usePhaseColorStore.setState({ overrides: { Al: '#ff0000' } });
    render(<Harness />);
    const input = await screen.findByLabelText('phaseMap.colorAria:Al.cif');
    fireEvent.contextMenu(input.parentElement);
    expect(usePhaseColorStore.getState().overrides).toEqual({});
  });

  it('picking a colour does not also select the phase', async () => {
    // The swatch sits inside a row whose click selects the phase for the
    // paint tools; a colour change must not arm a paint target.
    render(<Harness />);
    const input = await screen.findByLabelText('phaseMap.colorAria:Al.cif');
    const row = input.closest('[role="button"]');
    const before = row.style.background;
    fireEvent.click(input.parentElement);
    expect(row.style.background).toBe(before);
  });

  it('pushes saved overrides on mount, so a returning user sees their colours', async () => {
    usePhaseColorStore.setState({ overrides: { Al: '#123456' } });
    render(<Harness />);
    await waitFor(() => {
      expect(edsApi.setPhaseColors).toHaveBeenCalledWith({ Al: '#123456' });
    });
  });
});
